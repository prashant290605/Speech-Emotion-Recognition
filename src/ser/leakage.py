"""Executable leakage assertions.

The Phase 2 brief requires four hard assertions. Three live here; the fourth
(``map_label`` purity) is a property of :mod:`ser.labels` and is asserted in
``tests/test_labelmap.py``.

1. No speaker appears in more than one split within a corpus.
2. No utterance appears in both a source split and a target split.
3. ``target_test`` never appears in any fitted alignment object.

Assertion 3 defines a **contract that Phase 5 must satisfy**, not a check Phase 5
may opt into: every alignment records the identifiers it was fitted on, and the
runner asserts that set is disjoint from ``target_test``. The original study
fitted CORAL on target train *and test* concatenated, which is precisely what
this catches — and it caught nothing at the time because nothing was checked.

Identifiers are ``utterance_id`` strings, not positional indices. Positions shift
whenever a filter changes; an utterance id does not.
"""

from __future__ import annotations

from typing import Iterable, List, Protocol, Sequence, Set, runtime_checkable

from .splits import SOURCE_ROLES, TARGET_ROLES, PairSplit

__all__ = [
    "LeakageError",
    "FittedAlignment",
    "assert_speaker_disjoint",
    "assert_source_target_utterances_disjoint",
    "assert_alignment_blind_to_target_test",
    "assert_no_leakage",
    "check_pair_split",
]


class LeakageError(AssertionError):
    """A split or a fitted object violates the leakage contract."""


@runtime_checkable
class FittedAlignment(Protocol):
    """What Phase 5 alignment objects must expose.

    ``fitted_on_indices`` holds every utterance id the object was shown during
    ``fit``. It must never grow during ``transform``.
    """

    fitted_on_indices: Set[str]


def assert_speaker_disjoint(pair_split: PairSplit) -> None:
    """Assertion 1: no speaker (or session) is in two roles of the same corpus."""
    seen: dict[tuple[str, str], str] = {}
    for role, split in pair_split.splits().items():
        for group in split.group_ids:
            previous = seen.get((split.corpus, group))
            if previous is not None:
                raise LeakageError(
                    f"{pair_split.split_id}: {split.corpus} group {group!r} appears in "
                    f"both {previous!r} and {role!r}. Splits must be speaker-disjoint."
                )
            seen[(split.corpus, group)] = role


def assert_source_target_utterances_disjoint(pair_split: PairSplit) -> None:
    """Assertion 2: no utterance is on both the source and the target side.

    This is the assertion that makes in-domain pairs honest. When source and
    target are the same corpus, a naive implementation reports training data as
    a test score and nothing else would notice.
    """
    source = set()
    for role in SOURCE_ROLES:
        source |= pair_split.utterance_ids(role)
    target = set()
    for role in TARGET_ROLES:
        target |= pair_split.utterance_ids(role)

    overlap = source & target
    if overlap:
        sample = sorted(overlap)[:5]
        raise LeakageError(
            f"{pair_split.split_id}: {len(overlap)} utterance(s) appear on both the "
            f"source and target side, e.g. {sample}. "
            "For an in-domain pair this means test scores include training data."
        )


def assert_alignment_blind_to_target_test(
    alignment: FittedAlignment, pair_split: PairSplit
) -> None:
    """Assertion 3: a fitted alignment never saw ``target_test``."""
    fitted = getattr(alignment, "fitted_on_indices", None)
    if fitted is None:
        raise LeakageError(
            f"{type(alignment).__name__} does not expose 'fitted_on_indices'. "
            "Every alignment must record what it was fitted on so this assertion "
            "can be made; an alignment that cannot be checked must not be used."
        )

    held_out = pair_split.utterance_ids("target_test")
    overlap = set(fitted) & held_out
    if overlap:
        sample = sorted(overlap)[:5]
        raise LeakageError(
            f"{pair_split.split_id}: {type(alignment).__name__} was fitted on "
            f"{len(overlap)} target_test utterance(s), e.g. {sample}. "
            "target_test is touched exactly once, at scoring time."
        )

    allowed = (
        pair_split.utterance_ids("source_train")
        | pair_split.utterance_ids("source_val")
        | pair_split.utterance_ids("target_adapt")
    )
    unexpected = set(fitted) - allowed
    if unexpected:
        sample = sorted(unexpected)[:5]
        raise LeakageError(
            f"{pair_split.split_id}: {type(alignment).__name__} was fitted on "
            f"{len(unexpected)} utterance(s) outside this split entirely, e.g. "
            f"{sample}."
        )


def assert_no_leakage(pair_split: PairSplit) -> None:
    """Run every split-level assertion. Raises on the first violation."""
    assert_speaker_disjoint(pair_split)
    assert_source_target_utterances_disjoint(pair_split)


def check_pair_split(pair_split: PairSplit) -> List[str]:
    """Non-raising form: returns a list of violations, empty when clean.

    Used by the reporting commands, which should show every problem at once
    rather than stopping at the first.
    """
    problems: List[str] = []
    for check in (assert_speaker_disjoint, assert_source_target_utterances_disjoint):
        try:
            check(pair_split)
        except LeakageError as exc:
            problems.append(str(exc))

    for role, split in pair_split.splits().items():
        if len(split) == 0:
            problems.append(f"{pair_split.split_id}: {role} is empty.")
    return problems


def all_utterances_accounted_for(
    pair_split: PairSplit, expected: Iterable[str]
) -> Set[str]:
    """Utterances of the pair's corpora that landed in no split.

    Non-empty is not automatically wrong -- an in-domain partition can round
    speakers away -- but it should be visible rather than silent.
    """
    assigned: Set[str] = set()
    for role in pair_split.splits():
        assigned |= pair_split.utterance_ids(role)
    return set(expected) - assigned
