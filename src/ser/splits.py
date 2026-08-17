"""Speaker-disjoint splits, deterministic given a seed.

Four roles per corpus pair:

    source_train    labelled data the classifier fits on
    source_val      the ONLY surface any hyperparameter, alpha, or layer choice
                    may be selected on
    target_adapt    the ONLY target data any alignment method may see
    target_test     touched exactly once, at scoring time

Two properties this module exists to guarantee:

* **Speaker disjointness.** Grouping is by speaker, or by session for corpora
  where the session is the standard unit (IEMOCAP). No speaker appears in two
  roles of the same corpus.
* **In-domain pairs partition one corpus four ways.** Running the source split
  and the target split independently over the same corpus would place the same
  speakers in ``source_train`` and ``target_test``, so every in-domain number
  would be reporting training data. Instead the corpus is divided into a source
  side and a target side first, and the roles are carved out within each side.

Determinism is derived from ``(seed, corpus, side)`` rather than from a single
global RNG, so a corpus gets the same source-side partition in every pair where
it is the source. That makes results across different targets comparable, and it
means adding a pair to the grid does not perturb existing ones.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .manifest import ManifestRow

__all__ = [
    "Split",
    "PairSplit",
    "ROLES",
    "grouping_key_for",
    "make_pair_split",
    "split_class_counts",
]

ROLES = ("source_train", "source_val", "target_adapt", "target_test")

SOURCE_ROLES = ("source_train", "source_val")
TARGET_ROLES = ("target_adapt", "target_test")


@dataclass(frozen=True)
class Split:
    """One role's share of one corpus."""

    corpus: str
    role: str
    utterance_ids: Tuple[str, ...]
    group_ids: Tuple[str, ...]  # speakers, or sessions for session-split corpora

    def __len__(self) -> int:
        return len(self.utterance_ids)


@dataclass(frozen=True)
class PairSplit:
    """All four roles for one (source, target, seed)."""

    source_corpus: str
    target_corpus: str
    label_space: str
    seed: int
    split_id: str
    source_train: Split
    source_val: Split
    target_adapt: Split
    target_test: Split
    # Set when `splits.matched_source_train` capped source_train to the other
    # direction's size. None means the split was left at its natural size --
    # either matching is off, or this direction was already the smaller one.
    source_train_cap: Optional[int] = None

    @property
    def is_in_domain(self) -> bool:
        return self.source_corpus == self.target_corpus

    def splits(self) -> Dict[str, Split]:
        return {
            "source_train": self.source_train,
            "source_val": self.source_val,
            "target_adapt": self.target_adapt,
            "target_test": self.target_test,
        }

    def utterance_ids(self, role: str) -> frozenset[str]:
        return frozenset(self.splits()[role].utterance_ids)

    def sizes(self) -> Dict[str, int]:
        return {role: len(split) for role, split in self.splits().items()}


def grouping_key_for(corpus: str, config) -> str:
    """``'session_id'`` where the session is the standard unit, else ``'speaker_id'``.

    IEMOCAP is split on session because its ten actors appear in fixed pairs
    across five sessions; splitting on the speaker tag alone lets the same actor
    land on both sides.
    """
    if corpus == "iemocap" and config.splits.iemocap_split_unit == "session":
        return "session_id"
    return "speaker_id"


def _group_of(row: ManifestRow, key: str) -> str:
    value = getattr(row, key)
    # A corpus with no session structure falls back to the speaker, so a
    # misconfigured split unit cannot collapse every utterance into one group.
    return value if value else row.speaker_id


def _rng(seed: int, *parts: str) -> np.random.Generator:
    """RNG seeded from the run seed plus a stable label.

    Hashing the parts rather than mutating a shared stream means each
    (corpus, side) partition is independent of enumeration order.
    """
    digest = hashlib.sha256(("|".join((str(seed), *parts))).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _partition(groups: Sequence[str], ratio: float, rng: np.random.Generator) -> Tuple[List[str], List[str]]:
    """Split group ids in two, at least one group on each side."""
    if len(groups) < 2:
        raise ValueError(
            f"need at least 2 groups to split, got {len(groups)}: {list(groups)}"
        )
    shuffled = list(groups)
    rng.shuffle(shuffled)
    cut = int(round(len(shuffled) * ratio))
    cut = max(1, min(cut, len(shuffled) - 1))
    return sorted(shuffled[:cut]), sorted(shuffled[cut:])


def _make_split(
    rows: Sequence[ManifestRow],
    corpus: str,
    role: str,
    group_key: str,
    group_ids: Sequence[str],
    label_space: str,
) -> Split:
    wanted = set(group_ids)
    utterances = [
        row.utterance_id
        for row in rows
        if row.corpus == corpus
        and _group_of(row, group_key) in wanted
        # Utterances excluded from this label space are not part of any split.
        and (row.label_six if label_space == "six" else row.label_four)
    ]
    return Split(
        corpus=corpus,
        role=role,
        utterance_ids=tuple(sorted(utterances)),
        group_ids=tuple(sorted(wanted)),
    )


def make_pair_split(
    rows: Sequence[ManifestRow],
    config,
    source_corpus: str,
    target_corpus: str,
    seed: int,
    label_space: Optional[str] = None,
) -> PairSplit:
    """Build the four roles for one corpus pair at one seed.

    With ``splits.matched_source_train`` set, the cross-corpus ``source_train``
    is capped to whichever direction of the pair has fewer source-train
    utterances, so that a reported transfer asymmetry cannot be a training-set
    size effect in disguise. See :func:`_match_source_train`.
    """
    pair = _build_pair_split(rows, config, source_corpus, target_corpus, seed, label_space)

    matched = getattr(config.splits, "matched_source_train", False)
    if not matched or source_corpus == target_corpus:
        return pair

    # The cap is the smaller direction's natural size, so one direction is
    # untouched and the other comes down to meet it. Computing the reverse
    # split here is safe: _build_pair_split never consults the cap, so there is
    # no recursion.
    reverse = _build_pair_split(
        rows, config, target_corpus, source_corpus, seed, pair.label_space
    )
    cap = min(len(pair.source_train), len(reverse.source_train))
    if len(pair.source_train) <= cap:
        return pair

    return replace(
        pair,
        source_train=_match_source_train(rows, pair, cap, seed),
        source_train_cap=cap,
    )


def _match_source_train(
    rows: Sequence[ManifestRow], pair: PairSplit, cap: int, seed: int
) -> Split:
    """Subsample ``source_train`` to ``cap`` utterances, stratified by class.

    Three properties this has to preserve, all of them load-bearing:

    * **Speaker disjointness.** Utterances are only removed, never moved
      between roles, so every leakage guarantee from the original partition
      still holds. Dropping a speaker entirely is fine; splitting one across
      roles would not be, and cannot happen here.
    * **Class proportions.** Allocation is by largest remainder, so the capped
      split's class distribution matches the uncapped one as closely as an
      integer split allows. A uniform random subsample would shift the label
      prior and confound the very comparison the cap exists to enable.
    * **Determinism.** Seeded from the run seed plus a stable label, sorted on
      the way out, so the same (pair, seed) always yields the same subsample on
      any machine.
    """
    labels = {row.utterance_id: row for row in rows}
    space = pair.label_space

    def label_of(utterance_id: str) -> str:
        row = labels[utterance_id]
        return row.label_six if space == "six" else row.label_four

    by_class: Dict[str, List[str]] = {}
    for utterance_id in pair.source_train.utterance_ids:
        by_class.setdefault(label_of(utterance_id), []).append(utterance_id)

    total = len(pair.source_train)
    # Largest remainder: floor the proportional share, then hand out what is
    # left to the classes with the largest fractional parts.
    exact = {name: len(ids) * cap / total for name, ids in by_class.items()}
    quota = {name: int(value) for name, value in exact.items()}
    remaining = cap - sum(quota.values())
    for name in sorted(exact, key=lambda n: (-(exact[n] - quota[n]), n))[:remaining]:
        quota[name] += 1

    rng = _rng(seed, pair.source_corpus, pair.target_corpus, "matched_source_train")
    kept: List[str] = []
    for name in sorted(by_class):
        ids = sorted(by_class[name])
        rng.shuffle(ids)
        kept.extend(ids[: quota[name]])

    kept = sorted(kept)
    if len(kept) != cap:
        raise ValueError(
            f"matched subsample produced {len(kept)} utterances, expected {cap}"
        )

    return replace(
        pair.source_train,
        utterance_ids=tuple(kept),
        group_ids=tuple(sorted({labels[u].speaker_id for u in kept})),
    )


def _build_pair_split(
    rows: Sequence[ManifestRow],
    config,
    source_corpus: str,
    target_corpus: str,
    seed: int,
    label_space: Optional[str] = None,
) -> PairSplit:
    """The four roles at their natural sizes, before any matched-n cap."""
    if label_space is None:
        label_space = (
            config.labels.space_for_iemocap_pairs
            if "iemocap" in (source_corpus, target_corpus)
            else config.labels.space_for_other_pairs
        )

    split_id = f"{source_corpus}-{target_corpus}-s{seed}"

    if source_corpus == target_corpus:
        corpus = source_corpus
        key = grouping_key_for(corpus, config)
        groups = _groups_in(rows, corpus, key, label_space)

        # One corpus, four disjoint ways. See the module docstring.
        source_side, target_side = _partition(
            groups, config.splits.in_domain_source_ratio, _rng(seed, corpus, "in_domain")
        )
        train, val = _partition(
            source_side, config.splits.source_train_ratio, _rng(seed, corpus, "in_domain_source")
        )
        adapt, test = _partition(
            target_side, config.splits.target_adapt_ratio, _rng(seed, corpus, "in_domain_target")
        )
        assignments = {
            "source_train": (corpus, train),
            "source_val": (corpus, val),
            "target_adapt": (corpus, adapt),
            "target_test": (corpus, test),
        }
    else:
        source_key = grouping_key_for(source_corpus, config)
        target_key = grouping_key_for(target_corpus, config)

        train, val = _partition(
            _groups_in(rows, source_corpus, source_key, label_space),
            config.splits.source_train_ratio,
            _rng(seed, source_corpus, "source"),
        )
        adapt, test = _partition(
            _groups_in(rows, target_corpus, target_key, label_space),
            config.splits.target_adapt_ratio,
            _rng(seed, target_corpus, "target"),
        )
        assignments = {
            "source_train": (source_corpus, train),
            "source_val": (source_corpus, val),
            "target_adapt": (target_corpus, adapt),
            "target_test": (target_corpus, test),
        }

    built = {}
    for role, (corpus, group_ids) in assignments.items():
        key = grouping_key_for(corpus, config)
        built[role] = _make_split(rows, corpus, role, key, group_ids, label_space)

    return PairSplit(
        source_corpus=source_corpus,
        target_corpus=target_corpus,
        label_space=label_space,
        seed=seed,
        split_id=split_id,
        **built,
    )


def _groups_in(
    rows: Sequence[ManifestRow], corpus: str, key: str, label_space: str
) -> List[str]:
    groups = {
        _group_of(row, key)
        for row in rows
        if row.corpus == corpus
        and (row.label_six if label_space == "six" else row.label_four)
    }
    if not groups:
        raise ValueError(f"no utterances for corpus {corpus!r} in label space {label_space!r}")
    return sorted(groups)


def split_class_counts(
    rows: Sequence[ManifestRow], split: Split, label_space: str
) -> Counter:
    """Class counts within one split. Used for the A9 split-level priors."""
    wanted = set(split.utterance_ids)
    return Counter(
        (row.label_six if label_space == "six" else row.label_four)
        for row in rows
        if row.utterance_id in wanted
    )
