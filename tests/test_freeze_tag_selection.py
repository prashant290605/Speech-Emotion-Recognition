"""Which frozen experiment the analysis reads, and the ways it must not mix them.

``grid-freeze-v3`` was a string literal in two generators. It is still the
default and the published build still resolves to it, but a literal in a
plotting script is the wrong place for the identity of the experiment a paper
is about: a later prospective run could not be analysed without editing that
script, and -- worse -- the natural way to make it work would have been to
widen the filter, which is exactly how rows from two different frozen
configurations end up averaged into one number.

The property worth protecting is not configurability. It is that there is no
code path producing a union of freeze tags, and no code path silently
producing an empty selection, which downstream is indistinguishable from a
filter that worked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from ser.artifacts import (  # noqa: E402
    PUBLICATION_FREEZE_TAG,
    UnknownFreezeTag,
    filter_by_freeze_tag,
    select_freeze_tag,
)
from ser.utils.results import read_rows  # noqa: E402

LEDGER = REPO_ROOT / "results" / "runs.jsonl"


def rows_of(*tags):
    return [{"run_id": f"{tag}-{i}", "freeze_tag": tag, "macro_f1": 0.1 * i}
            for tag in tags for i in range(3)]


# -- the default is the publication tag -----------------------------------
def test_default_resolves_to_the_publication_tag():
    assert PUBLICATION_FREEZE_TAG == "grid-freeze-v3"
    assert select_freeze_tag(rows_of("grid-freeze-v3", "other")) == "grid-freeze-v3"


def test_default_selection_excludes_every_other_tag():
    selected = filter_by_freeze_tag(rows_of("grid-freeze-v3", "grid-freeze-v2", "future"))
    assert {r["freeze_tag"] for r in selected} == {"grid-freeze-v3"}
    assert len(selected) == 3


@pytest.mark.ledger
def test_default_selection_on_the_real_ledger_is_exactly_the_frozen_grid():
    """4986 rows, one tag, and nothing from v1, v2 or the unfrozen baselines."""
    if not LEDGER.exists():
        pytest.skip("ledger absent")
    rows = [r for r in read_rows(LEDGER) if r["status"] == "ok"]
    selected = filter_by_freeze_tag(rows)
    assert {r["freeze_tag"] for r in selected} == {PUBLICATION_FREEZE_TAG}
    assert len(selected) == 4986
    assert len(selected) < len(rows), "filter selected everything; it is not filtering"


# -- another tag cannot be blended in -------------------------------------
def test_an_explicit_tag_selects_only_that_tag():
    selected = filter_by_freeze_tag(rows_of("grid-freeze-v3", "prospective-v1"),
                                    "prospective-v1")
    assert {r["freeze_tag"] for r in selected} == {"prospective-v1"}


def test_there_is_no_way_to_request_a_union_of_tags():
    """A list, a comma-joined string and None-as-wildcard must all fail.

    Enumerated rather than argued: each of these is a plausible thing for a
    future caller to try, and each must be refused rather than quietly
    matching nothing or, worse, matching everything.
    """
    rows = rows_of("grid-freeze-v3", "prospective-v1")
    for attempt in (["grid-freeze-v3", "prospective-v1"],
                    "grid-freeze-v3,prospective-v1",
                    "*", ""):
        with pytest.raises((UnknownFreezeTag, TypeError)):
            filter_by_freeze_tag(rows, attempt)


def test_selecting_one_tag_never_returns_a_row_of_another():
    rows = rows_of("grid-freeze-v3", "prospective-v1", "grid-freeze-v2")
    for tag in ("grid-freeze-v3", "prospective-v1", "grid-freeze-v2"):
        assert all(r["freeze_tag"] == tag for r in filter_by_freeze_tag(rows, tag))


# -- absence fails loudly -------------------------------------------------
def test_a_tag_no_row_carries_raises_rather_than_returning_nothing():
    with pytest.raises(UnknownFreezeTag, match="no rows carry"):
        filter_by_freeze_tag(rows_of("grid-freeze-v3"), "grid-freeze-v9")


def test_the_error_names_the_tags_that_are_present():
    """So the fix is visible from the message, without opening the ledger."""
    with pytest.raises(UnknownFreezeTag) as excinfo:
        filter_by_freeze_tag(rows_of("grid-freeze-v2", "grid-freeze-v3"), "typo")
    message = str(excinfo.value)
    assert "grid-freeze-v2" in message and "grid-freeze-v3" in message
    assert "empty selection" in message


def test_the_default_also_fails_when_the_publication_tag_is_absent():
    """An analysis run against the wrong ledger stops instead of plotting nothing."""
    with pytest.raises(UnknownFreezeTag):
        filter_by_freeze_tag(rows_of("prospective-v1"))


def test_untagged_rows_are_never_selected():
    """Pre-freeze baseline rows carry freeze_tag None and must stay out."""
    rows = rows_of("grid-freeze-v3") + [{"run_id": "x", "freeze_tag": None}]
    selected = filter_by_freeze_tag(rows)
    assert all(r["freeze_tag"] == "grid-freeze-v3" for r in selected)
    assert len(selected) == 3
    # And the untagged rows are not reachable by asking for them by name.
    with pytest.raises(UnknownFreezeTag):
        filter_by_freeze_tag(rows, "None")


# -- the generators still bind to the default -----------------------------
def test_generators_resolve_the_tag_from_the_shared_constant():
    """No literal left behind in the plotting code."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import make_figures

    assert make_figures.STAGE2_TAG == PUBLICATION_FREEZE_TAG
    for name in ("tools/make_figures.py", "tools/make_tables.py"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert '"grid-freeze-v3"' not in text, f"{name} still hardcodes the tag"
