"""Phase 1: the reference integrity checker.

No test here touches the network. Crossref is stubbed, so the suite is
deterministic and the checker's *judgement* is what gets tested, not the API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ser.config import repo_root
from ser.refs import (
    TIER_CONFIRMED,
    TIER_FABRICATION,
    TIER_MANUAL,
    CrossrefRecord,
    Reference,
    audit,
    find_duplicate_coordinates,
    find_duplicate_titles,
    normalise_title,
    parse_bibtex,
    parse_bibliography,
    parse_citation_keys,
    render_report,
    surnames_from_authors,
)

TEX_PATH = repo_root() / "legacy" / "SER_Report.tex"


@pytest.fixture(scope="module")
def tex() -> str:
    return TEX_PATH.read_text(encoding="utf-8", errors="ignore")


@pytest.fixture(scope="module")
def references(tex) -> list[Reference]:
    return parse_bibliography(tex)


# -- parsing ---------------------------------------------------------------
def test_parses_all_seventeen_entries(references):
    assert len(references) == 17
    assert [r.index for r in references] == list(range(1, 18))


def test_entry_fields_are_parsed(references):
    first = references[0]
    assert first.key == "schuller2010cross"
    assert first.title.startswith("Cross-corpus acoustic emotion recognition")
    assert first.venue == "IEEE Transactions on Affective Computing"
    assert first.volume == "1"
    assert first.issue == "2"
    assert first.pages == "119-131"
    assert first.year == 2010
    assert "schuller" in first.surnames


def test_article_number_style_pages_are_parsed(references):
    by_key = {r.key: r for r in references}
    assert by_key["fu2023cross"].pages == "124"
    assert by_key["livingstone2018ryerson"].pages == "e0196391"


def test_missing_bibliography_raises():
    with pytest.raises(ValueError, match="thebibliography"):
        parse_bibliography("no bibliography here")


def test_bibtex_parser_preserves_nested_latex_and_author_surnames():
    references = parse_bibtex(
        """@article{sample,
  author = {Doe, Jane and van Rossum, Guido},
  title = {{A} {B}ibTeX title},
  journal = {Example Journal},
  volume = {7},
  number = {2},
  pages = {10--20},
  year = {2026}
}"""
    )
    assert len(references) == 1
    reference = references[0]
    assert reference.key == "sample"
    assert reference.title == "A BibTeX title"
    assert reference.surnames == ("doe", "vanrossum")
    assert reference.pages == "10-20"


# -- citation extraction ---------------------------------------------------
def test_citation_keys_exclude_the_bibliography_block(tex, references):
    cited = parse_citation_keys(tex)
    keys = {r.key for r in references}
    # Every cited key must be a real entry; the bibliography's own keys must not
    # count as citations of themselves.
    assert cited <= keys
    assert len(cited) == 15


def test_the_two_uncited_entries_are_the_suspect_ones(tex, references):
    cited = parse_citation_keys(tex)
    uncited = {r.key for r in references} - cited
    assert uncited == {"w2vprosody2023", "li2023cross"}


# -- normalisation ---------------------------------------------------------
def test_title_matching_is_case_insensitive(references):
    """[16] differs from [6] only by wav2vec2 vs Wav2Vec2."""
    by_key = {r.key: r for r in references}
    assert by_key["naderi2023cross"].title != by_key["w2vprosody2023"].title
    assert normalise_title(by_key["naderi2023cross"].title) == normalise_title(
        by_key["w2vprosody2023"].title
    )


def test_surname_extraction():
    assert surnames_from_authors("B. Schuller, B. Vlasenko, and A. Wendemuth") == (
        "schuller",
        "vlasenko",
        "wendemuth",
    )
    assert surnames_from_authors("") == ()


def test_duplicate_title_group_found(references):
    groups = find_duplicate_titles(references)
    found = {tuple(sorted(r.key for r in group)) for group in groups.values()}
    assert ("naderi2023cross", "w2vprosody2023") in found
    assert ("fu2023cross", "li2023cross") in found


def test_duplicate_coordinates_found_independently_of_title(references):
    """[17] and [7] both claim Entropy 25(1):124."""
    groups = find_duplicate_coordinates(references)
    found = {tuple(sorted(r.key for r in group)) for group in groups.values()}
    assert ("fu2023cross", "li2023cross") in found


# -- audit -----------------------------------------------------------------
def _record(title, surnames, *, volume=None, pages=None, year=None, similarity=1.0):
    return CrossrefRecord(
        doi="10.0000/test",
        title=title,
        surnames=tuple(surnames),
        authors_display=", ".join(surnames),
        venue="Test Venue",
        volume=volume,
        issue=None,
        pages=pages,
        year=year,
        similarity=similarity,
    )


def _ref(index, key, title, surnames, *, venue="V", volume="1", issue="1", pages="1-2", year=2023):
    return Reference(
        index=index,
        key=key,
        authors_raw=", ".join(surnames),
        surnames=tuple(surnames),
        title=title,
        venue=venue,
        volume=volume,
        issue=issue,
        pages=pages,
        year=year,
        raw="",
    )


def test_clean_entry_is_tier_a():
    ref = _ref(1, "good", "A real paper", ["smith"], volume="7", pages="1-2", year=2020)
    findings = audit(
        [ref],
        {"good"},
        lambda r: _record("A real paper", ["smith"], volume="7", pages="1-2", year=2020),
    )
    assert findings[0].tier == TIER_CONFIRMED
    assert findings[0].status == "VERIFIED"


def test_weak_crossref_match_does_not_produce_an_author_mismatch():
    """Regression. Comparing metadata against a *different* paper manufactures a
    fabrication signal -- it is how a real citation (wav2vec 2.0, Gretton) gets
    falsely accused when its venue simply is not indexed in Crossref."""
    ref = _ref(1, "real", "A kernel two-sample test", ["gretton", "smola"])
    findings = audit(
        [ref],
        {"real"},
        lambda r: _record("A composite kernel two-sample test", ["lv", "liu"], similarity=0.83),
    )
    finding = findings[0]

    assert "NOT-IN-CROSSREF" in finding.flags
    assert "AUTHOR-MISMATCH" not in finding.flags
    assert "VOLUME-MISMATCH" not in finding.flags
    assert finding.tier == TIER_MANUAL
    # And it must not link the wrong paper's DOI as if authoritative.
    assert "doi.org" not in finding.landing_url


def test_confirmed_entry_is_not_condemned_by_its_duplicate():
    """Regression. A duplicate pair is not symmetric: when one member is
    corroborated by Crossref and the other is not, only the other is suspect."""
    real = _ref(6, "real", "Same Title", ["naderi", "nasersharif"], volume="277", pages="110814")
    fake = _ref(16, "fake", "same title", ["ploszaj", "tarnowski"], volume="275", pages="110676")

    def lookup(reference):
        if reference.key == "real":
            return _record("Same Title", ["naderi", "nasersharif"], volume="277", pages="110814")
        return _record("Same Title", ["naderi", "nasersharif"], volume="277", pages="110814")

    findings = {f.reference.key: f for f in audit([real, fake], {"real"}, lookup)}

    assert findings["real"].tier == TIER_CONFIRMED
    assert "DUPLICATED-BY-OTHER" in findings["real"].flags
    assert "DUPLICATE-TITLE" not in findings["real"].flags

    assert findings["fake"].tier == TIER_FABRICATION
    assert "DUPLICATE-TITLE" in findings["fake"].flags
    assert "UNCITED" in findings["fake"].flags


def test_author_and_volume_mismatch_against_a_strong_match_is_flagged():
    """[9]: exact title, right venue and page, wrong authors and volume."""
    ref = _ref(
        9, "jafari", "Feature and classifier-level domain adaptation", ["jafari", "shahin"],
        volume="187", pages="110510", year=2025,
    )
    findings = audit(
        [ref],
        {"jafari"},
        lambda r: _record(
            "Feature and classifier-level domain adaptation",
            ["naeeni", "nasersharif"],
            volume="194",
            pages="110510",
            year=2025,
        ),
    )
    finding = findings[0]
    assert "AUTHOR-MISMATCH" in finding.flags
    assert "VOLUME-MISMATCH" in finding.flags
    assert finding.tier == TIER_FABRICATION


def test_uncited_alone_is_only_a_manual_flag():
    ref = _ref(1, "orphan", "Fine paper", ["smith"], volume="7", pages="1-2", year=2020)
    findings = audit(
        [ref],
        set(),
        lambda r: _record("Fine paper", ["smith"], volume="7", pages="1-2", year=2020),
    )
    assert findings[0].flags == ["UNCITED"]
    assert findings[0].tier == TIER_MANUAL


# -- end to end on the real bibliography -----------------------------------
def test_report_renders(references):
    findings = audit([references[0]], {references[0].key}, lambda r: None)
    report = render_report(findings, source="test.tex", offline=True)
    assert "# Reference integrity report" in report
    assert "cache-only mode" in report


def test_committed_report_reproduces_the_three_known_findings():
    """The report in reports/ must still name [9], [16] and [17]."""
    report_path = repo_root() / "reports" / "refs_report.md"
    if not report_path.exists():
        pytest.skip("run `ser check-refs` first")
    report = report_path.read_text(encoding="utf-8")

    assert "jafari2025feature" in report
    assert "w2vprosody2023" in report
    assert "li2023cross" in report
    # And must not have quietly condemned the genuine members of the pairs.
    assert "| 6 | `naderi2023cross`" in report
    assert "| 7 | `fu2023cross`" in report
