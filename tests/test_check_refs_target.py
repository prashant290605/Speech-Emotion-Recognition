"""Which manuscript the reference audit means by default.

``tools/check_refs.py`` used to default to ``legacy/SER_Report.tex``. A bare
run therefore audited the superseded pre-rebuild document and wrote its
findings over the current report: a check that passes while telling you nothing
about the paper you are submitting. It cost real time in Phase 3.

The supplement half matters just as much. The submission is two documents, and
``lipton2018bbse`` and ``saerens2002adjusting`` are cited only in supplementary
Section S2. Auditing the article alone reports them as uncited, and the obvious
"fix" for an uncited entry is to delete it -- which would remove two live
references. These tests pin both halves, and the bibliography is never edited
to satisfy the checker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_refs  # noqa: E402

from ser.refs import (  # noqa: E402
    CURRENT_PAPER,
    LEGACY_PAPER,
    parse_bibtex,
    parse_citation_keys,
    run_audit,
)
from ser.refs import _tex_tree  # noqa: E402

SUPPLEMENT_ONLY = ("lipton2018bbse", "saerens2002adjusting")


# -- the default target ----------------------------------------------------
def test_default_target_is_the_current_paper_not_the_legacy_report():
    assert CURRENT_PAPER["tex"] == "paper/main.tex"
    assert CURRENT_PAPER["bib"] == "paper/refs.bib"
    assert CURRENT_PAPER["supplement"] == "paper/supplementary.tex"
    assert "legacy" not in CURRENT_PAPER["tex"]


def test_default_output_does_not_overwrite_the_legacy_report():
    """Separate files, so one audit cannot silently replace the other."""
    assert CURRENT_PAPER["out"] != LEGACY_PAPER["out"]
    assert LEGACY_PAPER["out"] == "reports/refs_report.md"


def test_the_legacy_document_is_still_reachable_on_request():
    assert LEGACY_PAPER["tex"] == "legacy/SER_Report.tex"
    assert (REPO_ROOT / LEGACY_PAPER["tex"]).exists()
    assert LEGACY_PAPER["bib"] is None      # it carries an inline bibliography
    assert LEGACY_PAPER["supplement"] is None


def test_no_tool_still_points_at_the_legacy_report_by_default():
    """Guards the regression at its source, not only at its symptom."""
    for rel in ("tools/check_refs.py", "src/ser/cli.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'default="legacy/SER_Report.tex"' not in text, rel


# -- the supplement is part of the submission ------------------------------
@pytest.mark.parametrize("key", SUPPLEMENT_ONLY)
def test_these_entries_really_are_cited_only_in_the_supplement(key):
    """If this fails the fixture premise is gone and the tests below are moot."""
    article = parse_citation_keys(_tex_tree(REPO_ROOT / CURRENT_PAPER["tex"]))
    supplement = parse_citation_keys(_tex_tree(REPO_ROOT / CURRENT_PAPER["supplement"]))
    assert key not in article
    assert key in supplement


def test_supplement_citations_count_as_cited():
    article = parse_citation_keys(_tex_tree(REPO_ROOT / CURRENT_PAPER["tex"]))
    supplement = parse_citation_keys(_tex_tree(REPO_ROOT / CURRENT_PAPER["supplement"]))
    combined = article | supplement
    for key in SUPPLEMENT_ONLY:
        assert key in combined


def test_every_bibliography_entry_is_cited_once_the_supplement_counts():
    """The real claim: nothing in refs.bib is dead weight.

    Asserted over the union rather than the article alone, which is the whole
    point -- and without touching refs.bib to make it come out.
    """
    entries = {r.key for r in parse_bibtex(
        (REPO_ROOT / CURRENT_PAPER["bib"]).read_text(encoding="utf-8"))}
    cited = (parse_citation_keys(_tex_tree(REPO_ROOT / CURRENT_PAPER["tex"]))
             | parse_citation_keys(_tex_tree(REPO_ROOT / CURRENT_PAPER["supplement"])))
    assert entries - cited == set(), "uncited bibliography entries"


def test_dropping_the_supplement_is_what_makes_them_look_uncited():
    """The negative control. Without it, the test above proves nothing."""
    article = parse_citation_keys(_tex_tree(REPO_ROOT / CURRENT_PAPER["tex"]))
    entries = {r.key for r in parse_bibtex(
        (REPO_ROOT / CURRENT_PAPER["bib"]).read_text(encoding="utf-8"))}
    assert entries - article == set(SUPPLEMENT_ONLY)


# -- the audit runs end to end --------------------------------------------
@pytest.mark.ledger
def test_default_audit_reports_no_uncited_entries(tmp_path):
    """Offline, so it exercises the wiring without contacting Crossref."""
    out = tmp_path / "report.md"
    code = run_audit(
        tex_path=REPO_ROOT / CURRENT_PAPER["tex"],
        out_path=out,
        cache_path=REPO_ROOT / ".cache" / "crossref.json",
        bib_path=REPO_ROOT / CURRENT_PAPER["bib"],
        extra_tex_paths=[REPO_ROOT / CURRENT_PAPER["supplement"]],
        offline=True,
    )
    assert code in (0, 1)  # 1 = manual-resolution entries remain; not a wiring fault
    text = out.read_text(encoding="utf-8")
    assert "UNCITED" not in text
    for key in SUPPLEMENT_ONLY:
        assert key in text


@pytest.mark.ledger
def test_audit_without_the_supplement_flags_them(tmp_path):
    out = tmp_path / "report.md"
    run_audit(
        tex_path=REPO_ROOT / CURRENT_PAPER["tex"],
        out_path=out,
        cache_path=REPO_ROOT / ".cache" / "crossref.json",
        bib_path=REPO_ROOT / CURRENT_PAPER["bib"],
        offline=True,
    )
    assert "UNCITED" in out.read_text(encoding="utf-8")


def test_run_audit_refuses_a_missing_extra_root(tmp_path):
    code = run_audit(
        tex_path=REPO_ROOT / CURRENT_PAPER["tex"],
        out_path=tmp_path / "report.md",
        cache_path=tmp_path / "cache.json",
        bib_path=REPO_ROOT / CURRENT_PAPER["bib"],
        extra_tex_paths=[tmp_path / "absent.tex"],
        offline=True,
    )
    assert code == 2
    assert not (tmp_path / "report.md").exists()


# -- argument handling -----------------------------------------------------
def test_legacy_flag_rejects_a_conflicting_explicit_target(capsys):
    with pytest.raises(SystemExit):
        check_refs.main(["--legacy", "--tex", "paper/main.tex"])


def test_report_header_records_a_relative_source_not_a_local_path():
    """A committed report must not carry the path of the machine that built it."""
    report = REPO_ROOT / LEGACY_PAPER["out"]
    if not report.exists():
        pytest.skip("legacy report not generated")
    header = report.read_text(encoding="utf-8").splitlines()[2]
    assert ":" not in header.replace("Source:", ""), header
    assert "Users" not in header
