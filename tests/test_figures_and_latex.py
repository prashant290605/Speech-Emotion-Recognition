"""Phase 10/11: the properties figures and tables must have to be publishable."""

from __future__ import annotations

from pathlib import Path

import pytest

from ser.figures import (
    HATCHES,
    LINESTYLES,
    MARKERS,
    OKABE_ITO,
    series,
)
from ser.latex import escape, interval, number, table

REPO_ROOT = Path(__file__).resolve().parents[1]


# -- figure style ----------------------------------------------------------
def test_series_never_relies_on_colour_alone():
    """A figure that distinguishes series by colour only is unreadable in
    greyscale and to a colourblind reader. Every series must also differ in
    marker and linestyle."""
    styles = [series(i) for i in range(len(OKABE_ITO))]
    assert len({s["color"] for s in styles}) == len(styles)
    assert len({s["marker"] for s in styles}) == len(styles)
    assert len({str(s["linestyle"]) for s in styles}) == len(styles)


def test_series_wraps_rather_than_failing():
    assert series(len(OKABE_ITO))["color"] == series(0)["color"]
    assert series(0).keys() == {"color", "marker", "linestyle"}


def test_palette_is_the_published_okabe_ito_set():
    """Not an approximation. These eight hex values are the ones tested for
    deuteranopia, protanopia and tritanopia."""
    assert OKABE_ITO[0] == "#000000"
    assert set(OKABE_ITO) == {
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    }
    assert len(MARKERS) == len(LINESTYLES) == len(HATCHES) == len(OKABE_ITO)


def test_style_sets_embeddable_fonts():
    """Type 3 fonts are matplotlib's PDF default and are rejected by several
    publishers. Checked because the failure appears only at submission."""
    import matplotlib.pyplot as plt

    from ser.figures import use_style

    use_style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42


def test_every_committed_figure_has_a_vector_form():
    figures = REPO_ROOT / "figures"
    if not figures.exists():
        pytest.skip("figures not generated")
    pngs = sorted(p.stem for p in figures.glob("*.png"))
    pdfs = sorted(p.stem for p in figures.glob("*.pdf"))
    assert pngs == pdfs, "every PNG preview needs the PDF that is actually submitted"
    assert pdfs, "no figures found"


# -- LaTeX -----------------------------------------------------------------
def test_escape_handles_the_characters_rung_names_actually_contain():
    assert escape("mkmmd_full") == r"mkmmd\_full"
    assert escape("95% CI") == r"95\% CI"
    assert escape("a&b") == r"a\&b"
    assert escape("x^2") == r"x\textasciicircum{}2"
    assert escape("\\") == r"\textbackslash{}"


def test_number_prints_a_dash_for_a_genuine_absence():
    """A conditional MMD below the support threshold is not zero, and a table
    that renders it as zero is lying."""
    assert number(None) == "--"
    assert number(float("nan")) == "--"
    assert number(0.0) == "0.0000"
    assert number(0.5, 2) == "0.50"


def test_interval_degrades_to_the_mean_when_there_is_no_interval():
    assert interval(0.5, 0.4, 0.6) == "0.5000 [0.4000, 0.6000]"
    assert interval(0.5, None, None) == "0.5000"
    assert interval(0.5, float("nan"), float("nan")) == "0.5000"
    assert interval(None, 0.1, 0.2) == "--"


def test_table_refuses_a_ragged_body():
    """A row with the wrong cell count produces LaTeX that fails to compile at
    submission time rather than here."""
    with pytest.raises(ValueError, match="ragged"):
        table([["a", "b"], ["c"]], ["x", "y"], caption="c", label="l")


def test_table_emits_balanced_booktabs():
    text = table([["a", 1]], ["k", "v"], caption="Cap", label="demo",
                 notes=["a note"])
    for opening, closing in (("\\begin{table}", "\\end{table}"),
                             ("\\begin{tabular}", "\\end{tabular}")):
        assert text.count(opening) == text.count(closing) == 1
    for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert rule in text
    assert "\\label{tab:demo}" in text
    assert "a note" in text


def test_table_respects_escape_cells_for_the_header_too():
    """A header written as LaTeX must not be escaped into visible backslashes.
    This was a real bug: `$\\rho$` rendered as `\\$\\textbackslash{}rho\\$`."""
    raw = table([["x"]], ["$\\rho$"], caption="c", label="l", escape_cells=False)
    assert "$\\rho$" in raw
    assert "textbackslash" not in raw

    escaped = table([["x"]], ["a_b"], caption="c", label="l", escape_cells=True)
    assert r"a\_b" in escaped


def test_generated_tables_are_balanced_and_labelled():
    tables = REPO_ROOT / "tables"
    if not tables.exists():
        pytest.skip("tables not generated")
    files = sorted(tables.glob("*.tex"))
    assert files, "no tables found"
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}")
        assert text.count(r"\begin{table}") == text.count(r"\end{table}")
        assert f"\\label{{tab:{path.stem}}}" in text or r"\label{tab:" in text
        # Every table states the run filter it was computed from.
        assert "Filter:" in text or "replicates" in text, path.name
