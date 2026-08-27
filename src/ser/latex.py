"""LaTeX table emission. Tables regenerate from results; nothing is retyped.

Deliberately dependency-free at the LaTeX end: `booktabs` is the only package
required, and it is close to universal. No `siunitx`, no `multirow`, no
`tabularx` -- a table that fails to compile in the target venue's template is a
table that gets retyped by hand, which is exactly what this module exists to
prevent.

Every table carries a `\\label` derived from its filename and a caption that
states the run filter it was computed from. A number in a paper whose
provenance is not on the same page as it is a number nobody can check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

__all__ = ["escape", "number", "interval", "table", "write_table"]

_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def escape(text: object) -> str:
    """Escape LaTeX specials. Rung names like ``mkmmd_full`` need it."""
    out = []
    for character in str(text):
        if character == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(_ESCAPES.get(character, character))
    return "".join(out)


def number(value: Optional[float], places: int = 4, dash: str = "--") -> str:
    """A number, or an em-dash for a genuine absence.

    None is printed as a dash rather than as 0 or NaN. A missing conditional
    MMD below the support threshold is not a zero, and a table that renders it
    as one is lying.
    """
    if value is None:
        return dash
    try:
        if value != value:  # NaN
            return dash
    except TypeError:
        return escape(value)
    return f"{value:.{places}f}"


def interval(mean: Optional[float], lo: Optional[float], hi: Optional[float],
             places: int = 4) -> str:
    """``0.3335 [0.2550, 0.3856]``, or the mean alone when there is no interval."""
    if mean is None:
        return "--"
    if lo is None or hi is None or lo != lo or hi != hi:
        return number(mean, places)
    return f"{number(mean, places)} [{number(lo, places)}, {number(hi, places)}]"


def table(
    rows: Sequence[Sequence[object]],
    header: Sequence[str],
    *,
    caption: str,
    label: str,
    column_spec: Optional[str] = None,
    notes: Optional[Iterable[str]] = None,
    escape_cells: bool = True,
) -> str:
    """A complete ``table`` float using booktabs rules.

    ``notes`` become a small-font block under the rules -- the run filter, the
    seed count, the floor. They are part of the table, not the caption, so they
    survive a journal that truncates captions.
    """
    if any(len(row) != len(header) for row in rows):
        widths = sorted({len(row) for row in rows} | {len(header)})
        raise ValueError(f"ragged table: row widths {widths} against "
                         f"{len(header)} header cells")

    spec = column_spec or ("l" + "r" * (len(header) - 1))
    # Applies to the header too. Escaping a header the caller already wrote as
    # LaTeX turns its commands into visible backslashes.
    render = escape if escape_cells else (lambda v: str(v))

    lines = [
        r"\begin{table}[tb]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{tab:{label}}}",
        f"  \\begin{{tabular}}{{{spec}}}",
        r"    \toprule",
        "    " + " & ".join(render(h) for h in header) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(render(cell) for cell in row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    for note in notes or ():
        lines.append(r"  \\[2pt]")
        lines.append(r"  \begin{minipage}{\linewidth}\footnotesize " + note
                     + r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table(text: str, name: str, root: Path | str = "tables") -> Path:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.tex"
    path.write_text(text, encoding="utf-8")
    return path
