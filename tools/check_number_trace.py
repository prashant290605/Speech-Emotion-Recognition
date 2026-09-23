"""Trace manuscript outcome numbers to generated result reports.

Usage:
    python tools/check_number_trace.py

The paper contains two numeric classes. Outcome numbers in the abstract,
results, discussion, reproducibility section, conclusion, highlights, and
result tables must be in a generated report (allowing ordinary display rounding).
Corpus and fixed-design constants in Methods and the corpus-description table
are reported separately: they are reproducibility inputs, not outcomes.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SUPPLEMENT = PAPER / "supplementary.tex"
RESULT_SOURCES = (
    ROOT / "reports" / "translation_audit.md",
    ROOT / "reports" / "RESULTS.md",
    ROOT / "reports" / "zscore_global_bound.md",
    ROOT / "reports" / "phase9_reference_geometry.md",
    ROOT / "reports" / "calm_dropped_sensitivity.md",
    ROOT / "reports" / "neutral_excluded_sensitivity.md",
    ROOT / "reports" / "calm_dropped_diagnostics.md",
    ROOT / "reports" / "neutral_excluded_diagnostics.md",
)
NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![A-Za-z_])")
LATEX_SCIENTIFIC = re.compile(
    r"(?P<coefficient>[-+]?\d+(?:\.\d+)?)\\times\s*10\^\{(?P<exponent>[-+]?\d+)\}"
)
# Fixed analysis parameters which appear in an otherwise outcome-bearing table.
KNOWN_DESIGN_TOKENS = {
    ("decomposition.tex", "50"),
    ("calm_sensitivity.tex", "5"),
    ("calm_sensitivity.tex", "10.0"),
    ("calm_sensitivity.tex", "95"),
    ("neutral_excluded_sensitivity.tex", "5"),
    ("neutral_excluded_sensitivity.tex", "95"),
    ("label_harmonisation_diagnostics.tex", "5"),
    ("label_harmonisation_diagnostics.tex", "10.0"),
    # Supplementary S8 states the MK-MMD optimiser budget. These are fixed
    # experimental inputs of the same kind Methods carries, not outcomes:
    # 200 Adam steps, batch size 256, and the per-iteration step norm that
    # motivates the parameter-count-normalised learning rate.
    ("supplementary.tex", "200"),
    ("supplementary.tex", "256"),
    ("supplementary.tex", "0.77"),
    # Design inputs that moved out of Methods with the protocols they belong
    # to: the 192 RAVDESS calm utterances a control drops, and the minimum
    # per-class support below which a conditional MMD is reported as undefined
    # rather than estimated. Corpus and threshold constants, not outcomes.
    ("supplementary.tex", "192"),
    ("supplementary.tex", "50"),
}


@dataclass(frozen=True)
class Occurrence:
    path: Path
    line: int
    token: str


def text_without_noncontent(text: str, *, strip_latex_comments: bool = True) -> str:
    if strip_latex_comments:
        text = re.sub(r"(?m)(?<!\\)%.*$", "", text)
    text = re.sub(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{[^}]*\}", "", text)
    text = re.sub(r"\[CITE:[^\]]*\]", "", text, flags=re.S)
    text = LATEX_SCIENTIFIC.sub(
        lambda match: f"{match.group('coefficient')}e{match.group('exponent')}",
        text,
    )
    text = re.sub(r"\^\{[^{}]*\}", "", text)
    text = re.sub(r"(?<=\d)--(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text)
    return text.replace("{,}", "").replace(",", "").replace("−", "-")


def source_text(path: Path) -> str:
    """Return the manuscript text containing claims, not title-page metadata."""
    text = path.read_text(encoding="utf-8")
    if path == PAPER / "main.tex":
        match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
        if match is None:
            raise ValueError(f"abstract not found in {path}")
        text = match.group(1)
    return text_without_noncontent(text)


def occurrences(path: Path) -> list[Occurrence]:
    content = source_text(path)
    found: list[Occurrence] = []
    for number, line in enumerate(content.splitlines(), 1):
        for match in NUMBER.finditer(line):
            found.append(Occurrence(path, number, match.group()))
    return found


def decimal(token: str) -> Decimal | None:
    try:
        return Decimal(token.lstrip("+"))
    except InvalidOperation:
        return None


def is_traced(token: str, result_values: set[Decimal]) -> bool:
    value = decimal(token)
    if value is None:
        return False
    if value in result_values:
        return True
    mantissa, *exponent = token.lower().split("e", 1)
    if "." not in mantissa:
        return False
    places = len(mantissa.split(".", 1)[1])
    scale = int(exponent[0]) if exponent else 0
    tolerance = Decimal("0.5") * (Decimal(10) ** (scale - places))
    return any(abs(value - candidate) < tolerance for candidate in result_values)


def input_tables() -> list[Path]:
    """Return generated tables included by the manuscript or the supplement.

    The supplement is scanned as well. Moving a table out of the article must
    not move its numbers out of the trace: an unverified number is just as
    wrong in supplementary material, and the supplement is where the detailed
    robustness and diagnostic tables now live.
    """
    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    if SUPPLEMENT.exists():
        text += "\n" + SUPPLEMENT.read_text(encoding="utf-8")
    for section in (PAPER / "sections").glob("*.tex"):
        text += "\n" + section.read_text(encoding="utf-8")
    paths: list[Path] = []
    for match in re.finditer(r"\\input\{(\.\./tables/[^}]+)\}", text):
        path = PAPER / match.group(1)
        if not path.suffix:
            path = path.with_suffix(".tex")
        if path.exists():
            paths.append(path)
    return paths


def source_files() -> tuple[list[Path], list[Path]]:
    outcomes = [PAPER / "main.tex", PAPER / "highlights.txt"]
    if SUPPLEMENT.exists():
        outcomes.append(SUPPLEMENT)
    outcomes.extend(
        path for path in sorted((PAPER / "sections").glob("*.tex"))
        if path.name != "methods.tex"
    )
    tables = input_tables()
    outcomes.extend(path for path in tables if path.name != "corpora.tex")
    context = [PAPER / "sections" / "methods.tex"]
    context.extend(path for path in tables if path.name == "corpora.tex")
    return outcomes, context


def main() -> int:
    missing = [path for path in RESULT_SOURCES if not path.exists()]
    if missing:
        print("missing " + ", ".join(path.relative_to(ROOT).as_posix() for path in missing))
        return 2

    result_values = {
        value
        for source in RESULT_SOURCES
        for match in NUMBER.finditer(text_without_noncontent(
            source.read_text(encoding="utf-8"), strip_latex_comments=False
        ))
        if (value := decimal(match.group())) is not None
    }
    outcome_files, context_files = source_files()
    context_set = set(context_files)
    untraced_outcomes: list[Occurrence] = []
    context_only: list[Occurrence] = []
    traced = 0

    for path in [*outcome_files, *context_files]:
        for occurrence in occurrences(path):
            if is_traced(occurrence.token, result_values):
                traced += 1
            elif path in context_set or (path.name, occurrence.token) in KNOWN_DESIGN_TOKENS:
                context_only.append(occurrence)
            else:
                untraced_outcomes.append(occurrence)

    print(f"Generated result values parsed: {len(result_values)}")
    print(f"Outcome occurrences traced: {traced}")
    print(f"Context/design occurrences not in RESULTS: {len(context_only)}")
    for occurrence in context_only:
        print(f"  context: {occurrence.path.relative_to(ROOT)}:{occurrence.line} {occurrence.token}")
    if untraced_outcomes:
        print(f"\n{len(untraced_outcomes)} UNTRACED OUTCOME NUMBER(S):")
        for occurrence in untraced_outcomes:
            print(f"  {occurrence.path.relative_to(ROOT)}:{occurrence.line} {occurrence.token}")
        return 1
    print("\nall outcome numbers trace to generated result reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
