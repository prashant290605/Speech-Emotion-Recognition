"""Trace manuscript outcome numbers to reports/RESULTS.md.

Usage:
    python tools/check_number_trace.py

The paper contains two numeric classes. Outcome numbers in the abstract,
results, discussion, reproducibility section, conclusion, highlights, and
result tables must be in RESULTS.md (allowing ordinary display rounding).
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
RESULTS = ROOT / "reports" / "RESULTS.md"
NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![A-Za-z_])")
# Fixed analysis parameters which appear in an otherwise outcome-bearing table.
KNOWN_DESIGN_TOKENS = {("decomposition.tex", "50")}


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
    if "e" in token.lower() or "." not in token:
        return False
    places = len(token.lower().split("e", 1)[0].split(".", 1)[1])
    tolerance = Decimal("0.5") * (Decimal(10) ** -places)
    return any(abs(value - candidate) < tolerance for candidate in result_values)


def source_files() -> tuple[list[Path], list[Path]]:
    outcomes = [PAPER / "main.tex", PAPER / "highlights.txt"]
    outcomes.extend(
        path for path in sorted((PAPER / "sections").glob("*.tex"))
        if path.name != "methods.tex"
    )
    outcomes.extend(
        path for path in sorted((ROOT / "tables").glob("*.tex"))
        if path.name != "corpora.tex"
    )
    context = [PAPER / "sections" / "methods.tex", ROOT / "tables" / "corpora.tex"]
    return outcomes, context


def main() -> int:
    if not RESULTS.exists():
        print(f"missing {RESULTS.relative_to(ROOT)}")
        return 2

    result_values = {
        value
        for match in NUMBER.finditer(text_without_noncontent(
            RESULTS.read_text(encoding="utf-8"), strip_latex_comments=False
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

    print(f"RESULTS values parsed: {len(result_values)}")
    print(f"Outcome occurrences traced: {traced}")
    print(f"Context/design occurrences not in RESULTS: {len(context_only)}")
    for occurrence in context_only:
        print(f"  context: {occurrence.path.relative_to(ROOT)}:{occurrence.line} {occurrence.token}")
    if untraced_outcomes:
        print(f"\n{len(untraced_outcomes)} UNTRACED OUTCOME NUMBER(S):")
        for occurrence in untraced_outcomes:
            print(f"  {occurrence.path.relative_to(ROOT)}:{occurrence.line} {occurrence.token}")
        return 1
    print("\nall outcome numbers trace to reports/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
