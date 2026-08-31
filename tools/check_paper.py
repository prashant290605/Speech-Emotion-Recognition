#!/usr/bin/env python
"""Structural validation of the manuscript, in place of a LaTeX compile.

    python tools/check_paper.py

No TeX distribution is installed on the machine this was written on, so the
manuscript cannot be compiled here. This checks everything a compiler would
have caught that does not need TeX itself:

* every ``\\input`` resolves to a file that exists;
* every ``\\includegraphics`` resolves, given ``\\graphicspath``;
* every ``\\cite`` key exists in refs.bib, and every bib entry is cited;
* every ``\\ref`` has a matching ``\\label``;
* environments are balanced;
* no stray control character has been written into the source (a real defect
  in this project: a shell heredoc turned ``\\ref`` into a carriage return);
* every ``[CITE: ...]`` placeholder is listed in CITATIONS\\_NEEDED.md.

It does not check that the document typesets. That still needs a compile, and
the checklist says so.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper"


def strip_comments(text: str) -> str:
    """Remove LaTeX comments without eating an escaped percent."""
    return re.sub(r"(?<!\\)%.*", "", text)


def sources():
    """main.tex plus every file it \\inputs, in order."""
    main = PAPER / "main.tex"
    files = [main]
    for match in re.finditer(r"\\input\{([^}]+)\}", strip_comments(main.read_text(encoding="utf-8"))):
        target = match.group(1)
        if target.startswith("../"):
            continue  # generated tables, checked separately
        path = PAPER / (target if target.endswith(".tex") else target + ".tex")
        if path.exists():
            files.append(path)
    return files


def main() -> int:
    problems, notes = [], []

    if not (PAPER / "main.tex").exists():
        print("paper/main.tex missing")
        return 2

    files = sources()
    body = "\n".join(strip_comments(f.read_text(encoding="utf-8")) for f in files)

    # -- inputs resolve ----------------------------------------------------
    for path in files:
        raw = strip_comments(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"\\input\{([^}]+)\}", raw):
            target = match.group(1)
            # LaTeX resolves \input relative to the directory of the job's
            # main file, not the including file, so ../tables/x from a file in
            # sections/ still means <repo>/tables/x.
            candidate = (PAPER / target)
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            if not candidate.exists():
                problems.append(f"{path.name}: \\input{{{target}}} does not resolve")

    # -- graphics resolve --------------------------------------------------
    graphics_dirs = [PAPER / d for d in
                     re.findall(r"\\graphicspath\{\{([^}]+)\}\}", body)] or [PAPER]
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        name = match.group(1)
        if not any((d / name).exists() or (d / (name + ".pdf")).exists()
                   for d in graphics_dirs):
            problems.append(f"\\includegraphics{{{name}}} does not resolve")

    # -- citations ---------------------------------------------------------
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for match in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
        cited |= {k.strip() for k in match.group(1).split(",")}
    for key in sorted(cited - defined):
        problems.append(f"\\cite{{{key}}} has no entry in refs.bib")
    for key in sorted(defined - cited):
        notes.append(f"refs.bib entry never cited: {key}")

    # entries the checker deleted must not reappear
    for banned in ("w2vprosody2023", "li2023cross", "jafari2025feature"):
        if banned in defined:
            problems.append(f"refs.bib contains {banned}, which Phase 1 removed "
                            "or corrected")

    # -- refs / labels -----------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for table in (REPO_ROOT / "tables").glob("*.tex"):
        labels |= set(re.findall(r"\\label\{([^}]+)\}",
                                 table.read_text(encoding="utf-8")))
    # A dangling \ref into a section that is still a placeholder is expected
    # while the manuscript is being drafted section by section. It becomes a
    # defect once every section has been written, and the count below is what
    # makes that transition automatic rather than remembered.
    undrafted = {f.stem for f in files
                 if f.read_text(encoding="utf-8").lstrip().startswith("% placeholder")}
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    for key in sorted(refs - labels):
        if key.startswith("sec:") and undrafted:
            notes.append(f"forward \\ref{{{key}}} -- "
                        f"{len(undrafted)} section(s) still undrafted")
        else:
            problems.append(f"\\ref{{{key}}} has no \\label")
    for key in sorted(label for label in labels if label.startswith(("fig:", "tab:"))
                      and label not in refs):
        problems.append(f"\\label{{{key}}} is never referenced in the text")

    # -- balance -----------------------------------------------------------
    for name in ("document", "table", "figure", "tabular", "equation",
                 "description", "itemize", "enumerate", "abstract"):
        opened = len(re.findall(r"\\begin\{" + name + r"\*?\}", body))
        closed = len(re.findall(r"\\end\{" + name + r"\*?\}", body))
        if opened != closed:
            problems.append(f"environment {name}: {opened} begin vs {closed} end")

    # -- stray control characters -----------------------------------------
    for path in files:
        with open(path, encoding="utf-8", newline="") as handle:
            raw = handle.read()
        for bad, what in ((chr(13), "carriage return"), (chr(9), "tab"),
                          (chr(12), "form feed")):
            if bad in raw.replace(chr(13) + chr(10), ""):
                problems.append(f"{path.name}: contains a literal {what} -- a "
                                "collapsed backslash escape")

    # -- collapsed line breaks ---------------------------------------------
    # A line ending in exactly one backslash escapes the newline rather than
    # breaking the line. It is almost always a `\\` that lost a backslash on
    # the way in, and it happened for real in the title block of this paper.
    backslash = chr(92)
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            stripped = line.rstrip()
            if stripped.endswith(backslash) and not stripped.endswith(backslash * 2):
                problems.append(f"{path.name}:{number}: line ends in a single "
                                "backslash; a line break needs two")

    # -- placeholders ------------------------------------------------------
    placeholders = re.findall(r"\[CITE:\s*([^\]]+)\]", body)
    listed = ""
    needed = PAPER / "CITATIONS_NEEDED.md"
    if needed.exists():
        listed = needed.read_text(encoding="utf-8")
    lowered = listed.lower()
    for text in placeholders:
        words = [w for w in re.findall(r"[a-z]{4,}", text.lower())
                 if w not in {"reference", "standard", "with", "that", "from"}]
        distinctive = words[:4]
        if distinctive and not any(w in lowered for w in distinctive):
            problems.append(f"[CITE] placeholder not listed in "
                            f"CITATIONS_NEEDED.md: {text.strip()[:60]}...")

    # -- report ------------------------------------------------------------
    print(f"checked {len(files)} source file(s): "
          + ", ".join(f.name for f in files))
    print(f"  citations used   {len(cited)} / {len(defined)} bib entries")
    print(f"  labels defined   {len(labels)}")
    print(f"  CITE placeholders {len(placeholders)}")
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nno structural problems found "
          "(this is not a substitute for a LaTeX compile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
