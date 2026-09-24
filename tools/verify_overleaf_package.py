#!/usr/bin/env python
"""Statically validate a self-contained Elsevier/Overleaf upload archive.

Usage:
    python tools/verify_overleaf_package.py dist/Speech_Communication_submission_YYYYMMDD

This intentionally checks only facts available without a TeX installation:
archive structure, manuscript inputs, graphics, citation keys, required
BibTeX fields, and ZIP/staging consistency. A PDF compile remains the final
typesetting check.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.refs import parse_bibtex  # noqa: E402

INPUT = re.compile(r"\\input\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^]]*\])*\{([^}]+)\}")
ENTRY = re.compile(r"@(\w+)\{([^,]+),", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Flat staging directory")
    args = parser.parse_args()
    stage = args.package.resolve()
    zip_path = stage.with_suffix(".zip")
    problems: list[str] = []
    notes: list[str] = []

    if not stage.is_dir():
        parser.error(f"staging directory not found: {stage}")
    if not zip_path.is_file():
        problems.append(f"missing ZIP: {zip_path}")

    files = {
        path.relative_to(stage).as_posix(): path
        for path in stage.rglob("*")
        if path.is_file()
    }
    root_files = {path.name: path for path in stage.iterdir() if path.is_file()}
    nested_files = {name for name in files if "/" in name}
    allowed_nested = {"thumbnails/cas-email.jpeg"}
    for name in sorted(nested_files - allowed_nested):
        problems.append(f"unexpected nested file: {name}")
    for name in sorted(allowed_nested - nested_files):
        problems.append(f"missing required CAS asset: {name}")
    for required in ("main.tex", "refs.bib", "cas-dc.cls", "cas-common.sty",
                     "cas-model2-names.bst"):
        if required not in root_files:
            problems.append(f"missing required file: {required}")

    tex_files = [path for path in root_files.values() if path.suffix == ".tex"]
    text_by_file = {path.name: path.read_text(encoding="utf-8-sig") for path in tex_files}
    body = "\n".join(text_by_file.values())
    main_text = text_by_file.get("main.tex", "")
    if not re.search(r"\\documentclass(?:\[[^]]*\])?\{cas-dc\}", main_text):
        problems.append("main.tex does not use the official cas-dc class")
    if "\\bibliographystyle{cas-model2-names}" not in main_text:
        problems.append("main.tex does not use the CAS author-date bibliography style")

    for source, text in text_by_file.items():
        for target in INPUT.findall(text):
            if "/" in target or "\\" in target or ".." in target:
                problems.append(f"{source}: non-flat input path: {target}")
            elif f"{target}.tex" not in root_files and target not in root_files:
                problems.append(f"{source}: input does not resolve: {target}")
        for target in GRAPHIC.findall(text):
            name = target if Path(target).suffix else f"{target}.pdf"
            if "/" in target or "\\" in target or ".." in target:
                problems.append(f"{source}: non-flat graphic path: {target}")
            elif name not in root_files:
                problems.append(f"{source}: graphic does not resolve: {target}")
        chars = sorted({char for char in text if ord(char) > 127})
        if chars:
            notes.append(f"{source}: UTF-8 characters present: {''.join(chars)!r}")

    if "refs.bib" in root_files:
        bib_text = root_files["refs.bib"].read_text(encoding="utf-8-sig")
        chars = sorted({char for char in bib_text if ord(char) > 127})
        if chars:
            notes.append(f"refs.bib: UTF-8 characters present: {''.join(chars)!r}")
        references = parse_bibtex(bib_text)
        keys = {reference.key for reference in references}
        cited = {
            key.strip()
            for match in CITE.finditer(body)
            for key in match.group(1).split(",")
            if key.strip()
        }
        for key in sorted(cited - keys):
            problems.append(f"citation has no BibTeX record: {key}")
        for key in sorted(keys - cited):
            problems.append(f"uncited BibTeX record: {key}")
        raw_by_key = {
            match.group(2): match.group(1).lower()
            for match in ENTRY.finditer(bib_text)
        }
        for reference in references:
            if not reference.authors_raw or not reference.title or not reference.year:
                problems.append(f"{reference.key}: missing author, title, or year")
            kind = raw_by_key.get(reference.key, "")
            if kind == "article" and not reference.venue:
                problems.append(f"{reference.key}: article has no journal")
            if kind in {"inproceedings", "incollection"} and not reference.venue:
                problems.append(f"{reference.key}: proceedings entry has no booktitle")

    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
        zip_nested = {name for name in names if "/" in name}
        if zip_nested != allowed_nested:
            problems.append("ZIP contains files outside the required CAS thumbnail path")
        if set(names) != set(files):
            problems.append("ZIP contents do not match the staging directory")

    print(f"staging files: {len(files)}")
    print(f"TeX files: {len(tex_files)}")
    print(f"PDF figures: {sum(path.suffix == '.pdf' for path in files.values())}")
    for note in notes:
        print(f"note: {note}")
    if problems:
        print("\nPACKAGE PROBLEMS:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("\npackage checks passed (a TeX compile is still required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
