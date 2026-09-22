#!/usr/bin/env python
"""One command that regenerates the analysis and, optionally, compiles the paper.

    python tools/build_paper.py --check-only   # verify inputs, generate nothing
    python tools/build_paper.py --analysis     # regenerate reports/tables/figures
    python tools/build_paper.py                # the above, then compile the PDF

Before this existed, the build was a sequence recorded in prose in
``paper/BUILD_REPORT.md`` and a LaTeX compiler invoked by hand from a
gitignored directory. Nothing checked that the analysis had been regenerated
from the current ledger, and nothing checked the ledger had not moved.

Stages, in order, each refusing to continue when the one before it failed:

  1. frozen-ledger digest
  2. tracked analysis artifacts present
  3. reports          (per-class, results document)
  4. tables, figures
  5. translation audit
  6. structural check (tools/check_paper.py)
  7. number trace     (tools/check_number_trace.py)
  8. LaTeX compile    (Tectonic, if available)
  9. log inspection   (undefined references, citations, overfull boxes)

Deliberate non-behaviours. It does not download a compiler or any package: a
build that silently acquires a dependency is not a reproducible build, so a
missing Tectonic is reported with where to get it and the run stops at stage 7
rather than proceeding halfway. It does not write to any frozen ledger. It does
not pass ``--freeze-tag`` anywhere by default, so the published build always
resolves to the publication tag named in :mod:`ser.artifacts`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser import artifacts  # noqa: E402
from ser.freeze import LedgerDrift, assert_ledger_unchanged  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "output" / "paper"
VENDORED_TECTONIC = REPO_ROOT / ".tools" / "tectonic" / "bin" / "tectonic.exe"

# Section files and generated tables the compile needs at the flattened root.
PAPER_DIR = REPO_ROOT / "paper"


class BuildError(RuntimeError):
    """A stage failed. The build stops rather than producing a partial PDF."""


def banner(step: str, title: str) -> None:
    print(f"\n=== [{step}] {title} " + "=" * max(0, 58 - len(title)), flush=True)


def run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict | None = None) -> str:
    merged = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), **(env or {})}
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd), env=merged,
                               capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    label = " ".join(Path(c).name if c.endswith(".py") else c for c in command[:3])
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout[-4000:])
        sys.stderr.write(completed.stderr[-4000:])
        raise BuildError(f"{label} failed with exit code {completed.returncode}")
    print(f"  ok ({elapsed:.1f}s) {label}")
    return completed.stdout


# -- stages ----------------------------------------------------------------
def stage_ledger() -> None:
    banner("1/9", "frozen ledger digest")
    try:
        digest = assert_ledger_unchanged()
    except LedgerDrift as exc:
        raise BuildError(str(exc)) from exc
    print(f"  ok  results/runs.jsonl {digest}")


def stage_artifacts(strict: bool) -> list[tuple[str, str]]:
    banner("2/9", "tracked analysis artifacts")
    missing = artifacts.missing()
    for path, why in artifacts.REQUIRED_ARTIFACTS:
        mark = "MISSING" if (path, why) in missing else "ok     "
        print(f"  {mark} {path:<52} {why}")
    if missing and strict:
        raise BuildError(
            f"{len(missing)} required artifact(s) absent: "
            + ", ".join(path for path, _ in missing)
            + ". Build them with tools/merge_layer_sweep.py, "
              "tools/make_portable_manifest.py and "
              "tools/make_speaker_confusions.py on a machine that has the "
              "source data."
        )
    return missing


def stage_reports() -> None:
    banner("3/9", "reports")
    run([sys.executable, "tools/phase10_per_class.py"])
    run([sys.executable, "tools/make_results_doc.py"])


def stage_tables_and_figures() -> None:
    banner("4/9", "tables and figures")
    run([sys.executable, "tools/make_figures.py"])
    run([sys.executable, "tools/make_tables.py"])


def stage_audit() -> None:
    banner("5/9", "retrospective translation audit")
    run([sys.executable, "tools/audit_translation.py"])


def stage_checks() -> None:
    banner("6/9", "manuscript structural check")
    out = run([sys.executable, "tools/check_paper.py"])
    print("  " + out.strip().splitlines()[-1])
    banner("7/9", "number trace")
    out = run([sys.executable, "tools/check_number_trace.py"])
    print("  " + out.strip().splitlines()[-1])


def find_tectonic(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).exists():
            raise BuildError(f"--tectonic {explicit} does not exist")
        return explicit
    found = shutil.which("tectonic")
    if found:
        return found
    if VENDORED_TECTONIC.exists():
        return str(VENDORED_TECTONIC)
    raise BuildError(
        "Tectonic not found. Install it from https://tectonic-typesetting.github.io "
        "and put it on PATH, or pass --tectonic /path/to/tectonic. This script "
        "will not download a compiler: a build that acquires its own toolchain "
        "is not reproducible. Everything except the PDF has been produced; "
        "rerun with --analysis to skip the compile."
    )


def stage_compile(tectonic: str) -> Path:
    banner("8/9", "LaTeX compile")
    build = OUTPUT_DIR / "build"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    for source in [PAPER_DIR / "main.tex", PAPER_DIR / "refs.bib",
                   PAPER_DIR / "supplementary_provenance.tex"]:
        shutil.copy2(source, build)
    for source in (PAPER_DIR / "sections").glob("*.tex"):
        shutil.copy2(source, build)
    for source in (REPO_ROOT / "tables").glob("*.tex"):
        shutil.copy2(source, build)
    for source in (REPO_ROOT / "figures").glob("*.pdf"):
        shutil.copy2(source, build)
    vendor = PAPER_DIR / "vendor" / "cas"
    for name in ("cas-dc.cls", "cas-common.sty", "cas-model2-names.bst"):
        shutil.copy2(vendor / name, build)
    if (vendor / "thumbnails").exists():
        shutil.copytree(vendor / "thumbnails", build / "thumbnails", dirs_exist_ok=True)

    # The repository lays sections and tables out in subdirectories; the build
    # directory is flat, like the Overleaf package, so inputs are rewritten
    # rather than the manuscript carrying two path conventions.
    for tex in build.glob("*.tex"):
        text = tex.read_text(encoding="utf-8")
        rewritten = (text.replace(r"\graphicspath{{../figures/}}", r"\graphicspath{{./}}")
                         .replace(r"\input{sections/", r"\input{")
                         .replace(r"\input{../tables/", r"\input{"))
        if rewritten != text:
            tex.write_text(rewritten, encoding="utf-8", newline="\n")

    completed = subprocess.run(
        [tectonic, "-X", "compile", "main.tex", "--outdir", ".", "--keep-logs"],
        cwd=str(build), capture_output=True, text=True,
    )
    pdf = build / "main.pdf"
    if not pdf.exists():
        sys.stderr.write(completed.stderr[-4000:])
        raise BuildError("Tectonic produced no PDF")
    print(f"  ok  {pdf.relative_to(REPO_ROOT)}")
    return build


def stage_inspect(build: Path) -> dict:
    banner("9/9", "compile log inspection")
    log = (build / "main.log").read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on main\.xdv \((\d+) pages", log)
    undefined_ref = re.findall(r"Warning: Reference `([^']+)' .*undefined", log)
    undefined_cite = re.findall(r"Warning: Citation `([^']+)' .*undefined", log)
    overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*", log)

    report = {
        "pages": int(pages.group(1)) if pages else None,
        "undefined_references": sorted(set(undefined_ref)),
        "undefined_citations": sorted(set(undefined_cite)),
        "overfull_boxes": len(overfull),
        "overfull_detail": overfull,
    }
    print(f"  pages                 {report['pages']}")
    print(f"  undefined references  {len(report['undefined_references'])}")
    print(f"  undefined citations   {len(report['undefined_citations'])}")
    print(f"  overfull hboxes       {report['overfull_boxes']}")
    for item in overfull:
        print(f"    {item}")
    if report["undefined_references"] or report["undefined_citations"]:
        raise BuildError(
            "undefined references/citations in the compiled document: "
            f"{report['undefined_references']} {report['undefined_citations']}"
        )

    final = OUTPUT_DIR / "Speech_Communication.pdf"
    shutil.copy2(build / "main.pdf", final)
    print(f"  wrote {final.relative_to(REPO_ROOT)}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true",
                      help="verify ledger, artifacts and manuscript; generate nothing")
    mode.add_argument("--analysis", action="store_true",
                      help="regenerate reports, tables and figures; skip the LaTeX compile")
    parser.add_argument("--tectonic", help="path to a Tectonic binary")
    parser.add_argument("--allow-missing-artifacts", action="store_true",
                        help="continue when a tracked artifact is absent (degraded run)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_ledger()
    missing = stage_artifacts(strict=not args.allow_missing_artifacts and not args.check_only)

    if args.check_only:
        stage_checks()
        banner("--", "check-only complete")
        if missing:
            print(f"  {len(missing)} artifact(s) absent; a full build would stop here.")
            return 1
        print("  ledger, artifacts, structure and number trace all verified.")
        return 0

    stage_reports()
    stage_tables_and_figures()
    stage_audit()
    stage_checks()

    if args.analysis:
        banner("--", "analysis complete")
        print("  reports, tables, figures and the audit are regenerated.")
        print("  run without --analysis to compile the PDF.")
        return 0

    build = stage_compile(find_tectonic(args.tectonic))
    report = stage_inspect(build)
    banner("--", "build complete")
    print(f"  {report['pages']} pages, {report['overfull_boxes']} overfull box(es)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
