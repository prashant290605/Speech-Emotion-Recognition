#!/usr/bin/env python
"""Build the pipeline overview figure from its TikZ source.

    python tools/make_pipeline_figure.py [--tectonic PATH]

Compiles ``tools/pipeline_figure.tex`` and writes ``figures/pipeline.pdf`` (the
vector form the manuscript includes) and ``figures/pipeline.png`` (the preview
every committed figure has). The figure is a schematic, not a data figure, so
it does not read the result ledger and ``tools/make_figures.py`` does not
regenerate it.

It uses ``pdflatex`` when available and Tectonic otherwise, looked up the way
``tools/build_paper.py`` looks it up: ``--tectonic``, then ``PATH``, then the
vendored ``.tools/tectonic``. The PNG preview needs ``pdftoppm`` (poppler);
without it the PDF is still rebuilt, the committed PNG is left in place, and
the script says so. Like ``tools/build_paper.py``, it never downloads a
compiler.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "tools" / "pipeline_figure.tex"
FIGURES = REPO_ROOT / "figures"
VENDORED_TECTONIC = REPO_ROOT / ".tools" / "tectonic" / "bin" / "tectonic.exe"


def latex_command(tectonic: str | None) -> list[str]:
    if shutil.which("pdflatex"):
        return ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                SOURCE.name]
    for candidate in (tectonic, shutil.which("tectonic"),
                      str(VENDORED_TECTONIC) if VENDORED_TECTONIC.exists() else None):
        if candidate:
            return [candidate, "-X", "compile", SOURCE.name, "--outdir", ".",
                    "--keep-logs"]
    raise SystemExit("no LaTeX engine found: install pdflatex, put Tectonic on "
                     "PATH, or pass --tectonic /path/to/tectonic")


def compile_pdf(workdir: Path, tectonic: str | None) -> Path:
    shutil.copy2(SOURCE, workdir / SOURCE.name)
    command = latex_command(tectonic)
    completed = subprocess.run(command, cwd=workdir, capture_output=True,
                               text=True, errors="replace")
    pdf = workdir / SOURCE.with_suffix(".pdf").name
    if completed.returncode != 0 or not pdf.exists():
        sys.stderr.write(completed.stdout[-3000:] + completed.stderr[-3000:])
        raise SystemExit(f"{Path(command[0]).name} failed on {SOURCE.name}")
    log = workdir / SOURCE.with_suffix(".log").name
    if log.exists() and "Overfull" in log.read_text(errors="replace"):
        raise SystemExit("overfull box in the figure: a label does not fit")
    return pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tectonic", help="path to a Tectonic binary")
    args = parser.parse_args()

    FIGURES.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pdf = compile_pdf(Path(tmp), args.tectonic)
        shutil.copy2(pdf, FIGURES / "pipeline.pdf")
    print("  wrote figures/pipeline.pdf")

    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile",
                        str(FIGURES / "pipeline.pdf"),
                        str(FIGURES / "pipeline")], check=True)
        print("  wrote figures/pipeline.png")
    else:
        print("  pdftoppm (poppler) not found: figures/pipeline.png was NOT "
              "refreshed. The PDF is the submitted form; regenerate the PNG "
              "preview where poppler is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
