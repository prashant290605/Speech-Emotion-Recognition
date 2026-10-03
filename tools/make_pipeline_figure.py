#!/usr/bin/env python
"""Build the pipeline overview figure from its TikZ source.

    python tools/make_pipeline_figure.py

Compiles ``tools/pipeline_figure.tex`` and writes ``figures/pipeline.pdf`` (the
vector form the manuscript includes) and ``figures/pipeline.png`` (the preview
every committed figure has). The figure is a schematic, not a data figure, so
it does not read the result ledger and ``tools/make_figures.py`` does not
regenerate it.

It uses ``pdflatex`` when available and Tectonic otherwise, and ``pdftoppm``
for the preview. Like ``tools/build_paper.py``, it never downloads a compiler.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "tools" / "pipeline_figure.tex"
FIGURES = REPO_ROOT / "figures"


def compile_pdf(workdir: Path) -> Path:
    shutil.copy2(SOURCE, workdir / SOURCE.name)
    if shutil.which("pdflatex"):
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                   SOURCE.name]
    elif shutil.which("tectonic"):
        command = ["tectonic", SOURCE.name]
    else:
        raise SystemExit("neither pdflatex nor tectonic is on PATH")
    completed = subprocess.run(command, cwd=workdir, capture_output=True,
                               text=True, errors="replace")
    pdf = workdir / SOURCE.with_suffix(".pdf").name
    if completed.returncode != 0 or not pdf.exists():
        sys.stderr.write(completed.stdout[-3000:] + completed.stderr[-3000:])
        raise SystemExit(f"{command[0]} failed on {SOURCE.name}")
    log = workdir / SOURCE.with_suffix(".log").name
    if log.exists() and "Overfull" in log.read_text(errors="replace"):
        raise SystemExit("overfull box in the figure: a label does not fit")
    return pdf


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pdf = compile_pdf(Path(tmp))
        shutil.copy2(pdf, FIGURES / "pipeline.pdf")
        if not shutil.which("pdftoppm"):
            raise SystemExit("pdftoppm (poppler) is needed for the PNG preview")
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile",
                        str(FIGURES / "pipeline.pdf"),
                        str(FIGURES / "pipeline")], check=True)
    for name in ("pipeline.pdf", "pipeline.png"):
        print(f"  wrote figures/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
