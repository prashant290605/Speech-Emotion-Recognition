#!/usr/bin/env python
"""Generate the manuscript label-harmonisation diagnostic table from ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.latex import table, write_table  # noqa: E402

from report_calm_sensitivity import expected_count  # noqa: E402
from report_label_sensitivity_diagnostics import mean_conditional  # noqa: E402
from run_calm_sensitivity import load_spec  # noqa: E402


def display_corpus(corpus: str) -> str:
    return {"ravdess": "RAVDESS", "cremad": "CREMA-D"}.get(corpus, corpus.upper())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/label_harmonisation_summary.yaml")
    args = parser.parse_args(argv)
    summary_spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = []
    for relative_spec in summary_spec["controls"]:
        spec = load_spec(REPO_ROOT / relative_spec)
        path = REPO_ROOT / spec["diagnostics"]["output"]
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != expected_count(spec):
            raise RuntimeError(f"{path}: {len(records)}/{expected_count(spec)} records")
        control = spec["variant"]["name"].replace("_", " ")
        rungs = tuple(rung["alignment"] for rung in spec["rungs"])
        for source, target in spec["directions"]:
            for rung in rungs:
                group = [
                    record for record in records
                    if (record["source"], record["target"], record["alignment"])
                    == (source, target, rung)
                ]
                marginal = np.mean([record["marginal_raw_mmd2"] for record in group])
                conditional = np.mean([
                    mean_conditional(record, "raw_mmd") for record in group
                ])
                rows.append([
                    control,
                    f"{display_corpus(source)} $\\rightarrow$ {display_corpus(target)}",
                    f"\\texttt{{{rung}}}",
                    f"{marginal:.4f}",
                    f"{conditional:.4f}",
                ])
    text = table(
        rows,
        ["control", "pair", "rung", "marginal MMD$^2$", "mean conditional MMD$^2$"],
        caption=(
            "Label-harmonisation sensitivity of the descriptive MMD diagnostics. "
            "Each entry averages five seeds. Within one rung, marginal and "
            "class-conditional MMD$^2$ use the same source-target median RBF "
            "bandwidth. The bandwidth is recomputed after each map, so this "
            "table does not rank discrepancy across rungs under a fixed geometry."
        ),
        label="label-harmonisation-diagnostics",
        column_spec="lllrr",
        escape_cells=False,
        notes=[
            "Controls: HuBERT final-layer features, logistic regression, and "
            "\\texttt{none}, \\texttt{zscore}, \\texttt{mean\\_shift}, and "
            "\\texttt{coral} ($\\varepsilon=10.0$). Conditional values are an "
            "unweighted mean over the retained classes and read target labels only "
            "after the alignment leakage assertion. No conditional-to-marginal ratio is reported."
        ],
    )
    text = text.replace(r"\begin{table}[tb]", r"\begin{table*}[!t]", 1)
    text = text.rsplit(r"\end{table}", 1)[0] + r"\end{table*}" + "\n"
    output = write_table(
        text,
        "label_harmonisation_diagnostics",
        REPO_ROOT / "tables",
    )
    expected_output = REPO_ROOT / summary_spec["table"]
    if output != expected_output:
        raise RuntimeError(f"table path drift: {output} != {expected_output}")
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
