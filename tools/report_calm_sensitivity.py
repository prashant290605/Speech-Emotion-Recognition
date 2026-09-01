#!/usr/bin/env python
"""Summarise the pre-specified calm-drop label sensitivity from its own ledger."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.phase8 import seed_interval  # noqa: E402


ORDER = ("none", "zscore", "mean_shift", "coral")


def main() -> int:
    path = REPO_ROOT / "results/calm_dropped_sensitivity.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = 40
    if len(rows) != expected or any(row["status"] != "ok" for row in rows):
        raise RuntimeError(f"sensitivity incomplete: {len(rows)}/{expected} rows")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_corpus"], row["target_corpus"], row["alignment"])].append(row)

    lines = [
        "# RAVDESS calm-drop sensitivity",
        "",
        "This pre-specified sensitivity drops RAVDESS `calm` rather than merging it into `neutral`. It reuses cached HuBERT final-layer features, runs logistic regression, fixes CORAL at epsilon 10.0, and scores five speaker-disjoint seeds in each transfer direction. It is a robustness control, not a replacement full classifier grid.",
        "",
        "| direction | alignment | target macro-F1 | source-val macro-F1 | chance baseline | n train | n target test |",
        "|---|---|---|---|---|---|---|",
    ]
    for direction in (("ravdess", "cremad"), ("cremad", "ravdess")):
        for alignment in ORDER:
            group = groups[(*direction, alignment)]
            target = seed_interval([row["macro_f1"] for row in group])
            source = seed_interval([row["selection_source_val_macro_f1"] for row in group])
            chance = seed_interval([row["chance_macro_f1"] for row in group])
            lines.append(
                f"| {direction[0]}->{direction[1]} | {alignment} | "
                f"{target['mean']:.4f} [{target['lo']:.4f}, {target['hi']:.4f}] | "
                f"{source['mean']:.4f} [{source['lo']:.4f}, {source['hi']:.4f}] | "
                f"{chance['mean']:.4f} | {int(group[0]['n_train'])} | {int(group[0]['n_target_test'])} |"
            )

    lines.extend(["", "## Target-score differences from `none`", ""])
    for direction in (("ravdess", "cremad"), ("cremad", "ravdess")):
        baseline = seed_interval([
            row["macro_f1"] for row in groups[(*direction, "none")]
        ])["mean"]
        deltas = []
        for alignment in ORDER[1:]:
            aligned = seed_interval([
                row["macro_f1"] for row in groups[(*direction, alignment)]
            ])["mean"]
            deltas.append(f"`{alignment}` minus `none` = {aligned - baseline:+.4f}")
        lines.append(f"- {direction[0]}->{direction[1]}: " + "; ".join(deltas) + ".")
    output = REPO_ROOT / "reports/calm_dropped_sensitivity.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_lines = [
        "\\begin{table*}[!t]",
        "  \\centering",
        "  \\caption{Target macro-F1 in the pre-specified calm-drop sensitivity. "
        "RAVDESS \\texttt{calm} is dropped rather than merged into "
        "\\texttt{neutral}. Values are means over five speaker-disjoint seeds "
        "with 95\\% $t$-intervals.}",
        "  \\label{tab:calm-sensitivity}",
        "  \\small",
        "  \\begin{tabular}{lrr}",
        "    \\toprule",
        "    rung & RAVDESS $\\rightarrow$ CREMA-D & CREMA-D $\\rightarrow$ RAVDESS " + "\\\\",
        "    \\midrule",
    ]
    for alignment in ORDER:
        cells = []
        for direction in (("ravdess", "cremad"), ("cremad", "ravdess")):
            summary = seed_interval([
                row["macro_f1"] for row in groups[(*direction, alignment)]
            ])
            cells.append(
                f"{summary['mean']:.4f} [{summary['lo']:.4f}, {summary['hi']:.4f}]"
            )
        table_lines.append(
            f"    \\texttt{{{alignment}}} & {cells[0]} & {cells[1]} " + "\\\\"
        )
    forward_chance = seed_interval([
        row["chance_macro_f1"] for row in groups[("ravdess", "cremad", "none")]
    ])["mean"]
    reverse_chance = seed_interval([
        row["chance_macro_f1"] for row in groups[("cremad", "ravdess", "none")]
    ])["mean"]
    table_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\\\[2pt]",
        "  \\begin{minipage}{\\linewidth}\\footnotesize "
        "Filter: cached HuBERT final-layer features, logistic regression, "
        "and CORAL $\\varepsilon=10.0$. The chance baselines are "
        f"{forward_chance:.4f} and {reverse_chance:.4f}. This robustness "
        "control is not a replacement full classifier grid."
        "\\end{minipage}",
        "\\end{table*}",
    ])
    table_output = REPO_ROOT / "tables/calm_sensitivity.tex"
    table_output.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    print(f"wrote {table_output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
