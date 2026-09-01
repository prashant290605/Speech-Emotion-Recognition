#!/usr/bin/env python
"""Render the fixed-reference MMD diagnostic report and manuscript table."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.latex import table, write_table  # noqa: E402
from ser.phase8 import seed_interval  # noqa: E402

LADDER = ("none", "zscore", "mean_shift", "coral", "mkmmd_diag", "mkmmd_full")
PAPER_AGG = "last"


def load_records(path: Path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def fmt(values, places=4):
    stats = seed_interval(values)
    if stats["n"] == 0:
        return "--"
    if stats["n"] == 1:
        return f"{stats['mean']:.{places}f} (n=1)"
    return f"{stats['mean']:.{places}f} [{stats['lo']:.{places}f}, {stats['hi']:.{places}f}]"


def conditional_mean(record, key):
    values = [row[key] for row in record["conditional"] if row[key] is not None]
    return float(np.mean(values)) if values else None


def grouped(records):
    out = defaultdict(list)
    for row in records:
        out[(row["source"], row["target"], row["layer_agg"], row["alignment"])].append(row)
    return out


def available_ladder(records):
    present = {record["alignment"] for record in records}
    return tuple(rung for rung in LADDER if rung in present)


def write_markdown(records):
    groups = grouped(records)
    directions = sorted({(row["source"], row["target"]) for row in records})
    aggs = sorted({row["layer_agg"] for row in records})
    rungs = available_ladder(records)
    lines = [
        "# Fixed-reference MMD diagnostics",
        "",
        "The ZCA basis and RBF bandwidth are each derived from unaligned source-train features once per pair, seed, backbone and layer aggregation. Both are then held fixed across the six rungs. Raw MMD-squared is the primary marginal statistic. The normalised value divides it by a same-distribution source half-split scale and is reported as a finite-sample aid. Conditional values have class-specific null scales, so no conditional-to-marginal ratio is reported.",
        "",
    ]
    for agg in aggs:
        lines.extend([f"## `layer_agg={agg}`", ""])
        lines.append("| direction | rung | raw marginal MMD2 | marginal MMD2 / null | mean raw conditional MMD2 | conditional / class null |")
        lines.append("|---|---|---|---|---|---|")
        for source, target in directions:
            for rung in rungs:
                rows = groups[(source, target, agg, rung)]
                raw = [row["marginal"]["raw_mmd2"] for row in rows]
                normalised = [row["marginal"]["normalised"] for row in rows]
                conditional_raw = [conditional_mean(row, "raw_mmd") for row in rows]
                conditional_effect = [conditional_mean(row, "effect_size") for row in rows]
                lines.append(
                    f"| {source}->{target} | {rung} | {fmt(raw, 5)} | {fmt(normalised, 2)} | "
                    f"{fmt(conditional_raw, 5)} | {fmt(conditional_effect, 2)} |"
                )
        lines.append("")
    output = REPO_ROOT / "reports/phase9_reference_geometry.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_manuscript_table(records):
    groups = grouped(records)
    rungs = available_ladder(records)
    rows = []
    for source, target in sorted({(row["source"], row["target"]) for row in records}):
        for rung in rungs:
            records_for_rung = groups[(source, target, PAPER_AGG, rung)]
            raw = np.mean([row["marginal"]["raw_mmd2"] for row in records_for_rung])
            effect = np.mean([row["marginal"]["normalised"] for row in records_for_rung])
            conditional_raw = np.mean([conditional_mean(row, "raw_mmd") for row in records_for_rung])
            rows.append([
                f"{source.upper()} $\\rightarrow$ {target.upper()}",
                f"\\texttt{{{rung.replace('_', r'\\_')}}}",
                f"{raw:.5f}",
                f"{effect:.2f}",
                f"{conditional_raw:.5f}",
            ])
    text = table(
        rows,
        ["pair", "rung", "marginal MMD$^2$", "marginal / null", "mean conditional MMD$^2$"],
        caption=(
            "Marginal and class-conditional MMD diagnostics in a common reference "
            "geometry. The source-train ZCA basis and source-train median RBF bandwidth "
            "are fixed for every rung within a pair and seed."
        ),
        label="decomposition",
        column_spec="llrrr",
        escape_cells=False,
        notes=[
            "Filter: HuBERT, \\texttt{layer\\_agg=last}, evaluated alignment rungs, 5 speaker-disjoint seeds. "
            "Raw MMD$^2$ is the primary diagnostic. The marginal normaliser is the mean absolute "
            "MMD$^2$ over source half-splits. Conditional MMD$^2$ is an unweighted mean over the six "
            "classes; its class-specific normalised values are reported in the generated diagnostic report, "
            "but are not divided by the marginal value."
        ],
    )
    return write_table(text, "decomposition", REPO_ROOT / "tables")


def main() -> int:
    records = load_records(REPO_ROOT / "results/phase9_reference_geometry_controls_v2.jsonl")
    report = write_markdown(records)
    table_path = write_manuscript_table(records)
    print(f"wrote {report.relative_to(REPO_ROOT)}")
    print(f"wrote {table_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
