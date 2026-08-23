#!/usr/bin/env python
"""Settle frame dependence and depth divergence from the full-seed sweep.

    python tools/layer_sweep_v2_report.py

Writes reports/layer_sweep_v2.md.

Both questions are answered **per seed and then aggregated**, not from a single
correlation over pooled means. The Stage 1 version computed one rho per backbone
from 13 layer-means and reported its sign; with 5 seeds the sign itself can be
given an interval, which is the difference between "the frames disagreed once"
and "the frames disagree".
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.phase8 import seed_interval  # noqa: E402
from ser.utils.results import read_rows  # noqa: E402

LADDER = ["none", "zscore", "mean_shift", "coral", "mkmmd_diag", "mkmmd_full"]


def spearman(x, y):
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for position, index in enumerate(order):
            ranks[index] = float(position)
        return ranks

    rx, ry = rank(x), rank(y)
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load():
    rows = []
    for path in sorted(glob.glob(str(REPO_ROOT / "results/shards/sweep2_shard*.jsonl"))):
        rows += [r for r in read_rows(path) if r["status"] == "ok"]
    unique = {}
    for row in rows:
        unique[row["run_id"]] = row
    return list(unique.values())


def main() -> int:
    rows = load()
    if not rows:
        print("no sweep rows yet")
        return 1

    seeds = sorted({r["seed"] for r in rows})
    backbones = sorted({r["backbone"] for r in rows})
    directions = sorted({(r["source_corpus"], r["target_corpus"]) for r in rows})
    layers = sorted({r["layer_index"] for r in rows})

    out = ["# 13-layer sweep, full seed count", ""]
    out.append(f"{len(rows)} runs: {len(layers)} layers x {len(LADDER)} rungs x "
               f"{len(backbones)} backbones x {len(directions)} directions x "
               f"{len(seeds)} seeds, logreg.\n")
    out.append("Replaces the Stage 1 sweep (2 seeds, one classifier, one pair, rung "
               "`none`). Both questions below rested entirely on that.\n")
    out.append("---\n")

    # -- Q1: frame dependence ---------------------------------------------
    out.append("## 1. Frame dependence\n")
    out.append("Spearman rho between a layer's discrepancy and its target macro-F1, "
               "computed **within each seed** across the 13 layers, then averaged. "
               "The interval is a t-interval over seeds, so a sign that is not "
               "stable across seeds cannot look stable here.\n")
    out.append("| direction | backbone | rung | rho(own geometry) | rho(reference frame) | frames disagree in sign |")
    out.append("|---|---|---|---|---|---|")

    disagreements = 0
    comparisons = 0
    for source, target in directions:
        for backbone in backbones:
            for rung in LADDER:
                own_by_seed, ref_by_seed = [], []
                for seed in seeds:
                    cells = [
                        r for r in rows
                        if (r["source_corpus"], r["target_corpus"]) == (source, target)
                        and r["backbone"] == backbone and r["alignment"] == rung
                        and r["seed"] == seed
                    ]
                    if len(cells) < 3:
                        continue
                    cells.sort(key=lambda r: r["layer_index"])
                    target_scores = [r["macro_f1"] for r in cells]
                    own_by_seed.append(
                        spearman([r["marginal_mmd_normalised"] for r in cells], target_scores)
                    )
                    ref_by_seed.append(
                        spearman([r["marginal_mmd_reference"] for r in cells], target_scores)
                    )
                if not own_by_seed:
                    continue
                own = seed_interval(own_by_seed)
                ref = seed_interval(ref_by_seed)
                comparisons += 1
                # A sign disagreement counts only when BOTH intervals exclude
                # zero and they fall on opposite sides. Two point estimates of
                # opposite sign whose intervals both straddle zero is not a
                # disagreement, it is noise.
                own_sig = own["lo"] * own["hi"] > 0
                ref_sig = ref["lo"] * ref["hi"] > 0
                disagree = own_sig and ref_sig and own["mean"] * ref["mean"] < 0
                disagreements += disagree
                out.append(
                    f"| {source[:4]}->{target[:4]} | {backbone} | {rung} | "
                    f"{own['mean']:+.3f} [{own['lo']:+.3f}, {own['hi']:+.3f}] | "
                    f"{ref['mean']:+.3f} [{ref['lo']:+.3f}, {ref['hi']:+.3f}] | "
                    f"{'**YES**' if disagree else 'no'} |"
                )
    out.append("")
    out.append(f"**Sign disagreements: {disagreements} of {comparisons} "
               f"(backbone x rung x direction) cells.** A cell counts only when "
               "both intervals exclude zero and fall on opposite sides; two "
               "opposite-signed point estimates whose intervals straddle zero are "
               "noise, not disagreement.\n")

    # -- Q2: depth divergence ---------------------------------------------
    out.append("## 2. Depth divergence\n")
    out.append("Argmax layer on `source_val` against argmax layer on target "
               "macro-F1, computed per seed. The gap is reported as a mean with a "
               "t-interval over seeds rather than as a single number from pooled "
               "means, which is how the Stage 1 version reported it.\n")
    out.append("| direction | backbone | rung | argmax source_val | argmax target | gap (layers) | cost of choosing on source_val |")
    out.append("|---|---|---|---|---|---|---|")
    for source, target in directions:
        for backbone in backbones:
            for rung in LADDER:
                gaps, costs, val_args, tgt_args = [], [], [], []
                for seed in seeds:
                    cells = [
                        r for r in rows
                        if (r["source_corpus"], r["target_corpus"]) == (source, target)
                        and r["backbone"] == backbone and r["alignment"] == rung
                        and r["seed"] == seed
                    ]
                    if len(cells) < 3:
                        continue
                    best_val = max(cells, key=lambda r: r["selection_source_val_macro_f1"])
                    best_tgt = max(cells, key=lambda r: r["macro_f1"])
                    val_args.append(best_val["layer_index"])
                    tgt_args.append(best_tgt["layer_index"])
                    gaps.append(best_tgt["layer_index"] - best_val["layer_index"])
                    costs.append(best_tgt["macro_f1"] - best_val["macro_f1"])
                if not gaps:
                    continue
                gap = seed_interval([float(g) for g in gaps])
                cost = seed_interval(costs)
                out.append(
                    f"| {source[:4]}->{target[:4]} | {backbone} | {rung} | "
                    f"{np.median(val_args):.0f} | {np.median(tgt_args):.0f} | "
                    f"{gap['mean']:+.1f} [{gap['lo']:+.1f}, {gap['hi']:+.1f}] | "
                    f"{cost['mean']:+.4f} [{cost['lo']:+.4f}, {cost['hi']:+.4f}] |"
                )
    out.append("")

    # -- the curve itself --------------------------------------------------
    out.append("## 3. The curve, rung `none`, averaged over seeds\n")
    for source, target in directions:
        out.append(f"### {source} -> {target}\n")
        out.append("| layer | " + " | ".join(f"{b} val / target" for b in backbones) + " |")
        out.append("|---" * (len(backbones) + 1) + "|")
        for layer in layers:
            cells = []
            for backbone in backbones:
                g = [r for r in rows
                     if (r["source_corpus"], r["target_corpus"]) == (source, target)
                     and r["backbone"] == backbone and r["alignment"] == "none"
                     and r["layer_index"] == layer]
                cells.append(
                    f"{np.mean([r['selection_source_val_macro_f1'] for r in g]):.3f} / "
                    f"**{np.mean([r['macro_f1'] for r in g]):.3f}**" if g else "--"
                )
            out.append(f"| {layer} | " + " | ".join(cells) + " |")
        out.append("")

    path = REPO_ROOT / "reports/layer_sweep_v2.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path} ({len(rows)} runs, {disagreements}/{comparisons} sign disagreements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
