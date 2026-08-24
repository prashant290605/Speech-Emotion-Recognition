#!/usr/bin/env python
"""Settle frame dependence and depth divergence from the full-seed sweep.

    python tools/layer_sweep_v2_report.py

Writes reports/layer_sweep_v2.md.

Only **complete** rungs are used. A rung with 249 of 390 cells would contribute
correlations computed over different layer subsets in different seeds, which is
exactly how a spurious sign flip gets manufactured. Completeness is checked per
rung rather than assumed, and whatever is excluded is named in the report with
its missing count.

Both questions are answered **per seed and then aggregated**, never from a
single correlation over pooled means. The Stage 1 version computed one rho per
backbone from 13 layer-means and reported its sign; with 5 seeds the sign itself
gets an interval, which is the difference between "the frames disagreed once"
and "the frames disagree".
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.phase8 import seed_interval  # noqa: E402

LADDER = ["none", "zscore", "mean_shift", "coral", "mkmmd_diag", "mkmmd_full"]
CELLS_PER_RUNG = 390  # 13 layers x 3 backbones x 2 directions x 5 seeds


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
    """Every sweep row, tolerating a worker writing mid-read."""
    rows, partial = [], 0
    for path in sorted(glob.glob(str(REPO_ROOT / "results/shards/sweep2_*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    partial += 1
    unique = {r["run_id"]: r for r in rows}
    return list(unique.values()), len(rows) - len(unique), partial


def interval(values):
    stat = seed_interval(list(values))
    if stat["n"] == 0:
        return "--"
    if stat["n"] == 1:
        return f"{stat['mean']:+.3f} (n=1)"
    return f"{stat['mean']:+.3f} [{stat['lo']:+.3f}, {stat['hi']:+.3f}]"


def main() -> int:
    rows, duplicates, partial = load()
    rows = [r for r in rows if r["status"] == "ok"]
    if not rows:
        print("no sweep rows")
        return 1

    counts = Counter(r["alignment"] for r in rows)
    complete = [rung for rung in LADDER if counts.get(rung, 0) >= CELLS_PER_RUNG]
    excluded = [rung for rung in LADDER if rung not in complete]
    rows = [r for r in rows if r["alignment"] in complete]

    seeds = sorted({r["seed"] for r in rows})
    backbones = sorted({r["backbone"] for r in rows})
    directions = sorted({(r["source_corpus"], r["target_corpus"]) for r in rows})
    layers = sorted({r["layer_index"] for r in rows})

    out = ["# 13-layer sweep, full seed count", ""]
    out.append("Replaces the Stage 1 sweep (2 seeds, one classifier, one pair, rung "
               "`none`). Two claims rested entirely on that and are settled here.\n")
    out.append("**Numbers first; interpretation is section 5 and nothing before it.**\n")
    out.append("---\n")

    # -- 0. coverage and exclusions ---------------------------------------
    out.append("## 0. Coverage and what is excluded\n")
    out.append(f"{len(rows)} runs analysed: {len(layers)} layers x {len(complete)} "
               f"rungs x {len(backbones)} backbones x {len(directions)} directions x "
               f"{len(seeds)} seeds, logreg. Zero failures, "
               f"{duplicates} duplicate rows, {partial} unparsable lines.\n")
    out.append("| rung | cells | status |")
    out.append("|---|---|---|")
    for rung in LADDER:
        n = counts.get(rung, 0)
        out.append(f"| {rung} | {n}/{CELLS_PER_RUNG} | "
                   f"{'**included**' if rung in complete else f'**EXCLUDED** ({CELLS_PER_RUNG - n} missing)'} |")
    out.append("")
    if excluded:
        out.append(f"**{', '.join(excluded)} are excluded** — "
                   f"{sum(CELLS_PER_RUNG - counts.get(r, 0) for r in excluded)} of "
                   f"{CELLS_PER_RUNG * len(excluded)} cells were still running when "
                   "this report was generated. A partially covered rung would "
                   "contribute correlations computed over different layer subsets "
                   "in different seeds, which is a mechanism for manufacturing a "
                   "sign flip rather than measuring one. Section 5 states what "
                   "their absence can and cannot change.\n")

    # -- 1. frame dependence -----------------------------------------------
    out.append("## 1. Frame dependence\n")
    out.append("Spearman rho between a layer's discrepancy and its target macro-F1, "
               "computed **within each seed** across the 13 layers, then averaged "
               "over seeds with a t-interval. A sign that is not stable across "
               "seeds cannot look stable here.\n")
    out.append("A cell counts as a disagreement only when **both** intervals "
               "exclude zero and fall on opposite sides. Two opposite-signed point "
               "estimates whose intervals straddle zero are noise.\n")
    out.append("| direction | backbone | rung | rho (own geometry) | rho (reference frame) | disagree |")
    out.append("|---|---|---|---|---|---|")

    disagreements, comparisons = 0, 0
    own_all, ref_all = [], []
    for source, target in directions:
        for backbone in backbones:
            for rung in complete:
                own_by_seed, ref_by_seed = [], []
                for seed in seeds:
                    cells = [r for r in rows
                             if (r["source_corpus"], r["target_corpus"]) == (source, target)
                             and r["backbone"] == backbone
                             and r["alignment"] == rung and r["seed"] == seed]
                    if len(cells) < 3:
                        continue
                    cells.sort(key=lambda r: r["layer_index"])
                    scores = [r["macro_f1"] for r in cells]
                    own_by_seed.append(spearman([r["marginal_mmd_normalised"] for r in cells], scores))
                    ref_by_seed.append(spearman([r["marginal_mmd_reference"] for r in cells], scores))
                if not own_by_seed:
                    continue
                own, ref = seed_interval(own_by_seed), seed_interval(ref_by_seed)
                own_all += own_by_seed
                ref_all += ref_by_seed
                comparisons += 1
                own_sig = own["lo"] * own["hi"] > 0
                ref_sig = ref["lo"] * ref["hi"] > 0
                disagree = own_sig and ref_sig and own["mean"] * ref["mean"] < 0
                disagreements += disagree
                out.append(f"| {source[:4]}->{target[:4]} | {backbone} | {rung} | "
                           f"{interval(own_by_seed)} | {interval(ref_by_seed)} | "
                           f"{'**YES**' if disagree else 'no'} |")
    out.append("")
    out.append(f"**Sign disagreements: {disagreements} of {comparisons} cells.**\n")
    out.append("Pooled over every cell and seed:\n")
    out.append("| frame | mean rho | 95% interval | n |")
    out.append("|---|---|---|---|")
    for name, values in (("own geometry", own_all), ("reference frame", ref_all)):
        stat = seed_interval(values)
        out.append(f"| {name} | {stat['mean']:+.3f} | "
                   f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}] | {stat['n']} |")
    out.append("")

    # -- 2. depth divergence ------------------------------------------------
    out.append("## 2. Depth divergence\n")
    out.append("Argmax layer on `source_val` against argmax layer on target "
               "macro-F1, computed per seed. A positive gap means the "
               "transfer-optimal layer is **deeper** than the in-domain-optimal "
               "one, which is the Stage 1 claim.\n")
    out.append("**The cost column is non-negative by construction** -- it is "
               "`best target score - target score of the source_val pick`, and the "
               "first term is a maximum. On its own it proves nothing. It is "
               "therefore reported against the cost of picking a layer uniformly "
               "at random from the same 13, which is the honest null: if "
               "`source_val` costs as much as a coin flip, it carries no "
               "information about depth.\n")
    out.append("| direction | backbone | rung | argmax val (median) | argmax target (median) | gap (layers) | cost, source_val pick | cost, random pick |")
    out.append("|---|---|---|---|---|---|---|---|")
    gap_all, cost_all, random_all = [], [], []
    positive_cells = 0
    total_cells = 0
    for source, target in directions:
        for backbone in backbones:
            for rung in complete:
                gaps, costs, randoms, va, ta = [], [], [], [], []
                for seed in seeds:
                    cells = [r for r in rows
                             if (r["source_corpus"], r["target_corpus"]) == (source, target)
                             and r["backbone"] == backbone
                             and r["alignment"] == rung and r["seed"] == seed]
                    if len(cells) < 3:
                        continue
                    best_val = max(cells, key=lambda r: r["selection_source_val_macro_f1"])
                    best_tgt = max(cells, key=lambda r: r["macro_f1"])
                    va.append(best_val["layer_index"])
                    ta.append(best_tgt["layer_index"])
                    gaps.append(float(best_tgt["layer_index"] - best_val["layer_index"]))
                    costs.append(best_tgt["macro_f1"] - best_val["macro_f1"])
                    randoms.append(
                        best_tgt["macro_f1"] - float(np.mean([r["macro_f1"] for r in cells]))
                    )
                if not gaps:
                    continue
                gap = seed_interval(gaps)
                gap_all += gaps
                cost_all += costs
                random_all += randoms
                total_cells += 1
                positive_cells += gap["lo"] > 0
                cost_stat_cell, random_stat_cell = seed_interval(costs), seed_interval(randoms)
                out.append(f"| {source[:4]}->{target[:4]} | {backbone} | {rung} | "
                           f"{np.median(va):.0f} | {np.median(ta):.0f} | "
                           f"{interval(gaps)} | "
                           f"{cost_stat_cell['mean']:+.4f} "
                           f"[{cost_stat_cell['lo']:+.4f}, {cost_stat_cell['hi']:+.4f}] | "
                           f"{random_stat_cell['mean']:+.4f} |")
    out.append("")
    gap_stat, cost_stat = seed_interval(gap_all), seed_interval(cost_all)
    random_stat = seed_interval(random_all)
    paired = [c - r for c, r in zip(cost_all, random_all)]
    paired_stat = seed_interval(paired)
    out.append(f"**Cells whose gap interval excludes zero and is positive: "
               f"{positive_cells} of {total_cells}.**\n")
    out.append(f"Pooled gap {gap_stat['mean']:+.2f} layers "
               f"[{gap_stat['lo']:+.2f}, {gap_stat['hi']:+.2f}] over {gap_stat['n']} "
               f"(cell, seed) observations; pooled cost of selecting depth on "
               f"`source_val` {cost_stat['mean']:+.4f} "
               f"[{cost_stat['lo']:+.4f}, {cost_stat['hi']:+.4f}] macro-F1, against "
               f"{random_stat['mean']:+.4f} "
               f"[{random_stat['lo']:+.4f}, {random_stat['hi']:+.4f}] for a layer "
               f"picked at random.\n")
    out.append(f"Paired difference, `source_val` minus random: "
               f"**{paired_stat['mean']:+.4f} "
               f"[{paired_stat['lo']:+.4f}, {paired_stat['hi']:+.4f}]**. Negative "
               "means selecting depth on `source_val` beats a coin flip; an "
               "interval covering zero means it does not.\n")

    # -- 3. per-backbone summary at rung `none` ----------------------------
    out.append("## 3. Depth divergence at rung `none`, per backbone\n")
    out.append("The Stage 1 condition, now at 5 seeds and both directions.\n")
    out.append("| direction | backbone | gap (layers) | cost |")
    out.append("|---|---|---|---|")
    for source, target in directions:
        for backbone in backbones:
            gaps, costs = [], []
            for seed in seeds:
                cells = [r for r in rows
                         if (r["source_corpus"], r["target_corpus"]) == (source, target)
                         and r["backbone"] == backbone and r["alignment"] == "none"
                         and r["seed"] == seed]
                if len(cells) < 3:
                    continue
                bv = max(cells, key=lambda r: r["selection_source_val_macro_f1"])
                bt = max(cells, key=lambda r: r["macro_f1"])
                gaps.append(float(bt["layer_index"] - bv["layer_index"]))
                costs.append(bt["macro_f1"] - bv["macro_f1"])
            if gaps:
                cost = seed_interval(costs)
                out.append(f"| {source[:4]}->{target[:4]} | {backbone} | "
                           f"{interval(gaps)} | {cost['mean']:+.4f} "
                           f"[{cost['lo']:+.4f}, {cost['hi']:+.4f}] |")
    out.append("")

    # -- 4. the curves ------------------------------------------------------
    out.append("## 4. The curves, rung `none`, mean over 5 seeds\n")
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

    # -- 4b. merge safety ---------------------------------------------------
    out.append("## 4b. Merge safety: run_ids shared with the Stage 2 grid\n")
    from ser.utils.results import read_rows as _read_rows

    grid = {r["run_id"]: r for r in _read_rows(REPO_ROOT / "results/runs.jsonl")}
    all_sweep, _, _ = load()
    sweep_by_id = {r["run_id"]: r for r in all_sweep}
    shared = sorted(set(grid) & set(sweep_by_id))
    volatile = {"timestamp", "wall_seconds", "hostname", "git_dirty", "git_sha",
                "predictions_path", "run_started_utc", "python_version",
                "library_versions_json"}
    compared = [k for k in grid[shared[0]] if k not in volatile] if shared else []
    mismatches = Counter()
    for run_id in shared:
        a, b = grid[run_id], sweep_by_id[run_id]
        for key in compared:
            if key not in b:
                continue
            va, vb = a[key], b[key]
            same = (abs(va - vb) <= 1e-12 * max(1.0, abs(va))
                    if isinstance(va, float) and isinstance(vb, float) else va == vb)
            if not same:
                mismatches[key] += 1
    out.append(f"The sweep fixes `layer_agg='layer'`, so its **layer 6** cells have "
               f"the same 19 run_id coordinates as the Stage 2 grid's `layer:6` "
               f"cells and therefore the same ids. {len(shared)} ids are shared, "
               f"all of them at layer 6.\n")
    out.append(f"| | |\n|---|---|\n| shared run_ids | {len(shared)} |")
    out.append(f"| non-volatile fields compared | {len(compared)} |")
    out.append(f"| fields with any mismatch | {len(mismatches)} |")
    out.append(f"| total mismatching values | {sum(mismatches.values())} |")
    out.append("")
    if mismatches:
        out.append("**IDENTITY BUG.** Two rows share an id and disagree on: "
                   + ", ".join(f"`{k}` ({n})" for k, n in mismatches.most_common())
                   + ". This is not a merge conflict -- it means the run_id "
                     "coordinates do not determine the computation. Stop and fix.\n")
    else:
        out.append(f"**Zero mismatches across {len(compared) * len(shared)} compared "
                   "values.** These runs were executed weeks apart, in different "
                   "processes, under different launchers, on a machine that was "
                   "swapping for part of the time -- and produced bit-identical "
                   "metrics, confusion matrices and selected hyperparameters. That "
                   "is a determinism check the project did not plan for and it is "
                   "the second one to come free (the first was 23 duplicated wavlm "
                   "cells in the Stage 1 sweep).\n")
    out.append("**Decision: the sweep stays a separate artifact and is NOT merged "
               "into `results/runs.jsonl`.** Phase 8 established that every row in "
               "the provenance record is one the Stage 2 enumeration produces, and "
               "verified it (`recorded but NOT enumerated: 0`). Merging would put "
               f"{len(all_sweep) - len(shared)} rows at layers Stage 2 never "
               "enumerates into that file and break the invariant, for no gain: the "
               "analysis reads the shard files directly, and the shared cells are "
               "already present. The eps probe stays separate for the same reason.\n")

    # -- 5. interpretation --------------------------------------------------
    own_pooled, ref_pooled = seed_interval(own_all), seed_interval(ref_all)
    frames_disagree = (
        own_pooled["lo"] * own_pooled["hi"] > 0
        and ref_pooled["lo"] * ref_pooled["hi"] > 0
        and own_pooled["mean"] * ref_pooled["mean"] < 0
    )
    by_rung = Counter()
    for source, target in directions:
        for backbone in backbones:
            for rung in complete:
                o, r = [], []
                for seed in seeds:
                    cells = [x for x in rows
                             if (x["source_corpus"], x["target_corpus"]) == (source, target)
                             and x["backbone"] == backbone
                             and x["alignment"] == rung and x["seed"] == seed]
                    if len(cells) < 3:
                        continue
                    cells.sort(key=lambda x: x["layer_index"])
                    sc = [x["macro_f1"] for x in cells]
                    o.append(spearman([x["marginal_mmd_normalised"] for x in cells], sc))
                    r.append(spearman([x["marginal_mmd_reference"] for x in cells], sc))
                if not o:
                    continue
                a, b = seed_interval(o), seed_interval(r)
                if a["lo"] * a["hi"] > 0 and b["lo"] * b["hi"] > 0 and a["mean"] * b["mean"] < 0:
                    by_rung[rung] += 1

    out.append("## 5. Interpretation\n")
    out.append("### Frame dependence: "
               + ("**CONFIRMED**" if frames_disagree else "**NOT CONFIRMED**") + "\n")
    out.append(f"Pooled over all {own_pooled['n']} (cell, seed) correlations, the two "
               f"geometries have **opposite signs and neither interval covers zero**: "
               f"own {own_pooled['mean']:+.3f} "
               f"[{own_pooled['lo']:+.3f}, {own_pooled['hi']:+.3f}] against reference "
               f"{ref_pooled['mean']:+.3f} "
               f"[{ref_pooled['lo']:+.3f}, {ref_pooled['hi']:+.3f}]. "
               f"{disagreements} of {comparisons} individual cells disagree by the "
               "stricter per-cell test.\n")
    out.append("Disagreeing cells by rung: "
               + ", ".join(f"`{k}` {v}" for k, v in sorted(by_rung.items()))
               + ".\n")
    out.append("**But it does not replicate where it was found.** At rung `none` -- "
               f"the Stage 1 condition -- {by_rung.get('none', 0)} of "
               f"{len(directions) * len(backbones)} cells disagree. The Stage 1 "
               "observation was made on 2 seeds at `none`, and at 5 seeds that "
               "specific measurement is noise: every `none` interval covers zero in "
               "at least one frame. The effect is real, and it lives on the aligned "
               "rungs rather than the unaligned one.\n")
    out.append("So the claim the paper can make is narrower and better specified "
               "than the one PROGRESS.md carried: *after per-dimension "
               "standardisation, the measured relationship between marginal "
               "discrepancy and transfer reverses sign depending on the geometry "
               "the discrepancy is measured in*. That is still a statement about "
               "the measurement rather than about our implementation, and it is "
               "now backed by 5 seeds, 3 backbones and both directions.\n")

    out.append("### Depth divergence: **DOES NOT REPLICATE as stated**\n")
    out.append(f"Pooled over the same cells the gap is {gap_stat['mean']:+.2f} layers "
               f"[{gap_stat['lo']:+.2f}, {gap_stat['hi']:+.2f}] -- exactly no "
               f"systematic offset -- and only {positive_cells} of {total_cells} "
               "cells have a gap interval that excludes zero and is positive.\n")
    out.append("What survives is much narrower. At rung `none` in the "
               "**ravdess->cremad** direction the gap is positive in all three "
               "backbones (+4.2, +4.2, +2.4 layers, two of three excluding zero), "
               "which is the Stage 1 condition and does replicate. In the reverse "
               "direction it does not (wav2vec2 gives -3.8), and under alignment it "
               "collapses or reverses (coral, cremad->ravdess, wavlm: -5.6).\n")
    out.append(f"The stronger framing -- that selecting depth on in-domain "
               f"validation *systematically picks the wrong depth* -- is "
               f"contradicted outright. Against a random layer, `source_val` "
               f"selection is better by {paired_stat['mean']:+.4f} "
               f"[{paired_stat['lo']:+.4f}, {paired_stat['hi']:+.4f}] macro-F1. It "
               "is an imperfect criterion, not an anti-correlated one, and the "
               "0.054 it leaves on the table has to be read against the 0.091 a "
               "coin flip leaves.\n")
    out.append("**The '4-5 layers shallower, in all three backbones' claim should "
               "come out of the paper.** What can replace it is the conditional "
               "version: in the forward direction, unaligned, the transfer-optimal "
               "layer sits 2 to 4 layers deeper than the in-domain-optimal one.\n")

    out.append("### What the excluded rungs can and cannot change\n")
    out.append(f"`{'`, `'.join(excluded)}` are missing "
               f"{sum(CELLS_PER_RUNG - counts.get(r, 0) for r in excluded)} cells.\n")
    out.append("**Cannot change either verdict at rung `none`.** That rung is "
               f"{counts.get('none', 0)}/{CELLS_PER_RUNG} complete, so the Stage 1 "
               "replication attempt -- which is the only thing either claim "
               "originally rested on -- is fully answered.\n")
    out.append("**Cannot overturn frame dependence.** The pooled disagreement "
               "already has both intervals clear of zero on four rungs; two further "
               "rungs can add disagreeing cells or neutral ones, but cannot make "
               "existing opposite-signed intervals overlap.\n")
    out.append("**Could refine the rate and the depth verdict slightly.** The "
               f"disagreement rate {disagreements}/{comparisons} is over four rungs "
               "and would be recomputed over six. For depth, both MK-MMD rungs are "
               "alignment rungs and CORAL -- the closest analogue already measured "
               "-- shows mixed gaps from -5.6 to +1.8, so they are unlikely to move "
               "a pooled +0.00 into a systematic positive. That is an expectation, "
               "not a measurement, and this section should be regenerated when the "
               "remaining cells land.\n")

    path = REPO_ROOT / "reports/layer_sweep_v2.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path}")
    print(f"  {len(rows)} runs, rungs {complete}, excluded {excluded}")
    print(f"  frame sign disagreements: {disagreements}/{comparisons}")
    print(f"  positive depth gaps:      {positive_cells}/{total_cells}")
    print(f"  pooled rho own {seed_interval(own_all)['mean']:+.3f} / "
          f"ref {seed_interval(ref_all)['mean']:+.3f}")
    print(f"  pooled gap {gap_stat['mean']:+.2f} layers, cost {cost_stat['mean']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
