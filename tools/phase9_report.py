#!/usr/bin/env python
"""Phase 9 pass 1: the shift decomposition as numbers. No interpretation.

    python tools/phase9_report.py

Writes reports/phase9_tables.md from results/phase9_shift.jsonl.

Every cell is a mean with a 95% t-interval over the 5 seeds. Seeds are the
replication unit: each one is a different speaker-disjoint partition, so the
priors, the fitted alignment and the conditional supports all move with it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.analysis import assert_conditional_shift_firewall  # noqa: E402
from ser.config import load_config  # noqa: E402
from ser.phase8 import seed_interval  # noqa: E402

LADDER = ["none", "zscore", "mean_shift", "coral", "mkmmd_diag", "mkmmd_full"]


def fmt(values, places=4):
    interval = seed_interval(list(values))
    if interval["n"] == 0:
        return "--"
    if interval["n"] == 1:
        return f"{interval['mean']:.{places}f} (n=1)"
    return (f"{interval['mean']:.{places}f} "
            f"[{interval['lo']:.{places}f}, {interval['hi']:.{places}f}]")


def main() -> int:
    assert_conditional_shift_firewall()
    config = load_config()
    path = REPO_ROOT / "results/phase9_shift.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        print("no phase 9 records")
        return 1

    directions = sorted({(r["source"], r["target"]) for r in records})
    aggs = sorted({r["layer_agg"] for r in records})
    classes = records[0]["classes"]

    out = ["# Phase 9 pass 1 — the three-way shift decomposition", ""]
    out.append("Numbers only. Interpretation is in "
               "`reports/phase9_interpretation.md`.\n")
    out.append(f"{len(records)} records: {len(directions)} directions x "
               f"{len({r['seed'] for r in records})} seeds x {len(aggs)} aggregations "
               f"x {len(LADDER)} rungs, hubert, logreg. Intervals are t-intervals "
               "over seeds.\n")
    out.append("> **A10.** The conditional term reads target test labels. It is "
               "computed in `ser.analysis`, written only to "
               "`results/phase9_shift.jsonl`, and never to `results/runs.jsonl` "
               "or to any field of the frozen result schema. The firewall "
               "assertion runs before this report is generated and before the "
               "experiment that produced it.\n")
    out.append("---\n")

    # -- 1. label shift ----------------------------------------------------
    out.append("## 1. Label shift\n")
    out.append("KL(P_target || P_source) between the **realised partition** priors "
               "— not corpus-level counts. Splits are speaker-disjoint and speakers "
               "are not class-balanced, so the prior a seed actually trains on is "
               "not the corpus prior.\n")
    out.append("| direction | KL (nats) | KL (bits) | total variation | classes with no source support |")
    out.append("|---|---|---|---|---|")
    for source, target in directions:
        g = [r for r in records if (r["source"], r["target"]) == (source, target)]
        by_seed = {}
        for r in g:
            by_seed[r["seed"]] = r["label_shift"]
        kl = [v["kl_nats"] for v in by_seed.values()]
        bits = [v["kl_bits"] for v in by_seed.values()]
        tv = [v["total_variation"] for v in by_seed.values()]
        missing = sorted({c for v in by_seed.values()
                          for c in v["classes_without_source_support"]})
        out.append(f"| {source}->{target} | {fmt(kl, 5)} | {fmt(bits, 5)} | "
                   f"{fmt(tv, 4)} | {', '.join(missing) if missing else 'none'} |")
    out.append("")
    out.append("Realised priors, mean over seeds:\n")
    out.append("| direction | side | " + " | ".join(classes) + " |")
    out.append("|---" * (len(classes) + 2) + "|")
    for source, target in directions:
        g = {r["seed"]: r["label_shift"] for r in records
             if (r["source"], r["target"]) == (source, target)}
        for side in ("source_prior", "target_prior"):
            values = np.array([v[side] for v in g.values()])
            out.append(f"| {source}->{target} | {side.replace('_prior','')} | "
                       + " | ".join(f"{v:.3f}" for v in values.mean(axis=0)) + " |")
    out.append("")

    # -- 2. covariate vs conditional --------------------------------------
    out.append("## 2. Covariate shift against conditional shift\n")
    out.append("Both measured between the **same two sets** (aligned source_train "
               "and aligned target_test), so the two columns are directly "
               "comparable. The conditional column is the mean over classes whose "
               f"support clears {config.shift.conditional_mmd_min_support}.\n")
    for agg in aggs:
        out.append(f"### layer_agg = {agg}\n")
        out.append("| direction | rung | marginal, own frame | marginal, reference frame | "
                   "conditional (mean over classes) | classes defined |")
        out.append("|---|---|---|---|---|---|")
        for source, target in directions:
            for rung in LADDER:
                g = [r for r in records
                     if (r["source"], r["target"]) == (source, target)
                     and r["layer_agg"] == agg and r["alignment"] == rung]
                if not g:
                    continue
                own = [r["marginal_effect_own"] for r in g]
                ref = [r["marginal_effect_reference"] for r in g]
                per_seed_conditional = []
                defined = set()
                for r in g:
                    values = [c["effect_size"] for c in r["conditional"]
                              if c["effect_size"] is not None]
                    defined |= {c["class"] for c in r["conditional"]
                                if c["effect_size"] is not None}
                    if values:
                        per_seed_conditional.append(float(np.mean(values)))
                out.append(
                    f"| {source[:4]}->{target[:4]} | {rung} | {fmt(own, 2)} | "
                    f"{fmt(ref, 2)} | {fmt(per_seed_conditional, 2)} | "
                    f"{len(defined)}/{len(classes)} |"
                )
        out.append("")

    # -- 3. conditional per class -----------------------------------------
    out.append("## 3. Conditional shift per class\n")
    out.append("Per-class n accompanies every value, as A10 requires. A class below "
               f"the minimum support of {config.shift.conditional_mmd_min_support} on "
               "either side is reported as undefined, never as a number.\n")
    for source, target in directions:
        out.append(f"### {source} -> {target}, layer_agg = {aggs[0]}\n")
        out.append("| class | n source | n target | " +
                   " | ".join(f"{rung}" for rung in LADDER) + " |")
        out.append("|---" * (len(LADDER) + 3) + "|")
        for name in classes:
            cells, n_source, n_target = [], [], []
            for rung in LADDER:
                g = [r for r in records
                     if (r["source"], r["target"]) == (source, target)
                     and r["layer_agg"] == aggs[0] and r["alignment"] == rung]
                values = []
                for r in g:
                    entry = next(c for c in r["conditional"] if c["class"] == name)
                    n_source.append(entry["n_source"])
                    n_target.append(entry["n_target"])
                    if entry["effect_size"] is not None:
                        values.append(entry["effect_size"])
                cells.append(fmt(values, 1) if values else "**undefined**")
            out.append(f"| {name} | {int(np.mean(n_source))} | {int(np.mean(n_target))} | "
                       + " | ".join(cells) + " |")
        out.append("")

    # -- 4. the falsifiable test -------------------------------------------
    out.append("## 4. Label-shift correction — the falsifiable test\n")
    out.append("Near-zero prior KL **predicts** that a label-shift correction "
               "cannot help. Both estimators are validated against a planted shift "
               "in `tests/test_analysis_shift.py`, so a null here is a result "
               "rather than a broken implementation.\n")
    out.append("| direction | agg | rung | uncorrected | BBSE | delta | EM | delta |")
    out.append("|---|---|---|---|---|---|---|---|")
    for source, target in directions:
        for agg in aggs:
            for rung in LADDER:
                g = [r for r in records
                     if (r["source"], r["target"]) == (source, target)
                     and r["layer_agg"] == agg and r["alignment"] == rung]
                if not g:
                    continue
                base = [r["correction"]["uncorrected"] for r in g]
                bbse = [r["correction"]["bbse"] for r in g if "bbse" in r["correction"]]
                em = [r["correction"]["em"] for r in g]
                d_bbse = [r["correction"]["bbse"] - r["correction"]["uncorrected"]
                          for r in g if "bbse" in r["correction"]]
                d_em = [r["correction"]["em"] - r["correction"]["uncorrected"] for r in g]
                out.append(
                    f"| {source[:4]}->{target[:4]} | {agg} | {rung} | {fmt(base)} | "
                    f"{fmt(bbse) if bbse else 'unidentifiable'} | "
                    f"{fmt(d_bbse) if d_bbse else '--'} | {fmt(em)} | {fmt(d_em)} |"
                )
    out.append("")

    path = REPO_ROOT / "reports/phase9_tables.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
