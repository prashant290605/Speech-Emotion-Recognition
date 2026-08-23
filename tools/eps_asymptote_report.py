#!/usr/bin/env python
"""Does CORAL asymptote, and does it converge on mean_shift? Numbers only.

    python tools/eps_asymptote_report.py

Writes reports/eps_asymptote.md, combining the Stage 2 grid (eps up to 10) with
the off-grid probe at eps=100 and 1000, restricted to the conditions the probe
covers so the two are matched.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.phase8 import seed_interval  # noqa: E402
from ser.utils.results import read_rows  # noqa: E402


def fmt(values, places=4):
    interval = seed_interval(list(values))
    if interval["n"] == 0:
        return "--"
    if interval["n"] == 1:
        return f"{interval['mean']:.{places}f} (n=1)"
    return (f"{interval['mean']:.{places}f} "
            f"[{interval['lo']:.{places}f}, {interval['hi']:.{places}f}]")


def per_seed(rows, key):
    """Mean of `key` within each seed, so seeds stay the replication unit."""
    buckets = defaultdict(list)
    for r in rows:
        if r[key] is not None:
            buckets[r["seed"]].append(r[key])
    return [float(np.mean(v)) for v in buckets.values()]


def main() -> int:
    grid = [r for r in read_rows(REPO_ROOT / "results/runs.jsonl")
            if r["freeze_tag"] == "grid-freeze-v3" and r["blending"] == "none"]
    probe_path = REPO_ROOT / "results/eps_asymptote.jsonl"
    probe = [r for r in read_rows(probe_path)] if probe_path.exists() else []
    probe = [r for r in probe if r["status"] == "ok"]
    if not probe:
        print("no probe rows yet")
        return 1

    # Match the grid to exactly the probe's conditions.
    families = {r["classifier"] for r in probe}
    aggs = {r["layer_agg"] for r in probe}
    backbones = {r["backbone"] for r in probe}
    seeds = {r["seed"] for r in probe}
    matched = [r for r in grid
               if r["classifier"] in families and r["layer_agg"] in aggs
               and r["backbone"] in backbones and r["seed"] in seeds]

    directions = sorted({(r["source_corpus"], r["target_corpus"]) for r in probe})
    out = ["# CORAL's shrinkage asymptote", ""]
    out.append(f"Stage 2 grid (eps <= 10) restricted to the probe's conditions — "
               f"{', '.join(sorted(backbones))}, {', '.join(sorted(families))}, "
               f"{', '.join(sorted(aggs))} — plus {len(probe)} off-grid runs at "
               "eps=100 and 1000. Intervals are t-intervals over seeds.\n")
    out.append("`mean_shift` and `zscore` rows are the convergence targets: the "
               "derivation says CORAL tends to a global scalar rescale plus a mean "
               "shift as eps grows, which is `mean_shift` with one extra degree of "
               "freedom.\n")
    out.append("---\n")

    out.append("## Analytic convergence of the map\n")
    out.append("Distance from the fitted CORAL matrix to the nearest scaled "
               "identity, `||M - cI|| / ||M||`, and the fitted scalar against its "
               "predicted limit `sqrt(tr(C_t)/tr(C_s))`. hubert, `last`, seed 0. "
               "Reproduce with `python tools/eps_asymptote.py --analytic`.\n")
    out.append("| pair | eps | \\|\\|M-cI\\|\\|/\\|\\|M\\|\\| | c fitted | c predicted | ratio |")
    out.append("|---|---|---|---|---|---|")
    out.append("| ravdess->cremad | 0.0001 | 0.9254 | 2.4867 | 1.1327 | 2.195 |")
    out.append("| ravdess->cremad | 0.01 | 0.8401 | 1.6668 | 1.1327 | 1.472 |")
    out.append("| ravdess->cremad | 0.1 | 0.6904 | 1.3401 | 1.1327 | 1.183 |")
    out.append("| ravdess->cremad | 1 | 0.4009 | 1.1871 | 1.1327 | 1.048 |")
    out.append("| ravdess->cremad | 10 | 0.1436 | 1.1414 | 1.1327 | 1.008 |")
    out.append("| ravdess->cremad | 100 | 0.0343 | 1.1333 | 1.1327 | 1.001 |")
    out.append("| ravdess->cremad | 1000 | 0.0049 | 1.1327 | 1.1327 | 1.000 |")
    out.append("| cremad->ravdess | 0.0001 | 0.9630 | 1.3124 | 0.8764 | 1.497 |")
    out.append("| cremad->ravdess | 0.01 | 0.8899 | 1.1018 | 0.8764 | 1.257 |")
    out.append("| cremad->ravdess | 0.1 | 0.7502 | 0.9856 | 0.8764 | 1.125 |")
    out.append("| cremad->ravdess | 1 | 0.4463 | 0.9125 | 0.8764 | 1.041 |")
    out.append("| cremad->ravdess | 10 | 0.1571 | 0.8830 | 0.8764 | 1.008 |")
    out.append("| cremad->ravdess | 100 | 0.0344 | 0.8769 | 0.8764 | 1.001 |")
    out.append("| cremad->ravdess | 1000 | 0.0049 | 0.8764 | 0.8764 | 1.000 |")
    out.append("")

    out.append("## Empirical behaviour against eps\n")
    for source, target in directions:
        out.append(f"### {source} -> {target}\n")
        out.append("| rung | eps | runs | source_val | target macro-F1 | "
                   "effect, own frame | effect, reference frame |")
        out.append("|---|---|---|---|---|---|---|")
        pool = [r for r in matched
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        probe_pool = [r for r in probe
                      if (r["source_corpus"], r["target_corpus"]) == (source, target)]

        rows_for_eps = []
        for eps in sorted({r["alignment_eps"] for r in pool if r["alignment"] == "coral"
                           and r["alignment_eps"] is not None}):
            rows_for_eps.append((f"{eps:g}", [r for r in pool
                                              if r["alignment"] == "coral"
                                              and r["alignment_eps"] == eps]))
        for eps in sorted({r["alignment_eps"] for r in probe_pool}):
            rows_for_eps.append((f"{eps:g}", [r for r in probe_pool
                                              if r["alignment_eps"] == eps]))
        lw = [r for r in pool if r["alignment"] == "coral" and r["alignment_eps"] is None]
        if lw:
            rows_for_eps.append(("ledoit-wolf", lw))

        for label, g in rows_for_eps:
            if not g:
                continue
            out.append(
                f"| coral | {label} | {len(g)} | "
                f"{fmt(per_seed(g, 'selection_source_val_macro_f1'))} | "
                f"{fmt(per_seed(g, 'macro_f1'))} | "
                f"{fmt(per_seed(g, 'marginal_mmd_normalised'), 2)} | "
                f"{fmt(per_seed(g, 'marginal_mmd_reference'), 2)} |"
            )
        for rung in ("mean_shift", "zscore", "none"):
            g = [r for r in pool if r["alignment"] == rung]
            if not g:
                continue
            out.append(
                f"| **{rung}** | -- | {len(g)} | "
                f"{fmt(per_seed(g, 'selection_source_val_macro_f1'))} | "
                f"{fmt(per_seed(g, 'macro_f1'))} | "
                f"{fmt(per_seed(g, 'marginal_mmd_normalised'), 2)} | "
                f"{fmt(per_seed(g, 'marginal_mmd_reference'), 2)} |"
            )
        out.append("")

    path = REPO_ROOT / "reports/eps_asymptote.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path} ({len(probe)} probe runs, {len(matched)} matched grid runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
