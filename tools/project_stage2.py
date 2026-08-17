#!/usr/bin/env python
"""Project Stage 2 wall time from measured Stage 1 timings.

    python tools/project_stage2.py
    python tools/project_stage2.py --reverse-factor 3.0 --shards 4

Costs are taken from what Stage 1 actually did, not from a model: the median
``wall_seconds`` per (family, layer_agg, is-MK-MMD) cell, measured under the
same 4-shard contention Stage 2 will run under. That is why the totals divide
cleanly by the shard count -- the per-cell numbers already include the slowdown
from sharing the machine.

Two corrections are applied on top:

``--reverse-factor``
    cremad -> ravdess has 5972 source_train utterances against ravdess ->
    cremad's 988. SVM fitting and ``null_mmd_scale`` are both O(n^2) in that
    number, so the reverse direction is not free. Measure it with
    tools/calibrate_stage2.py and pass the result rather than guessing.

``--transformer-seconds``
    Stage 1's transformer probe was 12 runs; its median is noisier than the
    sklearn families'. Overridable.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.config import load_config  # noqa: E402
from ser.run_grid import STAGE2_SURVIVING, enumerate_stage  # noqa: E402
from ser.utils.results import read_rows  # noqa: E402


def measured_costs(results: Path) -> dict:
    """Median wall_seconds per (family, layer_agg, is_mkmmd) from Stage 1."""
    buckets = defaultdict(list)
    for row in read_rows(results):
        if row["classifier"].startswith("baseline") or row["status"] != "ok":
            continue
        if row["freeze_tag"] != "grid-freeze-v2":
            continue
        key = (row["classifier"], row["layer_agg"], row["alignment"].startswith("mkmmd"))
        buckets[key].append(row["wall_seconds"])
    return {k: median(v) for k, v in buckets.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/runs.jsonl")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--reverse-factor", type=float, default=1.0)
    parser.add_argument("--transformer-seconds", type=float, default=None)
    parser.add_argument("--one-direction", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    costs = measured_costs(REPO_ROOT / args.results)

    directions = [("ravdess", "cremad")]
    if not args.one_direction:
        directions.append(("cremad", "ravdess"))

    surviving = dict(
        STAGE2_SURVIVING,
        directions=directions,
        backbones=list(config.features.backbones),
        seeds=config.splits.seeds,
        transformer_seeds=config.splits.seeds[:2],
    )
    runs = enumerate_stage(config, 2, corpora=["ravdess", "cremad"], surviving=surviving)

    per_family = defaultdict(float)
    per_direction = defaultdict(float)
    missing = set()
    total = 0.0
    for run in runs:
        key = (run.classifier, run.layer_agg, run.alignment.startswith("mkmmd"))
        seconds = costs.get(key)
        if seconds is None:
            missing.add(key)
            seconds = costs.get((run.classifier, "last", key[2]), 300.0)
        if run.classifier == "transformer" and args.transformer_seconds:
            seconds = args.transformer_seconds
        if run.source == "cremad":
            seconds *= args.reverse_factor
        per_family[run.classifier] += seconds
        per_direction[(run.source, run.target)] += seconds
        total += seconds

    if missing:
        print(f"note: no Stage 1 timing for {sorted(missing)}; fell back to `last`\n")

    print(f"Stage 2: {len(runs)} runs")
    print(f"  directions      : {', '.join(f'{s}->{t}' for s, t in directions)}")
    print(f"  backbones       : {', '.join(surviving['backbones'])}")
    print(f"  seeds           : {surviving['seeds']} (transformer {surviving['transformer_seeds']})")
    print(f"  reverse factor  : {args.reverse_factor:.2f}x")
    print(f"  shards          : {args.shards}\n")

    print(f"{'family':<14} {'runs':>6} {'CPU hours':>11} {'wall hours':>11}")
    for family in sorted(per_family, key=lambda f: -per_family[f]):
        n = sum(1 for r in runs if r.classifier == family)
        print(f"{family:<14} {n:>6} {per_family[family]/3600:>11.1f} "
              f"{per_family[family]/3600/args.shards:>11.1f}")
    print(f"{'-'*44}")
    print(f"{'TOTAL':<14} {len(runs):>6} {total/3600:>11.1f} {total/3600/args.shards:>11.1f}")

    print(f"\n{'direction':<22} {'runs':>6} {'CPU hours':>11} {'wall hours':>11}")
    for (s, t), seconds in per_direction.items():
        n = sum(1 for r in runs if (r.source, r.target) == (s, t))
        print(f"{s+' -> '+t:<22} {n:>6} {seconds/3600:>11.1f} "
              f"{seconds/3600/args.shards:>11.1f}")

    budget = 72.0
    wall = total / 3600 / args.shards
    print(f"\nbudget {budget:.0f} h wall: {'FITS' if wall <= budget else 'OVER BY '
          f'{wall - budget:.1f} h'} ({wall:.1f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
