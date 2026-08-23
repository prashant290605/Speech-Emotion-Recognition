#!/usr/bin/env python
"""Re-run the 13-layer sweep at full seed count, across ladder rungs.

    python tools/layer_sweep_v2.py --shard 0 --n-shards 4

The Stage 1 sweep was 2 seeds, one classifier, one pair, and rung `none`. Two
claims currently rest on it and on nothing else:

* **frame dependence** -- that rho(discrepancy, target) has opposite signs in
  the rung's own geometry and in the fixed reference frame, on 2 of 3 backbones;
* **depth divergence** -- that the layer maximising `source_val` sits 4-5 layers
  shallower than the layer maximising target macro-F1.

This settles both or refutes both. Full seed count, both directions, all three
backbones, and all six ladder rungs rather than one, so the question becomes
"does the sign disagreement survive alignment" and not merely "does it exist
unaligned".

CORAL is deliberately NOT run at its `source_val`-selected eps of 10: at that shrinkage the covariance
term is almost entirely suppressed and the rung degenerates toward a scalar
rescale (see tools/eps_asymptote.py), so using it here would test the
degenerate case rather than covariance matching.

Rows are schema-valid and go to their own results file, never to
results/runs.jsonl: they are a targeted probe outside the frozen grid, and
mixing them in would break the invariant that every row in the provenance
record is one the Stage 2 enumeration produces.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

from ser.features.audio import warm_up_audio_stack  # noqa: E402

warm_up_audio_stack()

from ser.config import load_config  # noqa: E402
from ser.freeze import read_freeze_tag  # noqa: E402
from ser.run_grid import GridRun, _Context, execute_run, shard_of  # noqa: E402
from ser.utils.results import append_row, completed_run_ids, make_run_id  # noqa: E402

# All six rungs. A three-rung version was scoped first; measured cost came in at
# 8 s per cell, so the full ladder is affordable and the question becomes "does
# the sign disagreement survive alignment" rather than "does it exist unaligned".
RUNGS = [
    ("none", None, None),
    ("zscore", None, None),
    ("mean_shift", None, None),
    ("coral", 1e-2, None),
    ("mkmmd_diag", None, 0.01),
    ("mkmmd_full", None, 0.01),
]
DIRECTIONS = [("ravdess", "cremad"), ("cremad", "ravdess")]


def enumerate_sweep(config, *, backbones, seeds, layers):
    runs = []
    for source, target in DIRECTIONS:
        for backbone in backbones:
            for seed in seeds:
                for layer in layers:
                    for method, eps, lam in RUNGS:
                        runs.append(
                            GridRun(
                                source=source,
                                target=target,
                                seed=seed,
                                backbone=backbone,
                                feature_branch="ssl",
                                layer_agg="layer",
                                layer_index=layer,
                                alignment=method,
                                alignment_eps=eps,
                                alignment_lam=lam,
                                blending="none",
                                blend_alpha=None,
                                n_groups=None,
                                classifier="logreg",
                            )
                        )
    return runs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--results", default="results/layer_sweep_v2.jsonl")
    parser.add_argument("--backbones", default="hubert,wav2vec2,wavlm")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--heartbeat-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-from", default="results/shards/sweep2_*.jsonl",
        help="Glob of sweep files to treat as already done. Resuming from ALL "
             "of them, not just this worker's own file, is what lets the shard "
             "count change between restarts without redoing work.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    context = _Context(config)
    freeze_tag = read_freeze_tag() or ""
    out = REPO_ROOT / args.results
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = enumerate_sweep(
        config,
        backbones=args.backbones.split(","),
        seeds=[int(s) for s in args.seeds.split(",")],
        layers=range(config.features.n_layers),
    )
    ids = [make_run_id(r.coords(config)) for r in runs]
    assert len(set(ids)) == len(ids), "sweep enumeration produced duplicate run_ids"

    if args.shard is not None:
        runs = [r for r, i in zip(runs, ids) if shard_of(i, args.n_shards) == args.shard]

    already = set()
    for path in sorted(glob.glob(str(REPO_ROOT / args.resume_from))):
        already |= completed_run_ids(path)
    todo = [r for r in runs if make_run_id(r.coords(config)) not in already]
    # Decisive rungs first. `none` is where the Stage 1 finding was made and
    # `zscore`/`coral` are what test whether it survives alignment; the other
    # three broaden the answer but cannot change it. Ordering them first means a
    # sweep stopped early still answers the question it was run to answer,
    # instead of leaving every rung three-quarters finished.
    priority = {name: i for i, (name, _, _) in enumerate(RUNGS)}
    order = {"none": 0, "zscore": 1, "coral": 2}
    todo.sort(key=lambda r: (order.get(r.alignment, 3 + priority[r.alignment]),
                             r.source, r.backbone, r.seed, r.layer_index))
    print(f"sweep v2: {len(runs)} runs in this shard, {len(todo)} to execute, "
          f"{len(runs) - len(todo)} already complete", flush=True)
    if args.dry_run:
        return 0

    started = time.perf_counter()
    failed = 0
    for index, run in enumerate(todo, start=1):
        row = execute_run(run, context, freeze_tag)
        append_row(out, row)
        failed += row["status"] != "ok"
        if index % args.heartbeat_every == 0 or index == len(todo):
            elapsed = time.perf_counter() - started
            eta = (len(todo) - index) * elapsed / index
            print(f"  {index}/{len(todo)} ({index/len(todo):5.1%}) "
                  f"elapsed {elapsed/60:6.1f} min eta {eta/60:6.1f} min | {failed} failed",
                  flush=True)
    print(f"done: {len(todo)} runs, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
