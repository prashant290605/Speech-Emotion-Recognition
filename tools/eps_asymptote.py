#!/usr/bin/env python
"""Does CORAL degenerate as its shrinkage grows? Two tests.

    python tools/eps_asymptote.py --analytic
    python tools/eps_asymptote.py --runs --shard 0 --n-shards 4

Stage 1 and Stage 2 both found `source_val` monotone increasing in eps with the
argmax sitting on the grid boundary, and Phase 8 recorded that as an unresolved
defect ("the grid is still mis-centred"). It is not a defect. It is a property
of the estimator, and it is predictable in closed form.

CORAL's map is ``M = C_s^{-1/2} C_t^{1/2}`` with each covariance regularised as
``C + eps * tr(C)/d * I``. As eps grows both regularised covariances approach a
scaled identity, so

    M  ->  sqrt( tr(C_t) / tr(C_s) ) * I

and the rung collapses to *a global scalar rescale plus a mean shift* -- which
is `mean_shift` with one extra degree of freedom, and close to `zscore` when the
per-dimension scales are similar. Covariance matching, the thing CORAL exists to
do, is switched off continuously as eps rises.

``--analytic`` measures that convergence on real features: the relative distance
from M to the nearest scaled identity, and the ratio of the fitted scalar to its
predicted limit. No classifier, seconds to run.

``--runs`` extends the empirical grid to eps=100 and 1000 so the `source_val`
asymptote and the convergence of target macro-F1 and discrepancy toward
mean_shift can be read off directly.

Rows go to their own results file, not results/runs.jsonl: these are off-grid
probe points and mixing them in would break the invariant that every recorded
row is one the Stage 2 enumeration produces. No config change and no re-freeze
is involved -- ``build_alignment`` takes eps as an argument and never consults
the searched grid, so these runs carry the same facet hashes as Stage 2 and are
directly comparable to it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

from ser.features.audio import warm_up_audio_stack  # noqa: E402

warm_up_audio_stack()

import numpy as np  # noqa: E402

from ser.alignment import build_alignment  # noqa: E402
from ser.config import load_config  # noqa: E402
from ser.freeze import read_freeze_tag  # noqa: E402
from ser.numerics import covariance  # noqa: E402
from ser.run_grid import GridRun, _Context, execute_run, shard_of  # noqa: E402
from ser.utils.results import append_row, completed_run_ids, make_run_id  # noqa: E402

EPS_LADDER = [1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
NEW_EPS = [100.0, 1000.0]
DIRECTIONS = [("ravdess", "cremad"), ("cremad", "ravdess")]


def analytic(config, context, *, backbone="hubert", layer_spec="last") -> int:
    """How close is the fitted CORAL map to a scaled identity, against eps?"""
    print(f"CORAL map convergence, {backbone}, {layer_spec}\n")
    print(f"{'pair':<20} {'eps':>8} {'||M-cI||/||M||':>15} {'c fitted':>10} "
          f"{'c predicted':>12} {'ratio':>7}")
    for source, target in DIRECTIONS:
        pair = context.split(source, target, 0)
        X_s = context.loader(source, backbone).load(
            pair.source_train.utterance_ids, layer_spec=layer_spec
        )
        X_t = context.loader(target, backbone).load(
            pair.target_adapt.utterance_ids, layer_spec=layer_spec
        )
        # The limit the derivation predicts, computed from the raw covariances.
        limit = float(
            np.sqrt(np.trace(covariance(X_t)) / np.trace(covariance(X_s)))
        )
        for eps in EPS_LADDER:
            alignment = build_alignment("coral", config, eps=eps)
            alignment.fit(X_s, X_t, pair.target_adapt.utterance_ids,
                          pair.source_train.utterance_ids)
            M = alignment.transform_matrix
            d = M.shape[0]
            scalar = float(np.trace(M) / d)
            residual = float(
                np.linalg.norm(M - scalar * np.eye(d)) / np.linalg.norm(M)
            )
            print(f"{source+'->'+target:<20} {eps:>8g} {residual:>15.4f} "
                  f"{scalar:>10.4f} {limit:>12.4f} {scalar/limit:>7.3f}")
        print()
    return 0


def enumerate_runs(config, *, backbone, seeds, families, aggs):
    runs = []
    for source, target in DIRECTIONS:
        for seed in seeds:
            for family in families:
                for agg in aggs:
                    for eps in NEW_EPS:
                        runs.append(
                            GridRun(
                                source=source, target=target, seed=seed,
                                backbone=backbone, feature_branch="ssl",
                                layer_agg=agg,
                                layer_index=(config.classifiers.layer_candidates[1]
                                             if agg == "layer" else None),
                                alignment="coral", alignment_eps=eps,
                                alignment_lam=None, blending="none",
                                blend_alpha=None, n_groups=None, classifier=family,
                            )
                        )
    return runs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic", action="store_true")
    parser.add_argument("--runs", action="store_true")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--backbone", default="hubert")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--families", default="logreg,mlp")
    parser.add_argument("--aggs", default="last,layer")
    parser.add_argument("--results", default="results/eps_asymptote.jsonl")
    args = parser.parse_args(argv)

    config = load_config()
    context = _Context(config)

    if args.analytic:
        analytic(config, context, backbone=args.backbone)
    if not args.runs:
        return 0

    out = REPO_ROOT / args.results
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = enumerate_runs(
        config,
        backbone=args.backbone,
        seeds=[int(s) for s in args.seeds.split(",")],
        families=args.families.split(","),
        aggs=args.aggs.split(","),
    )
    ids = [make_run_id(r.coords(config)) for r in runs]
    assert len(set(ids)) == len(ids), "duplicate run_ids in the eps probe"
    if args.shard is not None:
        runs = [r for r, i in zip(runs, ids) if shard_of(i, args.n_shards) == args.shard]

    already = completed_run_ids(out)
    todo = [r for r in runs if make_run_id(r.coords(config)) not in already]
    print(f"eps probe: {len(runs)} runs in this shard, {len(todo)} to execute")

    freeze_tag = read_freeze_tag() or ""
    started = time.perf_counter()
    failed = 0
    for index, run in enumerate(todo, start=1):
        row = execute_run(run, context, freeze_tag)
        append_row(out, row)
        failed += row["status"] != "ok"
        if index % 10 == 0 or index == len(todo):
            elapsed = time.perf_counter() - started
            print(f"  {index}/{len(todo)} elapsed {elapsed/60:.1f} min | {failed} failed",
                  flush=True)
    print(f"done: {len(todo)} runs, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
