#!/usr/bin/env python
"""Measure the cost of the reverse direction before projecting Stage 2.

    python tools/calibrate_stage2.py --out <path>

Stage 1 ran ravdess -> cremad, where source_train is 988 utterances. Stage 2
adds cremad -> ravdess, where source_train is 5972 -- 6.0x. Several costs in a
run are superlinear in that number:

    SVM fit               O(n^2) to O(n^3)
    null_mmd_scale        O(n^2)   (half-splits of the aligned source)
    CORAL covariance      O(n d^2)

so projecting Stage 2 by assuming the two directions cost the same would be
wrong in the expensive direction. This runs a handful of real cells and reports
the measured ratio per family.

Writes full schema-valid rows to ``--out``. They are NOT written to
results/runs.jsonl: Stage 2 has not been launched, and a partial direction
sitting in the provenance record would misrepresent what has been run. The
run_ids are the real Stage 2 ids, so the rows can be merged in later if wanted.
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
    os.environ.setdefault(_var, "4")

from ser.features.audio import warm_up_audio_stack  # noqa: E402

warm_up_audio_stack()

from ser.config import load_config  # noqa: E402
from ser.freeze import read_freeze_tag  # noqa: E402
from ser.run_grid import GridRun, _Context, execute_run  # noqa: E402
from ser.utils.results import append_row  # noqa: E402

# One cheap rung and one expensive rung per family, at `last`. Enough to get a
# per-family ratio without paying for the whole cross-product.
RUNGS = [("none", None, None), ("mkmmd_full", None, 0.01)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/scratch/stage2_calibration.jsonl")
    parser.add_argument("--families", default="logreg,svm_linear,svm_rbf,mlp")
    parser.add_argument("--source", default="cremad")
    parser.add_argument("--target", default="ravdess")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    config = load_config()
    context = _Context(config)
    freeze_tag = read_freeze_tag() or ""
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"calibrating {args.source} -> {args.target}, seed {args.seed}\n")
    for family in args.families.split(","):
        for method, eps, lam in RUNGS:
            run = GridRun(
                source=args.source,
                target=args.target,
                seed=args.seed,
                backbone="hubert",
                feature_branch="ssl",
                layer_agg="last",
                layer_index=None,
                alignment=method,
                alignment_eps=eps,
                alignment_lam=lam,
                blending="none",
                blend_alpha=None,
                n_groups=None,
                classifier=family,
            )
            started = time.perf_counter()
            row = execute_run(run, context, freeze_tag)
            append_row(out, row)
            print(
                f"  {family:<12} {method:<12} {row['status']:<7} "
                f"{time.perf_counter() - started:>8.0f}s",
                flush=True,
            )
            if row["status"] != "ok":
                print(f"      {str(row.get('error_type'))}: {str(row.get('error_message'))[:200]}")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
