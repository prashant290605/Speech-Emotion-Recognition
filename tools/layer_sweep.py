#!/usr/bin/env python
"""Sweep all 13 transformer layers from the existing feature cache.

    python tools/layer_sweep.py

Stage 1 only ever compared `last` against `layer:6`, which makes the layer
finding a two-point contrast. The features for every layer are already cached,
so the full curve costs one logreg fit per (backbone, layer, seed) and nothing
else -- no extraction, no alignment search.

The rung is fixed at `none` on purpose. This measures the *intrinsic*
discrepancy of each layer's representation; letting an alignment rung vary too
would confound "which layer transfers" with "which rung helps".

Everything is selected on source_val. target_test is read once, at the end, for
the reported score only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

from ser.features.audio import warm_up_audio_stack  # noqa: E402

warm_up_audio_stack()

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from ser.classifiers import fit_and_select  # noqa: E402
from ser.config import load_config  # noqa: E402
from ser.mmd import (  # noqa: E402
    marginal_mmd,
    median_bandwidth,
    null_mmd_scale,
    reference_geometry,
)
from ser.run_grid import REFERENCE_GEOMETRY_EPS, _Context, _mmd_view  # noqa: E402
from ser.utils.seeding import set_all_seeds  # noqa: E402


def _effect(source: np.ndarray, target: np.ndarray, config, seed: int) -> float | None:
    bandwidth = median_bandwidth(source, target, seed=seed)
    raw = marginal_mmd(source, target, config, bandwidth=bandwidth, seed=seed)
    null = null_mmd_scale(
        source, config, bandwidth=bandwidth, n_repeats=5, seed=seed
    )["scale"]
    return raw / null if null > 0 else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="ravdess")
    parser.add_argument("--target", default="cremad")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--backbones", default="hubert,wav2vec2,wavlm")
    parser.add_argument("--out", default="results/layer_sweep.jsonl")
    args = parser.parse_args(argv)

    config = load_config()
    context = _Context(config)
    seeds = [int(s) for s in args.seeds.split(",")]
    backbones = args.backbones.split(",")
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["backbone"], r["layer"], r["seed"]))

    n_layers = config.features.n_layers
    total = len(backbones) * n_layers * len(seeds)
    started = time.perf_counter()
    i = 0

    for backbone in backbones:
        for layer in range(n_layers):
            for seed in seeds:
                i += 1
                if (backbone, layer, seed) in done:
                    continue
                set_all_seeds(seed)
                pair = context.split(args.source, args.target, seed)
                classes = list(config.labels.spaces[pair.label_space])
                spec = f"layer:{layer}"

                src = context.loader(args.source, backbone)
                tgt = context.loader(args.target, backbone)
                X_train = src.load(pair.source_train.utterance_ids, layer_spec=spec)
                X_val = src.load(pair.source_val.utterance_ids, layer_spec=spec)
                X_adapt = tgt.load(pair.target_adapt.utterance_ids, layer_spec=spec)
                X_test = tgt.load(pair.target_test.utterance_ids, layer_spec=spec)

                y_train = context.labels(pair, "source_train")
                y_val = context.labels(pair, "source_val")
                y_test = context.labels(pair, "target_test")

                mmd_src, mmd_tgt = _mmd_view(X_train), _mmd_view(X_adapt)
                own = _effect(mmd_src, mmd_tgt, config, seed)
                geometry = reference_geometry(mmd_src, eps=REFERENCE_GEOMETRY_EPS)
                ref = _effect(geometry(mmd_src), geometry(mmd_tgt), config, seed)

                selection = fit_and_select(
                    "logreg", X_train, y_train, X_val, y_val, classes, config, seed=seed
                )
                predicted = selection.predict(X_test)
                row = {
                    "backbone": backbone,
                    "layer": layer,
                    "seed": seed,
                    "source": args.source,
                    "target": args.target,
                    "effect_own": own,
                    "effect_reference": ref,
                    "source_val_macro_f1": selection.best_source_val_macro_f1,
                    "target_macro_f1": f1_score(
                        y_test, predicted, labels=classes, average="macro", zero_division=0
                    ),
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                }
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")

                elapsed = time.perf_counter() - started
                print(
                    f"[{i:>3}/{total}] {backbone:<9} layer:{layer:<2} seed {seed}  "
                    f"eff {own:>8.2f} / ref {ref:>9.2f}  "
                    f"val {row['source_val_macro_f1']:.4f}  "
                    f"tgt {row['target_macro_f1']:.4f}   ({elapsed/60:.1f} min)",
                    flush=True,
                )

    print(f"\ndone: {total} cells in {(time.perf_counter() - started)/60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
