#!/usr/bin/env python
"""Phase 9: the three-way shift decomposition, and a falsifiable test of it.

    python tools/phase9_shift.py

Writes results/phase9_shift.jsonl -- **not** results/runs.jsonl, and never a
schema row. The conditional term is derived from target test labels, and A10
forbids writing it anywhere the pipeline can read it back. The firewall
assertion runs before anything is computed.

Per (direction, seed, layer aggregation, rung):

``label shift``       KL(P_target || P_source) between the realised partition
                      priors, plus total variation. No features involved.
``covariate shift``   marginal MMD in both frames, on the same two sets the
                      conditional term uses, so the two are comparable.
``conditional shift`` MMD(X_src|y=k, X_tgt|y=k) per class, min-support
                      enforced, per-class n recorded.
``label correction``  BBSE and SLD-EM estimates of the target prior, applied to
                      a logreg fitted on the aligned source, scored before and
                      after. Near-zero prior KL predicts no gain; a gain would
                      falsify the decomposition.
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
    os.environ.setdefault(_var, "2")

from ser.features.audio import warm_up_audio_stack  # noqa: E402

warm_up_audio_stack()

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from ser.alignment import build_alignment  # noqa: E402
from ser.analysis import assert_conditional_shift_firewall  # noqa: E402
from ser.analysis.shift import (  # noqa: E402
    apply_prior_correction,
    bbse_weights,
    class_priors,
    conditional_mmd_by_class,
    em_prior_estimate,
    label_shift_kl,
)
from ser.config import load_config  # noqa: E402
from ser.leakage import assert_alignment_blind_to_target_test  # noqa: E402
from ser.metrics import macro_f1  # noqa: E402
from ser.mmd import (  # noqa: E402
    marginal_mmd,
    median_bandwidth,
    null_mmd_scale,
    reference_geometry,
)
from ser.run_grid import REFERENCE_GEOMETRY_EPS, _Context  # noqa: E402
from ser.utils.seeding import set_all_seeds  # noqa: E402

RUNGS = [
    ("none", None, None),
    ("zscore", None, None),
    ("mean_shift", None, None),
    ("coral", 1e-2, None),
    ("mkmmd_diag", None, 0.01),
    ("mkmmd_full", None, 0.01),
]
DIRECTIONS = [("ravdess", "cremad"), ("cremad", "ravdess")]


def _effect(A, B, config, seed):
    bandwidth = median_bandwidth(A, B, seed=seed)
    raw = marginal_mmd(A, B, config, bandwidth=bandwidth, seed=seed)
    null = null_mmd_scale(A, config, bandwidth=bandwidth, n_repeats=5, seed=seed)["scale"]
    return (float(raw / null) if null > 0 else None), float(raw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="hubert")
    parser.add_argument("--aggs", default="last,layer")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", default="results/phase9_shift.jsonl")
    args = parser.parse_args(argv)

    # A10: before any target label is touched.
    assert_conditional_shift_firewall()

    config = load_config()
    context = _Context(config)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    seeds = [int(s) for s in args.seeds.split(",")]
    aggs = args.aggs.split(",")
    total = len(DIRECTIONS) * len(seeds) * len(aggs) * len(RUNGS)
    started = time.perf_counter()
    index = 0

    for source, target in DIRECTIONS:
        for seed in seeds:
            pair = context.split(source, target, seed)
            classes = list(config.labels.spaces[pair.label_space])
            y_train = context.labels(pair, "source_train")
            y_val = context.labels(pair, "source_val")
            y_test = context.labels(pair, "target_test")

            # -- label shift: no features, no alignment, once per (pair, seed) --
            priors = label_shift_kl(y_train, y_test, classes)

            for agg in aggs:
                spec = "last" if agg == "last" else f"layer:{config.classifiers.layer_candidates[1]}"
                src_loader = context.loader(source, args.backbone)
                tgt_loader = context.loader(target, args.backbone)
                X_train = src_loader.load(pair.source_train.utterance_ids, layer_spec=spec)
                X_val = src_loader.load(pair.source_val.utterance_ids, layer_spec=spec)
                X_adapt = tgt_loader.load(pair.target_adapt.utterance_ids, layer_spec=spec)
                X_test = tgt_loader.load(pair.target_test.utterance_ids, layer_spec=spec)
                geometry = reference_geometry(X_train, eps=REFERENCE_GEOMETRY_EPS)

                for method, eps, lam in RUNGS:
                    index += 1
                    set_all_seeds(seed)
                    A_train, A_val, A_test = X_train, X_val, X_test

                    if method != "none":
                        alignment = build_alignment(
                            method, config, eps=eps, lam=lam, seed=seed
                        )
                        alignment.fit(
                            X_train, X_adapt,
                            pair.target_adapt.utterance_ids,
                            pair.source_train.utterance_ids,
                        )
                        assert_alignment_blind_to_target_test(alignment, pair)
                        A_train = alignment.transform(X_train, domain="source")
                        A_val = alignment.transform(X_val, domain="source")
                        A_test = alignment.transform(X_test, domain="target")

                    # -- covariate shift, on the same two sets the conditional
                    #    term uses, so the decomposition is internally comparable
                    own, own_raw = _effect(A_train, A_test, config, seed)
                    ref, _ = _effect(geometry(A_train), geometry(A_test), config, seed)

                    # -- conditional shift (A10: reads target test labels) -----
                    conditional = conditional_mmd_by_class(
                        A_train, y_train, A_test, y_test, classes, config, seed=seed
                    )

                    # -- label-shift correction as a falsifiable test ----------
                    model = LogisticRegression(
                        max_iter=config.classifiers.sklearn_max_iter, random_state=seed
                    ).fit(A_train, y_train)
                    order = list(model.classes_)
                    P_val = model.predict_proba(A_val)
                    P_test = model.predict_proba(A_test)
                    source_prior = class_priors(y_train, order)

                    predicted_val = model.predict(A_val)
                    confusion = np.zeros((len(order), len(order)))
                    for i, a in enumerate(order):
                        for j, b in enumerate(order):
                            confusion[i, j] = np.mean(
                                (np.asarray(y_val) == a) & (predicted_val == b)
                            )
                    mu = np.array([np.mean(model.predict(A_test) == c) for c in order])
                    weights = bbse_weights(confusion, mu)
                    em_prior = em_prior_estimate(P_test, source_prior)

                    baseline = macro_f1(y_test, list(model.predict(A_test)), classes)
                    scores = {"uncorrected": baseline}
                    if weights is not None and weights.sum() > 0:
                        bbse_prior = source_prior * weights
                        bbse_prior = bbse_prior / bbse_prior.sum()
                        decided = np.array(order)[
                            apply_prior_correction(P_test, source_prior, bbse_prior).argmax(1)
                        ]
                        scores["bbse"] = macro_f1(y_test, list(decided), classes)
                        scores["bbse_prior"] = bbse_prior.tolist()
                    decided = np.array(order)[
                        apply_prior_correction(P_test, source_prior, em_prior).argmax(1)
                    ]
                    scores["em"] = macro_f1(y_test, list(decided), classes)
                    scores["em_prior"] = em_prior.tolist()

                    record = {
                        "source": source, "target": target, "seed": seed,
                        "backbone": args.backbone, "layer_agg": agg,
                        "alignment": method, "alignment_eps": eps,
                        "alignment_lambda": lam,
                        "classes": classes,
                        "label_shift": priors,
                        "marginal_effect_own": own,
                        "marginal_effect_reference": ref,
                        "marginal_raw": own_raw,
                        "conditional": conditional,
                        "correction": scores,
                        "n_source_train": len(y_train),
                        "n_target_test": len(y_test),
                    }
                    with out.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")

                    elapsed = time.perf_counter() - started
                    print(f"[{index:>3}/{total}] {source[:4]}->{target[:4]} s{seed} "
                          f"{agg:<5} {method:<11} marg {own:>8.1f} "
                          f"KL {priors['kl_nats']:.4f} "
                          f"({elapsed/60:.1f} min)", flush=True)

    print(f"\ndone: {total} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
