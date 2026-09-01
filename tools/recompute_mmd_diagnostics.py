#!/usr/bin/env python
"""Recompute shift diagnostics in one source-defined reference geometry.

This is an analysis-only program. It reuses cached features and fits alignment
maps exactly as the grid did, but it never trains a classifier and never writes
to ``results/runs.jsonl``. Each record is appended to a separate JSONL file and
is skipped on a later invocation when its deterministic analysis id is present.

The reference ZCA basis and RBF median bandwidth are both derived from the
unaligned ``source_train`` features for a given pair, seed, backbone and layer
aggregation. They are then held fixed for all six alignment rungs. Target test
labels are read only after the alignment firewall assertion, to form the
class-conditional diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.alignment import build_alignment  # noqa: E402
from ser.analysis import assert_conditional_shift_firewall  # noqa: E402
from ser.analysis.shift import conditional_mmd_by_class  # noqa: E402
from ser.config import load_config  # noqa: E402
from ser.leakage import assert_alignment_blind_to_target_test  # noqa: E402
from ser.mmd import marginal_mmd, median_bandwidth, null_mmd_scale, reference_geometry  # noqa: E402
from ser.run_grid import REFERENCE_GEOMETRY_EPS, _Context  # noqa: E402
from ser.utils.runmeta import capture_runmeta, hash_payload  # noqa: E402
from ser.utils.seeding import set_all_seeds  # noqa: E402


ANALYSIS_VERSION = "fixed-reference-mmd-v1"
RUNGS = (
    ("none", None, None),
    ("zscore", None, None),
    ("mean_shift", None, None),
    ("coral", 1e-2, None),
    ("mkmmd_diag", None, 0.01),
    ("mkmmd_full", None, 0.01),
)
DIRECTIONS = (("ravdess", "cremad"), ("cremad", "ravdess"))


def analysis_id(config, *, source, target, seed, backbone, layer_agg):
    """Identifier for one map family before the alignment-rung coordinate."""
    return hash_payload(
        {
            "analysis": ANALYSIS_VERSION,
            "config_hash": config.config_hash,
            "source": source,
            "target": target,
            "seed": seed,
            "backbone": backbone,
            "layer_agg": layer_agg,
        }
    )


def record_id(base_id: str, alignment: str) -> str:
    return hash_payload({"base_id": base_id, "alignment": alignment})


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["record_id"])
    return ids


def mmd_summary(source, target, config, *, bandwidth, seed):
    """Raw unbiased MMD-squared and its source-half-split effect size."""
    raw = marginal_mmd(source, target, config, bandwidth=bandwidth, seed=seed)
    null = null_mmd_scale(
        source, config, bandwidth=bandwidth, n_repeats=5, seed=seed
    )["scale"]
    return {
        "raw_mmd2": float(raw),
        "null_abs_mmd2": float(null),
        "normalised": float(raw / null) if null > 0 else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="hubert")
    parser.add_argument("--aggs", default="last,layer")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--methods",
        default="none,zscore,mean_shift,coral,mkmmd_diag,mkmmd_full",
        help="Comma-separated subset of the six implemented alignment rungs.",
    )
    parser.add_argument("--out", default="results/phase9_reference_geometry.jsonl")
    args = parser.parse_args(argv)

    assert_conditional_shift_firewall()
    config = load_config()
    context = _Context(config)
    meta = capture_runmeta(config.config_hash, repo_root=REPO_ROOT)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_ids(out)
    seeds = [int(value) for value in args.seeds.split(",")]
    aggs = args.aggs.split(",")
    requested_methods = tuple(value for value in args.methods.split(",") if value)
    unknown_methods = sorted(set(requested_methods) - {name for name, _, _ in RUNGS})
    if unknown_methods:
        raise ValueError(f"unknown alignment methods: {unknown_methods}")
    rungs = tuple(rung for rung in RUNGS if rung[0] in requested_methods)
    if not rungs:
        raise ValueError("--methods selected no alignment rungs")

    planned = [
        (source, target, seed, agg, method, eps, lam)
        for source, target in DIRECTIONS
        for seed in seeds
        for agg in aggs
        for method, eps, lam in rungs
    ]
    todo = [
        item
        for item in planned
        if record_id(
            analysis_id(
                config, source=item[0], target=item[1], seed=item[2],
                backbone=args.backbone, layer_agg=item[3],
            ),
            item[4],
        ) not in completed
    ]
    print(f"reference MMD: {len(planned)} records, {len(todo)} remaining", flush=True)
    started = time.perf_counter()
    previous = None

    for index, (source, target, seed, agg, method, eps, lam) in enumerate(todo, start=1):
        key = (source, target, seed, agg)
        if key != previous:
            pair = context.split(source, target, seed)
            classes = list(config.labels.spaces[pair.label_space])
            y_train = context.labels(pair, "source_train")
            y_test = context.labels(pair, "target_test")
            spec = "last" if agg == "last" else f"layer:{config.classifiers.layer_candidates[1]}"
            source_loader = context.loader(source, args.backbone)
            target_loader = context.loader(target, args.backbone)
            X_train = source_loader.load(pair.source_train.utterance_ids, layer_spec=spec)
            X_adapt = target_loader.load(pair.target_adapt.utterance_ids, layer_spec=spec)
            X_test = target_loader.load(pair.target_test.utterance_ids, layer_spec=spec)
            geometry = reference_geometry(X_train, eps=REFERENCE_GEOMETRY_EPS)
            reference_bandwidth = median_bandwidth(geometry(X_train), seed=seed)
            base = analysis_id(
                config, source=source, target=target, seed=seed,
                backbone=args.backbone, layer_agg=agg,
            )
            previous = key

        set_all_seeds(seed)
        aligned_train = X_train
        aligned_test = X_test
        diagnostics = {}
        if method != "none":
            alignment = build_alignment(method, config, eps=eps, lam=lam, seed=seed)
            alignment.fit(
                X_train,
                X_adapt,
                pair.target_adapt.utterance_ids,
                pair.source_train.utterance_ids,
            )
            assert_alignment_blind_to_target_test(alignment, pair)
            aligned_train = alignment.transform(X_train, domain="source")
            aligned_test = alignment.transform(X_test, domain="target")
            diagnostics = dict(alignment.diagnostics)

        ref_train = geometry(aligned_train)
        ref_test = geometry(aligned_test)
        marginal = mmd_summary(
            ref_train, ref_test, config, bandwidth=reference_bandwidth, seed=seed
        )
        conditional = conditional_mmd_by_class(
            ref_train,
            y_train,
            ref_test,
            y_test,
            classes,
            config,
            seed=seed,
            bandwidth=reference_bandwidth,
        )
        row = {
            "analysis_version": ANALYSIS_VERSION,
            "record_id": record_id(base, method),
            "analysis_id": base,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": meta.git_sha,
            "git_dirty": meta.git_dirty,
            "config_hash": config.config_hash,
            "source": source,
            "target": target,
            "seed": seed,
            "backbone": args.backbone,
            "layer_agg": agg,
            "alignment": method,
            "alignment_eps": eps,
            "alignment_lambda": lam,
            "classes": classes,
            "n_source_train": len(y_train),
            "n_target_test": len(y_test),
            "reference_geometry": {
                "zca_source_train_eps": REFERENCE_GEOMETRY_EPS,
                "bandwidth_source_train": reference_bandwidth,
                "bandwidth_rule": "median_pairwise_distance_source_train",
            },
            "marginal": marginal,
            "conditional": conditional,
            "alignment_diagnostics": diagnostics,
        }
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

        elapsed = time.perf_counter() - started
        print(
            f"[{index:>3}/{len(todo)}] {source[:4]}->{target[:4]} s{seed} "
            f"{agg:<5} {method:<11} raw {marginal['raw_mmd2']:.5f} "
            f"xnull {marginal['normalised']:.2f} ({elapsed / 60:.1f} min)",
            flush=True,
        )

    print(f"done: {len(todo)} appended to {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
