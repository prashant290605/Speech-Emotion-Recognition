#!/usr/bin/env python
"""Compute label-harmonisation MMD diagnostics from a frozen sensitivity spec.

The analysis fits only on source-train and target-adapt features. Target-test
labels are read after the fitted-index assertion to calculate the descriptive
class-conditional diagnostic. Each rung receives its own source-target median
bandwidth, so raw and null-scaled MMD values are reported together and never
used as a conditional-to-marginal ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.alignment import build_alignment  # noqa: E402
from ser.analysis import assert_conditional_shift_firewall  # noqa: E402
from ser.analysis.shift import conditional_mmd_by_class  # noqa: E402
from ser.config import load_config  # noqa: E402
from ser.leakage import assert_alignment_blind_to_target_test  # noqa: E402
from ser.manifest import read_manifest  # noqa: E402
from ser.mmd import marginal_mmd, median_bandwidth, null_mmd_scale  # noqa: E402
from ser.utils.runmeta import capture_runmeta, hash_payload  # noqa: E402
from ser.utils.seeding import set_all_seeds  # noqa: E402

from run_calm_sensitivity import (  # noqa: E402
    SensitivityContext,
    load_spec,
    project_rows,
    sensitivity_config,
    variant_label_space,
)


ANALYSIS_VERSION = "label-sensitivity-mmd-v1"


def record_id(config, spec, *, source: str, target: str, seed: int, alignment: str) -> str:
    return hash_payload({
        "analysis_version": ANALYSIS_VERSION,
        "config_hash": config.config_hash,
        "label_map_hash": config.label_map_hash,
        "spec_version": spec["version"],
        "source": source,
        "target": target,
        "seed": seed,
        "backbone": spec["backbone"],
        "layer_agg": spec["layer_agg"],
        "alignment": alignment,
    })


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["record_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.config)
    spec = load_spec(spec_path)
    diagnostics_spec = spec.get("diagnostics")
    if not diagnostics_spec:
        raise ValueError("sensitivity spec requires a diagnostics section")
    if diagnostics_spec.get("bandwidth_rule") != "median_pairwise_distance_source_target":
        raise ValueError("unsupported diagnostics.bandwidth_rule")

    assert_conditional_shift_firewall()
    label_space = variant_label_space(spec)
    config = sensitivity_config(load_config(), spec, source_path=str(spec_path))
    rows = project_rows(
        read_manifest(config.resolve(config.paths.manifest)), config, label_space
    )
    context = SensitivityContext(config, rows, label_space)
    meta = capture_runmeta(config.config_hash, repo_root=REPO_ROOT)
    out = REPO_ROOT / diagnostics_spec["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_ids(out)
    rungs = tuple(spec["rungs"])
    planned = [
        (source, target, int(seed), rung)
        for source, target in spec["directions"]
        for seed in spec["seeds"]
        for rung in rungs
    ]
    todo = [
        item for item in planned
        if record_id(
            config, spec, source=item[0], target=item[1], seed=item[2],
            alignment=item[3]["alignment"],
        ) not in completed
    ]
    print(f"label diagnostics: {len(planned)} records, {len(todo)} remaining", flush=True)

    for index, (source, target, seed, rung) in enumerate(todo, start=1):
        pair = context.split(source, target, seed)
        classes = list(config.labels.spaces[pair.label_space])
        source_loader = context.loader(source, spec["backbone"])
        target_loader = context.loader(target, spec["backbone"])
        layer_spec = "last" if spec["layer_agg"] == "last" else spec["layer_agg"]
        X_train = source_loader.load(pair.source_train.utterance_ids, layer_spec=layer_spec)
        X_adapt = target_loader.load(pair.target_adapt.utterance_ids, layer_spec=layer_spec)
        X_test = target_loader.load(pair.target_test.utterance_ids, layer_spec=layer_spec)
        y_train = context.labels(pair, "source_train")
        y_test = context.labels(pair, "target_test")

        set_all_seeds(seed)
        alignment_name = rung["alignment"]
        aligned_train, aligned_test = X_train, X_test
        alignment_diagnostics = {}
        if alignment_name != "none":
            alignment = build_alignment(
                alignment_name,
                config,
                eps=rung.get("eps"),
                lam=rung.get("lam"),
                seed=seed,
            )
            alignment.fit(
                X_train,
                X_adapt,
                pair.target_adapt.utterance_ids,
                pair.source_train.utterance_ids,
            )
            assert_alignment_blind_to_target_test(alignment, pair)
            aligned_train = alignment.transform(X_train, domain="source")
            aligned_test = alignment.transform(X_test, domain="target")
            alignment_diagnostics = dict(alignment.diagnostics)

        bandwidth = median_bandwidth(aligned_train, aligned_test, seed=seed)
        marginal = marginal_mmd(
            aligned_train,
            aligned_test,
            config,
            bandwidth=bandwidth,
            max_samples=diagnostics_spec["max_samples"],
            seed=seed,
        )
        null = null_mmd_scale(
            aligned_train,
            config,
            bandwidth=bandwidth,
            n_repeats=diagnostics_spec["null_repeats"],
            max_samples=diagnostics_spec["max_samples"],
            seed=seed,
        )["scale"]
        conditional = conditional_mmd_by_class(
            aligned_train,
            y_train,
            aligned_test,
            y_test,
            classes,
            config,
            seed=seed,
            bandwidth=bandwidth,
            max_samples=diagnostics_spec["max_samples"],
        )
        row = {
            "analysis_version": ANALYSIS_VERSION,
            "record_id": record_id(
                config, spec, source=source, target=target, seed=seed,
                alignment=alignment_name,
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": meta.git_sha,
            "git_dirty": meta.git_dirty,
            "config_hash": config.config_hash,
            "label_map_hash": config.label_map_hash,
            "split_spec_hash": config.split_spec_hash,
            "variant": spec["variant"]["name"],
            "source": source,
            "target": target,
            "seed": seed,
            "backbone": spec["backbone"],
            "layer_agg": spec["layer_agg"],
            "alignment": alignment_name,
            "alignment_eps": rung.get("eps"),
            "alignment_lambda": rung.get("lam"),
            "classes": classes,
            "n_source_train": len(y_train),
            "n_target_test": len(y_test),
            "bandwidth": bandwidth,
            "bandwidth_rule": diagnostics_spec["bandwidth_rule"],
            "max_samples_per_mmd": diagnostics_spec["max_samples"],
            "null_repeats": diagnostics_spec["null_repeats"],
            "marginal_raw_mmd2": marginal,
            "marginal_null_abs_mmd2": null,
            "marginal_normalised": marginal / null if null > 0 else None,
            "conditional": conditional,
            "alignment_diagnostics": alignment_diagnostics,
        }
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[{index:>2}/{len(todo)}] {source[:4]}->{target[:4]} s{seed} "
            f"{alignment_name:<10} raw {marginal:.5f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
