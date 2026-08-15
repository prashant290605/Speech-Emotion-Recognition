"""Phase 6 acceptance run: every family, equal budget, selection on source_val.

Runs each (family, layer aggregation) condition end to end on one corpus pair
and reports two protocols side by side:

**Validated** — the condition with the best ``source_val`` macro-F1, and *its*
target score. What a practitioner without target labels could actually get.

**Oracle** — the best ``target_test`` macro-F1 over the same conditions,
labelled as an upper bound. This is what the original Table 1 reported without
saying so, and the gap between the two is a result in its own right.

Target scores are computed for every condition so the oracle can be reported,
but they are computed **after** selection and never feed back into it. The
selection surface is `source_val`, full stop.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .classifiers import FAMILIES, fit_and_select, supports_layer_agg
from .features.load import FeatureLoader
from .manifest import read_manifest
from .metrics import all_metrics
from .splits import make_pair_split
from .utils.seeding import set_all_seeds

__all__ = ["run_classifier_check"]


def _labels_for(pair, role, by_id) -> List[str]:
    space = pair.label_space
    return [
        (by_id[uid].label_six if space == "six" else by_id[uid].label_four)
        for uid in pair.splits()[role].utterance_ids
    ]


def _conditions(config, layer: int) -> List[Dict]:
    out = []
    for family in config.classifiers.families:
        for agg in config.classifiers.layer_agg_options:
            if not supports_layer_agg(family, agg):
                continue
            spec = {"last": "last", "layer": f"layer:{layer}", "weighted": "weighted"}[agg]
            out.append(
                {
                    "family": family,
                    "layer_agg": agg,
                    "layer_spec": spec,
                    "layer_index": layer if agg == "layer" else None,
                }
            )
    return out


def run_classifier_check(
    config,
    source: str,
    target: str,
    *,
    seed: int = 0,
    backbone: str = "hubert",
    layer: int = 6,
    families: Optional[Sequence[str]] = None,
) -> int:
    rows = read_manifest(config.resolve(config.paths.manifest))
    by_id = {row.utterance_id: row for row in rows}
    set_all_seeds(seed)

    pair = make_pair_split(rows, config, source, target, seed)
    classes = list(config.labels.spaces[pair.label_space])

    y_train = _labels_for(pair, "source_train", by_id)
    y_val = _labels_for(pair, "source_val", by_id)
    y_test = _labels_for(pair, "target_test", by_id)

    source_loader = FeatureLoader(config, source, backbone, rows)
    target_loader = FeatureLoader(config, target, backbone, rows)

    print(f"pair {source} -> {target} | seed {seed} | {backbone} | K={len(classes)}")
    print(
        f"source_train {len(y_train)}  source_val {len(y_val)}  "
        f"target_test {len(y_test)}"
    )
    print(f"budget: {config.classifiers.search_budget} trials per condition, identical")
    print()

    conditions = _conditions(config, layer)
    if families:
        conditions = [c for c in conditions if c["family"] in set(families)]

    header = (
        f"{'family':<12} {'layer_agg':<10} {'trials':>6} {'source_val':>11} "
        f"{'target':>8} {'epochs':>7} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))

    results: List[Dict] = []
    for condition in conditions:
        needs_segments = condition["family"] == "transformer"
        started = time.perf_counter()
        try:
            X_train = source_loader.load(
                pair.source_train.utterance_ids,
                layer_spec=condition["layer_spec"],
                segments=needs_segments,
            )
            X_val = source_loader.load(
                pair.source_val.utterance_ids,
                layer_spec=condition["layer_spec"],
                segments=needs_segments,
            )
            X_test = target_loader.load(
                pair.target_test.utterance_ids,
                layer_spec=condition["layer_spec"],
                segments=needs_segments,
            )

            selection = fit_and_select(
                condition["family"],
                X_train,
                y_train,
                X_val,
                y_val,
                classes,
                config,
                layer_agg=condition["layer_agg"],
                seed=seed,
            )

            # Target is scored only after selection is complete.
            predictions = selection.predict(X_test)
            metrics = all_metrics(y_test, list(predictions), classes)
            elapsed = time.perf_counter() - started

            print(
                f"{condition['family']:<12} {condition['layer_agg']:<10} "
                f"{selection.n_trials:>6} {selection.best_source_val_macro_f1:>11.4f} "
                f"{metrics['macro_f1']:>8.4f} "
                f"{selection.epochs_run if selection.epochs_run else '-':>7} "
                f"{elapsed:>6.0f}",
                flush=True,
            )
            results.append(
                {
                    **condition,
                    "n_trials": selection.n_trials,
                    "source_val": selection.best_source_val_macro_f1,
                    "target": metrics["macro_f1"],
                    "uar": metrics["uar"],
                    "collapsed": metrics["n_collapsed_classes"],
                    "epochs": selection.epochs_run,
                    "hyperparams": selection.as_hyperparams(),
                    "wall_seconds": elapsed,
                }
            )
        except Exception as exc:  # noqa: BLE001 - a failed condition is data
            print(f"{condition['family']:<12} {condition['layer_agg']:<10} FAILED: {exc}")
            results.append({**condition, "error": str(exc)})

    ok = [r for r in results if "error" not in r]
    if ok:
        validated = max(ok, key=lambda r: r["source_val"])
        oracle = max(ok, key=lambda r: r["target"])
        print()
        print(
            f"VALIDATED (selected on source_val): {validated['family']}/"
            f"{validated['layer_agg']}  source_val={validated['source_val']:.4f}  "
            f"target={validated['target']:.4f}"
        )
        print(
            f"ORACLE    (max over target, UPPER BOUND, not achievable without "
            f"target labels): {oracle['family']}/{oracle['layer_agg']}  "
            f"target={oracle['target']:.4f}"
        )
        print(f"GAP: {oracle['target'] - validated['target']:+.4f} macro-F1")

        budgets = {r["n_trials"] for r in ok}
        print(f"\nequal-budget check: trial counts across conditions = {budgets}")

    _write_report(config, source, target, seed, backbone, classes, results)
    return 0


def _write_report(config, source, target, seed, backbone, classes, results) -> None:
    ok = [r for r in results if "error" not in r]
    lines = ["# Classifier check — equal budget, selection on source_val", ""]
    lines.append(
        f"`{source} → {target}`, seed {seed}, {backbone}, K={len(classes)}. "
        f"Budget: **{config.classifiers.search_budget} trials per condition**, "
        "identical for every family. Generated by `ser classify-check`."
    )
    lines.append("")
    lines.append(
        "| family | layer agg | trials | source_val | target | epochs | collapsed |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            lines.append(
                f"| {r['family']} | {r['layer_agg']} | — | — | — | — | {r['error'][:60]} |"
            )
            continue
        lines.append(
            f"| {r['family']} | {r['layer_agg']} | {r['n_trials']} | "
            f"{r['source_val']:.4f} | {r['target']:.4f} | "
            f"{r['epochs'] or '—'} | {r['collapsed']} |"
        )
    lines.append("")

    if ok:
        validated = max(ok, key=lambda r: r["source_val"])
        oracle = max(ok, key=lambda r: r["target"])
        lines.append("## Validated vs oracle")
        lines.append("")
        lines.append("| protocol | condition | target macro-F1 |")
        lines.append("|---|---|---|")
        lines.append(
            f"| **validated** (selected on source_val) | "
            f"{validated['family']}/{validated['layer_agg']} | {validated['target']:.4f} |"
        )
        lines.append(
            f"| *oracle* (max over target_test — UPPER BOUND) | "
            f"{oracle['family']}/{oracle['layer_agg']} | {oracle['target']:.4f} |"
        )
        lines.append(f"| gap | | {oracle['target'] - validated['target']:+.4f} |")
        lines.append("")
        lines.append(
            "The oracle is **not achievable without target labels**. It is reported "
            "because the original Table 1 reported a grid maximum over target test "
            "without labelling it as one, and the gap quantifies how much that "
            "habit overstates cross-corpus transfer."
        )
        lines.append("")

    out = config.resolve(config.paths.reports_dir) / "classifier_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
