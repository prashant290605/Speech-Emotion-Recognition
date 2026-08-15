"""End-to-end sanity run over the alignment ladder.

Fits every rung on one corpus pair and reports, for each:

* marginal MMD² between source and target **before and after** alignment --
  the covariate-shift column of the Phase 9 decomposition, collected here
  because it is free once the features are loaded;
* covariance conditioning, so a near-singular fit is visible rather than
  inferred;
* the Phase 2 leakage assertion, run against the fitted object.

This trains nothing and selects nothing. Choosing eps and lambda on
``source_val`` needs a classifier, which is Phase 6.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np

from .alignment import LADDER, build_alignment
from .features.load import FeatureLoader
from .leakage import assert_alignment_blind_to_target_test
from .manifest import read_manifest
from .mmd import kernel_saturation, marginal_mmd, median_bandwidth, null_mmd_scale
from .numerics import SingularCovariance
from .splits import make_pair_split
from .utils.seeding import set_all_seeds

__all__ = ["run_alignment_check"]


def _conditions(config) -> List[Dict]:
    """Every rung, with its regularisation variants expanded."""
    out: List[Dict] = []
    for method in config.alignment.ladder_order():
        if method == "coral":
            for eps in config.alignment.coral_shrinkage:
                out.append({"method": method, "eps": eps, "label": f"coral(eps={eps:g})"})
            if config.alignment.coral_ledoit_wolf:
                out.append(
                    {"method": method, "ledoit_wolf": True, "label": "coral(ledoit-wolf)"}
                )
        elif method.startswith("mkmmd"):
            for lam in config.alignment.mmd_lambda_grid:
                out.append(
                    {"method": method, "lam": lam, "label": f"{method}(lam={lam:g})"}
                )
        else:
            out.append({"method": method, "label": method})
    return out


def run_alignment_check(
    config,
    source: str,
    target: str,
    *,
    seed: int = 0,
    backbone: str = "hubert",
    layer_spec: str = "layer:6",
    lambdas: Optional[Sequence[float]] = None,
) -> int:
    rows = read_manifest(config.resolve(config.paths.manifest))
    set_all_seeds(seed)

    pair = make_pair_split(rows, config, source, target, seed)
    source_loader = FeatureLoader(config, source, backbone, rows)
    target_loader = FeatureLoader(config, target, backbone, rows)

    X_source = source_loader.load(
        pair.source_train.utterance_ids, layer_spec=layer_spec
    )
    X_adapt = target_loader.load(
        pair.target_adapt.utterance_ids, layer_spec=layer_spec
    )
    X_test = target_loader.load(pair.target_test.utterance_ids, layer_spec=layer_spec)

    print(f"pair {source} -> {target} | seed {seed} | {backbone} | {layer_spec}")
    print(
        f"source_train {X_source.shape}  target_adapt {X_adapt.shape}  "
        f"target_test {X_test.shape}  dtype {X_source.dtype}"
    )
    print()

    # CORRECTION 1: the bandwidth is estimated ONCE, on the unaligned pair, and
    # reused for every rung. Re-estimating per rung makes the statistic
    # scale-dependent -- a rung that merely shrinks the features shrinks the
    # median pairwise distance too, the kernel widens to compensate, and the
    # reported MMD falls without anything having moved closer.
    bandwidth = median_bandwidth(X_source, X_adapt, seed=seed)
    null = null_mmd_scale(X_source, config, bandwidth=bandwidth, seed=seed)

    before = marginal_mmd(X_source, X_adapt, config, bandwidth=bandwidth, seed=seed)
    print(f"fixed bandwidth (unaligned median heuristic): {bandwidth:.4f}")
    print(
        f"null scale (same-distribution MMD^2, {null['n_repeats']} half-splits of "
        f"source_train): {null['scale']:.3e}"
    )
    print(f"marginal MMD^2 before alignment: {before:.6f}  "
          f"= {before / null['scale']:.1f}x null")
    print()

    header = (
        f"{'rung':<24} {'MMD^2@fixed':>12} {'sat':>8} {'effectsize':>11} "
        f"{'cond':>10} {'rank':>6}  leak"
    )
    print(header)
    print("-" * len(header))

    conditions = _conditions(config)
    if lambdas is not None:
        conditions = [
            c for c in conditions
            if not c["method"].startswith("mkmmd") or c.get("lam") in set(lambdas)
        ]

    results = []
    for condition in conditions:
        method = condition["method"]
        try:
            alignment = build_alignment(
                method,
                config,
                eps=condition.get("eps"),
                ledoit_wolf=condition.get("ledoit_wolf", False),
                lam=condition.get("lam"),
                seed=seed,
            )
            alignment.fit(
                X_source,
                X_adapt,
                pair.target_adapt.utterance_ids,
                pair.source_train.utterance_ids,
            )
            aligned = alignment.transform(X_source, domain="source")
            adapted = alignment.transform(X_adapt, domain="target")
            # Two statistics, because neither alone is trustworthy here.
            #
            # `after` uses the bandwidth fixed on the unaligned pair, so it is
            # directly comparable across rungs -- but it is INVALID for any rung
            # that changes the feature scale, because the kernel saturates.
            #
            # `normalised` is the effect size: MMD at a bandwidth appropriate to
            # the transformed data, divided by the same-distribution MMD of that
            # same transformed data. Both numerator and denominator move
            # together under rescaling, so the ratio cannot be manufactured by
            # shrinking or expanding the representation.
            after = marginal_mmd(
                aligned, adapted, config, bandwidth=bandwidth, seed=seed
            )
            saturation = kernel_saturation(
                aligned, adapted, bandwidth,
                config.alignment.mmd_bandwidth_multipliers,
            )
            own_bandwidth = median_bandwidth(aligned, adapted, seed=seed)
            own_mmd = marginal_mmd(
                aligned, adapted, config, bandwidth=own_bandwidth, seed=seed
            )
            own_null = null_mmd_scale(
                aligned, config, bandwidth=own_bandwidth, seed=seed
            )["scale"]
            normalised = own_mmd / own_null if own_null > 0 else float("nan")

            # The Phase 2 contract, on a real fitted object.
            assert_alignment_blind_to_target_test(alignment, pair)
            leakage = "OK"

            fields = alignment.row_fields()
            condition_number = fields["cov_condition_number"]
            effective_rank = fields["cov_effective_rank"]

            # ASCII in console output: Windows consoles default to cp1252 and
            # would render an em dash as mojibake. The markdown report is UTF-8.
            cond_text = "n/a" if condition_number is None else f"{condition_number:.3e}"
            rank_text = "n/a" if effective_rank is None else f"{effective_rank:.1f}"
            reduction = (before - after) / abs(before) * 100 if before else 0.0
            flag = " SATURATED" if saturation < 1e-2 else ""
            print(
                f"{condition['label']:<24} {after:12.6f} {saturation:8.1e} "
                f"{normalised:11.1f} {cond_text:>10} {rank_text:>6}  {leakage}{flag}"
            )
            results.append(
                {
                    "label": condition["label"],
                    "method": method,
                    "mmd2_before": before,
                    "mmd2_after": after,
                    "marginal_mmd_raw": after,
                    "marginal_mmd_normalised": normalised,
                    "kernel_saturation": saturation,
                    "own_bandwidth": own_bandwidth,
                    **fields,
                    "diagnostics": alignment.diagnostics,
                }
            )
        except SingularCovariance as exc:
            print(f"{condition['label']:<26} {'FAILED':>12}  {exc}")
            results.append({"label": condition["label"], "method": method,
                            "error": str(exc)})

    print()
    _write_report(config, source, target, seed, backbone, layer_spec, before, results, null, bandwidth)
    return 0


def _write_report(config, source, target, seed, backbone, layer_spec, before, results, null, bandwidth):
    lines = ["# Alignment ladder — sanity run", ""]
    lines.append(
        f"`{source} → {target}`, seed {seed}, {backbone}, layer spec `{layer_spec}`. "
        "Generated by `ser align-check`. Nothing is trained or selected here; "
        "choosing eps and lambda on `source_val` needs a classifier (Phase 6)."
    )
    lines.append("")
    lines.append(
        f"Bandwidth **{bandwidth:.4f}**, estimated once on the unaligned pair and "
        f"reused for every rung. Null scale **{null['scale']:.3e}** "
        f"(mean |MMD²| over {null['n_repeats']} half-splits of source_train, "
        f"{null['n_per_half']} per half) — the discrepancy attributable to finite "
        "sampling alone."
    )
    lines.append("")
    lines.append(
        f"Marginal MMD² before alignment: **{before:.6f}** "
        f"(**{before / null['scale']:.1f}x** null)"
    )
    lines.append("")
    lines.append(
        "`xnull` is the scale-invariant statistic: MMD² divided by the null "
        "scale. Rescaling the features cannot change it, so unlike a percentage "
        "reduction it cannot be manufactured by shrinking the representation."
    )
    lines.append("")
    lines.append(
        "| rung | MMD² after | xnull | reduction | cond. number | effective rank |"
    )
    lines.append("|---|---|---|---|---|---|")
    for entry in results:
        if "error" in entry:
            lines.append(f"| {entry['label']} | FAILED | — | — | {entry['error']} |")
            continue
        condition_number = entry.get("cov_condition_number")
        effective_rank = entry.get("cov_effective_rank")
        reduction = (before - entry["mmd2_after"]) / abs(before) * 100 if before else 0.0
        lines.append(
            f"| {entry['label']} | {entry['mmd2_after']:.6f} | "
            f"{entry['marginal_mmd_normalised']:.1f} | {reduction:.1f}% | "
            f"{'—' if condition_number is None else f'{condition_number:.3e}'} | "
            f"{'—' if effective_rank is None else f'{effective_rank:.1f}'} |"
        )
    lines.append("")
    lines.append(
        "Marginal MMD² is the covariate-shift measure Phase 9 needs. A rung that "
        "shrinks it a long way while transfer macro-F1 does not move is exactly "
        "the evidence amendment A8 predicts."
    )
    lines.append("")

    out = config.resolve(config.paths.reports_dir) / "alignment_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
