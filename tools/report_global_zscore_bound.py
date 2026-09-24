#!/usr/bin/env python
"""Report the globally corrected one-sided z-score comparison bound."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ser.phase8 import paired_cluster_bootstrap  # noqa: E402
from make_figures import Data, LADDER, PAIRS  # noqa: E402


FAMILYWISE_ALPHA = 0.05
N_BOOT = 2000
BOOTSTRAP_SEED = 17
ARROW = {
    ("ravdess", "cremad"): "RAVDESS->CREMA-D",
    ("cremad", "ravdess"): "CREMA-D->RAVDESS",
}


def one_sided_family_alpha(family_size: int) -> float:
    """Two-sided alpha with a one-sided Bonferroni familywise upper bound."""
    if family_size <= 0:
        raise ValueError("family_size must be positive")
    return 2 * FAMILYWISE_ALPHA / family_size


def paired_difference(
    data: Data, source: str, target: str, rung_a: str, rung_b: str,
    *, n_boot: int = N_BOOT, alpha: float = 0.05,
) -> dict:
    """Bootstrap target macro-F1 for a source-selected rung contrast."""
    pool = [
        row for row in data.main
        if (row["source_corpus"], row["target_corpus"]) == (source, target)
    ]
    buckets = defaultdict(list)
    for row in pool:
        if row["alignment"] in (rung_a, rung_b):
            buckets[
                (row["seed"], row["backbone"], row["layer_agg"], row["classifier"])
            ].append(row)

    arm_a, arm_b = defaultdict(list), defaultdict(list)
    for (seed, *_), rows in buckets.items():
        rows_a = [row for row in rows if row["alignment"] == rung_a]
        rows_b = [row for row in rows if row["alignment"] == rung_b]
        if not rows_a or not rows_b:
            continue
        arm_a[seed].append(
            data.confusion(max(rows_a, key=lambda row: row["selection_source_val_macro_f1"]))
        )
        arm_b[seed].append(
            data.confusion(max(rows_b, key=lambda row: row["selection_source_val_macro_f1"]))
        )

    speakers = data.n_speakers(source, target)
    return paired_cluster_bootstrap(
        dict(arm_a), dict(arm_b), speakers, n_boot=n_boot,
        seed=BOOTSTRAP_SEED, alpha=alpha,
    )


def calculate_global_bounds(data: Data, *, n_boot: int = N_BOOT) -> dict:
    """Return the largest possible advantage under one global eight-test family."""
    competitors = tuple(rung for rung in LADDER if rung not in ("none", "zscore"))
    family_size = len(PAIRS) * len(competitors)
    results = {}
    for source, target in PAIRS:
        comparisons = {}
        for rung in competitors:
            per_contrast = paired_difference(
                data, source, target, rung, "zscore", n_boot=n_boot, alpha=0.10
            )
            global_bound = paired_difference(
                data, source, target, rung, "zscore", n_boot=n_boot,
                alpha=one_sided_family_alpha(family_size),
            )
            comparisons[rung] = {
                "difference": per_contrast["diff"],
                "per_contrast_upper": per_contrast["hi"],
                "global_upper": global_bound["hi"],
                "n_seeds": per_contrast["n_seeds"],
            }
        limiting_rung = max(comparisons, key=lambda rung: comparisons[rung]["global_upper"])
        results[(source, target)] = {
            "limiting_rung": limiting_rung,
            "comparisons": comparisons,
        }
    return {"family_size": family_size, "directions": results}


def markdown_lines(summary: dict) -> list[str]:
    """Render the auditable result fragment used by the manuscript ledger."""
    family_size = summary["family_size"]
    confidence = 100 * (1 - FAMILYWISE_ALPHA / family_size)
    lines = [
        "## Global one-sided z-score bound",
        "",
        "The claim that no rung is shown to beat zscore covers four competitors "
        "in each of two directions. The resulting eight contrasts use one global "
        f"one-sided Bonferroni family. Each upper endpoint is the {confidence:.3f}th "
        "bootstrap percentile, so all eight upper bounds have at least 95% familywise coverage.",
        "",
        "| direction | rung determining bound | difference | per-contrast 95% upper | global 95% upper |",
        "|---|---|---:|---:|---:|",
    ]
    for direction, item in summary["directions"].items():
        rung = item["limiting_rung"]
        statistic = item["comparisons"][rung]
        lines.append(
            f"| {ARROW[direction]} | {rung} | {statistic['difference']:+.4f} | "
            f"{statistic['per_contrast_upper']:+.5f} | **{statistic['global_upper']:+.5f}** |"
        )
    lines.extend([
        "",
        "The selected rung is the one with the largest globally corrected upper bound "
        "within its direction. A non-positive upper bound does not prove a rung is worse; "
        "it only rules out an advantage at this confidence level.",
    ])
    return lines


def main() -> int:
    summary = calculate_global_bounds(Data())
    output = REPO_ROOT / "reports" / "zscore_global_bound.md"
    output.write_text("\n".join(markdown_lines(summary)) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
