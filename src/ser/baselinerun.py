"""Compute the chance floors for every pair and seed, and record them.

Writes one row per (baseline, pair, seed) to ``results/runs.jsonl`` with
``classifier="baseline_*"``. **Every** row -- including each baseline's own --
carries all three floor columns, so from here on no metric can be reported
without its floor beside it.

Floors are per pair, not global: after amendment A4 the IEMOCAP pairs run
4-class and RAVDESS↔CREMA-D runs 6-class, so chance is 0.250 for one and 0.167
for the other. A single headline "chance" number would be wrong for half the
grid.
"""

from __future__ import annotations

import json
import time
from itertools import product
from typing import Dict, List, Sequence

from .baselines import all_floors
from .manifest import read_manifest
from .splits import make_pair_split
from .utils.results import append_row, completed_run_ids, make_run_id, new_row
from .utils.runmeta import capture_runmeta
from .utils.seeding import set_all_seeds

__all__ = ["run_baselines"]

BASELINE_TO_COLUMN = {
    "uniform_random": "chance_macro_f1",
    "majority": "majority_macro_f1",
    "stratified_random": "prior_matched_macro_f1",
}


def _labels_for(pair_split, role, by_id, label_space) -> List[str]:
    ids = pair_split.splits()[role].utterance_ids
    return [
        (by_id[uid].label_six if label_space == "six" else by_id[uid].label_four)
        for uid in ids
    ]


def run_baselines(config, corpora: Sequence[str], *, force: bool = False) -> int:
    rows = read_manifest(config.resolve(config.paths.manifest))
    by_id = {row.utterance_id: row for row in rows}
    present = [name for name in corpora if any(r.corpus == name for r in rows)]

    results_path = config.results_path
    already = set() if force else completed_run_ids(results_path)

    written = 0
    skipped = 0
    summary: List[Dict] = []

    for source, target in product(present, repeat=2):
        for seed in config.splits.seeds:
            set_all_seeds(seed)
            pair = make_pair_split(rows, config, source, target, seed)
            classes = list(config.labels.spaces[pair.label_space])

            y_true = _labels_for(pair, "target_test", by_id, pair.label_space)
            source_labels = _labels_for(pair, "source_train", by_id, pair.label_space)

            started = time.perf_counter()
            floors = all_floors(
                y_true,
                classes,
                source_labels,
                n_draws=config.baselines.n_random_draws,
                seed=seed,
                ci_level=config.stats.ci_level,
            )
            elapsed = time.perf_counter() - started

            floor_columns = {
                column: floors[name].macro_f1
                for name, column in BASELINE_TO_COLUMN.items()
            }

            for name, result in floors.items():
                coords = {
                    "label_map_hash": config.label_map_hash,
                    "split_spec_hash": config.split_spec_hash,
                    "feature_spec_hash": config.feature_spec_hash,
                    "search_spec_hash": config.search_spec_hash,
                    "seed": seed,
                    "source_corpus": source,
                    "target_corpus": target,
                    "backbone": "none",
                    "layer_agg": "n/a",
                    "layer_index": None,
                    "feature_branch": "none",
                    "alignment": "none",
                    "blending": "none",
                    "blend_alpha": None,
                    "n_groups": None,
                    "classifier": f"baseline_{name}",
                    "split_id": pair.split_id,
                }
                run_id = make_run_id(coords)
                if run_id in already:
                    skipped += 1
                    continue

                meta = capture_runmeta(config.config_hash)
                sizes = pair.sizes()
                row = new_row(
                    **{**coords, **meta.as_row_fields()},
                    run_id=run_id,
                    n_classes=len(classes),
                    class_names=classes,
                    hyperparams_json=json.dumps(
                        {
                            "n_draws": result.n_draws,
                            "analytic_macro_f1": result.analytic_macro_f1,
                            "ci_low": result.ci_low,
                            "ci_high": result.ci_high,
                            "label_space": pair.label_space,
                            **result.details,
                        },
                        sort_keys=True,
                        default=str,
                    ),
                    n_train=sizes["source_train"],
                    n_val=sizes["source_val"],
                    n_target_adapt=sizes["target_adapt"],
                    n_target_test=sizes["target_test"],
                    macro_f1=result.macro_f1,
                    accuracy=result.accuracy,
                    uar=result.uar,
                    per_class_f1_json=json.dumps(result.per_class_f1, sort_keys=True),
                    confusion_json=json.dumps(result.confusion),
                    **floor_columns,
                    selection_source_val_macro_f1=None,
                    cov_condition_number=None,
                    cov_effective_rank=None,
                    n_search_trials=None,
                    marginal_mmd_raw=None,
                    marginal_mmd_normalised=None,
                    wall_seconds=round(elapsed / len(floors), 6),
                    status="ok",
                    error=None,
                )
                append_row(results_path, row)
                already.add(run_id)
                written += 1

            summary.append(
                {
                    "source": source,
                    "target": target,
                    "seed": seed,
                    "label_space": pair.label_space,
                    "n_classes": len(classes),
                    "n_target_test": len(y_true),
                    **{name: floors[name].macro_f1 for name in floors},
                    **{
                        f"{name}_analytic": floors[name].analytic_macro_f1
                        for name in floors
                    },
                }
            )

    _write_report(config, summary, present)

    print(f"wrote {written} baseline rows, skipped {skipped} already present")
    print(f"results: {results_path}")
    print()
    print(
        f"{'pair':<22} {'K':>2} {'seed':>4} {'uniform':>9} {'(analytic)':>11} "
        f"{'majority':>9} {'stratified':>11}"
    )
    for entry in summary:
        print(
            f"{entry['source'] + ' -> ' + entry['target']:<22} {entry['n_classes']:>2} "
            f"{entry['seed']:>4} {entry['uniform_random']:9.4f} "
            f"{entry['uniform_random_analytic']:11.4f} {entry['majority']:9.4f} "
            f"{entry['stratified_random']:11.4f}"
        )
    return 0


def _write_report(config, summary, present) -> None:
    lines = ["# Chance floors", ""]
    lines.append(
        "Computed from the **realised** `target_test` label distribution for each "
        "pair and seed, not from an assumed uniform one. Generated by "
        "`ser baselines`; every row also lives in `results/runs.jsonl` with "
        "`classifier=\"baseline_*\"`."
    )
    lines.append("")
    lines.append(
        "Floors are **per pair**. After amendment A4, IEMOCAP pairs run 4-class "
        "(chance 0.250) and RAVDESS↔CREMA-D runs 6-class (chance 0.167), so a "
        "single headline chance value would be wrong for half the grid."
    )
    lines.append("")
    lines.append("| pair | K | seed | n_test | uniform | uniform (analytic) | majority | stratified |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for entry in summary:
        lines.append(
            f"| {entry['source']} → {entry['target']} | {entry['n_classes']} | "
            f"{entry['seed']} | {entry['n_target_test']} | "
            f"{entry['uniform_random']:.4f} | {entry['uniform_random_analytic']:.4f} | "
            f"{entry['majority']:.4f} | {entry['stratified_random']:.4f} |"
        )
    lines.append("")
    lines.append(
        "`uniform` is the mean over "
        f"{config.baselines.n_random_draws} draws; `uniform (analytic)` is the "
        "closed form. They differ slightly because the closed form is a ratio of "
        "expectations rather than the expectation of a ratio -- the empirical "
        "value with its CI is the ground truth."
    )
    lines.append("")

    out = config.resolve(config.paths.reports_dir) / "baselines.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
