"""The staged grid runner.

Three stages, deliberately not a full factorial:

**Stage 0 — smoke gate.** One pair, one backbone, one seed, every alignment
rung. Halts if target macro-F1 does not clear the pair's own chance floor. The
point is to fail in minutes rather than to discover a broken pipeline forty
hours in.

**Stage 1 — screening.** One pair, one backbone, all axes, two seeds. Axes that
show no effect are pruned. **Pruning is decided on ``source_val`` only** — using
target scores to choose what to run is the same leak Phase 2 exists to prevent,
just moved one level up.

**Stage 2 — reduced factorial** over the surviving axes, both directions, all
backbones plus MFCC, five seeds.

Every run records what Phase 8 and Phase 9 will need and cannot recover later:
per-class precision/recall/F1/support, the confusion matrix, the selected
hyperparameters and trial count, source_val and target scores separately, epochs
to early stop, the alignment effect size, wall time — and the **predictions
themselves**, per utterance, so paired significance tests come for free.

Resumable: completed ``run_id``s are read off disk at start and skipped. A
crashed run writes ``status="failed"`` with its traceback and the runner
continues, because a silent skip leaves a hole nobody can see.
"""

from __future__ import annotations

import gzip
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from .alignment import build_alignment
from .blending import BLENDABLE_ALIGNMENTS
from .classifiers import fit_and_select, supports_layer_agg
from .features.load import FeatureLoader
from .freeze import assert_config_frozen, read_freeze_tag
from .leakage import assert_alignment_blind_to_target_test
from .manifest import read_manifest
from .metrics import all_metrics, confusion_matrix, macro_f1
from .mmd import marginal_mmd, median_bandwidth, null_mmd_scale, reference_geometry
from .splits import make_pair_split
from .utils.results import append_row, completed_run_ids, make_run_id, new_row
from .utils.runmeta import capture_runmeta
from .utils.seeding import set_all_seeds

__all__ = [
    "GridRun",
    "enumerate_stage",
    "enumerate_transformer_probe",
    "run_grid",
    "shard_of",
    "REFERENCE_GEOMETRY_EPS",
    "STAGE2_SURVIVING",
    "stage2_surviving",
]


def _stamp() -> str:
    """UTC timestamp for log lines, so a heartbeat is locatable in wall time."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# Shrinkage for the fixed reference geometry. Strong on purpose: the source
# covariance has an effective rank near 57 of 768, so a weak shrinkage would
# amplify hundreds of near-null directions and make the reference frame itself
# noise-dominated.
REFERENCE_GEOMETRY_EPS = 1e-2


# -- what Stage 1 pruned ---------------------------------------------------
# Recorded here rather than in configs/default.yaml because the config is
# FROZEN at grid-freeze-v2 and Stage 1's rows were produced against it. Editing
# it would change `search_spec_hash` and orphan all 372 Stage 1 rows.
#
# Every decision below is scored on source_val ONLY, and cross-checked against
# the 12 transformer probe runs. The rationale is in PROGRESS.md.
#
# NOT pruned, by instruction -- these three carry the paper's claims and are
# kept at full width whatever screening says:
#   * the alignment ladder   (the dose-response axis)
#   * layer aggregation      (carries the layer finding)
#   * backbone               (a stated contribution)
STAGE2_SURVIVING = {
    # Dropped 1e-3: never the source_val argmax in 18/18 conditions, and it is
    # interior to the retained grid -- 1e-4 and 1e-2 bracket it, and its
    # source_val (0.5713) sits between theirs, so it adds no shape.
    # 1e-4 is kept as the endpoint the "unregularised CORAL" reading rests on;
    # Ledoit-Wolf is kept because "does the standard automatic choice find the
    # good regime?" is a real question and its answer here is no.
    "coral_shrinkage": [1e-4, 1e-2, 1e-1, None],
    # Dropped 10.0 and 100.0. source_val is flat in lambda (spread 0.0016 diag /
    # 0.0054 full, i.e. argmax is noise), so the drop is made on mechanism, not
    # on score: at lambda >= 1 the W-to-identity penalty dominates and
    # mkmmd_full reverts to its CORAL warm start in 14/18 runs. Those cells are
    # not a sixth rung, they are CORAL with extra wall time. 1.0 is retained as
    # the anchor that shows where the fallback saturates.
    "mmd_lambda_grid": [0.001, 0.01, 0.1, 1.0],
    # One variant per rung for the transformer arm only -- see PROGRESS.md for
    # why the transformer cannot carry the full inner grid at any seed count.
    "transformer_coral_eps": 1e-1,
    "transformer_mmd_lambda": 0.01,
}


def stage2_surviving(config, *, corpora: Sequence[str]) -> Dict:
    """The full Stage 2 design: pruned inner grids plus the axes Stage 1 fixed.

    Stage 1 held source, backbone and seed count constant so it could afford to
    screen the inner grids. Stage 2 spends what the pruning saved on those three.

    One place, so the launcher and the wall-time projection cannot disagree
    about what is being run.
    """
    source, target = corpora[0], corpora[-1]
    return dict(
        STAGE2_SURVIVING,
        directions=[(source, target), (target, source)],
        backbones=list(config.features.backbones),
        seeds=list(config.splits.seeds),
        transformer_seeds=list(config.splits.seeds[:2]),
    )


@dataclass(frozen=True)
class GridRun:
    """One cell of the grid."""

    source: str
    target: str
    seed: int
    backbone: str
    feature_branch: str
    layer_agg: str
    layer_index: Optional[int]
    alignment: str
    alignment_eps: Optional[float]
    alignment_lam: Optional[float]
    blending: str
    blend_alpha: Optional[float]
    n_groups: Optional[int]
    classifier: str

    @property
    def layer_spec(self) -> str:
        if self.layer_agg == "last":
            return "last"
        if self.layer_agg == "weighted":
            return "weighted"
        return f"layer:{self.layer_index}"

    def coords(self, config) -> Dict:
        return {
            "label_map_hash": config.label_map_hash,
            "split_spec_hash": config.split_spec_hash,
            "feature_spec_hash": config.feature_spec_hash,
            "search_spec_hash": config.search_spec_hash,
            "seed": self.seed,
            "source_corpus": self.source,
            "target_corpus": self.target,
            "backbone": self.backbone,
            "layer_agg": self.layer_agg,
            "layer_index": self.layer_index,
            "feature_branch": self.feature_branch,
            "alignment": self.alignment,
            "alignment_eps": self.alignment_eps,
            "alignment_lambda": self.alignment_lam,
            "blending": self.blending,
            "blend_alpha": self.blend_alpha,
            "n_groups": self.n_groups,
            "classifier": self.classifier,
            "split_id": f"{self.source}-{self.target}-s{self.seed}",
        }


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------
def _alignment_variants(config, method: str) -> List[Dict]:
    """One entry per distinct setting of a rung's regularisation axis."""
    if method == "coral":
        variants = [{"eps": eps, "lam": None} for eps in config.alignment.coral_shrinkage]
        if config.alignment.coral_ledoit_wolf:
            variants.append({"eps": None, "lam": None, "ledoit_wolf": True})
        return variants
    if method.startswith("mkmmd"):
        return [{"eps": None, "lam": lam} for lam in config.alignment.mmd_lambda_grid]
    return [{"eps": None, "lam": None}]


def enumerate_transformer_probe(config, *, corpora: Sequence[str]) -> List[GridRun]:
    """A stratified transformer probe at the extremes of every axis screening prunes.

    Screening runs on sklearn + MLP, which consume a pooled vector. The
    transformer consumes an 8-segment sequence, so a pruning decision taken
    without it might not transfer to the one family whose input differs. This
    probe covers **both ends of the alignment ladder, every layer aggregation,
    and both epsilon extremes** — 12 runs — so that any decision that would flip
    for the transformer shows up before Stage 2 commits to it.

    It is a validity check on the pruning, not a measurement of the transformer.
    """
    source, target = corpora[0], corpora[-1]
    eps_low, eps_high = min(config.alignment.coral_shrinkage), max(
        config.alignment.coral_shrinkage
    )
    conditions = [
        ("none", None, None),
        ("coral", eps_low, None),
        ("coral", eps_high, None),
        ("mkmmd_full", None, min(config.alignment.mmd_lambda_grid)),
    ]
    runs = []
    for method, eps, lam in conditions:
        for agg in config.classifiers.layer_agg_options:
            runs.append(
                GridRun(
                    source=source,
                    target=target,
                    seed=config.splits.seeds[0],
                    backbone=next(iter(config.features.backbones)),
                    feature_branch="ssl",
                    layer_agg=agg,
                    layer_index=(
                        config.classifiers.layer_candidates[1] if agg == "layer" else None
                    ),
                    alignment=method,
                    alignment_eps=eps,
                    alignment_lam=lam,
                    blending="none",
                    blend_alpha=None,
                    n_groups=None,
                    classifier="transformer",
                )
            )
    return runs


def enumerate_stage(
    config,
    stage: int,
    *,
    corpora: Sequence[str],
    surviving: Optional[Dict] = None,
    families: Optional[Sequence[str]] = None,
) -> List[GridRun]:
    """Enumerate the runs for one stage.

    ``families`` restricts the classifier axis. Stage 1 runs without the
    transformer: at measured timings it is 80% of the screening budget, Stage 2
    re-runs it anyway, and every other axis's pruning decision is available
    without it. The transformer's coverage comes from
    :func:`enumerate_transformer_probe` instead.
    """
    surviving = surviving or {}

    if stage == 0:
        # Smoke gate: minimum that still exercises every rung end to end.
        return [
            GridRun(
                source=corpora[0],
                target=corpora[-1],
                seed=config.splits.seeds[0],
                backbone=next(iter(config.features.backbones)),
                feature_branch="ssl",
                layer_agg="last",
                layer_index=None,
                alignment=method,
                alignment_eps=(
                    min(config.alignment.coral_shrinkage) if method == "coral" else None
                ),
                alignment_lam=(
                    min(config.alignment.mmd_lambda_grid)
                    if method.startswith("mkmmd")
                    else None
                ),
                blending="none",
                blend_alpha=None,
                n_groups=None,
                classifier="logreg",
            )
            for method in config.alignment.ladder_order()
        ]

    if stage == 1:
        source, target = corpora[0], corpora[-1]
        backbone = next(iter(config.features.backbones))
        seeds = config.splits.seeds[:2]
        runs: List[GridRun] = []
        selected = list(families) if families else list(config.classifiers.families)
        for seed, method, agg, family in product(
            seeds,
            config.alignment.ladder_order(),
            config.classifiers.layer_agg_options,
            selected,
        ):
            if not supports_layer_agg(family, agg):
                continue
            for variant in _alignment_variants(config, method):
                runs.append(
                    GridRun(
                        source=source,
                        target=target,
                        seed=seed,
                        backbone=backbone,
                        feature_branch="ssl",
                        layer_agg=agg,
                        layer_index=(
                            config.classifiers.layer_candidates[1]
                            if agg == "layer"
                            else None
                        ),
                        alignment=method,
                        alignment_eps=variant.get("eps"),
                        alignment_lam=variant.get("lam"),
                        blending="none",
                        blend_alpha=None,
                        n_groups=None,
                        classifier=family,
                    )
                )
        return runs

    if stage == 2:
        if not surviving:
            raise ValueError(
                "stage 2 is not enumerable without the surviving axes recorded "
                "in PROGRESS.md after Stage 1. Pass surviving=STAGE2_SURVIVING."
            )
        return _enumerate_stage2(config, surviving, corpora=corpora, families=families)

    raise ValueError(
        f"stage {stage} is not enumerable yet. Stage 2 must be built from the "
        "surviving axes recorded in PROGRESS.md after Stage 1."
    )


def _stage2_variants(config, surviving: Dict, method: str, *, transformer: bool):
    """Inner-grid settings for one rung after Stage 1's pruning.

    The transformer takes one setting per rung. Its per-cell cost is 15-40x the
    sklearn families', so carrying the full inner grid for it would spend more
    wall time on epsilon and lambda -- both of which Stage 1 showed to be either
    monotone-to-the-boundary or flat -- than on every other axis combined.
    """
    if method == "coral":
        if transformer:
            return [{"eps": surviving["transformer_coral_eps"], "lam": None}]
        return [{"eps": eps, "lam": None} for eps in surviving["coral_shrinkage"]]
    if method.startswith("mkmmd"):
        if transformer:
            return [{"eps": None, "lam": surviving["transformer_mmd_lambda"]}]
        return [{"eps": None, "lam": lam} for lam in surviving["mmd_lambda_grid"]]
    return [{"eps": None, "lam": None}]


def _enumerate_stage2(
    config,
    surviving: Dict,
    *,
    corpora: Sequence[str],
    families: Optional[Sequence[str]] = None,
) -> List[GridRun]:
    """The reduced factorial: seeds and backbones over what Stage 1 left standing.

    Stage 1 fixed source, backbone and seed count to screen the inner grids.
    Stage 2 spends what that saved on the axes the paper actually reports:
    both transfer directions, all three backbones, and five seeds.

    The transformer runs as an explicitly reduced arm -- 2 seeds, one inner-grid
    setting per rung -- and must be reported with wider intervals than the rest.
    Its numbers are not pooled with the five-seed families.
    """
    directions = surviving.get("directions") or [(corpora[0], corpora[-1])]
    backbones = surviving.get("backbones") or list(config.features.backbones)
    seeds = surviving.get("seeds") or config.splits.seeds
    transformer_seeds = surviving.get("transformer_seeds") or config.splits.seeds[:2]
    selected = list(families) if families else list(config.classifiers.families)

    runs: List[GridRun] = []
    for family in selected:
        is_transformer = family == "transformer"
        family_seeds = transformer_seeds if is_transformer else seeds
        for (source, target), backbone, seed, method, agg in product(
            directions,
            backbones,
            family_seeds,
            config.alignment.ladder_order(),
            config.classifiers.layer_agg_options,
        ):
            if not supports_layer_agg(family, agg):
                continue
            for variant in _stage2_variants(
                config, surviving, method, transformer=is_transformer
            ):
                runs.append(
                    GridRun(
                        source=source,
                        target=target,
                        seed=seed,
                        backbone=backbone,
                        feature_branch="ssl",
                        layer_agg=agg,
                        layer_index=(
                            config.classifiers.layer_candidates[1]
                            if agg == "layer"
                            else None
                        ),
                        alignment=method,
                        alignment_eps=variant.get("eps"),
                        alignment_lam=variant.get("lam"),
                        blending="none",
                        blend_alpha=None,
                        n_groups=None,
                        classifier=family,
                    )
                )
    return runs


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
class _Context:
    """Caches loaders and splits so the grid does not re-read them per run."""

    def __init__(self, config):
        self.config = config
        self.rows = read_manifest(config.resolve(config.paths.manifest))
        self.by_id = {row.utterance_id: row for row in self.rows}
        self._loaders: Dict[tuple, FeatureLoader] = {}
        self._splits: Dict[tuple, object] = {}

    def loader(self, corpus: str, backbone: str) -> FeatureLoader:
        key = (corpus, backbone)
        if key not in self._loaders:
            self._loaders[key] = FeatureLoader(self.config, corpus, backbone, self.rows)
        return self._loaders[key]

    def split(self, source: str, target: str, seed: int):
        key = (source, target, seed)
        if key not in self._splits:
            self._splits[key] = make_pair_split(self.rows, self.config, source, target, seed)
        return self._splits[key]

    def labels(self, pair, role: str) -> List[str]:
        space = pair.label_space
        return [
            (self.by_id[uid].label_six if space == "six" else self.by_id[uid].label_four)
            for uid in pair.splits()[role].utterance_ids
        ]


def _write_predictions(results_path: Path, run_id: str, utterance_ids, predicted) -> str:
    """Store per-utterance predictions beside the results file.

    ~15 KB gzipped per run, and it makes per-class transfer, confusion
    structure, McNemar and bootstrap paired tests available in Phase 8/9 with no
    reruns at all.
    """
    directory = results_path.parent / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json.gz"
    payload = {
        "run_id": run_id,
        "utterance_ids": list(utterance_ids),
        "predicted": [str(p) for p in predicted],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return str(path.relative_to(results_path.parent))


def _floor_columns(y_test, y_train, class_names) -> Dict[str, float]:
    """The three chance floors for this run's realised target distribution."""
    from collections import Counter

    from .baselines import (
        analytic_majority_macro_f1,
        analytic_stratified_macro_f1,
        analytic_uniform_macro_f1,
    )

    target_counts = Counter(y_test)
    total = sum(target_counts.get(c, 0) for c in class_names)
    target_prior = [target_counts.get(c, 0) / total for c in class_names]

    source_counts = Counter(y_train)
    source_total = sum(source_counts.get(c, 0) for c in class_names)
    source_prior = [source_counts.get(c, 0) / source_total for c in class_names]
    majority_index = int(np.argmax(source_prior))

    return {
        "chance_macro_f1": analytic_uniform_macro_f1(target_prior),
        "majority_macro_f1": analytic_majority_macro_f1(target_prior, majority_index),
        "prior_matched_macro_f1": analytic_stratified_macro_f1(
            target_prior, source_prior
        ),
    }


def _per_class_stats(y_true, y_pred, class_names) -> Dict[str, Dict[str, float]]:
    matrix = confusion_matrix(y_true, y_pred, class_names)
    tp = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    actual = matrix.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
    return {
        "precision": {n: float(precision[i]) for i, n in enumerate(class_names)},
        "recall": {n: float(recall[i]) for i, n in enumerate(class_names)},
        "support": {n: int(actual[i]) for i, n in enumerate(class_names)},
    }


def execute_run(run: GridRun, context: _Context, freeze_tag: str) -> Dict:
    """Run one cell. Returns a schema-valid row, ok or failed."""
    config = context.config
    started = time.perf_counter()
    coords = run.coords(config)
    run_id = make_run_id(coords)
    meta = capture_runmeta(config.config_hash)

    def _row(**extra) -> Dict:
        return new_row(
            **{**coords, **meta.as_row_fields()},
            run_id=run_id,
            freeze_tag=freeze_tag or None,
            wall_seconds=round(time.perf_counter() - started, 3),
            **extra,
        )

    try:
        set_all_seeds(run.seed)
        pair = context.split(run.source, run.target, run.seed)
        classes = list(config.labels.spaces[pair.label_space])
        needs_segments = run.classifier == "transformer"

        source_loader = context.loader(run.source, run.backbone)
        target_loader = context.loader(run.target, run.backbone)

        X_train = source_loader.load(
            pair.source_train.utterance_ids,
            layer_spec=run.layer_spec,
            segments=needs_segments,
        )
        X_val = source_loader.load(
            pair.source_val.utterance_ids,
            layer_spec=run.layer_spec,
            segments=needs_segments,
        )
        X_adapt = target_loader.load(
            pair.target_adapt.utterance_ids,
            layer_spec=run.layer_spec,
            segments=needs_segments,
        )
        X_test = target_loader.load(
            pair.target_test.utterance_ids,
            layer_spec=run.layer_spec,
            segments=needs_segments,
        )

        y_train = context.labels(pair, "source_train")
        y_val = context.labels(pair, "source_val")
        y_test = context.labels(pair, "target_test")

        # -- alignment. Fitted on source_train and target_adapt only. --------
        effect_size = raw_mmd = reference_effect = None
        condition_number = effective_rank = None
        fallback_fired = None
        flat_train = _flatten(X_train)
        flat_adapt = _flatten(X_adapt)
        aligned_adapt = flat_adapt

        # ITEM B: one ZCA map from the UNALIGNED source_train, derived before any
        # rung touches the features, so it is identical for every rung and
        # cannot undo any rung's alignment.
        geometry = reference_geometry(_mmd_view(X_train), eps=REFERENCE_GEOMETRY_EPS)

        if run.alignment != "none":
            alignment = build_alignment(
                run.alignment,
                config,
                eps=run.alignment_eps,
                ledoit_wolf=(run.alignment == "coral" and run.alignment_eps is None),
                lam=run.alignment_lam,
                seed=run.seed,
            )
            alignment.fit(
                flat_train,
                flat_adapt,
                pair.target_adapt.utterance_ids,
                pair.source_train.utterance_ids,
            )
            # The Phase 2 contract, on the real fitted object, every run.
            assert_alignment_blind_to_target_test(alignment, pair)

            X_train = _unflatten(alignment.transform(flat_train, domain="source"), X_train)
            X_val = _unflatten(
                alignment.transform(_flatten(X_val), domain="source"), X_val
            )
            X_test = _unflatten(
                alignment.transform(_flatten(X_test), domain="target"), X_test
            )
            aligned_adapt = alignment.transform(flat_adapt, domain="target")

            fields = alignment.row_fields()
            condition_number = fields["cov_condition_number"]
            effective_rank = fields["cov_effective_rank"]
            fallback_fired = alignment.diagnostics.get("reverted_to_warm_start")

        # Effect size for EVERY rung including `none` -- the unaligned value is
        # the reference the whole ladder is read against, so leaving it null
        # would mean the covariate-shift column had a hole exactly where the
        # comparison starts.
        aligned_train = _mmd_view(X_train)
        mmd_adapt = _mmd_view(_unflatten(aligned_adapt, X_adapt))
        bandwidth = median_bandwidth(aligned_train, mmd_adapt, seed=run.seed)
        raw_mmd = marginal_mmd(
            aligned_train, mmd_adapt, config, bandwidth=bandwidth, seed=run.seed
        )
        null = null_mmd_scale(
            aligned_train, config, bandwidth=bandwidth, n_repeats=5, seed=run.seed
        )["scale"]
        effect_size = raw_mmd / null if null > 0 else None

        # Same measurement in the fixed reference geometry, so between-rung
        # comparison does not depend on which frame each rung happened to leave
        # the features in.
        ref_source = geometry(aligned_train)
        ref_target = geometry(mmd_adapt)
        ref_bandwidth = median_bandwidth(ref_source, ref_target, seed=run.seed)
        ref_null = null_mmd_scale(
            ref_source, config, bandwidth=ref_bandwidth, n_repeats=5, seed=run.seed
        )["scale"]
        reference_effect = (
            marginal_mmd(
                ref_source, ref_target, config, bandwidth=ref_bandwidth, seed=run.seed
            )
            / ref_null
            if ref_null > 0
            else None
        )

        # -- classifier. Selection on source_val only. -----------------------
        selection = fit_and_select(
            run.classifier,
            X_train,
            y_train,
            X_val,
            y_val,
            classes,
            config,
            layer_agg=run.layer_agg,
            seed=run.seed,
        )

        # Target is touched exactly once, here, after selection is complete.
        predicted = [str(p) for p in selection.predict(X_test)]
        metrics = all_metrics(y_test, predicted, classes)
        per_class = _per_class_stats(y_test, predicted, classes)

        predictions_path = _write_predictions(
            config.results_path, run_id, pair.target_test.utterance_ids, predicted
        )

        return _row(
            n_classes=len(classes),
            class_names=classes,
            hyperparams_json=json.dumps(
                {
                    **selection.as_hyperparams(),
                    "alignment_eps": run.alignment_eps,
                    "alignment_lambda": run.alignment_lam,
                },
                sort_keys=True,
                default=str,
            ),
            n_search_trials=selection.n_trials,
            n_train=len(y_train),
            n_val=len(y_val),
            n_target_adapt=len(pair.target_adapt.utterance_ids),
            n_target_test=len(y_test),
            macro_f1=metrics["macro_f1"],
            accuracy=metrics["accuracy"],
            uar=metrics["uar"],
            per_class_f1_json=json.dumps(metrics["per_class_f1"], sort_keys=True),
            per_class_precision_json=json.dumps(per_class["precision"], sort_keys=True),
            per_class_recall_json=json.dumps(per_class["recall"], sort_keys=True),
            per_class_support_json=json.dumps(per_class["support"], sort_keys=True),
            confusion_json=json.dumps(metrics["confusion"]),
            n_collapsed_classes=metrics["n_collapsed_classes"],
            epochs_run=selection.epochs_run,
            predictions_path=predictions_path,
            # Floors on every row, so no metric is ever reported without one.
            # Analytic rather than sampled: the closed forms are exact functions
            # of the realised priors and cost nothing per run.
            **_floor_columns(y_test, y_train, classes),
            selection_source_val_macro_f1=selection.best_source_val_macro_f1,
            cov_condition_number=condition_number,
            cov_effective_rank=effective_rank,
            marginal_mmd_raw=raw_mmd,
            marginal_mmd_normalised=effect_size,
            marginal_mmd_reference=reference_effect,
            mmd_fallback_fired=fallback_fired,
            status="ok",
            error=None,
        )

    except Exception:  # noqa: BLE001 - a crash is recorded, never skipped
        return _row(
            n_classes=0,
            class_names=[],
            hyperparams_json="{}",
            n_search_trials=None,
            n_train=0,
            n_val=0,
            n_target_adapt=0,
            n_target_test=0,
            macro_f1=None,
            accuracy=None,
            uar=None,
            per_class_f1_json=None,
            per_class_precision_json=None,
            per_class_recall_json=None,
            per_class_support_json=None,
            confusion_json=None,
            n_collapsed_classes=None,
            epochs_run=None,
            predictions_path=None,
            chance_macro_f1=None,
            majority_macro_f1=None,
            prior_matched_macro_f1=None,
            selection_source_val_macro_f1=None,
            cov_condition_number=None,
            cov_effective_rank=None,
            marginal_mmd_raw=None,
            marginal_mmd_normalised=None,
            marginal_mmd_reference=None,
            mmd_fallback_fired=None,
            status="failed",
            error=traceback.format_exc()[:4000],
        )


def shard_of(run_id: str, n_shards: int) -> int:
    """Which shard a run belongs to. Deterministic from the id alone.

    Hashing the id rather than slicing the list means a shard's membership does
    not change if the enumeration order changes, so a half-finished shard stays
    resumable across code edits that reorder runs.
    """
    return int(run_id[:8], 16) % n_shards


def run_grid(
    config,
    stage: int,
    *,
    corpora: Sequence[str],
    dry_run: bool = False,
    require_freeze: bool = True,
    families: Optional[Sequence[str]] = None,
    probe: bool = False,
    shard: Optional[int] = None,
    n_shards: int = 1,
    results_path: Optional[Path] = None,
    heartbeat_every: int = 5,
    surviving: Optional[Dict] = None,
) -> int:
    """Execute one stage, resumably, optionally as one shard of several."""
    freeze_tag = assert_config_frozen(config, require=require_freeze)
    label = f"shard {shard}/{n_shards}" if shard is not None else "single worker"
    print(f"[{_stamp()}] config frozen at tag: {freeze_tag or '(not required)'} | {label}",
          flush=True)

    context = _Context(config)
    present = [c for c in corpora if any(r.corpus == c for r in context.rows)]
    if stage == 2 and surviving is None:
        surviving = stage2_surviving(config, corpora=present)
    runs = (
        enumerate_transformer_probe(config, corpora=present)
        if probe
        else enumerate_stage(
            config, stage, corpora=present, families=families, surviving=surviving
        )
    )

    results_path = Path(results_path) if results_path else config.results_path

    # Resume reads the shard's OWN file plus the merged one, so a shard restarted
    # after a merge does not redo work that is already committed.
    already = completed_run_ids(results_path) | completed_run_ids(config.results_path)

    if shard is not None:
        runs = [r for r in runs if shard_of(make_run_id(r.coords(config)), n_shards) == shard]
    todo = [r for r in runs if make_run_id(r.coords(config)) not in already]

    print(f"stage {stage}: {len(runs)} runs enumerated, {len(todo)} to execute, "
          f"{len(runs) - len(todo)} already complete")
    if dry_run:
        for run in runs:
            print(f"  {run.source}->{run.target} s{run.seed} {run.backbone} "
                  f"{run.layer_agg} {run.alignment} {run.classifier}")
        return 0
    print()

    header = (
        f"{'alignment':<14} {'clf':<12} {'agg':<9} {'source_val':>11} "
        f"{'target':>8} {'chance':>7} {'effect':>8} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))

    completed: List[Dict] = []
    wall_start = time.perf_counter()
    for index, run in enumerate(todo, start=1):
        row = execute_run(run, context, freeze_tag)
        append_row(results_path, row)
        completed.append(row)

        if index % heartbeat_every == 0 or index == len(todo):
            elapsed = time.perf_counter() - wall_start
            rate = elapsed / index
            remaining = (len(todo) - index) * rate
            n_failed = sum(1 for r in completed if r["status"] == "failed")
            print(
                f"[{_stamp()}] HEARTBEAT {index}/{len(todo)} "
                f"({index/len(todo):5.1%}) elapsed {elapsed/60:6.1f} min "
                f"eta {remaining/60:6.1f} min | {n_failed} failed",
                flush=True,
            )

        chance = _chance_for(context, run, config)
        if row["status"] == "ok":
            effect = row["marginal_mmd_normalised"]
            print(
                f"{row['alignment']:<14} {row['classifier']:<12} {row['layer_agg']:<9} "
                f"{row['selection_source_val_macro_f1']:>11.4f} {row['macro_f1']:>8.4f} "
                f"{chance:>7.4f} "
                f"{'n/a' if effect is None else f'{effect:8.1f}'} "
                f"{row['wall_seconds']:>6.0f}",
                flush=True,
            )
        else:
            first_line = row["error"].strip().splitlines()[-1][:80]
            print(f"{row['alignment']:<14} {row['classifier']:<12} FAILED: {first_line}",
                  flush=True)

    elapsed = time.perf_counter() - wall_start
    failed = [r for r in completed if r["status"] == "failed"]
    ok = [r for r in completed if r["status"] == "ok"]

    print()
    print(f"{len(ok)} ok, {len(failed)} failed, {elapsed/60:.1f} min "
          f"({elapsed/max(len(completed),1):.0f} s/run)")

    if stage == 0:
        return _smoke_gate(context, config, ok, failed)
    if stage == 1 and shard is None:
        _stage1_report(config)
    return 1 if failed else 0


def _stage1_report(config) -> None:
    """Screening summary. Every number here is `source_val` or a diagnostic.

    Target scores are deliberately absent: pruning an axis on target
    performance is the Phase 2 leak moved one level up.
    """
    from statistics import fmean

    from .utils.results import read_rows

    rows = [
        r
        for r in read_rows(config.results_path)
        if r["status"] == "ok"
        and not r["classifier"].startswith("baseline")
        and r["selection_source_val_macro_f1"] is not None
    ]
    if not rows:
        return

    lines = ["# Stage 1 screening", ""]
    lines.append(
        "**Every figure here is `source_val` or a diagnostic. Target scores are "
        "deliberately excluded** — pruning an axis on target performance is the "
        "leak Phase 2 exists to prevent, moved one level up."
    )
    lines.append("")

    def _by(key, label):
        groups: Dict[object, List[float]] = {}
        for row in rows:
            groups.setdefault(row[key], []).append(
                row["selection_source_val_macro_f1"]
            )
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"| {label} | n | mean source_val | min | max |")
        lines.append("|---|---|---|---|---|")
        for value, scores in sorted(groups.items(), key=lambda kv: str(kv[0])):
            lines.append(
                f"| {value} | {len(scores)} | {fmean(scores):.4f} | "
                f"{min(scores):.4f} | {max(scores):.4f} |"
            )
        lines.append("")
        return groups

    _by("alignment", "alignment")
    _by("classifier", "classifier")
    _by("layer_agg", "layer_agg")

    # ITEM D: does CORAL's source_val cost recover as shrinkage increases?
    coral = [r for r in rows if r["alignment"] == "coral"]
    if coral:
        lines.append("## CORAL: source_val against shrinkage epsilon")
        lines.append("")
        lines.append(
            "CORAL cost 0.166 of `source_val` at eps=1e-4 in Stage 0. At an "
            "effective rank near 57 of 768, weak shrinkage amplifies hundreds of "
            "near-null directions, so that cost may be a property of the "
            "regularisation rather than of CORAL. **If `source_val` recovers at "
            "larger eps while target stays flat, the paper must say so.**"
        )
        lines.append("")
        lines.append("| eps | n | mean source_val | mean effect size |")
        lines.append("|---|---|---|---|")
        groups: Dict[object, List[Dict]] = {}
        for row in coral:
            groups.setdefault(row["alignment_eps"], []).append(row)
        for eps, group in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
            effects = [
                r["marginal_mmd_normalised"]
                for r in group
                if r["marginal_mmd_normalised"] is not None
            ]
            label = "ledoit-wolf" if eps is None else f"{eps:g}"
            lines.append(
                f"| {label} | {len(group)} | "
                f"{fmean(r['selection_source_val_macro_f1'] for r in group):.4f} | "
                f"{fmean(effects):.2f} |" if effects else
                f"| {label} | {len(group)} | "
                f"{fmean(r['selection_source_val_macro_f1'] for r in group):.4f} | — |"
            )
        lines.append("")

    # ITEM A: the fallback rate is a measured property of the method.
    mkmmd = [r for r in rows if r["alignment"].startswith("mkmmd")]
    if mkmmd:
        lines.append("## MK-MMD fallback rate")
        lines.append("")
        lines.append(
            "How often the optimiser failed to beat its own warm start. A rung "
            "that reverts most of the time is its warm start wearing a different "
            "label, and any table containing it must state this rate."
        )
        lines.append("")
        lines.append("| rung | n | fallback fired | rate |")
        lines.append("|---|---|---|---|")
        for name in ("mkmmd_diag", "mkmmd_full"):
            group = [r for r in mkmmd if r["alignment"] == name]
            if not group:
                continue
            fired = sum(1 for r in group if r["mmd_fallback_fired"])
            lines.append(
                f"| {name} | {len(group)} | {fired} | {fired/len(group):.1%} |"
            )
        lines.append("")

    lines.append("## Pruning decisions")
    lines.append("")
    lines.append(
        "_To be filled in by hand before Stage 2, with the rationale. An axis is "
        "pruned only on the evidence above._"
    )
    lines.append("")

    out = config.resolve(config.paths.reports_dir) / "stage1_screening.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"screening summary written to {out}")


def _chance_for(context, run: GridRun, config) -> float:
    """Analytic uniform-random macro-F1 on this pair's realised target labels."""
    from .baselines import analytic_uniform_macro_f1
    from collections import Counter

    pair = context.split(run.source, run.target, run.seed)
    classes = config.labels.spaces[pair.label_space]
    labels = context.labels(pair, "target_test")
    counts = Counter(labels)
    total = sum(counts.get(c, 0) for c in classes)
    prior = [counts.get(c, 0) / total for c in classes]
    return analytic_uniform_macro_f1(prior)


def _smoke_gate(context, config, ok: List[Dict], failed: List[Dict]) -> int:
    """Halt unless every rung clears its pair's own chance floor.

    A marginal result is not a pass. The point of the gate is to stop before
    committing days of compute to a pipeline that is quietly broken.
    """
    print("SMOKE GATE")
    if failed:
        print(f"  FAIL: {len(failed)} run(s) crashed")
        return 1
    if not ok:
        print("  FAIL: no runs completed")
        return 1

    problems = []
    for row in ok:
        chance = row["chance_macro_f1"]
        if chance is None:
            continue
        if row["macro_f1"] <= chance:
            problems.append(
                f"{row['alignment']}/{row['classifier']}: target "
                f"{row['macro_f1']:.4f} does not clear chance {chance:.4f}"
            )

    best = max(ok, key=lambda r: r["macro_f1"])
    worst = min(ok, key=lambda r: r["macro_f1"])
    print(f"  best  {best['alignment']}/{best['classifier']}: {best['macro_f1']:.4f}")
    print(f"  worst {worst['alignment']}/{worst['classifier']}: {worst['macro_f1']:.4f}")

    if problems:
        print("  HALT — do not proceed to Stage 1:")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("  PASS: every rung clears its pair's chance floor")
    return 0


def _flatten(X: np.ndarray) -> np.ndarray:
    """View the input as ``(-1, D)`` -- every 768-d vector as one observation.

    **Not** ``(N, everything_else)``. Concatenating the axes would fit the
    alignment in the flattened space, and that is unworkable as well as wrong:

        sklearn / MLP  `last`      768      covariance    768 x 768
        transformer    `last`     6144      covariance   6144 x 6144  (0.3 GB)
        MLP            `weighted` 9984      covariance   9984 x 9984  (0.8 GB)
        transformer    `weighted` 79872     covariance  ~51 GB -- OOM

    Stage 1 contains `mlp x weighted x coral`, so that would have failed hours
    into the run. It is also statistically wrong: a 6144-dimensional covariance
    from ~988 utterances has rank at most 987, and the conditioning would differ
    per family, making the alignment rung mean something different for the
    transformer than for everything else.

    Collapsing to ``(-1, D)`` instead keeps every family's alignment in the same
    768-dimensional space with the same conditioning, and the map is fitted on
    exactly the distribution of vectors it is applied to.
    """
    return X.reshape(-1, X.shape[-1]) if X.ndim > 2 else X


def _unflatten(flat: np.ndarray, like: np.ndarray) -> np.ndarray:
    return flat.reshape(like.shape) if like.ndim > 2 else flat


def _mmd_view(X: np.ndarray) -> np.ndarray:
    """One 768-d vector per **utterance**, for the discrepancy measurement.

    MMD compares distributions over utterances, so a segment sequence is
    mean-pooled first. Without this the transformer's effect size would be
    computed over 8x as many points drawn from a different (within-utterance)
    distribution, and would not be comparable to any other family's.
    """
    if X.ndim == 2:
        return X
    axes = tuple(range(1, X.ndim - 1))
    return X.mean(axis=axes)
