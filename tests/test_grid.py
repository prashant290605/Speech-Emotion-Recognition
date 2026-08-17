"""Phase 7: staged enumeration, the freeze guard, and what every run records."""

from __future__ import annotations

import gzip
import json

import pytest

from ser.classifiers import supports_layer_agg
from ser.config import load_config
from ser.run_grid import GridRun, enumerate_stage
from ser.utils.results import RUN_ID_FIELDS, make_run_id


@pytest.fixture(scope="module")
def config():
    return load_config()


CORPORA = ["ravdess", "cremad"]


# -- enumeration -----------------------------------------------------------
def test_stage0_covers_every_ladder_rung_exactly_once(config):
    """The gate exists to exercise the whole pipeline, so no rung may be absent."""
    runs = enumerate_stage(config, 0, corpora=CORPORA)
    assert [r.alignment for r in runs] == config.alignment.ladder_order()
    assert len({(r.source, r.target, r.seed, r.backbone) for r in runs}) == 1


def test_stage0_is_small_enough_to_be_a_gate(config):
    """A gate that takes as long as the grid is not a gate."""
    runs = enumerate_stage(config, 0, corpora=CORPORA)
    assert len(runs) == len(config.alignment.ladder_order())
    assert all(r.classifier == "logreg" for r in runs)
    assert all(r.blending == "none" for r in runs)


def test_stage0_passes_the_regularisation_axis_to_the_rungs_that_need_it(config):
    by_method = {r.alignment: r for r in enumerate_stage(config, 0, corpora=CORPORA)}
    assert by_method["coral"].alignment_eps == min(config.alignment.coral_shrinkage)
    assert by_method["mkmmd_full"].alignment_lam == min(config.alignment.mmd_lambda_grid)
    assert by_method["none"].alignment_eps is None
    assert by_method["none"].alignment_lam is None


def test_stage1_uses_two_seeds_and_spans_every_axis(config):
    runs = enumerate_stage(config, 1, corpora=CORPORA)

    assert {r.seed for r in runs} == set(config.splits.seeds[:2])
    assert {r.alignment for r in runs} == set(config.alignment.ladder_order())
    assert {r.classifier for r in runs} == set(config.classifiers.families)


def test_stage1_omits_combinations_that_cannot_exist(config):
    """`weighted` needs learnable parameters; sklearn families have none."""
    for run in enumerate_stage(config, 1, corpora=CORPORA):
        assert supports_layer_agg(run.classifier, run.layer_agg)


def test_stage1_expands_the_regularisation_grids(config):
    runs = enumerate_stage(config, 1, corpora=CORPORA)
    coral_eps = {r.alignment_eps for r in runs if r.alignment == "coral"}
    lambdas = {r.alignment_lam for r in runs if r.alignment == "mkmmd_full"}

    # Every searched epsilon, plus None for the Ledoit-Wolf variant.
    assert set(config.alignment.coral_shrinkage) <= coral_eps
    assert lambdas == set(config.alignment.mmd_lambda_grid)


def test_stage2_refuses_until_stage1_has_pruned(config):
    """Stage 2 is built from the surviving axes, which only Stage 1 can decide."""
    with pytest.raises(ValueError, match="surviving axes"):
        enumerate_stage(config, 2, corpora=CORPORA)


# -- Stage 2: the reduced factorial ----------------------------------------
@pytest.fixture(scope="module")
def surviving(config):
    """The real Stage 2 design, not a hand-built copy of it.

    Rebuilding the dict here would let the tests pass against a design the
    launcher does not run -- which is exactly what happened when the
    transformer's direction trim was added in one place and not the other.
    """
    from ser.run_grid import stage2_surviving

    return stage2_surviving(config, corpora=CORPORA)


@pytest.fixture(scope="module")
def stage2(config, surviving):
    return enumerate_stage(config, 2, corpora=CORPORA, surviving=surviving)


def test_stage2_keeps_the_protected_axes_at_full_width(config, stage2):
    """The ladder, layer aggregation and backbone carry the paper's claims.

    Screening may not narrow them however flat they look -- a dose-response
    axis with rungs removed is no longer a dose-response axis.
    """
    assert {r.alignment for r in stage2} == set(config.alignment.ladder_order())
    assert {r.backbone for r in stage2} == set(config.features.backbones)
    assert {r.layer_agg for r in stage2} == set(config.classifiers.layer_agg_options)

    # And at full width *within* each family, not just in the union.
    for family in config.classifiers.families:
        runs = [r for r in stage2 if r.classifier == family]
        assert {r.alignment for r in runs} == set(config.alignment.ladder_order())
        assert {r.backbone for r in runs} == set(config.features.backbones)
        assert {r.layer_agg for r in runs} == {
            agg for agg in config.classifiers.layer_agg_options
            if supports_layer_agg(family, agg)
        }


def test_stage2_prunes_only_the_inner_grids(config, stage2, surviving):
    """Pruning is allowed on epsilon and lambda and nowhere else."""
    sklearn_runs = [r for r in stage2 if r.classifier != "transformer"]
    coral_eps = {r.alignment_eps for r in sklearn_runs if r.alignment == "coral"}
    lambdas = {r.alignment_lam for r in sklearn_runs if r.alignment.startswith("mkmmd")}

    assert coral_eps == set(surviving["coral_shrinkage"])
    assert lambdas == set(surviving["mmd_lambda_grid"])
    # The dropped values are gone, and they are the only ones dropped.
    assert 1e-3 not in coral_eps
    assert {10.0, 100.0}.isdisjoint(lambdas)
    assert set(surviving["coral_shrinkage"]) < set(config.alignment.coral_shrinkage) | {None}


def test_stage2_covers_both_transfer_directions(stage2):
    assert {(r.source, r.target) for r in stage2} == {
        ("ravdess", "cremad"),
        ("cremad", "ravdess"),
    }


def test_stage2_transformer_is_an_explicitly_reduced_arm(config, stage2, surviving):
    """Reduced on seeds and inner grid only -- never on a protected axis.

    Its per-cell cost is 15-40x the sklearn families'. Carrying the full inner
    grid for it would spend more wall time on epsilon and lambda than on every
    other axis combined, and Stage 1 showed both to be flat or monotone.
    """
    transformer = [r for r in stage2 if r.classifier == "transformer"]
    others = [r for r in stage2 if r.classifier != "transformer"]

    assert {r.seed for r in transformer} == set(surviving["transformer_seeds"])
    assert len({r.seed for r in transformer}) < len({r.seed for r in others})

    # Exactly one inner-grid setting per rung.
    for method in ("coral", "mkmmd_diag", "mkmmd_full"):
        runs = [r for r in transformer if r.alignment == method]
        assert len({(r.alignment_eps, r.alignment_lam) for r in runs}) == 1

    # But still every rung, every backbone, every aggregation it supports.
    assert {r.alignment for r in transformer} == set(config.alignment.ladder_order())
    assert {r.backbone for r in transformer} == set(config.features.backbones)


def test_stage2_run_ids_are_distinct(config, stage2):
    ids = [make_run_id(r.coords(config)) for r in stage2]
    assert len(ids) == len(set(ids))


def test_stage2_does_not_reuse_a_stage1_run_id(config, surviving):
    """Stage 1 rows must resume, not be recomputed under a different meaning."""
    stage1 = {make_run_id(r.coords(config)) for r in enumerate_stage(config, 1, corpora=CORPORA)}
    stage2 = enumerate_stage(config, 2, corpora=CORPORA, surviving=surviving)
    overlap = {
        make_run_id(r.coords(config))
        for r in stage2
        if r.source == "ravdess" and r.backbone == "hubert" and r.seed in (0, 1)
    } & stage1
    # The overlap is real and intended: those cells are literally the same run.
    # What must not happen is an id shared by cells with different coordinates,
    # which the distinctness test above already rules out.
    assert overlap, "Stage 2 should resume the Stage 1 cells it re-enumerates"


def test_stage2_extends_the_coral_grid_past_its_stage1_argmax(config, stage2, surviving):
    """eps=1e-1 won 18/18 in Stage 1 *and* was the grid maximum. A hyperparameter
    selected at the edge of its range is not evidence the edge is optimal."""
    eps = {r.alignment_eps for r in stage2 if r.alignment == "coral"}
    assert {1.0, 10.0} <= eps, "the grid must extend past the Stage 1 argmax"
    assert max(e for e in eps if e is not None) > 1e-1


def test_transformer_arm_is_trimmed_on_direction_not_on_seeds(config, stage2, surviving):
    """The projection came in over the ceiling and something had to give.

    Cutting to one seed would leave the arm unable to report any spread, which
    is the entire point of calling it a reduced-seed arm; cutting a protected
    axis is not allowed. One direction is what is left.
    """
    transformer = [r for r in stage2 if r.classifier == "transformer"]
    others = [r for r in stage2 if r.classifier != "transformer"]

    assert len({(r.source, r.target) for r in transformer}) == 1
    assert len({(r.source, r.target) for r in others}) == 2
    # Still enough seeds to form an interval, and every protected axis intact.
    assert len({r.seed for r in transformer}) >= 2
    assert {r.alignment for r in transformer} == set(config.alignment.ladder_order())
    assert {r.backbone for r in transformer} == set(config.features.backbones)


def test_blending_arm_varies_alpha_and_runs_only_implemented_modes(config, stage2):
    """The axis was unscreened, so it is either varied or dropped. It is varied."""
    blended = [r for r in stage2 if r.blending != "none"]

    assert blended, "blend_alpha must be varied somewhere or removed from scope"
    assert {r.blending for r in blended} == {"scalar"}, "gaa has no run-path support"
    assert {r.blend_alpha for r in blended} == {0.0, 0.25, 0.5, 0.75}
    # alpha=1.0 is pure aligned, which the main grid already covers as
    # blending="none"; enumerating it twice is how the original counted 972
    # runs when only 756 were distinct.
    assert 1.0 not in {r.blend_alpha for r in blended}
    # It is a screening arm, not a full axis: one backbone, two seeds.
    assert len({r.backbone for r in blended}) == 1
    assert len({r.seed for r in blended}) == 2


def test_the_launcher_passes_partition_the_grid(config, surviving, stage2):
    """The launcher runs sklearn+MLP, then transformer. The two passes must
    tile the grid exactly -- no run in both, none in neither.

    The blending arm was appended regardless of the --families filter, so
    `--families transformer` enumerated 288 sklearn runs and the passes summed
    to 5274 against a grid of 4986.
    """
    sklearn_pass = enumerate_stage(
        config, 2, corpora=CORPORA, surviving=surviving,
        families=["logreg", "svm_linear", "svm_rbf", "mlp"],
    )
    transformer_pass = enumerate_stage(
        config, 2, corpora=CORPORA, surviving=surviving, families=["transformer"],
    )

    assert len(sklearn_pass) + len(transformer_pass) == len(stage2)
    a = {make_run_id(r.coords(config)) for r in sklearn_pass}
    b = {make_run_id(r.coords(config)) for r in transformer_pass}
    assert not a & b, "a run enumerated by both passes would be run twice"
    assert a | b == {make_run_id(r.coords(config)) for r in stage2}
    assert not [r for r in transformer_pass if r.blending != "none"]


def test_every_enumerated_blended_run_can_actually_be_run(config, stage2):
    """Guards the failure mode where a row is labelled with a blend_alpha whose
    features were never blended, which is what would have happened before the
    run path implemented blending at all."""
    from ser.run_grid import _blend_alpha_for

    for run in stage2:
        if run.blending == "none":
            continue
        assert _blend_alpha_for(run) == run.blend_alpha


def test_stage1_blending_axis_was_never_screened(config):
    """blend_alpha cannot be pruned on Stage 1 evidence -- Stage 1 never varied it.

    Guards against a later reader assuming the blending axis was screened
    because every other inner grid was.
    """
    assert {r.blending for r in enumerate_stage(config, 1, corpora=CORPORA)} == {"none"}
    assert len(config.blending.alpha_grid) > 1


def test_every_enumerated_run_has_a_distinct_run_id(config):
    runs = enumerate_stage(config, 1, corpora=CORPORA)
    ids = [make_run_id(r.coords(config)) for r in runs]
    assert len(ids) == len(set(ids))


def test_run_coords_cover_every_run_id_field(config):
    run = enumerate_stage(config, 0, corpora=CORPORA)[0]
    assert set(RUN_ID_FIELDS) <= set(run.coords(config))


def test_layer_spec_matches_the_aggregation(config):
    base = dict(
        source="ravdess", target="cremad", seed=0, backbone="hubert",
        feature_branch="ssl", alignment="none", alignment_eps=None,
        alignment_lam=None, blending="none", blend_alpha=None, n_groups=None,
        classifier="mlp",
    )
    assert GridRun(layer_agg="last", layer_index=None, **base).layer_spec == "last"
    assert GridRun(layer_agg="weighted", layer_index=None, **base).layer_spec == "weighted"
    assert GridRun(layer_agg="layer", layer_index=6, **base).layer_spec == "layer:6"


# -- the freeze guard ------------------------------------------------------
def test_grid_refuses_to_start_on_config_drift(config, tmp_path, monkeypatch):
    """A two-week grid must not run against a moving config."""
    from ser import run_grid as grid_module
    from ser.freeze import ConfigDrift

    def _drifted(*args, **kwargs):
        raise ConfigDrift("working config does not match the frozen tag")

    monkeypatch.setattr(grid_module, "assert_config_frozen", _drifted)
    with pytest.raises(ConfigDrift):
        grid_module.run_grid(config, 0, corpora=CORPORA)


def test_grid_records_the_freeze_tag_on_every_row():
    from ser.utils.results import FIELD_NAMES

    assert "freeze_tag" in FIELD_NAMES


# -- what a run must record ------------------------------------------------
def test_schema_carries_everything_phase_8_and_9_need():
    """Adding any of these after the grid has run means re-running it."""
    from ser.utils.results import FIELD_NAMES

    required = {
        "per_class_f1_json",
        "per_class_precision_json",
        "per_class_recall_json",
        "per_class_support_json",
        "confusion_json",
        "n_collapsed_classes",
        "predictions_path",
        "n_search_trials",
        "selection_source_val_macro_f1",
        "epochs_run",
        "marginal_mmd_normalised",
        "marginal_mmd_reference",
        "mmd_fallback_fired",
        "wall_seconds",
    }
    assert required <= set(FIELD_NAMES)


def test_predictions_round_trip(tmp_path):
    from ser.run_grid import _write_predictions

    results = tmp_path / "runs.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    relative = _write_predictions(
        results, "abc123", ["u1", "u2", "u3"], ["angry", "sad", "angry"]
    )

    path = results.parent / relative
    assert path.exists()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["utterance_ids"] == ["u1", "u2", "u3"]
    assert payload["predicted"] == ["angry", "sad", "angry"]


def test_floor_columns_are_computed_from_the_realised_distribution():
    from ser.run_grid import _floor_columns

    classes = ["angry", "happy", "neutral", "sad"]
    balanced = _floor_columns(classes * 25, classes * 25, classes)
    assert balanced["chance_macro_f1"] == pytest.approx(0.25, abs=1e-6)
    # Majority is the collapse floor and must sit below chance.
    assert balanced["majority_macro_f1"] < balanced["chance_macro_f1"]

    skewed = _floor_columns(["angry"] * 70 + classes[1:] * 10, classes * 25, classes)
    assert skewed["chance_macro_f1"] != pytest.approx(0.25, abs=1e-3)


def test_reference_geometry_is_one_fixed_map_for_every_rung():
    """It must not be re-derived per rung, or it becomes another degree of
    freedom rather than the thing that removes one."""
    import numpy as np

    from ser.mmd import reference_geometry

    rng = np.random.default_rng(0)
    source = rng.standard_normal((200, 12)) @ rng.standard_normal((12, 12))

    geometry = reference_geometry(source, eps=1e-2)

    # Same map applied twice gives the same answer, and it is linear, so it
    # cannot undo an alignment applied before it.
    a = rng.standard_normal((50, 12))
    np.testing.assert_allclose(geometry(a), geometry(a))
    np.testing.assert_allclose(
        geometry(a + 3.0) - geometry(a),
        (np.full_like(a, 3.0)) @ geometry.whitener,
        atol=1e-9,
    )


def test_reference_geometry_whitens_its_own_source():
    import numpy as np

    from ser.mmd import reference_geometry
    from ser.numerics import covariance

    rng = np.random.default_rng(1)
    source = rng.standard_normal((400, 8)) @ rng.standard_normal((8, 8))
    whitened = reference_geometry(source, eps=1e-6)(source)

    # Close to identity covariance: that is what makes it a reference frame.
    cov = covariance(whitened)
    np.testing.assert_allclose(np.diag(cov), np.ones(8), rtol=0.15)


def test_alignment_view_keeps_every_family_in_the_same_feature_space():
    """Each 768-d vector is one observation, NOT (N, everything_else).

    Concatenating the axes would fit the alignment in 6144 dimensions for the
    transformer and 79872 for transformer+weighted -- the latter needs a ~51 GB
    covariance and would OOM. It would also give each family a differently
    conditioned covariance, so the alignment rung would not mean the same thing
    across families.
    """
    import numpy as np

    from ser.run_grid import _flatten, _unflatten

    for shape in [(100, 768), (100, 8, 768), (100, 13, 768), (100, 13, 8, 768)]:
        X = np.zeros(shape)
        view = _flatten(X)
        assert view.shape[-1] == 768, f"{shape} aligned in {view.shape[-1]} dims"
        assert view.ndim == 2
        np.testing.assert_array_equal(_unflatten(view, X), X)


def test_alignment_view_round_trips_values_not_just_shapes():
    import numpy as np

    from ser.run_grid import _flatten, _unflatten

    X = np.arange(2 * 8 * 4, dtype=np.float64).reshape(2, 8, 4)
    assert _flatten(X).shape == (16, 4)
    np.testing.assert_array_equal(_unflatten(_flatten(X), X), X)


def test_mmd_view_gives_one_vector_per_utterance():
    """Discrepancy compares distributions over utterances. Without pooling, the
    transformer's effect size would come from 8x as many points drawn from a
    within-utterance distribution and would not compare to any other family."""
    import numpy as np

    from ser.run_grid import _mmd_view

    for shape in [(100, 768), (100, 8, 768), (100, 13, 768), (100, 13, 8, 768)]:
        assert _mmd_view(np.zeros(shape)).shape == (100, 768)

    X = np.stack([np.full((8, 4), i, dtype=np.float64) for i in range(3)])
    np.testing.assert_allclose(_mmd_view(X), np.array([[0.0] * 4, [1.0] * 4, [2.0] * 4]))
