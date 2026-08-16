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


def test_flatten_round_trips_a_segment_sequence():
    import numpy as np

    from ser.run_grid import _flatten, _unflatten

    X = np.arange(2 * 8 * 4, dtype=np.float64).reshape(2, 8, 4)
    flat = _flatten(X)
    assert flat.shape == (2, 32)
    np.testing.assert_array_equal(_unflatten(flat, X), X)
