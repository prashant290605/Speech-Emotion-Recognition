"""The frozen result schema must actually be frozen, and must actually reject.

A schema that silently accepts a malformed row is worse than no schema: it
lets a wrong number reach a table with full provenance attached.
"""

from __future__ import annotations

import json

import pytest

from ser.utils.results import (
    FIELD_NAMES,
    RUN_ID_FIELDS,
    SCHEMA_VERSION,
    SchemaError,
    append_row,
    completed_run_ids,
    make_run_id,
    new_row,
    read_rows,
    validate_row,
)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# config_hash is deliberately NOT here: it is recorded but is not a coordinate.
COORDS = {
    "label_map_hash": "labelmap00000000",
    "split_spec_hash": "splitspec0000000",
    "feature_spec_hash": "featspec00000000",
    "search_spec_hash": "searchspec000000",
    "seed": 0,
    "source_corpus": "ravdess",
    "target_corpus": "cremad",
    "backbone": "hubert",
    "layer_agg": "weighted",
    "layer_index": None,
    "feature_branch": "ssl",
    "alignment": "coral",
    "blending": "scalar",
    "blend_alpha": 0.5,
    "n_groups": None,
    "classifier": "logreg",
    "split_id": "seed0",
}


def _row(**overrides):
    values = dict(
        **COORDS,
        run_id=make_run_id(COORDS),
        config_hash="a" * 64,
        git_sha="0" * 40,
        git_dirty=False,
        timestamp="2026-08-10T12:00:00Z",
        hostname="testhost",
        lib_versions_json=json.dumps({"numpy": "1.26.4"}),
        n_classes=len(CLASS_NAMES),
        class_names=list(CLASS_NAMES),
        hyperparams_json=json.dumps({"C": 1.0}),
        n_train=100,
        n_val=25,
        n_target_adapt=50,
        n_target_test=50,
        macro_f1=0.42,
        accuracy=0.5,
        uar=0.44,
        per_class_f1_json=json.dumps({c: 0.4 for c in CLASS_NAMES}),
        confusion_json=json.dumps([[1] * 6 for _ in range(6)]),
        chance_macro_f1=0.167,
        majority_macro_f1=0.05,
        prior_matched_macro_f1=0.16,
        selection_source_val_macro_f1=0.61,
        cov_condition_number=1.5e4,
        cov_effective_rank=57.2,
        n_search_trials=20,
        marginal_mmd_raw=0.0018,
        marginal_mmd_normalised=2.0,
        wall_seconds=12.5,
        status="ok",
        error=None,
    )
    values.update(overrides)
    return new_row(**values)


# -- schema shape ----------------------------------------------------------
def test_every_field_from_the_brief_is_present():
    """The brief's schema is the contract. Guard against silent drift."""
    required = {
        "run_id", "git_sha", "config_hash", "timestamp", "seed",
        "source_corpus", "target_corpus", "n_classes", "class_names",
        "backbone", "layer_agg", "layer_index", "feature_branch",
        "alignment", "blending", "blend_alpha", "n_groups",
        "classifier", "hyperparams_json",
        "split_id", "n_train", "n_val", "n_target_adapt", "n_target_test",
        "macro_f1", "accuracy", "uar", "per_class_f1_json", "confusion_json",
        "chance_macro_f1", "majority_macro_f1", "prior_matched_macro_f1",
        "selection_source_val_macro_f1",
        "wall_seconds",
    }
    assert required <= set(FIELD_NAMES)


def test_schema_version_is_pinned():
    assert SCHEMA_VERSION == 4
    assert _row()["schema_version"] == 4


def test_the_four_facets_are_run_id_coordinates_and_config_hash_is_not():
    """config_hash was a coordinate and orphaned 60 completed rows when an
    unrelated section changed. The facets replace it; it is still recorded."""
    for facet in (
        "label_map_hash",
        "split_spec_hash",
        "feature_spec_hash",
        "search_spec_hash",
    ):
        assert facet in RUN_ID_FIELDS
    assert "config_hash" not in RUN_ID_FIELDS
    assert "config_hash" in FIELD_NAMES


def test_valid_row_validates():
    validate_row(_row())


# -- rejection -------------------------------------------------------------
def test_unknown_field_rejected():
    with pytest.raises(SchemaError, match="unknown field"):
        _row(oracle_macro_f1=0.9)


def test_missing_field_rejected():
    row = _row()
    del row["macro_f1"]
    with pytest.raises(SchemaError, match="missing field"):
        validate_row(row)


def test_non_nullable_field_rejects_none():
    with pytest.raises(SchemaError, match="not nullable"):
        _row(seed=None)


def test_wrong_type_rejected():
    with pytest.raises(SchemaError, match="expected int"):
        _row(n_train="100")


def test_bool_is_not_accepted_as_int():
    """bool subclasses int in Python; an int column must not accept True."""
    with pytest.raises(SchemaError, match="expected int"):
        _row(n_train=True)


def test_class_names_must_match_n_classes():
    with pytest.raises(SchemaError, match="does not match"):
        _row(n_classes=5)


def test_ok_status_requires_a_metric():
    with pytest.raises(SchemaError, match="requires a macro_f1"):
        _row(macro_f1=None)


def test_failed_status_requires_an_error():
    with pytest.raises(SchemaError, match="requires a non-empty 'error'"):
        _row(status="failed", macro_f1=None, error=None)


def test_failed_run_can_be_recorded_with_null_metrics():
    """Phase 7 needs this: a crash is recorded, never silently skipped."""
    row = _row(
        status="failed",
        error="Traceback (most recent call last): ...",
        macro_f1=None,
        accuracy=None,
        uar=None,
        per_class_f1_json=None,
        confusion_json=None,
    )
    validate_row(row)


def test_invalid_status_rejected():
    with pytest.raises(SchemaError, match="status must be"):
        _row(status="partial")


# -- run_id ----------------------------------------------------------------
def test_run_id_is_deterministic():
    assert make_run_id(COORDS) == make_run_id(dict(COORDS))


def test_run_id_ignores_key_order():
    shuffled = {k: COORDS[k] for k in reversed(list(COORDS))}
    assert make_run_id(shuffled) == make_run_id(COORDS)


def test_run_id_changes_with_every_coordinate():
    """No coordinate may be inert, or two distinct runs collide onto one id."""
    baseline = make_run_id(COORDS)
    alternatives = {
        "label_map_hash": "labelmap11111111",
        "split_spec_hash": "splitspec1111111",
        "feature_spec_hash": "featspec11111111",
        "search_spec_hash": "searchspec111111",
        "seed": 1,
        "source_corpus": "cremad",
        "target_corpus": "iemocap",
        "backbone": "wavlm",
        "layer_agg": "last",
        "layer_index": 6,
        "feature_branch": "fused",
        "alignment": "mmd",
        "blending": "gaa",
        "blend_alpha": 0.75,
        "n_groups": 16,
        "classifier": "svm",
        "split_id": "seed1",
    }
    for name in RUN_ID_FIELDS:
        changed = dict(COORDS, **{name: alternatives[name]})
        assert make_run_id(changed) != baseline, f"run_id ignores '{name}'"


def test_run_id_requires_all_coordinates():
    incomplete = dict(COORDS)
    del incomplete["seed"]
    with pytest.raises(SchemaError, match="missing coordinates"):
        make_run_id(incomplete)


# -- writing ---------------------------------------------------------------
def test_append_round_trip(tmp_path):
    path = tmp_path / "nested" / "runs.jsonl"
    row = _row()

    append_row(path, row)
    append_row(path, _row(seed=1, split_id="seed1"))

    rows = list(read_rows(path, validate=True))
    assert len(rows) == 2
    assert rows[0] == row
    assert completed_run_ids(path) == {rows[0]["run_id"], rows[1]["run_id"]}


def test_append_refuses_invalid_row(tmp_path):
    path = tmp_path / "runs.jsonl"
    bad = _row()
    bad["status"] = "nonsense"
    with pytest.raises(SchemaError):
        append_row(path, bad)
    assert not path.exists()


def test_read_rows_on_missing_file_is_empty(tmp_path):
    assert list(read_rows(tmp_path / "absent.jsonl")) == []


def test_corrupt_line_is_reported_with_location(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_row(path, _row())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(SchemaError, match="not valid JSON"):
        list(read_rows(path))
