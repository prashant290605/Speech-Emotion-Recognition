"""Config loading is strict, and the shipped default is internally consistent."""

from __future__ import annotations

import copy

import pytest
import yaml

from ser.config import Config, ConfigError, load_config, repo_root

DEFAULT_PATH = repo_root() / "configs" / "default.yaml"


@pytest.fixture()
def raw():
    with open(DEFAULT_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write(tmp_path, data) -> str:
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return str(path)


# -- the shipped default ---------------------------------------------------
def test_default_config_loads():
    config = load_config()
    assert isinstance(config, Config)
    assert config.project.name == "cross-corpus-ser"


def test_default_config_matches_the_brief():
    """Values the phase plan specifies explicitly must not drift."""
    config = load_config()

    assert config.labels.spaces["six"] == ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    # 'four', not the plan's original 'five': IEMOCAP fear (~40 utterances) is
    # cut, so IEMOCAP pairs run as canonical 4-class.
    assert config.labels.spaces["four"] == ["angry", "happy", "neutral", "sad"]
    assert config.labels.space_for_iemocap_pairs == "four"
    assert config.labels.space_for_other_pairs == "six"

    assert len(config.splits.seeds) >= 5
    assert config.splits.iemocap_split_unit == "session"

    assert config.features.n_layers == 13
    assert config.features.hidden_dim == 768
    assert config.features.n_segments == 8
    assert config.features.sample_rate == 16000

    assert config.blending.alpha_grid == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert config.alignment.mmd_bandwidth_multipliers == [0.25, 0.5, 1.0, 2.0, 4.0]
    # Regularisation is a searched axis on both rungs that need it.
    assert len(config.alignment.coral_shrinkage) >= 2
    assert config.alignment.coral_ledoit_wolf is True
    assert len(config.alignment.mmd_lambda_grid) >= 2
    assert all(isinstance(v, float) for v in config.alignment.mmd_lambda_grid)

    assert config.classifiers.search_budget == 20
    assert "last" in config.classifiers.layer_agg_options
    assert "weighted" in config.classifiers.layer_agg_options

    assert config.stats.bootstrap_resamples == 2000
    assert config.baselines.n_random_draws == 1000
    assert config.shift.conditional_mmd_min_support >= 2


def test_config_hash_is_stable_and_content_addressed(raw, tmp_path):
    baseline = load_config()

    # Reformatting the file must not change the hash...
    reordered = {k: raw[k] for k in reversed(list(raw))}
    assert load_config(_write(tmp_path, reordered)).config_hash == baseline.config_hash

    # ...but changing a value must.
    changed = copy.deepcopy(raw)
    changed["project"]["seed"] = 43
    assert load_config(_write(tmp_path, changed)).config_hash != baseline.config_hash


# -- strictness ------------------------------------------------------------
def test_unknown_key_rejected(raw, tmp_path):
    raw["features"]["nlayers"] = 13  # typo for n_layers
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, raw))


def test_unknown_section_rejected(raw, tmp_path):
    raw["extras"] = {"anything": 1}
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(_write(tmp_path, raw))


def test_missing_key_rejected(raw, tmp_path):
    del raw["splits"]["source_train_ratio"]
    with pytest.raises(ConfigError, match="missing key"):
        load_config(_write(tmp_path, raw))


def test_missing_section_rejected(raw, tmp_path):
    del raw["stats"]
    with pytest.raises(ConfigError, match="missing section"):
        load_config(_write(tmp_path, raw))


def test_missing_file_rejected():
    with pytest.raises(ConfigError, match="not found"):
        load_config("configs/does-not-exist.yaml")


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("splits", "source_train_ratio", 1.0, "strictly between"),
        ("splits", "iemocap_split_unit", "utterance", "session"),
        ("splits", "seeds", [0, 1], "at least 5 seeds"),
        ("splits", "seeds", [0, 0, 1, 2, 3], "duplicates"),
        ("alignment", "coral_shrinkage", [0.0], "must be positive"),
        ("alignment", "coral_shrinkage", [], "at least one epsilon"),
        ("alignment", "methods", ["none", "gfk"], "unknown method"),
        ("alignment", "mmd_lambda_grid", [0.1], "searched axis"),
        ("alignment", "mmd_lambda_grid", [-1.0, 1.0], "non-negative"),
        ("alignment", "mmd_batch_size", 2, "at least 4"),
        ("blending", "alpha_grid", [0.0, 1.5], r"\[0, 1\]"),
        ("blending", "n_groups", 1, "at least 2"),
        ("classifiers", "families", ["logreg", "randomforest"], "unknown family"),
        ("classifiers", "search_budget", 0, "at least 1"),
        ("classifiers", "layer_candidates", [4, 99], "outside"),
        ("features", "storage_dtype", "float64", "float16 or float32"),
        ("grid", "corpora", ["ravdess", "emodb"], "unknown corpus"),
        ("shift", "conditional_mmd_min_support", 1, "at least 2"),
        ("stats", "ci_level", 1.0, "strictly between"),
        ("stats", "correction", "bonferroni", "holm-bonferroni"),
        ("labels", "iemocap_subsets", "improv", "scripted"),
    ],
)
def test_out_of_range_values_rejected(raw, tmp_path, section, key, value, message):
    raw[section][key] = value
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, raw))


def test_label_space_must_be_sorted(raw, tmp_path):
    raw["labels"]["spaces"]["six"] = ["sad", "angry", "disgust", "fear", "happy", "neutral"]
    with pytest.raises(ConfigError, match="must be sorted"):
        load_config(_write(tmp_path, raw))


def test_transformer_requires_the_segment_cache(raw, tmp_path):
    """Decision A is enforced, not assumed: a Transformer over a pooled vector
    is not a sequence model, so that combination must not be configurable."""
    raw["features"]["segment_pooling_enabled"] = False
    with pytest.raises(ConfigError, match="Decision A"):
        load_config(_write(tmp_path, raw))


def test_dropping_the_transformer_allows_disabling_segments(raw, tmp_path):
    raw["features"]["segment_pooling_enabled"] = False
    raw["classifiers"]["families"] = ["logreg", "svm_linear", "svm_rbf", "mlp"]
    config = load_config(_write(tmp_path, raw))
    assert "transformer" not in config.classifiers.families


# -- the alignment ladder --------------------------------------------------
def test_ladder_is_ordered_by_moments_matched():
    config = load_config()
    assert config.alignment.ladder_order() == [
        "none",
        "zscore",
        "mean_shift",
        "coral",
        "mkmmd_diag",
        "mkmmd_full",
    ]


def test_mean_shift_is_a_distinct_condition_not_an_mmd_alias(raw, tmp_path):
    """The original's "mmd" was a mean shift. Carrying that name forward as an
    alias is how the misstatement survives into v2, so both must be nameable
    and neither may stand in for the other."""
    config = load_config()
    assert "mean_shift" in config.alignment.methods
    assert "mkmmd_full" in config.alignment.methods

    raw["alignment"]["methods"] = ["none", "mmd_mean_shift"]
    with pytest.raises(ConfigError, match="unknown method"):
        load_config(_write(tmp_path, raw))


def test_ladder_order_ignores_config_ordering(raw, tmp_path):
    raw["alignment"]["methods"] = ["mkmmd_full", "none", "coral"]
    config = load_config(_write(tmp_path, raw))
    assert config.alignment.ladder_order() == ["none", "coral", "mkmmd_full"]


# -- label decisions -------------------------------------------------------
def test_all_label_decisions_are_made():
    """Decided 2026-08-10. Phase 2 must not have to guess."""
    config = load_config()
    assert config.undecided() == []

    assert (
        config.require_decision("iemocap_label_source")
        == "majority_vote_discard_disagreement"
    )
    assert config.require_decision("iemocap_excited_to_happy") is True
    assert config.require_decision("iemocap_frustrated") == "drop"
    assert config.require_decision("iemocap_subsets") == "both"
    assert config.require_decision("iemocap_record_subset") is True
    assert config.require_decision("ravdess_calm_to_neutral") is True


def test_frustration_is_not_merged_into_anger():
    """Merging would make 'angry' ~30% of retained IEMOCAP, manufacturing the
    label-prior skew the paper's thesis claims to discover."""
    assert load_config().labels.iemocap_frustrated != "merge_angry"


def test_require_decision_halts_while_undecided(raw, tmp_path):
    raw["labels"]["iemocap_subsets"] = None
    config = load_config(_write(tmp_path, raw))
    assert config.undecided() == ["iemocap_subsets"]
    with pytest.raises(ConfigError, match="undecided"):
        config.require_decision("iemocap_subsets")


def test_invalid_frustration_decision_rejected(raw, tmp_path):
    raw["labels"]["iemocap_frustrated"] = "merge_frustrated"
    with pytest.raises(ConfigError, match="iemocap_frustrated"):
        load_config(_write(tmp_path, raw))


def test_annotation_rule_is_explicit_and_inside_the_label_map_hash(raw, tmp_path):
    """The label source determines the counts, hence the priors, hence the whole
    shift analysis. It must not be a silent property of the parsing code."""
    baseline = load_config()

    changed = copy.deepcopy(raw)
    changed["labels"]["iemocap_label_source"] = "any_annotator"
    assert load_config(_write(tmp_path, changed)).label_map_hash != baseline.label_map_hash


def test_invalid_label_source_rejected(raw, tmp_path):
    raw["labels"]["iemocap_label_source"] = "majority"
    with pytest.raises(ConfigError, match="iemocap_label_source"):
        load_config(_write(tmp_path, raw))


def test_subset_probe_requires_the_subset_to_be_recorded(raw, tmp_path):
    """The improvised/scripted pair cannot be built without the manifest column."""
    raw["labels"]["iemocap_record_subset"] = False
    with pytest.raises(ConfigError, match="requires labels.iemocap_record_subset"):
        load_config(_write(tmp_path, raw))


def test_subset_probe_requires_iemocap_in_the_grid(raw, tmp_path):
    raw["grid"]["corpora"] = ["ravdess", "cremad"]
    with pytest.raises(ConfigError, match="requires 'iemocap'"):
        load_config(_write(tmp_path, raw))


def test_space_reference_must_exist(raw, tmp_path):
    raw["labels"]["space_for_iemocap_pairs"] = "five"
    with pytest.raises(ConfigError, match="not a key of labels.spaces"):
        load_config(_write(tmp_path, raw))


# -- run_id coordinate hashes ----------------------------------------------
def test_label_map_hash_tracks_label_decisions(raw, tmp_path):
    """A changed mapping decision must change the hash, or Phase 7's resume
    merges runs scored against different label spaces."""
    baseline = load_config()

    changed = copy.deepcopy(raw)
    changed["labels"]["iemocap_frustrated"] = "merge_angry"
    assert load_config(_write(tmp_path, changed)).label_map_hash != baseline.label_map_hash

    changed = copy.deepcopy(raw)
    changed["labels"]["label_map_version"] = "v2"
    assert load_config(_write(tmp_path, changed)).label_map_hash != baseline.label_map_hash

    changed = copy.deepcopy(raw)
    changed["labels"]["spaces"]["four"] = ["angry", "fear", "happy", "neutral", "sad"]
    assert load_config(_write(tmp_path, changed)).label_map_hash != baseline.label_map_hash


def test_split_spec_hash_tracks_split_decisions(raw, tmp_path):
    baseline = load_config()

    changed = copy.deepcopy(raw)
    changed["splits"]["target_adapt_ratio"] = 0.4
    assert load_config(_write(tmp_path, changed)).split_spec_hash != baseline.split_spec_hash

    changed = copy.deepcopy(raw)
    changed["splits"]["iemocap_split_unit"] = "speaker"
    assert load_config(_write(tmp_path, changed)).split_spec_hash != baseline.split_spec_hash


def test_coordinate_hashes_are_insensitive_to_unrelated_changes(raw, tmp_path):
    """They pin semantics, not the whole file; config_hash covers the rest."""
    baseline = load_config()
    changed = copy.deepcopy(raw)
    changed["stats"]["bootstrap_resamples"] = 4000
    other = load_config(_write(tmp_path, changed))

    assert other.label_map_hash == baseline.label_map_hash
    assert other.split_spec_hash == baseline.split_spec_hash
    assert other.config_hash != baseline.config_hash


def test_require_decision_rejects_a_name_that_is_not_a_decision():
    with pytest.raises(ConfigError, match="not a labels decision"):
        load_config().require_decision("spaces_typo")


# -- paths -----------------------------------------------------------------
def test_relative_paths_resolve_against_repo_root():
    config = load_config()
    assert config.results_path == repo_root() / "results" / "runs.jsonl"
    assert config.resolve("data/manifest.csv").is_absolute()


# -- run_id coverage: every config key must be classified -------------------
def test_config_hash_is_not_a_run_id_coordinate():
    """It was, and editing an unrelated section orphaned 60 completed rows."""
    from ser.utils.results import RUN_ID_FIELDS

    assert "config_hash" not in RUN_ID_FIELDS
    for facet in Config.FACET_SECTIONS:
        assert facet in RUN_ID_FIELDS


def test_every_config_key_is_covered_by_a_facet_or_declared_inert(raw):
    """Mirror of the inert-run_id-field test. Fails when a new config key
    appears in neither category, because an unclassified key can change a
    result without changing run_id."""
    config = load_config()
    unclassified = []
    for section, values in raw.items():
        if not isinstance(values, dict):
            continue
        for key in values:
            try:
                config.classify_config_key(section, key)
            except ConfigError:
                unclassified.append(f"{section}.{key}")
    assert unclassified == [], f"unclassified config keys: {unclassified}"


def test_an_unknown_section_is_not_silently_treated_as_inert():
    config = load_config()
    with pytest.raises(ConfigError, match="neither covered"):
        config.classify_config_key("brand_new_section", "some_key")


def test_facet_hashes_are_narrower_than_config_hash(raw, tmp_path):
    """The whole point: an unrelated edit must not move a run's coordinates."""
    base = load_config()

    changed = copy.deepcopy(raw)
    changed["stats"]["bootstrap_resamples"] = 4000
    other = load_config(_write(tmp_path, changed))

    assert other.config_hash != base.config_hash
    # stats sits in search_spec_hash, so that facet moves...
    assert other.search_spec_hash != base.search_spec_hash
    # ...but the facets that decide the data and the features do not.
    assert other.label_map_hash == base.label_map_hash
    assert other.split_spec_hash == base.split_spec_hash
    assert other.feature_spec_hash == base.feature_spec_hash


def test_adding_a_seed_does_not_invalidate_existing_seeds(raw, tmp_path):
    """splits.seeds is excluded from split_spec_hash because the per-run seed is
    already a coordinate; hashing the list would make a sixth seed invalidate
    the five that already ran."""
    base = load_config()
    changed = copy.deepcopy(raw)
    changed["splits"]["seeds"] = [0, 1, 2, 3, 4, 5]
    assert load_config(_write(tmp_path, changed)).split_spec_hash == base.split_spec_hash


def test_feature_spec_hash_tracks_extraction_settings(raw, tmp_path):
    base = load_config()
    changed = copy.deepcopy(raw)
    changed["features"]["feature_version"] = "v2"
    assert load_config(_write(tmp_path, changed)).feature_spec_hash != base.feature_spec_hash


def test_search_spec_hash_tracks_the_searched_grids(raw, tmp_path):
    """Changing a grid changes which value could be selected, so it changes what
    the run means even with every other coordinate identical."""
    base = load_config()
    changed = copy.deepcopy(raw)
    changed["alignment"]["mmd_lambda_grid"] = [0.1, 1.0]
    assert load_config(_write(tmp_path, changed)).search_spec_hash != base.search_spec_hash
