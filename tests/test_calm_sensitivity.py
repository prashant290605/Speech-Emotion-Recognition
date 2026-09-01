from __future__ import annotations

import sys
from pathlib import Path

from ser.config import load_config
from ser.manifest import read_manifest
from ser.splits import make_pair_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_calm_sensitivity import (
    enumerate_runs,
    load_spec,
    project_rows,
    sensitivity_config,
    variant_label_space,
)
from run_label_sensitivity_diagnostics import record_id


def test_calm_drop_excludes_only_ravdess_calm_and_changes_label_hash():
    spec = load_spec(ROOT / "configs/calm_sensitivity.yaml")
    base = load_config()
    config = sensitivity_config(base, spec)
    rows = project_rows(read_manifest(config.resolve(config.paths.manifest)), config)
    labels = {row.utterance_id: row.label_six for row in rows}
    assert config.label_map_hash != base.label_map_hash
    assert all(
        labels[row.utterance_id] == ""
        for row in rows if row.corpus == "ravdess" and row.original_label == "calm"
    )
    assert all(
        labels[row.utterance_id] == "neutral"
        for row in rows if row.corpus == "ravdess" and row.original_label == "neutral"
    )


def test_calm_drop_splits_stay_speaker_disjoint_and_runs_are_distinct():
    spec = load_spec(ROOT / "configs/calm_sensitivity.yaml")
    config = sensitivity_config(load_config(), spec)
    rows = project_rows(read_manifest(config.resolve(config.paths.manifest)), config)
    pair = make_pair_split(rows, config, "ravdess", "cremad", 0, "six")
    groups = [set(split.group_ids) for split in pair.splits().values()]
    assert not any(a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    runs = enumerate_runs(config, spec)
    assert len(runs) == 40


def test_neutral_exclusion_uses_a_distinct_five_class_mapping_and_run_ids():
    spec = load_spec(ROOT / "configs/neutral_excluded_sensitivity.yaml")
    base = load_config()
    config = sensitivity_config(base, spec)
    label_space = variant_label_space(spec)
    rows = project_rows(read_manifest(config.resolve(config.paths.manifest)), config, label_space)
    labels = {row.utterance_id: row.label_four for row in rows}

    assert config.label_map_hash != base.label_map_hash
    assert config.labels.spaces[label_space] == ["angry", "disgust", "fear", "happy", "sad"]
    assert all(
        labels[row.utterance_id] == ""
        for row in rows
        if row.original_label in {"calm", "neutral"}
    )

    pair = make_pair_split(rows, config, "ravdess", "cremad", 0, label_space)
    groups = [set(split.group_ids) for split in pair.splits().values()]
    assert not any(a & b for index, a in enumerate(groups) for b in groups[index + 1:])
    assert len(enumerate_runs(config, spec)) == 40


def test_diagnostic_record_ids_include_the_label_mapping_and_alignment_rung():
    base = load_config()
    calm_spec = load_spec(ROOT / "configs/calm_sensitivity.yaml")
    neutral_spec = load_spec(ROOT / "configs/neutral_excluded_sensitivity.yaml")
    calm_config = sensitivity_config(base, calm_spec)
    neutral_config = sensitivity_config(base, neutral_spec)

    calm_none = record_id(
        calm_config, calm_spec, source="ravdess", target="cremad", seed=0, alignment="none"
    )
    calm_coral = record_id(
        calm_config, calm_spec, source="ravdess", target="cremad", seed=0, alignment="coral"
    )
    neutral_none = record_id(
        neutral_config, neutral_spec, source="ravdess", target="cremad", seed=0, alignment="none"
    )
    assert len({calm_none, calm_coral, neutral_none}) == 3
