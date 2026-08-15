"""Table-driven label mapping tests.

Covers every raw label string each corpus is known to produce, for every label
space, plus the purity property the Phase 2 brief requires.
"""

from __future__ import annotations

import pytest

from ser.config import load_config
from ser.labels import (
    KNOWN_CORPORA,
    LabelPolicy,
    map_label,
    raw_labels_for,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def policy(config) -> LabelPolicy:
    return LabelPolicy.from_config(config)


# -- exhaustive coverage ---------------------------------------------------
def _all_cases():
    for corpus in KNOWN_CORPORA:
        for raw in raw_labels_for(corpus):
            for space in ("six", "four"):
                yield corpus, raw, space


@pytest.mark.parametrize("corpus,raw,space", list(_all_cases()))
def test_every_raw_label_maps_to_a_class_or_explicit_none(corpus, raw, space, policy, config):
    """Phase 2 assertion 4: every raw label either lands in the space or is
    explicitly excluded. Never anything else, never an exception."""
    result = map_label(corpus, raw, space, policy)
    assert result is None or result in config.labels.spaces[space]


def test_the_table_covers_every_label_actually_in_the_manifest(config):
    """Ties the table above to reality: a corpus emitting a label the table does
    not know about would otherwise pass every test while being silently dropped."""
    from ser.manifest import read_manifest

    manifest_path = config.resolve(config.paths.manifest)
    if not manifest_path.exists():
        pytest.skip("manifest not built yet")

    rows = read_manifest(manifest_path)
    for corpus in {row.corpus for row in rows}:
        seen = {row.original_label for row in rows if row.corpus == corpus}
        known = set(raw_labels_for(corpus))
        assert seen <= known, f"{corpus}: manifest has labels absent from the table: {seen - known}"


# -- purity ----------------------------------------------------------------
def test_map_label_is_deterministic(policy):
    first = [map_label("ravdess", raw, "six", policy) for raw in raw_labels_for("ravdess")]
    second = [map_label("ravdess", raw, "six", policy) for raw in raw_labels_for("ravdess")]
    assert first == second


def test_map_label_does_not_mutate_its_policy(policy):
    before = (
        policy.iemocap_excited_to_happy,
        policy.iemocap_frustrated,
        policy.ravdess_calm_to_neutral,
        dict(policy.spaces),
    )
    for corpus in KNOWN_CORPORA:
        for raw in raw_labels_for(corpus):
            map_label(corpus, raw, "four", policy)
    after = (
        policy.iemocap_excited_to_happy,
        policy.iemocap_frustrated,
        policy.ravdess_calm_to_neutral,
        dict(policy.spaces),
    )
    assert before == after


def test_case_and_whitespace_are_normalised(policy):
    assert map_label("cremad", "  ANGRY ", "six", policy) == "angry"


# -- the A-series decisions ------------------------------------------------
def test_ravdess_calm_merges_into_neutral(policy):
    assert map_label("ravdess", "calm", "six", policy) == "neutral"


def test_ravdess_surprised_is_excluded(policy):
    """Not in either space; it has no counterpart in CREMA-D."""
    assert map_label("ravdess", "surprised", "six", policy) is None
    assert map_label("ravdess", "surprised", "four", policy) is None


def test_iemocap_excited_merges_into_happy(policy):
    assert map_label("iemocap", "excited", "four", policy) == "happy"


def test_iemocap_frustration_is_dropped_not_merged_into_anger(policy):
    """A3: merging would manufacture the prior skew the thesis claims to find."""
    assert policy.iemocap_frustrated == "drop"
    assert map_label("iemocap", "frustrated", "four", policy) is None


def test_iemocap_no_agreement_marker_is_excluded(policy):
    """'xxx' is the no-majority marker under majority_vote_discard_disagreement."""
    assert map_label("iemocap", "xxx", "four", policy) is None


def test_iemocap_fear_and_disgust_excluded_from_the_four_class_space(policy):
    """A4: ~40 fear and ~2 disgust utterances guarantee collapse."""
    assert map_label("iemocap", "fear", "four", policy) is None
    assert map_label("iemocap", "disgust", "four", policy) is None


def test_merge_angry_policy_would_change_the_mapping(policy):
    """The decision is real, not decorative: flipping it changes the output."""
    merged = LabelPolicy(
        spaces=policy.spaces,
        iemocap_label_source=policy.iemocap_label_source,
        iemocap_excited_to_happy=policy.iemocap_excited_to_happy,
        iemocap_frustrated="merge_angry",
        ravdess_calm_to_neutral=policy.ravdess_calm_to_neutral,
    )
    assert map_label("iemocap", "frustrated", "four", merged) == "angry"


# -- refusal to guess ------------------------------------------------------
def test_unknown_raw_label_raises_rather_than_returning_none(policy):
    """None must mean 'deliberately excluded' and nothing else."""
    with pytest.raises(ValueError, match="unrecognised raw label"):
        map_label("cremad", "bored", "six", policy)


def test_unknown_corpus_raises(policy):
    with pytest.raises(ValueError, match="unknown corpus"):
        map_label("emodb", "angry", "six", policy)


def test_unknown_label_space_raises(policy):
    with pytest.raises(ValueError, match="unknown label space"):
        map_label("cremad", "angry", "five", policy)


def test_policy_refuses_to_build_while_a_decision_is_unmade(config, tmp_path):
    import copy

    import yaml

    from ser.config import ConfigError, load_config as _load

    raw = copy.deepcopy(config.raw)
    raw["labels"]["iemocap_frustrated"] = None
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="undecided"):
        LabelPolicy.from_config(_load(str(path)))
