"""The matched-n control, and the convergence and blending guards.

All three exist to stop a specific defect reaching a table:

* an asymmetry between transfer directions that is really a training-set size
  difference,
* a baseline whose score reflects where its optimiser stopped,
* a row labelled with a `blend_alpha` whose features were never blended.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from ser.config import load_config
from ser.manifest import read_manifest
from ser.splits import make_pair_split


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def rows(config):
    path = config.resolve(config.paths.manifest)
    if not path.exists():
        pytest.skip("manifest not built yet; run `ser manifest`")
    return read_manifest(path)


@pytest.fixture(scope="module")
def label_of(rows):
    by_id = {row.utterance_id: row for row in rows}

    def lookup(utterance_id: str, space: str) -> str:
        row = by_id[utterance_id]
        return row.label_six if space == "six" else row.label_four

    return lookup


# -- matched n -------------------------------------------------------------
def test_both_directions_get_the_same_source_train_size(rows, config):
    """Without this, direction is confounded with 6x the training data."""
    for seed in config.splits.seeds:
        forward = make_pair_split(rows, config, "ravdess", "cremad", seed)
        reverse = make_pair_split(rows, config, "cremad", "ravdess", seed)
        assert len(forward.source_train) == len(reverse.source_train)


def test_only_the_larger_direction_is_subsampled(rows, config):
    """The cap is the smaller side's natural size, so it is never padded."""
    forward = make_pair_split(rows, config, "ravdess", "cremad", 0)
    reverse = make_pair_split(rows, config, "cremad", "ravdess", 0)

    assert forward.source_train_cap is None, "RAVDESS is already the smaller side"
    assert reverse.source_train_cap == len(forward.source_train)


def test_matched_subsample_preserves_class_proportions(rows, config, label_of):
    """A uniform subsample would shift the label prior and confound the very
    comparison the cap exists to enable.

    Compared against the UNCAPPED split, which is the only comparison that says
    anything -- checking the capped split against itself would pass whatever
    the allocation did.
    """
    from ser.splits import _build_pair_split

    capped = make_pair_split(rows, config, "cremad", "ravdess", 0)
    uncapped = _build_pair_split(rows, config, "cremad", "ravdess", 0)
    space = capped.label_space

    kept = Counter(label_of(u, space) for u in capped.source_train.utterance_ids)
    natural = Counter(label_of(u, space) for u in uncapped.source_train.utterance_ids)
    n_kept, n_natural = sum(kept.values()), sum(natural.values())

    assert n_kept < n_natural, "this pair should actually have been subsampled"
    for name, count in natural.items():
        # Largest remainder is exact to within one utterance per class.
        assert abs(kept[name] - count * n_kept / n_natural) <= 1.0
        assert abs(kept[name] / n_kept - count / n_natural) < 0.01


def test_matched_subsample_keeps_speakers_disjoint(rows, config):
    """Utterances are removed, never moved, so every leakage guarantee holds."""
    pair = make_pair_split(rows, config, "cremad", "ravdess", 0)
    train = pair.source_train
    for role, split in pair.splits().items():
        if role == "source_train":
            continue
        assert not set(train.utterance_ids) & set(split.utterance_ids)
        if split.corpus == train.corpus:
            assert not set(train.group_ids) & set(split.group_ids)


def test_matched_subsample_is_deterministic(rows, config):
    a = make_pair_split(rows, config, "cremad", "ravdess", 0).source_train
    b = make_pair_split(rows, config, "cremad", "ravdess", 0).source_train
    assert a.utterance_ids == b.utterance_ids
    assert a.group_ids == b.group_ids


def test_matched_subsample_is_a_subset_of_the_uncapped_split(rows, config):
    """It must select from the existing source_train, not re-partition."""
    from ser.splits import _build_pair_split

    uncapped = _build_pair_split(rows, config, "cremad", "ravdess", 0)
    capped = make_pair_split(rows, config, "cremad", "ravdess", 0)
    assert set(capped.source_train.utterance_ids) < set(
        uncapped.source_train.utterance_ids
    )


def test_matched_n_is_recorded_in_the_split_spec_hash(config):
    """A control nobody can see in the provenance stamp is not a control."""
    assert config.classify_config_key("splits", "matched_source_train") == (
        "facet:split_spec_hash"
    )


# -- convergence -----------------------------------------------------------
def test_max_iter_is_not_a_searched_hyperparameter(config):
    """Searching a convergence budget lets a trial win by stopping early."""
    from ser.classifiers import sample_params

    rng = np.random.default_rng(0)
    for _ in range(20):
        assert "max_iter" not in sample_params("logreg", rng, config)


def test_non_convergence_raises_instead_of_warning(config):
    from ser.classifiers import NotConverged, _assert_converged, _build_sklearn

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30))
    y = [["a", "b", "c"][i % 3] for i in range(200)]

    model = _build_sklearn("logreg", {"C": 1.0, "class_weight": None}, 0, config)
    model.max_iter = 1
    model.fit(X, y)
    with pytest.raises(NotConverged, match="max_iter"):
        _assert_converged(model, "logreg")


def test_unlimited_solvers_are_not_reported_as_non_converged(config):
    """libsvm uses max_iter=-1 for "no limit". Comparing against it made every
    SVC fit look non-converged the first time this guard was written."""
    from ser.classifiers import _assert_converged, _build_sklearn

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 8))
    y = [["a", "b"][i % 2] for i in range(120)]

    model = _build_sklearn("svm_rbf", {"C": 1.0, "gamma": 0.1, "class_weight": None}, 0, config)
    model.fit(X, y)
    assert model.max_iter == -1
    _assert_converged(model, "svm_rbf")  # must not raise


def test_selection_records_the_iteration_count(config):
    from ser.classifiers import fit_and_select

    rng = np.random.default_rng(0)
    X, Xv = rng.standard_normal((200, 20)), rng.standard_normal((60, 20))
    y = [["a", "b", "c"][i % 3] for i in range(200)]
    yv = [["a", "b", "c"][i % 3] for i in range(60)]

    result = fit_and_select("logreg", X, y, Xv, yv, ["a", "b", "c"], config, seed=0)
    assert result.solver_n_iter is not None and result.solver_n_iter > 0
    assert result.as_hyperparams()["solver_n_iter"] == result.solver_n_iter


# -- blending --------------------------------------------------------------
def test_blend_endpoints_are_the_two_unblended_conditions():
    """An inverted alpha would silently swap aligned for original everywhere."""
    from ser.blending import blend

    rng = np.random.default_rng(0)
    original, aligned = rng.standard_normal((50, 6)), rng.standard_normal((50, 6))

    np.testing.assert_allclose(blend(original, aligned, 0.0), original)
    np.testing.assert_allclose(blend(original, aligned, 1.0), aligned)


def test_unimplemented_blending_modes_refuse_to_run():
    """`gaa` is enumerable but has no run-path implementation. Running it would
    write rows labelled `gaa` whose features were never grouped."""
    from ser.run_grid import GridRun, _blend_alpha_for

    base = dict(
        source="ravdess", target="cremad", seed=0, backbone="hubert",
        feature_branch="ssl", layer_agg="last", layer_index=None,
        alignment="coral", alignment_eps=0.1, alignment_lam=None,
        n_groups=None, classifier="logreg",
    )
    with pytest.raises(NotImplementedError, match="gaa"):
        _blend_alpha_for(GridRun(blending="gaa", blend_alpha=None, **base))
    with pytest.raises(ValueError, match="blend_alpha"):
        _blend_alpha_for(GridRun(blending="scalar", blend_alpha=None, **base))
    assert _blend_alpha_for(GridRun(blending="scalar", blend_alpha=0.25, **base)) == 0.25


def test_blending_arm_only_touches_rungs_where_alpha_means_something(config):
    """With `none` and `zscore`, every alpha is the same features."""
    from ser.blending import BLENDABLE_ALIGNMENTS
    from ser.run_grid import STAGE2_SURVIVING

    for method in STAGE2_SURVIVING["blending_arm"]["alignments"]:
        assert method in BLENDABLE_ALIGNMENTS
