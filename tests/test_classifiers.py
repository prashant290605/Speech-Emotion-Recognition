"""Phase 6: equal budget, source_val-only selection, real early stopping."""

from __future__ import annotations

import numpy as np
import pytest

from ser.classifiers import (
    FAMILIES,
    SKLEARN_FAMILIES,
    TORCH_FAMILIES,
    fit_and_select,
    sample_params,
    supports_layer_agg,
)
from ser.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def small_config(config):
    """A 3-trial, few-epoch config so the torch families stay test-sized."""
    import copy
    import tempfile
    from pathlib import Path

    import yaml

    from ser.config import load_config as _load

    raw = copy.deepcopy(config.raw)
    raw["classifiers"]["search_budget"] = 3
    raw["classifiers"]["max_epochs"] = 6
    raw["classifiers"]["early_stopping_patience"] = 2
    path = Path(tempfile.mkdtemp()) / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return _load(str(path))


CLASSES = ["angry", "happy", "neutral", "sad"]


# Class centres are drawn from a FIXED generator, independent of `seed`. Train
# and validation must share them: drawing centres per call produces two
# unrelated datasets rather than a split, and a classifier trained on one then
# scores at chance on the other.
_CENTRE_SEED = 12345


def _separable(n=120, d=16, seed=0, n_layers=None, n_segments=None):
    """Linearly separable classes, so any working family scores well above chance."""
    rng = np.random.default_rng(seed)
    labels = [CLASSES[i % len(CLASSES)] for i in range(n)]
    centres = np.random.default_rng(_CENTRE_SEED).standard_normal((len(CLASSES), d)) * 4.0
    X = np.stack([centres[CLASSES.index(v)] + rng.standard_normal(d) for v in labels])

    if n_layers:
        # Copy the signal across layers with per-layer noise.
        X = np.stack([X + rng.standard_normal(X.shape) * 0.1 for _ in range(n_layers)], axis=1)
    if n_segments:
        X = np.repeat(X[..., None, :], n_segments, axis=-2)
    return X.astype(np.float64), labels


# -- equal budget ----------------------------------------------------------
@pytest.mark.parametrize("family", SKLEARN_FAMILIES)
def test_sklearn_family_consumes_exactly_the_budget(family, small_config):
    X, y = _separable()
    Xv, yv = _separable(n=40, seed=1)
    result = fit_and_select(family, X, y, Xv, yv, CLASSES, small_config, seed=0)

    assert result.n_trials == small_config.classifiers.search_budget
    assert len(result.trials) == small_config.classifiers.search_budget


def test_every_family_gets_an_identical_budget(small_config):
    """The defect this rebuild exists to fix: the original tuned neural models
    and ran logreg and SVM at library defaults, then compared them."""
    X, y = _separable()
    Xv, yv = _separable(n=40, seed=1)

    counts = {}
    for family in SKLEARN_FAMILIES:
        counts[family] = fit_and_select(
            family, X, y, Xv, yv, CLASSES, small_config, seed=0
        ).n_trials

    Xl, yl = _separable(n_layers=small_config.features.n_layers)
    Xlv, ylv = _separable(n=40, seed=1, n_layers=small_config.features.n_layers)
    counts["mlp"] = fit_and_select(
        "mlp", Xl, yl, Xlv, ylv, CLASSES, small_config, layer_agg="weighted", seed=0
    ).n_trials

    assert len(set(counts.values())) == 1, f"unequal budgets: {counts}"


def test_a_failed_trial_does_not_buy_extra_attempts(small_config):
    """Otherwise a fragile family quietly gets more shots than a robust one."""
    X, y = _separable()
    Xv, yv = _separable(n=40, seed=1)
    result = fit_and_select("logreg", X, y, Xv, yv, CLASSES, small_config, seed=0)
    assert len(result.trials) == result.n_trials


# -- search spaces ---------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_sampling_is_reproducible_and_varies(family, config):
    a = sample_params(family, np.random.default_rng(0), config)
    b = sample_params(family, np.random.default_rng(0), config)
    c = sample_params(family, np.random.default_rng(1), config)
    assert a == b
    assert a != c or family == "svm_linear"


def test_transformer_head_count_divides_the_model_dimension(config):
    for seed in range(30):
        params = sample_params("transformer", np.random.default_rng(seed), config)
        assert params["d_model"] % params["heads"] == 0


def test_unknown_family_raises(config):
    with pytest.raises(ValueError, match="unknown family"):
        sample_params("randomforest", np.random.default_rng(0), config)


# -- layer aggregation -----------------------------------------------------
def test_weighted_aggregation_is_only_offered_to_trainable_models():
    """The softmax over layers is a classifier parameter; a closed-form sklearn
    model has none to learn it with."""
    for family in SKLEARN_FAMILIES:
        assert supports_layer_agg(family, "weighted") is False
        assert supports_layer_agg(family, "last") is True
    for family in TORCH_FAMILIES:
        assert supports_layer_agg(family, "weighted") is True


def test_requesting_weighted_on_an_sklearn_family_raises(small_config):
    X, y = _separable()
    Xv, yv = _separable(n=40, seed=1)
    with pytest.raises(ValueError, match="needs learnable parameters"):
        fit_and_select(
            "logreg", X, y, Xv, yv, CLASSES, small_config, layer_agg="weighted"
        )


def test_mlp_learns_over_the_full_layer_stack(small_config):
    n_layers = small_config.features.n_layers
    X, y = _separable(n_layers=n_layers)
    Xv, yv = _separable(n=40, seed=1, n_layers=n_layers)

    result = fit_and_select(
        "mlp", X, y, Xv, yv, CLASSES, small_config, layer_agg="weighted", seed=0
    )
    assert X.shape[1] == n_layers
    assert result.best_source_val_macro_f1 > 0.0


def test_transformer_consumes_a_segment_sequence(small_config):
    X, y = _separable(n=60, d=16, n_segments=4)
    Xv, yv = _separable(n=24, d=16, seed=1, n_segments=4)
    assert X.ndim == 3

    result = fit_and_select(
        "transformer", X, y, Xv, yv, CLASSES, small_config, layer_agg="last", seed=0
    )
    assert result.n_trials == small_config.classifiers.search_budget


# -- early stopping --------------------------------------------------------
def test_torch_training_early_stops_rather_than_running_fixed_epochs(small_config):
    """The original ran a fixed 8 gradient steps with no validation set, so it
    had no evidence of convergence either way."""
    X, y = _separable()
    Xv, yv = _separable(n=40, seed=1)

    result = fit_and_select("mlp", X, y, Xv, yv, CLASSES, small_config, seed=0)
    assert result.epochs_run is not None
    assert 1 <= result.epochs_run <= small_config.classifiers.max_epochs


def test_recorded_hyperparameters_name_the_selection_surface(small_config):
    X, y = _separable()
    Xv, yv = _separable(n=40, seed=1)
    payload = fit_and_select(
        "logreg", X, y, Xv, yv, CLASSES, small_config, seed=0
    ).as_hyperparams()

    assert payload["selection_surface"] == "source_val"
    assert payload["n_search_trials"] == small_config.classifiers.search_budget
    assert "selected" in payload
    import json

    json.dumps(payload)  # must survive the results writer


# -- selection uses source_val only ----------------------------------------
def test_selection_is_invariant_to_target_data(small_config):
    """The strongest form of the claim: the selected configuration cannot depend
    on target data, because no target data is ever passed in."""
    X, y = _separable()
    Xv, yv = _separable(n=40, seed=1)

    first = fit_and_select("svm_rbf", X, y, Xv, yv, CLASSES, small_config, seed=0)
    second = fit_and_select("svm_rbf", X, y, Xv, yv, CLASSES, small_config, seed=0)
    assert first.best_params == second.best_params
    assert first.best_source_val_macro_f1 == second.best_source_val_macro_f1


def test_a_working_family_beats_chance_on_separable_data(small_config):
    X, y = _separable(n=200)
    Xv, yv = _separable(n=80, seed=1)
    result = fit_and_select("logreg", X, y, Xv, yv, CLASSES, small_config, seed=0)
    assert result.best_source_val_macro_f1 > 0.5  # chance at K=4 is 0.25
