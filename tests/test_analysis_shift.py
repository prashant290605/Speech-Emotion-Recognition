"""Phase 9: the A10 firewall, and the shift decomposition's estimators."""

from __future__ import annotations

import numpy as np
import pytest

from ser.analysis import FirewallViolation, assert_conditional_shift_firewall
from ser.analysis.shift import (
    apply_prior_correction,
    bbse_weights,
    class_priors,
    conditional_mmd_by_class,
    em_prior_estimate,
    label_shift_kl,
)
from ser.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


# -- A10 -------------------------------------------------------------------
def test_conditional_shift_firewall_holds():
    """The whole point of A10. If this fails, a target-label-derived quantity
    has become reachable from fitting or selection."""
    assert_conditional_shift_firewall()


def test_firewall_catches_a_schema_leak(monkeypatch):
    """A test that cannot fail is not a guard. Simulate the exact regression
    A10 warns about -- a "diagnostics" column carrying the conditional term."""
    import ser.utils.results as results

    monkeypatch.setattr(
        results, "FIELD_NAMES", results.FIELD_NAMES + ("conditional_mmd_json",)
    )
    with pytest.raises(FirewallViolation, match="conditional"):
        assert_conditional_shift_firewall()


def test_no_pipeline_module_imports_the_analysis_package():
    """Checked by reading the source, so it cannot be satisfied by convention."""
    import inspect

    from ser import alignment, blending, classifiers, run_grid

    for module in (alignment, classifiers, run_grid, blending):
        source = inspect.getsource(module)
        assert "analysis" not in source.replace("analysis layer", ""), module.__name__


# -- label shift -----------------------------------------------------------
def test_kl_is_zero_for_identical_priors_and_positive_otherwise():
    classes = ["a", "b"]
    assert label_shift_kl(["a", "b"], ["a", "b"], classes)["kl_nats"] == 0.0
    shifted = label_shift_kl(["a"] * 50 + ["b"] * 50, ["a"] * 90 + ["b"] * 10, classes)
    assert shifted["kl_nats"] > 0
    assert shifted["total_variation"] == pytest.approx(0.4)


def test_kl_direction_is_target_given_source():
    """KL is asymmetric and the direction is a claim. Target||source is the one
    that bounds how badly a source-trained prior misprices the target."""
    classes = ["a", "b"]
    forward = label_shift_kl(["a"] * 90 + ["b"] * 10, ["a"] * 50 + ["b"] * 50, classes)
    reverse = label_shift_kl(["a"] * 50 + ["b"] * 50, ["a"] * 90 + ["b"] * 10, classes)
    assert forward["kl_nats"] != pytest.approx(reverse["kl_nats"])
    # P_target is the first argument's *second* set, so check it landed right.
    assert forward["target_prior"] == pytest.approx([0.5, 0.5])
    assert forward["source_prior"] == pytest.approx([0.9, 0.1])


def test_a_class_missing_from_source_is_flagged_not_smoothed_away():
    result = label_shift_kl(["a"] * 10, ["a"] * 5 + ["b"] * 5, ["a", "b"])
    assert result["classes_without_source_support"] == ["b"]
    assert result["kl_nats"] > 10, "an unsupported class must dominate the KL"


# -- conditional shift -----------------------------------------------------
def test_conditional_mmd_reports_undefined_below_min_support(config):
    """A number computed on a few dozen samples would be read as a measurement.
    Per-class n must accompany every row, defined or not."""
    rng = np.random.default_rng(0)
    X_source, X_target = rng.standard_normal((120, 8)), rng.standard_normal((120, 8))
    y_source = ["a"] * 100 + ["b"] * 20
    y_target = ["a"] * 100 + ["b"] * 20

    rows = conditional_mmd_by_class(
        X_source, y_source, X_target, y_target, ["a", "b"], config, min_support=50
    )
    by_class = {r["class"]: r for r in rows}
    assert by_class["a"]["effect_size"] is not None
    assert by_class["b"]["effect_size"] is None
    assert "support below 50" in by_class["b"]["undefined_reason"]
    for row in rows:
        assert row["n_source"] > 0 and row["n_target"] > 0


def test_conditional_mmd_is_larger_for_a_class_that_was_moved(config):
    rng = np.random.default_rng(1)
    X_source = rng.standard_normal((200, 6))
    X_target = rng.standard_normal((200, 6))
    y = ["a"] * 100 + ["b"] * 100
    # Displace class b on the target side only.
    X_target[100:] += 6.0

    rows = {
        r["class"]: r
        for r in conditional_mmd_by_class(
            X_source, y, X_target, y, ["a", "b"], config, min_support=50
        )
    }
    assert rows["b"]["effect_size"] > rows["a"]["effect_size"]


# -- label-shift correction ------------------------------------------------
def test_bbse_returns_none_for_an_unidentifiable_confusion_matrix():
    """A classifier that cannot separate the classes on source cannot identify
    the target prior either; a huge unstable weight vector would hide that."""
    assert bbse_weights(np.full((2, 2), 0.25), np.array([0.5, 0.5])) is None


def test_bbse_and_em_recover_a_planted_prior_shift():
    """Both estimators must work on a case with a known answer, or a null result
    on the real data says nothing."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)

    def draw(n_a, n_b):
        X = np.vstack([rng.normal(-1.5, 1.0, (n_a, 3)), rng.normal(1.5, 1.0, (n_b, 3))])
        return X, np.array(["a"] * n_a + ["b"] * n_b)

    X_source, y_source = draw(2000, 2000)
    X_target, _ = draw(1600, 400)
    model = LogisticRegression(max_iter=2000).fit(X_source, y_source)
    order = list(model.classes_)
    source_prior = class_priors(y_source, order)

    estimated = em_prior_estimate(model.predict_proba(X_target), source_prior)
    assert estimated[0] == pytest.approx(0.8, abs=0.05)

    predicted = model.predict(X_source)
    confusion = np.array(
        [[np.mean((y_source == a) & (predicted == b)) for b in order] for a in order]
    )
    mu = np.array([np.mean(model.predict(X_target) == c) for c in order])
    weights = bbse_weights(confusion, mu)
    assert weights is not None
    assert weights[0] == pytest.approx(1.6, abs=0.15)


def test_prior_correction_is_identity_when_the_priors_match():
    """The falsifiable test depends on this: no estimated shift, no change."""
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])
    prior = np.array([0.5, 0.5])
    np.testing.assert_allclose(
        apply_prior_correction(probabilities, prior, prior), probabilities
    )


def test_prior_correction_moves_decisions_toward_the_upweighted_class():
    probabilities = np.array([[0.55, 0.45]])
    corrected = apply_prior_correction(
        probabilities, np.array([0.5, 0.5]), np.array([0.1, 0.9])
    )
    assert corrected[0, 1] > corrected[0, 0], "the upweighted class must win"
