"""Phase 8 statistics: the properties the reported intervals depend on."""

from __future__ import annotations

import numpy as np
import pytest

from ser.metrics import macro_f1
from ser.phase8 import (
    confusion_by_group,
    holm,
    macro_f1_from_confusion,
    paired_cluster_bootstrap,
    per_class_f1_from_confusion,
    seed_interval,
)

CLASSES = ["a", "b", "c", "d"]


def _confusion(y_true, y_pred, groups, n_groups=5):
    return confusion_by_group(y_true, y_pred, groups, len(CLASSES), n_groups)


def test_macro_f1_from_confusion_matches_the_projects_own_metric():
    """Everything in Phase 8 is scored from confusion tensors rather than from
    label lists, so the two paths must not be allowed to drift apart."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        y_true = rng.integers(0, 4, 300)
        y_pred = rng.integers(0, 4, 300)
        conf = _confusion(y_true, y_pred, rng.integers(0, 5, 300)).sum(axis=0)
        assert macro_f1_from_confusion(conf) == pytest.approx(
            macro_f1([CLASSES[i] for i in y_true], [CLASSES[i] for i in y_pred], CLASSES),
            abs=1e-12,
        )


def test_macro_f1_agrees_on_a_collapsed_prediction():
    """The collapse case is the one that matters -- it is the failure mode the
    floors exist to detect, and a mismatch there would misreport it."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 4, 200)
    y_pred = np.zeros(200, dtype=int)
    conf = _confusion(y_true, y_pred, rng.integers(0, 5, 200)).sum(axis=0)
    assert macro_f1_from_confusion(conf) == pytest.approx(
        macro_f1([CLASSES[i] for i in y_true], [CLASSES[0]] * 200, CLASSES), abs=1e-12
    )


def test_confusion_is_additive_over_groups():
    """The whole bootstrap rests on this: a replicate is a weighted sum of
    per-speaker confusions, not a rescan of the utterances."""
    rng = np.random.default_rng(2)
    y_true, y_pred = rng.integers(0, 4, 400), rng.integers(0, 4, 400)
    groups = rng.integers(0, 5, 400)
    per_group = _confusion(y_true, y_pred, groups)
    pooled = _confusion(y_true, y_pred, np.zeros(400, dtype=int), n_groups=1)[0]
    np.testing.assert_array_equal(per_group.sum(axis=0), pooled)

    # Doubling a group's weight equals duplicating its utterances.
    weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
    weighted = np.tensordot(weights, per_group, axes=(0, 0))
    keep = groups == 0
    duplicated = pooled + _confusion(
        y_true[keep], y_pred[keep], np.zeros(keep.sum(), dtype=int), n_groups=1
    )[0]
    np.testing.assert_array_equal(weighted, duplicated)


def test_per_class_f1_is_nan_for_a_class_absent_from_both():
    conf = np.zeros((4, 4))
    conf[0, 0] = 5
    values = per_class_f1_from_confusion(conf)
    assert values[0] == pytest.approx(1.0)
    assert np.isnan(values[1:]).all()


def test_paired_bootstrap_finds_no_difference_between_identical_arms():
    """A test that cannot return "no difference" is not a test."""
    rng = np.random.default_rng(3)
    arm = {
        seed: [_confusion(rng.integers(0, 4, 200), rng.integers(0, 4, 200),
                          rng.integers(0, 5, 200))]
        for seed in range(5)
    }
    stat = paired_cluster_bootstrap(arm, arm, {s: 5 for s in range(5)}, n_boot=200, seed=0)
    assert stat["diff"] == pytest.approx(0.0, abs=1e-12)
    assert stat["lo"] == pytest.approx(0.0, abs=1e-12)
    assert stat["hi"] == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_recovers_a_planted_difference():
    rng = np.random.default_rng(4)
    y_true = {seed: rng.integers(0, 4, 400) for seed in range(5)}
    groups = {seed: rng.integers(0, 8, 400) for seed in range(5)}
    good, bad = {}, {}
    for seed in range(5):
        truth = y_true[seed]
        # 80% correct against 30% correct.
        strong = np.where(rng.random(400) < 0.8, truth, rng.integers(0, 4, 400))
        weak = np.where(rng.random(400) < 0.3, truth, rng.integers(0, 4, 400))
        good[seed] = [confusion_by_group(truth, strong, groups[seed], 4, 8)]
        bad[seed] = [confusion_by_group(truth, weak, groups[seed], 4, 8)]

    stat = paired_cluster_bootstrap(good, bad, {s: 8 for s in range(5)}, n_boot=400, seed=0)
    assert stat["diff"] > 0.2
    assert stat["lo"] > 0, "a large planted effect must exclude zero"
    assert stat["p"] <= 2.0 / 400


def test_paired_bootstrap_p_value_is_floored_at_its_resolution():
    """A resampling test cannot resolve below 1/n_boot; reporting a smaller
    number would be fabricating precision the procedure does not have."""
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 4, 300)
    groups = rng.integers(0, 6, 300)
    perfect = {0: [confusion_by_group(truth, truth, groups, 4, 6)]}
    useless = {0: [confusion_by_group(truth, np.zeros(300, dtype=int), groups, 4, 6)]}
    stat = paired_cluster_bootstrap(perfect, useless, {0: 6}, n_boot=100, seed=0)
    assert stat["p"] == pytest.approx(1 / 100)
    assert stat["p_at_floor"] is True


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(6)
    a = {s: [_confusion(rng.integers(0, 4, 100), rng.integers(0, 4, 100),
                        rng.integers(0, 5, 100))] for s in range(3)}
    b = {s: [_confusion(rng.integers(0, 4, 100), rng.integers(0, 4, 100),
                        rng.integers(0, 5, 100))] for s in range(3)}
    ns = {s: 5 for s in range(3)}
    first = paired_cluster_bootstrap(a, b, ns, n_boot=100, seed=11)
    second = paired_cluster_bootstrap(a, b, ns, n_boot=100, seed=11)
    assert first == second


def test_holm_matches_worked_examples():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    # Monotone: an adjusted p can never fall below one ranked before it.
    adjusted = holm([0.001, 0.002, 0.3, 0.4])
    assert adjusted == sorted(adjusted)
    assert all(a <= 1.0 for a in holm([0.9, 0.9, 0.9]))


def test_holm_is_at_least_as_powerful_as_bonferroni():
    pvalues = [0.001, 0.02, 0.04, 0.5]
    assert all(
        h <= b + 1e-12
        for h, b in zip(holm(pvalues), [min(1.0, len(pvalues) * p) for p in pvalues])
    )


def test_seed_interval_refuses_to_invent_width_from_one_observation():
    single = seed_interval([0.4])
    assert single["mean"] == pytest.approx(0.4)
    assert np.isnan(single["lo"]) and np.isnan(single["hi"])

    five = seed_interval([0.30, 0.32, 0.34, 0.36, 0.38])
    assert five["lo"] < five["mean"] < five["hi"]
    assert five["n"] == 5


def test_seed_interval_drops_nulls_rather_than_imputing():
    """The discrepancy columns carry legitimate nulls -- rungs with no warm
    start, and Stage 0 rows predating a column."""
    assert seed_interval([0.1, None, 0.3])["n"] == 2
    assert seed_interval([None, None])["n"] == 0
