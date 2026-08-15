"""Phase 4: metrics, chance floors, statistics, and the sample-rate guard."""

from __future__ import annotations

import numpy as np
import pytest

from ser.baselines import (
    analytic_majority_macro_f1,
    analytic_stratified_macro_f1,
    analytic_uniform_macro_f1,
    all_floors,
    majority_class,
    stratified_random,
    uniform_random,
)
from ser.metrics import (
    accuracy,
    all_metrics,
    confusion_matrix,
    macro_f1,
    n_collapsed_classes,
    per_class_f1,
    uar,
)
from ser.stats import bootstrap_ci, holm_bonferroni, wilcoxon_signed_rank

SIX = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
FOUR = ["angry", "happy", "neutral", "sad"]


# -- metrics against sklearn -----------------------------------------------
def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    y_true = [SIX[i] for i in rng.integers(0, 6, 500)]
    y_pred = [SIX[i] for i in rng.integers(0, 6, 500)]

    assert macro_f1(y_true, y_pred, SIX) == pytest.approx(
        f1_score(y_true, y_pred, average="macro", labels=SIX, zero_division=0)
    )


def test_per_class_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(1)
    y_true = [SIX[i] for i in rng.integers(0, 6, 300)]
    y_pred = [SIX[i] for i in rng.integers(0, 6, 300)]

    mine = per_class_f1(y_true, y_pred, SIX)
    theirs = f1_score(y_true, y_pred, average=None, labels=SIX, zero_division=0)
    for i, name in enumerate(SIX):
        assert mine[name] == pytest.approx(theirs[i])


def test_confusion_matrix_matches_sklearn_and_is_true_by_predicted():
    from sklearn.metrics import confusion_matrix as sk_confusion

    y_true = ["angry", "angry", "sad", "happy"]
    y_pred = ["angry", "sad", "sad", "angry"]
    mine = confusion_matrix(y_true, y_pred, SIX)
    np.testing.assert_array_equal(mine, sk_confusion(y_true, y_pred, labels=SIX))
    # Rows are true, columns predicted: one angry was predicted sad.
    assert mine[SIX.index("angry")][SIX.index("sad")] == 1


def test_accuracy_and_uar_match_sklearn():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    rng = np.random.default_rng(2)
    y_true = [SIX[i] for i in rng.integers(0, 6, 400)]
    y_pred = [SIX[i] for i in rng.integers(0, 6, 400)]

    assert accuracy(y_true, y_pred, SIX) == pytest.approx(accuracy_score(y_true, y_pred))
    assert uar(y_true, y_pred, SIX) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )


def test_unpredicted_class_scores_zero_not_dropped():
    """Silently dropping a class the model never predicts is what makes a
    collapsed model look competent."""
    y_true = SIX[:]
    y_pred = ["angry"] * 6
    scores = per_class_f1(y_true, y_pred, SIX)
    assert set(scores) == set(SIX)
    assert all(scores[name] == 0.0 for name in SIX if name != "angry")
    assert n_collapsed_classes(y_pred, SIX) == 5


# -- analytic floors -------------------------------------------------------
def test_uniform_chance_is_one_over_k_for_a_balanced_target():
    assert analytic_uniform_macro_f1([1 / 6] * 6) == pytest.approx(1 / 6)
    assert analytic_uniform_macro_f1([1 / 4] * 4) == pytest.approx(1 / 4)


def test_the_two_floors_this_project_needs():
    """A4 made chance pair-dependent: 0.167 at K=6, 0.250 at K=4."""
    assert analytic_uniform_macro_f1([1 / 6] * 6) == pytest.approx(0.1667, abs=1e-3)
    assert analytic_uniform_macro_f1([1 / 4] * 4) == pytest.approx(0.2500, abs=1e-3)


def test_majority_collapse_floor_is_near_five_percent_at_k6():
    value = analytic_majority_macro_f1([1 / 6] * 6, 0)
    assert value == pytest.approx(2 * (1 / 6) / (1 / 6 + 1) / 6)
    assert value == pytest.approx(0.0476, abs=1e-3)


def test_majority_floor_depends_on_the_target_distribution():
    """The same constant predictor scores differently against different targets,
    which is why floors are computed per pair rather than once."""
    skewed = analytic_majority_macro_f1([0.5, 0.1, 0.2, 0.2], 0)
    flat = analytic_majority_macro_f1([0.25] * 4, 0)
    assert skewed > flat


def test_stratified_equals_uniform_when_the_source_prior_is_uniform():
    assert analytic_stratified_macro_f1([1 / 6] * 6, [1 / 6] * 6) == pytest.approx(
        analytic_uniform_macro_f1([1 / 6] * 6)
    )


# -- empirical floors ------------------------------------------------------
def _labels(prior, n, names, seed=0):
    rng = np.random.default_rng(seed)
    return [names[i] for i in rng.choice(len(names), size=n, p=prior)]


def test_uniform_empirical_mean_lands_within_ci_of_the_analytic_value():
    """The closed form is a ratio of expectations, not the expectation of a
    ratio, so agreement is a real check rather than a tautology."""
    y_true = _labels([1 / 6] * 6, 1200, SIX)
    result = uniform_random(y_true, SIX, n_draws=1000, seed=7)

    assert result.ci_low <= result.analytic_macro_f1 <= result.ci_high
    assert result.macro_f1 == pytest.approx(result.analytic_macro_f1, abs=0.01)


def test_uniform_floor_at_k4_is_a_quarter():
    y_true = _labels([1 / 4] * 4, 1200, FOUR)
    result = uniform_random(y_true, FOUR, n_draws=500, seed=3)
    assert result.macro_f1 == pytest.approx(0.25, abs=0.02)


def test_majority_baseline_is_deterministic_and_collapses():
    y_true = _labels([1 / 6] * 6, 600, SIX)
    source = ["angry"] * 100 + ["sad"] * 10
    result = majority_class(y_true, SIX, source)

    assert result.details["majority_class"] == "angry"
    assert result.macro_f1 == pytest.approx(result.analytic_macro_f1, abs=0.01)
    assert result.n_draws == 0
    # Exactly one class is ever predicted.
    assert sum(1 for v in result.per_class_f1.values() if v > 0) == 1


def test_stratified_tracks_the_source_prior():
    y_true = _labels([1 / 6] * 6, 900, SIX)
    source = _labels([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], 900, SIX, seed=5)
    result = stratified_random(y_true, SIX, source, n_draws=400, seed=11)
    assert result.macro_f1 == pytest.approx(result.analytic_macro_f1, abs=0.02)
    assert result.ci_low < result.macro_f1 < result.ci_high


def test_all_floors_are_ordered_as_expected():
    """Majority is the collapse floor and must sit below the random floors."""
    y_true = _labels([1 / 6] * 6, 900, SIX)
    source = _labels([1 / 6] * 6, 900, SIX, seed=2)
    floors = all_floors(y_true, SIX, source, n_draws=300, seed=1)

    assert floors["majority"].macro_f1 < floors["uniform_random"].macro_f1
    assert floors["majority"].macro_f1 < floors["stratified_random"].macro_f1


def test_floors_use_the_realised_target_distribution_not_a_uniform_assumption():
    skewed = _labels([0.7, 0.1, 0.1, 0.1], 800, FOUR)
    balanced = _labels([0.25] * 4, 800, FOUR, seed=4)
    source = _labels([0.25] * 4, 800, FOUR, seed=9)

    a = uniform_random(skewed, FOUR, n_draws=200, seed=1).macro_f1
    b = uniform_random(balanced, FOUR, n_draws=200, seed=1).macro_f1
    assert a != pytest.approx(b, abs=0.01)


# -- statistics ------------------------------------------------------------
def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = [SIX[i] for i in rng.integers(0, 6, 300)]
    y_pred = [SIX[i] for i in rng.integers(0, 6, 300)]

    ci = bootstrap_ci(
        y_true, y_pred, lambda a, b: macro_f1(a, b, SIX), n_resamples=300, seed=0
    )
    assert ci.low <= ci.point <= ci.high
    assert 0.0 <= ci.low < ci.high <= 1.0


def test_bootstrap_is_deterministic_given_a_seed():
    y_true = ["angry"] * 50 + ["sad"] * 50
    y_pred = ["angry"] * 60 + ["sad"] * 40
    kwargs = dict(metric=lambda a, b: macro_f1(a, b, SIX), n_resamples=200, seed=3)
    assert bootstrap_ci(y_true, y_pred, **kwargs) == bootstrap_ci(y_true, y_pred, **kwargs)


def test_wilcoxon_detects_a_consistent_difference():
    a = [0.40, 0.42, 0.38, 0.45, 0.41, 0.39]
    b = [0.30, 0.31, 0.29, 0.33, 0.30, 0.28]
    result = wilcoxon_signed_rank(a, b, name="a_vs_b")
    assert result.p_value < 0.05
    assert result.median_difference > 0
    assert result.n_pairs == 6


def test_wilcoxon_on_identical_inputs_does_not_raise():
    result = wilcoxon_signed_rank([0.3] * 5, [0.3] * 5)
    assert result.p_value == 1.0
    assert result.median_difference == 0.0


def test_holm_bonferroni_is_stricter_than_raw_and_monotone():
    corrected = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.2}, alpha=0.05)
    names = [name for name, *_ in corrected]
    assert names == ["a", "b", "c"]

    adjusted = [adj for _, _, adj, _ in corrected]
    assert adjusted == sorted(adjusted)  # monotone
    for _, raw, adj, _ in corrected:
        assert adj >= raw

    # b would pass uncorrected at 0.04 but not after correction.
    assert corrected[1][3] is False


def test_holm_bonferroni_handles_no_comparisons():
    assert holm_bonferroni({}) == []


# -- the sample-rate guard -------------------------------------------------
def test_assert_target_sample_rate_rejects_48k():
    from ser.config import load_config
    from ser.features.audio import assert_target_sample_rate

    config = load_config()
    assert_target_sample_rate(16000, config)
    with pytest.raises(ValueError, match="!= required 16000"):
        assert_target_sample_rate(48000, config)


def test_extractor_encode_rejects_non_16k_audio():
    """48 kHz would not error inside the model -- it would silently encode
    time-distorted speech. RAVDESS ships at 48 kHz, so this must be structural."""
    from ser.features.ssl import SSLExtractor

    extractor = SSLExtractor(
        backbone="hubert",
        checkpoint="test",
        n_layers=13,
        hidden_dim=768,
        n_segments=8,
        model=None,
        feature_extractor=None,
        input_normalised=True,
        expected_sample_rate=16000,
    )
    with pytest.raises(ValueError, match="expects 16000 Hz, got 48000"):
        extractor.encode(np.zeros(16000, dtype=np.float32), 48000)


def test_48k_file_is_resampled_to_16k_preserving_duration(tmp_path):
    import soundfile as sf

    from ser.config import load_config
    from ser.features.audio import load_audio

    config = load_config()
    path = tmp_path / "loud48.wav"
    sf.write(path, np.zeros(48000 * 2, dtype="float32"), 48000)  # 2 seconds

    waveform = load_audio(path, config)
    assert len(waveform) == pytest.approx(16000 * 2, rel=0.01)


def test_load_audio_refuses_a_disabled_target_rate(tmp_path):
    """Loading at native rate would mix 48 kHz RAVDESS with 16 kHz CREMA-D."""
    import copy

    import soundfile as sf
    import yaml

    from ser.config import load_config
    from ser.features.audio import load_audio

    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["features"]["sample_rate"] = 0
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(str(path))

    audio_path = tmp_path / "a.wav"
    sf.write(audio_path, np.zeros(1000, dtype="float32"), 48000)

    with pytest.raises(ValueError, match="features.sample_rate must be set"):
        load_audio(audio_path, config)
