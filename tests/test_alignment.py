"""Phase 5: the alignment ladder, its numerics, and blending."""

from __future__ import annotations

import numpy as np
import pytest

from ser.alignment import (
    LADDER,
    CoralAlignment,
    MeanShiftAlignment,
    MKMMDAlignment,
    NoAlignment,
    ZScoreAlignment,
    build_alignment,
)
from ser.blending import (
    BLENDABLE_ALIGNMENTS,
    blend,
    blend_groupwise,
    enumerate_blending,
    group_assignments,
)
from ser.config import load_config
from ser.mmd import median_bandwidth, multi_kernel_mmd2
from ser.numerics import (
    MAX_CONDITION_NUMBER,
    SingularCovariance,
    covariance,
    diagnose,
    inverse_sqrt_psd,
    ledoit_wolf_covariance,
    require_float64,
    shrink,
    upcast,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    d = 24
    A = rng.standard_normal((200, d)) @ rng.standard_normal((d, d)) * 0.5
    B = rng.standard_normal((250, d)) @ rng.standard_normal((d, d)) + 3.0
    return A.astype(np.float64), B.astype(np.float64)


def _ids(n, prefix="u"):
    return [f"{prefix}{i}" for i in range(n)]


# -- dtype discipline ------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_fit_rejects_low_precision(dtype, data):
    """Caches are float16. Fitting a covariance in half precision loses accuracy
    without announcing it, so the contract is asserted, not assumed."""
    A, B = data
    alignment = CoralAlignment(eps=1e-3)
    with pytest.raises(TypeError, match="must be float64"):
        alignment.fit(A.astype(dtype), B, _ids(len(B)), _ids(len(A), "s"))


def test_require_float64_refuses_to_silently_upcast():
    with pytest.raises(TypeError, match="must be float64"):
        require_float64(np.zeros(4, dtype=np.float32), "X")
    require_float64(np.zeros(4, dtype=np.float64), "X")


def test_require_float64_rejects_non_finite():
    bad = np.array([1.0, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        require_float64(bad, "X")


def test_upcast_is_the_sanctioned_promotion():
    assert upcast(np.zeros(4, dtype=np.float16)).dtype == np.float64


# -- covariance diagnostics ------------------------------------------------
def test_diagnose_reports_rank_deficiency():
    """n < d is the normal case here: 768 dims from ~1000 samples."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 40))
    diagnostics = diagnose(covariance(X), n_samples=10)

    assert diagnostics.rank_deficient
    assert diagnostics.numerical_rank <= 9
    assert diagnostics.effective_rank < 40


def test_effective_rank_is_lower_when_energy_concentrates():
    rng = np.random.default_rng(2)
    spread = covariance(rng.standard_normal((500, 20)))
    concentrated = np.diag([100.0] + [1e-4] * 19)

    assert diagnose(concentrated, 500).effective_rank < diagnose(spread, 500).effective_rank


def test_shrinkage_is_scale_aware():
    """The original added a fixed 1e-5*I regardless of feature scale, so the same
    nominal epsilon meant different things per backbone and layer."""
    cov = np.diag([100.0, 100.0])
    scaled = np.diag([0.01, 0.01])

    a = shrink(cov, 0.1)[0, 0] / cov[0, 0]
    b = shrink(scaled, 0.1)[0, 0] / scaled[0, 0]
    assert a == pytest.approx(b)


def test_shrinkage_improves_conditioning_monotonically():
    rng = np.random.default_rng(3)
    cov = covariance(rng.standard_normal((30, 60)))
    conditions = [
        diagnose(shrink(cov, eps), 30).condition_number
        for eps in (1e-4, 1e-3, 1e-2, 1e-1)
    ]
    assert conditions == sorted(conditions, reverse=True)


def test_singular_matrix_fails_loudly_rather_than_pseudo_inverting():
    """A pseudo-inverse here would produce a number that looks like a result."""
    singular = np.zeros((5, 5))
    singular[0, 0] = 1.0
    with pytest.raises(SingularCovariance, match="condition number"):
        inverse_sqrt_psd(singular, label="test")


def test_ledoit_wolf_is_parameter_free_and_well_conditioned():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((40, 80))
    cov, shrinkage = ledoit_wolf_covariance(X)
    assert 0.0 <= shrinkage <= 1.0
    assert diagnose(cov, 40).condition_number < MAX_CONDITION_NUMBER


# -- the rungs -------------------------------------------------------------
def test_none_is_the_identity(data):
    A, B = data
    alignment = NoAlignment().fit(A, B, _ids(len(B)), _ids(len(A), "s"))
    np.testing.assert_allclose(alignment.transform(A), A)


def test_zscore_gives_zero_mean_unit_variance_per_corpus(data):
    A, B = data
    alignment = ZScoreAlignment().fit(A, B, _ids(len(B)), _ids(len(A), "s"))

    for X, domain in ((A, "source"), (B, "target")):
        out = alignment.transform(X, domain=domain)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-9)


def test_zscore_survives_a_constant_dimension():
    """Dividing by a ~0 std would manufacture huge values from numerical noise."""
    rng = np.random.default_rng(5)
    A = rng.standard_normal((50, 4))
    A[:, 2] = 7.0
    B = rng.standard_normal((60, 4))

    out = ZScoreAlignment().fit(A, B, _ids(60), _ids(50, "s")).transform(A)
    assert np.isfinite(out).all()


def test_mean_shift_moves_the_source_mean_onto_the_target(data):
    A, B = data
    alignment = MeanShiftAlignment().fit(A, B, _ids(len(B)), _ids(len(A), "s"))
    np.testing.assert_allclose(
        alignment.transform(A).mean(axis=0), B.mean(axis=0), atol=1e-9
    )


def test_mean_shift_leaves_the_target_alone(data):
    A, B = data
    alignment = MeanShiftAlignment().fit(A, B, _ids(len(B)), _ids(len(A), "s"))
    np.testing.assert_allclose(alignment.transform(B, domain="target"), B)


def test_coral_with_identical_domains_is_approximately_identity():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((400, 12))
    alignment = CoralAlignment(eps=1e-6).fit(X, X.copy(), _ids(400), _ids(400, "s"))
    np.testing.assert_allclose(alignment.transform(X), X, atol=1e-6)


def test_coral_matches_the_target_covariance():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((600, 8))
    B = rng.standard_normal((600, 8)) * 4.0 + 2.0

    aligned = CoralAlignment(eps=1e-8).fit(A, B, _ids(600), _ids(600, "s")).transform(A)
    np.testing.assert_allclose(covariance(aligned), covariance(B), rtol=0.2, atol=0.2)


def test_coral_requires_regularisation():
    """768 dimensions from ~1000 samples is singular; there is no unregularised
    variant to fall back on."""
    with pytest.raises(ValueError, match="requires regularisation"):
        CoralAlignment()
    with pytest.raises(ValueError, match="not both"):
        CoralAlignment(eps=1e-3, ledoit_wolf=True)


def test_coral_records_eps_and_conditioning(data):
    """An unreported eps makes CORAL unreproducible."""
    A, B = data
    alignment = CoralAlignment(eps=1e-2).fit(A, B, _ids(len(B)), _ids(len(A), "s"))

    assert alignment.diagnostics["eps"] == 1e-2
    assert alignment.diagnostics["variant"] == "shrinkage"
    fields = alignment.row_fields()
    assert fields["cov_condition_number"] > 0
    assert fields["cov_effective_rank"] > 0


def test_coral_ledoit_wolf_variant_records_its_shrinkage(data):
    A, B = data
    alignment = CoralAlignment(ledoit_wolf=True).fit(A, B, _ids(len(B)), _ids(len(A), "s"))
    assert alignment.diagnostics["variant"] == "ledoit_wolf"
    assert 0.0 <= alignment.diagnostics["ledoit_wolf_shrinkage"] <= 1.0


# -- MMD -------------------------------------------------------------------
def test_mmd_of_a_distribution_with_itself_is_near_zero():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((200, 6))
    Y = rng.standard_normal((200, 6))
    assert abs(multi_kernel_mmd2(X, Y)) < 0.05


def test_mmd_detects_a_shifted_distribution():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((200, 6))
    Y = rng.standard_normal((200, 6)) + 5.0
    assert multi_kernel_mmd2(X, Y) > 0.5


def test_median_bandwidth_is_positive_and_reproducible():
    rng = np.random.default_rng(10)
    X, Y = rng.standard_normal((80, 5)), rng.standard_normal((90, 5))
    assert median_bandwidth(X, Y, seed=1) == median_bandwidth(X, Y, seed=1)
    assert median_bandwidth(X, Y, seed=1) > 0


def test_median_bandwidth_of_identical_points_does_not_divide_by_zero():
    X = np.ones((10, 3))
    assert median_bandwidth(X, X) == 1.0


@pytest.mark.parametrize("diagonal", [True, False])
def test_mkmmd_reduces_mmd_and_records_lambda(config, data, diagonal):
    A, B = data
    name = "mkmmd_diag" if diagonal else "mkmmd_full"
    alignment = build_alignment(name, config, lam=0.01, seed=0)
    alignment.fit(A, B, _ids(len(B)), _ids(len(A), "s"))

    assert alignment.diagnostics["lambda"] == 0.01
    assert alignment.diagnostics["diagonal"] is diagonal
    assert alignment.diagnostics["final_mmd2"] < alignment.diagnostics["initial_mmd2"]


def test_mkmmd_diagonal_has_far_fewer_parameters(config, data):
    """768 against ~590k. If diagonal matches full, the alignment was only
    rescaling dimensions -- a finding, not a detail."""
    A, B = data
    diag = build_alignment("mkmmd_diag", config, lam=1.0, seed=0)
    full = build_alignment("mkmmd_full", config, lam=1.0, seed=0)
    diag.fit(A, B, _ids(len(B)), _ids(len(A), "s"))
    full.fit(A, B, _ids(len(B)), _ids(len(A), "s"))

    assert diag.result.W.ndim == 1
    assert full.result.W.ndim == 2
    assert diag.result.W.size < full.result.W.size


def test_large_lambda_keeps_the_map_near_identity(config, data):
    """The identity anchor means lambda -> infinity degrades towards `none`
    rather than towards noise."""
    A, B = data
    weak = build_alignment("mkmmd_full", config, lam=1e-3, seed=0)
    strong = build_alignment("mkmmd_full", config, lam=1e3, seed=0)
    weak.fit(A, B, _ids(len(B)), _ids(len(A), "s"))
    strong.fit(A, B, _ids(len(B)), _ids(len(A), "s"))

    assert (
        strong.diagnostics["W_deviation_from_identity"]
        < weak.diagnostics["W_deviation_from_identity"]
    )


def test_mkmmd_requires_a_lambda(config):
    with pytest.raises(ValueError, match="requires a lambda"):
        build_alignment("mkmmd_full", config)


# -- the fitted-indices contract -------------------------------------------
@pytest.mark.parametrize("method", LADDER)
def test_every_rung_records_what_it_was_fitted_on(config, data, method):
    A, B = data
    alignment = build_alignment(
        method, config, eps=1e-2, lam=1.0, seed=0
    )
    target_ids = _ids(len(B), "t")
    source_ids = _ids(len(A), "s")
    alignment.fit(A, B, target_ids, source_ids)

    assert alignment.fitted_on_indices == set(target_ids) | set(source_ids)


@pytest.mark.parametrize("method", LADDER)
def test_transform_never_grows_the_fitted_index_set(config, data, method):
    A, B = data
    alignment = build_alignment(method, config, eps=1e-2, lam=1.0, seed=0)
    alignment.fit(A, B, _ids(len(B), "t"), _ids(len(A), "s"))

    before = set(alignment.fitted_on_indices)
    alignment.transform(A)
    alignment.transform(B, domain="target")
    assert alignment.fitted_on_indices == before


def test_transform_before_fit_raises(config):
    with pytest.raises(RuntimeError, match="before fit"):
        NoAlignment().transform(np.zeros((2, 2)))


def test_unknown_rung_raises(config):
    with pytest.raises(ValueError, match="unknown alignment"):
        build_alignment("gfk", config)


# -- blending --------------------------------------------------------------
def test_alpha_endpoints_are_pure_aligned_and_pure_original():
    original = np.zeros((5, 3))
    aligned = np.ones((5, 3))
    np.testing.assert_allclose(blend(original, aligned, 1.0), aligned)
    np.testing.assert_allclose(blend(original, aligned, 0.0), original)
    np.testing.assert_allclose(blend(original, aligned, 0.5), np.full((5, 3), 0.5))


def test_blend_rejects_alpha_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        blend(np.zeros((2, 2)), np.ones((2, 2)), 1.5)


def test_groupwise_blending_applies_per_group_alphas():
    original = np.zeros((4, 6))
    aligned = np.ones((4, 6))
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = blend_groupwise(original, aligned, groups, [0.0, 1.0])

    np.testing.assert_allclose(out[:, :3], 0.0)
    np.testing.assert_allclose(out[:, 3:], 1.0)


def test_group_assignments_cluster_dimensions_not_indices():
    rng = np.random.default_rng(11)
    signal = rng.standard_normal((100, 1))
    # Dimensions 0 and 3 move together; 1 and 2 are independent noise.
    X = np.column_stack(
        [signal[:, 0], rng.standard_normal(100), rng.standard_normal(100), signal[:, 0]]
    )
    groups = group_assignments(X, 2, seed=0)
    assert groups[0] == groups[3]


def test_blending_is_not_enumerated_where_it_is_a_mathematical_identity(config):
    """The original enumerated these anyway and reported 972 runs for 756."""
    for method in ("none", "zscore"):
        assert enumerate_blending(method, config) == [
            {"blending": "none", "blend_alpha": None, "n_groups": None}
        ]
    for method in BLENDABLE_ALIGNMENTS:
        assert len(enumerate_blending(method, config)) > 1
