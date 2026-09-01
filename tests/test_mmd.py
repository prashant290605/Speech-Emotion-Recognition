"""Properties of the MMD measurement rule."""

from __future__ import annotations

import numpy as np

from ser.mmd import median_bandwidth, multi_kernel_mmd2


def test_median_heuristic_rbf_mmd_is_invariant_to_global_scaling():
    """Scaling both samples scales the median bandwidth by the same factor.

    This is the exact property stated in the manuscript. It does not say that
    arbitrary learned affine maps leave MMD invariant; it covers the global
    rescaling case only.
    """
    rng = np.random.default_rng(7)
    source = rng.normal(size=(80, 5))
    target = rng.normal(loc=0.4, size=(75, 5))
    scale = 3.75

    bandwidth = median_bandwidth(source, target, seed=11)
    scaled_bandwidth = median_bandwidth(scale * source, scale * target, seed=11)
    np.testing.assert_allclose(
        scaled_bandwidth, abs(scale) * bandwidth, rtol=1e-12, atol=1e-12
    )

    original = multi_kernel_mmd2(source, target, bandwidth=bandwidth, seed=11)
    scaled = multi_kernel_mmd2(
        scale * source, scale * target, bandwidth=scaled_bandwidth, seed=11
    )
    np.testing.assert_allclose(scaled, original, rtol=1e-11, atol=1e-11)


def test_source_only_median_bandwidth_is_scale_equivariant():
    rng = np.random.default_rng(8)
    source = rng.normal(size=(90, 4))
    bandwidth = median_bandwidth(source, seed=2)
    scaled = median_bandwidth(-2.5 * source, seed=2)
    np.testing.assert_allclose(scaled, 2.5 * bandwidth, rtol=1e-12, atol=1e-12)
