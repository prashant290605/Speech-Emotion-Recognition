"""Blending between original and aligned features.

Three modes:

``none``
    Use the aligned features as-is.
``scalar``
    ``α · aligned + (1 − α) · original``, one α for the whole feature vector.
``gaa``
    k-means over feature *dimensions* into g groups, one α per group.

**α is selected on `source_val`, never on target test.** Selection needs a
classifier, so it happens in Phase 6/7; this module only provides the transform
and the enumeration rule.

Worth recording for the paper: in the original study α was **never searched**.
`fwaa` and `gaa` derived it from the magnitude of ``|X_aligned − X_orig|`` and
only `scalar` took an α argument, so its Table 3 compares three blending modes
at three different unspecified α values. Selecting α on `source_val` is a change
in kind, not a correction to the selection surface.

Enumeration rule: blending only applies when the alignment actually moved the
features in a way interpolation can trade off against -- ``mean_shift``,
``coral``, ``mkmmd_*``. With ``none`` and ``zscore`` the three modes are
mathematically identical, and enumerating them anyway is how the original grid
reported 972 runs when only 756 were distinct.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "BLENDABLE_ALIGNMENTS",
    "blend",
    "group_assignments",
    "blend_groupwise",
    "enumerate_blending",
]

BLENDABLE_ALIGNMENTS = ("mean_shift", "coral", "mkmmd_diag", "mkmmd_full")


def blend(original: np.ndarray, aligned: np.ndarray, alpha: float) -> np.ndarray:
    """``α · aligned + (1 − α) · original``.

    α=1 is pure aligned, α=0 pure original. Both are asserted in tests, because
    an off-by-one in this direction would silently invert the entire blending
    axis.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    original = np.asarray(original, dtype=np.float64)
    aligned = np.asarray(aligned, dtype=np.float64)
    if original.shape != aligned.shape:
        raise ValueError(
            f"shape mismatch: original {original.shape} vs aligned {aligned.shape}"
        )
    return alpha * aligned + (1.0 - alpha) * original


def group_assignments(
    reference: np.ndarray, n_groups: int, *, seed: int = 0
) -> np.ndarray:
    """k-means over feature *dimensions*, returning a group id per dimension.

    Dimensions are clustered by their behaviour across samples, so a group is a
    set of dimensions that move together -- not an arbitrary contiguous slice.
    The original used contiguous slices of the 768 dimensions, which groups
    dimensions by index rather than by anything meaningful.
    """
    from sklearn.cluster import KMeans

    reference = np.asarray(reference, dtype=np.float64)
    n_features = reference.shape[1]
    if n_groups > n_features:
        raise ValueError(f"n_groups {n_groups} exceeds {n_features} features")

    # Cluster the transposed matrix: each row is now one dimension's profile.
    profiles = reference.T
    kmeans = KMeans(n_clusters=n_groups, random_state=seed, n_init=10)
    return kmeans.fit_predict(profiles)


def blend_groupwise(
    original: np.ndarray,
    aligned: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    """Per-group α. ``groups[j]`` indexes ``alphas`` for feature ``j``."""
    original = np.asarray(original, dtype=np.float64)
    aligned = np.asarray(aligned, dtype=np.float64)
    groups = np.asarray(groups)

    if groups.shape[0] != original.shape[1]:
        raise ValueError(
            f"groups has {groups.shape[0]} entries for {original.shape[1]} features"
        )
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1:
        raise ValueError("alphas must be one-dimensional")
    if groups.max(initial=-1) >= len(alphas):
        raise ValueError("a group id has no corresponding alpha")
    if not ((alphas >= 0.0) & (alphas <= 1.0)).all():
        raise ValueError("every alpha must lie in [0, 1]")

    per_feature = alphas[groups]
    return per_feature * aligned + (1.0 - per_feature) * original


def enumerate_blending(alignment: str, config) -> List[dict]:
    """Distinct (mode, α) combinations for one alignment rung.

    Returns a single no-op entry for alignments where blending is a mathematical
    identity, which is what keeps the grid free of the original's duplicates.
    """
    if alignment not in BLENDABLE_ALIGNMENTS:
        return [{"blending": "none", "blend_alpha": None, "n_groups": None}]

    out: List[dict] = []
    for mode in config.blending.modes:
        if mode == "none":
            out.append({"blending": "none", "blend_alpha": None, "n_groups": None})
        elif mode == "scalar":
            for alpha in config.blending.alpha_grid:
                out.append(
                    {"blending": "scalar", "blend_alpha": float(alpha), "n_groups": None}
                )
        elif mode == "gaa":
            # Per-group alphas are selected inside the run on source_val, so
            # they are an output (hyperparams_json), not a grid coordinate.
            out.append(
                {
                    "blending": "gaa",
                    "blend_alpha": None,
                    "n_groups": config.blending.n_groups,
                }
            )
    return out
