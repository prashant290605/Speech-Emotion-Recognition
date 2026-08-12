from __future__ import annotations

import numpy as np


def _compute_covariance(X: np.ndarray) -> np.ndarray:
    """
    Compute the unbiased feature covariance matrix:
        cov = (X^T X) / (n - 1)

    X must already be centered and have shape (n_samples, n_features).
    """
    n_samples = X.shape[0]
    return (X.T @ X) / float(n_samples - 1)


def _svd_matrix_sqrt(matrix: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute matrix square-root factors from an SVD of a symmetric matrix.

    Returns:
        inv_sqrt: matrix^{-1/2}
        sqrt: matrix^{1/2}
    """
    U, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    singular_values = np.maximum(singular_values, eps)
    inv_sqrt = U @ np.diag(1.0 / np.sqrt(singular_values)) @ U.T
    sqrt = U @ np.diag(np.sqrt(singular_values)) @ U.T
    return inv_sqrt, sqrt


def coral_align(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Align source features (Xs) to target features (Xt) using CORAL.

    Args:
        Xs: (Ns, D) source features
        Xt: (Nt, D) target features
        eps: small regularization constant

    Returns:
        Xs_aligned: (Ns, D)
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    Xt = np.asarray(Xt, dtype=np.float64)

    if Xs.ndim != 2 or Xt.ndim != 2:
        raise ValueError("Xs and Xt must both be 2D arrays.")
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError("Xs and Xt must have the same feature dimension.")
    if Xs.shape[0] < 2 or Xt.shape[0] < 2:
        raise ValueError("Xs and Xt must each contain at least 2 samples.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    # Compute per-domain means so we can center features before covariance estimation.
    mean_s = Xs.mean(axis=0, keepdims=True)
    mean_t = Xt.mean(axis=0, keepdims=True)
    Xs_centered = Xs - mean_s
    Xt_centered = Xt - mean_t

    # Estimate regularized covariance matrices in feature space using the
    # explicit CORAL formula so the scaling matches the derivation exactly.
    d = Xs.shape[1]
    identity = np.eye(d, dtype=np.float64)
    Cs = _compute_covariance(Xs_centered) + eps * identity
    Ct = _compute_covariance(Xt_centered) + eps * identity

    # Use SVD for the whitening and recoloring transforms.
    # For symmetric PSD covariances, the left singular vectors define the same
    # orthogonal basis needed for matrix square roots.
    Cs_inv_sqrt, _ = _svd_matrix_sqrt(Cs, eps=eps)
    _, Ct_sqrt = _svd_matrix_sqrt(Ct, eps=eps)

    # Apply CORAL in centered space, then shift features to the target mean.
    Xs_aligned = Xs_centered @ Cs_inv_sqrt @ Ct_sqrt

    # CORAL can shrink variance severely when the source covariance is highly
    # rank-deficient relative to the feature dimension. Rescale the transformed
    # source so its total variance matches the target covariance trace.
    transformed_cov = np.cov(Xs_aligned, rowvar=False)
    transformed_trace = float(np.trace(transformed_cov))
    target_trace = float(np.trace(Ct))
    if transformed_trace <= 0.0 or not np.isfinite(transformed_trace):
        raise FloatingPointError("Invalid transformed covariance trace in CORAL.")
    scale = target_trace / transformed_trace
    Xs_aligned *= np.sqrt(scale)

    Xs_aligned = Xs_aligned + mean_t

    # Guard against silent numerical issues before returning to the caller.
    if not np.isfinite(Xs_aligned).all():
        raise FloatingPointError("CORAL produced NaN or Inf values.")

    return Xs_aligned


def test_coral() -> None:
    np.random.seed(42)
    Xs = np.random.randn(100, 768)
    Xt = np.random.randn(120, 768)

    Xs_aligned = coral_align(Xs, Xt)
    Cs_before = _compute_covariance(Xs - Xs.mean(axis=0, keepdims=True))
    Ct = _compute_covariance(Xt - Xt.mean(axis=0, keepdims=True))
    Cs_after = _compute_covariance(Xs_aligned - Xs_aligned.mean(axis=0, keepdims=True))

    print(f"Xs shape before: {Xs.shape}")
    print(f"Xt shape: {Xt.shape}")
    print(f"Xs aligned shape: {Xs_aligned.shape}")

    print(f"Source before mean/std: {Xs.mean():.6f} / {Xs.std():.6f}")
    print(f"Target mean/std: {Xt.mean():.6f} / {Xt.std():.6f}")
    print(f"Source after mean/std: {Xs_aligned.mean():.6f} / {Xs_aligned.std():.6f}")

    print(f"Source before covariance trace: {np.trace(Cs_before):.6f}")
    print(f"Target covariance trace: {np.trace(Ct):.6f}")
    print(f"Source after covariance trace: {np.trace(Cs_after):.6f}")
    print(
        "Covariance Frobenius difference before: "
        f"{np.linalg.norm(Cs_before - Ct, ord='fro'):.6f}"
    )
    print(
        "Covariance Frobenius difference after: "
        f"{np.linalg.norm(Cs_after - Ct, ord='fro'):.6f}"
    )

    print(
        "Mean comparison (target vs aligned): "
        f"{Xt.mean():.6f} vs {Xs_aligned.mean():.6f}"
    )
    print(
        "Std comparison (target vs aligned): "
        f"{Xt.std():.6f} vs {Xs_aligned.std():.6f}"
    )


if __name__ == "__main__":
    test_coral()
