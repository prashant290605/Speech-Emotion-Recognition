"""Numerical conditioning: dtype discipline and covariance diagnostics.

Two facts drive this module.

**Caches are float16.** That is a storage decision, and float16 has ~3 decimal
digits of precision. Forming a 768x768 covariance, whitening it, or taking a
matrix square root in float16 -- or even float32 -- loses accuracy in ways that
do not announce themselves. Every alignment upcasts to float64 first, and
:func:`require_float64` asserts it at the entry to each fit rather than trusting
callers.

**Every covariance here is rank-deficient.** d=768 with ~1000 source-train
samples gives a sample covariance of rank at most n-1, and the split sizes make
that a live constraint rather than a theoretical one. So:

* conditioning is *measured* and recorded on the run row, not assumed;
* regularisation is mandatory, not optional;
* a matrix that is still singular after regularisation makes the run **fail
  loudly**. Silently falling back to a pseudo-inverse would produce a number
  that looks like a result.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

__all__ = [
    "SingularCovariance",
    "CovarianceDiagnostics",
    "require_float64",
    "upcast",
    "covariance",
    "diagnose",
    "shrink",
    "ledoit_wolf_covariance",
    "inverse_sqrt_psd",
    "sqrt_psd",
]

# A regularised covariance worse-conditioned than this is treated as singular to
# working precision. float64 carries ~16 digits; 1e12 leaves a safety margin
# before whitening amplifies noise into the result.
MAX_CONDITION_NUMBER = 1e12


class SingularCovariance(RuntimeError):
    """A covariance is singular to working precision after regularisation."""


def require_float64(array: np.ndarray, name: str) -> np.ndarray:
    """Assert an array is float64. Raises rather than silently upcasting.

    Silently upcasting would hide the mistake it exists to catch: a caller that
    handed over a float16 cache slice and believed it was fitting in double
    precision.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.dtype != np.float64:
        raise TypeError(
            f"{name} must be float64 before any covariance or whitening step, "
            f"got {array.dtype}. Caches are float16; call upcast() explicitly."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def upcast(array: np.ndarray) -> np.ndarray:
    """float16/32 cache slice -> float64. The one sanctioned promotion."""
    return np.asarray(array, dtype=np.float64)


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """What was actually formed, recorded on the run row."""

    n_samples: int
    n_features: int
    condition_number: float
    effective_rank: float
    numerical_rank: int
    min_eigenvalue: float
    max_eigenvalue: float
    trace: float
    rank_deficient: bool

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def covariance(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    """Unbiased feature covariance of a float64, already-centred-or-not matrix."""
    require_float64(X, name)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"{name} needs at least 2 samples, got {X.shape[0]}")
    centred = X - X.mean(axis=0, keepdims=True)
    return (centred.T @ centred) / float(X.shape[0] - 1)


def diagnose(matrix: np.ndarray, n_samples: int) -> CovarianceDiagnostics:
    """Condition number, effective rank, and numerical rank of a PSD matrix.

    Effective rank is the entropy-based measure of Roy & Vetterli:
    ``exp(-sum p_i log p_i)`` over the normalised eigenvalue spectrum. Unlike a
    thresholded count it is continuous, so it distinguishes "768 directions with
    energy spread evenly" from "768 directions where five carry everything" --
    which is exactly the difference that decides whether whitening is stable.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    total = float(eigenvalues.sum())

    condition = float(largest / smallest) if smallest > 0 else float("inf")

    tolerance = largest * max(matrix.shape) * np.finfo(np.float64).eps
    numerical_rank = int((eigenvalues > tolerance).sum())

    if total > 0:
        p = eigenvalues / total
        nonzero = p[p > 0]
        effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    else:
        effective_rank = 0.0

    return CovarianceDiagnostics(
        n_samples=int(n_samples),
        n_features=int(matrix.shape[0]),
        condition_number=condition,
        effective_rank=effective_rank,
        numerical_rank=numerical_rank,
        min_eigenvalue=smallest,
        max_eigenvalue=largest,
        trace=total,
        rank_deficient=numerical_rank < matrix.shape[0],
    )


def shrink(cov: np.ndarray, eps: float) -> np.ndarray:
    """Scale-aware shrinkage: ``Cov + eps * trace(Cov)/d * I``.

    Scale-aware matters. The original study added a fixed ``1e-5 * I`` regardless
    of feature scale, so the same nominal epsilon meant something different for
    every backbone and layer. Anchoring to ``trace/d`` -- the mean eigenvalue --
    makes ``eps`` a comparable quantity across the whole grid.
    """
    if eps <= 0:
        raise ValueError(f"shrinkage eps must be positive, got {eps}")
    d = cov.shape[0]
    mean_eigenvalue = float(np.trace(cov)) / d
    if mean_eigenvalue <= 0:
        raise SingularCovariance("covariance has non-positive trace")
    return cov + eps * mean_eigenvalue * np.eye(d, dtype=np.float64)


def ledoit_wolf_covariance(X: np.ndarray, *, name: str = "X") -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrunk covariance. Parameter-free.

    Returns ``(covariance, shrinkage)``. The shrinkage coefficient is chosen
    analytically to minimise expected squared error, which is why this is worth
    carrying alongside the searched-epsilon variant: if the two agree, the
    epsilon search is not doing anything a principled estimator would not.
    """
    require_float64(X, name)
    from sklearn.covariance import LedoitWolf

    estimator = LedoitWolf(assume_centered=False).fit(X)
    return np.asarray(estimator.covariance_, dtype=np.float64), float(estimator.shrinkage_)


def _check_conditioning(diagnostics: CovarianceDiagnostics, label: str) -> None:
    if not np.isfinite(diagnostics.condition_number) or (
        diagnostics.condition_number > MAX_CONDITION_NUMBER
    ):
        raise SingularCovariance(
            f"{label}: condition number {diagnostics.condition_number:.3e} exceeds "
            f"{MAX_CONDITION_NUMBER:.0e} after regularisation "
            f"(n={diagnostics.n_samples}, d={diagnostics.n_features}, "
            f"effective rank {diagnostics.effective_rank:.1f}). "
            "Increase shrinkage or use the Ledoit-Wolf variant. Refusing to "
            "pseudo-invert: the result would look like a number."
        )


def _symmetric_power(matrix: np.ndarray, power: float, label: str) -> np.ndarray:
    """``matrix ** power`` via eigendecomposition, with a conditioning check."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    diagnostics = diagnose(matrix, n_samples=0)
    _check_conditioning(diagnostics, label)

    if eigenvalues.min() <= 0:
        raise SingularCovariance(
            f"{label}: smallest eigenvalue {eigenvalues.min():.3e} is not positive; "
            "the matrix is not positive definite after regularisation."
        )
    return (eigenvectors * np.power(eigenvalues, power)) @ eigenvectors.T


def inverse_sqrt_psd(matrix: np.ndarray, *, label: str = "covariance") -> np.ndarray:
    """``matrix ** -0.5`` for a symmetric positive-definite matrix."""
    return _symmetric_power(matrix, -0.5, label)


def sqrt_psd(matrix: np.ndarray, *, label: str = "covariance") -> np.ndarray:
    """``matrix ** 0.5`` for a symmetric positive-semidefinite matrix."""
    return _symmetric_power(matrix, 0.5, label)
