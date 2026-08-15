"""Multi-kernel MMD, and the affine map that minimises it.

The operator, stated once and completely, because the paper must state it and
the original study's did not:

    Learn W, b minimising   MMD²_k(W·X_src + b, X_target_adapt) + λ‖W − I‖²_F

with k a **sum** of Gaussian RBF kernels at bandwidths
``{0.25, 0.5, 1, 2, 4} × σ_median``, where ``σ_median`` is the median pairwise
Euclidean distance over the pooled sample. Optimised with Adam for a fixed step
budget. Fitted on ``source_train`` and ``target_adapt`` only.

Two things make this honest rather than decorative:

**λ is a searched axis over a wide range.** W is 768×768 ≈ 590k parameters fitted
on ~1000 samples. Without strong regularisation it interpolates the source into
the target and generalises to nothing. The identity anchor ‖W − I‖²_F means
λ→∞ recovers the identity map, so the ladder degrades gracefully towards `none`
rather than towards noise.

**A diagonal-W variant is a separate rung.** Diagonal W has 768 parameters, a
768× reduction. If it matches full W, the alignment was only ever rescaling
individual dimensions, and that is a finding about how much structure the method
uses -- not a detail.

``mean_shift`` is the same family: linear-kernel MMD reduces to ‖μ_s − μ_t‖²,
whose minimiser is exactly ``X + (μ_t − μ_s)``. That is why the ladder is an
ordered series rather than a set of arbitrary points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .numerics import require_float64

__all__ = [
    "median_bandwidth",
    "multi_kernel_mmd2",
    "marginal_mmd",
    "MMDFitResult",
    "fit_affine_mmd",
]


def median_bandwidth(X: np.ndarray, Y: np.ndarray, *, max_samples: int = 512,
                     seed: int = 0) -> float:
    """Median pairwise Euclidean distance over the pooled sample.

    Subsampled for tractability; seeded so the bandwidth is reproducible. A zero
    median (identical points) falls back to 1.0 rather than producing a
    degenerate kernel.
    """
    rng = np.random.default_rng(seed)
    pooled = np.vstack([X, Y])
    if pooled.shape[0] > max_samples:
        pooled = pooled[rng.choice(pooled.shape[0], max_samples, replace=False)]

    squared = _pairwise_sq_dists_numpy(pooled, pooled)
    upper = squared[np.triu_indices_from(squared, k=1)]
    median = float(np.sqrt(np.median(upper))) if upper.size else 0.0
    return median if median > 0 else 1.0


def _pairwise_sq_dists_numpy(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x2 = (X * X).sum(axis=1)[:, None]
    y2 = (Y * Y).sum(axis=1)[None, :]
    return np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)


def multi_kernel_mmd2(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    multipliers: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    bandwidth: Optional[float] = None,
    seed: int = 0,
) -> float:
    """Unbiased MMD² under a sum of Gaussian kernels.

    Unbiased means the diagonal self-similarity terms are excluded, so the value
    is an estimate of the population MMD² and can legitimately go slightly
    negative when the two samples are drawn from the same distribution. Clamping
    it at zero would hide exactly the case worth seeing.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if bandwidth is None:
        bandwidth = median_bandwidth(X, Y, seed=seed)

    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        raise ValueError("MMD needs at least 2 samples per sample set")

    xx = _pairwise_sq_dists_numpy(X, X)
    yy = _pairwise_sq_dists_numpy(Y, Y)
    xy = _pairwise_sq_dists_numpy(X, Y)

    total = 0.0
    for multiplier in multipliers:
        gamma = 1.0 / (2.0 * (multiplier * bandwidth) ** 2)
        k_xx = np.exp(-gamma * xx)
        k_yy = np.exp(-gamma * yy)
        k_xy = np.exp(-gamma * xy)

        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        total += (
            k_xx.sum() / (n * (n - 1))
            + k_yy.sum() / (m * (m - 1))
            - 2.0 * k_xy.mean()
        )
    return float(total)


def marginal_mmd(
    X_source: np.ndarray,
    X_target: np.ndarray,
    config,
    *,
    max_samples: int = 512,
    seed: int = 0,
) -> float:
    """Marginal MMD² between two feature sets.

    This is the covariate-shift column of the Phase 9 decomposition. Collected
    for every rung before and after alignment, which costs nothing here and
    saves re-deriving it later.
    """
    rng = np.random.default_rng(seed)

    def subsample(X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] > max_samples:
            return X[rng.choice(X.shape[0], max_samples, replace=False)]
        return X

    a, b = subsample(X_source), subsample(X_target)
    return multi_kernel_mmd2(
        a, b, multipliers=tuple(config.alignment.mmd_bandwidth_multipliers), seed=seed
    )


@dataclass
class MMDFitResult:
    W: np.ndarray
    b: np.ndarray
    diagonal: bool
    lam: float
    bandwidth: float
    steps: int
    initial_objective: float
    final_objective: float
    initial_mmd2: float
    final_mmd2: float
    history: List[float]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.diagonal:
            return X * self.W + self.b
        return X @ self.W.T + self.b


def fit_affine_mmd(
    X_source: np.ndarray,
    X_target: np.ndarray,
    config,
    *,
    lam: float,
    diagonal: bool,
    seed: int = 0,
) -> MMDFitResult:
    """Learn (W, b) minimising multi-kernel MMD² plus the identity penalty.

    Minibatched: full kernel matrices over several thousand target samples would
    dominate the runtime for no gain in gradient quality. Batches are drawn with
    a seeded generator, so the fit is reproducible.

    W is initialised at the identity, so step 0 is exactly the `none` rung and
    any improvement is attributable to the optimisation rather than to the
    initialisation.
    """
    import torch

    require_float64(X_source, "X_source")
    require_float64(X_target, "X_target")

    torch.manual_seed(seed)
    d = X_source.shape[1]
    multipliers = tuple(config.alignment.mmd_bandwidth_multipliers)
    bandwidth = median_bandwidth(X_source, X_target, seed=seed)

    source = torch.from_numpy(X_source)
    target = torch.from_numpy(X_target)

    if diagonal:
        W = torch.ones(d, dtype=torch.float64, requires_grad=True)
    else:
        W = torch.eye(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(d, dtype=torch.float64, requires_grad=True)

    identity = torch.ones(d, dtype=torch.float64) if diagonal else torch.eye(
        d, dtype=torch.float64
    )

    optimiser = torch.optim.Adam(
        [W, b], lr=config.alignment.mmd_learning_rate
    )
    generator = torch.Generator().manual_seed(seed)
    batch = int(getattr(config.alignment, "mmd_batch_size", 256))

    def apply(X):
        return X * W + b if diagonal else X @ W.T + b

    def objective(a, t):
        mmd2 = _torch_multi_kernel_mmd2(a, t, multipliers, bandwidth)
        penalty = lam * ((W - identity) ** 2).sum()
        return mmd2 + penalty, mmd2

    history: List[float] = []
    initial_objective = initial_mmd2 = None

    for step in range(config.alignment.mmd_steps):
        s_idx = torch.randint(0, source.shape[0], (min(batch, source.shape[0]),),
                              generator=generator)
        t_idx = torch.randint(0, target.shape[0], (min(batch, target.shape[0]),),
                              generator=generator)

        optimiser.zero_grad()
        loss, mmd2 = objective(apply(source[s_idx]), target[t_idx])
        if step == 0:
            initial_objective = float(loss.item())
            initial_mmd2 = float(mmd2.item())
        loss.backward()
        optimiser.step()
        history.append(float(loss.item()))

    with torch.no_grad():
        s_idx = torch.randint(0, source.shape[0], (min(batch, source.shape[0]),),
                              generator=generator)
        t_idx = torch.randint(0, target.shape[0], (min(batch, target.shape[0]),),
                              generator=generator)
        final_loss, final_mmd2 = objective(apply(source[s_idx]), target[t_idx])

    return MMDFitResult(
        W=W.detach().numpy().copy(),
        b=b.detach().numpy().copy(),
        diagonal=diagonal,
        lam=lam,
        bandwidth=bandwidth,
        steps=config.alignment.mmd_steps,
        initial_objective=float(initial_objective),
        final_objective=float(final_loss.item()),
        initial_mmd2=float(initial_mmd2),
        final_mmd2=float(final_mmd2.item()),
        history=history,
    )


def _torch_multi_kernel_mmd2(X, Y, multipliers, bandwidth):
    import torch

    xx = torch.cdist(X, X) ** 2
    yy = torch.cdist(Y, Y) ** 2
    xy = torch.cdist(X, Y) ** 2

    n, m = X.shape[0], Y.shape[0]
    total = torch.zeros((), dtype=X.dtype)
    for multiplier in multipliers:
        gamma = 1.0 / (2.0 * (multiplier * bandwidth) ** 2)
        k_xx = torch.exp(-gamma * xx)
        k_yy = torch.exp(-gamma * yy)
        k_xy = torch.exp(-gamma * xy)

        k_xx = k_xx - torch.diag(torch.diag(k_xx))
        k_yy = k_yy - torch.diag(torch.diag(k_yy))

        total = total + (
            k_xx.sum() / (n * (n - 1))
            + k_yy.sum() / (m * (m - 1))
            - 2.0 * k_xy.mean()
        )
    return total
