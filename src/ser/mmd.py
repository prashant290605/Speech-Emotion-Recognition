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
    "null_mmd_scale",
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
    bandwidth: Optional[float] = None,
    max_samples: int = 512,
    seed: int = 0,
) -> float:
    """Marginal MMD² between two feature sets.

    ``bandwidth`` should be **fixed once** from the unaligned source/target pair
    and reused for every rung. Re-estimating it per rung makes the statistic
    scale-dependent: an alignment that merely shrinks the features shrinks the
    median pairwise distance too, the kernel widens to compensate, and the
    reported MMD falls without any distributions having moved closer.
    """
    rng = np.random.default_rng(seed)

    def subsample(X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] > max_samples:
            return X[rng.choice(X.shape[0], max_samples, replace=False)]
        return X

    a, b = subsample(X_source), subsample(X_target)
    return multi_kernel_mmd2(
        a,
        b,
        multipliers=tuple(config.alignment.mmd_bandwidth_multipliers),
        bandwidth=bandwidth,
        seed=seed,
    )


def kernel_saturation(
    X: np.ndarray, Y: np.ndarray, bandwidth: float, multipliers, *, max_samples: int = 300
) -> float:
    """Fraction of kernel mass lost to saturation at the widest bandwidth.

    Returns the mean kernel value at the widest multiplier. Near 1 means the
    kernel cannot discriminate (everything is 'close'); near 0 means it has
    saturated the other way and everything reads as infinitely far apart, so
    MMD collapses to ~0 regardless of how the distributions actually overlap.

    Measured because a bandwidth fixed on the unaligned pair is not valid for a
    rung that changes the feature scale: z-scoring 768 dimensions moves the
    median pairwise distance from 1.5 to 35.6, i.e. 16.8x the fixed bandwidth,
    at which point every kernel value is ~1e-4 and the statistic is an artefact.
    """
    X = np.asarray(X, dtype=np.float64)[:max_samples]
    Y = np.asarray(Y, dtype=np.float64)[:max_samples]
    widest = max(multipliers) * bandwidth
    gamma = 1.0 / (2.0 * widest**2)
    return float(np.exp(-gamma * _pairwise_sq_dists_numpy(X, Y)).mean())


def null_mmd_scale(
    X: np.ndarray,
    config,
    *,
    bandwidth: Optional[float] = None,
    n_repeats: int = 10,
    max_samples: int = 512,
    seed: int = 0,
) -> Dict[str, float]:
    """Typical same-distribution MMD² at this sample size.

    Splits ``X`` into two random halves repeatedly and measures MMD² between
    them. Both halves come from one distribution, so this is the discrepancy
    attributable purely to finite sampling -- the natural denominator for an
    effect size.

    **Deviation, deliberate.** The brief says to divide by the *mean* null MMD².
    With the unbiased estimator that mean is zero by construction: measured on
    real data it came out at −3.5e-4 against a spread of 9.2e-4, so the ratio
    would be unstable and would flip sign. The scale returned here is therefore
    the mean **absolute** null MMD² (8.8e-4 on that same data, agreeing with the
    spread to within 5%). ``std`` is returned alongside so the two can be
    compared directly.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    multipliers = tuple(config.alignment.mmd_bandwidth_multipliers)

    values = []
    for _ in range(n_repeats):
        order = rng.permutation(X.shape[0])
        half = min(X.shape[0] // 2, max_samples)
        if half < 2:
            raise ValueError("need at least 4 samples to form two halves")
        a = X[order[:half]]
        b = X[order[half : 2 * half]]
        values.append(
            multi_kernel_mmd2(a, b, multipliers=multipliers, bandwidth=bandwidth)
        )

    array = np.asarray(values, dtype=np.float64)
    return {
        "scale": float(np.abs(array).mean()),
        "signed_mean": float(array.mean()),
        "std": float(array.std()),
        "n_repeats": n_repeats,
        "n_per_half": int(half),
    }


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
    grad_norms: List[float] = None
    learning_rate: float = 0.0

    @property
    def final_grad_norm(self) -> float:
        return float(self.grad_norms[-1]) if self.grad_norms else float("nan")

    @property
    def converged(self) -> bool:
        """Objective flat over the last 10% of steps.

        Not a guarantee of a global optimum -- only that further steps at this
        learning rate would not move it much.
        """
        if len(self.history) < 20:
            return False
        tail = np.asarray(self.history[-max(10, len(self.history) // 10):])
        span = float(tail.max() - tail.min())
        scale = abs(float(np.mean(tail))) + 1e-12
        return span / scale < 0.01

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
    W_init: Optional[np.ndarray] = None,
    b_init: Optional[np.ndarray] = None,
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

    if W_init is not None:
        # Warm start. The MMD optimum is at least as good as CORAL's solution,
        # because CORAL's W lies in the feasible set -- so starting there makes
        # "did the optimiser converge?" separable from "is the objective
        # attainable?". Documented as a distinct condition, not a silent default.
        start = np.asarray(W_init, dtype=np.float64)
        expected = (d,) if diagonal else (d, d)
        if start.shape != expected:
            raise ValueError(f"W_init shape {start.shape} != expected {expected}")
        W = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    elif diagonal:
        W = torch.ones(d, dtype=torch.float64, requires_grad=True)
    else:
        W = torch.eye(d, dtype=torch.float64, requires_grad=True)
    if b_init is not None:
        b = torch.tensor(np.asarray(b_init, dtype=np.float64), requires_grad=True)
    else:
        b = torch.zeros(d, dtype=torch.float64, requires_grad=True)

    identity = torch.ones(d, dtype=torch.float64) if diagonal else torch.eye(
        d, dtype=torch.float64
    )

    n_parameters = int(W.numel())
    learning_rate = config.alignment.learning_rate_for(n_parameters)
    optimiser = torch.optim.Adam([W, b], lr=learning_rate)
    grad_norms: List[float] = []
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
        grad_norms.append(
            float(
                torch.sqrt(
                    sum((p.grad**2).sum() for p in (W, b) if p.grad is not None)
                ).item()
            )
        )
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
        grad_norms=grad_norms,
        learning_rate=learning_rate,
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
