"""Uncertainty and significance.

Three tools, matching how the results are actually compared:

* **Bootstrap CI over test utterances** — for a single run's metric. Resampling
  utterances, not runs, is what expresses "how much of this number is the luck
  of which speakers landed in target_test".
* **Wilcoxon signed-rank** — paired across matched (pair, seed) combinations.
  Paired because the same splits are used for both conditions; non-parametric
  because macro-F1 across a handful of pairs is not plausibly normal.
* **Holm-Bonferroni** — the grid invites several comparisons, and reporting the
  best of them uncorrected is the same family of error this rebuild exists to
  remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BootstrapCI",
    "bootstrap_ci",
    "wilcoxon_signed_rank",
    "holm_bonferroni",
    "PairedTest",
]


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    level: float
    n_resamples: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric: Callable[[Sequence, Sequence], float],
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap over test utterances.

    Resamples (true, predicted) pairs with replacement, so the CI reflects
    uncertainty from which utterances were evaluated.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    n = len(y_true)
    if n == 0:
        raise ValueError("nothing to resample")

    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric([y_true[j] for j in idx], [y_pred[j] for j in idx])

    alpha = (1.0 - level) / 2.0
    return BootstrapCI(
        point=float(metric(y_true, y_pred)),
        low=float(np.quantile(scores, alpha)),
        high=float(np.quantile(scores, 1 - alpha)),
        level=level,
        n_resamples=n_resamples,
    )


@dataclass(frozen=True)
class PairedTest:
    name: str
    statistic: float
    p_value: float
    n_pairs: int
    median_difference: float


def wilcoxon_signed_rank(
    a: Sequence[float], b: Sequence[float], *, name: str = "comparison"
) -> PairedTest:
    """Paired Wilcoxon signed-rank over matched observations.

    ``a`` and ``b`` must be aligned: element *i* of each is the same
    (pair, seed) under two conditions.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if a.size == 0:
        raise ValueError("no observations")

    differences = a - b
    non_zero = differences[differences != 0]
    if non_zero.size == 0:
        # Identical throughout: no evidence of a difference, and scipy would raise.
        return PairedTest(name=name, statistic=0.0, p_value=1.0, n_pairs=int(a.size),
                          median_difference=0.0)

    from scipy.stats import wilcoxon

    result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return PairedTest(
        name=name,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_pairs=int(a.size),
        median_difference=float(np.median(differences)),
    )


def holm_bonferroni(
    p_values: Dict[str, float], *, alpha: float = 0.05
) -> List[Tuple[str, float, float, bool]]:
    """Holm-Bonferroni step-down correction.

    Returns ``(name, raw_p, adjusted_p, reject)`` sorted by raw p ascending.
    Adjusted p-values are made monotone, so a later comparison can never appear
    more significant than an earlier one with a smaller raw p.
    """
    if not p_values:
        return []

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)

    adjusted: List[float] = []
    running_max = 0.0
    for index, (_, raw) in enumerate(ordered):
        value = min(1.0, (m - index) * raw)
        running_max = max(running_max, value)
        adjusted.append(running_max)

    return [
        (name, raw, adjusted[i], adjusted[i] <= alpha)
        for i, (name, raw) in enumerate(ordered)
    ]
