"""Chance floors.

Every metric in this paper is reported beside its floor. A cross-corpus macro-F1
of 0.30 means nothing until you know whether chance is 0.17 or 0.25 — and after
amendment A4 it differs by pair, because IEMOCAP pairs run 4-class while
RAVDESS↔CREMA-D runs 6-class.

Three floors, all computed against the **realised** target-eval label
distribution rather than an assumed uniform one:

``uniform_random``
    Predict each class with probability 1/K, independent of the input.
``majority``
    Always predict the most frequent class in ``source_train``. This is the
    collapse floor, and it is what a model that has learned nothing but the
    source's dominant class scores.
``stratified_random``
    Sample predictions from the ``source_train`` class prior. The most honest
    floor: what a model that learned nothing about the input but everything
    about the source prior would score.

Each has a closed form *and* an empirical estimate over many draws. The analytic
values are expectations of a ratio approximated by the ratio of expectations, so
they are close but not exact; the empirical estimate with its CI is the ground
truth, and a test asserts the two agree.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import all_metrics, macro_f1

__all__ = [
    "BaselineResult",
    "analytic_uniform_macro_f1",
    "analytic_majority_macro_f1",
    "analytic_stratified_macro_f1",
    "uniform_random",
    "majority_class",
    "stratified_random",
    "all_floors",
]


@dataclass
class BaselineResult:
    name: str
    macro_f1: float
    accuracy: float
    uar: float
    per_class_f1: Dict[str, float]
    confusion: List[List[int]]
    analytic_macro_f1: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n_draws: int = 0
    details: Dict[str, object] = field(default_factory=dict)


def _target_prior(y_true: Sequence[str], class_names: Sequence[str]) -> np.ndarray:
    counts = Counter(y_true)
    total = sum(counts.get(name, 0) for name in class_names)
    if total == 0:
        raise ValueError("no evaluation labels")
    return np.array([counts.get(name, 0) / total for name in class_names], dtype=np.float64)


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------
def analytic_uniform_macro_f1(target_prior: Sequence[float]) -> float:
    """Expected macro-F1 of a uniform random predictor.

    For class k with target share p_k and prediction probability 1/K:
        precision_k = p_k, recall_k = 1/K
        F1_k = 2 p_k (1/K) / (p_k + 1/K)

    For a balanced target this collapses to 1/K, giving 0.167 at K=6 and 0.250
    at K=4 -- the two floors this project actually needs.
    """
    p = np.asarray(target_prior, dtype=np.float64)
    k = len(p)
    q = 1.0 / k
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + q > 0, 2 * p * q / (p + q), 0.0)
    return float(f1.mean())


def analytic_majority_macro_f1(
    target_prior: Sequence[float], majority_index: int
) -> float:
    """Expected macro-F1 of a constant predictor.

    Only the predicted class scores: precision = p_m, recall = 1, so
    F1_m = 2 p_m / (p_m + 1) and every other class contributes 0.
    At K=6 with a balanced target this is ~0.048 -- the collapse floor.
    """
    p = np.asarray(target_prior, dtype=np.float64)
    p_m = float(p[majority_index])
    f1_m = 2 * p_m / (p_m + 1.0) if (p_m + 1.0) > 0 else 0.0
    return float(f1_m / len(p))


def analytic_stratified_macro_f1(
    target_prior: Sequence[float], source_prior: Sequence[float]
) -> float:
    """Expected macro-F1 when predictions are drawn from the source prior.

    precision_k = p_k, recall_k = q_k, so F1_k = 2 p_k q_k / (p_k + q_k).
    """
    p = np.asarray(target_prior, dtype=np.float64)
    q = np.asarray(source_prior, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + q > 0, 2 * p * q / (p + q), 0.0)
    return float(f1.mean())


# --------------------------------------------------------------------------
# Empirical
# --------------------------------------------------------------------------
def _empirical(
    y_true: Sequence[str],
    class_names: Sequence[str],
    draw: callable,
    n_draws: int,
    rng: np.random.Generator,
    ci_level: float,
) -> tuple[float, float, float, np.ndarray]:
    scores = np.empty(n_draws, dtype=np.float64)
    last = None
    for i in range(n_draws):
        last = draw(rng)
        scores[i] = macro_f1(y_true, last, class_names)
    alpha = (1.0 - ci_level) / 2.0
    return (
        float(scores.mean()),
        float(np.quantile(scores, alpha)),
        float(np.quantile(scores, 1 - alpha)),
        last,
    )


def uniform_random(
    y_true: Sequence[str],
    class_names: Sequence[str],
    *,
    n_draws: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> BaselineResult:
    rng = np.random.default_rng(seed)
    names = list(class_names)
    n = len(y_true)

    def draw(generator: np.random.Generator):
        return [names[i] for i in generator.integers(0, len(names), size=n)]

    mean, low, high, last = _empirical(
        y_true, names, draw, n_draws, rng, ci_level
    )
    metrics = all_metrics(y_true, last, names)
    return BaselineResult(
        name="uniform_random",
        macro_f1=mean,
        accuracy=metrics["accuracy"],
        uar=metrics["uar"],
        per_class_f1=metrics["per_class_f1"],
        confusion=metrics["confusion"],
        analytic_macro_f1=analytic_uniform_macro_f1(_target_prior(y_true, names)),
        ci_low=low,
        ci_high=high,
        n_draws=n_draws,
        details={"note": "macro_f1 is the mean over draws; confusion is the last draw"},
    )


def majority_class(
    y_true: Sequence[str],
    class_names: Sequence[str],
    source_labels: Sequence[str],
) -> BaselineResult:
    """Always predict the most frequent ``source_train`` class.

    Deterministic, so no draws are needed. Uses the *source* majority because
    that is what a practitioner without target labels would do; using the target
    majority would be an oracle.
    """
    names = list(class_names)
    counts = Counter(source_labels)
    majority = max(names, key=lambda name: (counts.get(name, 0), name))
    predictions = [majority] * len(y_true)

    metrics = all_metrics(y_true, predictions, names)
    prior = _target_prior(y_true, names)
    return BaselineResult(
        name="majority",
        macro_f1=metrics["macro_f1"],
        accuracy=metrics["accuracy"],
        uar=metrics["uar"],
        per_class_f1=metrics["per_class_f1"],
        confusion=metrics["confusion"],
        analytic_macro_f1=analytic_majority_macro_f1(prior, names.index(majority)),
        n_draws=0,
        details={"majority_class": majority, "from": "source_train"},
    )


def stratified_random(
    y_true: Sequence[str],
    class_names: Sequence[str],
    source_labels: Sequence[str],
    *,
    n_draws: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> BaselineResult:
    """Sample predictions from the ``source_train`` class prior."""
    names = list(class_names)
    counts = Counter(source_labels)
    total = sum(counts.get(name, 0) for name in names)
    if total == 0:
        raise ValueError("no source labels to build a prior from")
    source_prior = np.array([counts.get(name, 0) / total for name in names])

    rng = np.random.default_rng(seed)
    n = len(y_true)

    def draw(generator: np.random.Generator):
        idx = generator.choice(len(names), size=n, p=source_prior)
        return [names[i] for i in idx]

    mean, low, high, last = _empirical(y_true, names, draw, n_draws, rng, ci_level)
    metrics = all_metrics(y_true, last, names)
    return BaselineResult(
        name="stratified_random",
        macro_f1=mean,
        accuracy=metrics["accuracy"],
        uar=metrics["uar"],
        per_class_f1=metrics["per_class_f1"],
        confusion=metrics["confusion"],
        analytic_macro_f1=analytic_stratified_macro_f1(
            _target_prior(y_true, names), source_prior
        ),
        ci_low=low,
        ci_high=high,
        n_draws=n_draws,
        details={"source_prior": source_prior.round(6).tolist()},
    )


def all_floors(
    y_true: Sequence[str],
    class_names: Sequence[str],
    source_labels: Sequence[str],
    *,
    n_draws: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> Dict[str, BaselineResult]:
    """All three floors for one (pair, seed).

    The returned dict is what every downstream result row carries in its
    ``chance_macro_f1`` / ``majority_macro_f1`` / ``prior_matched_macro_f1``
    columns, so no metric is ever reported without its floor beside it.
    """
    return {
        "uniform_random": uniform_random(
            y_true, class_names, n_draws=n_draws, seed=seed, ci_level=ci_level
        ),
        "majority": majority_class(y_true, class_names, source_labels),
        "stratified_random": stratified_random(
            y_true, class_names, source_labels, n_draws=n_draws, seed=seed,
            ci_level=ci_level,
        ),
    }
