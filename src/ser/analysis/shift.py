"""The three-way shift decomposition, and a falsifiable test of it.

Shift between two corpora decomposes into three pieces that different methods
address and that the literature routinely conflates:

``label shift``      P(y) differs; P(x|y) does not.
``covariate shift``  P(x) differs; P(y|x) does not.
``conditional shift``P(x|y) differs -- what neither of the above can fix.

The alignment ladder only ever addresses the second. If the measured label shift
is negligible and the conditional shift does not fall when the marginal does,
then the ladder is treating the wrong term, and that is a claim about the
problem rather than about our implementation of any method.

The BBSE test at the bottom is what makes the decomposition falsifiable rather
than descriptive: a near-zero prior KL *predicts* that a label-shift correction
cannot help. If it helps anyway, the decomposition is wrong and needs
investigating, not quietly reporting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..mmd import marginal_mmd, median_bandwidth, null_mmd_scale

__all__ = [
    "class_priors",
    "label_shift_kl",
    "conditional_mmd_by_class",
    "bbse_weights",
    "em_prior_estimate",
    "apply_prior_correction",
]


def class_priors(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """Empirical P(y) over ``classes``, in that order."""
    counts = np.array([sum(1 for y in labels if y == c) for c in classes], dtype=np.float64)
    total = counts.sum()
    if total == 0:
        raise ValueError("no labels to form a prior from")
    return counts / total


def label_shift_kl(
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    classes: Sequence[str],
    *,
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """KL(P_target || P_source) between realised class priors, in nats and bits.

    Direction matters and is chosen deliberately: KL(target || source) is the
    quantity that bounds how badly a source-trained prior misprices the target,
    which is the thing label-shift correction is supposed to fix.

    Computed from the **realised partitions**, not from corpus-level counts. The
    split is speaker-disjoint and speakers are not class-balanced, so the prior
    a given seed actually trains on is not the corpus prior.
    """
    p_source = class_priors(source_labels, classes)
    p_target = class_priors(target_labels, classes)
    # A zero in the source prior with mass in the target is infinite KL, which is
    # a real answer, not a numerical problem -- but it also means the class is
    # untrainable, so it is flagged rather than silently smoothed.
    unsupported = [
        c for c, ps, pt in zip(classes, p_source, p_target) if ps == 0 and pt > 0
    ]
    safe_source = np.clip(p_source, epsilon, None)
    mask = p_target > 0
    kl = float(np.sum(p_target[mask] * np.log(p_target[mask] / safe_source[mask])))
    return {
        "kl_nats": kl,
        "kl_bits": kl / np.log(2.0),
        "total_variation": float(0.5 * np.abs(p_target - p_source).sum()),
        "source_prior": p_source.tolist(),
        "target_prior": p_target.tolist(),
        "classes_without_source_support": unsupported,
    }


def conditional_mmd_by_class(
    X_source: np.ndarray,
    y_source: Sequence[str],
    X_target: np.ndarray,
    y_target: Sequence[str],
    classes: Sequence[str],
    config,
    *,
    seed: int = 0,
    min_support: Optional[int] = None,
) -> List[Dict]:
    """``MMD(X_src|y=k, X_tgt|y=k)`` per class, normalised by the same-class null.

    **This reads target labels.** It is behind the A10 firewall and may only be
    called from analysis code. See :mod:`ser.analysis`.

    Normalisation matches the marginal statistic: MMD^2 divided by the mean
    absolute MMD^2 over random half-splits of the source side of that class, so
    the value is scale-free and comparable across classes and rungs.

    A class with fewer than ``min_support`` samples on **either** side is
    reported with ``effect_size=None`` rather than a number. Class-conditional
    MMD on a few dozen samples is mostly estimator variance, and a number there
    would be read as a measurement.
    """
    if min_support is None:
        min_support = config.shift.conditional_mmd_min_support

    y_source = list(y_source)
    y_target = list(y_target)
    out: List[Dict] = []
    for name in classes:
        src_index = [i for i, y in enumerate(y_source) if y == name]
        tgt_index = [i for i, y in enumerate(y_target) if y == name]
        n_source, n_target = len(src_index), len(tgt_index)
        record = {
            "class": name,
            "n_source": n_source,
            "n_target": n_target,
            "effect_size": None,
            "raw_mmd": None,
            "undefined_reason": None,
        }
        if n_source < min_support or n_target < min_support:
            record["undefined_reason"] = (
                f"support below {min_support} (source {n_source}, target {n_target})"
            )
            out.append(record)
            continue

        A = np.asarray(X_source[src_index], dtype=np.float64)
        B = np.asarray(X_target[tgt_index], dtype=np.float64)
        bandwidth = median_bandwidth(A, B, seed=seed)
        raw = marginal_mmd(A, B, config, bandwidth=bandwidth, seed=seed)
        null = null_mmd_scale(A, config, bandwidth=bandwidth, n_repeats=5, seed=seed)["scale"]
        record["raw_mmd"] = float(raw)
        record["effect_size"] = float(raw / null) if null > 0 else None
        out.append(record)
    return out


# --------------------------------------------------------------------------
# Label-shift correction, as a test of the decomposition
# --------------------------------------------------------------------------
def bbse_weights(
    source_confusion: np.ndarray,
    target_predicted_prior: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> Optional[np.ndarray]:
    """Black-box shift estimation of ``q(y)/p(y)``.

    ``source_confusion[i, j]`` is the joint P(true=i, predicted=j) on held-out
    **source** data, and ``target_predicted_prior[j]`` is the distribution of
    predicted labels on unlabelled **target** data. Under label shift these
    satisfy ``C^T w_scaled = mu``, so the importance weights fall out of a solve.

    Reads no target labels: BBSE is a legitimate unsupervised correction, and
    only the evaluation of whether it helped uses them.

    Returns None when the confusion matrix is too near-singular for the solve to
    mean anything -- a classifier that cannot separate the classes on source
    cannot identify the target prior either, and returning a huge unstable
    weight vector would hide that.
    """
    C = np.asarray(source_confusion, dtype=np.float64)
    mu = np.asarray(target_predicted_prior, dtype=np.float64)
    k = C.shape[0]
    # Conditioning is judged on the RAW confusion matrix. Judging it after the
    # ridge is added measures the ridge, not the classifier: a rank-deficient C
    # plus 1e-6*I has a condition number around 1e6 and would sail through.
    if not np.all(np.isfinite(C)) or np.linalg.cond(C) > 1e8:
        return None
    try:
        weights = np.linalg.solve(C.T + ridge * np.eye(k), mu)
    except np.linalg.LinAlgError:  # pragma: no cover
        return None
    if not np.all(np.isfinite(weights)):
        return None
    # Negative weights are an identifiability failure, not a valid estimate.
    return np.clip(weights, 0.0, None)


def em_prior_estimate(
    probabilities: np.ndarray,
    source_prior: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Saerens-Latinne-Decaestecker EM estimate of the target prior.

    Iterates posterior reweighting to a fixed point using only the source-
    trained classifier's probabilities on unlabelled target data.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    source_prior = np.asarray(source_prior, dtype=np.float64)
    estimate = source_prior.copy()
    for _ in range(max_iter):
        ratio = np.divide(
            estimate, source_prior,
            out=np.zeros_like(estimate), where=source_prior > 0,
        )
        weighted = probabilities * ratio
        total = weighted.sum(axis=1, keepdims=True)
        posterior = np.divide(
            weighted, total, out=np.zeros_like(weighted), where=total > 0
        )
        updated = posterior.mean(axis=0)
        if np.max(np.abs(updated - estimate)) < tol:
            return updated
        estimate = updated
    return estimate


def apply_prior_correction(
    probabilities: np.ndarray, source_prior: np.ndarray, target_prior: np.ndarray
) -> np.ndarray:
    """Re-decide under an estimated target prior: ``p(y|x) * q(y)/p(y)``."""
    ratio = np.divide(
        np.asarray(target_prior, dtype=np.float64),
        np.asarray(source_prior, dtype=np.float64),
        out=np.zeros(len(target_prior)),
        where=np.asarray(source_prior) > 0,
    )
    adjusted = np.asarray(probabilities, dtype=np.float64) * ratio
    total = adjusted.sum(axis=1, keepdims=True)
    return np.divide(adjusted, total, out=np.zeros_like(adjusted), where=total > 0)
