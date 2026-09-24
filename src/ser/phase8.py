"""Phase 8 statistics: paired comparisons from the stored predictions.

Three decisions that shape everything here, stated once:

**Comparisons are paired at the utterance level, never unpaired means.** Two
rungs evaluated on the same target_test see the same utterances; comparing
their marginal means throws that pairing away and inflates the variance of the
difference. Every interval below is on a *difference* computed per utterance.

**The bootstrap resamples speakers, not utterances.** Utterances from one
speaker are not independent -- emotion recognition errors cluster hard by
talker -- so an utterance bootstrap would report intervals several times too
narrow. Resampling whole speakers is the standard fix and the one the
speaker-disjoint split design already assumes.

**Seeds are resampled too.** The seed is the replication unit for everything
the grid controls (split realisation, search draw, init). A CI that holds the
seed fixed describes one split, not the method. Each bootstrap replicate draws
seeds with replacement and then speakers within them, so the interval carries
both sources of variance.

Confusion matrices are accumulated per speaker once, then combined by
multiplicity per replicate. That is exact -- confusion counts are additive over
utterances -- and turns a 2000-replicate bootstrap from minutes into seconds.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "load_predictions",
    "confusion_by_group",
    "macro_f1_from_confusion",
    "per_class_f1_from_confusion",
    "cluster_bootstrap",
    "paired_cluster_bootstrap",
    "holm",
    "seed_interval",
]


def load_predictions(results_path: Path, row: Dict) -> Tuple[List[str], List[str]]:
    """Utterance ids and predicted labels for one run, in the stored order."""
    path = Path(results_path).parent / row["predictions_path"]
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["utterance_ids"], payload["predicted"]


def confusion_by_group(
    y_true: Sequence[int], y_pred: Sequence[int], groups: Sequence[int], n_classes: int, n_groups: int
) -> np.ndarray:
    """``(n_groups, n_classes, n_classes)`` counts, rows true, columns predicted.

    Precomputed once per run so a bootstrap replicate is a weighted sum rather
    than a re-scan of every utterance.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    flat = np.bincount(
        (groups * n_classes + y_true) * n_classes + y_pred,
        minlength=n_groups * n_classes * n_classes,
    )
    return flat.reshape(n_groups, n_classes, n_classes)


def macro_f1_from_confusion(conf: np.ndarray) -> float:
    """Unweighted mean per-class F1. Classes absent from truth AND prediction
    are dropped, matching :func:`ser.metrics.macro_f1`."""
    tp = np.diag(conf).astype(np.float64)
    support = conf.sum(axis=1).astype(np.float64)
    predicted = conf.sum(axis=0).astype(np.float64)
    denominator = support + predicted
    present = denominator > 0
    if not present.any():
        return float("nan")
    f1 = np.zeros_like(tp)
    f1[present] = 2.0 * tp[present] / denominator[present]
    return float(f1[present].mean())


def per_class_f1_from_confusion(conf: np.ndarray) -> np.ndarray:
    tp = np.diag(conf).astype(np.float64)
    denominator = conf.sum(axis=1) + conf.sum(axis=0)
    out = np.full(conf.shape[0], np.nan)
    present = denominator > 0
    out[present] = 2.0 * tp[present] / denominator[present]
    return out


def _replicate_scores(
    arm: Dict[int, List[np.ndarray]],
    seed_draw: np.ndarray,
    group_draws: Dict[int, np.ndarray],
) -> float:
    """Mean macro-F1 for one arm under one (seed, speaker) resample.

    ``arm`` maps seed -> list of per-condition ``(n_groups, K, K)`` tensors.
    Conditions are averaged within a seed, then seeds are averaged, so a seed
    with more conditions does not get more weight.
    """
    per_seed = []
    for seed in seed_draw:
        weights = group_draws[seed]
        scores = [
            macro_f1_from_confusion(np.tensordot(weights, conf, axes=(0, 0)))
            for conf in arm[seed]
        ]
        per_seed.append(float(np.mean(scores)))
    return float(np.mean(per_seed))


def paired_cluster_bootstrap(
    arm_a: Dict[int, List[np.ndarray]],
    arm_b: Dict[int, List[np.ndarray]],
    n_groups_by_seed: Dict[int, int],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap the mean paired difference (a - b) in target macro-F1.

    Both arms are scored on the *same* resampled speakers within the same
    resampled seeds, which is what makes the difference paired: the sampling
    noise common to both cancels instead of accumulating.

    Returns the observed difference, a percentile interval, and a two-sided
    bootstrap p-value floored at ``1 / n_boot`` -- a resampling test cannot
    resolve a p-value below its own resolution and reporting one would be
    fabricating precision.
    """
    seeds = sorted(set(arm_a) & set(arm_b))
    if not seeds:
        return {"n_seeds": 0, "diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}

    identity = {s: np.ones(n_groups_by_seed[s], dtype=np.float64) for s in seeds}
    observed = _replicate_scores(arm_a, np.array(seeds), identity) - _replicate_scores(
        arm_b, np.array(seeds), identity
    )

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        seed_draw = rng.choice(seeds, size=len(seeds), replace=True)
        draws = {}
        for s in set(seed_draw.tolist()):
            n = n_groups_by_seed[s]
            draws[s] = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        diffs[i] = _replicate_scores(arm_a, seed_draw, draws) - _replicate_scores(
            arm_b, seed_draw, draws
        )

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "n_seeds": len(seeds),
        "diff": float(observed),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(max(2.0 * tail, 1.0 / n_boot)),
        # True when no replicate fell on the other side of zero, so the p-value
        # is the bootstrap's resolution limit rather than a measured quantity
        # and must be reported as "<" that value.
        "p_at_floor": bool(tail == 0.0),
        "n_boot": int(n_boot),
    }


def cluster_bootstrap(
    arm: Dict[int, List[np.ndarray]],
    n_groups_by_seed: Dict[int, int],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Same resampling scheme for a single arm's mean macro-F1."""
    seeds = sorted(arm)
    identity = {s: np.ones(n_groups_by_seed[s], dtype=np.float64) for s in seeds}
    observed = _replicate_scores(arm, np.array(seeds), identity)

    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        seed_draw = rng.choice(seeds, size=len(seeds), replace=True)
        draws = {}
        for s in set(seed_draw.tolist()):
            n = n_groups_by_seed[s]
            draws[s] = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        values[i] = _replicate_scores(arm, seed_draw, draws)

    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n_seeds": len(seeds), "mean": float(observed), "lo": float(lo), "hi": float(hi)}


def seed_interval(values: Sequence[float], *, alpha: float = 0.05) -> Dict[str, float]:
    """Mean and t-interval over seeds, for quantities with no per-utterance form.

    Used for the discrepancy columns, which are properties of a fitted map
    rather than of a prediction and so cannot be bootstrapped over utterances.
    With one observation the interval is undefined and is reported as such
    rather than as zero width.
    """
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = float(np.mean(values))
    if n == 1:
        return {"n": 1, "mean": mean, "lo": float("nan"), "hi": float("nan")}
    from statistics import stdev

    # t critical values for the small n this project actually has.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    half = critical * stdev(values) / np.sqrt(n)
    return {"n": n, "mean": mean, "lo": mean - half, "hi": mean + half}


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Holm rather than Bonferroni because it is uniformly more powerful at the
    same family-wise error rate, and rather than Benjamini-Hochberg because the
    primary family is small and pre-registered, where controlling the
    family-wise rate is the stricter and more appropriate guarantee.
    """
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        value = (n - rank) * pvalues[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted
