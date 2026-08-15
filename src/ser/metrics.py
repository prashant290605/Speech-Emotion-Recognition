"""Evaluation metrics.

macro-F1 is the primary metric because class collapse is common under
cross-corpus transfer and accuracy hides it entirely. UAR is included because
most prior work reports it, and the comparison table needs a shared axis.

Every function takes explicit ``class_names`` and scores over *all* of them, so
a class the model never predicts contributes an F1 of 0 rather than vanishing
from the average. Silently dropping unpredicted classes is what makes a
collapsed model look competent.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

__all__ = [
    "confusion_matrix",
    "per_class_f1",
    "macro_f1",
    "accuracy",
    "uar",
    "all_metrics",
    "n_collapsed_classes",
]


def _as_indices(labels: Sequence, class_names: Sequence[str]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(class_names)}
    out = np.empty(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        if isinstance(label, (int, np.integer)):
            out[i] = int(label)
        else:
            if label not in lookup:
                raise ValueError(f"label {label!r} not in class_names {list(class_names)}")
            out[i] = lookup[label]
    return out


def confusion_matrix(
    y_true: Sequence, y_pred: Sequence, class_names: Sequence[str]
) -> np.ndarray:
    """Rows are true classes, columns predicted. Always ``(K, K)``."""
    k = len(class_names)
    true_idx = _as_indices(y_true, class_names)
    pred_idx = _as_indices(y_pred, class_names)
    if len(true_idx) != len(pred_idx):
        raise ValueError("y_true and y_pred differ in length")

    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (true_idx, pred_idx), 1)
    return matrix


def per_class_f1(
    y_true: Sequence, y_pred: Sequence, class_names: Sequence[str]
) -> Dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, class_names)
    tp = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

    return {name: float(f1[i]) for i, name in enumerate(class_names)}


def macro_f1(y_true: Sequence, y_pred: Sequence, class_names: Sequence[str]) -> float:
    """Unweighted mean F1 over every class in ``class_names``."""
    scores = per_class_f1(y_true, y_pred, class_names)
    return float(np.mean([scores[name] for name in class_names]))


def accuracy(y_true: Sequence, y_pred: Sequence, class_names: Sequence[str]) -> float:
    matrix = confusion_matrix(y_true, y_pred, class_names)
    total = matrix.sum()
    return float(np.diag(matrix).sum() / total) if total else 0.0


def uar(y_true: Sequence, y_pred: Sequence, class_names: Sequence[str]) -> float:
    """Unweighted average recall, over classes that actually occur in y_true.

    A class absent from the evaluation set has no recall to average; including
    it as 0 would penalise the model for the split rather than for its
    predictions.
    """
    matrix = confusion_matrix(y_true, y_pred, class_names)
    actual = matrix.sum(axis=1)
    present = actual > 0
    if not present.any():
        return 0.0
    recall = np.diag(matrix)[present] / actual[present]
    return float(recall.mean())


def n_collapsed_classes(
    y_pred: Sequence, class_names: Sequence[str]
) -> int:
    """Classes the model never predicts. The Phase 10 collapse diagnostic."""
    predicted = set(_as_indices(y_pred, class_names).tolist())
    return len(class_names) - len(predicted)


def all_metrics(
    y_true: Sequence, y_pred: Sequence, class_names: Sequence[str]
) -> Dict[str, object]:
    """Everything the result schema needs from one prediction vector."""
    return {
        "macro_f1": macro_f1(y_true, y_pred, class_names),
        "accuracy": accuracy(y_true, y_pred, class_names),
        "uar": uar(y_true, y_pred, class_names),
        "per_class_f1": per_class_f1(y_true, y_pred, class_names),
        "confusion": confusion_matrix(y_true, y_pred, class_names).tolist(),
        "n_collapsed_classes": n_collapsed_classes(y_pred, class_names),
    }
