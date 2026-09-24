"""Coordinate identities and a read-only audit of frozen translation pairs.

These utilities do not fit models. Row-vector convention: T(x) = x A + a.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AffineGeometry:
    relative_matrix: np.ndarray
    relative_offset: np.ndarray
    distance_metric: np.ndarray
    coefficient_metric: np.ndarray


def factor_affine(source_matrix, source_offset, target_matrix, target_offset):
    """Separate relative transport from common coordinates; target must invert.

    The returned metrics describe Euclidean distances and a weight-only L2
    penalty in the original target coordinates. No fitted-solver invariance is
    asserted. A singular source matrix is allowed.
    """
    a, b = (np.asarray(v, dtype=float) for v in (source_matrix, target_matrix))
    s, t = (np.asarray(v, dtype=float) for v in (source_offset, target_offset))
    if b.ndim != 2 or b.shape[0] != b.shape[1] or a.shape != b.shape:
        raise ValueError("source and target matrices must be square and same size")
    if s.shape != (b.shape[0],) or t.shape != s.shape:
        raise ValueError("offsets must match feature dimension")
    if not all(np.isfinite(v).all() for v in (a, b, s, t)):
        raise ValueError("affine parameters must be finite")
    inverse = np.linalg.solve(b, np.eye(len(b)))
    return AffineGeometry(a @ inverse, (s - t) @ inverse,
                          b @ b.T, inverse.T @ inverse)


def match_translation_rows(rows: list[dict[str, Any]], *, freeze_tag: str,
                           classifiers: list[str], match_fields: list[str]):
    """Require complete unique pairs, without selecting on either outcome.

    Equality is checked on stored, unrounded numeric values. The ledger only
    stores the winning validation score, not every candidate validation score.
    Therefore this audit cannot empirically certify the entire search surface.
    """
    groups: dict[tuple, dict] = {}
    seen = set()
    for row in rows:
        if (row.get("freeze_tag") != freeze_tag or row.get("blending") != "none"
                or row.get("classifier") not in classifiers
                or row.get("alignment") not in {"none", "mean_shift"}):
            continue
        if row.get("status") != "ok":
            raise ValueError(f"incomplete eligible run {row.get('run_id')}")
        if row["run_id"] in seen:
            raise ValueError(f"duplicate run_id {row['run_id']}")
        seen.add(row["run_id"])
        key = tuple(json.dumps(row[field], sort_keys=True) for field in match_fields)
        arms = groups.setdefault(key, {})
        arm = row["alignment"]
        if arm in arms:
            raise ValueError("multiple eligible rows in one arm; refine match fields")
        arms[arm] = row
    if not groups:
        raise ValueError("no eligible pairs")
    matched = []
    for key, arms in sorted(groups.items()):
        if set(arms) != {"none", "mean_shift"}:
            raise ValueError(f"unmatched eligible row: {key}")
        raw, shift = arms["none"], arms["mean_shift"]
        scores = [r["selection_source_val_macro_f1"] for r in (raw, shift)]
        if not all(v is not None and np.isfinite(v) for v in scores):
            raise ValueError("missing/nonfinite validation score")
        target = [r["macro_f1"] for r in (raw, shift)]
        if not all(v is not None and np.isfinite(v) for v in target):
            raise ValueError("missing/nonfinite target score")
        params = [json.loads(r["hyperparams_json"])["selected"] for r in (raw, shift)]
        matched.append({
            **{field: raw[field] for field in match_fields},
            "none_run_id": raw["run_id"], "mean_shift_run_id": shift["run_id"],
            "validation_equal": scores[0] == scores[1],
            "hyperparameters_equal": params[0] == params[1],
            "none_validation": scores[0], "mean_shift_validation": scores[1],
            "none_target": target[0], "mean_shift_target": target[1],
            "target_difference": target[1] - target[0],
            "chance_macro_f1": raw["chance_macro_f1"],
        })
    return matched
