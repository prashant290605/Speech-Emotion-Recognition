"""Prospective replication: candidate surface, result isolation, protocol loading.

Three things this module exists to guarantee, each of which the retrospective
audit could not:

**The candidate surface is shared and recorded.** The frozen ledger stores only
the winning candidate's validation score, so the existing audit can check that
the *selected* models agree and must take the rest of the search surface on
faith. Here the candidate list is generated once per (classifier, seed) and
handed to both arms of a matched cell, and every candidate's validation
predictions are digested and persisted. The theorem can then be audited at
candidate level, which is what it actually claims.

**The historical ledger cannot be written.** A prospective run that appended to
``results/runs.jsonl`` would destroy the provenance of the published
experiment. :func:`assert_isolated_output` refuses any such path, and it is
called by the runner before anything is opened for writing.

**The protocol is loaded, not reimplemented.** Every decision lives in
``configs/prospective_translation_v1.yaml``; this module reads it. A runner that
disagreed with the protocol document would be the whole exercise defeated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import repo_root

__all__ = [
    "PROTOCOL_PATH",
    "HISTORICAL_LEDGER",
    "ProtocolError",
    "IsolationError",
    "load_protocol",
    "protocol_digest",
    "assert_isolated_output",
    "candidate_list",
    "candidate_list_digest",
    "prediction_digest",
    "decision_digest",
    "TrialRecord",
    "subset_decision",
]

PROTOCOL_PATH = "configs/prospective_translation_v1.yaml"

# The published experiment's ledger. Nothing in this module may write to it.
HISTORICAL_LEDGER = "results/runs.jsonl"


class ProtocolError(RuntimeError):
    """The protocol is missing, malformed, or disagrees with the request."""


class IsolationError(RuntimeError):
    """A prospective output path would collide with historical results."""


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------
def load_protocol(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the protocol and check the fields the runner depends on."""
    import yaml

    target = Path(path) if path else repo_root() / PROTOCOL_PATH
    if not target.exists():
        raise ProtocolError(f"protocol not found: {target}")
    protocol = yaml.safe_load(target.read_bytes())
    if not isinstance(protocol, dict):
        raise ProtocolError(f"{target} is not a YAML mapping")

    required = ("protocol", "directions", "label_space", "features", "alignments",
                "classifiers", "search_budget", "splits", "equality", "runs",
                "trials", "output_root", "forbidden_outputs", "expected_runs")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ProtocolError(f"{target} is missing {missing}")

    if set(protocol["alignments"]) != {"none", "mean_shift"}:
        raise ProtocolError(
            f"this protocol is a translation replication; alignments must be "
            f"exactly none and mean_shift, found {protocol['alignments']}"
        )
    return protocol


def protocol_digest(path: Optional[Path] = None) -> str:
    """sha256 of the protocol file's bytes, for the provenance record."""
    target = Path(path) if path else repo_root() / PROTOCOL_PATH
    return hashlib.sha256(target.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Result isolation
# --------------------------------------------------------------------------
def assert_isolated_output(path: Path, protocol: Dict[str, Any],
                           *, root: Optional[Path] = None) -> Path:
    """Raise unless ``path`` is inside the protocol's own output tree.

    Checked on resolved paths, so ``results/prospective_translation_v1/../runs.jsonl``
    is refused as firmly as ``results/runs.jsonl``. This is the guard that keeps
    a prospective run from contaminating the published ledger.
    """
    base = (root or repo_root()).resolve()
    target = Path(path)
    resolved = (base / target).resolve() if not target.is_absolute() else target.resolve()

    for forbidden in protocol.get("forbidden_outputs", []) + [HISTORICAL_LEDGER]:
        if resolved == (base / forbidden).resolve():
            raise IsolationError(
                f"refusing to write prospective results to {forbidden}. The "
                "historical ledger is the provenance of the published "
                "experiment and is read-only here."
            )

    allowed = (base / protocol["output_root"]).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError:
        raise IsolationError(
            f"{resolved} is outside the protocol's output root {allowed}. "
            "Prospective results live in their own tree."
        ) from None
    return resolved


# --------------------------------------------------------------------------
# The candidate surface
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialRecord:
    """One candidate, evaluated on source validation.

    Digests rather than raw vectors: a prediction vector per candidate per run
    would be 2400 vectors, and the audit only needs to know whether two arms
    produced the same thing. The digest is of the exact predicted labels, so
    equality of digests is equality of predictions.
    """

    candidate_index: int
    params: Dict[str, Any]
    source_val_macro_f1: float
    prediction_digest: str
    decision_digest: str
    converged: bool
    n_iter: Optional[int] = None

    def as_row(self, **extra: Any) -> Dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "params": self.params,
            "source_val_macro_f1": self.source_val_macro_f1,
            "prediction_digest": self.prediction_digest,
            "decision_digest": self.decision_digest,
            "converged": self.converged,
            "n_iter": self.n_iter,
            **extra,
        }


def candidate_list(classifier: str, seed: int, protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The candidates a (classifier, seed) cell evaluates, in order.

    Generated from the seed alone, so the ``none`` and ``mean_shift`` arms of a
    matched cell necessarily receive an identical list in an identical order.
    Sampling inside each arm -- as the frozen grid does -- would make the two
    arms' surfaces merely similar, and the theorem is about identity.
    """
    budget = int(protocol["search_budget"])
    rng = np.random.default_rng(seed)
    out: List[Dict[str, Any]] = []
    for _ in range(budget):
        if classifier == "svm_rbf":
            out.append({
                "C": float(10.0 ** rng.uniform(-2, 4)),
                "gamma": float(10.0 ** rng.uniform(-5, 0)),
            })
        elif classifier == "logreg":
            out.append({"C": float(10.0 ** rng.uniform(-3, 4))})
        else:
            raise ProtocolError(
                f"no candidate space defined for {classifier!r}; this protocol "
                f"covers {protocol['classifiers']}"
            )
    return out


def candidate_list_digest(candidates: Sequence[Dict[str, Any]]) -> str:
    """Digest of a candidate list, for proving both arms received the same one."""
    payload = json.dumps(list(candidates), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prediction_digest(labels: Sequence[Any]) -> str:
    """Digest of predicted labels, in order. Exact: no rounding anywhere."""
    payload = json.dumps([str(v) for v in labels], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decision_digest(values: np.ndarray, *, atol: float) -> str:
    """Digest of decision-function values quantised to the protocol tolerance.

    Quantising before hashing is what makes a digest usable for a float
    comparison: two runs agreeing to within ``atol`` produce the same digest.
    The tolerance comes from the protocol and is fixed before execution, so it
    cannot be widened after seeing a disagreement.
    """
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("decision values must be finite")
    quantised = np.round(array / atol).astype(np.int64)
    return hashlib.sha256(quantised.tobytes()).hexdigest()


# --------------------------------------------------------------------------
# The one open design decision
# --------------------------------------------------------------------------
def subset_decision(counts: Dict[str, Dict[str, int]],
                    protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the pre-committed improvised-vs-both rule to metadata counts.

    ``counts`` maps subset -> class -> per-session counts, as produced by
    ``tools/prospective_counts.py``. The rule and its thresholds are fixed in
    the protocol; this only evaluates them, and it records why, so the decision
    is auditable rather than asserted.
    """
    spec = protocol["iemocap_subset"]
    limits = spec["thresholds"]
    improvised = counts.get("improvised", {})

    reasons: List[str] = []
    classes = protocol.get("classes") or ["angry", "happy", "neutral", "sad"]
    for label in classes:
        per_session = improvised.get(label, {})
        total = sum(per_session.values())
        if total < limits["min_utterances_per_class"]:
            reasons.append(
                f"{label}: {total} utterances < {limits['min_utterances_per_class']}")
        present = sum(1 for n in per_session.values() if n > 0)
        if present < limits["min_sessions_containing_each_class"]:
            reasons.append(
                f"{label}: present in {present} sessions < "
                f"{limits['min_sessions_containing_each_class']}")
        thin = [s for s, n in per_session.items()
                if n < limits["min_utterances_per_class_per_session"]]
        if thin:
            reasons.append(
                f"{label}: below {limits['min_utterances_per_class_per_session']} "
                f"in sessions {sorted(thin)}")

    chosen = "improvised" if not reasons else spec["fallback"]
    return {
        "chosen": chosen,
        "rule": spec["rule"],
        "thresholds": limits,
        "failed_checks": reasons,
        "counts": counts,
    }


# --------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------
def match_prospective_pairs(runs: Sequence[Dict[str, Any]],
                            trials: Sequence[Dict[str, Any]],
                            protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Join none/mean_shift arms and compare them at candidate level.

    The retrospective audit can only compare selected models. This compares
    every candidate, which is what Proposition 1 actually claims, and reports
    both levels separately so a reader can see which one the evidence covers.

    Fails closed on an incomplete or ambiguous join, exactly as the
    retrospective matcher does: a missing arm, a duplicate run id or two
    eligible rows in one arm each raise rather than being resolved by a
    heuristic. No target quantity is read here.
    """
    fields = list(protocol["match_fields"])
    groups: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    seen: set = set()
    for row in runs:
        if row.get("protocol") != protocol["protocol"]:
            continue
        if row.get("status") != "ok":
            raise ProtocolError(f"incomplete prospective run {row.get('run_id')}")
        if row["run_id"] in seen:
            raise ProtocolError(f"duplicate run_id {row['run_id']}")
        seen.add(row["run_id"])
        key = tuple(json.dumps(row[f], sort_keys=True) for f in fields)
        arms = groups.setdefault(key, {})
        arm = row["alignment"]
        if arm in arms:
            raise ProtocolError("two eligible rows in one arm; refine match fields")
        arms[arm] = row
    if not groups:
        raise ProtocolError("no eligible prospective pairs")

    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for trial in trials:
        by_run.setdefault(trial["run_id"], []).append(trial)

    matched: List[Dict[str, Any]] = []
    for key, arms in sorted(groups.items()):
        if set(arms) != {"none", "mean_shift"}:
            raise ProtocolError(f"unmatched eligible cell: {key}")
        raw, shift = arms["none"], arms["mean_shift"]

        left = sorted(by_run.get(raw["run_id"], []), key=lambda t: t["candidate_index"])
        right = sorted(by_run.get(shift["run_id"], []), key=lambda t: t["candidate_index"])
        if not left or len(left) != len(right):
            raise ProtocolError(
                f"trial surfaces differ in size for {raw['run_id']}/{shift['run_id']}: "
                f"{len(left)} vs {len(right)}"
            )
        if [t["candidate_index"] for t in left] != [t["candidate_index"] for t in right]:
            raise ProtocolError("candidate indices differ across arms")
        if [t["params"] for t in left] != [t["params"] for t in right]:
            raise ProtocolError(
                "the two arms did not receive the same candidate list; the "
                "comparison is not matched and the audit is void"
            )

        prediction_equal = sum(a["prediction_digest"] == b["prediction_digest"]
                               for a, b in zip(left, right))
        score_equal = sum(a["source_val_macro_f1"] == b["source_val_macro_f1"]
                          for a, b in zip(left, right))
        decision_equal = sum(a["decision_digest"] == b["decision_digest"]
                             for a, b in zip(left, right))

        matched.append({
            **{f: raw[f] for f in fields},
            "none_run_id": raw["run_id"],
            "mean_shift_run_id": shift["run_id"],
            "candidates": len(left),
            "candidate_prediction_equal": prediction_equal,
            "candidate_score_equal": score_equal,
            "candidate_decision_equal": decision_equal,
            "selected_score_equal":
                raw["selection_source_val_macro_f1"] == shift["selection_source_val_macro_f1"],
            "selected_params_equal":
                json.loads(raw["hyperparams_json"])["selected"]
                == json.loads(shift["hyperparams_json"])["selected"],
            "none_target": raw["macro_f1"],
            "mean_shift_target": shift["macro_f1"],
            "target_difference": shift["macro_f1"] - raw["macro_f1"],
            "chance_macro_f1": raw["chance_macro_f1"],
        })
    return matched
