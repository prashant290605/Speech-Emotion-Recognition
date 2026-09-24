"""Per-speaker confusion counts: the sufficient statistic for every published interval.

Every cluster-bootstrap number in this paper is computed the same way. A
replicate draws seeds with replacement, draws that seed's target-test speakers
with replacement, and then forms a weighted sum of per-speaker confusion
matrices before scoring:

    np.tensordot(weights, per_speaker_confusions, axes=(0, 0))

The per-utterance predictions are used for exactly one thing: building those
per-speaker confusion matrices in the first place. Once built, no downstream
statistic reads them again. So the confusions *are* the sufficient statistic
for macro-F1, per-class precision/recall/F1, the pooled confusion matrix, and
every paired arm difference derived from them -- and they compress hard: K x K
counts per speaker instead of one label per utterance, with 6 classes and
thousands of utterances.

This matters for review rather than for speed. The predictions live in 7730
gzipped files that are gitignored, which made the manuscript's intervals
unreproducible from a checkout. The artifact this module reads and writes is a
single tracked file.

**What is preserved exactly.** The speaker ordering. A bootstrap replicate
indexes speakers positionally under a fixed ``default_rng`` seed, so a
different ordering would give different draws and different intervals from the
same data. The artifact stores speaker ids in the order the prediction-based
path produced them -- ``sorted(...)`` over the target-test speaker set -- and
the loader rebuilds that order, so replicates are reproduced draw for draw and
not merely in distribution.

**What is not preserved.** Which utterance received which label. Anything that
needs per-utterance identity -- McNemar on individual items, an utterance-level
bootstrap, error inspection by clip -- still needs ``results/predictions/``,
which remains the original source and is not deleted.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import repo_root

__all__ = [
    "SPEAKER_CONFUSIONS",
    "SpeakerConfusions",
    "build_record",
    "write_records",
    "load",
]

SPEAKER_CONFUSIONS = "results/speaker_confusions.jsonl.gz"


def build_record(run_id: str, class_names: Sequence[str], speaker_names: Sequence[str],
                 confusions: np.ndarray, **extra: Any) -> Dict[str, Any]:
    """One run's record. Counts are integers; the format stores them as such."""
    array = np.asarray(confusions)
    if array.shape != (len(speaker_names), len(class_names), len(class_names)):
        raise ValueError(
            f"{run_id}: confusion shape {array.shape} does not match "
            f"{len(speaker_names)} speakers x {len(class_names)} classes"
        )
    if not np.all(array == array.astype(np.int64)):
        raise ValueError(f"{run_id}: confusion counts must be integral")
    return {
        "run_id": run_id,
        "class_names": list(class_names),
        "speakers": list(speaker_names),
        # Flattened row-major per speaker: rows true, columns predicted, which
        # is the convention ser.phase8.confusion_by_group produces.
        "confusions": array.astype(np.int64).reshape(len(speaker_names), -1).tolist(),
        **extra,
    }


def write_records(records: Iterable[Dict[str, Any]], path: Path) -> Tuple[int, str]:
    """Write deterministically and return ``(count, sha256)``.

    Sorted by ``run_id`` and gzipped with ``mtime=0`` so the artifact's digest
    depends on its contents and not on when it was written.
    """
    import hashlib

    ordered = sorted(records, key=lambda record: record["run_id"])
    payload = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in ordered
    ).encode("utf-8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw:
        # filename="" and mtime=0 keep the gzip header out of the picture.
        # Without them the header carries the output file's own name and the
        # time of writing, so two byte-identical payloads produce two
        # different files and the artifact cannot be compared by digest.
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(payload)
    return len(ordered), hashlib.sha256(payload).hexdigest()


class SpeakerConfusions:
    """Loaded per-speaker confusions, indexed by ``run_id``."""

    def __init__(self, records: Dict[str, Dict[str, Any]]):
        self._records = records
        self._cache: Dict[str, np.ndarray] = {}

    def __contains__(self, run_id: str) -> bool:
        return run_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    def class_names(self, run_id: str) -> List[str]:
        return list(self._records[run_id]["class_names"])

    def speakers(self, run_id: str) -> List[str]:
        """Speaker ids in the order the bootstrap indexes them."""
        return list(self._records[run_id]["speakers"])

    def tensor(self, run_id: str) -> np.ndarray:
        """``(n_speakers, K, K)`` counts for one run."""
        if run_id not in self._cache:
            record = self._records[run_id]
            k = len(record["class_names"])
            self._cache[run_id] = np.asarray(
                record["confusions"], dtype=np.int64
            ).reshape(len(record["speakers"]), k, k)
        return self._cache[run_id]

    def speaker_index(self, run_id: str) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.speakers(run_id))}


def load(path: Optional[Path] = None, *, root: Optional[Path] = None) -> SpeakerConfusions:
    target = Path(path) if path is not None else (root or repo_root()) / SPEAKER_CONFUSIONS
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Build it with "
            "`python tools/make_speaker_confusions.py` on a machine that has "
            "results/predictions/."
        )
    records: Dict[str, Dict[str, Any]] = {}
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["run_id"]] = record
    return SpeakerConfusions(records)
