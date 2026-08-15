"""The manifest: one canonical row per audio file.

Everything downstream counts from this file and nothing else. Counts are never
hardcoded anywhere in the pipeline -- they are derived here by walking the raw
corpora, so a corpus that is incomplete on disk shows up as a number that
disagrees with the published expectation rather than as a silently smaller
experiment.

Audio *content* is read only for the integrity hash. Duration and sample rate
come from the file header.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from .labels import (
    CREMAD_EMOTION_CODES,
    RAVDESS_EMOTION_CODES,
    LabelPolicy,
    map_label,
)

__all__ = [
    "ManifestRow",
    "MANIFEST_COLUMNS",
    "CORPUS_EXPECTATIONS",
    "CountMismatch",
    "build_manifest",
    "write_manifest",
    "read_manifest",
    "verify_expected_counts",
]

MANIFEST_COLUMNS = (
    "corpus",
    "file_path",
    "utterance_id",
    "speaker_id",
    "session_id",
    "subset",
    "original_label",
    "label_six",
    "label_four",
    "duration_s",
    "sample_rate",
    "sha256",
)


@dataclass(frozen=True)
class ManifestRow:
    corpus: str
    file_path: str
    utterance_id: str
    speaker_id: str
    session_id: str  # "" where the corpus has no session structure
    subset: str  # IEMOCAP scripted/improvised; "" elsewhere
    original_label: str
    label_six: str  # "" means excluded from the 6-class space
    label_four: str  # "" means excluded from the 4-class space
    duration_s: float
    sample_rate: int
    sha256: str


# Published corpus sizes. Deliberately NOT config values: a verification
# threshold a user can edit is not a verification. If a corpus legitimately
# changes, this constant changes in a reviewed commit.
#
#   RAVDESS speech: 24 actors x 60 trials = 1440.
#   CREMA-D: 91 actors, 7442 clips.
CORPUS_EXPECTATIONS: Dict[str, Dict[str, int]] = {
    "ravdess": {"files": 1440, "speakers": 24},
    "cremad": {"files": 7442, "speakers": 91},
}

# Proportion by which an observed count may differ before the build halts.
COUNT_TOLERANCE = 0.01


class CountMismatch(RuntimeError):
    """Observed corpus size differs materially from the published expectation."""


# --------------------------------------------------------------------------
# Per-corpus parsing
# --------------------------------------------------------------------------
# 03-01-05-01-01-01-12.wav
#  modality-vocalChannel-emotion-intensity-statement-repetition-actor
_RAVDESS_STEM = re.compile(
    r"^(?P<modality>\d{2})-(?P<channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})$"
)

# 1001_DFA_ANG_XX.wav -> actor, sentence, emotion, intensity
_CREMAD_STEM = re.compile(
    r"^(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>[A-Z]{3})_(?P<level>[A-Z]{2})$"
)


def _iter_ravdess(root: Path) -> Iterator[tuple[Path, str, str, str]]:
    """Yield (path, utterance_id, speaker_id, original_label)."""
    for path in sorted(root.rglob("*.wav")):
        match = _RAVDESS_STEM.match(path.stem)
        if not match:
            raise ValueError(f"RAVDESS filename does not parse: {path}")
        # Modality 03 = audio-only, channel 01 = speech. Song is a separate
        # download and must not be mixed in.
        if match.group("channel") != "01":
            continue
        emotion = RAVDESS_EMOTION_CODES[match.group("emotion")]
        yield path, f"ravdess/{path.stem}", f"ravdess_{match.group('actor')}", emotion


def _iter_cremad(root: Path) -> Iterator[tuple[Path, str, str, str]]:
    for path in sorted(root.rglob("*.wav")):
        match = _CREMAD_STEM.match(path.stem)
        if not match:
            raise ValueError(f"CREMA-D filename does not parse: {path}")
        emotion = CREMAD_EMOTION_CODES[match.group("emotion")]
        yield path, f"cremad/{path.stem}", f"cremad_{match.group('actor')}", emotion


_ITERATORS = {"ravdess": _iter_ravdess, "cremad": _iter_cremad}


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------
def build_manifest(
    roots: Dict[str, Path],
    policy: LabelPolicy,
    *,
    verify_counts: bool = True,
    progress: Optional[callable] = None,
) -> List[ManifestRow]:
    """Walk each corpus root and produce one row per audio file.

    Args:
        roots: corpus name -> directory containing its audio.
        policy: resolved label decisions.
        verify_counts: halt if an observed count differs from the published
            expectation by more than :data:`COUNT_TOLERANCE`.
    """
    import soundfile as sf

    rows: List[ManifestRow] = []
    for corpus, root in roots.items():
        if corpus not in _ITERATORS:
            raise ValueError(f"no manifest parser for corpus {corpus!r}")
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"{corpus}: {root} does not exist")

        corpus_rows: List[ManifestRow] = []
        for path, utterance_id, speaker_id, original_label in _ITERATORS[corpus](root):
            info = sf.info(str(path))
            corpus_rows.append(
                ManifestRow(
                    corpus=corpus,
                    file_path=path.as_posix(),
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                    session_id="",
                    subset="",
                    original_label=original_label,
                    label_six=map_label(corpus, original_label, "six", policy) or "",
                    label_four=map_label(corpus, original_label, "four", policy) or "",
                    duration_s=round(info.frames / info.samplerate, 6),
                    sample_rate=int(info.samplerate),
                    sha256=_sha256(path),
                )
            )
            if progress and len(corpus_rows) % 500 == 0:
                progress(corpus, len(corpus_rows))

        if verify_counts:
            verify_expected_counts(corpus, corpus_rows)
        rows.extend(corpus_rows)

    return rows


def verify_expected_counts(corpus: str, rows: Sequence[ManifestRow]) -> None:
    """Halt if the corpus on disk is materially smaller or larger than published.

    A partial download is the failure this catches. It is far cheaper to stop
    here than to discover a missing 8% of CREMA-D after the grid has run.
    """
    expectation = CORPUS_EXPECTATIONS.get(corpus)
    if expectation is None:
        return

    n_files = len(rows)
    n_speakers = len({row.speaker_id for row in rows})
    problems = []

    expected_files = expectation["files"]
    if abs(n_files - expected_files) > expected_files * COUNT_TOLERANCE:
        problems.append(f"{n_files} files, expected {expected_files}")
    if n_speakers != expectation["speakers"]:
        problems.append(f"{n_speakers} speakers, expected {expectation['speakers']}")

    if problems:
        raise CountMismatch(
            f"{corpus}: " + "; ".join(problems) + ". "
            "The corpus on disk does not match its published size. Re-check the "
            "download before building anything on top of it."
        )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def write_manifest(rows: Iterable[ManifestRow], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
            count += 1
    return count


def read_manifest(path: Path) -> List[ManifestRow]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}. Run `ser manifest` first.")
    rows: List[ManifestRow] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(
                f"manifest columns {reader.fieldnames} != expected {list(MANIFEST_COLUMNS)}"
            )
        for record in reader:
            rows.append(
                ManifestRow(
                    corpus=record["corpus"],
                    file_path=record["file_path"],
                    utterance_id=record["utterance_id"],
                    speaker_id=record["speaker_id"],
                    session_id=record["session_id"],
                    subset=record["subset"],
                    original_label=record["original_label"],
                    label_six=record["label_six"],
                    label_four=record["label_four"],
                    duration_s=float(record["duration_s"]),
                    sample_rate=int(record["sample_rate"]),
                    sha256=record["sha256"],
                )
            )
    return rows
