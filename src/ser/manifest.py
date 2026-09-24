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
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence

from .labels import (
    CREMAD_EMOTION_CODES,
    RAVDESS_EMOTION_CODES,
    LabelPolicy,
    map_label,
)

__all__ = [
    "ManifestRow",
    "ParsedUtterance",
    "IEMOCAP_EMOTION_CODES",
    "MANIFEST_COLUMNS",
    "PORTABLE_COLUMNS",
    "PORTABLE_MANIFEST",
    "CORPUS_PATH_KEYS",
    "CORPUS_EXPECTATIONS",
    "CountMismatch",
    "build_manifest",
    "write_manifest",
    "read_manifest",
    "to_portable",
    "write_portable_manifest",
    "read_portable_manifest",
    "corpus_roots",
    "resolve_audio_path",
    "load_for_analysis",
    "verify_expected_counts",
]

# Where a corpus's audio lives, per config. The manifest walker takes roots as
# an argument; this mapping is what the portable form needs in order to strip a
# machine-local prefix off and to put it back again.
CORPUS_PATH_KEYS: Dict[str, str] = {
    "ravdess": "raw_ravdess",
    "cremad": "raw_cremad",
    "iemocap": "raw_iemocap",
}

PORTABLE_MANIFEST = "data/manifest_portable.csv"

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
    #   IEMOCAP: 5 sessions x 2 actors = 10 speakers, 10039 segmented
    #   sentences across improvised and scripted dialogues.
    "iemocap": {"files": 10039, "speakers": 10},
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


class ParsedUtterance(NamedTuple):
    """What a corpus parser yields, before durations and hashes are read.

    ``session_id`` and ``subset`` default to empty because only IEMOCAP has
    them. RAVDESS and CREMA-D rows are byte-identical to what they were before
    these fields existed, which a regression test asserts.
    """

    path: Path
    utterance_id: str
    speaker_id: str
    original_label: str
    session_id: str = ""
    subset: str = ""


def _iter_ravdess(root: Path) -> Iterator[ParsedUtterance]:
    """Yield one ParsedUtterance per audio-only speech file."""
    for path in sorted(root.rglob("*.wav")):
        match = _RAVDESS_STEM.match(path.stem)
        if not match:
            raise ValueError(f"RAVDESS filename does not parse: {path}")
        # Modality 03 = audio-only, channel 01 = speech. Song is a separate
        # download and must not be mixed in.
        if match.group("channel") != "01":
            continue
        emotion = RAVDESS_EMOTION_CODES[match.group("emotion")]
        yield ParsedUtterance(path, f"ravdess/{path.stem}",
                              f"ravdess_{match.group('actor')}", emotion)


def _iter_cremad(root: Path) -> Iterator[ParsedUtterance]:
    for path in sorted(root.rglob("*.wav")):
        match = _CREMAD_STEM.match(path.stem)
        if not match:
            raise ValueError(f"CREMA-D filename does not parse: {path}")
        emotion = CREMAD_EMOTION_CODES[match.group("emotion")]
        yield ParsedUtterance(path, f"cremad/{path.stem}",
                              f"cremad_{match.group('actor')}", emotion)


# --------------------------------------------------------------------------
# IEMOCAP
# --------------------------------------------------------------------------
# IEMOCAP's labels live in annotation files, not in filenames, so this parser
# reads them. `dialog/EmoEvaluation/<dialog>.txt` holds one summary line per
# utterance:
#
#   [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
#
# That summary label is already the majority vote across annotators, with `xxx`
# recorded where no majority exists. Reading it is therefore exactly the
# configured `majority_vote_discard_disagreement` policy; the per-evaluator
# files under EmoEvaluation/Categorical/ implement a *different* policy and are
# deliberately not read here.
_IEMOCAP_SUMMARY = re.compile(
    r"^\[[\d.]+\s*-\s*[\d.]+\]\s+"
    r"(?P<utt>Ses\d{2}[FM]_\w+)\s+"
    r"(?P<code>[a-z]{3})\s+"
    r"\[[-\d.,\s]+\]\s*$"
)

# Utterance id: Ses01F_impro01_F000 / Ses05M_script03_2_M012.
# The leading Ses01F names the *dialog*; the trailing F000/M012 names the
# speaker of this utterance, which is the one that matters for grouping.
_IEMOCAP_UTT = re.compile(
    r"^Ses(?P<session>\d{2})(?P<dialog_gender>[FM])_"
    r"(?P<dialog>(?P<kind>impro|script)\w*?)_"
    r"(?P<speaker_gender>[FM])(?P<index>\d{3})$"
)

# The three-letter codes IEMOCAP writes, mapped onto the raw-label vocabulary
# ser.labels already declares for this corpus. Every code is listed: an
# unmapped one raises rather than being silently dropped, because a silent drop
# is indistinguishable from a deliberate exclusion.
IEMOCAP_EMOTION_CODES: Mapping[str, str] = {
    "ang": "angry",
    "dis": "disgust",
    "exc": "excited",
    "fea": "fear",
    "fru": "frustrated",
    "hap": "happy",
    "neu": "neutral",
    "oth": "other",
    "sad": "sad",
    "sur": "surprised",
    "xxx": "xxx",
}


def _iemocap_annotations(root: Path) -> Dict[str, str]:
    """utterance stem -> raw label, read from every EmoEvaluation summary file."""
    labels: Dict[str, str] = {}
    files = sorted(root.rglob("EmoEvaluation/*.txt"))
    if not files:
        raise FileNotFoundError(
            f"no EmoEvaluation annotation files under {root}. IEMOCAP labels "
            "come from annotations, not filenames; a release without "
            "dialog/EmoEvaluation cannot be parsed."
        )
    for path in files:
        if path.parent.name != "EmoEvaluation":
            continue  # skip Categorical/, Attribute/, Self-evaluation/
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _IEMOCAP_SUMMARY.match(line.strip())
            if not match:
                continue
            code = match.group("code")
            if code not in IEMOCAP_EMOTION_CODES:
                raise ValueError(
                    f"{path.name}: unrecognised IEMOCAP emotion code {code!r}. "
                    f"Known: {sorted(IEMOCAP_EMOTION_CODES)}. Refusing to guess."
                )
            utterance = match.group("utt")
            mapped = IEMOCAP_EMOTION_CODES[code]
            if labels.get(utterance, mapped) != mapped:
                raise ValueError(
                    f"{utterance} annotated as both {labels[utterance]!r} and "
                    f"{mapped!r}; the release is inconsistent."
                )
            labels[utterance] = mapped
    if not labels:
        raise ValueError(f"EmoEvaluation files under {root} contained no summary lines")
    return labels


def _iter_iemocap(root: Path) -> Iterator[ParsedUtterance]:
    """Yield one ParsedUtterance per segmented sentence with an annotation.

    Audio comes from ``sentences/wav``; labels come from the annotations. A wav
    with no annotation raises rather than being skipped: a partial release
    should fail loudly, not quietly become a smaller corpus.
    """
    labels = _iemocap_annotations(root)
    seen: set[str] = set()
    for path in sorted(root.rglob("sentences/wav/*/*.wav")):
        stem = path.stem
        match = _IEMOCAP_UTT.match(stem)
        if not match:
            raise ValueError(f"IEMOCAP filename does not parse: {path}")
        if stem not in labels:
            raise ValueError(
                f"{stem} has audio but no EmoEvaluation entry. Refusing to "
                "silently exclude it."
            )
        if stem in seen:
            raise ValueError(f"duplicate IEMOCAP utterance {stem}")
        seen.add(stem)
        session = int(match.group("session"))
        yield ParsedUtterance(
            path=path,
            utterance_id=f"iemocap/{stem}",
            # Two actors per session; the trailing F/M identifies which one
            # spoke this utterance. Grouping on session (the configured unit)
            # keeps both of a session's actors on the same side of a split.
            speaker_id=f"iemocap_Ses{session:02d}{match.group('speaker_gender')}",
            original_label=labels[stem],
            session_id=f"iemocap_session{session}",
            subset="improvised" if match.group("kind") == "impro" else "scripted",
        )


_ITERATORS = {
    "ravdess": _iter_ravdess,
    "cremad": _iter_cremad,
    "iemocap": _iter_iemocap,
}


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
        for parsed in _ITERATORS[corpus](root):
            path = parsed.path
            utterance_id = parsed.utterance_id
            original_label = parsed.original_label
            info = sf.info(str(path))
            corpus_rows.append(
                ManifestRow(
                    corpus=corpus,
                    file_path=path.as_posix(),
                    utterance_id=utterance_id,
                    speaker_id=parsed.speaker_id,
                    session_id=parsed.session_id,
                    subset=parsed.subset,
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


# --------------------------------------------------------------------------
# The portable manifest
# --------------------------------------------------------------------------
# data/manifest.csv records an absolute ``file_path``, which is correct for the
# machine that built it and useless anywhere else. So it is gitignored, and
# analysis that needs only speaker ids and labels became transitively
# unrunnable from a checkout. The portable form separates the two concerns:
#
#   SCIENTIFIC MANIFEST     what an utterance is: corpus, speaker, session,
#                           subset, raw and mapped labels, duration, sample
#                           rate, and the audio content hash. All portable,
#                           and tracked.
#
#   LOCAL AUDIO RESOLUTION  where the bytes are on this machine: a corpus-
#                           relative path plus configured roots, joined at the
#                           moment audio is actually read.
#
# Analysis reads the first and must never need the second. Feature extraction
# needs both, and is the only thing that does.
PORTABLE_COLUMNS = (
    "corpus",
    "relative_path",   # POSIX, relative to that corpus's configured root
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


def corpus_roots(config) -> Dict[str, Path]:
    """Configured audio root per corpus, absolute, for corpora that have one."""
    roots: Dict[str, Path] = {}
    for corpus, key in CORPUS_PATH_KEYS.items():
        value = getattr(config.paths, key, None)
        if value:
            roots[corpus] = config.resolve(value)
    return roots


def _relative_to_root(file_path: str, root: Path) -> str:
    """Corpus-relative POSIX path, or a failure that names the mismatch.

    Deliberately strict. Falling back to the basename when a path does not sit
    under its configured root would silently discard directory structure that
    another corpus layout may depend on, and would hide a misconfigured root.
    """
    absolute = Path(file_path)
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        # The same directory reached by a different spelling (case, separators,
        # a symlink) is ordinary on Windows; compare resolved forms before
        # giving up.
        try:
            return Path(os.path.realpath(absolute)).relative_to(
                Path(os.path.realpath(root))
            ).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"{file_path!r} is not under its configured corpus root {root}. "
                "The portable manifest cannot record a corpus-relative path for "
                "it; check paths.raw_* against the manifest that was built."
            ) from exc


def to_portable(rows: Sequence[ManifestRow], roots: Dict[str, Path]) -> List[Dict[str, str]]:
    """Scientific fields plus a corpus-relative path, deterministically ordered.

    Sorted by (corpus, utterance_id) so the artifact and its digest do not
    depend on the order the filesystem happened to be walked in.
    """
    out: List[Dict[str, str]] = []
    for row in rows:
        root = roots.get(row.corpus)
        if root is None:
            raise ValueError(f"no configured root for corpus {row.corpus!r}")
        out.append({
            "corpus": row.corpus,
            "relative_path": _relative_to_root(row.file_path, root),
            "utterance_id": row.utterance_id,
            "speaker_id": row.speaker_id,
            "session_id": row.session_id,
            "subset": row.subset,
            "original_label": row.original_label,
            "label_six": row.label_six,
            "label_four": row.label_four,
            "duration_s": f"{row.duration_s:.6f}",
            "sample_rate": str(row.sample_rate),
            "sha256": row.sha256,
        })
    out.sort(key=lambda record: (record["corpus"], record["utterance_id"]))
    return out


def write_portable_manifest(records: Iterable[Dict[str, str]], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PORTABLE_COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
            count += 1
    return count


def read_portable_manifest(path: Path) -> List[ManifestRow]:
    """Load the portable manifest as :class:`ManifestRow` objects.

    ``file_path`` on the returned rows holds the **corpus-relative** path, not
    an absolute one. Anything that needs to open the audio must go through
    :func:`resolve_audio_path`; analysis should not need it at all.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"portable manifest not found: {path}. Build it with "
            "`python tools/make_portable_manifest.py`."
        )
    rows: List[ManifestRow] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PORTABLE_COLUMNS:
            raise ValueError(
                f"portable manifest columns {reader.fieldnames} != "
                f"expected {list(PORTABLE_COLUMNS)}"
            )
        for record in reader:
            rows.append(ManifestRow(
                corpus=record["corpus"],
                file_path=record["relative_path"],
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
            ))
    return rows


def resolve_audio_path(row: ManifestRow, roots: Dict[str, Path]) -> Path:
    """Absolute path to this utterance's audio on this machine."""
    root = roots.get(row.corpus)
    if root is None:
        raise ValueError(f"no configured root for corpus {row.corpus!r}")
    candidate = Path(row.file_path)
    return candidate if candidate.is_absolute() else root / candidate


def load_for_analysis(config, *, prefer_portable: bool = True) -> List[ManifestRow]:
    """Manifest rows for analysis, preferring the tracked portable form.

    Analysis needs speaker ids and labels and never audio, so it should read the
    artifact a reviewer actually has. The machine-local manifest remains the
    fallback for a working copy that has built one but not yet exported the
    portable form.
    """
    portable = config.resolve(PORTABLE_MANIFEST)
    if prefer_portable and portable.exists():
        return read_portable_manifest(portable)
    return read_manifest(config.resolve(config.paths.manifest))
