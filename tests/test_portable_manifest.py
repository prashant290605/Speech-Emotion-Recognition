"""The tracked scientific manifest, and the separation it enforces.

``data/manifest.csv`` records absolute audio paths, so it cannot be committed,
and analysis that needs only speaker ids and labels inherited that restriction.
The portable manifest carries the scientific metadata with a corpus-relative
path; audio resolution happens against configured roots at extraction time.

The test that matters most here is the one asserting no machine-local path
survives into the tracked file. Everything else guards the claim that the
export changed the representation and not the science.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from ser.config import load_config  # noqa: E402
from ser.manifest import (  # noqa: E402
    CORPUS_EXPECTATIONS,
    PORTABLE_COLUMNS,
    PORTABLE_MANIFEST,
    ManifestRow,
    corpus_roots,
    load_for_analysis,
    read_portable_manifest,
    resolve_audio_path,
    to_portable,
)

PORTABLE = REPO_ROOT / PORTABLE_MANIFEST

# Anything that would pin the file to one machine or one operating system.
ABSOLUTE_PATH = re.compile(
    r"(^|,)\s*(?:[A-Za-z]:[\\/]|\\\\|/home/|/Users/|/mnt/|/media/|/var/|/opt/)",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def rows():
    if not PORTABLE.exists():
        pytest.skip(f"{PORTABLE_MANIFEST} not built")
    return read_portable_manifest(PORTABLE)


# -- the portability property ---------------------------------------------
def test_no_absolute_path_appears_anywhere_in_the_tracked_manifest():
    """Line by line, so a failure names the row rather than the file."""
    if not PORTABLE.exists():
        pytest.skip("not built")
    offenders = []
    with open(PORTABLE, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if ABSOLUTE_PATH.search(line):
                offenders.append((number, line.strip()[:120]))
    assert not offenders, f"absolute paths in {PORTABLE_MANIFEST}: {offenders[:3]}"


def test_relative_paths_use_posix_separators_and_are_relative(rows):
    for row in rows:
        assert "\\" not in row.file_path, row.file_path
        assert not Path(row.file_path).is_absolute(), row.file_path
        assert not row.file_path.startswith("/"), row.file_path


def test_no_username_leaks_into_the_tracked_manifest():
    """A home-directory name is machine-identifying even without a drive letter."""
    if not PORTABLE.exists():
        pytest.skip("not built")
    text = PORTABLE.read_text(encoding="utf-8")
    home = Path.home().name
    assert home.lower() not in text.lower(), f"home directory name {home!r} present"


# -- corpus membership is unchanged ---------------------------------------
def test_row_and_speaker_counts_match_the_published_sizes(rows):
    for corpus, expectation in CORPUS_EXPECTATIONS.items():
        present = [r for r in rows if r.corpus == corpus]
        if not present:
            continue
        assert len(present) == expectation["files"], corpus
        assert len({r.speaker_id for r in present}) == expectation["speakers"], corpus


def test_utterance_ids_are_unique(rows):
    ids = [r.utterance_id for r in rows]
    assert len(set(ids)) == len(ids)


def test_columns_are_the_declared_portable_schema():
    if not PORTABLE.exists():
        pytest.skip("not built")
    with open(PORTABLE, "r", encoding="utf-8", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    assert header == PORTABLE_COLUMNS
    assert "file_path" not in header, "the absolute-path column must not survive"


def test_labels_match_the_local_manifest_where_one_exists(rows):
    """The export must not have changed any label or corpus assignment."""
    config = load_config()
    local = config.resolve(config.paths.manifest)
    if not local.exists():
        pytest.skip("local manifest absent; nothing to compare against")
    from ser.manifest import read_manifest

    reference = {r.utterance_id: r for r in read_manifest(local)}
    assert set(reference) == {r.utterance_id for r in rows}
    for row in rows:
        other = reference[row.utterance_id]
        assert (row.corpus, row.speaker_id, row.original_label,
                row.label_six, row.label_four, row.sha256) == \
               (other.corpus, other.speaker_id, other.original_label,
                other.label_six, other.label_four, other.sha256)


# -- audio resolution is a separate step ----------------------------------
def test_resolution_rejoins_a_relative_path_to_its_configured_root():
    roots = {"ravdess": Path("/data/RAVDESS")}
    row = ManifestRow(
        corpus="ravdess", file_path="Actor_01/03-01-01-01-01-01-01.wav",
        utterance_id="ravdess/x", speaker_id="ravdess_01", session_id="", subset="",
        original_label="neutral", label_six="neutral", label_four="neutral",
        duration_s=1.0, sample_rate=16000, sha256="0" * 64,
    )
    assert resolve_audio_path(row, roots) == Path("/data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav")


def test_resolution_leaves_an_absolute_path_alone():
    """The local manifest's rows still work, so extraction is unaffected."""
    roots = {"ravdess": Path("/data/RAVDESS")}
    row = ManifestRow(
        corpus="ravdess", file_path="/elsewhere/a.wav", utterance_id="ravdess/x",
        speaker_id="ravdess_01", session_id="", subset="", original_label="neutral",
        label_six="neutral", label_four="neutral", duration_s=1.0,
        sample_rate=16000, sha256="0" * 64,
    )
    assert resolve_audio_path(row, roots) == Path("/elsewhere/a.wav")


def test_resolution_refuses_an_unconfigured_corpus():
    row = ManifestRow(
        corpus="nowhere", file_path="a.wav", utterance_id="x", speaker_id="s",
        session_id="", subset="", original_label="neutral", label_six="neutral",
        label_four="neutral", duration_s=1.0, sample_rate=16000, sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="no configured root"):
        resolve_audio_path(row, {})


def test_export_refuses_a_path_outside_its_corpus_root():
    """A misconfigured root must fail, not silently become a basename."""
    row = ManifestRow(
        corpus="ravdess", file_path="/somewhere/else/a.wav", utterance_id="x",
        speaker_id="s", session_id="", subset="", original_label="neutral",
        label_six="neutral", label_four="neutral", duration_s=1.0,
        sample_rate=16000, sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="not under its configured corpus root"):
        to_portable([row], {"ravdess": Path("/data/RAVDESS")})


# -- analysis reads the tracked form --------------------------------------
def test_analysis_loader_prefers_the_portable_manifest():
    if not PORTABLE.exists():
        pytest.skip("not built")
    config = load_config()
    loaded = load_for_analysis(config)
    assert len(loaded) == len(read_portable_manifest(PORTABLE))
    assert all(not Path(r.file_path).is_absolute() for r in loaded)


def test_analysis_loader_needs_no_audio_on_disk():
    """Metadata only: nothing in the analysis path touches a wav file."""
    if not PORTABLE.exists():
        pytest.skip("not built")
    config = load_config()
    roots = corpus_roots(config)
    assert roots, "no corpus roots configured"
    rows = load_for_analysis(config)
    assert {r.corpus for r in rows} <= set(roots)
    assert all(r.speaker_id and r.utterance_id for r in rows)
