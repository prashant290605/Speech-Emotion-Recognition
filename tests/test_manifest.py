"""Manifest construction, count verification, and the A8 prior check."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ser.config import load_config
from ser.datastats import (
    EXPECTED_PRIOR_KL,
    KL_TOLERANCE,
    class_counts,
    js_distance,
    kl_divergence,
    pairwise_prior_shift,
    prior_vector,
    verify_against_a8,
)
from ser.labels import LabelPolicy
from ser.manifest import (
    CORPUS_EXPECTATIONS,
    MANIFEST_COLUMNS,
    CountMismatch,
    ManifestRow,
    build_manifest,
    read_manifest,
    verify_expected_counts,
    write_manifest,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def policy(config):
    return LabelPolicy.from_config(config)


def _write_wav(path, seconds=0.2, sr=16000):
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(seconds * sr), dtype="float32"), sr)


# -- building --------------------------------------------------------------
def test_builds_rows_from_ravdess_filenames(tmp_path, policy):
    root = tmp_path / "RAVDESS"
    # 03=audio-only, 01=speech, 05=angry, ..., actor 12
    _write_wav(root / "Actor_12" / "03-01-05-01-01-01-12.wav")
    _write_wav(root / "Actor_12" / "03-01-02-01-01-01-12.wav")  # calm

    rows = build_manifest({"ravdess": root}, policy, verify_counts=False)
    by_label = {r.original_label: r for r in rows}

    assert set(by_label) == {"angry", "calm"}
    assert by_label["angry"].speaker_id == "ravdess_12"
    assert by_label["angry"].utterance_id == "ravdess/03-01-05-01-01-01-12"
    assert by_label["angry"].sample_rate == 16000
    assert by_label["angry"].duration_s == pytest.approx(0.2, abs=1e-3)
    assert len(by_label["angry"].sha256) == 64
    # calm merges into neutral under the settled decision
    assert by_label["calm"].label_six == "neutral"


def test_ravdess_song_channel_is_excluded(tmp_path, policy):
    """Channel 02 is song, a separate download; mixing it in would corrupt counts."""
    root = tmp_path / "RAVDESS"
    _write_wav(root / "Actor_01" / "03-01-05-01-01-01-01.wav")  # speech
    _write_wav(root / "Actor_01" / "03-02-05-01-01-01-01.wav")  # song

    rows = build_manifest({"ravdess": root}, policy, verify_counts=False)
    assert len(rows) == 1
    assert rows[0].utterance_id.endswith("03-01-05-01-01-01-01")


def test_builds_rows_from_cremad_filenames(tmp_path, policy):
    root = tmp_path / "CREMA-D"
    _write_wav(root / "AudioWAV" / "1001_DFA_ANG_XX.wav")

    rows = build_manifest({"cremad": root}, policy, verify_counts=False)
    assert rows[0].speaker_id == "cremad_1001"
    assert rows[0].original_label == "angry"
    assert rows[0].label_six == "angry"
    assert rows[0].label_four == "angry"


def test_unparseable_filename_raises(tmp_path, policy):
    root = tmp_path / "CREMA-D"
    _write_wav(root / "AudioWAV" / "not_a_valid_name.wav")
    with pytest.raises(ValueError, match="does not parse"):
        build_manifest({"cremad": root}, policy, verify_counts=False)


def test_missing_root_raises(tmp_path, policy):
    with pytest.raises(FileNotFoundError):
        build_manifest({"ravdess": tmp_path / "absent"}, policy, verify_counts=False)


# -- count verification ----------------------------------------------------
def _row(speaker):
    return ManifestRow(
        corpus="ravdess",
        file_path="x.wav",
        utterance_id="ravdess/x",
        speaker_id=speaker,
        session_id="",
        subset="",
        original_label="angry",
        label_six="angry",
        label_four="angry",
        duration_s=1.0,
        sample_rate=16000,
        sha256="0" * 64,
    )


def test_short_corpus_halts():
    """A partial download must stop the build, not shrink the experiment."""
    with pytest.raises(CountMismatch, match="does not match its published size"):
        verify_expected_counts("ravdess", [_row(f"s{i}") for i in range(100)])


def test_correct_corpus_passes():
    expected = CORPUS_EXPECTATIONS["ravdess"]
    rows = [
        _row(f"ravdess_{i % expected['speakers']:02d}") for i in range(expected["files"])
    ]
    verify_expected_counts("ravdess", rows)


def test_unknown_corpus_is_not_checked():
    verify_expected_counts("emodb", [])


# -- round trip ------------------------------------------------------------
def test_write_read_round_trip(tmp_path):
    rows = [_row("ravdess_01"), _row("ravdess_02")]
    path = tmp_path / "manifest.csv"
    assert write_manifest(rows, path) == 2

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(MANIFEST_COLUMNS)
    assert read_manifest(path) == rows


def test_read_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `ser manifest`"):
        read_manifest(tmp_path / "absent.csv")


# -- divergence maths ------------------------------------------------------
def test_kl_of_identical_priors_is_zero():
    assert kl_divergence([0.25] * 4, [0.25] * 4) == pytest.approx(0.0)


def test_kl_is_asymmetric_and_js_is_not():
    p, q = [0.7, 0.3], [0.4, 0.6]
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))
    assert js_distance(p, q) == pytest.approx(js_distance(q, p))


def test_kl_is_infinite_when_the_target_lacks_a_class():
    """Not smoothed: a class absent from the target is a finding."""
    assert math.isinf(kl_divergence([0.5, 0.5], [1.0, 0.0]))


def test_prior_vector_returns_none_without_support():
    from collections import Counter

    assert prior_vector(Counter(), ["a", "b"]) is None


# -- against the real manifest ---------------------------------------------
@pytest.fixture(scope="module")
def manifest_rows(config):
    path = config.resolve(config.paths.manifest)
    if not path.exists():
        pytest.skip("manifest not built yet")
    return read_manifest(path)


def test_real_corpora_match_published_sizes(manifest_rows):
    for corpus, expectation in CORPUS_EXPECTATIONS.items():
        rows = [r for r in manifest_rows if r.corpus == corpus]
        if not rows:
            continue
        assert len(rows) == expectation["files"]
        assert len({r.speaker_id for r in rows}) == expectation["speakers"]


def test_utterance_ids_are_unique(manifest_rows):
    ids = [r.utterance_id for r in manifest_rows]
    assert len(ids) == len(set(ids))


def test_ravdess_is_not_balanced_at_the_six_class_intersection(manifest_rows):
    """The draft's "RAVDESS is exactly balanced" line is wrong: it is balanced at
    8 classes, not at the 6-class intersection after calm merges into neutral."""
    if not any(r.corpus == "ravdess" for r in manifest_rows):
        pytest.skip("RAVDESS not in manifest")
    counts = class_counts(manifest_rows, "ravdess", "six")
    assert counts["neutral"] == 288
    assert counts["angry"] == 192
    assert len(set(counts.values())) > 1


def test_manifest_priors_agree_with_amendment_a8(config, manifest_rows):
    """A8's near-zero prior shift reframed all of Phase 9. Verify it against
    real data rather than the published counts it was derived from."""
    present = sorted({r.corpus for r in manifest_rows})
    shifts = pairwise_prior_shift(manifest_rows, config, present)
    assert shifts, "no pairs to check"
    assert verify_against_a8(shifts) == []

    for shift in shifts:
        expected = EXPECTED_PRIOR_KL.get((shift["source"], shift["target"]))
        if expected is not None:
            assert abs(shift["kl"] - expected) <= KL_TOLERANCE
