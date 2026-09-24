"""IEMOCAP manifest parsing, against a fixture that mirrors the real release.

IEMOCAP is the one corpus in this project whose labels are not in its
filenames. They live in ``dialog/EmoEvaluation/<dialog>.txt``, whose summary
line already carries the majority vote across annotators, with ``xxx`` where no
majority exists -- which is exactly the configured
``majority_vote_discard_disagreement`` policy. Reading anything else, such as
the per-evaluator files under ``EmoEvaluation/Categorical/``, would silently
implement a different policy.

The corpus is not present on this machine, so these tests build a fixture with
the real directory layout, the real summary-line format, the real utterance-id
grammar and the real three-letter codes. That is enough to pin parsing,
grouping, the label policy and the failure modes; it is not a substitute for
running against the release, which is recorded as an outstanding data-access
requirement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ser.labels import IEMOCAP_RAW_LABELS, LabelPolicy, map_label
from ser.manifest import (
    CORPUS_EXPECTATIONS,
    IEMOCAP_EMOTION_CODES,
    _iemocap_annotations,
    _iter_iemocap,
)

# One line per utterance, tab-separated, exactly as the release writes it.
SUMMARY = "[{start} - {end}]\t{utt}\t{code}\t[{v}, {a}, {d}]"


def write_release(root: Path, dialogs):
    """Build a miniature IEMOCAP tree: dialog/EmoEvaluation + sentences/wav."""
    for session, dialog, utterances in dialogs:
        evaluation = root / f"Session{session}" / "dialog" / "EmoEvaluation"
        evaluation.mkdir(parents=True, exist_ok=True)
        lines = [
            "% [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]",
            "",
        ]
        audio_dir = root / f"Session{session}" / "sentences" / "wav" / dialog
        audio_dir.mkdir(parents=True, exist_ok=True)
        for index, (utt, code) in enumerate(utterances):
            lines.append(SUMMARY.format(start=f"{index}.0000", end=f"{index + 1}.0000",
                                        utt=utt, code=code, v="2.5", a="2.5", d="2.5"))
            lines.append(f"C-E1:\t{code};\t()")
            lines.append("")
            (audio_dir / f"{utt}.wav").write_bytes(b"")
        (evaluation / f"{dialog}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def release(tmp_path):
    write_release(tmp_path, [
        (1, "Ses01F_impro01", [
            ("Ses01F_impro01_F000", "neu"),
            ("Ses01F_impro01_M000", "ang"),
            ("Ses01F_impro01_F001", "xxx"),
            ("Ses01F_impro01_M001", "exc"),
            ("Ses01F_impro01_F002", "fru"),
        ]),
        (1, "Ses01M_script01_1", [
            ("Ses01M_script01_1_M000", "sad"),
            ("Ses01M_script01_1_F000", "hap"),
        ]),
        (5, "Ses05M_impro03", [
            ("Ses05M_impro03_M000", "oth"),
            ("Ses05M_impro03_F000", "sur"),
        ]),
    ])
    return tmp_path


@pytest.fixture()
def parsed(release):
    return list(_iter_iemocap(release))


# -- parsing ---------------------------------------------------------------
def test_every_annotated_utterance_is_parsed(parsed):
    assert len(parsed) == 9


def test_utterance_ids_are_unique_and_namespaced(parsed):
    ids = [p.utterance_id for p in parsed]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("iemocap/") for i in ids)


def test_parsing_is_deterministic(release):
    first = [tuple(p) for p in _iter_iemocap(release)]
    second = [tuple(p) for p in _iter_iemocap(release)]
    assert first == second
    assert first == sorted(first, key=lambda r: str(r[0]))


def test_audio_paths_exist(parsed):
    assert all(p.path.exists() for p in parsed)


def test_labels_come_from_annotations_not_filenames(parsed):
    by_id = {p.utterance_id: p.original_label for p in parsed}
    assert by_id["iemocap/Ses01F_impro01_F000"] == "neutral"
    assert by_id["iemocap/Ses01F_impro01_M000"] == "angry"
    assert by_id["iemocap/Ses01F_impro01_M001"] == "excited"
    assert by_id["iemocap/Ses01F_impro01_F001"] == "xxx"


def test_only_declared_raw_labels_are_produced(parsed):
    assert {p.original_label for p in parsed} <= set(IEMOCAP_RAW_LABELS)


def test_emotion_codes_cover_the_declared_vocabulary():
    assert set(IEMOCAP_EMOTION_CODES.values()) == set(IEMOCAP_RAW_LABELS)
    assert len(IEMOCAP_EMOTION_CODES) == 11


# -- session, speaker and subset -------------------------------------------
def test_session_ids_come_from_the_dialog_name(parsed):
    by_id = {p.utterance_id: p.session_id for p in parsed}
    assert by_id["iemocap/Ses01F_impro01_F000"] == "iemocap_session1"
    assert by_id["iemocap/Ses05M_impro03_M000"] == "iemocap_session5"


def test_speaker_is_the_utterance_speaker_not_the_dialog_leader(parsed):
    """Ses01F_impro01_M000 is spoken by session 1's *male* actor.

    Taking the gender from the dialog name instead would put both actors of a
    session under one id and make speaker-level reasoning wrong.
    """
    by_id = {p.utterance_id: p.speaker_id for p in parsed}
    assert by_id["iemocap/Ses01F_impro01_F000"] == "iemocap_Ses01F"
    assert by_id["iemocap/Ses01F_impro01_M000"] == "iemocap_Ses01M"
    assert by_id["iemocap/Ses01M_script01_1_F000"] == "iemocap_Ses01F"


def test_two_speakers_per_session(parsed):
    sessions = {}
    for p in parsed:
        sessions.setdefault(p.session_id, set()).add(p.speaker_id)
    assert sessions["iemocap_session1"] == {"iemocap_Ses01F", "iemocap_Ses01M"}


def test_improvised_and_scripted_are_distinguished(parsed):
    by_id = {p.utterance_id: p.subset for p in parsed}
    assert by_id["iemocap/Ses01F_impro01_F000"] == "improvised"
    assert by_id["iemocap/Ses01M_script01_1_M000"] == "scripted"
    assert {p.subset for p in parsed} == {"improvised", "scripted"}


def test_every_speaker_belongs_to_exactly_one_session(parsed):
    """The property session-level grouping depends on."""
    owner = {}
    for p in parsed:
        owner.setdefault(p.speaker_id, set()).add(p.session_id)
    assert all(len(v) == 1 for v in owner.values())


# -- the four-class mapping ------------------------------------------------
@pytest.fixture(scope="module")
def policy():
    from ser.config import load_config

    return LabelPolicy.from_config(load_config())


def test_four_class_mapping_matches_the_configured_policy(parsed, policy):
    mapped = {p.utterance_id: map_label("iemocap", p.original_label, "four", policy)
              for p in parsed}
    assert mapped["iemocap/Ses01F_impro01_F000"] == "neutral"
    assert mapped["iemocap/Ses01F_impro01_M000"] == "angry"
    assert mapped["iemocap/Ses01M_script01_1_M000"] == "sad"
    assert mapped["iemocap/Ses01M_script01_1_F000"] == "happy"


def test_excited_merges_into_happy_under_the_configured_policy(parsed, policy):
    assert map_label("iemocap", "excited", "four", policy) == "happy"


@pytest.mark.parametrize("raw", ["frustrated", "xxx", "other", "surprised"])
def test_unusable_labels_are_excluded_not_reassigned(raw, policy):
    assert map_label("iemocap", raw, "four", policy) is None


def test_exclusion_is_explicit_rather_than_an_unknown_label(policy):
    """None means deliberately excluded; an unknown label must raise."""
    with pytest.raises(ValueError, match="unrecognised raw label"):
        map_label("iemocap", "bored", "four", policy)


# -- failure modes ---------------------------------------------------------
def test_missing_annotation_directory_is_refused(tmp_path):
    (tmp_path / "Session1" / "sentences" / "wav" / "Ses01F_impro01").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="EmoEvaluation"):
        list(_iter_iemocap(tmp_path))


def test_audio_without_an_annotation_is_refused(release):
    orphan = release / "Session1" / "sentences" / "wav" / "Ses01F_impro01"
    (orphan / "Ses01F_impro01_F009.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="no EmoEvaluation entry"):
        list(_iter_iemocap(release))


def test_an_unknown_emotion_code_is_refused(tmp_path):
    write_release(tmp_path, [(1, "Ses01F_impro01", [("Ses01F_impro01_F000", "zzz")])])
    with pytest.raises(ValueError, match="unrecognised IEMOCAP emotion code"):
        list(_iter_iemocap(tmp_path))


def test_an_unparseable_filename_is_refused(release):
    bad = release / "Session1" / "sentences" / "wav" / "Ses01F_impro01" / "not_an_iemocap_id.wav"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="does not parse"):
        list(_iter_iemocap(release))


def test_contradictory_annotations_are_refused(tmp_path):
    write_release(tmp_path, [
        (1, "Ses01F_impro01", [("Ses01F_impro01_F000", "neu")]),
        (1, "Ses01F_impro02", [("Ses01F_impro01_F000", "ang")]),
    ])
    with pytest.raises(ValueError, match="annotated as both"):
        _iemocap_annotations(tmp_path)


def test_per_evaluator_files_are_not_read_as_summaries(tmp_path):
    """Categorical/ holds a different policy's labels and must be ignored."""
    write_release(tmp_path, [(1, "Ses01F_impro01", [("Ses01F_impro01_F000", "neu")])])
    categorical = tmp_path / "Session1" / "dialog" / "EmoEvaluation" / "Categorical"
    categorical.mkdir(parents=True)
    (categorical / "Ses01F_impro01_e1_cat.txt").write_text(
        SUMMARY.format(start="0.0", end="1.0", utt="Ses01F_impro01_F000",
                       code="ang", v="1", a="1", d="1") + "\n",
        encoding="utf-8")
    assert _iemocap_annotations(tmp_path) == {"Ses01F_impro01_F000": "neutral"}


# -- published size --------------------------------------------------------
def test_published_size_is_recorded_so_a_partial_release_fails():
    expectation = CORPUS_EXPECTATIONS["iemocap"]
    assert expectation == {"files": 10039, "speakers": 10}


def test_the_fixture_is_not_mistaken_for_a_complete_release(release):
    from ser.manifest import CountMismatch, verify_expected_counts, ManifestRow

    rows = [ManifestRow(corpus="iemocap", file_path=str(p.path),
                        utterance_id=p.utterance_id, speaker_id=p.speaker_id,
                        session_id=p.session_id, subset=p.subset,
                        original_label=p.original_label, label_six="", label_four="",
                        duration_s=1.0, sample_rate=16000, sha256="0" * 64)
            for p in _iter_iemocap(release)]
    with pytest.raises(CountMismatch):
        verify_expected_counts("iemocap", rows)
