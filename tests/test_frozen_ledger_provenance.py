"""The frozen-ledger damage detector.

``results/runs.jsonl`` is the provenance record every table, figure and the
retrospective audit is generated from, and until now nothing would have noticed
if it changed. These tests cover the detector added in :mod:`ser.freeze`: that
it verifies the real artifact, that it *fires* when the bytes move, and that its
scope stays narrow enough to remain switched on.

The detector is not a lock. It does not make the file read-only and it does not
prevent a future experiment from writing its own result file. It converts a
silent corruption into a loud failure, which is the property that was missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ser.freeze import (
    FROZEN_LEDGER,
    LEDGER_MARKER,
    LedgerDrift,
    assert_ledger_unchanged,
    expected_ledger_digest,
    ledger_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_marker_file_holds_a_single_well_formed_digest():
    path = REPO_ROOT / LEDGER_MARKER
    assert path.exists(), f"{LEDGER_MARKER} is missing"
    recorded = path.read_text(encoding="utf-8").strip()
    assert len(recorded) == 64
    assert all(c in "0123456789abcdef" for c in recorded)
    assert recorded == expected_ledger_digest()


@pytest.mark.ledger
def test_the_real_frozen_ledger_verifies():
    ledger = REPO_ROOT / FROZEN_LEDGER
    if not ledger.exists():
        pytest.skip("frozen ledger not present")
    assert assert_ledger_unchanged(ledger) == expected_ledger_digest()


def test_digest_is_over_bytes_and_is_streamed(tmp_path):
    """Chunked reading must give the same answer as hashing in one go."""
    import hashlib

    payload = (b'{"run_id":"a"}\n' * 200_000)  # comfortably past the 1 MiB chunk
    target = tmp_path / "runs.jsonl"
    target.write_bytes(payload)
    assert ledger_digest(target) == hashlib.sha256(payload).hexdigest()


def _repo_with_marker(tmp_path, digest, payload=b"row\n"):
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / LEDGER_MARKER).write_text(digest + "\n", encoding="utf-8")
    (tmp_path / FROZEN_LEDGER).write_bytes(payload)
    return tmp_path


def test_guard_fires_when_the_ledger_changes(tmp_path):
    """The point of the whole exercise: a changed byte is an error, loudly."""
    import hashlib

    payload = b"row\n"
    root = _repo_with_marker(tmp_path, hashlib.sha256(payload).hexdigest(), payload)
    assert assert_ledger_unchanged(root=root)  # unchanged: passes

    (root / FROZEN_LEDGER).write_bytes(payload + b"tampered\n")
    with pytest.raises(LedgerDrift) as excinfo:
        assert_ledger_unchanged(root=root)
    message = str(excinfo.value)
    assert "does not match its recorded digest" in message
    assert FROZEN_LEDGER in message


def test_guard_fires_on_truncation(tmp_path):
    """Truncation is the realistic accident -- a stray shell redirect."""
    import hashlib

    payload = b"a\nb\nc\n"
    root = _repo_with_marker(tmp_path, hashlib.sha256(payload).hexdigest(), payload)
    (root / FROZEN_LEDGER).write_bytes(b"")
    with pytest.raises(LedgerDrift):
        assert_ledger_unchanged(root=root)


def test_missing_expectation_is_an_error_by_default_and_optional_on_request(tmp_path):
    """An unverified ledger is not provenance, so the default is to refuse.

    ``require=False`` exists for a repository that has not recorded a digest
    yet; it returns an empty string rather than silently reporting success.
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / FROZEN_LEDGER).write_bytes(b"row\n")
    assert expected_ledger_digest(root=tmp_path) is None
    with pytest.raises(LedgerDrift):
        assert_ledger_unchanged(root=tmp_path)
    assert assert_ledger_unchanged(root=tmp_path, require=False) == ""


def test_empty_marker_counts_as_no_expectation(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / LEDGER_MARKER).write_text("   \n", encoding="utf-8")
    (tmp_path / FROZEN_LEDGER).write_bytes(b"row\n")
    assert expected_ledger_digest(root=tmp_path) is None


def test_guard_does_not_constrain_other_result_files(tmp_path):
    """Scope check: the detector names one artifact and only that one.

    A guard that complained about every new result file would be disabled
    within a week, so the narrowness is a feature and is asserted as one.
    """
    import hashlib

    payload = b"row\n"
    root = _repo_with_marker(tmp_path, hashlib.sha256(payload).hexdigest(), payload)
    (root / "results" / "new_experiment.jsonl").write_bytes(b"anything at all\n")
    (root / "results" / "calm_dropped_sensitivity.jsonl").write_bytes(b"and this\n")
    assert assert_ledger_unchanged(root=root)


@pytest.mark.ledger
def test_audit_tool_refuses_a_drifted_ledger(tmp_path):
    """End to end: the analysis entry point stops rather than reporting.

    Runs the real script against a copy of the real configuration whose input
    points at a deliberately corrupted copy of the ledger, in a temporary
    directory. The committed ledger is never opened for writing.
    """
    import subprocess
    import sys

    import yaml

    ledger = REPO_ROOT / FROZEN_LEDGER
    if not ledger.exists():
        pytest.skip("frozen ledger not present")

    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / LEDGER_MARKER).write_text(expected_ledger_digest() + "\n", encoding="utf-8")
    # One line short of the real ledger: eligible pairs would still be found,
    # so only the digest can catch this.
    lines = ledger.read_bytes().split(b"\n")
    (tmp_path / FROZEN_LEDGER).write_bytes(b"\n".join(lines[:-2]) + b"\n")

    config = yaml.safe_load((REPO_ROOT / "configs" / "audit_translation.yaml").read_text("utf-8"))
    config_path = tmp_path / "configs" / "audit_translation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "audit_translation.py"),
         "--config", str(config_path)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode != 0
    assert "does not match its recorded digest" in completed.stderr
    assert not (tmp_path / "reports" / "translation_audit.json").exists()
