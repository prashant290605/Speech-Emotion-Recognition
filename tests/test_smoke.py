"""Phase 0 acceptance: the reproducibility spine works end to end.

This is the smoke test the phase brief requires. It runs the real CLI command
against a temporary results file and checks that a row makes the full round
trip: config -> seed -> provenance -> schema validation -> JSONL -> re-read.
"""

from __future__ import annotations

import json

from ser.cli import main
from ser.utils.results import SCHEMA_VERSION, SMOKE_CORPUS, FIELD_NAMES, is_smoke_row, read_rows


def test_smoke_writes_one_valid_row(tmp_path):
    out = tmp_path / "runs.jsonl"

    assert main(["smoke", "--out", str(out)]) == 0

    rows = list(read_rows(out, validate=True))
    assert len(rows) == 1

    row = rows[0]
    assert row["schema_version"] == SCHEMA_VERSION
    assert set(row) == set(FIELD_NAMES)
    assert row["status"] == "ok"
    assert row["source_corpus"] == SMOKE_CORPUS
    assert is_smoke_row(row)

    # Provenance is populated, not placeholder.
    assert row["run_id"]
    assert len(row["config_hash"]) == 64
    assert row["timestamp"].endswith("Z")
    assert json.loads(row["lib_versions_json"])

    # Structured columns are parseable JSON of the right shape.
    per_class = json.loads(row["per_class_f1_json"])
    confusion = json.loads(row["confusion_json"])
    assert set(per_class) == set(row["class_names"])
    assert len(confusion) == row["n_classes"]
    assert all(len(r) == row["n_classes"] for r in confusion)


def test_smoke_is_append_only(tmp_path):
    """A second invocation adds a row; it never rewrites the file."""
    out = tmp_path / "runs.jsonl"

    main(["smoke", "--out", str(out)])
    first = out.read_text(encoding="utf-8")

    main(["smoke", "--out", str(out)])
    second = out.read_text(encoding="utf-8")

    assert second.startswith(first)
    assert len(list(read_rows(out, validate=True))) == 2


def test_unbuilt_phase_commands_exit_two(capsys):
    """Running a future phase fails loudly and names the phase that owns it."""
    assert main(["select"]) == 2
    assert "Phase 8" in capsys.readouterr().err


def test_inventory_runs(capsys):
    assert main(["inventory"]) == 0
    out = capsys.readouterr().out
    assert "INVENTORY" in out
    assert "legacy assets" in out
