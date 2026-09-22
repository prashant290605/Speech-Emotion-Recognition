"""The canonical 13-layer sweep artifact, and the merge that produces it.

The frame-dependence table and figure rest on 2340 sweep runs. Only 180 of them
share ids with the grid ledger, so until this artifact existed 2160 lived only
in a gitignored directory and that manuscript subsection could not be
reproduced from a checkout. These tests pin the packaging: that it preserves
every row, that it is deterministic, that it refuses a conflict rather than
picking a winner, and that its provenance sidecar describes what it was built
from.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from merge_layer_sweep import (  # noqa: E402
    EXPECTED_LAYERS,
    EXPECTED_ROWS,
    MergeError,
    check_against_grid,
    load_shards,
    serialise,
)

from ser.artifacts import LAYER_SWEEP, LAYER_SWEEP_PROVENANCE, read_layer_sweep  # noqa: E402
from ser.utils.results import field_disagreements  # noqa: E402

pytestmark = pytest.mark.ledger

ARTIFACT = REPO_ROOT / LAYER_SWEEP
PROVENANCE = REPO_ROOT / LAYER_SWEEP_PROVENANCE


@pytest.fixture(scope="module")
def artifact_rows():
    if not ARTIFACT.exists():
        pytest.skip(f"{LAYER_SWEEP} not built")
    rows, source = read_layer_sweep(REPO_ROOT)
    assert source == LAYER_SWEEP, "canonical artifact present but not preferred"
    return rows


@pytest.fixture(scope="module")
def provenance():
    if not PROVENANCE.exists():
        pytest.skip(f"{LAYER_SWEEP_PROVENANCE} not built")
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


# -- contents --------------------------------------------------------------
def test_row_count_and_layer_coverage(artifact_rows):
    assert len(artifact_rows) == EXPECTED_ROWS
    assert len({r["layer_index"] for r in artifact_rows}) == EXPECTED_LAYERS
    assert {r["status"] for r in artifact_rows} == {"ok"}
    assert {r["freeze_tag"] for r in artifact_rows} == {"grid-freeze-v3"}


def test_run_ids_are_unique(artifact_rows):
    ids = [r["run_id"] for r in artifact_rows]
    assert len(set(ids)) == len(ids)


def test_every_shard_row_survives_unchanged():
    """Packaging preserves rows, including the volatile provenance columns.

    Asserted as full dictionary equality rather than only on scientific fields:
    the artifact is meant to be the shards, not a summary of them, and a
    dropped column would be a silent loss of provenance.
    """
    if not ARTIFACT.exists():
        pytest.skip("artifact not built")
    shard_rows, _sources, _collapsed = load_shards(REPO_ROOT)
    canonical = {r["run_id"]: r for r in
                 (json.loads(line) for line in
                  ARTIFACT.read_text(encoding="utf-8").splitlines() if line.strip())}
    assert set(canonical) == set(shard_rows)
    assert all(canonical[i] == shard_rows[i] for i in shard_rows)


# -- determinism -----------------------------------------------------------
def test_merge_is_deterministic():
    """Same inputs, same bytes: sorted by run_id, fields in schema order."""
    if not ARTIFACT.exists():
        pytest.skip("artifact not built")
    rows, _sources, _collapsed = load_shards(REPO_ROOT)
    forwards = serialise(rows)
    backwards = serialise(dict(reversed(list(rows.items()))))
    assert forwards == backwards, "merge output depends on input order"
    on_disk = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert hashlib.sha256(forwards.encode("utf-8")).hexdigest() == on_disk, (
        "the artifact on disk is not what a fresh merge produces; "
        "rerun tools/merge_layer_sweep.py"
    )


def test_rows_are_sorted_by_run_id(artifact_rows):
    ids = [r["run_id"] for r in artifact_rows]
    assert ids == sorted(ids)


# -- the identity check the merge exists to make --------------------------
def test_rows_shared_with_the_grid_agree_on_every_nonvolatile_field():
    """180 ids appear in both the sweep and the frozen grid, and must match.

    This is not a merge convenience. Two rows carrying one run_id and different
    results would mean the 19 run_id coordinates do not determine the
    computation, which is a schema defect, not a data-cleaning problem.
    """
    if not ARTIFACT.exists():
        pytest.skip("artifact not built")
    rows, _sources, _collapsed = load_shards(REPO_ROOT)
    agreement = check_against_grid(rows, REPO_ROOT)
    assert agreement["shared"] == 180
    assert agreement["mismatches"] == 0
    assert agreement["fields_compared"] == 57


def test_a_conflicting_duplicate_is_refused(tmp_path):
    """Two different results under one id abort the merge, not one of them."""
    (tmp_path / "results" / "shards").mkdir(parents=True)
    base = {"run_id": "abc", "macro_f1": 0.5, "wall_seconds": 1.0, "status": "ok"}
    (tmp_path / "results" / "shards" / "sweep2_a.jsonl").write_text(
        json.dumps(base) + "\n", encoding="utf-8")
    (tmp_path / "results" / "shards" / "sweep2_b.jsonl").write_text(
        json.dumps({**base, "macro_f1": 0.9}) + "\n", encoding="utf-8")
    with pytest.raises(MergeError, match="different content"):
        load_shards(tmp_path)


def test_an_identical_duplicate_is_collapsed(tmp_path):
    """A restarted worker re-committing the same row is fine, and counted."""
    (tmp_path / "results" / "shards").mkdir(parents=True)
    base = {"run_id": "abc", "macro_f1": 0.5, "wall_seconds": 1.0, "status": "ok"}
    (tmp_path / "results" / "shards" / "sweep2_a.jsonl").write_text(
        json.dumps(base) + "\n", encoding="utf-8")
    (tmp_path / "results" / "shards" / "sweep2_b.jsonl").write_text(
        json.dumps({**base, "wall_seconds": 99.0}) + "\n", encoding="utf-8")
    rows, sources, collapsed = load_shards(tmp_path)
    assert len(rows) == 1 and collapsed == 1 and len(sources) == 2
    assert not field_disagreements(rows["abc"], base)


# -- provenance ------------------------------------------------------------
def test_provenance_records_every_source_shard_and_its_digest(provenance):
    assert provenance["rows"] == EXPECTED_ROWS
    assert provenance["sources"], "no source shards recorded"
    assert sum(s["rows"] for s in provenance["sources"]) == EXPECTED_ROWS
    for source in provenance["sources"]:
        assert len(source["sha256"]) == 64
        shard = REPO_ROOT / "results" / "shards" / source["shard"]
        if shard.exists():
            assert hashlib.sha256(shard.read_bytes()).hexdigest() == source["sha256"], \
                f"{source['shard']} has changed since the artifact was built"


def test_provenance_digest_matches_the_artifact_on_disk(provenance):
    assert provenance["artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()


def test_provenance_records_the_grid_agreement(provenance):
    assert provenance["grid_agreement"]["shared"] == 180
    assert provenance["grid_agreement"]["mismatches"] == 0
