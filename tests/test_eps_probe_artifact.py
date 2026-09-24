"""The packaged CORAL shrinkage probe.

Same gap as the layer sweep, found the same way: the clean-clone simulation
regenerated a different epsilon-asymptote table because 85 of the probe's 120
rows lived only in gitignored ``results/shards/eps_*.jsonl``. These tests pin
the packaging and, in particular, that the artifact carries the whole probe
rather than the tracked third of it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from merge_eps_probe import EXPECTED_ROWS, MergeError, load_sources, serialise  # noqa: E402

from ser.artifacts import EPS_PROBE, EPS_PROBE_PROVENANCE, read_eps_probe  # noqa: E402

pytestmark = pytest.mark.ledger

ARTIFACT = REPO_ROOT / EPS_PROBE
PROVENANCE = REPO_ROOT / EPS_PROBE_PROVENANCE


@pytest.fixture(scope="module")
def rows():
    if not ARTIFACT.exists():
        pytest.skip(f"{EPS_PROBE} not built")
    loaded, source = read_eps_probe(REPO_ROOT)
    assert source == EPS_PROBE, "canonical artifact present but not preferred"
    return loaded


def test_the_whole_probe_is_present_not_only_the_tracked_part(rows):
    """120 rows. The previously tracked ledger held 35 of them."""
    assert len(rows) == EXPECTED_ROWS
    assert {r["status"] for r in rows} == {"ok"}
    tracked = REPO_ROOT / "results" / "eps_asymptote.jsonl"
    if tracked.exists():
        partial = sum(1 for line in tracked.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        assert partial < len(rows), "artifact is no larger than the partial ledger"


def test_run_ids_are_unique_and_sorted(rows):
    ids = [r["run_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_every_source_row_survives_unchanged():
    if not ARTIFACT.exists():
        pytest.skip("not built")
    source_rows, _sources, _collapsed = load_sources(REPO_ROOT)
    canonical = {r["run_id"]: r for r in
                 (json.loads(line) for line in
                  ARTIFACT.read_text(encoding="utf-8").splitlines() if line.strip())}
    assert set(canonical) == set(source_rows)
    assert all(canonical[i] == source_rows[i] for i in source_rows)


def test_merge_is_deterministic():
    if not ARTIFACT.exists():
        pytest.skip("not built")
    source_rows, _sources, _collapsed = load_sources(REPO_ROOT)
    forwards = serialise(source_rows)
    backwards = serialise(dict(reversed(list(source_rows.items()))))
    assert forwards == backwards
    assert hashlib.sha256(forwards.encode("utf-8")).hexdigest() == \
        hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()


def test_the_artifact_is_never_folded_into_itself():
    """A rebuild must not read its own previous output as a source.

    Without the guard the glob ``results/eps_*.jsonl`` matches the artifact,
    and every rebuild would be merging the artifact with its own inputs. It
    would still produce the right rows, which is exactly why this is worth
    asserting rather than trusting.
    """
    if not ARTIFACT.exists():
        pytest.skip("not built")
    _rows, sources, _collapsed = load_sources(REPO_ROOT)
    names = {s["source"] for s in sources}
    assert EPS_PROBE not in names
    assert "results/eps_asymptote.jsonl" in names


def test_provenance_records_sources_and_digest():
    if not PROVENANCE.exists():
        pytest.skip("not built")
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["rows"] == EXPECTED_ROWS
    assert provenance["artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert sum(s["rows"] for s in provenance["sources"]) == EXPECTED_ROWS
    for source in provenance["sources"]:
        path = REPO_ROOT / source["source"]
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_a_conflicting_duplicate_is_refused(tmp_path):
    (tmp_path / "results" / "shards").mkdir(parents=True)
    base = {"run_id": "abc", "macro_f1": 0.5, "wall_seconds": 1.0, "status": "ok"}
    (tmp_path / "results" / "eps_a.jsonl").write_text(
        json.dumps(base) + "\n", encoding="utf-8")
    (tmp_path / "results" / "shards" / "eps_b.jsonl").write_text(
        json.dumps({**base, "macro_f1": 0.9}) + "\n", encoding="utf-8")
    with pytest.raises(MergeError, match="different content"):
        load_sources(tmp_path)
