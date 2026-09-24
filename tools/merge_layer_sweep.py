#!/usr/bin/env python
"""Package the 13-layer sweep shards into one tracked, canonical result file.

    python tools/merge_layer_sweep.py --check     # verify, write nothing
    python tools/merge_layer_sweep.py             # write the canonical artifact

Why this exists. The frame-dependence analysis rests on 2340 sweep runs that
lived only in ``results/shards/``, which is gitignored on the stated grounds
that shard output is "merged into runs.jsonl, then redundant". For these rows
that was never true: only 180 of them share ids with the grid, so 2160 existed
nowhere a clean clone could reach, and one manuscript subsection was therefore
not reproducible from a checkout. This tool packages them, without recomputing
anything.

**It is packaging, not analysis.** Every row is copied through unchanged at the
scientific-field level. Nothing is recomputed, no model is fitted, and
``results/runs.jsonl`` is neither read for merging nor written.

Four things are verified before anything is written, and any of them failing
aborts the whole operation rather than producing a partial file:

1. every ``run_id`` is unique across the shard set, or duplicates are
   byte-equivalent on every non-volatile field;
2. the total row count matches the expected count;
3. rows sharing an id with ``results/runs.jsonl`` agree on every non-volatile
   field, which is a real check of whether the run_id coordinates determine
   the computation;
4. the merge is deterministic: sorted by ``run_id``, with rows serialised in
   the frozen schema's field order.

The sidecar ``results/layer_sweep_v2.provenance.json`` records each source
shard's name, row count and SHA256, plus the digest of the artifact itself, so
the packaging step can be re-verified later without the shards being present.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.utils.results import (  # noqa: E402
    FIELD_NAMES,
    field_disagreements,
    validate_row,
)

SHARD_GLOB = "results/shards/sweep2_*.jsonl"
CANONICAL = "results/layer_sweep_v2.jsonl"
PROVENANCE = "results/layer_sweep_v2.provenance.json"
GRID = "results/runs.jsonl"

# The sweep as designed: 13 layers x 3 backbones x 6 rungs x 5 seeds x 2
# directions. Stated rather than derived, so a missing shard is a loud failure
# instead of a smaller experiment that still looks complete.
EXPECTED_ROWS = 2340
EXPECTED_LAYERS = 13


class MergeError(RuntimeError):
    """The shards cannot be packaged as they stand."""


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_shards(root: Path):
    """Return (rows_by_id, per-shard provenance). Raises on a real conflict."""
    rows: dict[str, dict] = {}
    origin: dict[str, str] = {}
    sources = []
    collapsed = 0
    for path in sorted(root.glob(SHARD_GLOB.replace("results/", "results/"))):
        text = path.read_bytes()
        count = 0
        for line in text.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            run_id = row["run_id"]
            if run_id in rows:
                # A restarted worker re-committing an identical row is fine and
                # is collapsed. Two different results under one id is not: it
                # would mean the coordinates do not determine the computation.
                disagreements = field_disagreements(rows[run_id], row)
                if disagreements:
                    raise MergeError(
                        f"run_id {run_id} appears in {origin[run_id]} and "
                        f"{path.name} with different content: "
                        + ", ".join(f"{k}={v[0]!r} vs {v[1]!r}"
                                    for k, v in sorted(disagreements.items()))
                    )
                collapsed += 1
                continue
            rows[run_id] = row
            origin[run_id] = path.name
        sources.append({"shard": path.name, "rows": count,
                        "sha256": hashlib.sha256(text).hexdigest()})
    if not sources:
        raise MergeError(f"no shards matched {SHARD_GLOB}")
    return rows, sources, collapsed


def check_against_grid(rows: dict[str, dict], root: Path):
    """Rows sharing an id with the frozen grid must describe the same run."""
    grid_path = root / GRID
    if not grid_path.exists():
        return {"shared": 0, "fields_compared": 0, "values_compared": 0,
                "mismatches": 0, "note": "grid ledger absent; check skipped"}
    grid: dict[str, dict] = {}
    with open(grid_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["run_id"] in rows:
                    grid[row["run_id"]] = row
    shared = sorted(grid)
    if not shared:
        return {"shared": 0, "fields_compared": 0, "values_compared": 0, "mismatches": 0}

    problems = {}
    compared = 0
    for run_id in shared:
        disagreements = field_disagreements(grid[run_id], rows[run_id])
        compared = max(compared, sum(
            1 for k in grid[run_id]
            if k not in {"timestamp", "wall_seconds", "hostname", "git_dirty",
                         "git_sha", "predictions_path", "run_started_utc",
                         "python_version", "library_versions_json"}
        ))
        if disagreements:
            problems[run_id] = disagreements
    if problems:
        first = next(iter(sorted(problems)))
        raise MergeError(
            f"{len(problems)} of {len(shared)} rows shared with {GRID} disagree "
            f"with the grid on a non-volatile field; e.g. {first}: "
            + ", ".join(f"{k}={v[0]!r} vs {v[1]!r}"
                        for k, v in sorted(problems[first].items()))
            + ". This is not a merge conflict. It means the run_id coordinates "
              "do not determine the computation. Stop and investigate."
        )
    return {"shared": len(shared), "fields_compared": compared,
            "values_compared": len(shared) * compared, "mismatches": 0}


def serialise(rows: dict[str, dict]) -> str:
    """Deterministic: sorted by run_id, fields in frozen schema order."""
    lines = []
    for run_id in sorted(rows):
        row = rows[run_id]
        validate_row(row)
        lines.append(json.dumps({name: row[name] for name in FIELD_NAMES},
                                separators=(",", ":")))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify and report; write nothing")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    root = args.root

    rows, sources, collapsed = load_shards(root)
    print(f"shards read: {len(sources)}")
    for source in sources:
        print(f"  {source['shard']:<28} {source['rows']:>5} rows  {source['sha256'][:16]}")
    print(f"unique run_ids: {len(rows)} (identical duplicates collapsed: {collapsed})")

    if len(rows) != EXPECTED_ROWS:
        raise MergeError(
            f"expected {EXPECTED_ROWS} unique rows, found {len(rows)}. A shard "
            "is missing or an extra one is present; refusing to write a "
            "canonical artifact that silently describes a different experiment."
        )
    layers = sorted({r["layer_index"] for r in rows.values()})
    if len(layers) != EXPECTED_LAYERS:
        raise MergeError(f"expected {EXPECTED_LAYERS} layers, found {layers}")
    statuses = {r["status"] for r in rows.values()}
    if statuses != {"ok"}:
        raise MergeError(f"expected every row to be ok, found statuses {statuses}")

    agreement = check_against_grid(rows, root)
    print(f"shared with {GRID}: {agreement['shared']} ids, "
          f"{agreement['fields_compared']} non-volatile fields, "
          f"{agreement['values_compared']} values, "
          f"{agreement['mismatches']} mismatches")

    payload = serialise(rows)
    encoded = payload.encode("utf-8")
    artifact_digest = hashlib.sha256(encoded).hexdigest()

    target = root / CANONICAL
    if args.check:
        if not target.exists():
            print(f"CHECK: {CANONICAL} does not exist yet")
            return 1
        current = _digest(target)
        ok = current == artifact_digest
        print(f"CHECK: {CANONICAL} {'matches' if ok else 'DIFFERS FROM'} a fresh merge")
        print(f"  on disk {current}\n  merged  {artifact_digest}")
        return 0 if ok else 1

    target.write_bytes(encoded)
    provenance = {
        "artifact": CANONICAL,
        "artifact_sha256": artifact_digest,
        "rows": len(rows),
        "layers": layers,
        "freeze_tags": sorted({r["freeze_tag"] for r in rows.values()}),
        "schema_versions": sorted({r["schema_version"] for r in rows.values()}),
        "identical_duplicates_collapsed": collapsed,
        "sources": sources,
        "grid_agreement": agreement,
        "generator": "tools/merge_layer_sweep.py",
        "note": ("Packaging of existing frozen rows. No run was executed and no "
                 "value was recomputed. The source shards are retained."),
    }
    (root / PROVENANCE).write_text(json.dumps(provenance, indent=2) + "\n",
                                   encoding="utf-8")
    print(f"wrote {CANONICAL} ({len(encoded)} bytes)\n  sha256 {artifact_digest}")
    print(f"wrote {PROVENANCE}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MergeError as exc:
        print(f"MERGE REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
