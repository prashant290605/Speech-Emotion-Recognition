#!/usr/bin/env python
"""Merge Stage 1 shard files into results/runs.jsonl.

    python tools/merge_shards.py --dry-run
    python tools/merge_shards.py

Refuses to merge if the shards disagree: a ``run_id`` appearing in two shards
with different content means the sharding was not disjoint, and silently keeping
one of them would hide that. Identical duplicates are fine (a restarted worker
can legitimately re-commit an identical row) and are collapsed.

Nothing is overwritten: rows already in ``results/runs.jsonl`` are kept, and only
ids not already present are appended.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.utils.results import (  # noqa: E402
    FIELD_NAMES,
    append_row,
    completed_run_ids,
    read_rows,
    validate_row,
)


def _content_key(row: dict) -> str:
    """Everything except provenance that varies between identical re-runs."""
    volatile = {"timestamp", "wall_seconds", "hostname", "git_dirty"}
    return json.dumps(
        {k: v for k, v in sorted(row.items()) if k not in volatile}, sort_keys=True
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="results/shards/*.jsonl")
    parser.add_argument("--into", default="results/runs.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    shard_files = sorted(glob.glob(str(REPO_ROOT / args.pattern)))
    if not shard_files:
        print(f"no shard files matching {args.pattern}")
        return 0

    by_id: dict[str, list[dict]] = defaultdict(list)
    per_file = Counter()
    for path in shard_files:
        for row in read_rows(path, validate=True):
            by_id[row["run_id"]].append(row)
            per_file[Path(path).name] += 1

    for name, count in per_file.items():
        print(f"  {name:<40} {count:>5} rows")
    print(f"\n{sum(per_file.values())} rows across {len(shard_files)} shards, "
          f"{len(by_id)} distinct run_ids")

    conflicts = {
        run_id: rows
        for run_id, rows in by_id.items()
        if len({_content_key(r) for r in rows}) > 1
    }
    if conflicts:
        print(
            f"\nREFUSING TO MERGE: {len(conflicts)} run_id(s) appear in more than one "
            "shard with DIFFERENT content, e.g. "
            f"{sorted(conflicts)[:3]}.\nThe shards were not disjoint. Keeping one "
            "arbitrarily would hide that.",
            file=sys.stderr,
        )
        return 2

    target = REPO_ROOT / args.into
    existing = completed_run_ids(target)
    new = {run_id: rows[0] for run_id, rows in by_id.items() if run_id not in existing}

    print(f"{len(existing)} rows already in {args.into}; {len(new)} to append")
    if args.dry_run:
        print("dry run; nothing written")
        return 0

    for row in new.values():
        validate_row(row)
        append_row(target, row)

    merged = list(read_rows(target, validate=True))
    ids = [r["run_id"] for r in merged]
    unique = len(set(ids)) == len(ids)
    print(f"\nmerged: {len(merged)} rows, {len(set(ids))} unique run_ids  "
          f"{'OK' if unique else 'DUPLICATES PRESENT'}")
    return 0 if unique else 1


if __name__ == "__main__":
    raise SystemExit(main())
