#!/usr/bin/env python
"""Migrate results/runs.jsonl between schema versions.

The result schema is frozen, and `ser.utils.results` says a change requires a
version bump *and a migration*. This is that migration. Regenerating instead
would have worked for the cheap baseline rows, but it would also have replaced
their provenance -- new timestamps, a new git SHA -- for rows whose numbers did
not change. A migration keeps every `run_id` and every stamp intact and adds
only the new columns.

    python tools/migrate_results.py --dry-run
    python tools/migrate_results.py

v2 -> v3: adds cov_condition_number and cov_effective_rank, both null. No row
written before v3 formed a covariance, so null is the truthful value rather
than a placeholder.

The original file is copied to <name>.v<N>.bak before anything is written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.utils.results import SCHEMA_VERSION, FIELD_NAMES, validate_row  # noqa: E402

MIGRATIONS = {
    2: {"to": 3, "adds": {"cov_condition_number": None, "cov_effective_rank": None}},
}


def migrate_row(row: dict) -> dict:
    version = row.get("schema_version")
    while version in MIGRATIONS:
        step = MIGRATIONS[version]
        for name, default in step["adds"].items():
            row.setdefault(name, default)
        version = step["to"]
        row["schema_version"] = version
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="results/runs.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    path = REPO_ROOT / args.path
    if not path.exists():
        print(f"nothing to migrate: {path} does not exist")
        return 0

    original = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    versions = {row.get("schema_version") for row in original}
    print(f"{len(original)} rows at schema version(s) {sorted(versions)}")

    if versions == {SCHEMA_VERSION}:
        print(f"already at v{SCHEMA_VERSION}; nothing to do")
        return 0

    migrated = [migrate_row(dict(row)) for row in original]
    for index, row in enumerate(migrated, start=1):
        extra = set(row) - set(FIELD_NAMES)
        if extra:
            print(f"row {index}: unexpected field(s) {sorted(extra)}", file=sys.stderr)
            return 1
        try:
            validate_row(row)
        except Exception as exc:  # noqa: BLE001 - report and stop
            print(f"row {index} invalid after migration: {exc}", file=sys.stderr)
            return 1

    print(f"all {len(migrated)} rows validate at v{SCHEMA_VERSION}")
    if args.dry_run:
        print("dry run; nothing written")
        return 0

    backup = path.with_suffix(f".v{min(versions)}.bak")
    shutil.copy2(path, backup)
    print(f"backed up to {backup.name}")

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in migrated:
            handle.write(
                json.dumps({name: row[name] for name in FIELD_NAMES}, separators=(",", ":"))
                + "\n"
            )
    print(f"migrated {len(migrated)} rows in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
