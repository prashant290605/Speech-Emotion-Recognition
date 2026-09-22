#!/usr/bin/env python
"""Package the CORAL shrinkage probe into one tracked, canonical result file.

    python tools/merge_eps_probe.py --check     # verify, write nothing
    python tools/merge_eps_probe.py             # write the canonical artifact

The epsilon-asymptote table and figure rest on 120 probe runs. Only 35 of them
were tracked; the other 85 lived in gitignored ``results/shards/eps_*.jsonl``,
so the table could not be regenerated from a checkout. This has the same shape
as the layer-sweep gap and the same fix.

Packaging, not analysis: rows are copied through unchanged, nothing is
recomputed, and neither the existing ``results/eps_asymptote.jsonl`` nor
``results/runs.jsonl`` is modified. The sources are retained.
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

from ser.artifacts import EPS_PROBE, EPS_PROBE_PROVENANCE, EPS_PROBE_SOURCES  # noqa: E402
from ser.utils.results import FIELD_NAMES, field_disagreements, validate_row  # noqa: E402

# The probe as designed and as the manuscript reports it.
EXPECTED_ROWS = 120


class MergeError(RuntimeError):
    """The probe files cannot be packaged as they stand."""


def load_sources(root: Path):
    rows: dict[str, dict] = {}
    origin: dict[str, str] = {}
    sources = []
    collapsed = 0
    target_name = Path(EPS_PROBE).name
    for pattern in EPS_PROBE_SOURCES:
        for path in sorted(glob.glob(str(root / pattern))):
            path = Path(path)
            if path.name == target_name:
                continue  # never fold a previous build of the artifact into itself
            text = path.read_bytes()
            count = 0
            for line in text.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                run_id = row["run_id"]
                if run_id in rows:
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
            sources.append({
                "source": path.relative_to(root).as_posix(),
                "rows": count,
                "sha256": hashlib.sha256(text).hexdigest(),
            })
    if not sources:
        raise MergeError(f"no probe files matched {EPS_PROBE_SOURCES}")
    return rows, sources, collapsed


def serialise(rows: dict[str, dict]) -> str:
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
    parser.add_argument("--check", action="store_true", help="verify only; write nothing")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    root = args.root

    rows, sources, collapsed = load_sources(root)
    print(f"sources read: {len(sources)}")
    for source in sources:
        print(f"  {source['source']:<40} {source['rows']:>4} rows  {source['sha256'][:16]}")
    print(f"unique run_ids: {len(rows)} (identical duplicates collapsed: {collapsed})")

    if len(rows) != EXPECTED_ROWS:
        raise MergeError(
            f"expected {EXPECTED_ROWS} unique probe rows, found {len(rows)}. "
            "A source file is missing; refusing to write a canonical artifact "
            "that silently describes a smaller probe."
        )
    statuses = {r["status"] for r in rows.values()}
    if statuses != {"ok"}:
        raise MergeError(f"expected every row to be ok, found statuses {statuses}")

    payload = serialise(rows).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    target = root / EPS_PROBE

    if args.check:
        if not target.exists():
            print(f"CHECK: {EPS_PROBE} does not exist yet")
            return 1
        current = hashlib.sha256(target.read_bytes()).hexdigest()
        ok = current == digest
        print(f"CHECK: {EPS_PROBE} {'matches' if ok else 'DIFFERS FROM'} a fresh merge")
        return 0 if ok else 1

    target.write_bytes(payload)
    (root / EPS_PROBE_PROVENANCE).write_text(json.dumps({
        "artifact": EPS_PROBE,
        "artifact_sha256": digest,
        "rows": len(rows),
        "epsilons": sorted({r["alignment_eps"] for r in rows.values()
                            if r.get("alignment_eps") is not None}),
        "freeze_tags": sorted({str(r["freeze_tag"]) for r in rows.values()}),
        "identical_duplicates_collapsed": collapsed,
        "sources": sources,
        "generator": "tools/merge_eps_probe.py",
        "note": ("Packaging of existing frozen rows. No run was executed and no "
                 "value recomputed. The source files are retained."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {EPS_PROBE} ({len(payload)} bytes)\n  sha256 {digest}")
    print(f"wrote {EPS_PROBE_PROVENANCE}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MergeError as exc:
        print(f"MERGE REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
