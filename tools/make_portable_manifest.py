#!/usr/bin/env python
"""Export the machine-local manifest as a tracked, portable one.

    python tools/make_portable_manifest.py --check   # verify, write nothing
    python tools/make_portable_manifest.py

The local manifest records absolute audio paths, so it cannot be committed and
analysis that needs only speaker ids and labels inherited that restriction.
This writes ``data/manifest_portable.csv``: the same scientific fields, with a
corpus-relative path in place of the absolute one, sorted deterministically.

Nothing about corpus membership or label mapping changes here. The published
corpus sizes and speaker counts are re-verified against
``CORPUS_EXPECTATIONS`` before anything is written, so an export built from a
partial download fails rather than becoming a smaller experiment that still
looks complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.config import load_config  # noqa: E402
from ser.labels import LabelPolicy, map_label  # noqa: E402
from ser.manifest import (  # noqa: E402
    CORPUS_EXPECTATIONS,
    PORTABLE_MANIFEST,
    corpus_roots,
    read_manifest,
    read_portable_manifest,
    to_portable,
    write_portable_manifest,
)

PROVENANCE = "data/manifest_portable.provenance.json"


class ExportError(RuntimeError):
    """The manifest cannot be exported as it stands."""


def verify(rows) -> dict:
    """Counts, speakers and label-map inputs, checked before export."""
    report = {}
    for corpus, expectation in CORPUS_EXPECTATIONS.items():
        present = [r for r in rows if r.corpus == corpus]
        if not present:
            continue
        speakers = {r.speaker_id for r in present}
        if len(present) != expectation["files"]:
            raise ExportError(
                f"{corpus}: {len(present)} rows, expected {expectation['files']}"
            )
        if len(speakers) != expectation["speakers"]:
            raise ExportError(
                f"{corpus}: {len(speakers)} speakers, expected {expectation['speakers']}"
            )
        report[corpus] = {"rows": len(present), "speakers": len(speakers)}

    ids = Counter(r.utterance_id for r in rows)
    duplicates = [k for k, v in ids.items() if v > 1]
    if duplicates:
        raise ExportError(f"{len(duplicates)} duplicate utterance_ids, e.g. {duplicates[:3]}")
    return report


def verify_label_mapping(rows, config) -> dict:
    """Recompute every mapped label from its raw label and require agreement.

    The export copies ``label_six``/``label_four`` across rather than
    recomputing them, so this is the check that the copied values are the ones
    the current policy produces. A disagreement means the tracked manifest
    would carry a label mapping the code no longer implements.
    """
    policy = LabelPolicy.from_config(config)
    mismatches = 0
    for row in rows:
        for space, stored in (("six", row.label_six), ("four", row.label_four)):
            expected = map_label(row.corpus, row.original_label, space, policy) or ""
            if expected != stored:
                mismatches += 1
    if mismatches:
        raise ExportError(
            f"{mismatches} stored label values disagree with the current label "
            "policy. The manifest was built under a different policy; rebuild it "
            "with `ser manifest` before exporting."
        )
    return {"label_map_hash": config.label_map_hash, "checked": len(rows) * 2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = parser.parse_args()

    config = load_config()
    local = config.resolve(config.paths.manifest)
    target = config.resolve(PORTABLE_MANIFEST)

    if not local.exists():
        if args.check and target.exists():
            rows = read_portable_manifest(target)
            report = verify(rows)
            print(f"CHECK: {PORTABLE_MANIFEST} present, {len(rows)} rows, "
                  f"{report} (local manifest absent; contents not re-derived)")
            return 0
        raise ExportError(
            f"local manifest {local} not found and cannot be re-derived without "
            "the raw corpora. Run `ser manifest` on a machine that has them."
        )

    rows = read_manifest(local)
    report = verify(rows)
    labels = verify_label_mapping(rows, config)
    records = to_portable(rows, corpus_roots(config))

    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    payload = buffer.getvalue().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    print(f"rows: {len(records)}")
    for corpus, stats in sorted(report.items()):
        print(f"  {corpus:<8} {stats['rows']:>5} files  {stats['speakers']:>3} speakers")
    print(f"label policy: {labels['label_map_hash']} ({labels['checked']} values checked)")
    print(f"sha256: {digest}")

    if args.check:
        if not target.exists():
            print(f"CHECK: {PORTABLE_MANIFEST} does not exist yet")
            return 1
        current = hashlib.sha256(target.read_bytes()).hexdigest()
        ok = current == digest
        print(f"CHECK: {PORTABLE_MANIFEST} {'matches' if ok else 'DIFFERS FROM'} a fresh export")
        return 0 if ok else 1

    written = write_portable_manifest(records, target)
    (config.resolve(PROVENANCE)).write_text(json.dumps({
        "artifact": PORTABLE_MANIFEST,
        "sha256": digest,
        "rows": written,
        "corpora": report,
        "label_map_hash": labels["label_map_hash"],
        "generator": "tools/make_portable_manifest.py",
        "note": ("Scientific metadata only. Audio paths are corpus-relative and "
                 "are resolved against paths.raw_* at extraction time. No label "
                 "mapping or corpus membership is changed by this export."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {PORTABLE_MANIFEST} ({written} rows) and {PROVENANCE}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExportError as exc:
        print(f"EXPORT REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
