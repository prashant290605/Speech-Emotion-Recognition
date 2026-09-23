#!/usr/bin/env python
"""Evaluate the pre-committed IEMOCAP subset rule on metadata. Fits nothing.

    python tools/prospective_counts.py

The prospective protocol leaves exactly one decision open: improvised-only
versus improvised-plus-scripted. Making that choice after seeing an outcome
would destroy prospectivity, so the rule and its thresholds are fixed in
``configs/prospective_translation_v1.yaml`` and this script only evaluates
them.

**It reads labels, sessions and subsets. It does not load a feature, fit a
model, or compute any target score** -- counting utterances by class is
metadata inspection, which the protocol explicitly permits.

The output is written next to the other prospective artifacts so the freeze
commit can record the counts the decision was taken on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.config import load_config  # noqa: E402
from ser.labels import LabelPolicy, map_label  # noqa: E402
from ser.manifest import corpus_roots, read_manifest  # noqa: E402
from ser.prospective import load_protocol, subset_decision  # noqa: E402


def counts_from_manifest(rows, policy, label_space):
    """subset -> class -> session -> count, after the four-class mapping."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for row in rows:
        if row.corpus != "iemocap":
            continue
        mapped = map_label(row.corpus, row.original_label, label_space, policy)
        if mapped is None:
            continue  # deliberately excluded by the frozen label policy
        session = int(row.session_id.rsplit("session", 1)[-1]) if row.session_id else 0
        out[row.subset or "unknown"][mapped][session] += 1
    return {s: {c: dict(sessions) for c, sessions in classes.items()}
            for s, classes in out.items()}


def merge_subsets(counts):
    """The 'both' branch: improvised and scripted pooled."""
    merged = defaultdict(lambda: defaultdict(int))
    for classes in counts.values():
        for label, sessions in classes.items():
            for session, n in sessions.items():
                merged[label][session] += n
    return {c: dict(s) for c, s in merged.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest to count (default: the configured one)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the counts (default: the protocol's output root)")
    args = parser.parse_args()

    config = load_config()
    protocol = load_protocol()
    policy = LabelPolicy.from_config(config)

    manifest_path = args.manifest or config.resolve(config.paths.manifest)
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        print("Build it with `ser manifest` once IEMOCAP is in place.", file=sys.stderr)
        return 2

    rows = read_manifest(manifest_path)
    if not any(r.corpus == "iemocap" for r in rows):
        roots = corpus_roots(config)
        print("the manifest contains no IEMOCAP rows.", file=sys.stderr)
        print(f"Expected the release at {roots.get('iemocap')}.", file=sys.stderr)
        return 2

    counts = counts_from_manifest(rows, policy, protocol["label_space"])
    decision = subset_decision(counts, protocol)

    print(f"label space: {protocol['label_space']}  policy: {config.label_map_hash}")
    for subset in sorted(counts):
        total = sum(sum(s.values()) for s in counts[subset].values())
        print(f"\n{subset}  ({total} utterances after mapping)")
        for label in sorted(counts[subset]):
            sessions = counts[subset][label]
            per = " ".join(f"s{s}={n}" for s, n in sorted(sessions.items()))
            print(f"  {label:<9} {sum(sessions.values()):>5}   {per}")

    print(f"\npooled (both subsets):")
    for label, sessions in sorted(merge_subsets(counts).items()):
        print(f"  {label:<9} {sum(sessions.values()):>5}")

    print(f"\nrule: {decision['rule']}")
    print(f"thresholds: {decision['thresholds']}")
    if decision["failed_checks"]:
        print("failed checks:")
        for reason in decision["failed_checks"]:
            print(f"  - {reason}")
    print(f"\nDECISION: {decision['chosen']}")
    print("Record this in configs/prospective_translation_v1.yaml "
          "(iemocap_subset.decided) before freezing.")

    out = args.out or (REPO_ROOT / protocol["output_root"] / "subset_counts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": protocol["protocol"],
        "label_space": protocol["label_space"],
        "label_map_hash": config.label_map_hash,
        "manifest": str(manifest_path.name),
        "decision": decision,
        "pooled": merge_subsets(counts),
        "note": ("Metadata only. No feature was loaded, no model fitted and no "
                 "target score computed to produce this file."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
