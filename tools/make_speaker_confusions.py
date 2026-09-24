#!/usr/bin/env python
"""Derive the compact per-speaker confusion artifact from stored predictions.

    python tools/make_speaker_confusions.py --verify   # build in memory, prove equivalence
    python tools/make_speaker_confusions.py            # write the artifact

The cluster bootstraps behind every interval in the paper read per-utterance
predictions only to build per-speaker confusion matrices, then never touch them
again. Those matrices are therefore the sufficient statistic, and they are
small: K x K counts per speaker rather than one label per utterance.

This exists because ``results/predictions/`` is 7730 gitignored files, which
made the published intervals unreproducible from a checkout. Nothing here is
recomputed from a model; it is a projection of stored predictions onto the
statistic the analysis actually consumes. ``results/predictions/`` is not
deleted and remains the original source.

``--verify`` recomputes macro-F1, per-class F1, the pooled confusion matrix and
a full deterministic bootstrap both ways -- from predictions, and from the
compact artifact -- and requires them to agree exactly before the artifact is
treated as a substitute.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.config import load_config  # noqa: E402
from ser.manifest import load_for_analysis  # noqa: E402
from ser.phase8 import (  # noqa: E402
    confusion_by_group,
    load_predictions,
    macro_f1_from_confusion,
    paired_cluster_bootstrap,
    per_class_f1_from_confusion,
)
from ser.speaker_stats import SPEAKER_CONFUSIONS, build_record, load, write_records  # noqa: E402
from ser.utils.results import read_rows  # noqa: E402

RESULTS = REPO_ROOT / "results/runs.jsonl"
PROVENANCE = "results/speaker_confusions.provenance.json"


def from_predictions(row, label, speaker, index, classes):
    """The original path: predictions -> per-speaker confusions."""
    ids, predicted = load_predictions(RESULTS, row)
    names = sorted({speaker[u] for u in ids})
    lookup = {n: i for i, n in enumerate(names)}
    tensor = confusion_by_group(
        [index[label[u]] for u in ids],
        [index[p] for p in predicted],
        [lookup[speaker[u]] for u in ids],
        len(classes), len(names),
    )
    return names, tensor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="prove equivalence against the prediction files; write nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N runs (development aid)")
    args = parser.parse_args()

    config = load_config()
    manifest = load_for_analysis(config)
    label = {r.utterance_id: r.label_six for r in manifest}
    speaker = {r.utterance_id: r.speaker_id for r in manifest}

    rows = [r for r in read_rows(RESULTS)
            if r["status"] == "ok" and r.get("predictions_path")]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("no rows with stored predictions", file=sys.stderr)
        return 2
    classes = list(rows[0]["class_names"])
    index = {name: i for i, name in enumerate(classes)}

    records, tensors = [], {}
    for n, row in enumerate(rows, 1):
        if list(row["class_names"]) != classes:
            raise SystemExit(f"{row['run_id']}: unexpected class_names {row['class_names']}")
        names, tensor = from_predictions(row, label, speaker, index, classes)
        tensors[row["run_id"]] = (names, tensor)
        records.append(build_record(
            row["run_id"], classes, names, tensor,
            source_corpus=row["source_corpus"], target_corpus=row["target_corpus"],
            seed=row["seed"],
        ))
        if n % 500 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

    print(f"runs projected: {len(records)}")

    # -- equivalence ------------------------------------------------------
    reloaded = {r["run_id"]: r for r in records}
    worst_macro = worst_class = 0.0
    conf_exact = tensor_exact = 0
    for run_id, (names, tensor) in tensors.items():
        record = reloaded[run_id]
        k = len(classes)
        back = np.asarray(record["confusions"], dtype=np.int64).reshape(len(names), k, k)
        if np.array_equal(back, tensor):
            tensor_exact += 1
        if record["speakers"] != names:
            raise SystemExit(f"{run_id}: speaker ordering not preserved")
        pooled_a, pooled_b = tensor.sum(axis=0), back.sum(axis=0)
        if np.array_equal(pooled_a, pooled_b):
            conf_exact += 1
        worst_macro = max(worst_macro, abs(macro_f1_from_confusion(pooled_a)
                                           - macro_f1_from_confusion(pooled_b)))
        fa, fb = per_class_f1_from_confusion(pooled_a), per_class_f1_from_confusion(pooled_b)
        worst_class = max(worst_class, float(np.nanmax(np.abs(fa - fb))))

    print(f"per-speaker tensors identical: {tensor_exact}/{len(tensors)}")
    print(f"pooled confusions identical:   {conf_exact}/{len(tensors)}")
    print(f"max |macro-F1 difference|:     {worst_macro:.3e}")
    print(f"max |per-class F1 difference|: {worst_class:.3e}")

    # A real paired bootstrap, both ways, under identical rng seeds.
    by_pair = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("freeze_tag") != "grid-freeze-v3" or row.get("blending") != "none":
            continue
        key = (row["source_corpus"], row["target_corpus"], row["alignment"])
        by_pair[key][row["seed"]].append(row["run_id"])

    checked = 0
    worst_boot = 0.0
    for (source, target, rung) in sorted(by_pair):
        if rung not in ("none", "mean_shift"):
            continue
        other = (source, target, "none" if rung == "mean_shift" else "mean_shift")
        if other not in by_pair:
            continue
        def arm(spec, source_of):
            return {seed: [source_of(rid) for rid in sorted(ids)]
                    for seed, ids in spec.items()}
        groups = {seed: len(tensors[ids[0]][0])
                  for seed, ids in by_pair[(source, target, rung)].items()}
        a_pred = arm(by_pair[(source, target, rung)], lambda r: tensors[r][1])
        b_pred = arm(by_pair[other], lambda r: tensors[r][1])
        a_art = arm(by_pair[(source, target, rung)],
                    lambda r: np.asarray(reloaded[r]["confusions"], dtype=np.int64)
                    .reshape(len(reloaded[r]["speakers"]), len(classes), len(classes)))
        b_art = arm(by_pair[other],
                    lambda r: np.asarray(reloaded[r]["confusions"], dtype=np.int64)
                    .reshape(len(reloaded[r]["speakers"]), len(classes), len(classes)))
        one = paired_cluster_bootstrap(a_pred, b_pred, groups, n_boot=200, seed=17)
        two = paired_cluster_bootstrap(a_art, b_art, groups, n_boot=200, seed=17)
        for key in ("diff", "lo", "hi", "p"):
            worst_boot = max(worst_boot, abs(one[key] - two[key]))
        checked += 1
    print(f"paired bootstraps compared:    {checked}")
    print(f"max |bootstrap difference|:    {worst_boot:.3e}")

    equivalent = (tensor_exact == len(tensors) and conf_exact == len(tensors)
                  and worst_macro == 0.0 and worst_class == 0.0 and worst_boot == 0.0)
    print(f"EXACT EQUIVALENCE: {'yes' if equivalent else 'NO'}")
    if not equivalent:
        print("refusing to treat the artifact as a substitute", file=sys.stderr)
        return 1
    if args.verify:
        return 0

    count, digest = write_records(records, REPO_ROOT / SPEAKER_CONFUSIONS)
    size = (REPO_ROOT / SPEAKER_CONFUSIONS).stat().st_size
    (REPO_ROOT / PROVENANCE).write_text(json.dumps({
        "artifact": SPEAKER_CONFUSIONS,
        "payload_sha256": digest,
        "runs": count,
        "classes": classes,
        "compressed_bytes": size,
        "derived_from": "results/predictions/ (retained, not tracked)",
        "generator": "tools/make_speaker_confusions.py",
        "equivalence": {
            "per_speaker_tensors_identical": tensor_exact,
            "pooled_confusions_identical": conf_exact,
            "max_abs_macro_f1_difference": worst_macro,
            "max_abs_per_class_f1_difference": worst_class,
            "paired_bootstraps_compared": checked,
            "max_abs_bootstrap_difference": worst_boot,
        },
        "note": ("Sufficient statistic for every published cluster-bootstrap "
                 "interval. Per-utterance identity is NOT preserved; anything "
                 "needing it must read results/predictions/."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SPEAKER_CONFUSIONS} ({size} bytes, payload sha256 {digest})")
    print(f"wrote {PROVENANCE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
