#!/usr/bin/env python
"""Phase 10: which emotions transfer and which collapse.

    python tools/phase10_per_class.py

Writes reports/per_class.md and reports/per_class.json.

Everything is computed from the **stored per-utterance predictions**, not from
the per-class columns on the result rows, so precision, recall and F1 all come
from one confusion matrix and cannot disagree with each other.

Intervals are the same paired cluster bootstrap Phase 8 used: target_test
speakers and seeds resampled together. A per-class interval that resampled
utterances would be far too narrow, because errors on one talker are strongly
correlated -- and per-class support is small enough that this matters more here
than anywhere else in the project.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.config import load_config  # noqa: E402
from ser.manifest import read_manifest  # noqa: E402
from ser.phase8 import confusion_by_group, load_predictions  # noqa: E402
from ser.utils.results import read_rows  # noqa: E402

RESULTS = REPO_ROOT / "results/runs.jsonl"
STAGE2_TAG = "grid-freeze-v3"
N_BOOT = 2000
PAIRS = [("ravdess", "cremad"), ("cremad", "ravdess")]


def per_class_scores(conf: np.ndarray) -> dict:
    """Precision, recall, F1 and support from one confusion matrix."""
    tp = np.diag(conf).astype(float)
    support = conf.sum(axis=1).astype(float)   # true
    predicted = conf.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1e-12), np.nan)
        recall = np.where(support > 0, tp / np.maximum(support, 1e-12), np.nan)
        f1 = np.where(support + predicted > 0,
                      2 * tp / np.maximum(support + predicted, 1e-12), np.nan)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support,
            "predicted": predicted}


def main() -> int:
    config = load_config()
    manifest = read_manifest(config.resolve(config.paths.manifest))
    label = {r.utterance_id: r.label_six for r in manifest}
    speaker = {r.utterance_id: r.speaker_id for r in manifest}

    rows = [r for r in read_rows(RESULTS)
            if r["freeze_tag"] == STAGE2_TAG and r["blending"] == "none"
            and r["status"] == "ok"]
    classes = list(rows[0]["class_names"])
    index = {name: i for i, name in enumerate(classes)}

    payload = {"classes": classes, "n_boot": N_BOOT, "pairs": {}}
    out = ["# Phase 10 — per-class transfer", ""]
    out.append("Computed from the stored per-utterance predictions of the "
               "**validated** configuration in each (pair, seed): the run with the "
               "best `source_val`, which is what the protocol would actually "
               "deploy. Precision, recall and F1 all come from the same confusion "
               "matrix, so they cannot disagree.\n")
    out.append(f"Intervals are a paired cluster bootstrap over target_test speakers "
               f"and seeds, {N_BOOT} replicates -- the same scheme as Phase 8. "
               "Per-class support is small, so an utterance-level interval would be "
               "badly over-confident here.\n")
    out.append("---\n")

    for source, target in PAIRS:
        pool = [r for r in rows
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        by_seed = defaultdict(list)
        for r in pool:
            by_seed[r["seed"]].append(r)

        chosen, tensors, n_speakers = {}, {}, {}
        for seed in sorted(by_seed):
            best = max(by_seed[seed], key=lambda r: r["selection_source_val_macro_f1"])
            ids, predicted = load_predictions(RESULTS, best)
            names = sorted({speaker[u] for u in ids})
            lookup = {n: i for i, n in enumerate(names)}
            tensors[seed] = confusion_by_group(
                [index[label[u]] for u in ids],
                [index[p] for p in predicted],
                [lookup[speaker[u]] for u in ids],
                len(classes), len(names),
            )
            n_speakers[seed] = len(names)
            chosen[seed] = best

        seeds = sorted(tensors)
        observed = per_class_scores(sum(t.sum(axis=0) for t in tensors.values()))

        rng = np.random.default_rng(17)
        draws = {k: np.empty((N_BOOT, len(classes))) for k in ("precision", "recall", "f1")}
        for b in range(N_BOOT):
            seed_draw = rng.choice(seeds, size=len(seeds), replace=True)
            total = np.zeros((len(classes), len(classes)))
            for s in seed_draw:
                n = n_speakers[s]
                weights = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
                total += np.tensordot(weights, tensors[s], axes=(0, 0))
            scores = per_class_scores(total)
            for key in draws:
                draws[key][b] = scores[key]

        out.append(f"## {source} -> {target}\n")
        out.append(f"Validated configuration per seed, {len(seeds)} seeds. "
                   f"Chance floor {np.mean([r['chance_macro_f1'] for r in pool]):.4f}, "
                   f"majority floor {np.mean([r['majority_macro_f1'] for r in pool]):.4f}.\n")
        out.append("| class | support | predicted | precision | recall | F1 |")
        out.append("|---|---|---|---|---|---|")
        pair_payload = {"classes": {}, "seeds": seeds,
                        "selected": {str(s): {
                            "alignment": chosen[s]["alignment"],
                            "layer_agg": chosen[s]["layer_agg"],
                            "classifier": chosen[s]["classifier"],
                            "backbone": chosen[s]["backbone"],
                            "source_val": chosen[s]["selection_source_val_macro_f1"],
                            "target_macro_f1": chosen[s]["macro_f1"],
                        } for s in seeds}}
        for i, name in enumerate(classes):
            cells = []
            entry = {"support": int(observed["support"][i]),
                     "predicted": int(observed["predicted"][i])}
            for key in ("precision", "recall", "f1"):
                lo, hi = np.percentile(draws[key][:, i], [2.5, 97.5])
                value = observed[key][i]
                cells.append(f"{value:.4f} [{lo:.4f}, {hi:.4f}]")
                entry[key] = {"value": float(value), "lo": float(lo), "hi": float(hi)}
            pair_payload["classes"][name] = entry
            out.append(f"| {name} | {int(observed['support'][i])} | "
                       f"{int(observed['predicted'][i])} | " + " | ".join(cells) + " |")
        macro_draws = np.nanmean(draws["f1"], axis=1)
        macro_lo, macro_hi = np.percentile(macro_draws, [2.5, 97.5])
        macro = float(np.nanmean(observed["f1"]))
        out.append(f"| **macro** | {int(observed['support'].sum())} | "
                   f"{int(observed['predicted'].sum())} | | | "
                   f"**{macro:.4f} [{macro_lo:.4f}, {macro_hi:.4f}]** |")
        out.append("")
        pair_payload["macro_f1"] = {"value": macro, "lo": float(macro_lo),
                                    "hi": float(macro_hi)}

        order = sorted(range(len(classes)), key=lambda i: observed["f1"][i])
        out.append(f"Weakest to strongest: "
                   + ", ".join(f"`{classes[i]}` {observed['f1'][i]:.3f}" for i in order)
                   + ".\n")
        collapsed = [classes[i] for i in range(len(classes))
                     if observed["predicted"][i] == 0]
        out.append(("**Never predicted at all: " + ", ".join(f"`{c}`" for c in collapsed)
                    + ".**\n") if collapsed
                   else "No class collapsed to zero predictions.\n")
        payload["pairs"][f"{source}->{target}"] = pair_payload

    # -- which classes behave the same in both directions ------------------
    out.append("## Consistency across directions\n")
    out.append("A class that transfers badly in both directions is a property of "
               "the label, not of one corpus being the source.\n")
    out.append("| class | F1 ravdess->cremad | F1 cremad->ravdess | both below macro |")
    out.append("|---|---|---|---|")
    a = payload["pairs"]["ravdess->cremad"]
    b = payload["pairs"]["cremad->ravdess"]
    for name in classes:
        fa, fb = a["classes"][name]["f1"]["value"], b["classes"][name]["f1"]["value"]
        below = fa < a["macro_f1"]["value"] and fb < b["macro_f1"]["value"]
        out.append(f"| {name} | {fa:.4f} | {fb:.4f} | {'**yes**' if below else 'no'} |")
    out.append("")

    (REPO_ROOT / "reports/per_class.md").write_text("\n".join(out), encoding="utf-8")
    (REPO_ROOT / "reports/per_class.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote reports/per_class.md and reports/per_class.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
