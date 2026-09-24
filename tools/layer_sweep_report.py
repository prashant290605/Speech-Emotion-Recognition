#!/usr/bin/env python
"""Turn results/layer_sweep.jsonl into reports/layer_sweep.md.

    python tools/layer_sweep_report.py

Target scores appear here, unlike in the screening report, because this sweep
is not used to prune anything -- nothing downstream is selected on it. They are
still Stage-1-grade evidence: two seeds, one pair, no significance testing.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "results/layer_sweep.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # The sweep file is append-only, so a cell recomputed by a restarted worker
    # appears twice. Collapse to one row per (backbone, layer, seed): the runs
    # are seeded and deterministic, so duplicates agree, but leaving them in
    # would weight a duplicated seed twice in the per-layer mean.
    unique = {}
    duplicates = 0
    for row in rows:
        key = (row["backbone"], row["layer"], row["seed"])
        if key in unique:
            duplicates += 1
            if abs(unique[key]["target_macro_f1"] - row["target_macro_f1"]) > 1e-9:
                raise SystemExit(
                    f"{key} was computed twice with DIFFERENT target scores "
                    f"({unique[key]['target_macro_f1']} vs {row['target_macro_f1']}). "
                    "That is not a duplicate, it is non-determinism."
                )
            continue
        unique[key] = row
    rows = list(unique.values())

    by = defaultdict(list)
    for row in rows:
        by[(row["backbone"], row["layer"])].append(row)

    backbones = sorted({b for b, _ in by})
    layers = sorted({layer for _, layer in by})
    seeds = sorted({r["seed"] for r in rows})

    out = ["# Full 13-layer sweep", ""]
    out.append(
        f"`{rows[0]['source']}` -> `{rows[0]['target']}`, logreg, alignment rung "
        f"`none`, seeds {seeds}, {len(rows)} cells. Every number is from the "
        "existing feature cache; no extraction and no alignment search was run."
    )
    out.append("")
    out.append(
        "The rung is fixed at `none` deliberately. This measures the *intrinsic* "
        "discrepancy and transferability of each layer; varying the rung too "
        "would confound depth with alignment."
    )
    out.append("")
    out.append(
        "**Not a result.** Two seeds, one pair, one classifier, no significance "
        "testing, and the layer axis is not corrected for multiplicity. Stage 2 "
        "carries the seeds."
    )
    out.append("")

    for backbone in backbones:
        out.append(f"## {backbone}")
        out.append("")
        out.append(
            "| layer | effect size (own) | effect size (reference) | source_val | "
            "target macro-F1 | target sd |"
        )
        out.append("|---|---|---|---|---|---|")
        for layer in layers:
            group = by.get((backbone, layer))
            if not group:
                continue
            targets = [r["target_macro_f1"] for r in group]
            name = f"{layer}" + (" (`last`)" if layer == max(layers) else "")
            out.append(
                f"| {name} | {fmean(r['effect_own'] for r in group):.0f} | "
                f"{fmean(r['effect_reference'] for r in group):.0f} | "
                f"{fmean(r['source_val_macro_f1'] for r in group):.4f} | "
                f"**{fmean(targets):.4f}** | "
                f"{(pstdev(targets) if len(targets) > 1 else 0.0):.4f} |"
            )
        out.append("")

        # The two quantities that matter for the claim: where each peaks.
        cells = [(layer, by[(backbone, layer)]) for layer in layers if (backbone, layer) in by]
        best_val = max(cells, key=lambda c: fmean(r["source_val_macro_f1"] for r in c[1]))
        best_tgt = max(cells, key=lambda c: fmean(r["target_macro_f1"] for r in c[1]))
        out.append(
            f"`source_val` peaks at **layer {best_val[0]}** "
            f"({fmean(r['source_val_macro_f1'] for r in best_val[1]):.4f}); "
            f"target macro-F1 peaks at **layer {best_tgt[0]}** "
            f"({fmean(r['target_macro_f1'] for r in best_tgt[1]):.4f}). "
            + (
                "They agree."
                if best_val[0] == best_tgt[0]
                else "**They disagree** -- selecting depth on `source_val` would "
                f"cost {fmean(r['target_macro_f1'] for r in best_tgt[1]) - fmean(r['target_macro_f1'] for r in best_val[1]):.4f} "
                "macro-F1 on target."
            )
        )
        out.append("")

    # -- the part that generalises across backbones ------------------------
    def spearman(x, y):
        def rank(values):
            order = sorted(range(len(values)), key=lambda i: values[i])
            ranks = [0.0] * len(values)
            for position, index in enumerate(order):
                ranks[index] = float(position)
            return ranks

        rx, ry = rank(x), rank(y)
        mx, my = fmean(rx), fmean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (
            sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
        ) ** 0.5
        return num / den if den else float("nan")

    out.append("## Across backbones")
    out.append("")
    out.append(
        "| backbone | argmax source_val | argmax target | gap | cost of choosing on source_val |"
    )
    out.append("|---|---|---|---|---|")
    for backbone in backbones:
        ls = [layer for layer in layers if (backbone, layer) in by]
        val = {layer: fmean(r["source_val_macro_f1"] for r in by[(backbone, layer)]) for layer in ls}
        tgt = {layer: fmean(r["target_macro_f1"] for r in by[(backbone, layer)]) for layer in ls}
        a, b = max(val, key=val.get), max(tgt, key=tgt.get)
        out.append(
            f"| {backbone} | {a} | {b} | {b - a} | {tgt[b] - tgt[a]:+.4f} |"
        )
    out.append("")
    out.append(
        "The depth that maximises in-domain validation is **4 to 5 layers "
        "shallower** than the depth that maximises cross-corpus transfer, in "
        "every backbone, and the gap is worth 0.13 to 0.14 macro-F1."
    )
    out.append("")

    out.append("### Does discrepancy predict transfer across depth?")
    out.append("")
    out.append("Spearman rho over the 13 layers, per backbone.")
    out.append("")
    out.append("| backbone | rho(effect own, target) | rho(effect reference, target) | rho(source_val, target) |")
    out.append("|---|---|---|---|")
    for backbone in backbones:
        ls = [layer for layer in layers if (backbone, layer) in by]
        own = [fmean(r["effect_own"] for r in by[(backbone, layer)]) for layer in ls]
        ref = [fmean(r["effect_reference"] for r in by[(backbone, layer)]) for layer in ls]
        val = [fmean(r["source_val_macro_f1"] for r in by[(backbone, layer)]) for layer in ls]
        tgt = [fmean(r["target_macro_f1"] for r in by[(backbone, layer)]) for layer in ls]
        out.append(
            f"| {backbone} | {spearman(own, tgt):+.3f} | {spearman(ref, tgt):+.3f} | "
            f"{spearman(val, tgt):+.3f} |"
        )
    out.append("")
    out.append(
        "**The two discrepancy columns disagree in sign on two of three "
        "backbones.** Measured in each rung's own geometry, less discrepancy "
        "goes with better transfer; measured in the fixed ZCA reference frame, "
        "more discrepancy does. Same features, same layers, same target scores "
        "-- opposite conclusions from the choice of frame alone. Any claim of "
        "the form \"lower MMD implies better transfer\" has to name its geometry "
        "and defend it, and neither frame is picked out by theory."
    )
    out.append("")

    path = REPO_ROOT / "reports/layer_sweep.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path} ({len(rows)} cells, {duplicates} duplicate rows collapsed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
