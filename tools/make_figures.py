#!/usr/bin/env python
"""Phase 10: every figure in the paper, regenerated from the result files.

    python tools/make_figures.py            # all
    python tools/make_figures.py ladder     # one

Nothing here is hand-made and nothing is typed in. Each figure reads
`results/` and re-derives its own numbers, so a figure cannot drift from the
tables the way a pasted screenshot can.

Style rules live in `ser.figures`: vector PDF at journal column width,
Okabe-Ito palette, and colour never carrying meaning on its own -- every series
also has a distinct marker and linestyle, every bar group a distinct hatch.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.config import load_config  # noqa: E402
from ser.figures import (  # noqa: E402
    COLUMN_WIDTH,
    DOUBLE_WIDTH,
    HATCHES,
    OKABE_ITO,
    annotate_floor,
    new_figure,
    save,
    series,
    use_style,
)
from ser.manifest import read_manifest  # noqa: E402
from ser.phase8 import (  # noqa: E402
    cluster_bootstrap,
    confusion_by_group,
    load_predictions,
    seed_interval,
)
from ser.utils.results import read_rows  # noqa: E402

RESULTS = REPO_ROOT / "results/runs.jsonl"
STAGE2_TAG = "grid-freeze-v3"
LADDER = ["none", "zscore", "mean_shift", "coral", "mkmmd_diag", "mkmmd_full"]
PAIRS = [("ravdess", "cremad"), ("cremad", "ravdess")]
LABEL = {"ravdess": "RAVDESS", "cremad": "CREMA-D"}
N_BOOT = 500  # figures; the tables use 2000
FIGURE_DIR = REPO_ROOT / "figures"


def emit(fig, name):
    """Write into <repo>/figures regardless of the working directory."""
    return save(fig, name, FIGURE_DIR)


def pair_title(source, target):
    return f"{LABEL[source]} $\\rightarrow$ {LABEL[target]}"


class Data:
    """Result rows, predictions and per-speaker confusions, loaded once."""

    def __init__(self):
        self.config = load_config()
        manifest = read_manifest(self.config.resolve(self.config.paths.manifest))
        self.label = {r.utterance_id: r.label_six for r in manifest}
        self.speaker = {r.utterance_id: r.speaker_id for r in manifest}
        rows = [r for r in read_rows(RESULTS) if r["status"] == "ok"]
        self.stage2 = [r for r in rows if r["freeze_tag"] == STAGE2_TAG]
        self.main = [r for r in self.stage2 if r["blending"] == "none"]
        self.classes = list(self.main[0]["class_names"])
        self.index = {n: i for i, n in enumerate(self.classes)}
        self._speakers, self._conf = {}, {}

    def speakers(self, row):
        key = (row["source_corpus"], row["target_corpus"], row["seed"])
        if key not in self._speakers:
            ids, _ = load_predictions(RESULTS, row)
            names = sorted({self.speaker[u] for u in ids})
            self._speakers[key] = {n: i for i, n in enumerate(names)}
        return self._speakers[key]

    def confusion(self, row):
        if row["run_id"] not in self._conf:
            ids, predicted = load_predictions(RESULTS, row)
            lookup = self.speakers(row)
            self._conf[row["run_id"]] = confusion_by_group(
                [self.index[self.label[u]] for u in ids],
                [self.index[p] for p in predicted],
                [lookup[self.speaker[u]] for u in ids],
                len(self.classes), len(lookup),
            )
        return self._conf[row["run_id"]]

    def n_speakers(self, source, target):
        return {seed: len(v) for (s, t, seed), v in self._speakers.items()
                if (s, t) == (source, target)}

    def arm(self, rows, keys=("seed", "backbone", "layer_agg", "classifier")):
        """seed -> [confusion per condition], picking the source_val best in each."""
        buckets = defaultdict(list)
        for r in rows:
            buckets[tuple(r[k] for k in keys)].append(r)
        out = defaultdict(list)
        for group in buckets.values():
            best = max(group, key=lambda r: r["selection_source_val_macro_f1"])
            out[best["seed"]].append(self.confusion(best))
        return dict(out)


def sweep_rows():
    rows = {}
    for path in sorted(glob.glob(str(REPO_ROOT / "results/shards/sweep2_*.jsonl"))):
        for r in read_rows(path):
            if r["status"] == "ok":
                rows[r["run_id"]] = r
    return list(rows.values())


def eps_probe_rows():
    rows = {}
    for pattern in ("results/eps_*.jsonl", "results/shards/eps_*.jsonl"):
        for path in sorted(glob.glob(str(REPO_ROOT / pattern))):
            for r in read_rows(path):
                if r["status"] == "ok":
                    rows[r["run_id"]] = r
    return list(rows.values())


# ---------------------------------------------------------------------------
def figure_ladder(data):
    """Step, then plateau: target macro-F1 by rung with discrepancy alongside."""
    fig, axes = new_figure(DOUBLE_WIDTH, 2.9, ncols=2, sharey=True)
    for column, (source, target) in enumerate(PAIRS):
        axis = axes[column]
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        means, los, his, effects = [], [], [], []
        for rung in LADDER:
            g = [r for r in pool if r["alignment"] == rung]
            stat = cluster_bootstrap(data.arm(g), data.n_speakers(source, target),
                                     n_boot=N_BOOT, seed=17)
            means.append(stat["mean"]); los.append(stat["lo"]); his.append(stat["hi"])
            effects.append(np.mean([r["marginal_mmd_normalised"] for r in g]))

        x = np.arange(len(LADDER))
        style = series(0)
        axis.errorbar(x, means,
                      yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                      color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                      label="target macro-F1", zorder=3)
        annotate_floor(axis, float(np.mean([r["chance_macro_f1"] for r in pool])))

        twin = axis.twinx()
        twin.set_yscale("log")
        dstyle = series(5)
        twin.plot(x, effects, color=dstyle["color"], marker=dstyle["marker"],
                  linestyle=dstyle["linestyle"], alpha=0.85,
                  label="marginal discrepancy", zorder=2)
        twin.set_ylabel("discrepancy ($\\times$ null, log)", color=dstyle["color"])
        twin.tick_params(axis="y", labelcolor=dstyle["color"], labelsize=6)
        twin.grid(False)
        twin.spines["top"].set_visible(False)

        # Shade the plateau so the "one step, then flat" reading is visible
        # without reading the numbers off the axis.
        axis.axvspan(0.5, len(LADDER) - 0.5, color="0.85", alpha=0.35, zorder=0)
        axis.annotate("plateau", xy=(3.5, 0.955), xycoords=("data", "axes fraction"),
                      ha="center", fontsize=6, color="0.35")

        axis.set_xticks(x)
        axis.set_xticklabels(LADDER, rotation=30, ha="right")
        axis.set_title(pair_title(source, target))
        axis.set_xlabel("alignment rung (moments matched $\\rightarrow$)")
        if column == 0:
            axis.set_ylabel("target macro-F1")
            handles = [
                axis.lines[0],
                twin.lines[0],
            ]
            axis.legend(handles, ["target macro-F1", "marginal discrepancy"],
                        loc="upper left", fontsize=6)
    fig.suptitle("Alignment buys one step, then nothing", fontsize=9, y=1.02)
    return emit(fig, "ladder")


def figure_decomposition():
    """Marginal falls; conditional does not follow it down."""
    records = [json.loads(line) for line in
               (REPO_ROOT / "results/phase9_shift.jsonl").read_text().splitlines()
               if line.strip()]
    fig, axes = new_figure(DOUBLE_WIDTH, 2.9, ncols=2,
                           gridspec_kw={"wspace": 0.42})
    agg = "last"
    for column, (source, target) in enumerate(PAIRS):
        axis = axes[column]
        marg, cond, ratio = [], [], []
        for rung in LADDER:
            g = [r for r in records
                 if (r["source"], r["target"]) == (source, target)
                 and r["layer_agg"] == agg and r["alignment"] == rung]
            m = float(np.mean([r["marginal_effect_own"] for r in g]))
            c = float(np.mean([np.mean([x["effect_size"] for x in r["conditional"]
                                        if x["effect_size"] is not None]) for r in g]))
            marg.append(m); cond.append(c); ratio.append(c / m)

        x = np.arange(len(LADDER))
        for i, (values, name) in enumerate(((marg, "marginal"), (cond, "conditional"))):
            style = series(i + 1)
            axis.plot(x, values, label=name, **style)
        axis.set_yscale("log")
        axis.set_xticks(x)
        axis.set_xticklabels(LADDER, rotation=30, ha="right")
        axis.set_title(pair_title(source, target))
        axis.set_ylabel("discrepancy ($\\times$ null, log)" if column == 0 else "")

        twin = axis.twinx()
        rstyle = series(3)
        twin.plot(x, ratio, color=rstyle["color"], marker=rstyle["marker"],
                  linestyle=rstyle["linestyle"], alpha=0.9)
        twin.set_ylim(0, 1.15)
        twin.set_ylabel("conditional / marginal" if column == len(PAIRS) - 1 else "",
                        color=rstyle["color"])
        twin.tick_params(axis="y", labelcolor=rstyle["color"], labelsize=6)
        twin.grid(False)
        twin.spines["top"].set_visible(False)
        for position, value, align in ((0, ratio[0], "left"),
                                       (len(LADDER) - 1, ratio[-1], "right")):
            twin.annotate(f"{value:.2f}", xy=(position, value),
                          xytext=(0, 7), textcoords="offset points",
                          fontsize=6.5, fontweight="bold", color=rstyle["color"],
                          ha=align, va="bottom")
        if column == 0:
            axis.legend(loc="lower left", fontsize=6)
    fig.suptitle("Alignment removes the marginal term, not the conditional one",
                 fontsize=9, y=1.02)
    return emit(fig, "decomposition")


def figure_validated_vs_oracle(data):
    """What the protocol picks, per seed, against what the grid contained."""
    fig, axes = new_figure(DOUBLE_WIDTH, 2.9, ncols=2, sharey=True)
    for column, (source, target) in enumerate(PAIRS):
        axis = axes[column]
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        by_seed = defaultdict(list)
        for r in pool:
            by_seed[r["seed"]].append(r)
        seeds = sorted(by_seed)

        validated, oracle, picked = [], [], []
        for seed in seeds:
            best = max(by_seed[seed], key=lambda r: r["selection_source_val_macro_f1"])
            validated.append(best["macro_f1"])
            oracle.append(max(r["macro_f1"] for r in by_seed[seed]))
            picked.append(best["alignment"])

        x = np.arange(len(seeds))
        width = 0.38
        for i, (values, name) in enumerate(((validated, "validated"), (oracle, "oracle"))):
            axis.bar(x + (i - 0.5) * width, values, width,
                     color=OKABE_ITO[[2, 4][i]], edgecolor="black", linewidth=0.6,
                     hatch=HATCHES[i * 2], label=name, zorder=3)
        # Mark the seeds where selection fell back to `none`: the point of the
        # figure is not the gap size but that the criterion sometimes cannot
        # tell the unaligned condition from the aligned ones.
        for i, rung in enumerate(picked):
            axis.annotate(rung, xy=(x[i] - 0.5 * width, validated[i]),
                          xytext=(0, 3), textcoords="offset points",
                          rotation=90, ha="center", va="bottom", fontsize=5.5,
                          color="#D55E00" if rung == "none" else "0.3",
                          fontweight="bold" if rung == "none" else "normal")
        annotate_floor(axis, float(np.mean([r["chance_macro_f1"] for r in pool])))
        axis.set_xticks(x)
        axis.set_xticklabels([f"seed {s}" for s in seeds])
        axis.set_title(pair_title(source, target))
        axis.set_ylim(0, max(oracle) * 1.55)
        if column == 0:
            axis.set_ylabel("target macro-F1")
            axis.legend(loc="upper left", fontsize=6, ncol=2)
        fallbacks = sum(1 for p in picked if p == "none")
        axis.annotate(f"selected `none` on {fallbacks} of {len(seeds)} seeds",
                      xy=(0.98, 0.86), xycoords="axes fraction", ha="right",
                      fontsize=6.5, color="#D55E00" if fallbacks else "0.35",
                      fontweight="bold" if fallbacks else "normal")
    fig.suptitle("Selection on source validation cannot see the alignment step",
                 fontsize=9, y=1.02)
    return emit(fig, "validated_vs_oracle")


def figure_eps_asymptote(data):
    """CORAL is selected into being mean_shift."""
    probe = eps_probe_rows()
    families = {r["classifier"] for r in probe}
    aggs = {r["layer_agg"] for r in probe}
    backbones = {r["backbone"] for r in probe}
    matched = [r for r in data.main if r["classifier"] in families
               and r["layer_agg"] in aggs and r["backbone"] in backbones]

    fig, axes = new_figure(DOUBLE_WIDTH, 2.9, ncols=2, sharex=True)
    for column, (source, target) in enumerate(PAIRS):
        axis = axes[column]
        pool = [r for r in matched
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        pp = [r for r in probe
              if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        eps_values, val, tgt = [], [], []
        for eps in sorted({r["alignment_eps"] for r in pool
                           if r["alignment"] == "coral" and r["alignment_eps"] is not None}):
            g = [r for r in pool if r["alignment"] == "coral" and r["alignment_eps"] == eps]
            eps_values.append(eps)
            val.append(np.mean([r["selection_source_val_macro_f1"] for r in g]))
            tgt.append(np.mean([r["macro_f1"] for r in g]))
        for eps in sorted({r["alignment_eps"] for r in pp}):
            g = [r for r in pp if r["alignment_eps"] == eps]
            eps_values.append(eps)
            val.append(np.mean([r["selection_source_val_macro_f1"] for r in g]))
            tgt.append(np.mean([r["macro_f1"] for r in g]))

        for i, (values, name) in enumerate(((val, "source_val"), (tgt, "target macro-F1"))):
            axis.plot(eps_values, values, label=name, **series(i + 1))
        shift = [r for r in pool if r["alignment"] == "mean_shift"]
        for i, (key, name) in enumerate((("selection_source_val_macro_f1", "mean_shift source_val"),
                                         ("macro_f1", "mean_shift target"))):
            axis.axhline(float(np.mean([r[key] for r in shift])),
                         color=OKABE_ITO[i + 1], linestyle=(0, (1, 1.5)),
                         linewidth=1.0, alpha=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("CORAL shrinkage $\\epsilon$")
        axis.set_title(pair_title(source, target))
        if column == 0:
            axis.set_ylabel("macro-F1")
            axis.legend(loc="center left", fontsize=6)
    fig.suptitle("As $\\epsilon$ grows CORAL degenerates to a scalar rescale "
                 "plus a mean shift", fontsize=9, y=1.02)
    return emit(fig, "eps_asymptote")


def figure_frame_dependence(data):
    """Two geometries, opposite answers -- and the case that adjudicates."""
    rows = sweep_rows()
    seeds = sorted({r["seed"] for r in rows})

    def spearman(x, y):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            out = [0.0] * len(v)
            for pos, i in enumerate(order):
                out[i] = float(pos)
            return out
        rx, ry = rank(x), rank(y)
        if len(set(rx)) < 2 or len(set(ry)) < 2:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])

    own_all, ref_all, tags = [], [], []
    for source, target in PAIRS:
        for backbone in sorted({r["backbone"] for r in rows}):
            for rung in LADDER:
                o, r_ = [], []
                for seed in seeds:
                    cells = [x for x in rows
                             if (x["source_corpus"], x["target_corpus"]) == (source, target)
                             and x["backbone"] == backbone and x["alignment"] == rung
                             and x["seed"] == seed]
                    if len(cells) < 3:
                        continue
                    cells.sort(key=lambda x: x["layer_index"])
                    sc = [x["macro_f1"] for x in cells]
                    o.append(spearman([x["marginal_mmd_normalised"] for x in cells], sc))
                    r_.append(spearman([x["marginal_mmd_reference"] for x in cells], sc))
                if o:
                    own_all.append(seed_interval(o))
                    ref_all.append(seed_interval(r_))
                    tags.append(f"{source[:4]}/{backbone[:4]}/{rung}")

    fig, axes = new_figure(DOUBLE_WIDTH, 3.0, ncols=2,
                           gridspec_kw={"width_ratios": [1.25, 1]})

    # -- panel A: every cell, both frames ---------------------------------
    axis = axes[0]
    order = np.argsort([s["mean"] for s in own_all])
    y = np.arange(len(order))
    for i, (stats, name) in enumerate(((own_all, "own geometry"),
                                       (ref_all, "reference frame"))):
        style = series(i + 1)
        means = [stats[j]["mean"] for j in order]
        errs = [[means[k] - stats[j]["lo"] for k, j in enumerate(order)],
                [stats[j]["hi"] - means[k] for k, j in enumerate(order)]]
        axis.errorbar(means, y + (i - 0.5) * 0.32, xerr=errs, fmt=style["marker"],
                      color=style["color"], markersize=3, linewidth=0.8,
                      label=name, linestyle="none")
    axis.axvline(0, color="0.3", linewidth=0.9)
    axis.set_yticks([])
    axis.set_xlabel("Spearman $\\rho$ (discrepancy vs target macro-F1)")
    axis.set_ylabel("36 cells: direction $\\times$ backbone $\\times$ rung")
    axis.set_title("The two frames disagree about the sign")
    axis.legend(loc="upper left", fontsize=6)
    pooled_own = np.mean([s["mean"] for s in own_all])
    pooled_ref = np.mean([s["mean"] for s in ref_all])
    axis.annotate(f"pooled  own {pooled_own:+.2f}   ref {pooled_ref:+.2f}",
                  xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=6)

    # -- panel B: the case where the answer is known ----------------------
    axis = axes[1]
    probe = eps_probe_rows()
    families = {r["classifier"] for r in probe}
    aggs = {r["layer_agg"] for r in probe}
    backbones = {r["backbone"] for r in probe}
    matched = [r for r in data.main if r["classifier"] in families
               and r["layer_agg"] in aggs and r["backbone"] in backbones]
    source, target = PAIRS[0]
    pool = [r for r in matched
            if (r["source_corpus"], r["target_corpus"]) == (source, target)]
    pp = [r for r in probe
          if (r["source_corpus"], r["target_corpus"]) == (source, target)]
    eps_values, own, ref = [], [], []
    for eps in sorted({r["alignment_eps"] for r in pool
                       if r["alignment"] == "coral" and r["alignment_eps"] is not None}):
        g = [r for r in pool if r["alignment"] == "coral" and r["alignment_eps"] == eps]
        eps_values.append(eps)
        own.append(np.mean([r["marginal_mmd_normalised"] for r in g]))
        ref.append(np.mean([r["marginal_mmd_reference"] for r in g]))
    for eps in sorted({r["alignment_eps"] for r in pp}):
        g = [r for r in pp if r["alignment_eps"] == eps]
        eps_values.append(eps)
        own.append(np.mean([r["marginal_mmd_normalised"] for r in g]))
        ref.append(np.mean([r["marginal_mmd_reference"] for r in g]))
    shift = [r for r in pool if r["alignment"] == "mean_shift"]
    for i, (values, key, name) in enumerate((
            (own, "marginal_mmd_normalised", "own geometry"),
            (ref, "marginal_mmd_reference", "reference frame"))):
        style = series(i + 1)
        axis.plot(eps_values, values, label=name, **style)
        axis.axhline(float(np.mean([r[key] for r in shift])), color=style["color"],
                     linestyle=(0, (1, 1.5)), linewidth=1.0, alpha=0.8)
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("CORAL shrinkage $\\epsilon$")
    axis.set_ylabel("discrepancy ($\\times$ null, log)")
    axis.set_title("Where the answer is known")
    axis.annotate("dotted = `mean_shift`, the analytic limit",
                  xy=(0.98, 0.05), xycoords="axes fraction", ha="right",
                  fontsize=6, color="0.35")
    axis.legend(loc="upper left", fontsize=6)
    fig.suptitle("The discrepancy-transfer relationship has no sign until the "
                 "geometry is fixed", fontsize=9, y=1.03)
    return emit(fig, "frame_dependence")


def figure_confusion(data):
    """Confusion matrices done properly: both axes labelled, shared scale."""
    import matplotlib.pyplot as plt

    fig, axes = new_figure(DOUBLE_WIDTH, 3.7, ncols=2)
    matrices = []
    for source, target in PAIRS:
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        by_seed = defaultdict(list)
        for r in pool:
            by_seed[r["seed"]].append(r)
        total = np.zeros((len(data.classes), len(data.classes)))
        for seed in sorted(by_seed):
            best = max(by_seed[seed], key=lambda r: r["selection_source_val_macro_f1"])
            total += data.confusion(best).sum(axis=0)
        # Row-normalised: the question is "given the true class, what was
        # predicted", and raw counts would just show the class priors.
        matrices.append(total / np.maximum(total.sum(axis=1, keepdims=True), 1e-12))

    for column, ((source, target), matrix) in enumerate(zip(PAIRS, matrices)):
        axis = axes[column]
        image = axis.imshow(matrix, cmap="cividis", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(data.classes)))
        axis.set_yticks(range(len(data.classes)))
        axis.set_xticklabels(data.classes, rotation=45, ha="right")
        axis.set_yticklabels(data.classes)
        axis.set_xlabel("predicted class")
        if column == 0:
            axis.set_ylabel("true class")
        axis.set_title(pair_title(source, target))
        axis.grid(False)
        # Value in every cell, in whichever of black/white has contrast, so the
        # figure is readable without the colourbar at all.
        for i in range(len(data.classes)):
            for j in range(len(data.classes)):
                value = matrix[i, j]
                axis.text(j, i, f"{value:.2f}", ha="center", va="center",
                          fontsize=6, color="white" if value < 0.62 else "black")
    bar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    bar.set_label("proportion of true class (rows sum to 1)")
    fig.suptitle("Confusion of the validated configuration, summed over 5 seeds",
                 fontsize=9, y=1.0)
    return emit(fig, "confusion")


def figure_per_class(data):
    """Which emotions transfer, with intervals, both directions."""
    payload = json.loads((REPO_ROOT / "reports/per_class.json").read_text())
    classes = payload["classes"]
    fig, axis = new_figure(DOUBLE_WIDTH, 2.7)
    x = np.arange(len(classes))
    width = 0.38
    for i, (source, target) in enumerate(PAIRS):
        entry = payload["pairs"][f"{source}->{target}"]
        values = [entry["classes"][c]["f1"]["value"] for c in classes]
        lo = [values[k] - entry["classes"][c]["f1"]["lo"] for k, c in enumerate(classes)]
        hi = [entry["classes"][c]["f1"]["hi"] - values[k] for k, c in enumerate(classes)]
        axis.bar(x + (i - 0.5) * width, values, width, yerr=[lo, hi],
                 color=OKABE_ITO[[2, 4][i]], edgecolor="black", linewidth=0.6,
                 hatch=HATCHES[i * 2], label=pair_title(source, target),
                 error_kw={"linewidth": 0.8}, zorder=3)
        axis.axhline(entry["macro_f1"]["value"], color=OKABE_ITO[[2, 4][i]],
                     linestyle=(0, (4, 1.5)), linewidth=1.0, alpha=0.9)
    chance = float(np.mean([r["chance_macro_f1"] for r in data.main]))
    annotate_floor(axis, chance, x=0.30)
    axis.set_xticks(x)
    axis.set_xticklabels(classes)
    axis.set_ylabel("per-class F1")
    axis.set_xlabel("emotion")
    axis.legend(loc="upper right", fontsize=6)
    axis.annotate("dashed = that direction's macro-F1", xy=(0.02, 0.93),
                  xycoords="axes fraction", fontsize=6, color="0.35")
    below = [c for c in classes
             if all(payload["pairs"][f"{s2}->{t2}"]["classes"][c]["f1"]["value"]
                    < payload["pairs"][f"{s2}->{t2}"]["macro_f1"]["value"]
                    for s2, t2 in PAIRS)]
    fig.suptitle(", ".join(below)
                 + " fall below macro-F1 in both directions", fontsize=9, y=1.02)
    return emit(fig, "per_class_f1")


FIGURES = {
    "ladder": lambda d: figure_ladder(d),
    "decomposition": lambda d: figure_decomposition(),
    "validated_vs_oracle": lambda d: figure_validated_vs_oracle(d),
    "eps_asymptote": lambda d: figure_eps_asymptote(d),
    "frame_dependence": lambda d: figure_frame_dependence(d),
    "confusion": lambda d: figure_confusion(d),
    "per_class_f1": lambda d: figure_per_class(d),
}


def main(argv=None) -> int:
    wanted = list(argv or sys.argv[1:]) or list(FIGURES)
    unknown = [n for n in wanted if n not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {unknown}. Known: {list(FIGURES)}")
        return 2
    use_style()
    data = Data()
    for name in wanted:
        pdf, _ = FIGURES[name](data)
        print(f"  {name:<22} -> {pdf.relative_to(REPO_ROOT).as_posix()}")
    print(f"\n{len(wanted)} figure(s) written to figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
