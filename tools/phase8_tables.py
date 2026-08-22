#!/usr/bin/env python
"""Phase 8 pass 1: the primary tables. Numbers only, no interpretation.

    python tools/phase8_tables.py

Writes reports/phase8_tables.md.

The primary comparison family is declared in PRIMARY below, before any result
is computed, and Holm correction is applied within exactly that family.
Everything else in this file reports an interval and says nothing about
significance -- which is the point of fixing the family in advance.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.config import load_config  # noqa: E402
from ser.manifest import read_manifest  # noqa: E402
from ser.phase8 import (  # noqa: E402
    cluster_bootstrap,
    confusion_by_group,
    holm,
    load_predictions,
    macro_f1_from_confusion,
    paired_cluster_bootstrap,
    per_class_f1_from_confusion,
    seed_interval,
)
from ser.utils.results import read_rows  # noqa: E402

RESULTS = REPO_ROOT / "results/runs.jsonl"
STAGE2_TAG = "grid-freeze-v3"
N_BOOT = 2000

# ---------------------------------------------------------------------------
# THE PRIMARY COMPARISON FAMILY -- fixed before any number below was computed.
#
# Two questions carry the paper, so the family is the smallest set that can
# answer them. Each entry is run on both transfer directions, giving 14 tests,
# and Holm is applied across all 14 at once: both directions are reported as
# primary results, so both belong inside the correction.
#
# Everything not listed here gets an interval and no significance claim.
# ---------------------------------------------------------------------------
PRIMARY_LADDER = [
    ("L1", "none", "zscore"),
    ("L2", "none", "mean_shift"),
    ("L3", "none", "coral"),
    ("L4", "none", "mkmmd_diag"),
    ("L5", "none", "mkmmd_full"),
]
PRIMARY_AGG = [
    # `weighted` exists only for the torch families, so A2 is restricted to
    # them and A1 is computed on the same restriction for comparability.
    ("A1", "last", "layer"),
    ("A2", "last", "weighted"),
]

PAIRS = [("ravdess", "cremad"), ("cremad", "ravdess")]


def fmt(mean, lo, hi, places=4):
    if mean != mean:
        return "--"
    if lo != lo:
        return f"{mean:.{places}f} (n=1)"
    return f"{mean:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"


class Data:
    """Rows, labels, speakers and the per-speaker confusion cache."""

    def __init__(self):
        self.config = load_config()
        manifest = read_manifest(self.config.resolve(self.config.paths.manifest))
        self.label = {r.utterance_id: r.label_six for r in manifest}
        self.speaker = {r.utterance_id: r.speaker_id for r in manifest}
        rows = list(read_rows(RESULTS))
        self.baselines = [r for r in rows if r["classifier"].startswith("baseline")]
        self.stage2 = [r for r in rows if r["freeze_tag"] == STAGE2_TAG]
        self.main = [r for r in self.stage2 if r["blending"] == "none"]
        self.blend = [r for r in self.stage2 if r["blending"] != "none"]
        self.classes = list(self.stage2[0]["class_names"])
        self.index = {name: i for i, name in enumerate(self.classes)}
        self._speaker_index = {}
        self._conf = {}

    def pair_key(self, row):
        return (row["source_corpus"], row["target_corpus"], row["seed"])

    def speakers(self, row):
        """Speaker index for this (pair, seed)'s target_test, built once."""
        key = self.pair_key(row)
        if key not in self._speaker_index:
            ids, _ = load_predictions(RESULTS, row)
            names = sorted({self.speaker[u] for u in ids})
            self._speaker_index[key] = {n: i for i, n in enumerate(names)}
        return self._speaker_index[key]

    def confusion(self, row):
        """``(n_speakers, K, K)`` for one run, cached."""
        run_id = row["run_id"]
        if run_id not in self._conf:
            ids, predicted = load_predictions(RESULTS, row)
            speakers = self.speakers(row)
            y_true = [self.index[self.label[u]] for u in ids]
            y_pred = [self.index[p] for p in predicted]
            groups = [speakers[self.speaker[u]] for u in ids]
            self._conf[run_id] = confusion_by_group(
                y_true, y_pred, groups, len(self.classes), len(speakers)
            )
        return self._conf[run_id]

    def n_speakers(self, source, target):
        out = {}
        for (s, t, seed), index in self._speaker_index.items():
            if (s, t) == (source, target):
                out[seed] = len(index)
        return out


def best_by_source_val(rows):
    return max(rows, key=lambda r: r["selection_source_val_macro_f1"])


# ---------------------------------------------------------------------------
def table_integrity(data, out):
    rows = list(read_rows(RESULTS))
    out.append("## 0. Integrity\n")
    out.append("| | |")
    out.append("|---|---|")
    out.append(f"| total rows | {len(rows)} |")
    out.append(f"| unique run_ids | {len({r['run_id'] for r in rows})} |")
    out.append(f"| status=ok | {sum(1 for r in rows if r['status'] == 'ok')} |")
    out.append(f"| status=failed | {sum(1 for r in rows if r['status'] != 'ok')} |")
    out.append(f"| Phase 4 baselines (no freeze tag) | {len(data.baselines)} |")
    out.append(f"| Stage 0 (grid-freeze-v1) | {sum(1 for r in rows if r['freeze_tag'] == 'grid-freeze-v1')} |")
    out.append(f"| Stage 1 (grid-freeze-v2) | {sum(1 for r in rows if r['freeze_tag'] == 'grid-freeze-v2')} |")
    out.append(f"| Stage 2 (grid-freeze-v3) | {len(data.stage2)} |")
    out.append(f"| schema versions present | {sorted({r['schema_version'] for r in rows})} |")
    out.append("")

    solver = defaultdict(list)
    for r in data.stage2:
        if r["solver_n_iter"] is not None:
            solver[r["classifier"]].append(r["solver_n_iter"])
    out.append("### Convergence\n")
    out.append("| family | runs with a solver count | min | median | max | cap | at cap |")
    out.append("|---|---|---|---|---|---|---|")
    for family in sorted(solver):
        v = solver[family]
        out.append(f"| {family} | {len(v)} | {min(v)} | {median(v):.0f} | {max(v)} | "
                   f"{data.config.classifiers.sklearn_max_iter} | "
                   f"{sum(1 for x in v if x >= data.config.classifiers.sklearn_max_iter)} |")
    not_converged = sum(
        (json.loads(r["hyperparams_json"] or "{}").get("n_not_converged") or 0)
        for r in data.stage2
    )
    failed_trials = sum(
        (json.loads(r["hyperparams_json"] or "{}").get("n_failed_trials") or 0)
        for r in data.stage2
    )
    out.append("")
    out.append(f"NotConverged trials: **{not_converged}**. Failed trials of any kind: "
               f"**{failed_trials}** of {len(data.stage2) * data.config.classifiers.search_budget} attempted.\n")

    out.append("### MK-MMD fallback\n")
    out.append("| rung | runs | reverted to warm start | rate |")
    out.append("|---|---|---|---|")
    for rung in ("mkmmd_diag", "mkmmd_full"):
        g = [r for r in data.stage2 if r["alignment"] == rung]
        fired = sum(1 for r in g if r["mmd_fallback_fired"])
        out.append(f"| {rung} | {len(g)} | {fired} | {fired / len(g):.1%} |")
    out.append("")
    out.append("| rung | lambda | reverted | rate |")
    out.append("|---|---|---|---|")
    for rung in ("mkmmd_diag", "mkmmd_full"):
        g = [r for r in data.stage2 if r["alignment"] == rung]
        for lam in sorted({r["alignment_lambda"] for r in g}):
            h = [r for r in g if r["alignment_lambda"] == lam]
            out.append(f"| {rung} | {lam:g} | {sum(1 for r in h if r['mmd_fallback_fired'])}/{len(h)} | "
                       f"{sum(1 for r in h if r['mmd_fallback_fired']) / len(h):.1%} |")
    out.append("")

    total = sum(r["wall_seconds"] for r in data.stage2)
    out.append("### Cost against projection\n")
    out.append("| | projected | actual |")
    out.append("|---|---|---|")
    out.append(f"| CPU hours | 269.3 | {total / 3600:.1f} |")
    out.append(f"| wall hours at 4 shards | 67.3 | {total / 3600 / 4:.1f} |")
    out.append("")
    out.append("| family | runs | CPU hours | share |")
    out.append("|---|---|---|---|")
    for family in sorted({r["classifier"] for r in data.stage2}):
        g = [r["wall_seconds"] for r in data.stage2 if r["classifier"] == family]
        out.append(f"| {family} | {len(g)} | {sum(g) / 3600:.1f} | {sum(g) / total:.1%} |")
    out.append("")


def table_validated_oracle(data, out):
    out.append("## 1. Validated vs oracle\n")
    out.append("Validated = the configuration with the best `source_val` in that "
               "(pair, seed), scored on target. Oracle = the best target score in "
               "the same grid slice. **The oracle column is an upper bound that "
               "no protocol can reach; it is not a result.**\n")
    for label, subset in [
        ("full grid", data.main),
        ("sklearn + MLP only (all 5 seeds)",
         [r for r in data.main if r["classifier"] != "transformer"]),
    ]:
        out.append(f"### {label}\n")
        out.append("| pair | n seeds | validated | oracle | gap | chance | majority |")
        out.append("|---|---|---|---|---|---|---|")
        for source, target in PAIRS:
            g = [r for r in subset if (r["source_corpus"], r["target_corpus"]) == (source, target)]
            by_seed = defaultdict(list)
            for r in g:
                by_seed[r["seed"]].append(r)
            validated, oracle, gaps = [], [], []
            for seed in sorted(by_seed):
                pool = by_seed[seed]
                v = best_by_source_val(pool)["macro_f1"]
                o = max(r["macro_f1"] for r in pool)
                validated.append(v)
                oracle.append(o)
                gaps.append(o - v)
            vi, oi, gi = seed_interval(validated), seed_interval(oracle), seed_interval(gaps)
            chance = seed_interval([r["chance_macro_f1"] for r in g])
            majority = seed_interval([r["majority_macro_f1"] for r in g])
            out.append(
                f"| {source}->{target} | {len(validated)} | {fmt(**{k: vi[k] for k in ('mean','lo','hi')})} | "
                f"{fmt(**{k: oi[k] for k in ('mean','lo','hi')})} | "
                f"{fmt(**{k: gi[k] for k in ('mean','lo','hi')})} | "
                f"{chance['mean']:.4f} | {majority['mean']:.4f} |"
            )
        out.append("")


def arms_for(data, rows, split_key, value_a, value_b, *, families=None):
    """Matched arms for a paired comparison, per pair.

    Within each condition -- everything except ``split_key`` -- the
    `source_val` best run is taken on each side, so the comparison is between
    the two options *as the validated protocol would use them*, not between an
    arbitrary variant of each.
    """
    out = {}
    for source, target in PAIRS:
        pool = [r for r in rows if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        if families:
            pool = [r for r in pool if r["classifier"] in families]
        buckets = defaultdict(list)
        for r in pool:
            keys = ("seed", "backbone", "layer_agg", "classifier", "alignment")
            cond = tuple(r[k] for k in keys if k != split_key)
            buckets[cond].append(r)

        arm_a, arm_b = defaultdict(list), defaultdict(list)
        for cond, group in buckets.items():
            side_a = [r for r in group if r[split_key] == value_a]
            side_b = [r for r in group if r[split_key] == value_b]
            if not side_a or not side_b:
                continue
            seed = group[0]["seed"]
            arm_a[seed].append(data.confusion(best_by_source_val(side_a)))
            arm_b[seed].append(data.confusion(best_by_source_val(side_b)))
        out[(source, target)] = (dict(arm_a), dict(arm_b))
    return out


def run_primary(data, out):
    out.append("## Primary comparisons\n")
    out.append("Declared in `tools/phase8_tables.py` before any result was computed. "
               "Paired cluster bootstrap over target_test speakers and seeds, "
               f"{N_BOOT} replicates, Holm-corrected across all "
               f"{(len(PRIMARY_LADDER) + len(PRIMARY_AGG)) * len(PAIRS)} tests.\n")

    tests = []
    for spec, key, families in (
        (PRIMARY_LADDER, "alignment", None),
        (PRIMARY_AGG, "layer_agg", ("mlp", "transformer")),
    ):
        for tag, a, b in spec:
            arms = arms_for(data, data.main, key, a, b, families=families)
            for pair, (arm_a, arm_b) in arms.items():
                stat = paired_cluster_bootstrap(
                    arm_b, arm_a, data.n_speakers(*pair), n_boot=N_BOOT, seed=17
                )
                stat["n_conditions"] = sum(len(v) for v in arm_a.values())
                tests.append(
                    {"tag": tag, "name": f"{b} - {a}", "pair": pair, **stat, "holm": None}
                )

    valid = [t for t in tests if t["n_seeds"] > 0]
    for test, adjusted in zip(valid, holm([t["p"] for t in valid])):
        test["holm"] = adjusted

    out.append("| id | comparison | pair | seeds | conditions | difference in target macro-F1 [95% CI] | p | Holm p |")
    out.append("|---|---|---|---|---|---|---|---|")
    for t in tests:
        if t["n_seeds"] == 0:
            out.append(f"| {t['tag']} | {t['name']} | {t['pair'][0]}->{t['pair'][1]} | 0 | 0 | "
                       "not estimable | -- | -- |")
            continue
        out.append(
            f"| {t['tag']} | {t['name']} | {t['pair'][0]}->{t['pair'][1]} | {t['n_seeds']} | "
            f"{t['n_conditions']} | "
            f"{t['diff']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}] | "
            f"{'<' if t['p_at_floor'] else ''}{t['p']:.4f} | "
            f"{'<' if t['p_at_floor'] else ''}{t['holm']:.4f} |"
        )
    out.append("")
    return tests


# ---------------------------------------------------------------------------
def table_ladder(data, out):
    out.append("## 2. Alignment ladder\n")
    out.append("Target macro-F1 by rung, with **both** discrepancy columns. The two "
               "frames are reported together throughout; neither is presented alone.\n")
    ladder = data.config.alignment.ladder_order()
    for source, target in PAIRS:
        out.append(f"### {source} -> {target}\n")
        out.append("| rung | runs | target macro-F1 [95% CI] | effect size, own geometry | "
                   "effect size, reference frame |")
        out.append("|---|---|---|---|---|")
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        for rung in ladder:
            g = [r for r in pool if r["alignment"] == rung]
            if not g:
                continue
            arm = defaultdict(list)
            buckets = defaultdict(list)
            for r in g:
                buckets[(r["seed"], r["backbone"], r["layer_agg"], r["classifier"])].append(r)
            for (seed, *_), group in buckets.items():
                arm[seed].append(data.confusion(best_by_source_val(group)))
            stat = cluster_bootstrap(dict(arm), data.n_speakers(source, target),
                                     n_boot=N_BOOT, seed=17)
            own = seed_interval([r["marginal_mmd_normalised"] for r in g])
            ref = seed_interval([r["marginal_mmd_reference"] for r in g])
            out.append(
                f"| {rung} | {len(g)} | {fmt(stat['mean'], stat['lo'], stat['hi'])} | "
                f"{own['mean']:.2f} [{own['lo']:.2f}, {own['hi']:.2f}] | "
                f"{ref['mean']:.2f} [{ref['lo']:.2f}, {ref['hi']:.2f}] |"
            )
        floors = seed_interval([r["chance_macro_f1"] for r in pool])
        maj = seed_interval([r["majority_macro_f1"] for r in pool])
        out.append(f"| *chance floor* | | *{floors['mean']:.4f}* | | |")
        out.append(f"| *majority floor* | | *{maj['mean']:.4f}* | | |")
        out.append("")


def table_layer(data, out):
    out.append("## 3. Layer aggregation\n")
    for source, target in PAIRS:
        out.append(f"### {source} -> {target}\n")
        out.append("| aggregation | runs | families | target macro-F1 [95% CI] |")
        out.append("|---|---|---|---|")
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        for agg in ("last", "layer", "weighted"):
            g = [r for r in pool if r["layer_agg"] == agg]
            if not g:
                continue
            arm = defaultdict(list)
            buckets = defaultdict(list)
            for r in g:
                buckets[(r["seed"], r["backbone"], r["classifier"])].append(r)
            for (seed, *_), group in buckets.items():
                arm[seed].append(data.confusion(best_by_source_val(group)))
            stat = cluster_bootstrap(dict(arm), data.n_speakers(source, target),
                                     n_boot=N_BOOT, seed=17)
            families = ",".join(sorted({r["classifier"] for r in g}))
            out.append(f"| {agg} | {len(g)} | {families} | "
                       f"{fmt(stat['mean'], stat['lo'], stat['hi'])} |")
        out.append("")

    sweep = REPO_ROOT / "results/layer_sweep.jsonl"
    if sweep.exists():
        rows = [json.loads(line) for line in sweep.read_text().splitlines() if line.strip()]
        unique = {}
        for r in rows:
            unique.setdefault((r["backbone"], r["layer"], r["seed"]), r)
        by = defaultdict(list)
        for r in unique.values():
            by[(r["backbone"], r["layer"])].append(r)
        out.append("### 13-layer curve (Stage 1 artefact, logreg, rung `none`, 2 seeds)\n")
        out.append("Carried forward unchanged from Stage 1. Stage 2 did not re-run the "
                   "sweep, so this is **2 seeds and one classifier**, not full seed count.\n")
        backbones = sorted({b for b, _ in by})
        out.append("| layer | " + " | ".join(f"{b} source_val / target" for b in backbones) + " |")
        out.append("|---" * (len(backbones) + 1) + "|")
        for layer in sorted({layer for _, layer in by}):
            cells = []
            for backbone in backbones:
                g = by.get((backbone, layer), [])
                cells.append(
                    f"{np.mean([r['source_val_macro_f1'] for r in g]):.3f} / "
                    f"**{np.mean([r['target_macro_f1'] for r in g]):.3f}**" if g else "--"
                )
            out.append(f"| {layer} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("| backbone | argmax source_val | argmax target | gap |")
        out.append("|---|---|---|---|")
        for backbone in backbones:
            layers = sorted({layer for b, layer in by if b == backbone})
            val = {layer: np.mean([r["source_val_macro_f1"] for r in by[(backbone, layer)]])
                   for layer in layers}
            tgt = {layer: np.mean([r["target_macro_f1"] for r in by[(backbone, layer)]])
                   for layer in layers}
            a, b = max(val, key=val.get), max(tgt, key=tgt.get)
            out.append(f"| {backbone} | {a} | {b} | {b - a} |")
        out.append("")


def table_family(data, out):
    out.append("## 4. Classifier family\n")
    out.append("**The transformer is a reduced arm**: 2 seeds, primary direction only, "
               "one inner-grid setting per rung. Its interval is wider for that reason "
               "and its row is not comparable to the five-seed families.\n")
    for source, target in PAIRS:
        out.append(f"### {source} -> {target}\n")
        out.append("| family | runs | seeds | target macro-F1 [95% CI] |")
        out.append("|---|---|---|---|")
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        for family in ("logreg", "svm_linear", "svm_rbf", "mlp", "transformer"):
            g = [r for r in pool if r["classifier"] == family]
            if not g:
                out.append(f"| {family} | 0 | 0 | not run in this direction |")
                continue
            arm = defaultdict(list)
            buckets = defaultdict(list)
            for r in g:
                buckets[(r["seed"], r["backbone"], r["layer_agg"])].append(r)
            for (seed, *_), group in buckets.items():
                arm[seed].append(data.confusion(best_by_source_val(group)))
            stat = cluster_bootstrap(dict(arm), data.n_speakers(source, target),
                                     n_boot=N_BOOT, seed=17)
            flag = " *(reduced arm)*" if family == "transformer" else ""
            out.append(f"| {family}{flag} | {len(g)} | {stat['n_seeds']} | "
                       f"{fmt(stat['mean'], stat['lo'], stat['hi'])} |")
        out.append("")


def table_direction(data, out):
    out.append("## 5. Direction (matched-n)\n")
    out.append("**Both directions are matched-n**: cross-corpus `source_train` is capped "
               "to the smaller direction's size, so CREMA-D contributes 988 training "
               "utterances rather than 5972. Any asymmetry below is therefore not a "
               "training-set size effect. Full-n reverse has not been run.\n")
    out.append("| direction | runs | source_train n | target_test n | seeds | "
               "target macro-F1 [95% CI] | chance | majority |")
    out.append("|---|---|---|---|---|---|---|---|")
    for source, target in PAIRS:
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)
                and r["classifier"] != "transformer"]
        arm = defaultdict(list)
        buckets = defaultdict(list)
        for r in pool:
            buckets[(r["seed"], r["backbone"], r["layer_agg"], r["classifier"])].append(r)
        for (seed, *_), group in buckets.items():
            arm[seed].append(data.confusion(best_by_source_val(group)))
        stat = cluster_bootstrap(dict(arm), data.n_speakers(source, target),
                                 n_boot=N_BOOT, seed=17)
        n_train = sorted({r["source_train_n"] for r in pool})
        caps = sorted({str(r["source_train_cap"]) for r in pool})
        out.append(
            f"| {source}->{target} (matched-n) | {len(pool)} | "
            f"{n_train[0]}-{n_train[-1]} (cap {'/'.join(caps)}) | "
            f"{sorted({r['n_target_test'] for r in pool})[0]}-"
            f"{sorted({r['n_target_test'] for r in pool})[-1]} | "
            f"{stat['n_seeds']} | {fmt(stat['mean'], stat['lo'], stat['hi'])} | "
            f"{np.mean([r['chance_macro_f1'] for r in pool]):.4f} | "
            f"{np.mean([r['majority_macro_f1'] for r in pool]):.4f} |"
        )
    out.append("")
    out.append("Excludes the transformer, which ran the primary direction only and would "
               "otherwise weight one side of this comparison.\n")


def table_blending(data, out):
    out.append("## 6. Blending (the alpha arm)\n")
    out.append(f"{len(data.blend)} runs. `scalar` mode only; `gaa` is not implemented "
               "and was not run.\n")
    if not data.blend:
        out.append("_no blending runs_\n")
        return
    out.append("alpha=0 discards the alignment entirely, so the two rungs' alpha=0 "
               "rows are different `run_id`s computed over identical features. They "
               "agree to every reported digit, which is an end-to-end check on the "
               "blending path -- and 18 runs per direction of duplicated compute.\n")
    out.append("The alpha=1.00 row is drawn from the main grid at **exactly the "
               "inner-grid setting the arm used** (coral eps=0.1, mkmmd_full "
               "lambda=0.01) and the same backbone, seeds and families -- not "
               "pooled over the main grid's other eps and lambda values, which "
               "would make it a different experiment.\n")
    for source, target in PAIRS:
        blended = [r for r in data.blend
                   if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        if not blended:
            continue
        out.append(f"### {source} -> {target}\n")
        out.append("| alignment | alpha | runs | target macro-F1 [95% CI] |")
        out.append("|---|---|---|---|")
        for rung in sorted({r["alignment"] for r in blended}):
            arm_rows = [r for r in blended if r["alignment"] == rung]
            alphas = sorted({r["blend_alpha"] for r in arm_rows})
            spec = {
                "backbone": {r["backbone"] for r in arm_rows},
                "seed": {r["seed"] for r in arm_rows},
                "classifier": {r["classifier"] for r in arm_rows},
                "layer_agg": {r["layer_agg"] for r in arm_rows},
                "alignment_eps": {r["alignment_eps"] for r in arm_rows},
                "alignment_lambda": {r["alignment_lambda"] for r in arm_rows},
            }
            reference = [
                r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)
                and r["alignment"] == rung
                and all(r[key] in allowed for key, allowed in spec.items())
            ]
            for alpha in alphas + [1.0]:
                g = ([r for r in arm_rows if r["blend_alpha"] == alpha]
                     if alpha != 1.0 else reference)
                if not g:
                    continue
                arm = defaultdict(list)
                buckets = defaultdict(list)
                for r in g:
                    buckets[(r["seed"], r["backbone"], r["layer_agg"], r["classifier"])].append(r)
                for (seed, *_), group in buckets.items():
                    arm[seed].append(data.confusion(best_by_source_val(group)))
                stat = cluster_bootstrap(dict(arm), data.n_speakers(source, target),
                                         n_boot=500, seed=17)
                note = " *(= blending none)*" if alpha == 1.0 else ""
                out.append(f"| {rung} | {alpha:.2f}{note} | {len(g)} | "
                           f"{fmt(stat['mean'], stat['lo'], stat['hi'])} |")
        out.append("")


def table_per_class(data, out):
    out.append("## 7. Per-class F1, from the stored predictions\n")
    out.append("Computed by summing the per-speaker confusion matrices of the "
               "**validated** configuration in each (pair, seed), then taking per-class "
               "F1 of the total. Interval is over seeds.\n")
    for source, target in PAIRS:
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        by_seed = defaultdict(list)
        for r in pool:
            by_seed[r["seed"]].append(r)
        per_seed = []
        supports = np.zeros(len(data.classes))
        for seed in sorted(by_seed):
            chosen = best_by_source_val(by_seed[seed])
            conf = data.confusion(chosen).sum(axis=0)
            per_seed.append(per_class_f1_from_confusion(conf))
            supports = conf.sum(axis=1)
        matrix = np.vstack(per_seed)
        out.append(f"### {source} -> {target}\n")
        out.append("| class | support | F1 [95% CI over seeds] |")
        out.append("|---|---|---|")
        for i, name in enumerate(data.classes):
            interval = seed_interval(matrix[:, i].tolist())
            out.append(f"| {name} | {int(supports[i])} | "
                       f"{fmt(interval['mean'], interval['lo'], interval['hi'])} |")
        macro = seed_interval(np.nanmean(matrix, axis=1).tolist())
        out.append(f"| **macro** | {int(supports.sum())} | "
                   f"**{fmt(macro['mean'], macro['lo'], macro['hi'])}** |")
        out.append("")


def table_secondary_ladder(data, out):
    out.append("## 8. Secondary — differences among the aligned rungs\n")
    out.append("**Not in the primary family. Intervals only; no significance is "
               "claimed or implied for any row here.** The primary tests compare "
               "each rung against `none`; this table is the remaining question of "
               "whether the rungs differ from *each other*, which is what "
               "\"flat across the ladder\" refers to. 500 replicates.\n")
    rungs = [r for r in data.config.alignment.ladder_order() if r != "none"]
    for source, target in PAIRS:
        out.append(f"### {source} -> {target}\n")
        out.append("| comparison | conditions | difference in target macro-F1 [95% CI] |")
        out.append("|---|---|---|")
        widest = None
        for i, a in enumerate(rungs):
            for b in rungs[i + 1:]:
                arms = arms_for(data, data.main, "alignment", a, b)
                arm_a, arm_b = arms[(source, target)]
                stat = paired_cluster_bootstrap(
                    arm_b, arm_a, data.n_speakers(source, target), n_boot=500, seed=17
                )
                if stat["n_seeds"] == 0:
                    continue
                out.append(
                    f"| {b} - {a} | {sum(len(v) for v in arm_a.values())} | "
                    f"{stat['diff']:+.4f} [{stat['lo']:+.4f}, {stat['hi']:+.4f}] |"
                )
                if widest is None or abs(stat["diff"]) > abs(widest[1]):
                    widest = (f"{b} - {a}", stat["diff"], stat["lo"], stat["hi"])
        if widest:
            out.append("")
            out.append(f"Largest absolute difference among aligned rungs: "
                       f"**{widest[0]} = {widest[1]:+.4f} "
                       f"[{widest[2]:+.4f}, {widest[3]:+.4f}]**.")
        out.append("")


def table_stage1_recheck(data, out):
    """Stage 1 observations re-measured at full seed count. Numbers only."""
    out.append("## 9. Stage 1 observations, re-measured\n")
    out.append("The Stage 1 figure is quoted from `reports/stage1_analysis.md` "
               "(2 seeds, hubert, ravdess->cremad, pre-selection). The Stage 2 "
               "column is the same quantity at full seed count. **Verdicts are in "
               "pass 2, not here.**\n")

    out.append("### CORAL `source_val` against eps (the mis-centred grid)\n")
    out.append("| eps | Stage 1 source_val | Stage 2 source_val | Stage 2 target macro-F1 |")
    out.append("|---|---|---|---|")
    stage1 = {1e-4: 0.5592, 1e-3: 0.5713, 1e-2: 0.6032, 1e-1: 0.6523, None: 0.6096}
    coral = [r for r in data.main if r["alignment"] == "coral"]
    for eps in sorted({r["alignment_eps"] for r in coral}, key=lambda e: (e is None, e)):
        g = [r for r in coral if r["alignment_eps"] == eps]
        label = "ledoit-wolf" if eps is None else f"{eps:g}"
        was = stage1.get(eps)
        out.append(
            f"| {label} | {f'{was:.4f}' if was else 'not in grid'} | "
            f"{np.mean([r['selection_source_val_macro_f1'] for r in g]):.4f} | "
            f"{np.mean([r['macro_f1'] for r in g]):.4f} |"
        )
    best = max(
        {r["alignment_eps"] for r in coral if r["alignment_eps"] is not None},
        key=lambda e: np.mean([r["selection_source_val_macro_f1"] for r in coral
                               if r["alignment_eps"] == e]),
    )
    out.append("")
    out.append(f"`source_val` argmax over the numeric eps grid: **{best:g}** "
               f"(grid maximum is {max(data.config.alignment.coral_shrinkage):g}).\n")

    out.append("### Alignment gain by layer aggregation (the interaction)\n")
    out.append("| pair | aggregation | target, `none` | target, aligned | gain | Stage 1 gain |")
    out.append("|---|---|---|---|---|---|")
    stage1_gain = {"last": 0.042, "layer": 0.168, "weighted": 0.095}
    for source, target in PAIRS:
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        for agg in ("last", "layer", "weighted"):
            g = [r for r in pool if r["layer_agg"] == agg]
            if not g:
                continue
            unaligned = [r["macro_f1"] for r in g if r["alignment"] == "none"]
            aligned = [r["macro_f1"] for r in g if r["alignment"] != "none"]
            if not unaligned or not aligned:
                continue
            gain = np.mean(aligned) - np.mean(unaligned)
            was = stage1_gain[agg] if source == "ravdess" else None
            out.append(
                f"| {source}->{target} | {agg} | {np.mean(unaligned):.4f} | "
                f"{np.mean(aligned):.4f} | {gain:+.4f} | "
                f"{f'{was:+.3f}' if was is not None else 'not measured'} |"
            )
    out.append("")

    out.append("### Frame dependence across the ladder\n")
    out.append("Spearman rho over the six rung means, target macro-F1 against each "
               "discrepancy column. Six points, so this is a coarse re-measurement "
               "of the layer-sweep finding on a different axis, not a replication "
               "of it.\n")

    def spearman(x, y):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            out_ = [0.0] * len(v)
            for pos, idx in enumerate(order):
                out_[idx] = float(pos)
            return out_
        rx, ry = rank(x), rank(y)
        return float(np.corrcoef(rx, ry)[0, 1])

    out.append("| pair | rho(own geometry, target) | rho(reference frame, target) |")
    out.append("|---|---|---|")
    for source, target in PAIRS:
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        own, ref, tgt = [], [], []
        for rung in data.config.alignment.ladder_order():
            g = [r for r in pool if r["alignment"] == rung]
            if not g:
                continue
            own.append(np.mean([r["marginal_mmd_normalised"] for r in g]))
            ref.append(np.mean([r["marginal_mmd_reference"] for r in g]))
            tgt.append(np.mean([r["macro_f1"] for r in g]))
        out.append(f"| {source}->{target} | {spearman(own, tgt):+.3f} | "
                   f"{spearman(ref, tgt):+.3f} |")
    out.append("")

    out.append("### MK-MMD lambda on `source_val`\n")
    out.append("| rung | lambda | source_val | target macro-F1 | fallback rate |")
    out.append("|---|---|---|---|---|")
    for rung in ("mkmmd_diag", "mkmmd_full"):
        g = [r for r in data.main if r["alignment"] == rung]
        values = []
        for lam in sorted({r["alignment_lambda"] for r in g}):
            h = [r for r in g if r["alignment_lambda"] == lam]
            values.append(np.mean([r["selection_source_val_macro_f1"] for r in h]))
            out.append(
                f"| {rung} | {lam:g} | {values[-1]:.4f} | "
                f"{np.mean([r['macro_f1'] for r in h]):.4f} | "
                f"{sum(1 for r in h if r['mmd_fallback_fired']) / len(h):.1%} |"
            )
        out.append(f"| {rung} | **spread** | **{max(values) - min(values):.4f}** | | |")
    out.append("")


def main() -> int:
    data = Data()
    out = ["# Phase 8 pass 1 — tables", ""]
    out.append("Numbers only. No interpretation appears in this file; pass 2 lives in "
               "`reports/phase8_interpretation.md`.\n")
    out.append("Every target figure is a mean over seeds with a 95% interval from a "
               "**paired cluster bootstrap** that resamples target_test speakers and "
               "seeds together. Discrepancy columns use a t-interval over seeds instead: "
               "they are properties of a fitted map, not of a prediction, so they have no "
               "per-utterance form to bootstrap.\n")
    out.append("---\n")
    table_integrity(data, out)
    table_validated_oracle(data, out)
    tests = run_primary(data, out)
    table_ladder(data, out)
    table_layer(data, out)
    table_family(data, out)
    table_direction(data, out)
    table_blending(data, out)
    table_per_class(data, out)
    table_secondary_ladder(data, out)
    table_stage1_recheck(data, out)

    path = REPO_ROOT / "reports/phase8_tables.md"
    path.write_text("\n".join(out), encoding="utf-8")
    (REPO_ROOT / "reports/phase8_primary.json").write_text(
        json.dumps([{k: (list(v) if isinstance(v, tuple) else v) for k, v in t.items()}
                    for t in tests], indent=2),
        encoding="utf-8",
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
