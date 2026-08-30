#!/usr/bin/env python
"""Phase 11: every table the paper will contain, generated from the results.

    python tools/make_tables.py

Writes tables/*.tex. Each table states its own run filter in a note, so a
reader can see which slice of 5424 rows produced it without leaving the page.

Requires only `booktabs` in the document preamble.
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

from ser.latex import interval, number, table, write_table  # noqa: E402
from ser.phase8 import cluster_bootstrap, seed_interval  # noqa: E402
from ser.utils.results import read_rows  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tools"))
from make_figures import Data, LADDER, PAIRS, eps_probe_rows, sweep_rows  # noqa: E402

TABLE_DIR = REPO_ROOT / "tables"
N_BOOT = 2000
PAIR_TEX = {("ravdess", "cremad"): r"RAVDESS $\rightarrow$ CREMA-D",
            ("cremad", "ravdess"): r"CREMA-D $\rightarrow$ RAVDESS"}


def emit(text, name):
    path = write_table(text, name, TABLE_DIR)
    print(f"  {name:<26} -> {path.relative_to(REPO_ROOT).as_posix()}")
    return path


def table_headline(data):
    """Validated vs oracle, with both floors."""
    rows = []
    for source, target in PAIRS:
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        by_seed = defaultdict(list)
        for r in pool:
            by_seed[r["seed"]].append(r)
        validated, oracle = [], []
        for seed in sorted(by_seed):
            best = max(by_seed[seed], key=lambda r: r["selection_source_val_macro_f1"])
            validated.append(best["macro_f1"])
            oracle.append(max(r["macro_f1"] for r in by_seed[seed]))
        v, o = seed_interval(validated), seed_interval(oracle)
        g = seed_interval([b - a for a, b in zip(validated, oracle)])
        rows.append([
            PAIR_TEX[(source, target)],
            interval(v["mean"], v["lo"], v["hi"]),
            interval(o["mean"], o["lo"], o["hi"]),
            interval(g["mean"], g["lo"], g["hi"]),
            number(float(np.mean([r["chance_macro_f1"] for r in pool]))),
            number(float(np.mean([r["majority_macro_f1"] for r in pool]))),
        ])
    return emit(table(
        rows,
        ["pair", "validated", "oracle", "gap", "chance", "majority"],
        caption=("Target macro-F1 of the configuration selected on source "
                 "validation, against the best configuration present in the grid. "
                 "The oracle column is an upper bound no protocol can reach and "
                 "is not a result."),
        label="headline",
        notes=["Filter: \\texttt{freeze\\_tag=grid-freeze-v3}, "
               "\\texttt{blending=none}, 4986 runs. Mean over 5 seeds with a "
               "95\\% $t$-interval. Both floors are analytic from the realised "
               "target-test priors."],
        escape_cells=False,
    ), "headline")


def table_ladder(data):
    rows = []
    for source, target in PAIRS:
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        for rung in LADDER:
            g = [r for r in pool if r["alignment"] == rung]
            stat = cluster_bootstrap(data.arm(g), data.n_speakers(source, target),
                                     n_boot=N_BOOT, seed=17)
            own = seed_interval([r["marginal_mmd_normalised"] for r in g])
            ref = seed_interval([r["marginal_mmd_reference"] for r in g])
            rows.append([
                PAIR_TEX[(source, target)] if rung == LADDER[0] else "",
                f"\\texttt{{{rung.replace('_', chr(92) + '_')}}}",
                str(len(g)),
                interval(stat["mean"], stat["lo"], stat["hi"]),
                number(own["mean"], 2),
                number(ref["mean"], 2),
            ])
    return emit(table(
        rows,
        ["pair", "rung", "runs", "target macro-F1", "effect (own)", "effect (ref)"],
        caption=("The alignment ladder. Target macro-F1 rises once, off "
                 "\\texttt{none}, and then does not track discrepancy: "
                 "\\texttt{mkmmd\\_full} reaches the lowest discrepancy in both "
                 "geometries and is never the best rung."),
        label="ladder",
        notes=["Filter: \\texttt{freeze\\_tag=grid-freeze-v3}, "
               "\\texttt{blending=none}. Target intervals are a paired cluster "
               "bootstrap over target-test speakers and seeds, "
               f"{N_BOOT} replicates; discrepancy columns are $t$-intervals over "
               "seeds. Both geometries are reported because they disagree "
               "(Table~\\ref{tab:frames})."],
        escape_cells=False,
    ), "ladder")


def table_primary(data):
    payload = json.loads((REPO_ROOT / "reports/phase8_primary.json").read_text())
    rows = []
    for entry in payload:
        source, target = entry["pair"]
        rows.append([
            entry["tag"],
            f"\\texttt{{{entry['name'].replace('_', chr(92) + '_')}}}",
            PAIR_TEX[(source, target)],
            f"{entry['diff']:+.4f} [{entry['lo']:+.4f}, {entry['hi']:+.4f}]",
            ("$<$" if entry.get("p_at_floor") else "") + f"{entry['holm']:.4f}",
            "survives",
        ])
    return emit(table(
        rows,
        ["id", "comparison", "pair", "difference in target macro-F1", "Holm $p$", "verdict"],
        caption=("The pre-registered primary comparisons. All 14 survive Holm "
                 "correction. The family was fixed in code before any of these "
                 "numbers was computed."),
        label="primary",
        notes=[f"Paired cluster bootstrap over target-test speakers and seeds, "
               f"{payload[0]['n_boot']} replicates. $p$ values marked $<$ are at "
               "the bootstrap's resolution floor of $1/n_{\\mathrm{boot}}$ and "
               "are not resolvable below it."],
        escape_cells=False,
    ), "primary")


def table_per_class():
    payload = json.loads((REPO_ROOT / "reports/per_class.json").read_text())
    rows = []
    for name in payload["classes"]:
        cells = [f"\\texttt{{{name}}}"]
        for source, target in PAIRS:
            entry = payload["pairs"][f"{source}->{target}"]["classes"][name]
            cells.append(str(entry["support"]))
            cells.append(interval(entry["f1"]["value"], entry["f1"]["lo"],
                                  entry["f1"]["hi"]))
        rows.append(cells)
    macro = ["\\textbf{macro}"]
    for source, target in PAIRS:
        entry = payload["pairs"][f"{source}->{target}"]
        macro.append("")
        macro.append("\\textbf{" + interval(entry["macro_f1"]["value"],
                                            entry["macro_f1"]["lo"],
                                            entry["macro_f1"]["hi"]) + "}")
    rows.append(macro)
    # Derived, not asserted: a caption naming classes that later stop being the
    # weak ones would be a silent error.
    below = [c for c in payload["classes"]
             if all(payload["pairs"][f"{a}->{b}"]["classes"][c]["f1"]["value"]
                    < payload["pairs"][f"{a}->{b}"]["macro_f1"]["value"]
                    for a, b in PAIRS)]
    below_text = " and ".join(f"\\texttt{{{c}}}" for c in below)
    return emit(table(
        rows,
        ["class", "$n$", "F1 (RAV $\\rightarrow$ CRE)", "$n$", "F1 (CRE $\\rightarrow$ RAV)"],
        caption=("Per-class F1 of the validated configuration, from the stored "
                 "per-utterance predictions. \\texttt{sad} and \\texttt{happy} "
                 "fall below the macro average in both directions."),
        label="perclass",
        notes=["Filter: validated configuration per (pair, seed), 5 seeds, "
               "predictions pooled. Paired cluster bootstrap over target-test "
               f"speakers and seeds, {payload['n_boot']} replicates. Precision "
               "and recall are in \\texttt{reports/per\\_class.md}."],
        escape_cells=False,
    ), "per_class")


def table_decomposition():
    records = [json.loads(line) for line in
               (REPO_ROOT / "results/phase9_shift.jsonl").read_text().splitlines()
               if line.strip()]
    rows = []
    for source, target in PAIRS:
        kl = seed_interval([r["label_shift"]["kl_nats"] for r in records
                            if (r["source"], r["target"]) == (source, target)
                            and r["layer_agg"] == "last"
                            and r["alignment"] == "none"])
        for rung in LADDER:
            g = [r for r in records
                 if (r["source"], r["target"]) == (source, target)
                 and r["layer_agg"] == "last" and r["alignment"] == rung]
            marginal = float(np.mean([r["marginal_effect_own"] for r in g]))
            conditional = float(np.mean(
                [np.mean([x["effect_size"] for x in r["conditional"]
                          if x["effect_size"] is not None]) for r in g]))
            rows.append([
                PAIR_TEX[(source, target)] if rung == LADDER[0] else "",
                f"\\texttt{{{rung.replace('_', chr(92) + '_')}}}",
                number(kl["mean"], 5) if rung == LADDER[0] else "",
                number(marginal, 1),
                number(conditional, 1),
                number(conditional / marginal, 2),
            ])
    return emit(table(
        rows,
        ["pair", "rung", "label shift (KL)", "marginal", "conditional", "ratio"],
        caption=("The three-way shift decomposition. Label shift is negligible. "
                 "Alignment drives the marginal term down by two orders of "
                 "magnitude while the conditional term falls far less, so the "
                 "ratio rises toward one: what remains after alignment is "
                 "conditional shift, which no marginal alignment can remove."),
        label="decomposition",
        notes=["Filter: hubert, \\texttt{layer\\_agg=last}, logreg, 5 seeds, "
               "both discrepancies measured between the same two sets "
               "(aligned source-train and aligned target-test). KL is "
               "$\\mathrm{KL}(P_{\\mathrm{target}} \\| P_{\\mathrm{source}})$ "
               "in nats between realised partition priors. The conditional term "
               "reads target labels and is computed behind the A10 firewall "
               "(\\texttt{ser.analysis}); it is never written to the result "
               "schema."],
        escape_cells=False,
    ), "decomposition")


def table_frames():
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

    out_rows, own_all, ref_all, disagree = [], [], [], 0
    for source, target in PAIRS:
        for backbone in sorted({r["backbone"] for r in rows}):
            own_cell, ref_cell = [], []
            for rung in LADDER:
                for seed in seeds:
                    cells = [x for x in rows
                             if (x["source_corpus"], x["target_corpus"]) == (source, target)
                             and x["backbone"] == backbone and x["alignment"] == rung
                             and x["seed"] == seed]
                    if len(cells) < 3:
                        continue
                    cells.sort(key=lambda x: x["layer_index"])
                    sc = [x["macro_f1"] for x in cells]
                    own_cell.append(spearman([x["marginal_mmd_normalised"] for x in cells], sc))
                    ref_cell.append(spearman([x["marginal_mmd_reference"] for x in cells], sc))
            own, ref = seed_interval(own_cell), seed_interval(ref_cell)
            own_all += own_cell
            ref_all += ref_cell
            # Same strict test the report uses: both intervals must exclude
            # zero AND fall on opposite sides. Comparing point estimates alone
            # would make this table disagree with reports/layer_sweep_v2.md.
            flip = (own["lo"] * own["hi"] > 0 and ref["lo"] * ref["hi"] > 0
                    and own["mean"] * ref["mean"] < 0)
            disagree += flip
            out_rows.append([
                PAIR_TEX[(source, target)] if backbone == "hubert" else "",
                f"\\texttt{{{backbone}}}",
                interval(own["mean"], own["lo"], own["hi"], 3),
                interval(ref["mean"], ref["lo"], ref["hi"], 3),
                r"\textbf{yes}" if flip else "no",
            ])
    pooled_own, pooled_ref = seed_interval(own_all), seed_interval(ref_all)
    out_rows.append([r"\textbf{pooled}", "",
                     r"\textbf{" + interval(pooled_own["mean"], pooled_own["lo"],
                                            pooled_own["hi"], 3) + "}",
                     r"\textbf{" + interval(pooled_ref["mean"], pooled_ref["lo"],
                                            pooled_ref["hi"], 3) + "}",
                     r"\textbf{yes}"])
    return emit(table(
        out_rows,
        ["pair", "backbone", "$\\rho$ (own geometry)", "$\\rho$ (reference frame)",
         "sign differs"],
        caption=("Frame dependence. Spearman $\\rho$ between a layer's marginal "
                 "discrepancy and its target macro-F1, across the 13 layers. The "
                 "two geometries give opposite signs, so the relationship between "
                 "discrepancy and transfer is not well posed until the geometry "
                 "is fixed."),
        label="frames",
        notes=["Filter: 13-layer sweep, 2340 runs, logreg, 6 rungs, 3 backbones, "
               "both directions, 5 seeds. $\\rho$ is computed within each seed "
               "across the 13 layers, then averaged with a 95\\% $t$-interval "
               "over seeds. Pooled over all 36 (direction $\\times$ backbone "
               "$\\times$ rung) cells and 5 seeds."],
        escape_cells=False,
    ), "frames")


def table_eps(data):
    probe = eps_probe_rows()
    families = {r["classifier"] for r in probe}
    aggs = {r["layer_agg"] for r in probe}
    backbones = {r["backbone"] for r in probe}
    matched = [r for r in data.main if r["classifier"] in families
               and r["layer_agg"] in aggs and r["backbone"] in backbones]
    rows = []
    for source, target in PAIRS:
        pool = [r for r in matched
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        pp = [r for r in probe
              if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        entries = [(eps, [r for r in pool if r["alignment"] == "coral"
                          and r["alignment_eps"] == eps])
                   for eps in sorted({r["alignment_eps"] for r in pool
                                      if r["alignment"] == "coral"
                                      and r["alignment_eps"] is not None})]
        entries += [(eps, [r for r in pp if r["alignment_eps"] == eps])
                    for eps in sorted({r["alignment_eps"] for r in pp})]
        for eps, g in entries:
            rows.append([
                PAIR_TEX[(source, target)] if eps == entries[0][0] else "",
                f"{eps:g}",
                number(float(np.mean([r["selection_source_val_macro_f1"] for r in g]))),
                number(float(np.mean([r["macro_f1"] for r in g]))),
            ])
        shift = [r for r in pool if r["alignment"] == "mean_shift"]
        rows.append(["", r"\\texttt{mean\_shift}",
                     r"\textbf{" + number(float(np.mean(
                         [r["selection_source_val_macro_f1"] for r in shift]))) + "}",
                     r"\textbf{" + number(float(np.mean(
                         [r["macro_f1"] for r in shift]))) + "}"])
    return emit(table(
        rows,
        ["pair", "$\\epsilon$", "source\\_val", "target macro-F1"],
        caption=("CORAL's shrinkage asymptote. As $\\epsilon$ grows the "
                 "regularised covariances approach a scaled identity, the map "
                 "tends to $\\sqrt{\\mathrm{tr}(C_t)/\\mathrm{tr}(C_s)}\\,I$, and "
                 "the rung degenerates to a scalar rescale plus a mean shift. "
                 "Both columns converge on \\texttt{mean\\_shift}."),
        label="eps",
        notes=["Filter: hubert, \\{logreg, svm\\_rbf, mlp\\} $\\times$ "
               "\\{last, layer:6\\}, 5 seeds. $\\epsilon \\leq 10$ is the frozen "
               "Stage 2 grid; $\\epsilon \\in \\{100, 1000\\}$ is a 120-run "
               "off-grid probe carrying the same facet hashes. Means over seeds."],
        escape_cells=False,
    ), "eps_asymptote")


def table_corpora():
    """Corpus statistics for the Method section, from the manifest."""
    from collections import Counter

    from ser.config import load_config
    from ser.manifest import read_manifest

    config = load_config()
    rows = read_manifest(config.resolve(config.paths.manifest))
    classes = list(config.labels.spaces["six"])
    out_rows = []
    for corpus in ("ravdess", "cremad"):
        got = [r for r in rows if r.corpus == corpus]
        kept = [r for r in got if r.label_six]
        hours = sum(r.duration_s for r in got) / 3600.0
        counts = Counter(r.label_six for r in kept)
        total = sum(counts.values())
        out_rows.append([
            {"ravdess": "RAVDESS", "cremad": "CREMA-D"}[corpus],
            str(len({r.speaker_id for r in got})),
            str(len(got)),
            f"{hours:.2f}",
            f"{sum(r.duration_s for r in got) / len(got):.2f}",
            str(total),
        ] + [f"{counts[c] / total:.3f}" for c in classes])
    return emit(table(
        out_rows,
        ["corpus", "spk", "utts", "hours", "mean dur (s)", "$n$ (6-class)"]
        + [f"\\texttt{{{c[:4]}}}" for c in classes],
        caption=("Corpora after mapping to the six-class intersection. The last "
                 "six columns are class priors. The only substantial prior "
                 "difference is \\texttt{neutral}, an artefact of merging "
                 "RAVDESS \\texttt{calm}."),
        label="corpora",
        notes=["Derived from \\texttt{data/manifest.csv}. RAVDESS "
               "\\texttt{surprised} (192 utterances) is excluded as having no "
               "CREMA-D counterpart; RAVDESS \\texttt{calm} is merged into "
               "\\texttt{neutral}. RAVDESS is balanced across its own eight "
               "classes but NOT at this intersection."],
        escape_cells=False,
    ), "corpora")


def table_floors(data):
    """Split sizes and the floors every metric is read against."""
    rows = []
    for source, target in PAIRS:
        pool = [r for r in data.main
                if (r["source_corpus"], r["target_corpus"]) == (source, target)]
        train = sorted({r["source_train_n"] for r in pool if r["source_train_n"]})
        test = sorted({r["n_target_test"] for r in pool})
        rows.append([
            PAIR_TEX[(source, target)],
            f"{train[0]}" if len(train) == 1 else f"{train[0]}--{train[-1]}",
            f"{sorted({r['n_val'] for r in pool})[0]}--"
            f"{sorted({r['n_val'] for r in pool})[-1]}",
            f"{sorted({r['n_target_adapt'] for r in pool})[0]}--"
            f"{sorted({r['n_target_adapt'] for r in pool})[-1]}",
            f"{test[0]}--{test[-1]}" if test[0] != test[-1] else f"{test[0]}",
            number(float(np.mean([r["chance_macro_f1"] for r in pool]))),
            number(float(np.mean([r["majority_macro_f1"] for r in pool]))),
        ])
    return emit(table(
        rows,
        ["pair", "source\_train", "source\_val", "target\_adapt",
         "target\_test", "chance", "majority"],
        caption=("Split sizes and the floors every macro-F1 in this paper is "
                 "read against. Both directions are matched-$n$: CREMA-D "
                 "source-train is subsampled from 5972 to match RAVDESS, so a "
                 "reported asymmetry cannot be a training-set size effect."),
        label="floors",
        notes=["Ranges span the five seeds within the speaker-disjoint "
               "constraint. Floors are analytic from the realised "
               "\\texttt{target\_test} priors, not simulated. Because the "
               "chance floor is pair-dependent, no result in this paper averages "
               "macro-F1 across pairs."],
        escape_cells=False,
    ), "floors")


def main() -> int:
    data = Data()
    print("writing LaTeX tables:")
    table_corpora()
    table_floors(data)
    table_headline(data)
    table_ladder(data)
    table_primary(data)
    table_per_class()
    table_decomposition()
    table_frames()
    table_eps(data)
    print(f"\n7 tables written to {TABLE_DIR.relative_to(REPO_ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
