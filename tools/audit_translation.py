"""Generate a retrospective audit from existing runs; never train or edit them.

The table is drawn in two panels. The first is the exact prediction test:
classifier families whose fitting rule satisfies Proposition 1's assumptions.
The second is scope evidence: families the manuscript already predicts are
outside those assumptions, reported so the boundary is confirmed on frozen
rows rather than only asserted. The split comes from the configuration, not
from this script, so a reader can check which side a family was placed on and
why without reading code.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ser.analysis.affine_audit import match_translation_rows
from ser.freeze import FROZEN_LEDGER, assert_ledger_unchanged

DIRECTION_TEX = {"ravdess": r"RAVDESS $\to$ CREMA-D", "cremad": r"CREMA-D $\to$ RAVDESS"}


def summarise(matched, classifiers):
    """One descriptive row per (direction, classifier), in configured order."""
    groups = defaultdict(list)
    for row in matched:
        groups[(row["source_corpus"], row["target_corpus"], row["classifier"])].append(row)
    order = {name: index for index, name in enumerate(classifiers)}
    summaries = []
    for (source, target, classifier), rows in sorted(
        groups.items(), key=lambda item: (order[item[0][2]], item[0][0])
    ):
        summaries.append(dict(
            source=source, target=target, classifier=classifier, n=len(rows),
            validation_equal=sum(r["validation_equal"] for r in rows),
            hyperparameters_equal=sum(r["hyperparameters_equal"] for r in rows),
            both_equal=sum(r["validation_equal"] and r["hyperparameters_equal"] for r in rows),
            positive_target_differences=sum(r["target_difference"] > 0 for r in rows),
            mean_target_difference=mean(r["target_difference"] for r in rows),
            mean_none_target=mean(r["none_target"] for r in rows),
            mean_shift_target=mean(r["mean_shift_target"] for r in rows),
            chance=mean(r["chance_macro_f1"] for r in rows),
            seeds=sorted({r["seed"] for r in rows}),
            layer_aggregations=sorted({r["layer_agg"] for r in rows}),
            exact_prediction=None,  # filled by the caller, which knows the split
        ))
    return summaries


def markdown(report, summaries, config):
    names = config["classifier_names"]
    lines = [
        "# Source-translation audit", "",
        "Retrospective analysis of frozen runs. No training.",
        "Stored-score equality is exact, not rounded. The full trial surface is not stored.",
        "Rows share seeds and are not independent experimental replications.",
        "",
        "`exact prediction` marks the families whose fitting rule satisfies "
        "Proposition 1's assumptions. The remaining families are reported as "
        "scope evidence for a boundary the manuscript states in advance; they "
        "are not counterexamples to the proposition.",
        "",
        f"Input SHA256: `{report['input_sha256']}`",
        f"Config SHA256: `{report['config_sha256']}`", "",
        "| Direction | Classifier | Exact prediction | Pairs | Equal validation | "
        "Equal hyperparameters | Both equal | Positive target differences | "
        "Mean target difference | Chance |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['source']} -> {s['target']} | {names[s['classifier']]} | "
            f"{'yes' if s['exact_prediction'] else 'no'} | {s['n']} | "
            f"{s['validation_equal']} | {s['hyperparameters_equal']} | {s['both_equal']} | "
            f"{s['positive_target_differences']} | {s['mean_target_difference']:+.4f} | "
            f"{s['chance']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def latex(summaries):
    exact = [s for s in summaries if s["exact_prediction"]]
    boundary = [s for s in summaries if not s["exact_prediction"]]
    tex = [
        r"\begin{table*}[!t]", r"\centering\small",
        r"\caption{Retrospective translation audit on frozen runs. Each cell is "
        r"one matched \texttt{none}/\texttt{mean\_shift} pair sharing direction, "
        r"seed, backbone, layer setting, classifier and every split, label-map, "
        r"feature and search identifier. Equal validation and equal parameters "
        r"count exact equality of the stored winning source-validation score and "
        r"the selected classifier parameters. $\Delta$ is mean target macro-F1 "
        r"for mean shift minus none; Positive counts cells with a target "
        r"improvement. Chance is the uniform-random baseline. The upper panel is "
        r"the exact prediction of Proposition 1. The lower panel reports families "
        r"whose fitting rules fall outside its assumptions, which the proposition "
        r"does not cover and which are therefore scope evidence rather than "
        r"counterexamples. Cells share seeds; no independence-based test is "
        r"reported and no general claim about mean shift follows from the $\Delta$ "
        r"column.}",
        r"\label{tab:translation-audit}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Direction & Head & Pairs & Equal val. & Equal params & $\Delta$ & Positive & Chance \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textbf{Stationary kernel, Gram-determined fit: "
        r"Proposition 1 applies}} \\",
    ]

    def body(rows):
        for s in rows:
            tex.append(
                f"{DIRECTION_TEX[s['source']]} & {s['name']} & {s['n']} & "
                f"{s['validation_equal']} & {s['hyperparameters_equal']} & "
                f"{s['mean_target_difference']:+.4f} & "
                f"{s['positive_target_differences']} & {s['chance']:.4f} " + r"\\"
            )

    body(exact)
    if boundary:
        tex.extend([
            r"\midrule",
            r"\multicolumn{8}{l}{\textbf{Penalised intercept or iterative solver: "
            r"outside those assumptions}} \\",
        ])
        body(boundary)
    tex.extend([
        r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
        r"\begin{minipage}{\linewidth}\footnotesize "
        r"Filter: \texttt{grid-freeze-v3}, \texttt{blending=none}; "
        r"matched \texttt{none}/\texttt{mean\_shift} rows only. Every family "
        r"spans three backbones and five seeds; the MLP additionally has the "
        r"learned \texttt{weighted} layer aggregation, which the closed-form "
        r"heads cannot use, hence its larger cell count. The reduced-seed "
        r"transformer arm has no balanced matched cells and is excluded by "
        r"construction. Generated by \texttt{tools/audit\_translation.py} from "
        r"the configuration \texttt{configs/audit\_translation.yaml}. The JSON "
        r"report retains input/configuration hashes and both run IDs for every "
        r"pair. Means are descriptive, not independent-cell inference."
        r"\end{minipage}", r"\end{table*}",
    ])
    return "\n".join(tex) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/audit_translation.yaml"))
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)

    source = Path(config["input"])
    # Refuse to summarise a ledger that is no longer the one this analysis was
    # written against. The digest lives in configs/FROZEN_LEDGER.sha256 and is
    # not repeated here; see ser.freeze.
    if source.as_posix().endswith(FROZEN_LEDGER):
        assert_ledger_unchanged(source)

    data = source.read_bytes()
    matched = match_translation_rows(
        [json.loads(line) for line in data.splitlines() if line.strip()],
        freeze_tag=config["freeze_tag"], classifiers=config["classifiers"],
        match_fields=config["match_fields"],
    )

    exact = set(config["exact_prediction_classifiers"])
    unknown = exact - set(config["classifiers"])
    if unknown:
        raise ValueError(f"exact_prediction_classifiers not in classifiers: {sorted(unknown)}")
    summaries = summarise(matched, config["classifiers"])
    for s in summaries:
        s["exact_prediction"] = s["classifier"] in exact
        s["name"] = config["classifier_names"][s["classifier"]]

    report = dict(input=config["input"], input_sha256=hashlib.sha256(data).hexdigest(),
                  config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                  exact_prediction_classifiers=sorted(exact),
                  summaries=summaries, matches=matched)
    Path(config["output_json"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = markdown(report, summaries, config)
    Path(config["output_markdown"]).write_text(text, encoding="utf-8")
    Path(config["output_table"]).write_text(latex(summaries), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
