"""Generate a retrospective audit from existing runs; never train or edit them."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/audit_translation.yaml"))
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    data = Path(config["input"]).read_bytes()
    matched = match_translation_rows(
        [json.loads(line) for line in data.splitlines() if line.strip()],
        freeze_tag=config["freeze_tag"], classifiers=config["classifiers"],
        match_fields=config["match_fields"],
    )
    groups = defaultdict(list)
    for row in matched:
        groups[(row["source_corpus"], row["target_corpus"], row["classifier"])].append(row)
    summaries = []
    for (source, target, classifier), rows in sorted(groups.items()):
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
        ))
    report = dict(input=config["input"], input_sha256=hashlib.sha256(data).hexdigest(),
                  config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                  summaries=summaries, matches=matched)
    Path(config["output_json"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Source-translation audit", "", "Retrospective analysis of frozen runs. No training.",
             "Stored-score equality is exact, not rounded. The full trial surface is not stored.",
             "Rows share seeds and are not independent experimental replications.", "",
             f"Input SHA256: `{report['input_sha256']}`", f"Config SHA256: `{report['config_sha256']}`", "",
             "| Direction | Classifier | Pairs | Equal validation | Equal hyperparameters | Both equal | Positive target differences | Mean target difference | Chance |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{table*}[!t]", r"\centering\small",
           r"\caption{Retrospective translation audit on frozen runs. Each row contains matched cells across three backbones, two fixed layer settings and five seeds. Equal validation and equal hyperparameters count exact equality of the stored winning score and selected classifier parameters. $\Delta$ is mean target macro-F1 for mean shift minus none; chance is the uniform-random baseline. Cells share seeds; no independence-based test is reported.}",
           r"\label{tab:translation-audit}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Direction & Head & Pairs & Equal val. & Equal params & $\Delta$ & Positive & Chance \\", r"\midrule"]
    for s in summaries:
        direction = f"{s['source']} -> {s['target']}"
        lines.append(f"| {direction} | {s['classifier']} | {s['n']} | {s['validation_equal']} | {s['hyperparameters_equal']} | {s['both_equal']} | {s['positive_target_differences']} | {s['mean_target_difference']:+.4f} | {s['chance']:.4f} |")
        name = "RBF SVM" if s["classifier"] == "svm_rbf" else "Logistic"
        direction_tex = r"RAVDESS $\to$ CREMA-D" if s["source"] == "ravdess" else r"CREMA-D $\to$ RAVDESS"
        tex.append(f"{direction_tex} & {name} & {s['n']} & {s['validation_equal']} & {s['hyperparameters_equal']} & {s['mean_target_difference']:+.4f} & {s['positive_target_differences']} & {s['chance']:.4f} " + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    Path(config["output_markdown"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(config["output_table"]).write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
