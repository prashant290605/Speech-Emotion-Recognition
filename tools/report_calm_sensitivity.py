#!/usr/bin/env python
"""Render a label-harmonisation sensitivity report from its immutable YAML spec."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.config import load_config  # noqa: E402
from ser.manifest import load_for_analysis  # noqa: E402
from ser.phase8 import (  # noqa: E402
    confusion_by_group,
    load_predictions,
    paired_cluster_bootstrap,
    seed_interval,
)

from run_calm_sensitivity import (  # noqa: E402
    SPEC_PATH,
    load_spec,
    project_rows,
    sensitivity_config,
    variant_label_space,
)


N_BOOT = 2000
BOOTSTRAP_SEED = 17
FAMILYWISE_ALPHA = 0.05
ROBUSTNESS_CONTRASTS = 12


def expected_count(spec: dict) -> int:
    return len(spec["directions"]) * len(spec["seeds"]) * len(spec["rungs"])


def summary(rows, key: str) -> dict:
    return seed_interval([row[key] for row in rows])


def interval_text(values: dict) -> str:
    return f"{values['mean']:.4f} [{values['lo']:.4f}, {values['hi']:.4f}]"


def one_sided_family_alpha(family_size: int) -> float:
    """Return a two-sided alpha whose lower endpoint has FWER 1-alpha."""
    if family_size <= 0:
        raise ValueError("family_size must be positive")
    return 2 * FAMILYWISE_ALPHA / family_size


def title(name: str) -> str:
    return name.replace("_", " ").title()


def display_direction(direction: tuple[str, str]) -> str:
    return f"{direction[0]}->{direction[1]}"


def display_corpus(corpus: str) -> str:
    return {"ravdess": "RAVDESS", "cremad": "CREMA-D"}.get(corpus, corpus.upper())


class SensitivityPredictionData:
    """Reconstruct paired target-test confusions for one sensitivity variant."""

    def __init__(self, spec: dict, results_path: Path):
        base = load_config()
        self.label_space = variant_label_space(spec)
        config = sensitivity_config(base, spec)
        rows = project_rows(
            load_for_analysis(config),
            config,
            self.label_space,
        )
        self.by_id = {row.utterance_id: row for row in rows}
        self.results_path = results_path
        self._confusions: dict[str, object] = {}
        self._speaker_groups: dict[tuple[str, str, int], dict[str, int]] = {}

    def _label(self, utterance_id: str) -> str:
        row = self.by_id[utterance_id]
        return row.label_six if self.label_space == "six" else row.label_four

    def confusion(self, row: dict):
        run_id = row["run_id"]
        if run_id in self._confusions:
            return self._confusions[run_id]
        utterance_ids, predicted = load_predictions(self.results_path, row)
        classes = list(row["class_names"])
        index = {label: position for position, label in enumerate(classes)}
        labels = [self._label(utterance_id) for utterance_id in utterance_ids]
        if any(label not in index for label in labels):
            raise RuntimeError(f"{run_id}: target labels drifted from the stored class list")
        if any(label not in index for label in predicted):
            raise RuntimeError(f"{run_id}: predictions drifted from the stored class list")

        speakers = sorted({self.by_id[utterance_id].speaker_id for utterance_id in utterance_ids})
        key = (row["source_corpus"], row["target_corpus"], int(row["seed"]))
        groups = {speaker: position for position, speaker in enumerate(speakers)}
        previous = self._speaker_groups.setdefault(key, groups)
        if previous != groups:
            raise RuntimeError(f"{run_id}: target-test speaker groups do not match within a seed")
        confusion = confusion_by_group(
            [index[label] for label in labels],
            [index[label] for label in predicted],
            [groups[self.by_id[utterance_id].speaker_id] for utterance_id in utterance_ids],
            len(classes),
            len(groups),
        )
        self._confusions[run_id] = confusion
        return confusion

    def n_speakers(self, source: str, target: str, seed: int) -> int:
        return len(self._speaker_groups[(source, target, seed)])


def paired_differences(
    groups: dict, directions: tuple[tuple[str, str], ...], order: tuple[str, ...],
    data: SensitivityPredictionData, *, n_boot: int = N_BOOT,
    bootstrap_seed: int = BOOTSTRAP_SEED, alpha: float = 0.05,
) -> dict[tuple[str, str], dict[str, dict]]:
    """Return aligned-minus-none target macro-F1 intervals for each control."""
    output = {}
    for direction in directions:
        baseline = {int(row["seed"]): row for row in groups[(*direction, "none")]}
        if len(baseline) != len(groups[(*direction, "none")]):
            raise RuntimeError(f"{direction}: duplicate baseline rows")
        output[direction] = {"none": None}
        for alignment in order:
            if alignment == "none":
                continue
            aligned = {int(row["seed"]): row for row in groups[(*direction, alignment)]}
            if set(aligned) != set(baseline):
                raise RuntimeError(f"{direction}, {alignment}: unpaired seed sets")
            aligned_arm = {
                seed: [data.confusion(aligned[seed])]
                for seed in sorted(aligned)
            }
            baseline_arm = {
                seed: [data.confusion(baseline[seed])]
                for seed in sorted(baseline)
            }
            n_groups = {
                seed: data.n_speakers(direction[0], direction[1], seed)
                for seed in sorted(baseline)
            }
            output[direction][alignment] = paired_cluster_bootstrap(
                aligned_arm,
                baseline_arm,
                n_groups,
                n_boot=n_boot,
                seed=bootstrap_seed,
                alpha=alpha,
            )
    return output


def difference_text(values: dict | None) -> str:
    if values is None:
        return "--"
    return f"{values['diff']:+.4f} [{values['lo']:+.4f}, {values['hi']:+.4f}]"


def lower_bound_text(values: dict | None) -> str:
    if values is None:
        return "--"
    return f"{values['lo']:+.4f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SPEC_PATH))
    args = parser.parse_args(argv)
    spec = load_spec(Path(args.config))
    path = REPO_ROOT / spec["output"]
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = expected_count(spec)
    if len(rows) != expected or any(row["status"] != "ok" for row in rows):
        raise RuntimeError(f"sensitivity incomplete: {len(rows)}/{expected} rows")

    order = tuple(rung["alignment"] for rung in spec["rungs"])
    if len(order) != len(set(order)) or "none" not in order:
        raise ValueError("sensitivity rungs must be unique and include none")
    directions = tuple(tuple(direction) for direction in spec["directions"])
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_corpus"], row["target_corpus"], row["alignment"])].append(row)
    prediction_data = SensitivityPredictionData(spec, path)
    differences = paired_differences(groups, directions, order, prediction_data)
    familywise = paired_differences(
        groups, directions, order, prediction_data,
        alpha=one_sided_family_alpha(ROBUSTNESS_CONTRASTS),
    )

    variant = spec["variant"]
    description = variant["description"]
    caption_description = description[:1].upper() + description[1:]
    lines = [
        f"# {title(variant['name'])} sensitivity",
        "",
        f"This pre-specified sensitivity {description} It reuses cached "
        f"{spec['backbone']} {spec['layer_agg']}-layer features, runs "
        f"{spec['classifier']}, and scores {len(spec['seeds'])} speaker-disjoint "
        "seeds in each transfer direction. It is a robustness control, not a "
        "replacement for the full classifier grid.",
        "",
        "| direction | alignment | target macro-F1 | source-val macro-F1 | chance baseline | n train | n target test |",
        "|---|---|---|---|---|---|---|",
    ]
    for direction in directions:
        for alignment in order:
            group = groups[(*direction, alignment)]
            target = summary(group, "macro_f1")
            source = summary(group, "selection_source_val_macro_f1")
            chance = summary(group, "chance_macro_f1")
            lines.append(
                f"| {display_direction(direction)} | {alignment} | {interval_text(target)} | "
                f"{interval_text(source)} | {chance['mean']:.4f} | "
                f"{int(group[0]['n_train'])} | {int(group[0]['n_target_test'])} |"
            )

    lines.extend([
        "",
        "## Paired target-score differences from `none`",
        "",
        f"Each interval is a paired cluster bootstrap over target-test speakers and seeds, {N_BOOT} replicates.",
        f"Each lower bound is one-sided Bonferroni-simultaneous at 95% familywise coverage over all {ROBUSTNESS_CONTRASTS} contrasts in Tables 7 and 8.",
        "",
    ])
    for direction in directions:
        deltas = []
        for alignment in order:
            if alignment == "none":
                continue
            deltas.append(
                f"`{alignment}` minus `none` = {difference_text(differences[direction][alignment])}; "
                f"global lower bound = {lower_bound_text(familywise[direction][alignment])}"
            )
        lines.append(f"- {display_direction(direction)}: " + "; ".join(deltas) + ".")

    report_output = REPO_ROOT / spec["report"]
    report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    score_headings = ["rung", *[
        f"{display_corpus(source)} $\\rightarrow$ {display_corpus(target)}"
        for source, target in directions
    ]]
    table_lines = [
        "\\begin{table*}[!t]",
        "  \\centering",
        "  \\caption{Target macro-F1 in the pre-specified label-harmonisation "
       f"sensitivity. {caption_description} Values are means over {len(spec['seeds'])} "
       "speaker-disjoint seeds with 95\\% $t$-intervals. Each $\\Delta$ is aligned "
        "minus \\texttt{none}, with a paired 95\\% cluster-bootstrap interval and a "
        "one-sided Bonferroni-simultaneous lower bound at 95\\% familywise coverage "
        "over all 12 Table~7--8 contrasts.}",
        f"  \\label{{tab:{spec['table_label']}}}",
        "  \\small",
        "  Target macro-F1 [95\\% $t$-interval]\\\\[2pt]",
        "  \\begin{tabular}{l" + "r" * len(directions) + "}",
        "    \\toprule",
        "    " + " & ".join(score_headings) + " " + "\\\\",
        "    \\midrule",
    ]
    for alignment in order:
        cells = [f"\\texttt{{{alignment.replace('_', r'\_')}}}"]
        for direction in directions:
            cells.append(interval_text(summary(groups[(*direction, alignment)], "macro_f1")))
        table_lines.append("    " + " & ".join(cells) + " " + "\\\\")
    table_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\\\[5pt]",
        "  Paired target-macro-F1 change from \\texttt{none} [95\\% cluster-bootstrap interval]\\\\[2pt]",
        "  \\begin{tabular}{llrr}",
        "    \\toprule",
        "    rung & direction & $\\Delta$ target macro-F1 & global 95\\% lower bound \\\\",
        "    \\midrule",
    ])
    for alignment in order:
        if alignment == "none":
            continue
        for source, target in directions:
            direction = (source, target)
            table_lines.append(
                "    " + " & ".join([
                   f"\\texttt{{{alignment.replace('_', r'\_')}}}",
                   f"{display_corpus(source)} $\\rightarrow$ {display_corpus(target)}",
                   difference_text(differences[direction][alignment]),
                    lower_bound_text(familywise[direction][alignment]),
                ]) + " " + "\\\\"
            )
    chance = [
        summary(groups[(*direction, "none")], "chance_macro_f1")["mean"]
        for direction in directions
    ]
    rung_names = ", ".join(
        rung["alignment"].replace("_", r"\_") for rung in spec["rungs"]
    )
    table_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\\\[2pt]",
        "  \\begin{minipage}{\\linewidth}\\footnotesize "
        f"Filter: cached {spec['backbone']} {spec['layer_agg']}-layer features, "
        f"{spec['classifier']}, and {rung_names}. "
       "The chance baselines are " + " and ".join(f"{value:.4f}" for value in chance) + ". "
       f"Paired differences resample target-test speakers and seeds ({N_BOOT} replicates). "
        "The lower bounds use a single 12-contrast, one-sided Bonferroni family. "
        "This robustness control is not a replacement for the full classifier grid."
        "\\end{minipage}",
        "\\end{table*}",
    ])
    table_output = REPO_ROOT / spec["table"]
    table_output.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"wrote {report_output.relative_to(REPO_ROOT)}")
    print(f"wrote {table_output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
