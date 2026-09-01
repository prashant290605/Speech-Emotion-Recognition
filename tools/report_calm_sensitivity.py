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

from ser.phase8 import seed_interval  # noqa: E402

from run_calm_sensitivity import SPEC_PATH, load_spec  # noqa: E402


def expected_count(spec: dict) -> int:
    return len(spec["directions"]) * len(spec["seeds"]) * len(spec["rungs"])


def summary(rows, key: str) -> dict:
    return seed_interval([row[key] for row in rows])


def interval_text(values: dict) -> str:
    return f"{values['mean']:.4f} [{values['lo']:.4f}, {values['hi']:.4f}]"


def title(name: str) -> str:
    return name.replace("_", " ").title()


def display_direction(direction: tuple[str, str]) -> str:
    return f"{direction[0]}->{direction[1]}"


def display_corpus(corpus: str) -> str:
    return {"ravdess": "RAVDESS", "cremad": "CREMA-D"}.get(corpus, corpus.upper())


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
        "replacement full classifier grid.",
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

    lines.extend(["", "## Target-score differences from `none`", ""])
    for direction in directions:
        baseline = summary(groups[(*direction, "none")], "macro_f1")["mean"]
        deltas = []
        for alignment in order:
            if alignment == "none":
                continue
            aligned = summary(groups[(*direction, alignment)], "macro_f1")["mean"]
            deltas.append(f"`{alignment}` minus `none` = {aligned - baseline:+.4f}")
        lines.append(f"- {display_direction(direction)}: " + "; ".join(deltas) + ".")

    report_output = REPO_ROOT / spec["report"]
    report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    headings = ["rung", *[
        f"{display_corpus(source)} $\\rightarrow$ {display_corpus(target)}"
        for source, target in directions
    ]]
    table_lines = [
        "\\begin{table*}[!t]",
        "  \\centering",
        "  \\caption{Target macro-F1 in the pre-specified label-harmonisation "
        f"sensitivity. {caption_description} Values are means over {len(spec['seeds'])} "
        "speaker-disjoint seeds with 95\\% $t$-intervals.}",
        f"  \\label{{tab:{spec['table_label']}}}",
        "  \\small",
        "  \\begin{tabular}{l" + "r" * len(directions) + "}",
        "    \\toprule",
        "    " + " & ".join(headings) + " " + "\\\\",
        "    \\midrule",
    ]
    for alignment in order:
        cells = [f"\\texttt{{{alignment}}}"]
        for direction in directions:
            cells.append(interval_text(summary(groups[(*direction, alignment)], "macro_f1")))
        table_lines.append("    " + " & ".join(cells) + " " + "\\\\")
    chance = [
        summary(groups[(*direction, "none")], "chance_macro_f1")["mean"]
        for direction in directions
    ]
    table_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\\\[2pt]",
        "  \\begin{minipage}{\\linewidth}\\footnotesize "
        f"Filter: cached {spec['backbone']} {spec['layer_agg']}-layer features, "
        f"{spec['classifier']}, and {', '.join(rung['alignment'] for rung in spec['rungs'])}. "
        "The chance baselines are " + " and ".join(f"{value:.4f}" for value in chance) + ". "
        "This robustness control is not a replacement full classifier grid."
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
