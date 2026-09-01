#!/usr/bin/env python
"""Render an adaptive-geometry label-sensitivity diagnostic report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ser.phase8 import seed_interval  # noqa: E402

from report_calm_sensitivity import expected_count, interval_text  # noqa: E402
from run_calm_sensitivity import load_spec  # noqa: E402


def mean_conditional(record: dict, key: str) -> float | None:
    values = [row[key] for row in record["conditional"] if row[key] is not None]
    return float(np.mean(values)) if values else None


def summary(rows: list[dict], key: str) -> dict:
    return seed_interval([row[key] for row in rows if row[key] is not None])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    spec = load_spec(Path(args.config))
    diagnostic_spec = spec["diagnostics"]
    path = REPO_ROOT / diagnostic_spec["output"]
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = expected_count(spec)
    if len(records) != expected:
        raise RuntimeError(f"diagnostics incomplete: {len(records)}/{expected} records")
    groups = defaultdict(list)
    for record in records:
        groups[(record["source"], record["target"], record["alignment"])].append(record)
    directions = tuple(tuple(direction) for direction in spec["directions"])
    rungs = tuple(rung["alignment"] for rung in spec["rungs"])

    lines = [
        f"# {spec['variant']['name'].replace('_', ' ').title()} diagnostics",
        "",
        f"This report uses the {diagnostic_spec['bandwidth_rule']} rule separately "
        "for every rung and seed. The reported raw and null-scaled values are "
        "therefore descriptive adaptive-geometry diagnostics. They are not "
        "comparable under one fixed RBF scale, and no conditional-to-marginal "
        "ratio is reported.",
        "",
        "| direction | rung | raw marginal MMD2 | marginal MMD2 / null | mean raw conditional MMD2 | mean conditional / class null |",
        "|---|---|---|---|---|---|",
    ]
    for direction in directions:
        for rung in rungs:
            rows = groups[(*direction, rung)]
            marginal_raw = summary(rows, "marginal_raw_mmd2")
            marginal_effect = summary(rows, "marginal_normalised")
            conditional_raw = seed_interval([
                mean_conditional(row, "raw_mmd") for row in rows
                if mean_conditional(row, "raw_mmd") is not None
            ])
            conditional_effect = seed_interval([
                mean_conditional(row, "effect_size") for row in rows
                if mean_conditional(row, "effect_size") is not None
            ])
            lines.append(
                f"| {direction[0]}->{direction[1]} | {rung} | "
                f"{interval_text(marginal_raw)} | {interval_text(marginal_effect)} | "
                f"{interval_text(conditional_raw)} | {interval_text(conditional_effect)} |"
            )
    output = REPO_ROOT / diagnostic_spec["report"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
