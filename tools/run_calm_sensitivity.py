#!/usr/bin/env python
"""Run the pre-specified RAVDESS calm-drop sensitivity from cached features."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.features.load import FeatureLoader  # noqa: E402
from ser.labels import LabelPolicy, map_label  # noqa: E402
from ser.manifest import read_manifest  # noqa: E402
from ser.run_grid import GridRun, execute_run  # noqa: E402
from ser.splits import make_pair_split  # noqa: E402
from ser.config import load_config  # noqa: E402
from ser.utils.results import append_row, completed_run_ids, make_run_id  # noqa: E402


def load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    required = {"version", "output", "variant", "backbone", "classifier", "layer_agg", "seeds", "directions", "rungs"}
    missing = required - set(spec or {})
    if missing:
        raise ValueError(f"sensitivity config missing {sorted(missing)}")
    if spec["variant"].get("ravdess_calm_to_neutral") is not False:
        raise ValueError("this runner is only for the calm-dropped control")
    return spec


def sensitivity_config(base, spec):
    """Clone only semantic config fields, making a distinct label-map hash."""
    raw = copy.deepcopy(base.raw)
    raw["labels"]["ravdess_calm_to_neutral"] = False
    raw["labels"]["label_map_version"] = spec["version"]
    raw["project"]["results_path"] = spec["output"]
    labels = replace(
        base.labels,
        ravdess_calm_to_neutral=False,
        label_map_version=spec["version"],
    )
    project = replace(base.project, results_path=spec["output"])
    return replace(base, project=project, labels=labels, raw=raw, source_path=str(SPEC_PATH))


def project_rows(rows, config):
    policy = LabelPolicy.from_config(config)
    projected = []
    for row in rows:
        label = map_label(row.corpus, row.original_label, "six", policy) or ""
        projected.append(replace(row, label_six=label))
    return projected


class SensitivityContext:
    def __init__(self, config, rows):
        self.config = config
        self.rows = rows
        self.by_id = {row.utterance_id: row for row in rows}
        self._loaders = {}
        self._splits = {}

    def loader(self, corpus, backbone):
        key = (corpus, backbone)
        if key not in self._loaders:
            self._loaders[key] = FeatureLoader(self.config, corpus, backbone, self.rows)
        return self._loaders[key]

    def split(self, source, target, seed):
        key = (source, target, seed)
        if key not in self._splits:
            self._splits[key] = make_pair_split(self.rows, self.config, source, target, seed, "six")
        return self._splits[key]

    def labels(self, pair, role):
        return [self.by_id[uid].label_six for uid in pair.splits()[role].utterance_ids]


def enumerate_runs(config, spec):
    runs = []
    for source, target in spec["directions"]:
        for seed in spec["seeds"]:
            for rung in spec["rungs"]:
                runs.append(GridRun(
                    source=source, target=target, seed=int(seed),
                    backbone=spec["backbone"], feature_branch="ssl",
                    layer_agg=spec["layer_agg"], layer_index=None,
                    alignment=rung["alignment"], alignment_eps=rung.get("eps"),
                    alignment_lam=rung.get("lam"), blending="none",
                    blend_alpha=None, n_groups=None, classifier=spec["classifier"],
                ))
    ids = [make_run_id(run.coords(config)) for run in runs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("calm sensitivity produced colliding run ids")
    return runs


SPEC_PATH = REPO_ROOT / "configs/calm_sensitivity.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SPEC_PATH))
    args = parser.parse_args(argv)
    spec = load_spec(Path(args.config))
    config = sensitivity_config(load_config(), spec)
    rows = project_rows(read_manifest(config.resolve(config.paths.manifest)), config)
    context = SensitivityContext(config, rows)
    output = config.results_path
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_run_ids(output)
    runs = enumerate_runs(config, spec)
    todo = [run for run in runs if make_run_id(run.coords(config)) not in completed]
    print(f"calm sensitivity: {len(runs)} planned, {len(todo)} remaining", flush=True)
    started = time.perf_counter()
    for index, run in enumerate(todo, start=1):
        row = execute_run(run, context, freeze_tag=spec["version"])
        append_row(output, row)
        elapsed = time.perf_counter() - started
        print(f"[{index:>2}/{len(todo)}] {run.source[:4]}->{run.target[:4]} s{run.seed} "
              f"{run.alignment:<10} {row['status']:<6} {row['macro_f1']} "
              f"({elapsed / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
