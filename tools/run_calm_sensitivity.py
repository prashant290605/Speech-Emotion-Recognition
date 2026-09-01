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
    variant = spec["variant"]
    if variant.get("ravdess_calm_to_neutral") is not False:
        raise ValueError("label-sensitivity controls must drop RAVDESS calm")
    label_space = variant.get("label_space", "six")
    if not isinstance(label_space, str) or not label_space:
        raise ValueError("variant.label_space must be a non-empty string")
    classes = variant.get("classes")
    if classes is not None and (not isinstance(classes, list) or classes != sorted(set(classes))):
        raise ValueError("variant.classes must be a sorted list of unique labels")
    if label_space != "six" and not classes:
        raise ValueError("a custom variant.label_space requires variant.classes")
    return spec


def variant_label_space(spec: dict) -> str:
    return spec["variant"].get("label_space", "six")


def sensitivity_config(base, spec, *, source_path: str | None = None):
    """Clone only semantic config fields, making a distinct label-map hash."""
    raw = copy.deepcopy(base.raw)
    label_space = variant_label_space(spec)
    spaces = copy.deepcopy(base.labels.spaces)
    if spec["variant"].get("classes") is not None:
        spaces[label_space] = list(spec["variant"]["classes"])
    raw["labels"]["ravdess_calm_to_neutral"] = False
    raw["labels"]["label_map_version"] = spec["version"]
    raw["labels"]["spaces"] = copy.deepcopy(spaces)
    raw["labels"]["space_for_other_pairs"] = label_space
    raw["project"]["results_path"] = spec["output"]
    labels = replace(
        base.labels,
        ravdess_calm_to_neutral=False,
        label_map_version=spec["version"],
        spaces=spaces,
        space_for_other_pairs=label_space,
    )
    project = replace(base.project, results_path=spec["output"])
    return replace(
        base,
        project=project,
        labels=labels,
        raw=raw,
        source_path=source_path or str(SPEC_PATH),
    )


def project_rows(rows, config, label_space: str = "six"):
    """Project one requested label space into the manifest field splits use."""
    policy = LabelPolicy.from_config(config)
    projected = []
    for row in rows:
        label = map_label(row.corpus, row.original_label, label_space, policy) or ""
        field = "label_six" if label_space == "six" else "label_four"
        projected.append(replace(row, **{field: label}))
    return projected


class SensitivityContext:
    def __init__(self, config, rows, label_space: str):
        self.config = config
        self.rows = rows
        self.label_space = label_space
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
            self._splits[key] = make_pair_split(
                self.rows, self.config, source, target, seed, self.label_space
            )
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
    label_space = variant_label_space(spec)
    config = sensitivity_config(load_config(), spec, source_path=str(Path(args.config)))
    rows = project_rows(
        read_manifest(config.resolve(config.paths.manifest)), config, label_space
    )
    context = SensitivityContext(config, rows, label_space)
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
