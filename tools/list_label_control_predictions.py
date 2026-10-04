#!/usr/bin/env python
"""List the per-utterance prediction files the label-harmonisation controls need.

    python tools/list_label_control_predictions.py           # print the paths
    python tools/list_label_control_predictions.py --check   # also report missing files

The paired intervals of the two label-harmonisation controls (supplementary
Section S1, ``tools/report_calm_sensitivity.py``) are rebuilt from per-utterance
predictions, not from the committed per-speaker confusions, which cover only
the main ledger. ``results/predictions/`` is gitignored, so a clean clone cannot
regenerate those intervals. Committing the files this script lists closes that
gap; the manuscript's Data availability statement and the README then no
longer need their one exception.

On the machine that holds ``results/predictions/``:

    python tools/list_label_control_predictions.py --check
    python tools/list_label_control_predictions.py | xargs git add -f            # bash
    python tools/list_label_control_predictions.py | ForEach-Object { git add -f $_ }  # PowerShell

Paths are printed relative to the repository root with forward slashes, which
git accepts on every platform. The ledgers are read from the ``output`` field of
the two control specifications, so the list cannot drift from them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PureWindowsPath

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS = ("configs/calm_sensitivity.yaml", "configs/neutral_excluded_sensitivity.yaml")


def prediction_files() -> list[str]:
    files = []
    for spec_path in SPECS:
        spec = yaml.safe_load((REPO_ROOT / spec_path).read_text(encoding="utf-8"))
        ledger = Path(spec["output"])
        for line in (REPO_ROOT / ledger).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok" or not row.get("predictions_path"):
                raise SystemExit(f"{ledger}: run {row.get('run_id')} has no stored predictions")
            # Stored relative to the ledger, with the writer's separator.
            relative = ledger.parent.joinpath(*PureWindowsPath(row["predictions_path"]).parts)
            files.append(relative.as_posix())
    if len(set(files)) != len(files):
        raise SystemExit("duplicate prediction paths across the control ledgers")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="report how many of the files exist locally")
    args = parser.parse_args()
    files = prediction_files()
    if args.check:
        missing = [f for f in files if not (REPO_ROOT / f).is_file()]
        print(f"{len(files)} prediction files needed, {len(files) - len(missing)} present, "
              f"{len(missing)} missing", file=sys.stderr)
        for f in missing:
            print(f"  missing {f}", file=sys.stderr)
        return 1 if missing else 0
    try:
        for f in files:
            print(f)
    except BrokenPipeError:  # a consumer such as `head` stopped reading early
        sys.stderr.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
