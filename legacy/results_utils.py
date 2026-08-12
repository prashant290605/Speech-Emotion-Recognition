from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESULTS_DIR = Path("results")
RESULTS_JSON_PATH = RESULTS_DIR / "results.json"


def save_result(filename: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def append_experiment_record(record: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_JSON_PATH.exists():
        try:
            with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "experiments" not in payload:
                payload = {"experiments": []}
        except Exception:
            payload = {"experiments": []}
    else:
        payload = {"experiments": []}

    payload["experiments"].append(record)

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return RESULTS_JSON_PATH
