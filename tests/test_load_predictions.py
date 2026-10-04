"""Per-utterance prediction paths must resolve on every platform.

The grid ran on Windows, so ``predictions_path`` in the committed ledgers uses
backslashes. Read naively on Linux or macOS, ``predictions\\<id>.json.gz`` is a
single file name, and every analysis that needs utterance-level predictions
(the label-harmonisation controls among them) fails with FileNotFoundError
even when the files are present.
"""

from __future__ import annotations

import gzip
import json

import pytest

from ser.phase8 import load_predictions


@pytest.mark.parametrize("stored", [
    "predictions\\run1.json.gz",   # written on Windows, as in the ledgers
    "predictions/run1.json.gz",    # written on POSIX
])
def test_load_predictions_accepts_either_separator(tmp_path, stored):
    (tmp_path / "predictions").mkdir()
    payload = {"utterance_ids": ["u1", "u2"], "predicted": ["angry", "sad"]}
    with gzip.open(tmp_path / "predictions" / "run1.json.gz", "wt",
                   encoding="utf-8") as handle:
        json.dump(payload, handle)

    ids, predicted = load_predictions(tmp_path / "runs.jsonl",
                                      {"predictions_path": stored})

    assert ids == ["u1", "u2"]
    assert predicted == ["angry", "sad"]


def test_load_predictions_still_fails_loudly_when_the_file_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path / "runs.jsonl",
                         {"predictions_path": "predictions\\missing.json.gz"})
