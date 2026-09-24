"""The compact sufficient statistic behind every cluster-bootstrap interval.

Per-utterance predictions are read for exactly one purpose: to build
per-speaker confusion matrices. Every published interval is then a weighted sum
of those matrices. So the matrices are sufficient, and they are small enough to
track, which matters because the predictions are 7730 gitignored files.

"Sufficient" is asserted here rather than argued. The tests reconstruct the
statistics both ways -- from stored predictions and from the artifact -- and
require exact agreement, including a full deterministic bootstrap under
identical seeds, because the replicates index speakers positionally and would
diverge if the ordering were not preserved.

What the artifact deliberately does not preserve is which utterance received
which label. That is stated and tested too, so nobody later mistakes it for a
drop-in replacement for the predictions.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from ser import speaker_stats  # noqa: E402
from ser.config import load_config  # noqa: E402
from ser.manifest import load_for_analysis  # noqa: E402
from ser.phase8 import (  # noqa: E402
    confusion_by_group,
    load_predictions,
    macro_f1_from_confusion,
    paired_cluster_bootstrap,
    per_class_f1_from_confusion,
)
from ser.speaker_stats import SPEAKER_CONFUSIONS, build_record, write_records  # noqa: E402
from ser.utils.results import read_rows  # noqa: E402

ARTIFACT = REPO_ROOT / SPEAKER_CONFUSIONS
RESULTS = REPO_ROOT / "results" / "runs.jsonl"
PREDICTIONS = REPO_ROOT / "results" / "predictions"

# Enough runs to exercise both directions and several arms without making the
# suite slow; the generator's --verify mode covers all 5364.
SAMPLE = 40


@pytest.fixture(scope="module")
def compact():
    if not ARTIFACT.exists():
        pytest.skip(f"{SPEAKER_CONFUSIONS} not built")
    return speaker_stats.load(ARTIFACT)


@pytest.fixture(scope="module")
def sample_rows():
    if not RESULTS.exists() or not PREDICTIONS.exists():
        pytest.skip("ledger or predictions absent")
    # One direction only. The two directions have different target-test
    # speaker counts, and the bootstrap indexes speakers positionally within a
    # seed, so pooling them would be a malformed comparison rather than a
    # stricter one.
    rows = [r for r in read_rows(RESULTS)
            if r["status"] == "ok" and r.get("predictions_path")
            and r.get("freeze_tag") == "grid-freeze-v3" and r.get("blending") == "none"
            and r.get("alignment") in {"none", "mean_shift"}
            and r.get("classifier") == "svm_rbf"
            and (r["source_corpus"], r["target_corpus"]) == ("cremad", "ravdess")]
    return rows[:SAMPLE]


@pytest.fixture(scope="module")
def reference(sample_rows):
    """Per-speaker confusions rebuilt from the prediction files."""
    config = load_config()
    manifest = load_for_analysis(config)
    label = {r.utterance_id: r.label_six for r in manifest}
    speaker = {r.utterance_id: r.speaker_id for r in manifest}
    out = {}
    for row in sample_rows:
        classes = list(row["class_names"])
        index = {n: i for i, n in enumerate(classes)}
        ids, predicted = load_predictions(RESULTS, row)
        names = sorted({speaker[u] for u in ids})
        lookup = {n: i for i, n in enumerate(names)}
        out[row["run_id"]] = (names, confusion_by_group(
            [index[label[u]] for u in ids],
            [index[p] for p in predicted],
            [lookup[speaker[u]] for u in ids],
            len(classes), len(names),
        ))
    return out


pytestmark = pytest.mark.ledger


# -- exact equivalence -----------------------------------------------------
def test_per_speaker_tensors_are_identical(compact, reference):
    for run_id, (_names, tensor) in reference.items():
        assert run_id in compact, run_id
        assert np.array_equal(compact.tensor(run_id), tensor), run_id


def test_speaker_ordering_is_preserved(compact, reference):
    """Positional draws make this load-bearing, not cosmetic."""
    for run_id, (names, _tensor) in reference.items():
        assert compact.speakers(run_id) == names, run_id
        assert compact.speakers(run_id) == sorted(compact.speakers(run_id))


def test_pooled_confusion_macro_f1_and_per_class_f1_agree_exactly(compact, reference):
    for run_id, (_names, tensor) in reference.items():
        a, b = tensor.sum(axis=0), compact.tensor(run_id).sum(axis=0)
        assert np.array_equal(a, b)
        assert macro_f1_from_confusion(a) == macro_f1_from_confusion(b)
        fa, fb = per_class_f1_from_confusion(a), per_class_f1_from_confusion(b)
        assert np.array_equal(np.nan_to_num(fa, nan=-1), np.nan_to_num(fb, nan=-1))


def test_stored_row_macro_f1_is_recovered_from_the_artifact(compact, sample_rows):
    """Against the ledger's own recorded score, not only against predictions.

    This is the end-to-end statement: the artifact reproduces the number the
    frozen experiment wrote down, so an interval built on it is an interval
    about the published result.
    """
    for row in sample_rows:
        pooled = compact.tensor(row["run_id"]).sum(axis=0)
        assert macro_f1_from_confusion(pooled) == pytest.approx(row["macro_f1"], abs=1e-12)


def test_a_full_paired_bootstrap_is_reproduced_draw_for_draw(compact, reference, sample_rows):
    """Identical rng seed, identical interval and p-value, not merely similar."""
    by_seed_pred, by_seed_art, groups = {}, {}, {}
    other_pred, other_art = {}, {}
    for row in sample_rows:
        seed, run_id = row["seed"], row["run_id"]
        target = (by_seed_pred, by_seed_art) if row["alignment"] == "mean_shift" \
            else (other_pred, other_art)
        target[0].setdefault(seed, []).append(reference[run_id][1])
        target[1].setdefault(seed, []).append(compact.tensor(run_id))
        groups[seed] = len(reference[run_id][0])
    if not (by_seed_pred and other_pred):
        pytest.skip("sample did not contain both arms")

    one = paired_cluster_bootstrap(by_seed_pred, other_pred, groups, n_boot=300, seed=17)
    two = paired_cluster_bootstrap(by_seed_art, other_art, groups, n_boot=300, seed=17)
    for key in ("diff", "lo", "hi", "p", "n_seeds"):
        assert one[key] == two[key], key


# -- format properties -----------------------------------------------------
def test_counts_are_integers_and_sum_to_the_test_set_size(compact, sample_rows):
    for row in sample_rows:
        tensor = compact.tensor(row["run_id"])
        assert tensor.dtype == np.int64
        assert int(tensor.sum()) == row["n_target_test"]


def test_written_artifact_is_deterministic(tmp_path):
    """Sorted, and gzipped with a fixed mtime, so the digest is content-only."""
    records = [
        build_record("b", ["x", "y"], ["s1"], np.array([[[1, 0], [0, 2]]])),
        build_record("a", ["x", "y"], ["s2"], np.array([[[3, 1], [1, 0]]])),
    ]
    first, digest_one = write_records(records, tmp_path / "one.gz")
    second, digest_two = write_records(list(reversed(records)), tmp_path / "two.gz")
    assert first == second == 2
    assert digest_one == digest_two
    assert (tmp_path / "one.gz").read_bytes() == (tmp_path / "two.gz").read_bytes()
    with gzip.open(tmp_path / "one.gz", "rt", encoding="utf-8") as handle:
        order = [json.loads(line)["run_id"] for line in handle if line.strip()]
    assert order == ["a", "b"]


def test_non_integral_counts_are_refused():
    with pytest.raises(ValueError, match="integral"):
        build_record("a", ["x", "y"], ["s"], np.array([[[0.5, 0.0], [0.0, 1.0]]]))


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="does not match"):
        build_record("a", ["x", "y", "z"], ["s"], np.array([[[1, 0], [0, 1]]]))


def test_missing_artifact_names_the_tool_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_speaker_confusions"):
        speaker_stats.load(tmp_path / "absent.jsonl.gz")


def test_per_utterance_identity_is_not_claimed(compact, sample_rows):
    """Documented loss, asserted so it cannot be forgotten.

    Anything needing which clip got which label must read results/predictions/.
    """
    record_keys = set(compact._records[sample_rows[0]["run_id"]])
    assert "utterance_ids" not in record_keys
    assert "predicted" not in record_keys
