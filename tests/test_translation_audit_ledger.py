"""Pin the retrospective translation audit to the actual frozen ledger.

``test_affine_audit.py`` exercises ``match_translation_rows`` on synthetic rows:
it proves the matcher fails closed and compares without rounding. It cannot
prove that the numbers the manuscript reports are the numbers the matcher
currently produces from ``results/runs.jsonl``. Nothing did, until this module.

What is pinned here is the science and the provenance, not the presentation:

* the identity of the ledger, by digest,
* the exact-equality result for the RBF SVM, which is the manuscript's
  empirical centrepiece and must never drift silently,
* the completeness of the join -- every eligible row is represented,
* the *irrelevance of target performance* to matching and inclusion, asserted
  behaviourally rather than by reading the source.

Cell counts per configured family are pinned too, because a change in them is
either a deliberate configuration decision (which should update this file in
the same commit) or a regression. Descriptive quantities that would make the
test brittle without protecting a claim -- mean target differences, chance
baselines, per-family hyperparameter-agreement counts -- are deliberately not
pinned; they live in the generated report.

Read-only throughout. No test here writes to the ledger or the configuration.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from ser.analysis.affine_audit import match_translation_rows
from ser.freeze import assert_ledger_unchanged, expected_ledger_digest, ledger_digest

pytestmark = pytest.mark.ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "results" / "runs.jsonl"
CONFIG = REPO_ROOT / "configs" / "audit_translation.yaml"

# The manuscript's central empirical claim, stated as data. Both directions,
# all cells, exact equality of the stored winning validation score and of the
# selected hyperparameters.
RBF_EXACT_EQUALITY = {
    ("ravdess", "cremad"): 30,
    ("cremad", "ravdess"): 30,
}

# Matched pairs per (classifier, direction) under the committed configuration.
# The MLP has more cells because it is the only family here that can use the
# learned `weighted` layer aggregation.
EXPECTED_PAIRS = {
    ("svm_rbf", "ravdess", "cremad"): 30, ("svm_rbf", "cremad", "ravdess"): 30,
    ("logreg", "ravdess", "cremad"): 30, ("logreg", "cremad", "ravdess"): 30,
    ("svm_linear", "ravdess", "cremad"): 30, ("svm_linear", "cremad", "ravdess"): 30,
    ("mlp", "ravdess", "cremad"): 45, ("mlp", "cremad", "ravdess"): 45,
}


@pytest.fixture(scope="module")
def config():
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} not present")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows():
    if not LEDGER.exists():
        pytest.skip(f"{LEDGER} not present; nothing to audit")
    with open(LEDGER, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def matched(rows, config):
    return match_translation_rows(
        rows, freeze_tag=config["freeze_tag"], classifiers=config["classifiers"],
        match_fields=config["match_fields"],
    )


# -- provenance ------------------------------------------------------------
def test_frozen_ledger_matches_its_recorded_digest():
    """The audit's conclusions are about one specific file. This is that file."""
    if not LEDGER.exists():
        pytest.skip("ledger not present")
    assert assert_ledger_unchanged(LEDGER) == expected_ledger_digest()


def test_recorded_digest_is_the_one_the_published_report_quotes():
    """The report embeds `input_sha256`; it must be the digest under guard.

    Two copies of a hash that can disagree is the failure mode the single
    marker file exists to prevent, so the one place they are allowed to meet
    is here, where a disagreement fails a test.
    """
    report = REPO_ROOT / "reports" / "translation_audit.json"
    if not report.exists() or not LEDGER.exists():
        pytest.skip("generated report or ledger not present")
    published = json.loads(report.read_text(encoding="utf-8"))["input_sha256"]
    assert published == expected_ledger_digest() == ledger_digest(LEDGER)


# -- the central result ----------------------------------------------------
def test_rbf_svm_has_exact_validation_equality_in_every_matched_cell(matched):
    """Proposition 1's empirical consequence, per direction, with no rounding."""
    counts = {}
    for row in matched:
        if row["classifier"] != "svm_rbf":
            continue
        key = (row["source_corpus"], row["target_corpus"])
        counts.setdefault(key, {"n": 0, "validation": 0, "hyperparameters": 0})
        counts[key]["n"] += 1
        counts[key]["validation"] += bool(row["validation_equal"])
        counts[key]["hyperparameters"] += bool(row["hyperparameters_equal"])

    assert set(counts) == set(RBF_EXACT_EQUALITY)
    for direction, expected in RBF_EXACT_EQUALITY.items():
        observed = counts[direction]
        assert observed["n"] == expected, direction
        assert observed["validation"] == expected, f"{direction}: validation equality"
        assert observed["hyperparameters"] == expected, f"{direction}: parameter equality"


def test_rbf_equality_is_of_the_stored_values_not_a_rounded_comparison(matched):
    """The scores compared are identical floats, not merely close ones."""
    rbf = [r for r in matched if r["classifier"] == "svm_rbf"]
    assert rbf
    for row in rbf:
        assert row["none_validation"] == row["mean_shift_validation"]


def test_configured_families_outside_the_assumptions_are_present_and_reported(matched, config):
    """Scope evidence is retained, not filtered out when it is inconvenient.

    The audit exists to report every eligible match. A family disappearing from
    the output because its numbers were untidy is precisely the failure this
    asserts against.
    """
    exact = set(config["exact_prediction_classifiers"])
    boundary = [c for c in config["classifiers"] if c not in exact]
    assert boundary, "configuration declares no boundary families"
    present = {r["classifier"] for r in matched}
    assert set(config["classifiers"]) <= present


# -- completeness of the join ---------------------------------------------
def test_matched_pair_counts_are_stable(matched):
    counts = {}
    for row in matched:
        key = (row["classifier"], row["source_corpus"], row["target_corpus"])
        counts[key] = counts.get(key, 0) + 1
    assert counts == EXPECTED_PAIRS


def test_every_eligible_ledger_row_is_represented_exactly_once(rows, config, matched):
    """Derived from the ledger rather than hardcoded, so it cross-checks the
    pinned counts instead of restating them.

    Each matched pair consumes exactly two eligible rows, and the audit refuses
    duplicates, so the eligible population must be twice the pair count and the
    collected run identifiers must be distinct.
    """
    eligible = [
        r for r in rows
        if r.get("freeze_tag") == config["freeze_tag"]
        and r.get("blending") == "none"
        and r.get("classifier") in set(config["classifiers"])
        and r.get("alignment") in {"none", "mean_shift"}
    ]
    assert len(eligible) == 2 * len(matched)

    used = [r["none_run_id"] for r in matched] + [r["mean_shift_run_id"] for r in matched]
    assert len(set(used)) == len(used)
    assert set(used) == {r["run_id"] for r in eligible}


# -- target performance is not an input to the join ------------------------
def test_target_macro_f1_does_not_participate_in_matching_or_inclusion(rows, config, matched):
    """Behavioural proof, not a reading of the source.

    Every target score in the ledger copy is overwritten with one constant, the
    audit is re-run, and the pairing is required to be identical: the same run
    identifiers paired the same way, with the same validation and parameter
    equality verdicts. If any target quantity reached the join or the inclusion
    rule, this would not hold.
    """
    perturbed = copy.deepcopy(rows)
    for row in perturbed:
        if row.get("macro_f1") is not None:
            row["macro_f1"] = 0.5

    again = match_translation_rows(
        perturbed, freeze_tag=config["freeze_tag"], classifiers=config["classifiers"],
        match_fields=config["match_fields"],
    )
    assert len(again) == len(matched)

    def signature(entries):
        return sorted(
            (e["none_run_id"], e["mean_shift_run_id"],
             e["validation_equal"], e["hyperparameters_equal"])
            for e in entries
        )

    assert signature(again) == signature(matched)
    assert all(e["target_difference"] == 0.0 for e in again)


def test_matching_does_not_mutate_the_loaded_ledger(rows, config):
    """The audit is read-only at the object level as well as on disk."""
    before = copy.deepcopy(rows)
    match_translation_rows(
        rows, freeze_tag=config["freeze_tag"], classifiers=config["classifiers"],
        match_fields=config["match_fields"],
    )
    assert rows == before
