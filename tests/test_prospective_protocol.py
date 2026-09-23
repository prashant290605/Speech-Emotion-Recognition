"""The prospective replication's protocol, candidate surface and isolation.

The experiment cannot run — IEMOCAP is not on this machine — but almost
everything that makes it *prospective* can be tested now, and is:

- the protocol is machine-readable, so the runner cannot disagree with the
  document a reader was shown;
- both arms of a matched cell provably receive the same candidate list, which
  is the property the retrospective audit could not establish;
- the historical ledger cannot be written to;
- the audit fails closed on an incomplete or ambiguous join, and never reads a
  target score to decide what to compare;
- the equality tolerance is fixed in the protocol, not chosen later.

A synthetic RBF fixture also checks candidate-level invariance end to end, so
the endpoint itself is exercised before any real data exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.svm import SVC

from ser.prospective import (
    HISTORICAL_LEDGER,
    IsolationError,
    ProtocolError,
    assert_isolated_output,
    candidate_list,
    candidate_list_digest,
    decision_digest,
    load_protocol,
    match_prospective_pairs,
    prediction_digest,
    protocol_digest,
    subset_decision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


# -- the protocol ----------------------------------------------------------
def test_protocol_is_loadable_and_names_itself(protocol):
    assert protocol["protocol"] == "prospective-translation-v1"
    assert len(protocol_digest()) == 64


def test_protocol_is_a_translation_replication_only(protocol):
    """No z-score, CORAL, MK-MMD or blending may creep in."""
    assert set(protocol["alignments"]) == {"none", "mean_shift"}
    assert protocol["blending"] == "none"


def test_protocol_fixes_the_design_before_execution(protocol):
    assert protocol["directions"] == [["cremad", "iemocap"], ["iemocap", "cremad"]]
    assert protocol["label_space"] == "four"
    assert protocol["splits"]["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["features"]["backbones"] == ["hubert", "wav2vec2", "wavlm"]
    assert protocol["features"]["layer"] == "last"
    assert protocol["features"]["pooling"] == "mean"
    assert protocol["classifiers"]["primary"] == "svm_rbf"
    assert protocol["classifiers"]["contrast"] == "logreg"
    assert protocol["expected_runs"] == 120
    assert protocol["expected_trials"] == 2400


def test_target_sign_is_explicitly_not_predicted(protocol):
    target = [e for e in protocol["secondary_endpoints"] if e["name"] == "target_difference"]
    assert target and target[0]["predicted_sign"] == "none"


def test_logistic_regression_equality_is_not_predicted(protocol):
    contrast = [e for e in protocol["secondary_endpoints"]
                if e["name"] == "implementation_contrast"]
    assert contrast and contrast[0]["predicted"] == "not_predicted"


def test_adaptive_expansion_is_forbidden(protocol):
    assert protocol["stopping"]["adaptive_expansion"] == "forbidden"


def test_equality_tolerance_is_fixed_in_the_protocol(protocol):
    """Chosen before results exist, and matching the Proposition 1 tests."""
    assert protocol["equality"]["validation_predictions"] == "exact"
    assert protocol["equality"]["validation_macro_f1"] == "exact"
    assert protocol["equality"]["decision_function_atol"] == 1e-12
    assert protocol["equality"]["on_unexpected_inequality"] == "investigate"


def test_no_target_quantity_is_a_match_field(protocol):
    forbidden = {"macro_f1", "accuracy", "uar", "target", "chance_macro_f1"}
    assert not (set(protocol["match_fields"]) & forbidden)


def test_the_protocol_is_still_marked_draft():
    """It must not claim to be frozen while the corpus is absent."""
    text = (REPO_ROOT / "configs" / "prospective_translation_v1.yaml").read_text(encoding="utf-8")
    assert "NOT FROZEN" in text
    assert (REPO_ROOT / "docs" / "prospective_translation_v1.md").exists()


# -- the candidate surface -------------------------------------------------
def test_both_arms_receive_an_identical_candidate_list(protocol):
    """The property the retrospective audit could not establish.

    The frozen grid samples inside each run, so the two arms' surfaces are only
    similar. Generating from the seed alone makes them identical, which is what
    a claim about every candidate requires.
    """
    for classifier in ("svm_rbf", "logreg"):
        for seed in protocol["splits"]["seeds"]:
            none_arm = candidate_list(classifier, seed, protocol)
            shift_arm = candidate_list(classifier, seed, protocol)
            assert none_arm == shift_arm
            assert candidate_list_digest(none_arm) == candidate_list_digest(shift_arm)


def test_candidate_lists_differ_across_seeds_and_classifiers(protocol):
    a = candidate_list_digest(candidate_list("svm_rbf", 0, protocol))
    b = candidate_list_digest(candidate_list("svm_rbf", 1, protocol))
    c = candidate_list_digest(candidate_list("logreg", 0, protocol))
    assert len({a, b, c}) == 3


def test_candidate_list_honours_the_search_budget(protocol):
    assert len(candidate_list("svm_rbf", 0, protocol)) == protocol["search_budget"]


def test_an_unsupported_classifier_is_refused(protocol):
    with pytest.raises(ProtocolError, match="no candidate space"):
        candidate_list("transformer", 0, protocol)


def test_prediction_digest_is_exact_and_order_sensitive():
    assert prediction_digest(["a", "b"]) == prediction_digest(["a", "b"])
    assert prediction_digest(["a", "b"]) != prediction_digest(["b", "a"])


def test_decision_digest_absorbs_drift_within_the_tolerance_only():
    base = np.array([1.0, -2.0, 0.5])
    assert decision_digest(base, atol=1e-12) == \
        decision_digest(base + 1e-15, atol=1e-12)
    assert decision_digest(base, atol=1e-12) != \
        decision_digest(base + 1e-6, atol=1e-12)


# -- result isolation ------------------------------------------------------
def test_the_historical_ledger_cannot_be_written(protocol):
    with pytest.raises(IsolationError, match="read-only here"):
        assert_isolated_output(Path(HISTORICAL_LEDGER), protocol, root=REPO_ROOT)


def test_traversal_out_of_the_output_root_is_refused(protocol):
    with pytest.raises(IsolationError):
        assert_isolated_output(
            Path("results/prospective_translation_v1/../runs.jsonl"),
            protocol, root=REPO_ROOT)


def test_paths_outside_the_output_root_are_refused(protocol):
    for candidate in ("results/other.jsonl", "reports/RESULTS.md", "data/manifest.csv"):
        with pytest.raises(IsolationError):
            assert_isolated_output(Path(candidate), protocol, root=REPO_ROOT)


def test_the_protocols_own_paths_are_accepted(protocol):
    for key in ("runs", "trials", "provenance", "audit_json", "audit_markdown"):
        assert_isolated_output(Path(protocol[key]), protocol, root=REPO_ROOT)


def test_no_prospective_result_tree_exists_yet():
    """Nothing has been run; the report must be able to say so truthfully."""
    assert not (REPO_ROOT / "results" / "prospective_translation_v1").exists()


# -- the subset decision rule ---------------------------------------------
def full_counts(n):
    return {c: {s: n for s in range(1, 6)} for c in ("angry", "happy", "neutral", "sad")}


def test_adequate_improvised_counts_select_improvised(protocol):
    assert subset_decision({"improvised": full_counts(40)}, protocol)["chosen"] == "improvised"


def test_a_thin_class_falls_back_to_both(protocol):
    counts = full_counts(40)
    counts["angry"] = {1: 5, 2: 5, 3: 5, 4: 5, 5: 5}
    decision = subset_decision({"improvised": counts}, protocol)
    assert decision["chosen"] == "both"
    assert any("angry" in r for r in decision["failed_checks"])


def test_a_class_missing_from_a_session_falls_back_to_both(protocol):
    counts = full_counts(40)
    counts["sad"] = {1: 40, 2: 40, 3: 40, 4: 40}
    assert subset_decision({"improvised": counts}, protocol)["chosen"] == "both"


def test_the_decision_records_why(protocol):
    decision = subset_decision({"improvised": full_counts(1)}, protocol)
    assert decision["failed_checks"]
    assert decision["thresholds"] == protocol["iemocap_subset"]["thresholds"]


def test_the_subset_is_still_undecided(protocol):
    """It cannot be decided without the corpus, and must not be pre-filled."""
    assert protocol["iemocap_subset"]["decided"] is None


# -- the audit -------------------------------------------------------------
def prospective_row(alignment, run_id, protocol, **extra):
    base = {
        "protocol": protocol["protocol"], "status": "ok", "run_id": run_id,
        "alignment": alignment, "source_corpus": "cremad", "target_corpus": "iemocap",
        "seed": 0, "backbone": "hubert", "layer_agg": "last", "layer_index": None,
        "feature_branch": "ssl", "classifier": "svm_rbf",
        "label_map_hash": "h", "split_spec_hash": "h", "feature_spec_hash": "h",
        "search_spec_hash": "h", "split_id": "cremad-iemocap-s0", "n_classes": 4,
        "class_names": ["angry", "happy", "neutral", "sad"],
        "n_train": 100, "n_val": 25, "n_target_adapt": 50, "n_target_test": 50,
        "n_search_trials": 2,
        "selection_source_val_macro_f1": 0.5,
        "hyperparams_json": json.dumps({"selected": {"C": 1.0, "gamma": 0.1}}),
        "macro_f1": 0.3, "chance_macro_f1": 0.25,
    }
    base.update(extra)
    return base


def trial_rows(run_id, *, predictions="p", scores=(0.4, 0.5)):
    return [{"run_id": run_id, "candidate_index": i,
             "params": {"C": 1.0 + i, "gamma": 0.1},
             "source_val_macro_f1": scores[i],
             "prediction_digest": f"{predictions}{i}",
             "decision_digest": f"d{i}"}
            for i in range(2)]


def test_audit_reports_candidate_level_equality(protocol):
    runs = [prospective_row("none", "a", protocol),
            prospective_row("mean_shift", "b", protocol, macro_f1=0.4)]
    trials = trial_rows("a") + trial_rows("b")
    matched = match_prospective_pairs(runs, trials, protocol)
    assert len(matched) == 1
    cell = matched[0]
    assert cell["candidates"] == 2
    assert cell["candidate_prediction_equal"] == 2
    assert cell["candidate_score_equal"] == 2
    assert cell["selected_score_equal"] and cell["selected_params_equal"]
    assert cell["target_difference"] == pytest.approx(0.1)


def test_audit_detects_a_candidate_level_difference(protocol):
    runs = [prospective_row("none", "a", protocol),
            prospective_row("mean_shift", "b", protocol)]
    trials = trial_rows("a") + trial_rows("b", predictions="q")
    cell = match_prospective_pairs(runs, trials, protocol)[0]
    assert cell["candidate_prediction_equal"] == 0


def test_audit_refuses_arms_that_got_different_candidates(protocol):
    runs = [prospective_row("none", "a", protocol),
            prospective_row("mean_shift", "b", protocol)]
    trials = trial_rows("a")
    other = trial_rows("b")
    other[0]["params"] = {"C": 999.0, "gamma": 0.1}
    with pytest.raises(ProtocolError, match="same candidate list"):
        match_prospective_pairs(runs, trials + other, protocol)


@pytest.mark.parametrize("damage", ["missing_arm", "duplicate", "failed", "short_surface"])
def test_audit_fails_closed(protocol, damage):
    runs = [prospective_row("none", "a", protocol),
            prospective_row("mean_shift", "b", protocol)]
    trials = trial_rows("a") + trial_rows("b")
    if damage == "missing_arm":
        runs.pop()
    elif damage == "duplicate":
        runs.append(prospective_row("none", "a", protocol))
    elif damage == "failed":
        runs[1]["status"] = "failed"
    else:
        trials = trial_rows("a") + trial_rows("b")[:1]
    with pytest.raises(ProtocolError):
        match_prospective_pairs(runs, trials, protocol)


def test_target_score_does_not_affect_the_join(protocol):
    """Behavioural: overwrite every target value, get the same pairing."""
    runs = [prospective_row("none", "a", protocol),
            prospective_row("mean_shift", "b", protocol, macro_f1=0.9)]
    trials = trial_rows("a") + trial_rows("b")
    first = match_prospective_pairs(runs, trials, protocol)
    for row in runs:
        row["macro_f1"] = 0.5
    second = match_prospective_pairs(runs, trials, protocol)
    assert [(c["none_run_id"], c["mean_shift_run_id"], c["candidate_prediction_equal"])
            for c in first] == \
           [(c["none_run_id"], c["mean_shift_run_id"], c["candidate_prediction_equal"])
            for c in second]


# -- the endpoint itself, on a synthetic fixture --------------------------
def test_candidate_level_rbf_invariance_holds_on_a_synthetic_fixture(protocol):
    """Exercise the primary endpoint before any real data exists.

    Every candidate in the list is fitted on both arms and its validation
    predictions digested, exactly as the runner will. This is the prospective
    endpoint in miniature: if the machinery were wrong, it would be wrong here.
    """
    rng = np.random.default_rng(4)
    centres = np.array([[0., 0., 0.], [3., 1., -1.], [-2., 2., 1.], [1., -3., 2.]])
    train = np.vstack([c + 0.5 * rng.normal(size=(12, 3)) for c in centres])
    labels = np.repeat(["angry", "happy", "neutral", "sad"], 12)
    validation = np.vstack([c + 0.5 * rng.normal(size=(5, 3)) for c in centres])
    delta = np.array([1.5, -0.8, 2.0])
    atol = protocol["equality"]["decision_function_atol"]

    equal_predictions = equal_decisions = 0
    candidates = candidate_list("svm_rbf", 0, protocol)
    for params in candidates:
        fit = dict(kernel="rbf", random_state=0, tol=1e-9, **params)
        plain = SVC(**fit).fit(train, labels)
        shifted = SVC(**fit).fit(train + delta, labels)
        if prediction_digest(plain.predict(validation)) == \
           prediction_digest(shifted.predict(validation + delta)):
            equal_predictions += 1
        if decision_digest(plain.decision_function(validation), atol=atol) == \
           decision_digest(shifted.decision_function(validation + delta), atol=atol):
            equal_decisions += 1

    assert equal_predictions == len(candidates)
    assert equal_decisions == len(candidates)
