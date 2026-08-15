"""Phase 2 leakage assertions.

Three of the brief's four assertions live here; the fourth (``map_label``
purity) is in ``tests/test_labelmap.py``. Where a manifest is available these
run over every pair and seed the config declares, not a sampled subset.
"""

from __future__ import annotations

from itertools import product

import pytest

from ser.config import load_config
from ser.leakage import (
    LeakageError,
    assert_alignment_blind_to_target_test,
    assert_no_leakage,
    assert_source_target_utterances_disjoint,
    assert_speaker_disjoint,
    check_pair_split,
)
from ser.manifest import ManifestRow, read_manifest
from ser.splits import ROLES, make_pair_split


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def rows(config):
    path = config.resolve(config.paths.manifest)
    if not path.exists():
        pytest.skip("manifest not built yet; run `ser manifest`")
    return read_manifest(path)


@pytest.fixture(scope="module")
def corpora(rows):
    return sorted({row.corpus for row in rows})


def _all_pair_splits(rows, config, corpora):
    for source, target in product(corpora, repeat=2):
        for seed in config.splits.seeds:
            yield make_pair_split(rows, config, source, target, seed)


# -- synthetic fixtures ----------------------------------------------------
def _row(corpus, speaker, uid, label="angry"):
    return ManifestRow(
        corpus=corpus,
        file_path=f"{uid}.wav",
        utterance_id=uid,
        speaker_id=speaker,
        session_id="",
        subset="",
        original_label=label,
        label_six=label,
        label_four=label if label != "disgust" else "",
        duration_s=1.0,
        sample_rate=16000,
        sha256="0" * 64,
    )


@pytest.fixture()
def toy_rows():
    out = []
    for corpus in ("ravdess", "cremad"):
        for speaker in range(8):
            for utt in range(4):
                out.append(
                    _row(corpus, f"{corpus}_{speaker}", f"{corpus}/s{speaker}u{utt}")
                )
    return out


# -- assertion 1: speaker disjointness -------------------------------------
def test_speakers_are_disjoint_across_every_pair_and_seed(rows, config, corpora):
    for pair in _all_pair_splits(rows, config, corpora):
        assert_speaker_disjoint(pair)


def test_speaker_disjointness_is_actually_checked(toy_rows, config):
    """The assertion must fail on a violating split, or it proves nothing."""
    pair = make_pair_split(toy_rows, config, "ravdess", "cremad", 0)
    broken = type(pair)(
        **{
            **{k: getattr(pair, k) for k in ("source_corpus", "target_corpus", "label_space", "seed", "split_id")},
            "source_train": pair.source_train,
            # Give source_val a speaker that source_train already owns.
            "source_val": type(pair.source_val)(
                corpus=pair.source_val.corpus,
                role="source_val",
                utterance_ids=pair.source_val.utterance_ids,
                group_ids=pair.source_train.group_ids,
            ),
            "target_adapt": pair.target_adapt,
            "target_test": pair.target_test,
        }
    )
    with pytest.raises(LeakageError, match="speaker-disjoint"):
        assert_speaker_disjoint(broken)


# -- assertion 2: no utterance on both sides -------------------------------
def test_source_and_target_utterances_never_overlap(rows, config, corpora):
    for pair in _all_pair_splits(rows, config, corpora):
        assert_source_target_utterances_disjoint(pair)


def test_in_domain_pairs_partition_one_corpus_four_ways(rows, config, corpora):
    """The failure this catches reports training data as an in-domain test score."""
    for corpus in corpora:
        for seed in config.splits.seeds:
            pair = make_pair_split(rows, config, corpus, corpus, seed)
            assert pair.is_in_domain

            groups = [set(pair.splits()[role].group_ids) for role in ROLES]
            for i, left in enumerate(groups):
                for right in groups[i + 1 :]:
                    assert not (left & right)

            ids = [pair.utterance_ids(role) for role in ROLES]
            for i, left in enumerate(ids):
                for right in ids[i + 1 :]:
                    assert not (left & right)


def test_overlap_is_actually_detected(toy_rows, config):
    pair = make_pair_split(toy_rows, config, "ravdess", "ravdess", 0)
    broken = type(pair)(
        source_corpus=pair.source_corpus,
        target_corpus=pair.target_corpus,
        label_space=pair.label_space,
        seed=pair.seed,
        split_id=pair.split_id,
        source_train=pair.source_train,
        source_val=pair.source_val,
        target_adapt=pair.target_adapt,
        # target_test reusing source_train's utterances is the in-domain bug.
        target_test=type(pair.target_test)(
            corpus=pair.target_test.corpus,
            role="target_test",
            utterance_ids=pair.source_train.utterance_ids,
            group_ids=pair.target_test.group_ids,
        ),
    )
    with pytest.raises(LeakageError, match="both the source and target side"):
        assert_source_target_utterances_disjoint(broken)


# -- assertion 3: alignment never sees target_test -------------------------
class _Alignment:
    """Minimal stand-in for the Phase 5 interface."""

    def __init__(self, fitted_on_indices):
        self.fitted_on_indices = set(fitted_on_indices)


class _AlignmentWithoutRecord:
    pass


def test_alignment_fitted_on_target_adapt_passes(rows, config, corpora):
    for pair in _all_pair_splits(rows, config, corpora):
        assert_alignment_blind_to_target_test(
            _Alignment(pair.utterance_ids("target_adapt")), pair
        )
        break


def test_alignment_fitted_on_target_test_is_rejected(rows, config, corpora):
    """Exactly the original study's bug: CORAL fitted on target train AND test."""
    pair = next(_all_pair_splits(rows, config, corpora))
    leaky = _Alignment(
        pair.utterance_ids("target_adapt") | pair.utterance_ids("target_test")
    )
    with pytest.raises(LeakageError, match="target_test utterance"):
        assert_alignment_blind_to_target_test(leaky, pair)


def test_alignment_fitted_on_unrelated_data_is_rejected(rows, config, corpora):
    pair = next(_all_pair_splits(rows, config, corpora))
    with pytest.raises(LeakageError, match="outside this split"):
        assert_alignment_blind_to_target_test(_Alignment({"nowhere/u1"}), pair)


def test_alignment_that_cannot_be_checked_is_rejected(rows, config, corpora):
    """An object with no record is a failure, not a pass by omission."""
    pair = next(_all_pair_splits(rows, config, corpora))
    with pytest.raises(LeakageError, match="does not expose"):
        assert_alignment_blind_to_target_test(_AlignmentWithoutRecord(), pair)


# -- determinism -----------------------------------------------------------
def test_splits_are_deterministic_given_a_seed(rows, config, corpora):
    for source, target in product(corpora, repeat=2):
        a = make_pair_split(rows, config, source, target, 0)
        b = make_pair_split(rows, config, source, target, 0)
        assert a == b


def test_different_seeds_give_different_partitions(rows, config, corpora):
    source, target = corpora[0], corpora[-1]
    a = make_pair_split(rows, config, source, target, config.splits.seeds[0])
    b = make_pair_split(rows, config, source, target, config.splits.seeds[1])
    assert a.target_test.group_ids != b.target_test.group_ids


def test_a_corpus_gets_the_same_source_partition_across_targets(rows, config, corpora):
    """Keeps results comparable across targets and stops a new pair from
    perturbing existing ones."""
    if len(corpora) < 2:
        pytest.skip("needs two corpora")
    source = corpora[0]
    others = [c for c in corpora if c != source]
    first = make_pair_split(rows, config, source, others[0], 0)
    for other in others[1:]:
        assert make_pair_split(rows, config, source, other, 0).source_train.group_ids == (
            first.source_train.group_ids
        )


# -- content ---------------------------------------------------------------
def test_splits_contain_only_labelled_utterances(rows, config, corpora):
    """RAVDESS 'surprised' has no class in the 6-class space and must not appear."""
    by_id = {row.utterance_id: row for row in rows}
    for pair in _all_pair_splits(rows, config, corpora):
        for role in ROLES:
            for uid in pair.utterance_ids(role):
                row = by_id[uid]
                label = row.label_six if pair.label_space == "six" else row.label_four
                assert label, f"{uid} has no label in {pair.label_space}"


def test_no_split_is_empty(rows, config, corpora):
    for pair in _all_pair_splits(rows, config, corpora):
        assert check_pair_split(pair) == []


def test_assert_no_leakage_runs_every_split_check(rows, config, corpora):
    for pair in _all_pair_splits(rows, config, corpora):
        assert_no_leakage(pair)
