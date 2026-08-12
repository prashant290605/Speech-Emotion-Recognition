"""The reproducibility spine: seeding actually determines draws, and provenance
is actually captured."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from ser.utils.runmeta import RunMeta, capture_runmeta, hash_payload, utc_timestamp
from ser.utils.seeding import set_all_seeds


# -- seeding ---------------------------------------------------------------
def _draw():
    return (
        random.random(),
        np.random.rand(4).tolist(),
    )


def test_same_seed_gives_same_draws():
    set_all_seeds(1234)
    first = _draw()
    set_all_seeds(1234)
    assert _draw() == first


def test_different_seeds_give_different_draws():
    set_all_seeds(0)
    first = _draw()
    set_all_seeds(1)
    assert _draw() != first


def test_seed_is_returned_so_the_recorded_value_is_the_applied_one():
    assert set_all_seeds(7) == 7


def test_torch_is_seeded_when_available():
    torch = pytest.importorskip("torch")

    set_all_seeds(99)
    first = torch.rand(8)
    set_all_seeds(99)
    assert torch.equal(first, torch.rand(8))

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.parametrize("bad", [-1, 1.5, "42", True, None])
def test_invalid_seeds_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        set_all_seeds(bad)


# -- hashing ---------------------------------------------------------------
def test_hash_payload_is_order_independent():
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})


def test_hash_payload_is_content_sensitive():
    assert hash_payload({"a": 1}) != hash_payload({"a": 2})


def test_hash_payload_distinguishes_types():
    assert hash_payload({"a": 1}) != hash_payload({"a": "1"})


def test_hash_payload_is_sha256_length():
    assert len(hash_payload({"a": 1})) == 64


# -- provenance ------------------------------------------------------------
def test_capture_runmeta_populates_every_field():
    meta = capture_runmeta("deadbeef")

    assert isinstance(meta, RunMeta)
    assert meta.config_hash == "deadbeef"
    assert meta.timestamp.endswith("Z")
    assert meta.hostname
    assert meta.python_version
    assert meta.platform
    assert isinstance(meta.git_dirty, bool)
    assert meta.lib_versions["numpy"] == np.__version__


def test_runmeta_row_fields_are_json_serialisable():
    fields = capture_runmeta("cafe").as_row_fields()
    assert set(fields) == {
        "git_sha",
        "git_dirty",
        "config_hash",
        "timestamp",
        "hostname",
        "lib_versions_json",
    }
    assert json.loads(fields["lib_versions_json"])
    json.dumps(fields)


def test_runmeta_outside_a_git_repo_is_marked_unknown_not_faked(tmp_path):
    """A missing repo must never look like a real commit."""
    meta = capture_runmeta("x", repo_root=tmp_path)
    assert meta.git_sha == "unknown"
    # Unknown provenance is treated as dirty: it is the conservative reading.
    assert meta.git_dirty is True


def test_runmeta_is_immutable():
    meta = capture_runmeta("x")
    with pytest.raises(Exception):
        meta.git_sha = "tampered"  # type: ignore[misc]


def test_utc_timestamp_format():
    stamp = utc_timestamp()
    assert len(stamp) == 20
    assert stamp[4] == "-" and stamp[10] == "T" and stamp.endswith("Z")
