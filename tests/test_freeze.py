"""The config freeze is mechanical, not a convention."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from ser.config import load_config
from ser.freeze import (
    FROZEN_MARKER,
    ConfigDrift,
    assert_config_frozen,
    freeze_status,
    frozen_config_hash,
    read_freeze_tag,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(root), check=True, capture_output=True)


@pytest.fixture()
def tagged_repo(tmp_path):
    """A throwaway git repo with a tagged config, so nothing touches the real one."""
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    _git(root.parent, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")

    config_path = root / "configs" / "default.yaml"
    config_path.write_text(yaml.safe_dump({"project": {"seed": 42}}), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    _git(root, "tag", "grid-freeze-test")
    return root, config_path


class _FakeConfig:
    def __init__(self, config_hash):
        self.config_hash = config_hash


# -- reading the marker ----------------------------------------------------
def test_absent_marker_means_not_frozen(tmp_path):
    assert read_freeze_tag(tmp_path) is None


def test_marker_is_read_and_stripped(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / FROZEN_MARKER).write_text("  grid-freeze-v1  \n", encoding="utf-8")
    assert read_freeze_tag(tmp_path) == "grid-freeze-v1"


def test_empty_marker_means_not_frozen(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / FROZEN_MARKER).write_text("\n", encoding="utf-8")
    assert read_freeze_tag(tmp_path) is None


# -- hashing the tagged config ---------------------------------------------
def test_frozen_hash_reads_the_tagged_revision(tagged_repo):
    root, config_path = tagged_repo
    frozen = frozen_config_hash("grid-freeze-test", root=root)

    # Change the working copy; the tagged hash must not move.
    config_path.write_text(yaml.safe_dump({"project": {"seed": 43}}), encoding="utf-8")
    assert frozen_config_hash("grid-freeze-test", root=root) == frozen


def test_missing_tag_is_reported_clearly(tagged_repo):
    root, _ = tagged_repo
    with pytest.raises(ConfigDrift, match="Does the tag exist"):
        frozen_config_hash("no-such-tag", root=root)


def test_comparison_is_semantic_not_textual(tagged_repo):
    """Reformatting or a comment change is not drift; a value change is."""
    root, config_path = tagged_repo
    frozen = frozen_config_hash("grid-freeze-test", root=root)

    config_path.write_text(
        "# a comment that did not exist before\nproject:\n    seed:   42\n",
        encoding="utf-8",
    )
    reparsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    from ser.utils.runmeta import hash_payload

    assert hash_payload(reparsed) == frozen


# -- the guard -------------------------------------------------------------
def test_matching_config_passes(tagged_repo):
    root, _ = tagged_repo
    (root / FROZEN_MARKER).write_text("grid-freeze-test", encoding="utf-8")
    frozen = frozen_config_hash("grid-freeze-test", root=root)

    assert assert_config_frozen(_FakeConfig(frozen), root=root) == "grid-freeze-test"


def test_drifted_config_is_refused(tagged_repo):
    """The whole point: a two-week grid must not run against a moving config."""
    root, _ = tagged_repo
    (root / FROZEN_MARKER).write_text("grid-freeze-test", encoding="utf-8")

    with pytest.raises(ConfigDrift, match="does not match the frozen tag"):
        assert_config_frozen(_FakeConfig("a" * 64), root=root)


def test_unfrozen_config_is_refused_when_required(tagged_repo):
    root, _ = tagged_repo
    with pytest.raises(ConfigDrift, match="not frozen"):
        assert_config_frozen(_FakeConfig("a" * 64), root=root, require=True)


def test_unfrozen_config_is_tolerated_when_not_required(tagged_repo):
    root, _ = tagged_repo
    assert assert_config_frozen(_FakeConfig("a" * 64), root=root, require=False) == ""


def test_freeze_status_reports_all_three_parts(tagged_repo):
    root, _ = tagged_repo
    (root / FROZEN_MARKER).write_text("grid-freeze-test", encoding="utf-8")
    frozen = frozen_config_hash("grid-freeze-test", root=root)

    tag, recorded, matches = freeze_status(_FakeConfig(frozen), root=root)
    assert (tag, recorded, matches) == ("grid-freeze-test", frozen, True)


# -- the real repository ---------------------------------------------------
def test_freeze_tag_is_a_recorded_column():
    from ser.utils.results import FIELD_NAMES, RUN_ID_FIELDS

    assert "freeze_tag" in FIELD_NAMES
    # Recorded, not a coordinate: refreezing must not invalidate completed runs.
    assert "freeze_tag" not in RUN_ID_FIELDS


def test_real_repo_freeze_state_is_self_consistent():
    """If this repo declares a freeze, the working config must match it."""
    tag = read_freeze_tag()
    if tag is None:
        pytest.skip("repository is not frozen")
    _, _, matches = freeze_status(load_config())
    assert matches, f"working config has drifted from frozen tag {tag!r}"
