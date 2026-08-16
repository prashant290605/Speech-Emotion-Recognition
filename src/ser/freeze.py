"""Config freeze: the grid refuses to run against a config that has drifted.

A freeze that is only a convention will not survive a two-week grid. This makes
it mechanical:

* ``configs/FROZEN`` holds the name of a git tag.
* The tagged commit's ``configs/default.yaml`` is the frozen config.
* Before the grid starts, the working config is compared against it and the
  runner **refuses to start** if they differ semantically.
* The tag and the frozen hash are recorded on every run row.

Comparison is on the *parsed* config, not the file bytes, so reformatting or a
comment change is not drift. Only a value change is.

Why this matters more than it looks: with ``config_hash`` no longer a ``run_id``
coordinate (schema v4), an edit mid-grid no longer orphans completed runs — it
silently produces runs that are *not comparable* to the ones before it, under
the same ids. The freeze is what closes that gap.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Tuple

import yaml

from .config import ConfigError, repo_root
from .utils.runmeta import hash_payload

__all__ = [
    "FROZEN_MARKER",
    "ConfigDrift",
    "read_freeze_tag",
    "frozen_config_hash",
    "assert_config_frozen",
    "freeze_status",
]

FROZEN_MARKER = "configs/FROZEN"


class ConfigDrift(RuntimeError):
    """The working config differs from the frozen one."""


def read_freeze_tag(root: Optional[Path] = None) -> Optional[str]:
    """Tag name in ``configs/FROZEN``, or None if the config is not frozen."""
    path = (root or repo_root()) / FROZEN_MARKER
    if not path.exists():
        return None
    tag = path.read_text(encoding="utf-8").strip()
    return tag or None


def _git_show(root: Path, ref: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "show", ref],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def frozen_config_hash(
    tag: str, *, root: Optional[Path] = None, config_path: str = "configs/default.yaml"
) -> str:
    """Hash of the config as it stood at ``tag``.

    Hashes the parsed mapping, exactly as ``Config.config_hash`` does, so
    formatting and comments are not drift.
    """
    root = root or repo_root()
    text = _git_show(root, f"{tag}:{config_path}")
    if text is None:
        raise ConfigDrift(
            f"cannot read {config_path} at tag {tag!r}. Does the tag exist? "
            f"Create it with: git tag {tag}"
        )
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ConfigDrift(f"{config_path} at {tag!r} is not a YAML mapping")
    return hash_payload(parsed)


def freeze_status(config, *, root: Optional[Path] = None) -> Tuple[Optional[str], Optional[str], bool]:
    """``(tag, frozen_hash, matches)``. ``tag`` is None when not frozen."""
    tag = read_freeze_tag(root)
    if tag is None:
        return None, None, False
    frozen = frozen_config_hash(tag, root=root)
    return tag, frozen, frozen == config.config_hash


def assert_config_frozen(config, *, root: Optional[Path] = None, require: bool = True) -> str:
    """Raise unless the working config matches the frozen tag.

    Args:
        require: when True, an *absent* freeze is also an error. The grid runner
            uses that; smaller commands may pass False.

    Returns:
        The tag name.
    """
    tag, frozen, matches = freeze_status(config, root=root)

    if tag is None:
        if require:
            raise ConfigDrift(
                f"the config is not frozen. Write a git tag name into "
                f"{FROZEN_MARKER} and tag the commit before running the grid:\n"
                f"    git tag grid-freeze-v1 && echo grid-freeze-v1 > {FROZEN_MARKER}\n"
                "A two-week grid run against a moving config produces rows that "
                "are not comparable to each other."
            )
        return ""

    if not matches:
        raise ConfigDrift(
            f"working config does not match the frozen tag {tag!r}.\n"
            f"  frozen  {frozen}\n"
            f"  working {config.config_hash}\n"
            "Either revert the working config, or freeze again deliberately "
            "(new tag, new marker) and accept that earlier rows are not "
            "comparable to later ones."
        )
    return tag
