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

The same argument applies one level down, to the ledger those runs were written
into. ``configs/FROZEN_LEDGER.sha256`` records the digest of the historical
frozen ledger, and :func:`assert_ledger_unchanged` refuses to read it once the
bytes have moved. This is deliberately a detector, not a lock: it does not make
the file read-only, does not forbid writing a *new* result file, and does not
claim to prevent damage. It makes damage loud, which is what was missing.
``results/runs.jsonl`` is the provenance record every table, figure and the
retrospective audit is generated from, and a truncation or a stray rewrite
would otherwise be discovered only by reading a diff.
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
    "LEDGER_MARKER",
    "FROZEN_LEDGER",
    "ConfigDrift",
    "LedgerDrift",
    "read_freeze_tag",
    "frozen_config_hash",
    "assert_config_frozen",
    "freeze_status",
    "ledger_digest",
    "expected_ledger_digest",
    "assert_ledger_unchanged",
]

FROZEN_MARKER = "configs/FROZEN"

# The historical frozen ledger and the single place its expected digest lives.
# One location on purpose: a hash repeated in several files is a hash that will
# eventually disagree with itself, and then nobody knows which copy is right.
LEDGER_MARKER = "configs/FROZEN_LEDGER.sha256"
FROZEN_LEDGER = "results/runs.jsonl"


class ConfigDrift(RuntimeError):
    """The working config differs from the frozen one."""


class LedgerDrift(RuntimeError):
    """The historical frozen result ledger is not the file it was."""


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


# --------------------------------------------------------------------------
# Ledger provenance
# --------------------------------------------------------------------------
def ledger_digest(path: Optional[Path] = None, *, root: Optional[Path] = None) -> str:
    """sha256 of the ledger file's bytes, streamed.

    Bytes rather than parsed rows, deliberately. A reformatting that preserved
    every value would still change the artifact the published hash refers to,
    and the audit report quotes that hash; agreeing with it has to mean the
    same file, not an equivalent one.
    """
    import hashlib

    target = Path(path) if path is not None else (root or repo_root()) / FROZEN_LEDGER
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_ledger_digest(*, root: Optional[Path] = None) -> Optional[str]:
    """The recorded digest, or None when no expectation has been written."""
    path = (root or repo_root()) / LEDGER_MARKER
    if not path.exists():
        return None
    recorded = path.read_text(encoding="utf-8").strip()
    return recorded or None


def assert_ledger_unchanged(
    path: Optional[Path] = None, *, root: Optional[Path] = None, require: bool = True
) -> str:
    """Raise unless the frozen ledger still hashes to its recorded digest.

    Scope is deliberately narrow. This guards **one** historical artifact, the
    ledger named by :data:`FROZEN_LEDGER`. It says nothing about sensitivity
    ledgers, shard files, or any result file a future experiment writes; those
    are expected to grow, and a guard that complained about them would be
    turned off within a week and then protect nothing.

    Args:
        require: when True, a missing expectation is itself an error. Analysis
            entry points that read the frozen ledger use that. A caller working
            on a repository that has not recorded one may pass False.

    Returns:
        The digest that was verified.
    """
    expected = expected_ledger_digest(root=root)
    if expected is None:
        if require:
            raise LedgerDrift(
                f"no expected digest recorded in {LEDGER_MARKER}. Write the "
                f"sha256 of {FROZEN_LEDGER} there before reading it as frozen "
                "evidence; an unverified ledger is not provenance."
            )
        return ""

    actual = ledger_digest(path, root=root)
    if actual != expected:
        raise LedgerDrift(
            f"{FROZEN_LEDGER} does not match its recorded digest.\n"
            f"  expected {expected}\n"
            f"  actual   {actual}\n"
            f"The frozen ledger has changed. Nothing downstream of it -- tables, "
            "figures, the retrospective audit -- should be regenerated until the "
            "change is understood. Restore the file from git "
            "(`git checkout -- results/runs.jsonl`) if this was accidental, or "
            f"update {LEDGER_MARKER} in a reviewed commit if it was not."
        )
    return actual
