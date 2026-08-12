"""Provenance capture.

Every row in ``results/runs.jsonl`` carries a :class:`RunMeta`. The point is
that a reviewer can take any single number in any table and recover the exact
commit, config, and library set that produced it -- and can tell whether the
tree was dirty at the time, which is the failure mode that silently detaches
results from code.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

__all__ = ["RunMeta", "capture_runmeta", "hash_payload", "utc_timestamp"]

# Libraries whose version can change a number. Recorded on every row.
TRACKED_LIBRARIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "librosa",
    "soundfile",
    "transformers",
    "torch",
    "torchaudio",
)


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, second resolution, always suffixed ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_payload(payload: Any) -> str:
    """Stable sha256 over any JSON-serialisable payload.

    Canonicalised with sorted keys and no insignificant whitespace, so the hash
    depends on the config's *content*, not on YAML key order or formatting.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunMeta:
    """Immutable provenance stamp for one run."""

    git_sha: str
    git_dirty: bool
    git_branch: str
    config_hash: str
    timestamp: str
    hostname: str
    username: str
    python_version: str
    platform: str
    lib_versions: Dict[str, str] = field(default_factory=dict)

    def as_row_fields(self) -> Dict[str, Any]:
        """Flatten into the subset of result-schema columns this owns."""
        return {
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "lib_versions_json": json.dumps(self.lib_versions, sort_keys=True),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def capture_runmeta(config_hash: str, *, repo_root: Path | None = None) -> RunMeta:
    """Snapshot the current environment.

    Args:
        config_hash: From ``Config.config_hash``. Passed in rather than computed
            here so this module stays independent of the config schema.
        repo_root: Directory to interrogate for git state. Defaults to the
            repository this file lives in.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    sha, dirty, branch = _git_state(root)

    return RunMeta(
        git_sha=sha,
        git_dirty=dirty,
        git_branch=branch,
        config_hash=config_hash,
        timestamp=utc_timestamp(),
        hostname=_safe(socket.gethostname, "unknown"),
        username=_safe(getpass.getuser, "unknown"),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        lib_versions=_library_versions(),
    )


def _repo_root() -> Path:
    # src/ser/utils/runmeta.py -> repo root is four levels up.
    return Path(__file__).resolve().parents[3]


def _git_state(root: Path) -> tuple[str, bool, str]:
    """Return ``(sha, dirty, branch)``, degrading to sentinels outside a repo.

    A missing git binary or a non-repository directory is not an error -- the
    code must still run from an unpacked archive -- but it is recorded as
    ``"unknown"`` so it can never be mistaken for a real commit.
    """
    sha = _git(root, "rev-parse", "HEAD") or "unknown"
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"

    status = _git(root, "status", "--porcelain")
    # Distinguish "clean tree" (empty string) from "git unavailable" (None).
    dirty = bool(status) if status is not None else True

    return sha, dirty, branch


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _library_versions() -> Dict[str, str]:
    from importlib import metadata

    versions: Dict[str, str] = {}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _safe(fn, default: str) -> str:
    try:
        return str(fn())
    except Exception:  # pragma: no cover - environment dependent
        return default
