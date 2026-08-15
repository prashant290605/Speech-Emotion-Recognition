"""Content-addressed feature caches.

A cache is keyed by the manifest rows it covers, the backbone, and the feature
version. A key hit is **never overwritten** -- if the inputs are the same the
features are the same, and if the inputs changed the key changed.

Deviation from the phase brief, deliberate: the brief specifies
``sha256(manifest_rows)`` over the whole manifest. Caches here are keyed
**per corpus** instead, over exactly the rows they contain. The intent is
identical (a cache invalidates when its inputs change) but adding IEMOCAP later
then costs only IEMOCAP's extraction rather than invalidating RAVDESS and
CREMA-D, which on CPU is the difference between an afternoon and a day.

Writes are atomic: arrays go to a temporary directory which is renamed into
place only once every file is complete and fsynced. A killed extraction leaves
either nothing or a whole cache, never a half-written one that would later read
as valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..manifest import ManifestRow
from ..utils.runmeta import utc_timestamp

__all__ = [
    "CACHE_META_NAME",
    "cache_key",
    "manifest_rows_sha256",
    "CacheEntry",
    "load_entry",
    "entry_path",
]

CACHE_META_NAME = "meta.json"


def manifest_rows_sha256(rows: Sequence[ManifestRow]) -> str:
    """Hash the identity and content of the rows a cache covers.

    Includes each row's own file hash, so a corpus whose audio changed produces
    a different key even if the file list is identical.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.utterance_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(row.sha256.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def cache_key(
    rows: Sequence[ManifestRow], backbone: str, feature_version: str
) -> str:
    """sha256(manifest rows) + backbone + feature_version, as a short hex key."""
    payload = "|".join(
        (manifest_rows_sha256(rows), backbone, feature_version)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def entry_path(cache_dir: Path, corpus: str, backbone: str, key: str) -> Path:
    return Path(cache_dir) / f"{corpus}__{backbone}__{key}"


@dataclass
class CacheEntry:
    """One corpus x one backbone worth of features."""

    path: Path
    meta: Dict[str, Any]

    # -- reading -----------------------------------------------------------
    def array(self, name: str, *, mmap: bool = True) -> np.ndarray:
        target = self.path / f"{name}.npy"
        if not target.exists():
            raise FileNotFoundError(f"{name}.npy not in {self.path}")
        return np.load(target, mmap_mode="r" if mmap else None)

    def has(self, name: str) -> bool:
        return (self.path / f"{name}.npy").exists()

    @property
    def utterance_ids(self) -> list[str]:
        return list(self.meta["utterance_ids"])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CacheEntry({self.meta.get('corpus')}/{self.meta.get('backbone')}, n={self.meta.get('n_utterances')})"


def load_entry(path: Path) -> CacheEntry:
    path = Path(path)
    meta_path = path / CACHE_META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(f"no cache metadata at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return CacheEntry(path=path, meta=meta)


def write_entry(
    destination: Path,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, Any],
) -> CacheEntry:
    """Write a cache atomically. Refuses to overwrite an existing key.

    Returns the entry. Raises FileExistsError on a key hit -- callers check
    ``destination.exists()`` first and skip; reaching here with a live path is a
    bug, not a reason to clobber hours of extraction.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. A cache key hit is never overwritten."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=".tmp_", dir=str(destination.parent)))
    try:
        for name, array in arrays.items():
            target = staging / f"{name}.npy"
            np.save(target, array)
            with open(target, "rb+") as handle:
                os.fsync(handle.fileno())

        meta = dict(meta)
        meta["written_at"] = utc_timestamp()
        meta["arrays"] = {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        }
        meta_path = staging / CACHE_META_NAME
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return load_entry(destination)


def preprocessing_metadata(config) -> Dict[str, Any]:
    """The fixed preprocessing contract, recorded on every cache."""
    return {
        "sample_rate": config.features.sample_rate,
        "mono": config.features.mono,
        "peak_normalise": config.features.peak_normalise,
        "standardised": False,  # Phase 5 condition, never a preprocessing default
    }


def library_metadata() -> Dict[str, str]:
    """Versions that can change a feature value."""
    from importlib import metadata

    versions = {}
    for name in ("torch", "torchaudio", "transformers", "librosa", "soundfile", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover
            versions[name] = "not-installed"
    return versions
