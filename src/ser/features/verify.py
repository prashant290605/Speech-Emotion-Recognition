"""Cache verification.

Four assertions, per the phase brief:

1. Row count matches the manifest.
2. No NaN or Inf anywhere.
3. Shapes are exactly what the config declares.
4. **Utterance ordering matches manifest ordering exactly.**

(4) is the one that would be silent. Every downstream index into these arrays --
every split, every label lookup -- assumes row *i* of the cache is row *i* of the
corpus in the manifest. A reordering would misalign features and labels across
the board and still produce plausible-looking numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..manifest import ManifestRow
from .cache import cache_key, entry_path, load_entry, manifest_rows_sha256
from .extract import MFCC_BACKBONE, rows_for
from .mfcc import MFCC_DIM

__all__ = ["verify_entry", "verify_all"]


def verify_entry(
    rows: Sequence[ManifestRow], config, corpus: str, backbone: str
) -> Dict:
    """Verify one (corpus, backbone) cache. Returns a result dict with problems."""
    corpus_rows = rows_for(rows, corpus)
    key = cache_key(corpus_rows, backbone, config.features.feature_version)
    path = entry_path(config.resolve(config.paths.cache_dir), corpus, backbone, key)

    result = {
        "corpus": corpus,
        "backbone": backbone,
        "key": key,
        "path": path,
        "present": path.exists(),
        "problems": [],
        "bytes": 0,
    }
    if not path.exists():
        result["problems"].append("cache absent")
        return result

    entry = load_entry(path)
    problems: List[str] = result["problems"]
    n = len(corpus_rows)

    # 1. row count
    if entry.meta["n_utterances"] != n:
        problems.append(f"meta says {entry.meta['n_utterances']} utterances, manifest has {n}")

    # 4. ordering -- checked before touching arrays, since everything else
    #    assumes it holds.
    cached_ids = entry.utterance_ids
    expected_ids = [row.utterance_id for row in corpus_rows]
    if cached_ids != expected_ids:
        if sorted(cached_ids) == sorted(expected_ids):
            first = next(
                i for i, (a, b) in enumerate(zip(cached_ids, expected_ids)) if a != b
            )
            problems.append(
                f"utterance ORDER differs from the manifest, first at index {first}: "
                f"cache has {cached_ids[first]!r}, manifest has {expected_ids[first]!r}"
            )
        else:
            problems.append("utterance ids differ from the manifest (not just order)")

    if entry.meta.get("manifest_rows_sha256") != manifest_rows_sha256(corpus_rows):
        problems.append("manifest_rows_sha256 does not match the current manifest")

    expected_shapes = _expected_shapes(config, backbone, n)
    for name, expected in expected_shapes.items():
        if not entry.has(name):
            if name == "segments" and not config.features.segment_pooling_enabled:
                continue
            problems.append(f"missing array {name!r}")
            continue

        array = entry.array(name)
        result["bytes"] += (path / f"{name}.npy").stat().st_size

        # 3. shape
        if tuple(array.shape) != expected:
            problems.append(f"{name}: shape {tuple(array.shape)} != expected {expected}")
            continue

        # 2. finite. Chunked so a multi-GB memmap is not fully materialised.
        bad = _count_nonfinite(array)
        if bad:
            problems.append(f"{name}: {bad} non-finite value(s)")

    return result


def _expected_shapes(config, backbone: str, n: int) -> Dict[str, tuple]:
    if backbone == MFCC_BACKBONE:
        return {"mfcc": (n, MFCC_DIM)}

    shapes = {"layers": (n, config.features.n_layers, config.features.hidden_dim)}
    if config.features.segment_pooling_enabled:
        shapes["segments"] = (
            n,
            config.features.n_layers,
            config.features.n_segments,
            config.features.hidden_dim,
        )
    return shapes


def _count_nonfinite(array: np.ndarray, chunk: int = 512) -> int:
    total = 0
    for start in range(0, array.shape[0], chunk):
        block = np.asarray(array[start : start + chunk], dtype=np.float32)
        total += int((~np.isfinite(block)).sum())
    return total


def verify_all(
    rows: Sequence[ManifestRow], config, corpora: Sequence[str], backbones: Sequence[str]
) -> List[Dict]:
    return [
        verify_entry(rows, config, corpus, backbone)
        for corpus in corpora
        for backbone in backbones
        if rows_for(rows, corpus)
    ]
