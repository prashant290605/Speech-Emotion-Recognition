"""Extraction driver: one cache per (corpus, backbone), skipped on a key hit.

Why batch size 1, given this is the expensive phase on a CPU-only machine:

Batched inference needs padding, and a padded frame that reaches the mean
corrupts the pooled vector -- silently, and worst for the shortest utterances.
Masking fixes that in principle, but ``facebook/wav2vec2-base`` is documented as
degrading under masked batched inference (it was pretrained without an attention
mask, and its feature extractor sets ``return_attention_mask=False``). Treating
one backbone differently from the other two would put an unmeasured confound
straight into the backbone comparison.

So: batch 1, and recover the wall time with **process-level parallelism** -- one
process per backbone, each with a slice of the CPU threads. That is a scheduling
change with no numerical consequences at all.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..manifest import ManifestRow
from ..utils.runmeta import capture_runmeta
from .audio import load_audio
from .cache import (
    cache_key,
    entry_path,
    library_metadata,
    load_entry,
    manifest_rows_sha256,
    preprocessing_metadata,
    write_entry,
)
from .mfcc import MFCC_DIM, extract_mfcc
from .ssl import SSLExtractor, extract_corpus

__all__ = ["MFCC_BACKBONE", "plan_extraction", "extract_one", "run_extraction"]

# MFCC is cached alongside the SSL backbones under a reserved name, so the same
# key/skip/verify machinery covers it.
MFCC_BACKBONE = "mfcc"


def rows_for(rows: Sequence[ManifestRow], corpus: str) -> List[ManifestRow]:
    """Rows of one corpus, in manifest order. Ordering is the contract."""
    return [row for row in rows if row.corpus == corpus]


def plan_extraction(
    rows: Sequence[ManifestRow], config, corpora: Sequence[str], backbones: Sequence[str]
) -> List[dict]:
    """Enumerate work units and mark which already exist."""
    cache_dir = config.resolve(config.paths.cache_dir)
    plan = []
    for corpus in corpora:
        corpus_rows = rows_for(rows, corpus)
        if not corpus_rows:
            continue
        for backbone in backbones:
            key = cache_key(corpus_rows, backbone, config.features.feature_version)
            path = entry_path(cache_dir, corpus, backbone, key)
            plan.append(
                {
                    "corpus": corpus,
                    "backbone": backbone,
                    "key": key,
                    "path": path,
                    "n_rows": len(corpus_rows),
                    "exists": path.exists(),
                }
            )
    return plan


def _base_meta(config, corpus: str, backbone: str, key: str, corpus_rows) -> Dict:
    meta = capture_runmeta(config.config_hash)
    return {
        "cache_key": key,
        "corpus": corpus,
        "backbone": backbone,
        "feature_version": config.features.feature_version,
        "n_utterances": len(corpus_rows),
        "utterance_ids": [row.utterance_id for row in corpus_rows],
        "manifest_rows_sha256": manifest_rows_sha256(corpus_rows),
        "preprocessing": preprocessing_metadata(config),
        "libraries": library_metadata(),
        "git_sha": meta.git_sha,
        "git_dirty": meta.git_dirty,
        "config_hash": config.config_hash,
    }


def extract_one(
    rows: Sequence[ManifestRow],
    config,
    corpus: str,
    backbone: str,
    *,
    progress: Optional[Callable[[str, str, int, int, float], None]] = None,
) -> dict:
    """Extract one (corpus, backbone). No-op on a cache key hit."""
    corpus_rows = rows_for(rows, corpus)
    if not corpus_rows:
        raise ValueError(f"no manifest rows for corpus {corpus!r}")

    key = cache_key(corpus_rows, backbone, config.features.feature_version)
    destination = entry_path(config.resolve(config.paths.cache_dir), corpus, backbone, key)

    if destination.exists():
        entry = load_entry(destination)
        return {
            "status": "cached",
            "corpus": corpus,
            "backbone": backbone,
            "key": key,
            "path": destination,
            "n": entry.meta["n_utterances"],
            "wall_seconds": 0.0,
        }

    meta = _base_meta(config, corpus, backbone, key, corpus_rows)

    if backbone == MFCC_BACKBONE:
        started = time.perf_counter()
        matrix = np.empty((len(corpus_rows), MFCC_DIM), dtype=np.float32)
        for index, row in enumerate(corpus_rows):
            matrix[index] = extract_mfcc(load_audio(row.file_path, config), config)
            if progress and (index + 1) % 500 == 0:
                progress(corpus, backbone, index + 1, len(corpus_rows),
                         time.perf_counter() - started)
        arrays = {"mfcc": matrix}
        wall = time.perf_counter() - started
        meta["mfcc"] = {
            "n_coefficients": config.features.mfcc_n_coefficients,
            "deltas": config.features.mfcc_deltas,
            "pooling": list(config.features.mfcc_pooling),
            "dim": MFCC_DIM,
        }
    else:
        extractor = SSLExtractor.load(backbone, config)
        meta["checkpoint"] = extractor.checkpoint
        meta["input_normalised_by_feature_extractor"] = extractor.input_normalised
        meta["n_layers"] = extractor.n_layers
        meta["hidden_dim"] = extractor.hidden_dim
        meta["n_segments"] = extractor.n_segments

        def _forward(done: int, total: int, elapsed: float) -> None:
            if progress:
                progress(corpus, backbone, done, total, elapsed)

        result = extract_corpus(
            corpus_rows,
            extractor,
            config,
            want_segments=config.features.segment_pooling_enabled,
            progress=_forward,
        )
        arrays = result["arrays"]
        wall = result["wall_seconds"]

    meta["extraction_wall_seconds"] = round(wall, 2)
    write_entry(destination, arrays, meta)

    return {
        "status": "extracted",
        "corpus": corpus,
        "backbone": backbone,
        "key": key,
        "path": destination,
        "n": len(corpus_rows),
        "wall_seconds": wall,
    }


def run_extraction(
    rows: Sequence[ManifestRow],
    config,
    corpora: Sequence[str],
    backbones: Sequence[str],
    *,
    progress: Optional[Callable] = None,
) -> List[dict]:
    results = []
    for unit in plan_extraction(rows, config, corpora, backbones):
        results.append(
            extract_one(rows, config, unit["corpus"], unit["backbone"], progress=progress)
        )
    return results
