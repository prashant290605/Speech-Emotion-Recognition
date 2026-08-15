"""Load cached features for a set of utterances, in a specified order.

The cache is stored in manifest order; a split is an arbitrary subset in its own
order. This module is the single place those are reconciled, and it does so by
**utterance id**, never by position. Positional indexing would work right up
until a filter changed, then silently misalign features from labels.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .aggregate import aggregate_layers
from .cache import cache_key, entry_path, load_entry
from .extract import MFCC_BACKBONE, rows_for

__all__ = ["load_features", "FeatureLoader"]


class FeatureLoader:
    """Cached lookup from utterance id to feature row, for one corpus/backbone."""

    def __init__(self, config, corpus: str, backbone: str, manifest_rows) -> None:
        corpus_rows = rows_for(manifest_rows, corpus)
        key = cache_key(corpus_rows, backbone, config.features.feature_version)
        path = entry_path(config.resolve(config.paths.cache_dir), corpus, backbone, key)
        if not path.exists():
            raise FileNotFoundError(
                f"no cache for {corpus}/{backbone} at {path}. Run `ser extract` first."
            )
        self.entry = load_entry(path)
        self.corpus = corpus
        self.backbone = backbone
        self.index: Dict[str, int] = {
            uid: i for i, uid in enumerate(self.entry.utterance_ids)
        }

    def rows_for_ids(self, utterance_ids: Sequence[str]) -> np.ndarray:
        missing = [uid for uid in utterance_ids if uid not in self.index]
        if missing:
            raise KeyError(
                f"{self.corpus}/{self.backbone}: {len(missing)} utterance(s) not in "
                f"the cache, e.g. {missing[:3]}. Cache and manifest are out of sync."
            )
        return np.asarray([self.index[uid] for uid in utterance_ids], dtype=np.int64)

    def load(
        self,
        utterance_ids: Sequence[str],
        *,
        layer_spec: str = "last",
        segments: bool = False,
    ) -> np.ndarray:
        """Features for these ids, in the order given, as float64.

        Shapes, by ``(layer_spec, segments)``:

            ('last'|'layer:k', False)  (n, d)          pooled vector
            ('last'|'layer:k', True)   (n, S, d)       segment sequence
            ('weighted',       False)  (n, L, d)       unreduced layer stack
            ('weighted',       True)   (n, L, S, d)    layers x segments

        ``weighted`` returns the stack unreduced because the softmax over layers
        is a parameter of the classifier, not of the cache.
        """
        indices = self.rows_for_ids(utterance_ids)

        if self.backbone == MFCC_BACKBONE:
            if segments or layer_spec != "last":
                raise ValueError("the MFCC cache has no layer or segment axis")
            matrix = np.asarray(self.entry.array("mfcc"))[indices]
            return np.asarray(matrix, dtype=np.float64)

        name = "segments" if segments else "layers"
        if segments and not self.entry.has("segments"):
            raise FileNotFoundError(
                f"{self.corpus}/{self.backbone} has no segment cache; the "
                "Transformer family requires features.segment_pooling_enabled"
            )
        stack = np.asarray(self.entry.array(name))[indices]
        aggregated = aggregate_layers(stack, layer_spec)
        # float64 before anything touches a covariance. See ser.numerics.
        return np.asarray(aggregated, dtype=np.float64)


def load_features(
    config,
    manifest_rows,
    corpus: str,
    backbone: str,
    utterance_ids: Sequence[str],
    *,
    layer_spec: str = "last",
) -> np.ndarray:
    """One-shot convenience wrapper around :class:`FeatureLoader`."""
    loader = FeatureLoader(config, corpus, backbone, manifest_rows)
    return loader.load(utterance_ids, layer_spec=layer_spec)
