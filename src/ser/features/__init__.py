"""Feature extraction and caching (Phase 3).

Extract once, reuse forever. The design point is that **all hidden layers** are
cached, not just the last one, so layer aggregation becomes a free experimental
condition downstream instead of an unexamined default.

    ser.features.cache      cache keys, metadata, atomic writes
    ser.features.ssl        per-layer mean- and segment-pooled SSL features
    ser.features.mfcc       13 MFCC + delta + delta-delta, mean- and std-pooled
    ser.features.aggregate  layer cache + spec -> a matrix
"""

from .cache import CacheEntry, cache_key, load_entry
from .aggregate import aggregate_layers, parse_layer_spec

__all__ = [
    "CacheEntry",
    "cache_key",
    "load_entry",
    "aggregate_layers",
    "parse_layer_spec",
]
