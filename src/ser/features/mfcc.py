"""MFCC features: 13 coefficients + delta + delta-delta, mean- and std-pooled.

13 x 3 = 39 channels, pooled two ways = 78 dimensions. The paper may use the
mean half alone; std-pooling is nearly free and frequently helps, so both are
stored and the choice stays downstream.

Layout of the 78-vector is fixed and documented so a later slice cannot silently
take the wrong half:

    [ 0:13]  mean of base MFCC
    [13:26]  mean of delta
    [26:39]  mean of delta-delta
    [39:52]  std  of base MFCC
    [52:65]  std  of delta
    [65:78]  std  of delta-delta
"""

from __future__ import annotations

import numpy as np

__all__ = ["MFCC_DIM", "mean_slice", "std_slice", "extract_mfcc"]

MFCC_DIM = 78


def mean_slice(n_coefficients: int = 13) -> slice:
    """The mean-pooled half."""
    return slice(0, 3 * n_coefficients)


def std_slice(n_coefficients: int = 13) -> slice:
    """The std-pooled half."""
    return slice(3 * n_coefficients, 6 * n_coefficients)


def extract_mfcc(waveform: np.ndarray, config) -> np.ndarray:
    """Return the 78-dimensional pooled MFCC vector for one utterance."""
    import librosa

    n = config.features.mfcc_n_coefficients
    base = librosa.feature.mfcc(
        y=np.asarray(waveform, dtype=np.float32),
        sr=config.features.sample_rate,
        n_mfcc=n,
    )

    if config.features.mfcc_deltas:
        # librosa's delta needs at least 9 frames by default; very short clips
        # would otherwise raise. Widths must be odd and <= the frame count.
        width = min(9, base.shape[1] if base.shape[1] % 2 == 1 else base.shape[1] - 1)
        if width >= 3:
            delta = librosa.feature.delta(base, order=1, width=width)
            delta2 = librosa.feature.delta(base, order=2, width=width)
        else:
            # Too short for a finite difference; zeros are honest here.
            delta = np.zeros_like(base)
            delta2 = np.zeros_like(base)
        stacked = np.concatenate([base, delta, delta2], axis=0)
    else:
        stacked = base

    parts = []
    for pooling in config.features.mfcc_pooling:
        if pooling == "mean":
            parts.append(stacked.mean(axis=1))
        elif pooling == "std":
            parts.append(stacked.std(axis=1))
        else:  # pragma: no cover - config validation rejects this
            raise ValueError(f"unknown pooling {pooling!r}")

    return np.concatenate(parts).astype(np.float32)
