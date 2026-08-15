"""Per-layer SSL feature extraction.

For each utterance and **each of the 13 hidden states** (CNN feature-encoder
output plus 12 transformer layers) two poolings are produced from a single
forward pass:

    layers    (N, 13, 768)      mean over all frames
    segments  (N, 13, 8, 768)   mean within each of 8 uniform temporal segments

Both come from one pass because the pass is the expensive part; computing them
separately would double a CPU-bound job for nothing. They are stored as separate
arrays so the segment cache can be skipped or deleted independently.

Batch size is 1 by design. Batching requires padding, and a padded frame that
reaches the mean would corrupt the pooled vector -- silently, and worst for the
shortest utterances. On CPU the per-call overhead is small next to the transformer
itself, so correctness costs almost nothing here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..manifest import ManifestRow
from .audio import load_audio

__all__ = ["SSLExtractor", "segment_pool", "extract_corpus"]


def segment_pool(frames: np.ndarray, n_segments: int) -> np.ndarray:
    """Mean-pool ``(T, D)`` frames into ``(n_segments, D)``.

    Segment boundaries are uniform over time. When ``T < n_segments`` the
    boundaries are clamped so every segment still contains at least one frame;
    short utterances therefore repeat frames across segments rather than
    producing NaNs from an empty mean.
    """
    n_frames = frames.shape[0]
    if n_frames == 0:
        raise ValueError("cannot segment-pool zero frames")

    pooled = np.empty((n_segments, frames.shape[1]), dtype=np.float32)
    for i in range(n_segments):
        start = (i * n_frames) // n_segments
        end = ((i + 1) * n_frames) // n_segments
        end = max(end, start + 1)
        start = min(start, n_frames - 1)
        end = min(end, n_frames)
        pooled[i] = frames[start:end].mean(axis=0)
    return pooled


@dataclass
class SSLExtractor:
    """One loaded backbone, reused across a corpus."""

    backbone: str
    checkpoint: str
    n_layers: int
    hidden_dim: int
    n_segments: int
    model: object
    feature_extractor: object
    input_normalised: bool

    @classmethod
    def load(cls, backbone: str, config) -> "SSLExtractor":
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        checkpoint = config.features.backbones[backbone]
        feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint, output_hidden_states=True)
        model.eval()
        torch.set_grad_enabled(False)

        return cls(
            backbone=backbone,
            checkpoint=checkpoint,
            n_layers=config.features.n_layers,
            hidden_dim=config.features.hidden_dim,
            n_segments=config.features.n_segments,
            model=model,
            feature_extractor=feature_extractor,
            # Model-specific input scaling belongs to the model, not to our
            # preprocessing contract. Recorded so it is never ambiguous.
            input_normalised=bool(getattr(feature_extractor, "do_normalize", False)),
        )

    def encode(self, waveform: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(layers (13, 768), segments (13, 8, 768))`` for one utterance."""
        import torch

        inputs = self.feature_extractor(
            waveform, sampling_rate=sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(input_values=inputs.input_values)

        hidden = outputs.hidden_states  # tuple of (1, T, D), length n_layers
        if len(hidden) != self.n_layers:
            raise ValueError(
                f"{self.backbone}: expected {self.n_layers} hidden states, got {len(hidden)}. "
                "features.n_layers does not match the checkpoint."
            )

        means = np.empty((self.n_layers, self.hidden_dim), dtype=np.float32)
        segments = np.empty(
            (self.n_layers, self.n_segments, self.hidden_dim), dtype=np.float32
        )
        for index, state in enumerate(hidden):
            frames = state[0].numpy()
            if frames.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"{self.backbone}: hidden dim {frames.shape[1]} != "
                    f"configured {self.hidden_dim}"
                )
            means[index] = frames.mean(axis=0)
            segments[index] = segment_pool(frames, self.n_segments)

        return means, segments


def extract_corpus(
    rows: Sequence[ManifestRow],
    extractor: SSLExtractor,
    config,
    *,
    want_segments: bool = True,
    progress: Optional[Callable[[int, int, float], None]] = None,
    progress_every: int = 100,
) -> dict:
    """Extract every row, in manifest order.

    Ordering is the manifest's, unconditionally. `verify_cache` asserts it, and
    every downstream index into these arrays depends on it.
    """
    n = len(rows)
    layers = np.empty((n, extractor.n_layers, extractor.hidden_dim), dtype=np.float16)
    segments = (
        np.empty(
            (n, extractor.n_layers, extractor.n_segments, extractor.hidden_dim),
            dtype=np.float16,
        )
        if want_segments
        else None
    )

    started = time.perf_counter()
    for index, row in enumerate(rows):
        waveform = load_audio(row.file_path, config)
        mean_pooled, segment_pooled = extractor.encode(
            waveform, config.features.sample_rate
        )
        layers[index] = mean_pooled.astype(np.float16)
        if segments is not None:
            segments[index] = segment_pooled.astype(np.float16)

        if progress and (index + 1) % progress_every == 0:
            progress(index + 1, n, time.perf_counter() - started)

    arrays = {"layers": layers}
    if segments is not None:
        arrays["segments"] = segments
    return {"arrays": arrays, "wall_seconds": time.perf_counter() - started}
