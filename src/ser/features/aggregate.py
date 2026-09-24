"""Turn a cached layer stack into the matrix a classifier consumes.

Specs:

    last        the final hidden state -- what the original study used, kept as a
                comparison condition so the cost of that choice is measurable
    layer:k     a single hidden state, 0 = CNN output, 12 = last transformer layer
    mean:a-b    unweighted mean over an inclusive layer range
    weighted    the full stack, unreduced

``weighted`` deliberately returns ``(N, L, D)`` rather than a pooled matrix. The
softmax weights over layers are **learnable parameters owned by the classifier**,
trained jointly with the head; baking them into the cache would make them a
preprocessing constant and quietly remove the thing being measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["LayerSpec", "parse_layer_spec", "aggregate_layers", "spec_output_dim"]

_LAYER = re.compile(r"^layer:(?P<k>\d+)$")
_MEAN = re.compile(r"^mean:(?P<a>\d+)-(?P<b>\d+)$")


@dataclass(frozen=True)
class LayerSpec:
    kind: str  # "last" | "layer" | "mean" | "weighted"
    index: Optional[int] = None
    start: Optional[int] = None
    stop: Optional[int] = None  # inclusive

    @property
    def reduces(self) -> bool:
        """False for ``weighted``, which leaves reduction to the classifier."""
        return self.kind != "weighted"

    def as_row_fields(self) -> Tuple[str, Optional[int]]:
        """``(layer_agg, layer_index)`` for the result schema."""
        return self.kind, self.index


def parse_layer_spec(spec: str, n_layers: int) -> LayerSpec:
    """Parse a spec string, validating indices against the cached layer count."""
    text = spec.strip().lower()

    if text == "last":
        return LayerSpec(kind="last", index=n_layers - 1)
    if text == "weighted":
        return LayerSpec(kind="weighted")

    match = _LAYER.match(text)
    if match:
        k = int(match.group("k"))
        if not 0 <= k < n_layers:
            raise ValueError(f"layer:{k} out of range for {n_layers} layers")
        return LayerSpec(kind="layer", index=k)

    match = _MEAN.match(text)
    if match:
        a, b = int(match.group("a")), int(match.group("b"))
        if not 0 <= a <= b < n_layers:
            raise ValueError(
                f"mean:{a}-{b} out of range for {n_layers} layers, or a > b"
            )
        return LayerSpec(kind="mean", start=a, stop=b)

    raise ValueError(
        f"unrecognised layer spec {spec!r}. Expected 'last', 'layer:k', "
        "'mean:a-b', or 'weighted'."
    )


def aggregate_layers(layers: np.ndarray, spec: str | LayerSpec) -> np.ndarray:
    """Apply a spec to a cached stack.

    Args:
        layers: ``(N, L, D)`` mean-pooled, or ``(N, L, S, D)`` segment-pooled.
            Aggregation is over the layer axis either way, so the segment axis
            passes through untouched.

    Returns:
        ``(N, D)`` / ``(N, S, D)`` for reducing specs, or the input stack
        unchanged for ``weighted``.
    """
    if layers.ndim not in (3, 4):
        raise ValueError(f"expected (N, L, D) or (N, L, S, D), got {layers.shape}")

    parsed = spec if isinstance(spec, LayerSpec) else parse_layer_spec(spec, layers.shape[1])

    if parsed.kind == "weighted":
        return layers

    # float16 is a storage format, not an arithmetic one: averaging in it loses
    # precision for no saving, since the result is small.
    if parsed.kind in ("last", "layer"):
        return np.asarray(layers[:, parsed.index], dtype=np.float32)

    window = layers[:, parsed.start : parsed.stop + 1]
    return np.asarray(window, dtype=np.float32).mean(axis=1)


def spec_output_dim(spec: LayerSpec, n_layers: int, hidden_dim: int) -> int:
    """Feature width a spec yields, for sizing a classifier's input."""
    if spec.kind == "weighted":
        return hidden_dim  # after the classifier's own weighted sum
    return hidden_dim
