"""Single entry point for making a run reproducible.

Every script in this project calls :func:`set_all_seeds` exactly once, before
any random draw, and records the seed it used in the result row. Nothing else
in the codebase is permitted to call ``random.seed`` / ``np.random.seed`` /
``torch.manual_seed`` directly -- one place, so that "what was the seed?" always
has a single answer.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_all_seeds", "seed_worker"]


def set_all_seeds(seed: int, *, strict_determinism: bool = False) -> int:
    """Seed every RNG this project can reach and pin cuDNN to deterministic mode.

    Args:
        seed: Non-negative integer seed.
        strict_determinism: Additionally call
            ``torch.use_deterministic_algorithms(True)``. This makes non-
            deterministic CUDA kernels raise instead of silently varying, which
            is what you want when chasing a reproducibility failure. It is off
            by default because a handful of ops have no deterministic
            implementation and would abort an otherwise valid run.

    Returns:
        The seed, so callers can write ``seed = set_all_seeds(cfg.seed)`` and
        have the recorded value provably be the one that was applied.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    # Affects hash randomisation for str/bytes in *child* processes only; the
    # current interpreter has already fixed its hash seed. Set anyway so any
    # subprocess we spawn inherits it.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch = _try_import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic convolution algorithm selection. `benchmark=True` would
        # let cuDNN autotune per input shape, which changes results run to run.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if strict_determinism:
            # Required by CUBLAS for deterministic GEMMs on CUDA >= 10.2.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``.

    Torch gives each worker a distinct base seed derived from the parent's RNG;
    this propagates it to ``random`` and ``numpy`` inside the worker, which
    torch does not do for you.
    """
    torch = _try_import_torch()
    if torch is None:  # pragma: no cover - only reachable without torch
        return
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _try_import_torch():
    """Import torch if present.

    Phases 0-2 do no tensor work, and the leakage/label tests must be runnable
    in a bare environment, so torch is an optional import here rather than a
    hard dependency of the seeding helper.
    """
    try:
        import torch  # noqa: PLC0415 - deliberately lazy
    except ImportError:  # pragma: no cover - torch is pinned in requirements
        return None
    return torch
