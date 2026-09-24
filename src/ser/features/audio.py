"""The fixed preprocessing contract.

Mono, resample to 16 kHz, peak normalise. Identical for every corpus and every
backbone, and recorded in each cache's metadata.

Explicitly **not** done here: standardisation. Whether features are z-scored is
an experimental condition in Phase 5, not a preprocessing default, and baking it
in at extraction time would make the `none` alignment rung unmeasurable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_audio", "peak_normalise", "warm_up_audio_stack"]

_WARMED_UP = False


def warm_up_audio_stack() -> None:
    """Initialise librosa's numba/scipy path before torch is ever imported.

    On this platform (conda numpy+MKL alongside a pip torch) both ship their own
    ``libiomp5md.dll``. If torch's OpenMP initialises first, the first call into
    ``librosa.feature.mfcc`` aborts the process with "OMP: Error #15". Initialise
    them the other way round and both coexist for the rest of the run, including
    MFCC calls made after torch is loaded.

    This is import ordering only -- it changes no numerical result. The
    documented alternative, ``KMP_DUPLICATE_LIB_OK=TRUE``, is explicitly
    described by Intel as able to "silently produce incorrect results", which is
    not a trade this project can make.

    Call once, before anything touches torch. Idempotent and costs ~1 second.
    """
    global _WARMED_UP
    if _WARMED_UP:
        return

    import librosa

    # Exercise the *whole* MFCC path, not just the transform. librosa's delta
    # goes through scipy.signal, which links its own OpenMP separately -- a
    # warm-up that skips it leaves exactly that library to initialise later,
    # after torch, and the abort still happens. One second of audio gives enough
    # frames for the default delta width.
    buffer = np.zeros(16000, dtype=np.float32)
    base = librosa.feature.mfcc(y=buffer, sr=16000, n_mfcc=13)
    librosa.feature.delta(base, order=1)
    librosa.feature.delta(base, order=2)
    _WARMED_UP = True


def peak_normalise(waveform: np.ndarray) -> np.ndarray:
    """Scale to unit peak. Silence is returned unchanged rather than amplified."""
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0.0:
        return (waveform / peak).astype(np.float32)
    return waveform.astype(np.float32)


def assert_target_sample_rate(sample_rate: int, config) -> None:
    """Refuse any rate other than the configured one.

    RAVDESS ships at 48 kHz and CREMA-D at 16 kHz, and every SSL backbone here
    expects 16 kHz. Feeding 48 kHz audio to the model would not error -- it would
    silently produce features for speech running at a third of its true rate,
    and the numbers would look plausible. Timing evidence says the resampling is
    correct today; this makes it structural so it cannot regress.
    """
    expected = config.features.sample_rate
    if int(sample_rate) != int(expected):
        raise ValueError(
            f"sample rate {sample_rate} != required {expected}. Audio must be "
            "resampled before feature extraction; a mismatch here would produce "
            "plausible-looking features for time-distorted speech."
        )


def load_audio(path: str | Path, config) -> np.ndarray:
    """Load one file under the fixed contract, as float32 mono at the target rate.

    ``librosa.load`` resamples to ``sr``; the assertion below is a guard against
    the config being changed to disable resampling, not against librosa.
    """
    import librosa

    target_rate = config.features.sample_rate
    if not target_rate:
        raise ValueError(
            "features.sample_rate must be set. Loading at native rate would mix "
            "48 kHz RAVDESS and 16 kHz CREMA-D in one feature space."
        )

    waveform, sample_rate = librosa.load(
        str(path), sr=target_rate, mono=config.features.mono
    )
    assert_target_sample_rate(sample_rate, config)

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if config.features.peak_normalise:
        waveform = peak_normalise(waveform)
    return waveform
