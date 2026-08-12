from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample


RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

SIMPLIFIED_EMOTION_MAP = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": "surprise",
}

TARGET_SAMPLE_RATE = 16000
CREMAD_LABEL_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
COMMON_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def load_audio(file_path: Union[str, Path]) -> tuple[np.ndarray, int]:
    """
    Load audio from disk.

    Returns:
        signal: Audio samples as a 1D NumPy array (mono).
        sr: Original sample rate.
    """
    signal, sr = sf.read(str(file_path))
    return signal, sr


def resample_audio(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a 1D audio signal using scipy.signal.resample.
    """
    if orig_sr == target_sr:
        return signal

    target_length = int(round(len(signal) * float(target_sr) / float(orig_sr)))
    if target_length <= 0:
        return signal

    return resample(signal, target_length)


def preprocess_audio(signal: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """
    Preprocess audio:
    1) Convert to mono (safety check),
    2) Resample to 16 kHz,
    3) Normalize amplitude.
    """
    signal = np.asarray(signal)

    if signal.ndim > 1:
        # Convert multi-channel audio to mono by averaging channels.
        signal = np.mean(signal, axis=1)

    if sr != TARGET_SAMPLE_RATE:
        signal = resample_audio(signal, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    peak = np.max(np.abs(signal)) if signal.size > 0 else 0.0
    if peak > 0:
        signal = signal / peak

    return signal.astype(np.float32), sr


def extract_label(filename: str) -> str:
    """
    Extract and simplify label from RAVDESS filename.

    Format example: 03-01-05-01-01-01-01.wav
    Emotion ID is the 3rd field.
    """
    stem = Path(filename).stem
    parts = stem.split("-")

    if len(parts) < 7:
        raise ValueError(f"Invalid RAVDESS filename format: {filename}")

    emotion_id = parts[2]
    if emotion_id not in RAVDESS_EMOTION_MAP:
        raise ValueError(f"Unknown emotion id '{emotion_id}' in filename: {filename}")

    official_label = RAVDESS_EMOTION_MAP[emotion_id]
    return SIMPLIFIED_EMOTION_MAP[official_label]


def extract_speaker_id(filename: str) -> int:
    """
    Extract speaker/actor ID from RAVDESS filename.

    Format example: 03-01-05-01-01-01-24.wav
    Speaker ID is the last field.
    """
    stem = Path(filename).stem
    parts = stem.split("-")

    if len(parts) < 7:
        raise ValueError(f"Invalid RAVDESS filename format: {filename}")

    return int(parts[-1])


def build_dataset(root_path: Union[str, Path], load_audio_data: bool = True) -> List[Dict]:
    """
    Recursively load all .wav files from a RAVDESS root directory and build
    a ready-to-use dataset list.
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    dataset: List[Dict] = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() != ".wav":
            continue

        label = extract_label(file_path.name)
        speaker_id = extract_speaker_id(file_path.name)
        sample = {
            "file_path": str(file_path.resolve()),
            "label": label,
            "speaker_id": speaker_id,
        }

        if load_audio_data:
            signal, sr = load_audio(file_path)
            signal, sr = preprocess_audio(signal, sr)
            sample["signal"] = signal
            sample["sample_rate"] = sr

        dataset.append(sample)

    return dataset


def extract_label_cremad(filename: str) -> Union[str, None]:
    """
    Extract and map CREMA-D emotion label from filename.

    Format example: 1001_DFA_ANG_XX.wav
    Emotion code is the 3rd underscore-separated field.

    Returns:
        Unified label string when mapped, otherwise None.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None

    emotion_code = parts[2].upper()
    return CREMAD_LABEL_MAP.get(emotion_code)


def extract_speaker_id_cremad(filename: str) -> int:
    """
    Extract speaker ID from CREMA-D filename.

    Format example: 1001_DFA_ANG_XX.wav
    Speaker ID is the first numeric field.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 1:
        raise ValueError(f"Invalid CREMA-D filename format: {filename}")

    speaker_part = parts[0]
    if not speaker_part.isdigit():
        raise ValueError(f"Invalid CREMA-D speaker id in filename: {filename}")

    return int(speaker_part)


def build_cremad_dataset(
    root_path: Union[str, Path], load_audio_data: bool = True
) -> List[Dict]:
    """
    Recursively load all .wav files from a CREMA-D root directory and build
    a ready-to-use dataset list.

    Samples with labels outside the unified map are discarded.
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    dataset: List[Dict] = []

    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() != ".wav":
            continue

        label = extract_label_cremad(file_path.name)
        if label is None:
            continue

        speaker_id = extract_speaker_id_cremad(file_path.name)
        sample = {
            "file_path": str(file_path.resolve()),
            "label": label,
            "speaker_id": speaker_id,
        }

        if load_audio_data:
            signal, sr = load_audio(file_path)
            signal, sr = preprocess_audio(signal, sr)
            sample["signal"] = signal
            sample["sample_rate"] = sr

        dataset.append(sample)

    return dataset


def validate_common_label_set(
    dataset: List[Dict], expected_labels: List[str] = COMMON_LABELS
) -> bool:
    """
    Check whether dataset labels match the expected cross-dataset label set exactly.
    """
    dataset_labels = sorted({item["label"] for item in dataset})
    return dataset_labels == sorted(expected_labels)


def to_dataframe(dataset: List[Dict]) -> pd.DataFrame:
    """Convert dataset list to pandas DataFrame."""
    return pd.DataFrame(dataset)


def print_validation_stats(dataset: List[Dict]) -> None:
    """Print required validation stats."""
    labels = [item["label"] for item in dataset]
    speakers = [item["speaker_id"] for item in dataset]

    class_counts = Counter(labels)

    print(f"Total samples: {len(dataset)}")
    print("Class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count}")
    print(f"Number of unique speakers: {len(set(speakers))}")


if __name__ == "__main__":
    ravdess_root = Path("Radvess")
    cremad_root = Path("Crema D")
    dataset = build_cremad_dataset(cremad_root, load_audio_data=False)
    print_validation_stats(dataset)
    labels_ok = validate_common_label_set(dataset)
    print(f"Label set matches expected common set: {labels_ok}")
