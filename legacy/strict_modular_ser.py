"""Strict modular SER pipeline.

This module implements the main research pipeline used in the repository:
SSL extraction -> SSL alignment -> SSL blending -> SSL/MFCC fusion -> classifier.

Datasets are expected to exist locally and are intentionally not version-controlled.
Paths can be provided either by placing dataset folders in the repository root or by
setting `SER_DATA_ROOT` to a shared dataset directory.
"""

from __future__ import annotations

import contextlib
from collections import Counter
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModel


RANDOM_SEED = 42
DEFAULT_SR = 16000
FEATURE_CACHE_DIR = Path("./feature_cache")
RESULTS_DIR = Path("./results")
CONFUSION_DIR = RESULTS_DIR / "confusion"
RESULTS_JSON_PATH = RESULTS_DIR / "results.json"
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.getenv("SER_DATA_ROOT", str(PROJECT_ROOT))).expanduser()
# Dataset folders are resolved relative to SER_DATA_ROOT or the repository root.
DATASET_PATHS = {
    "ravdess": DATASET_ROOT / "Radvess",
    "crema": DATASET_ROOT / "Crema D",
    "crema-d": DATASET_ROOT / "Crema D",
    "cremad": DATASET_ROOT / "Crema D",
    "iemocap": DATASET_ROOT / "IEMOCAP",
    "mead": DATASET_ROOT / "MAED",
    "maed": DATASET_ROOT / "MAED",
}
LABELS_6_CLASS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
LABELS_5_CLASS = ["angry", "fear", "happy", "neutral", "sad"]
VALID_BACKBONES = ["wav2vec2", "hubert", "wavlm"]
SSL_MODEL_MAP = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "wavlm": "microsoft/wavlm-base",
}
_SSL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _get_ssl_model(model_name: str) -> Tuple[Any, Any]:
    if model_name not in VALID_BACKBONES:
        raise ValueError(
            f"Invalid backbone: {model_name}. Use wav2vec2, hubert, or wavlm."
        )

    if model_name not in _SSL_CACHE:
        hf_name = SSL_MODEL_MAP[model_name]
        print(f"[MODEL LOAD] feature extractor: {hf_name}", flush=True)
        processor = AutoFeatureExtractor.from_pretrained(hf_name, local_files_only=True)
        print(f"[MODEL LOAD] encoder: {hf_name}", flush=True)
        model = AutoModel.from_pretrained(hf_name, local_files_only=True)
        model.eval()
        _SSL_CACHE[model_name] = (processor, model)
        print(f"[MODEL READY] {hf_name}", flush=True)
    else:
        print(f"[MODEL CACHE HIT] {SSL_MODEL_MAP[model_name]}", flush=True)

    return _SSL_CACHE[model_name]


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = DEFAULT_SR) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)


def _covariance(X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")
    return (X.T @ X) / float(n_samples - 1)


def _matrix_sqrt_factors(matrix: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    U, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    singular_values = np.maximum(singular_values, eps)
    inv_sqrt = U @ np.diag(1.0 / np.sqrt(singular_values)) @ U.T
    sqrt = U @ np.diag(np.sqrt(singular_values)) @ U.T
    return inv_sqrt, sqrt


def _stable_seed_from_name(name: str) -> int:
    seed = RANDOM_SEED
    for idx, char in enumerate(name):
        seed = (seed * 131 + (idx + 1) * ord(char)) % (2**32)
    return seed


def _canonical_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_PATHS:
        raise ValueError(f"Unsupported dataset: {name}")
    if key in {"crema-d", "cremad"}:
        return "crema"
    if key == "maed":
        return "mead"
    return key


def _allowed_labels(label_mode: str) -> List[str]:
    if label_mode == "6-class":
        return LABELS_6_CLASS
    if label_mode == "5-class":
        return LABELS_5_CLASS
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _preprocess_audio_file(file_path: Path) -> np.ndarray:
    try:
        waveform, _ = librosa.load(str(file_path), sr=DEFAULT_SR, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file: {file_path}") from exc

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
    if peak > 0.0:
        waveform = waveform / peak
    return waveform.astype(np.float32)


def _normalize_label(raw_label: str, label_mode: str, dataset_name: str) -> str | None:
    label = raw_label.strip().lower()

    common_map = {
        "ang": "angry",
        "angry": "angry",
        "dis": "disgust",
        "disgust": "disgust",
        "disgusted": "disgust",
        "fea": "fear",
        "fear": "fear",
        "fearful": "fear",
        "hap": "happy",
        "happy": "happy",
        "neu": "neutral",
        "neutral": "neutral",
        "sad": "sad",
        "fru": "frustrated",
        "frustrated": "frustrated",
        "exc": "excited",
        "excited": "excited",
        "calm": "neutral",
        "sur": None,
        "surprised": None,
        "contempt": None,
        "xxx": None,
        "oth": None,
        "other": None,
    }
    label = common_map.get(label, label)

    if dataset_name == "iemocap":
        iemocap_map = {
            "excited": "happy",
            "frustrated": "angry",
            "fear": "fear",
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
            "angry": "angry",
            "disgust": None,
        }
        label = iemocap_map.get(label, label)

    if label_mode == "5-class":
        if label == "disgust":
            return None
        if label in {"excited", "frustrated"}:
            raise ValueError(
                f"Unmapped IEMOCAP label '{raw_label}' remained after normalization."
            )

    allowed = set(_allowed_labels(label_mode))
    if label not in allowed:
        return None
    return label


def _speaker_split(
    records: List[Dict],
    dataset_name: str,
    train_ratio: float = 0.8,
) -> Tuple[List[Dict], List[Dict]]:
    speakers = sorted({record["speaker_id"] for record in records})
    if len(speakers) < 2:
        raise ValueError(f"{dataset_name} needs at least two speakers for speaker split.")

    best_split: Tuple[List[Dict], List[Dict]] | None = None
    expected_labels = {record["label"] for record in records}
    n_train = min(max(1, int(round(len(speakers) * train_ratio))), len(speakers) - 1)

    for offset in range(64):
        rng = np.random.default_rng(_stable_seed_from_name(f"{dataset_name}:{offset}"))
        shuffled = speakers.copy()
        rng.shuffle(shuffled)
        train_speakers = set(shuffled[:n_train])
        test_speakers = set(shuffled[n_train:])
        train_records = [record for record in records if record["speaker_id"] in train_speakers]
        test_records = [record for record in records if record["speaker_id"] in test_speakers]

        if not train_records or not test_records:
            continue

        train_labels = {record["label"] for record in train_records}
        test_labels = {record["label"] for record in test_records}
        if train_labels == expected_labels and test_labels == expected_labels:
            best_split = (train_records, test_records)
            break
        if best_split is None:
            best_split = (train_records, test_records)

    if best_split is None:
        raise RuntimeError(f"Could not create a valid speaker split for {dataset_name}.")

    train_records, test_records = best_split
    train_speakers = {record["speaker_id"] for record in train_records}
    test_speakers = {record["speaker_id"] for record in test_records}
    if train_speakers & test_speakers:
        raise AssertionError(f"Speaker overlap detected in {dataset_name} split.")
    return train_records, test_records


def _print_dataset_summary(dataset_name: str, labels: List[str]) -> None:
    distribution = dict(sorted(Counter(labels).items()))
    print(f"{dataset_name}: samples={len(labels)}")
    print(f"{dataset_name}: class_distribution={distribution}")


def _validate_split(dataset_name: str, train_records: List[Dict], test_records: List[Dict]) -> None:
    train_speakers = {record["speaker_id"] for record in train_records}
    test_speakers = {record["speaker_id"] for record in test_records}
    if train_speakers & test_speakers:
        raise AssertionError(f"Speaker overlap detected for {dataset_name}.")


def _parse_ravdess_records(label_mode: str) -> List[Dict]:
    root = DATASET_PATHS["ravdess"]
    if not root.exists():
        raise FileNotFoundError(f"RAVDESS path not found: {root}")

    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }
    records: List[Dict] = []

    for file_path in sorted(root.rglob("*.wav")):
        parts = file_path.stem.split("-")
        if len(parts) != 7:
            continue
        if parts[1] != "01":
            continue
        raw_label = emotion_map.get(parts[2])
        if raw_label is None:
            continue
        label = _normalize_label(raw_label, label_mode, "ravdess")
        if label is None:
            continue
        records.append(
            {
                "audio": _preprocess_audio_file(file_path),
                "label": label,
                "speaker_id": f"ravdess_{parts[-1]}",
                "path": str(file_path),
            }
        )

    return records


def _parse_crema_records(label_mode: str) -> List[Dict]:
    root = DATASET_PATHS["crema"]
    if not root.exists():
        raise FileNotFoundError(f"CREMA-D path not found: {root}")

    records: List[Dict] = []
    for file_path in sorted(root.rglob("*.wav")):
        parts = file_path.stem.split("_")
        if len(parts) < 3:
            continue
        label = _normalize_label(parts[2], label_mode, "crema")
        if label is None:
            continue
        speaker_part = parts[0]
        if not speaker_part.isdigit():
            continue
        records.append(
            {
                "audio": _preprocess_audio_file(file_path),
                "label": label,
                "speaker_id": f"crema_{speaker_part}",
                "path": str(file_path),
            }
        )

    return records


def _parse_iemocap_annotations(annotation_file: Path, label_mode: str) -> List[Dict]:
    pattern = re.compile(r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+(?P<utt>\S+)\s+(?P<label>\w+)\s+\[")
    utterances: List[Dict] = []

    for line in annotation_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        utterance_id = match.group("utt")
        label = _normalize_label(match.group("label"), label_mode, "iemocap")
        if label is None:
            continue
        dialog_id = "_".join(utterance_id.split("_")[:-1])
        audio_path = annotation_file.parents[2] / "sentences" / "wav" / dialog_id / f"{utterance_id}.wav"
        if not audio_path.exists():
            continue
        speaker_id = utterance_id.split("_")[0]
        utterances.append(
            {
                "audio": _preprocess_audio_file(audio_path),
                "label": label,
                "speaker_id": f"iemocap_{speaker_id}",
                "path": str(audio_path),
            }
        )

    return utterances


def _parse_iemocap_records(label_mode: str) -> List[Dict]:
    if label_mode != "5-class":
        raise ValueError("IEMOCAP is supported only in 5-class label mode.")

    root = DATASET_PATHS["iemocap"]
    if not root.exists():
        raise FileNotFoundError(f"IEMOCAP path not found: {root}")

    records: List[Dict] = []
    for annotation_file in sorted(root.glob("Session*/dialog/EmoEvaluation/*.txt")):
        records.extend(_parse_iemocap_annotations(annotation_file, label_mode))
    return records


def _parse_mead_records(label_mode: str) -> List[Dict]:
    root = DATASET_PATHS["mead"]
    if not root.exists():
        raise FileNotFoundError(f"MEAD path not found: {root}")

    records: List[Dict] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in {".wav", ".m4a"}:
            continue
        parts = file_path.relative_to(root).parts
        if len(parts) < 4:
            continue
        speaker_id = parts[0]
        raw_label = parts[1]
        label = _normalize_label(raw_label, label_mode, "mead")
        if label is None:
            continue
        records.append(
            {
                "audio": _preprocess_audio_file(file_path),
                "label": label,
                "speaker_id": f"mead_{speaker_id}",
                "path": str(file_path),
            }
        )

    return records


def _load_records(dataset_name: str, label_mode: str) -> List[Dict]:
    dataset_name = _canonical_dataset_name(dataset_name)
    if dataset_name == "ravdess":
        records = _parse_ravdess_records(label_mode)
    elif dataset_name == "crema":
        records = _parse_crema_records(label_mode)
    elif dataset_name == "iemocap":
        records = _parse_iemocap_records(label_mode)
    elif dataset_name == "mead":
        records = _parse_mead_records(label_mode)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not records:
        raise RuntimeError(f"No valid samples found for dataset={dataset_name}, label_mode={label_mode}")
    return records


def encode_labels(
    src_labels: List[str],
    tgt_labels: List[str],
    label_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    allowed = _allowed_labels(label_mode)
    label_to_int = {label: idx for idx, label in enumerate(allowed)}
    unknown_src = sorted(set(src_labels) - set(label_to_int))
    unknown_tgt = sorted(set(tgt_labels) - set(label_to_int))
    if unknown_src or unknown_tgt:
        raise ValueError(
            f"Unknown labels encountered. src={unknown_src}, tgt={unknown_tgt}, mode={label_mode}"
        )
    src_encoded = np.array([label_to_int[label] for label in src_labels], dtype=np.int64)
    tgt_encoded = np.array([label_to_int[label] for label in tgt_labels], dtype=np.int64)
    return src_encoded, tgt_encoded, label_to_int


def load_cross_domain(
    src_name: str,
    tgt_name: str,
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str], str]:
    src_name = _canonical_dataset_name(src_name)
    tgt_name = _canonical_dataset_name(tgt_name)
    label_mode = "5-class" if "iemocap" in {src_name, tgt_name} else "6-class"

    src_train_audio, src_train_labels, _, _ = load_dataset(src_name, label_mode)
    _, _, tgt_test_audio, tgt_test_labels = load_dataset(tgt_name, label_mode)

    src_unique = sorted(set(src_train_labels))
    tgt_unique = sorted(set(tgt_test_labels))

    print(f"cross_domain: src={src_name}, tgt={tgt_name}, label_mode={label_mode}")
    _print_dataset_summary(f"{src_name}_train", src_train_labels)
    _print_dataset_summary(f"{tgt_name}_test", tgt_test_labels)

    if src_unique != tgt_unique:
        raise AssertionError(f"Label mismatch between src and tgt: {src_unique} vs {tgt_unique}")
    assert len(src_unique) == len(tgt_unique)

    return src_train_audio, src_train_labels, tgt_test_audio, tgt_test_labels, label_mode


def extract_ssl(audio: np.ndarray, sr: int, model_name: str) -> np.ndarray:
    processor, model = _get_ssl_model(model_name)
    waveform = _resample_if_needed(audio, sr, DEFAULT_SR)
    device = next(model.parameters()).device

    inputs = processor(waveform, sampling_rate=DEFAULT_SR, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        frame_embeddings = outputs.last_hidden_state[0]
        pooled = frame_embeddings.mean(dim=0)

    pooled_np = pooled.detach().cpu().numpy().astype(np.float32)
    if pooled_np.shape != (768,):
        raise ValueError(
            f"Expected SSL feature shape (768,) for model {model_name}, got {pooled_np.shape}"
        )
    return pooled_np


def extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    base_mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(base_mfcc, order=1)
    delta2 = librosa.feature.delta(base_mfcc, order=2)
    mfcc_39 = np.concatenate([base_mfcc, delta, delta2], axis=0)
    pooled = mfcc_39.mean(axis=1)
    return pooled.astype(np.float32)


def align_ssl(X_src: np.ndarray, X_tgt: np.ndarray, method: str) -> np.ndarray:
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)

    if method == "none":
        return X_src.astype(np.float32)
    if X_src.ndim != 2 or X_tgt.ndim != 2:
        raise ValueError("SSL alignment expects 2D feature matrices.")
    if X_src.shape[1] != X_tgt.shape[1]:
        raise ValueError("Source and target SSL features must have matching dimensions.")

    if method == "mmd":
        mu_src = X_src.mean(axis=0, keepdims=True)
        mu_tgt = X_tgt.mean(axis=0, keepdims=True)
        X_aligned = X_src + (mu_tgt - mu_src)
        return X_aligned.astype(np.float32)

    if method == "sa":
        from sklearn.decomposition import PCA

        d = X_src.shape[1]
        k = min(100, d, X_src.shape[0], X_tgt.shape[0])
        if k < 1:
            raise ValueError("SA requires at least one sample in both source and target.")

        pca_src = PCA(n_components=k, svd_solver="full")
        pca_tgt = PCA(n_components=k, svd_solver="full")
        pca_src.fit(X_src)
        pca_tgt.fit(X_tgt)

        U_s = pca_src.components_.T
        U_t = pca_tgt.components_.T
        M = U_s.T @ U_t
        X_aligned = X_src @ U_s @ M @ U_t.T
        return X_aligned.astype(np.float32)

    if method != "coral":
        raise ValueError(f"Unsupported alignment method: {method}")

    eps = 1e-5
    mean_src = X_src.mean(axis=0, keepdims=True)
    mean_tgt = X_tgt.mean(axis=0, keepdims=True)
    X_src_centered = X_src - mean_src
    X_tgt_centered = X_tgt - mean_tgt

    identity = np.eye(X_src.shape[1], dtype=np.float64)
    cov_src = _covariance(X_src_centered) + eps * identity
    cov_tgt = _covariance(X_tgt_centered) + eps * identity

    cov_src_inv_sqrt, _ = _matrix_sqrt_factors(cov_src, eps=eps)
    _, cov_tgt_sqrt = _matrix_sqrt_factors(cov_tgt, eps=eps)

    X_aligned = X_src_centered @ cov_src_inv_sqrt @ cov_tgt_sqrt
    X_aligned = X_aligned + mean_tgt
    return X_aligned.astype(np.float32)


def blend_ssl(
    X_orig: np.ndarray,
    X_aligned: np.ndarray,
    method: str,
    alpha: float = None,
) -> np.ndarray:
    X_orig = np.asarray(X_orig, dtype=np.float32)
    X_aligned = np.asarray(X_aligned, dtype=np.float32)

    if method == "none":
        return X_orig
    if X_orig.shape != X_aligned.shape:
        raise ValueError("Original and aligned SSL features must have the same shape.")

    if method == "scalar":
        if alpha is None:
            raise ValueError("alpha must be provided for scalar blending.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        return alpha * X_aligned + (1.0 - alpha) * X_orig

    if method == "fwaa":
        diff = np.abs(X_aligned - X_orig)
        alpha_feature = diff / (diff.max(axis=0, keepdims=True) + 1e-8)
        alpha_feature = np.clip(alpha_feature, 0.0, 1.0).astype(np.float32)
        return alpha_feature * X_aligned + (1.0 - alpha_feature) * X_orig

    if method == "gaa":
        n_samples, n_features = X_orig.shape
        n_groups = 16
        group_edges = np.linspace(0, n_features, num=n_groups + 1, dtype=int)
        alpha_group = np.zeros((n_features,), dtype=np.float32)
        diff = np.abs(X_aligned - X_orig)
        group_diffs = np.zeros((n_groups,), dtype=np.float32)

        for group_idx in range(n_groups):
            start = group_edges[group_idx]
            end = group_edges[group_idx + 1]
            if end <= start:
                continue
            group_diffs[group_idx] = float(diff[:, start:end].mean())

        denom = float(group_diffs.max()) + 1e-8
        for group_idx in range(n_groups):
            start = group_edges[group_idx]
            end = group_edges[group_idx + 1]
            if end <= start:
                continue
            alpha_group[start:end] = group_diffs[group_idx] / denom

        alpha_group = np.clip(alpha_group, 0.0, 1.0).reshape(1, n_features)
        alpha_group = np.broadcast_to(alpha_group, (n_samples, n_features)).astype(np.float32)
        return alpha_group * X_aligned + (1.0 - alpha_group) * X_orig

    raise ValueError(f"Unsupported blending method: {method}")


def fuse_features(X_ssl: np.ndarray, X_mfcc: np.ndarray) -> np.ndarray:
    return np.concatenate([X_ssl, X_mfcc], axis=-1)


class _MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _APLinHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.attn = nn.Linear(input_dim, input_dim)
        self.cls = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=-1)
        x_weighted = x * weights
        return self.cls(x_weighted)


class _TransformerHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        seq_len = 16
        token_dim = int(np.ceil(input_dim / seq_len))
        padded_dim = seq_len * token_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=1,
            dim_feedforward=max(128, token_dim * 2),
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.input_dim = input_dim
        self.padded_dim = padded_dim
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Linear(padded_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padded_dim > self.input_dim:
            pad_width = self.padded_dim - self.input_dim
            x = torch.nn.functional.pad(x, (0, pad_width))
        tokens = x.view(x.shape[0], self.seq_len, self.token_dim)
        encoded = self.encoder(tokens)
        flattened = encoded.reshape(x.shape[0], -1)
        return self.cls(flattened)


def _train_torch_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> tuple[float, float, np.ndarray]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    input_dim = X_train.shape[1]
    num_classes = int(np.max(y_train)) + 1

    if model_name == "mlp":
        model = _MLPHead(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "aplin":
        model = _APLinHead(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "transformer":
        model = _TransformerHead(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported torch classifier: {model_name}")

    train_inputs = torch.from_numpy(X_train_scaled)
    train_targets = torch.from_numpy(y_train)
    test_inputs = torch.from_numpy(X_test_scaled)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(8):
        optimizer.zero_grad()
        logits = model(train_inputs)
        loss = criterion(logits, train_targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(test_inputs).argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    return accuracy, macro_f1, predictions


def train_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
):
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    elif model_name == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_SEED)
    elif model_name in {"mlp", "aplin", "transformer"}:
        return _train_torch_classifier(X_train, y_train, X_test, y_test, model_name)
    else:
        raise ValueError(f"Unsupported classifier: {model_name}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    return accuracy, macro_f1, predictions


def load_dataset(name: str, label_mode: str = "6-class"):
    dataset_name = _canonical_dataset_name(name)
    records = _load_records(dataset_name, label_mode)
    train_records, test_records = _speaker_split(records, dataset_name)
    _validate_split(dataset_name, train_records, test_records)

    X_audio_train = [record["audio"] for record in train_records]
    y_train = [record["label"] for record in train_records]
    X_audio_test = [record["audio"] for record in test_records]
    y_test = [record["label"] for record in test_records]

    _print_dataset_summary(f"{dataset_name}_train", y_train)
    _print_dataset_summary(f"{dataset_name}_test", y_test)
    return X_audio_train, y_train, X_audio_test, y_test


def _extract_feature_matrix(
    audio_batch: List[np.ndarray],
    extractor,
    desc: str | None = None,
    **kwargs,
) -> np.ndarray:
    if desc is not None:
        print(f"[FEATURES] {desc}: {len(audio_batch)} samples", flush=True)
    iterator = audio_batch
    if desc is not None:
        iterator = tqdm(audio_batch, desc=desc, leave=True)
    features = [extractor(audio, DEFAULT_SR, **kwargs) for audio in iterator]
    if desc is not None:
        print(f"[FEATURES DONE] {desc}: shape=({len(features)}, {features[0].shape[0]})", flush=True)
    return np.vstack(features).astype(np.float32)


def _dataset_feature_cache_path(dataset_name: str, backbone: str, label_mode: str) -> Path:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_dataset = _canonical_dataset_name(dataset_name)
    safe_backbone = backbone.strip().lower()
    return FEATURE_CACHE_DIR / f"{safe_dataset}_{safe_backbone}.npz"


def _load_or_compute_dataset_features(
    dataset_name: str,
    label_mode: str,
    backbone: str,
) -> Dict[str, np.ndarray]:
    cache_path = _dataset_feature_cache_path(dataset_name, backbone, label_mode)
    if cache_path.exists():
        print(f"[CACHE HIT] {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        payload = {
            "X_ssl_train": cached["X_ssl_train"].astype(np.float32),
            "X_ssl_test": cached["X_ssl_test"].astype(np.float32),
            "X_mfcc_train": cached["X_mfcc_train"].astype(np.float32),
            "X_mfcc_test": cached["X_mfcc_test"].astype(np.float32),
            "y_train": cached["y_train"].astype(str),
            "y_test": cached["y_test"].astype(str),
        }
        assert payload["X_ssl_train"].shape[0] == len(payload["y_train"])
        assert payload["X_ssl_test"].shape[0] == len(payload["y_test"])
        return payload

    print(f"[CACHE MISS] {cache_path}")
    X_audio_train, y_train, X_audio_test, y_test = load_dataset(dataset_name, label_mode)
    X_ssl_train = _extract_feature_matrix(
        X_audio_train,
        extract_ssl,
        desc=f"{dataset_name} train SSL [{backbone}]",
        model_name=backbone,
    )
    X_ssl_test = _extract_feature_matrix(
        X_audio_test,
        extract_ssl,
        desc=f"{dataset_name} test SSL [{backbone}]",
        model_name=backbone,
    )
    X_mfcc_train = _extract_feature_matrix(
        X_audio_train,
        extract_mfcc,
        desc=f"{dataset_name} train MFCC",
    )
    X_mfcc_test = _extract_feature_matrix(
        X_audio_test,
        extract_mfcc,
        desc=f"{dataset_name} test MFCC",
    )

    payload = {
        "X_ssl_train": X_ssl_train.astype(np.float32),
        "X_ssl_test": X_ssl_test.astype(np.float32),
        "X_mfcc_train": X_mfcc_train.astype(np.float32),
        "X_mfcc_test": X_mfcc_test.astype(np.float32),
        "y_train": np.asarray(y_train, dtype=str),
        "y_test": np.asarray(y_test, dtype=str),
    }
    np.savez(cache_path, **payload)
    print(f"[CACHE SAVED] {cache_path}", flush=True)
    assert payload["X_ssl_train"].shape[0] == len(payload["y_train"])
    assert payload["X_ssl_test"].shape[0] == len(payload["y_test"])
    return payload


def _subset_split(
    X_ssl: np.ndarray,
    X_mfcc: np.ndarray,
    y: np.ndarray,
    limit: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if limit is None:
        return X_ssl, X_mfcc, y
    limit = min(limit, len(y))
    return X_ssl[:limit], X_mfcc[:limit], y[:limit]


def run_experiment(config: dict):
    backbone = config["backbone"]
    if backbone not in VALID_BACKBONES:
        raise ValueError(
            f"Invalid backbone: {backbone}. Use wav2vec2, hubert, or wavlm."
        )

    label_mode = "5-class" if "iemocap" in {
        _canonical_dataset_name(config["src_dataset"]),
        _canonical_dataset_name(config["tgt_dataset"]),
    } else "6-class"
    ultra_debug = bool(config.get("ultra_debug"))
    debug_limit = 200 if config.get("debug_subset") else None

    if ultra_debug:
        silent_stream = io.StringIO()
        with contextlib.redirect_stdout(silent_stream):
            X_audio_src_train, y_src_train_raw, _, _ = load_dataset(
                config["src_dataset"],
                label_mode,
            )
            X_audio_tgt_train, y_tgt_train_raw, X_audio_tgt_test, y_tgt_test_raw = load_dataset(
                config["tgt_dataset"],
                label_mode,
            )

        X_audio_src_train = X_audio_src_train[:50]
        y_src_train_raw = np.asarray(y_src_train_raw[:50], dtype=str)
        X_audio_tgt_train = X_audio_tgt_train[:50]
        y_tgt_train_raw = np.asarray(y_tgt_train_raw[:50], dtype=str)
        X_audio_tgt_test = X_audio_tgt_test[:50]
        y_tgt_test_raw = np.asarray(y_tgt_test_raw[:50], dtype=str)

        X_ssl_src_train = _extract_feature_matrix(
            X_audio_src_train,
            extract_ssl,
            desc=f"{config['src_dataset']} ultra_debug train SSL [{backbone}]",
            model_name=backbone,
        )
        X_mfcc_src_train = _extract_feature_matrix(
            X_audio_src_train,
            extract_mfcc,
            desc=f"{config['src_dataset']} ultra_debug train MFCC",
        )
        X_ssl_tgt_train = _extract_feature_matrix(
            X_audio_tgt_train,
            extract_ssl,
            desc=f"{config['tgt_dataset']} ultra_debug train SSL [{backbone}]",
            model_name=backbone,
        )
        X_ssl_tgt_test = _extract_feature_matrix(
            X_audio_tgt_test,
            extract_ssl,
            desc=f"{config['tgt_dataset']} ultra_debug test SSL [{backbone}]",
            model_name=backbone,
        )
        X_mfcc_tgt_test = _extract_feature_matrix(
            X_audio_tgt_test,
            extract_mfcc,
            desc=f"{config['tgt_dataset']} ultra_debug test MFCC",
        )
        y_src_train = y_src_train_raw
        y_tgt_test = y_tgt_test_raw
        X_ssl_tgt_all = np.vstack([X_ssl_tgt_train, X_ssl_tgt_test]).astype(np.float32)
    else:
        src_features = _load_or_compute_dataset_features(
            config["src_dataset"],
            label_mode,
            backbone,
        )
        tgt_features = _load_or_compute_dataset_features(
            config["tgt_dataset"],
            label_mode,
            backbone,
        )

        X_ssl_src_train, X_mfcc_src_train, y_src_train = _subset_split(
            src_features["X_ssl_train"],
            src_features["X_mfcc_train"],
            src_features["y_train"],
            debug_limit,
        )
        X_ssl_tgt_test, X_mfcc_tgt_test, y_tgt_test = _subset_split(
            tgt_features["X_ssl_test"],
            tgt_features["X_mfcc_test"],
            tgt_features["y_test"],
            debug_limit,
        )

        X_ssl_tgt_all = np.vstack(
            [
                tgt_features["X_ssl_train"],
                tgt_features["X_ssl_test"],
            ]
        ).astype(np.float32)
        if debug_limit is not None:
            X_ssl_tgt_all = X_ssl_tgt_all[: min(debug_limit, X_ssl_tgt_all.shape[0])]

    X_src_ssl_aligned = align_ssl(
        X_ssl_src_train,
        X_ssl_tgt_all,
        method=config["alignment"],
    )

    if not ultra_debug:
        print("CHECK ALIGNMENT:")
        print("orig mean:", float(X_ssl_src_train.mean()))
        print("aligned mean:", float(X_src_ssl_aligned.mean()))

    if config["blending"] == "none":
        X_src_ssl_final = X_src_ssl_aligned.astype(np.float32)
    else:
        X_src_ssl_final = blend_ssl(
            X_ssl_src_train,
            X_src_ssl_aligned,
            method=config["blending"],
            alpha=config.get("alpha"),
        )

    X_train = fuse_features(X_src_ssl_final, X_mfcc_src_train)
    X_test = fuse_features(X_ssl_tgt_test, X_mfcc_tgt_test)

    if ultra_debug:
        y_tgt_train = np.asarray(y_tgt_train_raw, dtype=str)
    else:
        y_tgt_train = np.asarray(tgt_features["y_train"], dtype=str)

    y_src_train_raw = np.asarray(y_src_train, dtype=str)
    y_tgt_train_raw = np.asarray(y_tgt_train, dtype=str)
    y_tgt_test_raw = np.asarray(y_tgt_test, dtype=str)

    all_labels = sorted(set(y_src_train_raw) | set(y_tgt_train_raw) | set(y_tgt_test_raw))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    print("LABEL CHECK:")
    print("src labels:", sorted(set(y_src_train_raw)))
    print("tgt labels:", sorted(set(y_tgt_test_raw)))
    print("encoder classes:", list(label_encoder.classes_))

    valid_labels = set(label_encoder.classes_)

    src_mask = np.array([label in valid_labels for label in y_src_train_raw], dtype=bool)
    X_train = X_train[src_mask]
    y_src_train_raw = y_src_train_raw[src_mask]

    test_mask = np.array([label in valid_labels for label in y_tgt_test_raw], dtype=bool)
    X_test = X_test[test_mask]
    y_tgt_test_raw = y_tgt_test_raw[test_mask]

    y_train = label_encoder.transform(y_src_train_raw)
    y_test = label_encoder.transform(y_tgt_test_raw)

    accuracy, macro_f1, predictions = train_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        model_name=config["classifier"],
    )
    print(
        f"[CLASSIFIER DONE] {config['classifier']} | "
        f"acc={float(accuracy):.4f} | f1={float(macro_f1):.4f}",
        flush=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    confusion_name = (
        f"{config['src_dataset']}_{config['tgt_dataset']}_{config['backbone']}_"
        f"{config['alignment']}_{config['blending']}_{config['classifier']}.png"
    )
    plt.savefig(CONFUSION_DIR / confusion_name)
    plt.close()

    result_entry = {
        "src": config["src_dataset"],
        "tgt": config["tgt_dataset"],
        "backbone": config["backbone"],
        "alignment": config["alignment"],
        "blending": config["blending"],
        "alpha": config.get("alpha"),
        "classifier": config["classifier"],
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }
    if RESULTS_JSON_PATH.exists():
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as handle:
            existing_results = json.load(handle)
        if isinstance(existing_results, dict):
            existing_results = [existing_results]
        elif not isinstance(existing_results, list):
            raise TypeError(
                f"Unsupported JSON root type in {RESULTS_JSON_PATH}: "
                f"{type(existing_results).__name__}"
            )
    else:
        existing_results = []
    existing_results.append(result_entry)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(existing_results, handle, indent=2)

    print(
        "RUN COMPLETE:",
        config["src_dataset"],
        "→",
        config["tgt_dataset"],
        "|",
        config["backbone"],
        "|",
        config["alignment"],
        "|",
        config["blending"],
        "|",
        config["classifier"],
        "| F1:",
        float(macro_f1),
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "predictions": label_encoder.inverse_transform(predictions).tolist(),
    }


if __name__ == "__main__":
    src_audio, src_labels, tgt_audio, tgt_labels, mode = load_cross_domain("ravdess", "crema")
    print(f"source_samples={len(src_audio)}")
    print(f"target_samples={len(tgt_audio)}")
    print(f"label_mode={mode}")
    print(f"source_distribution={dict(sorted(Counter(src_labels).items()))}")
    print(f"target_distribution={dict(sorted(Counter(tgt_labels).items()))}")
