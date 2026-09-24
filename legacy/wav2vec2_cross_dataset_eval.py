from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from coral import coral_align
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from ravdess_preprocessing import (
    build_cremad_dataset,
    build_dataset,
    load_audio,
    preprocess_audio,
)


RANDOM_SEED = 42
MODEL_NAME = "facebook/wav2vec2-base"
COMMON_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Caching and performance settings
FEATURES_ROOT = Path("features")
RAVDESS_FEATURE_DIR = FEATURES_ROOT / "ravdess"
CREMAD_FEATURE_DIR = FEATURES_ROOT / "crema_d"
BATCH_SIZE = 8
FORCE_RECOMPUTE = False
USE_CORAL = True
USE_HYBRID = True
CORAL_ALPHA = 0.5
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Results/logging paths
RESULTS_DIR = Path("results")
RESULTS_JSON_PATH = RESULTS_DIR / "results.json"
RESULTS_CSV_PATH = RESULTS_DIR / "results.csv"
SPLIT_INFO_PATH = RESULTS_DIR / "split_info.json"
FINAL_COMPARISON_PATH = RESULTS_DIR / "final_comparison.csv"
ALPHA_SWEEP_RESULTS_PATH = RESULTS_DIR / "alpha_sweep_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSOR: Optional[Wav2Vec2Processor] = None
WAV2VEC2_MODEL: Optional[Wav2Vec2Model] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SER wav2vec2 + MFCC experiments")
    parser.add_argument(
        "--run_mode",
        choices=["full", "debug"],
        default="full",
        help="full uses full dataset (default); debug uses a small subset.",
    )
    return parser.parse_args()


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_feature_dirs() -> None:
    RAVDESS_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    CREMAD_FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_wav2vec2() -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    global PROCESSOR, WAV2VEC2_MODEL
    if PROCESSOR is None or WAV2VEC2_MODEL is None:
        PROCESSOR = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        WAV2VEC2_MODEL = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        WAV2VEC2_MODEL.to(DEVICE)
        WAV2VEC2_MODEL.eval()
    return PROCESSOR, WAV2VEC2_MODEL


def filter_common_labels(dataset: List[Dict]) -> List[Dict]:
    allowed = set(COMMON_LABELS)
    return [sample for sample in dataset if sample["label"] in allowed]


def extract_ravdess_speaker_id_from_path(file_path: str) -> int:
    actor_folder = Path(file_path).parent.name
    if not actor_folder.lower().startswith("actor_"):
        raise ValueError(f"Invalid RAVDESS actor folder: {actor_folder}")
    speaker_part = actor_folder.split("_")[-1]
    if not speaker_part.isdigit():
        raise ValueError(f"Invalid RAVDESS actor id: {actor_folder}")
    return int(speaker_part)


def format_ravdess_dataset_for_split(dataset: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for sample in dataset:
        path = sample.get("path", sample.get("file_path"))
        if path is None:
            continue
        try:
            speaker_id = extract_ravdess_speaker_id_from_path(path)
        except Exception as exc:
            print(f"[WARN] Could not parse speaker for {path}: {exc}. Skipping.")
            continue

        row = dict(sample)
        row["path"] = path
        row["speaker"] = speaker_id
        out.append(row)
    return out


def split_dataset_by_speaker(
    dataset: List[Dict], test_size: float = 0.2, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    speaker_ids = sorted({sample["speaker"] for sample in dataset})
    if len(speaker_ids) < 2:
        raise RuntimeError("Need at least 2 speakers for speaker-independent split.")

    rng = np.random.default_rng(seed)
    shuffled = speaker_ids.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_size)))
    test_speakers = set(shuffled[:n_test])
    train_speakers = set(shuffled[n_test:])

    if not train_speakers:
        train_speakers = {shuffled[-1]}
        test_speakers = set(shuffled[:-1])

    train_dataset = [s for s in dataset if s["speaker"] in train_speakers]
    test_dataset = [s for s in dataset if s["speaker"] in test_speakers]

    if train_speakers & test_speakers:
        raise RuntimeError("Speaker overlap detected in split.")

    return train_dataset, test_dataset


def save_split_info(train_speakers: List[int], test_speakers: List[int], seed: int) -> None:
    ensure_results_dir()
    payload = {
        "seed": seed,
        "train_speakers": train_speakers,
        "test_speakers": test_speakers,
    }
    with open(SPLIT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_dataset_cache_dir(file_path: str) -> Path:
    parts_lower = [p.lower() for p in Path(file_path).parts]
    if "radvess" in parts_lower:
        return RAVDESS_FEATURE_DIR
    if "crema d" in parts_lower:
        return CREMAD_FEATURE_DIR
    return FEATURES_ROOT


def build_feature_file_path(file_path: str) -> Path:
    src = str(Path(file_path).resolve())
    src_hash = hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]
    stem = Path(file_path).stem.replace(" ", "_")
    cache_dir = get_dataset_cache_dir(file_path)
    return cache_dir / f"{src_hash}_{stem}.npy"


def save_npy_atomic(target_path: Path, array: np.ndarray) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp.npy")
    np.save(tmp_path, array.astype(np.float32))
    tmp_path.replace(target_path)


def extract_wav2vec_features(signal: np.ndarray, sr: int) -> np.ndarray:
    if sr != 16000:
        raise ValueError(f"Expected sample_rate=16000, got {sr}")

    processor, model = load_wav2vec2()
    signal = signal.astype(np.float32, copy=False)

    inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    attention_mask = (
        inputs.attention_mask.to(DEVICE) if "attention_mask" in inputs else None
    )

    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[0].detach().cpu().numpy()

    return hidden.astype(np.float32)


def extract_wav2vec_features_batch(signals: List[np.ndarray]) -> List[np.ndarray]:
    if not signals:
        return []

    processor, model = load_wav2vec2()
    safe_signals = [sig.astype(np.float32, copy=False) for sig in signals]

    inputs = processor(
        safe_signals,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(DEVICE)
    attention_mask = (
        inputs.attention_mask.to(DEVICE) if "attention_mask" in inputs else None
    )

    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = model._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )

    hidden_np = hidden_states.detach().cpu().numpy().astype(np.float32)
    if feature_attention_mask is not None:
        mask_np = feature_attention_mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = np.ones(hidden_np.shape[:2], dtype=bool)

    embeddings: List[np.ndarray] = []
    for i in range(hidden_np.shape[0]):
        embeddings.append(hidden_np[i][mask_np[i]].astype(np.float32))
    return embeddings


def get_or_compute_features(
    filepath: str,
    signal: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> Optional[np.ndarray]:
    feature_path = build_feature_file_path(filepath)
    if feature_path.exists() and not FORCE_RECOMPUTE:
        try:
            return np.load(feature_path).astype(np.float32)
        except Exception as exc:
            print(f"[WARN] Failed loading cache for {filepath}: {exc}. Recomputing.")

    try:
        if signal is None or sr is None:
            raw_signal, raw_sr = load_audio(filepath)
            signal, sr = preprocess_audio(raw_signal, raw_sr)
        embedding = extract_wav2vec_features(signal, sr)
        save_npy_atomic(feature_path, embedding)
        return embedding
    except Exception as exc:
        print(f"[WARN] Failed processing file {filepath}: {exc}. Skipping.")
        return None


def pool_features(embeddings: np.ndarray, mode: str = "mean_std") -> np.ndarray:
    mean_vec = embeddings.mean(axis=0)
    if mode == "mean":
        return mean_vec.astype(np.float32)
    if mode == "max":
        max_vec = embeddings.max(axis=0)
        return max_vec.astype(np.float32)
    if mode == "mean_std":
        std_vec = embeddings.std(axis=0)
        return np.concatenate([mean_vec, std_vec]).astype(np.float32)
    raise ValueError(f"Unsupported pooling mode: {mode}")


def _load_or_prepare_signal(sample: Dict) -> Tuple[Optional[np.ndarray], Optional[int]]:
    path = sample.get("path", sample.get("file_path"))
    if path is None:
        return None, None
    if "signal" in sample and "sample_rate" in sample:
        return sample["signal"], sample["sample_rate"]

    try:
        raw_signal, raw_sr = load_audio(path)
        signal, sr = preprocess_audio(raw_signal, raw_sr)
        return signal, sr
    except Exception as exc:
        print(f"[WARN] Corrupted or unreadable audio {path}: {exc}")
        return None, None


def prepare_features(
    dataset: List[Dict],
    pooling_mode: str = "mean_std",
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    ensure_feature_dirs()
    _ = load_wav2vec2()

    n_total = len(dataset)
    feature_rows: List[np.ndarray] = []
    labels: List[str] = []

    for start in range(0, n_total, batch_size):
        batch = dataset[start : start + batch_size]
        batch_embeddings: List[Optional[np.ndarray]] = [None] * len(batch)
        batch_labels: List[Optional[str]] = [None] * len(batch)

        compute_signals: List[np.ndarray] = []
        compute_indices: List[int] = []
        compute_paths: List[str] = []
        compute_cache_paths: List[Path] = []

        for i, sample in enumerate(batch):
            global_idx = start + i + 1
            path = sample.get("path", sample.get("file_path"))
            if path is None:
                print(f"Processing file {global_idx} / {n_total} [SKIPPED] missing path")
                continue

            label = sample["label"]
            cache_path = build_feature_file_path(path)

            if cache_path.exists() and not FORCE_RECOMPUTE:
                try:
                    emb = np.load(cache_path).astype(np.float32)
                    batch_embeddings[i] = emb
                    batch_labels[i] = label
                    print(f"Processing file {global_idx} / {n_total} [LOADED] {path}")
                    continue
                except Exception as exc:
                    print(f"[WARN] Cache read failed ({path}): {exc}. Will recompute.")

            signal, sr = _load_or_prepare_signal(sample)
            if signal is None or sr is None:
                print(f"Processing file {global_idx} / {n_total} [SKIPPED] {path}")
                continue

            compute_signals.append(signal)
            compute_indices.append(i)
            compute_paths.append(path)
            compute_cache_paths.append(cache_path)

        if compute_signals:
            try:
                computed_embeddings = extract_wav2vec_features_batch(compute_signals)
                for emb, slot_i, path_i, cache_i in zip(
                    computed_embeddings,
                    compute_indices,
                    compute_paths,
                    compute_cache_paths,
                ):
                    save_npy_atomic(cache_i, emb)
                    batch_embeddings[slot_i] = emb
                    batch_labels[slot_i] = batch[slot_i]["label"]
                    global_idx = start + slot_i + 1
                    print(f"Processing file {global_idx} / {n_total} [COMPUTED] {path_i}")
            except Exception as exc:
                print(f"[WARN] Batch compute failed: {exc}. Falling back to one-by-one.")
                for path_i, slot_i in zip(compute_paths, compute_indices):
                    global_idx = start + slot_i + 1
                    emb = get_or_compute_features(path_i)
                    if emb is None:
                        print(f"Processing file {global_idx} / {n_total} [SKIPPED] {path_i}")
                        continue
                    batch_embeddings[slot_i] = emb
                    batch_labels[slot_i] = batch[slot_i]["label"]
                    print(f"Processing file {global_idx} / {n_total} [COMPUTED] {path_i}")

        for emb, label in zip(batch_embeddings, batch_labels):
            if emb is None or label is None:
                continue
            feature_rows.append(pool_features(emb, mode=pooling_mode))
            labels.append(label)

    if not feature_rows:
        raise RuntimeError("No valid features were prepared. Check audio/cache files.")

    X = np.vstack(feature_rows).astype(np.float32)
    y = np.array(labels)
    return X, y


def extract_mfcc_features(filepath: str, n_mfcc: int = 13) -> np.ndarray:
    signal, sr = librosa.load(filepath, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T.astype(np.float32)


def prepare_mfcc_features(
    dataset: List[Dict], pooling_mode: str, n_mfcc: int = 13
) -> Tuple[np.ndarray, np.ndarray]:
    X_rows: List[np.ndarray] = []
    y_rows: List[str] = []

    n_total = len(dataset)
    for idx, sample in enumerate(dataset, start=1):
        path = sample.get("path", sample.get("file_path"))
        if path is None:
            print(f"MFCC file {idx} / {n_total} [SKIPPED] missing path")
            continue
        try:
            mfcc = extract_mfcc_features(path, n_mfcc=n_mfcc)
            pooled = pool_features(mfcc, mode=pooling_mode)
            X_rows.append(pooled)
            y_rows.append(sample["label"])
            print(f"MFCC file {idx} / {n_total} [DONE] {path}")
        except Exception as exc:
            print(f"[WARN] MFCC failed for {path}: {exc}. Skipping.")

    if not X_rows:
        raise RuntimeError("No MFCC features prepared. Check audio files.")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.array(y_rows)
    return X, y


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000, random_state=RANDOM_SEED, class_weight="balanced"
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    return acc, macro_f1, cm


def filter_to_seen_labels(
    X: np.ndarray, y_raw: np.ndarray, seen_labels: set[str]
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.array([label in seen_labels for label in y_raw], dtype=bool)
    return X[mask], y_raw[mask]


def append_result_record(
    feature_type: str,
    pooling_method: str,
    evaluation_type: str,
    accuracy: float,
    macro_f1: float,
    confusion_matrix_array: np.ndarray,
) -> None:
    ensure_results_dir()
    record = {
        "feature_type": feature_type,
        "pooling_method": pooling_method,
        "evaluation_type": evaluation_type,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix_array.tolist(),
    }

    if RESULTS_JSON_PATH.exists():
        try:
            with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "experiments" not in payload:
                payload = {"experiments": []}
        except Exception:
            payload = {"experiments": []}
    else:
        payload = {"experiments": []}

    payload["experiments"].append(record)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_exists = RESULTS_CSV_PATH.exists()
    with open(RESULTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_type",
                "pooling_method",
                "evaluation_type",
                "accuracy",
                "macro_f1",
                "confusion_matrix",
            ],
        )
        if not csv_exists:
            writer.writeheader()
        row = dict(record)
        row["confusion_matrix"] = json.dumps(record["confusion_matrix"])
        writer.writerow(row)


def export_confusion_matrix_csv(
    cm: np.ndarray, feature_type: str, pooling_method: str, eval_type: str
) -> None:
    ensure_results_dir()
    out_path = RESULTS_DIR / f"cm_{feature_type}_{pooling_method}_{eval_type}.csv"
    np.savetxt(out_path, cm.astype(int), delimiter=",", fmt="%d")


def append_final_comparison_row(
    feature_type: str,
    pooling_method: str,
    in_domain_accuracy: float,
    in_domain_f1: float,
    cross_domain_accuracy: float,
    cross_domain_f1: float,
) -> None:
    ensure_results_dir()
    csv_exists = FINAL_COMPARISON_PATH.exists()
    with open(FINAL_COMPARISON_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_type",
                "pooling_method",
                "in_domain_accuracy",
                "in_domain_f1",
                "cross_domain_accuracy",
                "cross_domain_f1",
            ],
        )
        if not csv_exists:
            writer.writeheader()
        writer.writerow(
            {
                "feature_type": feature_type,
                "pooling_method": pooling_method,
                "in_domain_accuracy": float(in_domain_accuracy),
                "in_domain_f1": float(in_domain_f1),
                "cross_domain_accuracy": float(cross_domain_accuracy),
                "cross_domain_f1": float(cross_domain_f1),
            }
        )


def main() -> None:
    args = parse_args()
    set_seed()
    ensure_feature_dirs()
    ensure_results_dir()
    print(f"Using device: {DEVICE}")
    print(f"FORCE_RECOMPUTE={FORCE_RECOMPUTE}")
    print(f"Batch size={BATCH_SIZE}")
    print(f"Run mode={args.run_mode}")

    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(cremad_raw)

    if args.run_mode == "debug":
        ravdess = ravdess[: min(120, len(ravdess))]
        cremad = cremad[: min(120, len(cremad))]

    ravdess_train, ravdess_test = split_dataset_by_speaker(
        ravdess, test_size=0.2, seed=RANDOM_SEED
    )

    train_speakers = sorted({s["speaker"] for s in ravdess_train})
    test_speakers = sorted({s["speaker"] for s in ravdess_test})
    print(f"Train speakers: {train_speakers}")
    print(f"Test speakers: {test_speakers}")
    print(f"Speaker overlap: {len(set(train_speakers) & set(test_speakers))}")
    save_split_info(train_speakers, test_speakers, RANDOM_SEED)

    pooling_methods = ["mean", "max", "mean_std"]
    wav2vec2_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    mfcc_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    alpha_sweep_results: Dict[str, List[Dict[str, float]]] = {}

    for pooling_method in pooling_methods:
        print(f"\n=== Running Pooling: {pooling_method} ===")

        X_train_in, y_train_in_raw = prepare_features(
            ravdess_train, pooling_mode=pooling_method, batch_size=BATCH_SIZE
        )
        X_test_in, y_test_in_raw = prepare_features(
            ravdess_test, pooling_mode=pooling_method, batch_size=BATCH_SIZE
        )

        label_encoder_in = LabelEncoder()
        label_encoder_in.fit(y_train_in_raw)
        seen_in = set(label_encoder_in.classes_)

        X_train_in, y_train_in_raw = filter_to_seen_labels(X_train_in, y_train_in_raw, seen_in)
        X_test_in, y_test_in_raw = filter_to_seen_labels(X_test_in, y_test_in_raw, seen_in)

        y_train_in = label_encoder_in.transform(y_train_in_raw)
        y_test_in = label_encoder_in.transform(y_test_in_raw)

        model_in = train_model(X_train_in, y_train_in)
        acc_in, f1_in, cm_in = evaluate_model(model_in, X_test_in, y_test_in)

        X_train_cross, y_train_cross_raw = prepare_features(
            ravdess, pooling_mode=pooling_method, batch_size=BATCH_SIZE
        )
        X_test_cross, y_test_cross_raw = prepare_features(
            cremad, pooling_mode=pooling_method, batch_size=BATCH_SIZE
        )

        X_w2v_train = X_train_cross
        X_w2v_test = X_test_cross

        if USE_HYBRID:
            X_mfcc_train, y_train_cross_raw_mfcc = prepare_mfcc_features(
                ravdess, pooling_mode=pooling_method
            )
            X_mfcc_test, y_test_cross_raw_mfcc = prepare_mfcc_features(
                cremad, pooling_mode=pooling_method
            )

            if not np.array_equal(y_train_cross_raw, y_train_cross_raw_mfcc):
                raise RuntimeError("Hybrid feature mismatch: wav2vec2/MFCC train labels differ.")
            if not np.array_equal(y_test_cross_raw, y_test_cross_raw_mfcc):
                raise RuntimeError("Hybrid feature mismatch: wav2vec2/MFCC test labels differ.")

        label_encoder_cross = LabelEncoder()
        label_encoder_cross.fit(y_train_cross_raw)
        seen_cross = set(label_encoder_cross.classes_)

        X_w2v_train, y_train_cross_raw = filter_to_seen_labels(
            X_w2v_train, y_train_cross_raw, seen_cross
        )
        X_w2v_test, y_test_cross_raw = filter_to_seen_labels(
            X_w2v_test, y_test_cross_raw, seen_cross
        )

        if USE_HYBRID:
            X_mfcc_train, y_train_cross_raw_mfcc = filter_to_seen_labels(
                X_mfcc_train, y_train_cross_raw_mfcc, seen_cross
            )
            X_mfcc_test, y_test_cross_raw_mfcc = filter_to_seen_labels(
                X_mfcc_test, y_test_cross_raw_mfcc, seen_cross
            )

            if not np.array_equal(y_train_cross_raw, y_train_cross_raw_mfcc):
                raise RuntimeError("Hybrid feature mismatch after filtering: train labels differ.")
            if not np.array_equal(y_test_cross_raw, y_test_cross_raw_mfcc):
                raise RuntimeError("Hybrid feature mismatch after filtering: test labels differ.")

        y_train_cross = label_encoder_cross.transform(y_train_cross_raw)
        y_test_cross = label_encoder_cross.transform(y_test_cross_raw)

        if USE_CORAL:
            print("Applying CORAL alignment...")
            print("Before CORAL:", X_w2v_train.mean(), X_w2v_train.std())
            X_w2v_aligned = coral_align(X_w2v_train, X_w2v_test)
        else:
            X_w2v_aligned = X_w2v_train

        alpha_results: List[Dict[str, float]] = []
        selected_alpha_metrics: Optional[Tuple[float, float, np.ndarray]] = None

        for alpha in ALPHAS:
            if USE_CORAL:
                X_w2v_final = alpha * X_w2v_aligned + (1.0 - alpha) * X_w2v_train
            else:
                X_w2v_final = X_w2v_train

            if USE_HYBRID:
                X_train_final = np.concatenate([X_w2v_final, X_mfcc_train], axis=1)
                X_test_final = np.concatenate([X_w2v_test, X_mfcc_test], axis=1)
            else:
                X_train_final = X_w2v_final
                X_test_final = X_w2v_test

            print("Hybrid enabled:", USE_HYBRID)
            print("Alpha:", alpha)
            print("Final feature shape:", X_train_final.shape)

            model_cross = train_model(X_train_final, y_train_cross)
            acc_cross_alpha, f1_cross_alpha, cm_cross_alpha = evaluate_model(
                model_cross, X_test_final, y_test_cross
            )
            alpha_results.append(
                {
                    "alpha": float(alpha),
                    "cross_domain_acc": float(acc_cross_alpha),
                    "cross_domain_f1": float(f1_cross_alpha),
                }
            )

            if abs(alpha - CORAL_ALPHA) < 1e-12:
                print("After CORAL:", X_w2v_final.mean(), X_w2v_final.std())
                selected_alpha_metrics = (acc_cross_alpha, f1_cross_alpha, cm_cross_alpha)

        alpha_sweep_results[pooling_method] = alpha_results

        if selected_alpha_metrics is None:
            raise RuntimeError(f"Configured CORAL alpha {CORAL_ALPHA} not found in ALPHAS.")

        acc_cross, f1_cross, cm_cross = selected_alpha_metrics

        print("\nAlpha | Cross-Domain Acc | Cross-Domain F1")
        for result in alpha_results:
            print(
                f"{result['alpha']:<5.2f} | "
                f"{result['cross_domain_acc']:.4f}           | "
                f"{result['cross_domain_f1']:.4f}"
            )

        wav2vec2_results[pooling_method] = {
            "in_domain": {"accuracy": float(acc_in), "f1": float(f1_in)},
            "cross_domain": {"accuracy": float(acc_cross), "f1": float(f1_cross)},
        }

        append_result_record("wav2vec2", pooling_method, "in_domain", acc_in, f1_in, cm_in)
        append_result_record(
            "wav2vec2", pooling_method, "cross_domain", acc_cross, f1_cross, cm_cross
        )
        export_confusion_matrix_csv(cm_in, "wav2vec2", pooling_method, "in_domain")
        export_confusion_matrix_csv(cm_cross, "wav2vec2", pooling_method, "cross_domain")

    for pooling_method in pooling_methods:
        print(f"\n=== Running MFCC Pooling: {pooling_method} ===")

        X_train_in_mfcc, y_train_in_raw_mfcc = prepare_mfcc_features(
            ravdess_train, pooling_mode=pooling_method
        )
        X_test_in_mfcc, y_test_in_raw_mfcc = prepare_mfcc_features(
            ravdess_test, pooling_mode=pooling_method
        )

        label_encoder_in_mfcc = LabelEncoder()
        label_encoder_in_mfcc.fit(y_train_in_raw_mfcc)
        seen_in_mfcc = set(label_encoder_in_mfcc.classes_)

        X_train_in_mfcc, y_train_in_raw_mfcc = filter_to_seen_labels(
            X_train_in_mfcc, y_train_in_raw_mfcc, seen_in_mfcc
        )
        X_test_in_mfcc, y_test_in_raw_mfcc = filter_to_seen_labels(
            X_test_in_mfcc, y_test_in_raw_mfcc, seen_in_mfcc
        )

        y_train_in_mfcc = label_encoder_in_mfcc.transform(y_train_in_raw_mfcc)
        y_test_in_mfcc = label_encoder_in_mfcc.transform(y_test_in_raw_mfcc)

        model_in_mfcc = train_model(X_train_in_mfcc, y_train_in_mfcc)
        acc_in_mfcc, f1_in_mfcc, cm_in_mfcc = evaluate_model(
            model_in_mfcc, X_test_in_mfcc, y_test_in_mfcc
        )

        X_train_cross_mfcc, y_train_cross_raw_mfcc = prepare_mfcc_features(
            ravdess, pooling_mode=pooling_method
        )
        X_test_cross_mfcc, y_test_cross_raw_mfcc = prepare_mfcc_features(
            cremad, pooling_mode=pooling_method
        )

        label_encoder_cross_mfcc = LabelEncoder()
        label_encoder_cross_mfcc.fit(y_train_cross_raw_mfcc)
        seen_cross_mfcc = set(label_encoder_cross_mfcc.classes_)

        X_train_cross_mfcc, y_train_cross_raw_mfcc = filter_to_seen_labels(
            X_train_cross_mfcc, y_train_cross_raw_mfcc, seen_cross_mfcc
        )
        X_test_cross_mfcc, y_test_cross_raw_mfcc = filter_to_seen_labels(
            X_test_cross_mfcc, y_test_cross_raw_mfcc, seen_cross_mfcc
        )

        y_train_cross_mfcc = label_encoder_cross_mfcc.transform(y_train_cross_raw_mfcc)
        y_test_cross_mfcc = label_encoder_cross_mfcc.transform(y_test_cross_raw_mfcc)

        model_cross_mfcc = train_model(X_train_cross_mfcc, y_train_cross_mfcc)
        acc_cross_mfcc, f1_cross_mfcc, cm_cross_mfcc = evaluate_model(
            model_cross_mfcc, X_test_cross_mfcc, y_test_cross_mfcc
        )

        mfcc_results[pooling_method] = {
            "in_domain": {"accuracy": float(acc_in_mfcc), "f1": float(f1_in_mfcc)},
            "cross_domain": {"accuracy": float(acc_cross_mfcc), "f1": float(f1_cross_mfcc)},
        }

        append_result_record("mfcc", pooling_method, "in_domain", acc_in_mfcc, f1_in_mfcc, cm_in_mfcc)
        append_result_record(
            "mfcc", pooling_method, "cross_domain", acc_cross_mfcc, f1_cross_mfcc, cm_cross_mfcc
        )
        export_confusion_matrix_csv(cm_in_mfcc, "mfcc", pooling_method, "in_domain")
        export_confusion_matrix_csv(cm_cross_mfcc, "mfcc", pooling_method, "cross_domain")

    with open(ALPHA_SWEEP_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(alpha_sweep_results, f, indent=2)

    print("\n=== WAV2VEC2 RESULTS ===")
    print("Pooling     | In-Domain Acc | In-Domain F1 | Cross-Domain Acc | Cross-Domain F1")
    for pooling_method in pooling_methods:
        in_acc = wav2vec2_results[pooling_method]["in_domain"]["accuracy"]
        in_f1 = wav2vec2_results[pooling_method]["in_domain"]["f1"]
        cr_acc = wav2vec2_results[pooling_method]["cross_domain"]["accuracy"]
        cr_f1 = wav2vec2_results[pooling_method]["cross_domain"]["f1"]
        print(f"{pooling_method:<11} | {in_acc:.4f}        | {in_f1:.4f}       | {cr_acc:.4f}           | {cr_f1:.4f}")

    print("\n=== MFCC RESULTS ===")
    print("Pooling     | In-Domain Acc | In-Domain F1 | Cross-Domain Acc | Cross-Domain F1")
    for pooling_method in pooling_methods:
        in_acc = mfcc_results[pooling_method]["in_domain"]["accuracy"]
        in_f1 = mfcc_results[pooling_method]["in_domain"]["f1"]
        cr_acc = mfcc_results[pooling_method]["cross_domain"]["accuracy"]
        cr_f1 = mfcc_results[pooling_method]["cross_domain"]["f1"]
        print(f"{pooling_method:<11} | {in_acc:.4f}        | {in_f1:.4f}       | {cr_acc:.4f}           | {cr_f1:.4f}")

    print("\n=== FINAL COMPARISON ===")
    print("Method        | Model      | In-Domain F1 | Cross-Domain F1")
    for pooling_method in pooling_methods:
        print(
            f"{pooling_method:<13} | {'wav2vec2':<10} | "
            f"{wav2vec2_results[pooling_method]['in_domain']['f1']:.4f}        | "
            f"{wav2vec2_results[pooling_method]['cross_domain']['f1']:.4f}"
        )
        print(
            f"{pooling_method:<13} | {'MFCC':<10} | "
            f"{mfcc_results[pooling_method]['in_domain']['f1']:.4f}        | "
            f"{mfcc_results[pooling_method]['cross_domain']['f1']:.4f}"
        )

        append_final_comparison_row(
            "wav2vec2",
            pooling_method,
            wav2vec2_results[pooling_method]["in_domain"]["accuracy"],
            wav2vec2_results[pooling_method]["in_domain"]["f1"],
            wav2vec2_results[pooling_method]["cross_domain"]["accuracy"],
            wav2vec2_results[pooling_method]["cross_domain"]["f1"],
        )
        append_final_comparison_row(
            "mfcc",
            pooling_method,
            mfcc_results[pooling_method]["in_domain"]["accuracy"],
            mfcc_results[pooling_method]["in_domain"]["f1"],
            mfcc_results[pooling_method]["cross_domain"]["accuracy"],
            mfcc_results[pooling_method]["cross_domain"]["f1"],
        )

    best_in_w2v = max(pooling_methods, key=lambda m: wav2vec2_results[m]["in_domain"]["f1"])
    best_in_mfcc = max(pooling_methods, key=lambda m: mfcc_results[m]["in_domain"]["f1"])
    best_cross_w2v = max(pooling_methods, key=lambda m: wav2vec2_results[m]["cross_domain"]["f1"])
    best_cross_mfcc = max(pooling_methods, key=lambda m: mfcc_results[m]["cross_domain"]["f1"])

    best_in_model = (
        "wav2vec2"
        if wav2vec2_results[best_in_w2v]["in_domain"]["f1"] >= mfcc_results[best_in_mfcc]["in_domain"]["f1"]
        else "MFCC"
    )
    best_cross_model = (
        "wav2vec2"
        if wav2vec2_results[best_cross_w2v]["cross_domain"]["f1"] >= mfcc_results[best_cross_mfcc]["cross_domain"]["f1"]
        else "MFCC"
    )

    pooling_overall_scores = {}
    for pooling_method in pooling_methods:
        pooling_overall_scores[pooling_method] = (
            wav2vec2_results[pooling_method]["in_domain"]["f1"]
            + wav2vec2_results[pooling_method]["cross_domain"]["f1"]
            + mfcc_results[pooling_method]["in_domain"]["f1"]
            + mfcc_results[pooling_method]["cross_domain"]["f1"]
        ) / 4.0
    best_pooling_overall = max(pooling_overall_scores, key=pooling_overall_scores.get)

    print(
        f"\nSummary: Better in-domain model = {best_in_model}; "
        f"better cross-domain model = {best_cross_model}; "
        f"best pooling overall = {best_pooling_overall}."
    )


if __name__ == "__main__":
    main()
