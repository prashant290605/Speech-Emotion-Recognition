from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from progress_utils import log_step
from results_utils import append_experiment_record, save_result
from ravdess_preprocessing import build_cremad_dataset, build_dataset
from wav2vec2_cross_dataset_eval import (
    build_feature_file_path,
    filter_common_labels,
    filter_to_seen_labels,
    format_ravdess_dataset_for_split,
    get_or_compute_features,
    pool_features,
    set_seed,
    train_model,
    evaluate_model,
)


def prepare_cached_wav2vec_features(
    dataset: list[dict], pooling_mode: str = "mean"
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    n_total = len(dataset)

    for idx, sample in enumerate(dataset, start=1):
        path = sample.get("path", sample.get("file_path"))
        if path is None:
            continue

        cache_path = build_feature_file_path(path)
        if cache_path.exists():
            embedding = np.load(cache_path).astype(np.float32)
            status = "LOADED"
        else:
            log_step(f"[wav2vec2] {idx}/{n_total} COMPUTING {Path(path).name}")
            embedding = get_or_compute_features(path)
            if embedding is None:
                raise FileNotFoundError(
                    f"Failed to compute wav2vec2 embedding for {path}: {cache_path}"
                )
            status = "COMPUTED"
        feature_rows.append(pool_features(embedding, mode=pooling_mode))
        labels.append(sample["label"])

        if cache_path.exists() and status == "LOADED":
            if idx == 1 or idx == n_total or idx % 100 == 0:
                log_step(f"[wav2vec2] {idx}/{n_total} LOADED")
        else:
            log_step(f"[wav2vec2] {idx}/{n_total} {status}")

    if not feature_rows:
        raise RuntimeError("No cached wav2vec2 features were loaded.")

    return np.vstack(feature_rows).astype(np.float32), np.array(labels)


def main() -> None:
    set_seed()
    log_step("[phase0] Loading datasets")

    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(cremad_raw)

    log_step("[phase0] Preparing wav2vec2 features")
    X_train, y_train_raw = prepare_cached_wav2vec_features(ravdess, pooling_mode="mean")
    X_test, y_test_raw = prepare_cached_wav2vec_features(cremad, pooling_mode="mean")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw)
    seen_labels = set(label_encoder.classes_)

    X_train, y_train_raw = filter_to_seen_labels(X_train, y_train_raw, seen_labels)
    X_test, y_test_raw = filter_to_seen_labels(X_test, y_test_raw, seen_labels)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    log_step("[phase0] Training Logistic Regression")
    model = train_model(X_train, y_train)
    log_step("[phase0] Evaluating")
    accuracy, macro_f1, confusion = evaluate_model(model, X_test, y_test)

    save_result(
        "phase0_baseline.json",
        {
            "phase": "phase0_baseline",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "Logistic Regression",
            "feature": "wav2vec2",
            "pooling": "mean",
            "accuracy": float(accuracy),
            "f1_score": float(macro_f1),
        },
    )
    append_experiment_record(
        {
            "phase": "phase0_baseline",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "Logistic Regression",
            "method": "Baseline",
            "feature_type": "wav2vec2",
            "pooling_method": "mean",
            "evaluation_type": "cross_domain",
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "confusion_matrix": confusion.tolist(),
        }
    )

    print("===== PHASE 0 BASELINE =====")
    print("Train: RAVDESS → Test: CREMA-D")
    print("Model: Logistic Regression")
    print("Feature: wav2vec2 (mean pooling)")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
