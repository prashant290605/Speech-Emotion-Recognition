from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from coral import coral_align
from phase0_baseline import prepare_cached_wav2vec_features
from progress_utils import log_step
from results_utils import append_experiment_record, save_result
from ravdess_preprocessing import build_cremad_dataset, build_dataset
from wav2vec2_cross_dataset_eval import (
    filter_common_labels,
    filter_to_seen_labels,
    format_ravdess_dataset_for_split,
    prepare_mfcc_features,
    set_seed,
    train_model,
    evaluate_model,
)


def prepare_mfcc_features_silent(
    dataset: list[dict], pooling_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_mfcc_features(dataset, pooling_mode=pooling_mode)


def compute_vector_alpha(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    mu_s = X_source.mean(axis=0)
    mu_t = X_target.mean(axis=0)
    sigma_s = X_source.std(axis=0)
    sigma_t = X_target.std(axis=0)

    alpha = np.abs(mu_s - mu_t) / (sigma_s + sigma_t + 1e-6)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha.astype(np.float32)


def main() -> None:
    set_seed()
    log_step("[phase5] Loading datasets")

    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(cremad_raw)

    log_step("[phase5] Preparing wav2vec2 features")
    X_w2v_train, y_train_raw = prepare_cached_wav2vec_features(
        ravdess, pooling_mode="mean"
    )
    X_w2v_test, y_test_raw = prepare_cached_wav2vec_features(
        cremad, pooling_mode="mean"
    )

    log_step("[phase5] Preparing MFCC features")
    X_mfcc_train, y_train_mfcc_raw = prepare_mfcc_features_silent(
        ravdess, pooling_mode="mean"
    )
    X_mfcc_test, y_test_mfcc_raw = prepare_mfcc_features_silent(
        cremad, pooling_mode="mean"
    )

    if not np.array_equal(y_train_raw, y_train_mfcc_raw):
        raise RuntimeError("Label mismatch between wav2vec2 and MFCC train features.")
    if not np.array_equal(y_test_raw, y_test_mfcc_raw):
        raise RuntimeError("Label mismatch between wav2vec2 and MFCC test features.")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw)
    seen_labels = set(label_encoder.classes_)

    X_w2v_train, y_train_raw = filter_to_seen_labels(
        X_w2v_train, y_train_raw, seen_labels
    )
    X_w2v_test, y_test_raw = filter_to_seen_labels(
        X_w2v_test, y_test_raw, seen_labels
    )
    X_mfcc_train, y_train_mfcc_raw = filter_to_seen_labels(
        X_mfcc_train, y_train_mfcc_raw, seen_labels
    )
    X_mfcc_test, y_test_mfcc_raw = filter_to_seen_labels(
        X_mfcc_test, y_test_mfcc_raw, seen_labels
    )

    if not np.array_equal(y_train_raw, y_train_mfcc_raw):
        raise RuntimeError("Label mismatch after filtering train features.")
    if not np.array_equal(y_test_raw, y_test_mfcc_raw):
        raise RuntimeError("Label mismatch after filtering test features.")

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    log_step("[phase5] Applying CORAL and computing vector alpha")
    X_w2v_aligned = coral_align(X_w2v_train, X_w2v_test)
    alpha = compute_vector_alpha(X_w2v_train, X_w2v_test)

    if alpha.shape[0] != X_w2v_train.shape[1]:
        raise RuntimeError("Vector alpha dimension does not match wav2vec2 features.")

    X_w2v_final = alpha * X_w2v_aligned + (1.0 - alpha) * X_w2v_train
    X_train_final = np.concatenate([X_w2v_final, X_mfcc_train], axis=1)
    X_test_final = np.concatenate([X_w2v_test, X_mfcc_test], axis=1)

    log_step("[phase5] Training Logistic Regression")
    model = train_model(X_train_final, y_train)
    accuracy, macro_f1, confusion = evaluate_model(model, X_test_final, y_test)

    save_result(
        "phase5_fwaa_vector_alpha.json",
        {
            "phase": "phase5_fwaa_vector_alpha",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "Logistic Regression",
            "method": "Feature-wise Adaptive Alignment (FWAA)",
            "feature": "wav2vec2 + MFCC",
            "pooling": "mean",
            "alpha_summary": {
                "min": float(alpha.min()),
                "max": float(alpha.max()),
                "mean": float(alpha.mean()),
            },
            "accuracy": float(accuracy),
            "f1_score": float(macro_f1),
        },
    )
    append_experiment_record(
        {
            "phase": "phase5_fwaa_vector_alpha",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "Logistic Regression",
            "method": "Feature-wise Adaptive Alignment (FWAA)",
            "feature_type": "wav2vec2 + MFCC",
            "pooling_method": "mean",
            "evaluation_type": "cross_domain",
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "confusion_matrix": confusion.tolist(),
        }
    )

    print("===== PHASE 5 (FWAA - VECTOR ALPHA) =====")
    print()
    print("Model: Logistic Regression")
    print("Method: Feature-wise Adaptive Alignment (FWAA)")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
