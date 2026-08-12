from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from coral import coral_align
from phase0_baseline import prepare_cached_wav2vec_features
from progress_utils import log_step
from results_utils import append_experiment_record, save_result
from ravdess_preprocessing import build_cremad_dataset, build_dataset
from svm_utils import train_svm_model
from wav2vec2_cross_dataset_eval import (
    filter_common_labels,
    filter_to_seen_labels,
    format_ravdess_dataset_for_split,
    prepare_mfcc_features,
)


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def prepare_mfcc_features_silent(dataset: list[dict], pooling_mode: str) -> tuple[np.ndarray, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_mfcc_features(dataset, pooling_mode=pooling_mode)


def main() -> None:
    log_step("[phase11] Loading datasets")
    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(cremad_raw)

    log_step("[phase11] Preparing wav2vec2 features")
    X_w2v_train, y_train_raw = prepare_cached_wav2vec_features(ravdess, pooling_mode="mean")
    X_w2v_test, y_test_raw = prepare_cached_wav2vec_features(cremad, pooling_mode="mean")
    log_step("[phase11] Preparing MFCC features")
    X_mfcc_train, y_train_mfcc_raw = prepare_mfcc_features_silent(ravdess, pooling_mode="mean")
    X_mfcc_test, y_test_mfcc_raw = prepare_mfcc_features_silent(cremad, pooling_mode="mean")

    if not np.array_equal(y_train_raw, y_train_mfcc_raw):
        raise RuntimeError("Label mismatch between wav2vec2 and MFCC train features.")
    if not np.array_equal(y_test_raw, y_test_mfcc_raw):
        raise RuntimeError("Label mismatch between wav2vec2 and MFCC test features.")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw)
    seen_labels = set(label_encoder.classes_)
    X_w2v_train, y_train_raw = filter_to_seen_labels(X_w2v_train, y_train_raw, seen_labels)
    X_w2v_test, y_test_raw = filter_to_seen_labels(X_w2v_test, y_test_raw, seen_labels)
    X_mfcc_train, y_train_mfcc_raw = filter_to_seen_labels(X_mfcc_train, y_train_mfcc_raw, seen_labels)
    X_mfcc_test, y_test_mfcc_raw = filter_to_seen_labels(X_mfcc_test, y_test_mfcc_raw, seen_labels)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    log_step("[phase11] Applying CORAL")
    X_w2v_aligned = coral_align(X_w2v_train, X_w2v_test)
    rows: list[dict[str, object]] = []

    print("===== PHASE 11 RESULTS (SVM + METHOD) =====")
    print()
    print("Alpha | Accuracy | F1 Score")
    print("--------------------------------")
    for alpha in ALPHAS:
        log_step(f"[phase11] Training alpha={alpha:.2f}")
        X_w2v_final = alpha * X_w2v_aligned + (1.0 - alpha) * X_w2v_train
        X_train_final = np.concatenate([X_w2v_final, X_mfcc_train], axis=1)
        X_test_final = np.concatenate([X_w2v_test, X_mfcc_test], axis=1)
        model = train_svm_model(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        confusion = confusion_matrix(y_test, y_pred)
        row = {
            "alpha": float(alpha),
            "accuracy": float(accuracy),
            "f1_score": float(macro_f1),
            "confusion_matrix": confusion.tolist(),
        }
        rows.append(row)
        append_experiment_record(
            {
                "phase": "phase11_svm_method",
                "train_dataset": "RAVDESS",
                "test_dataset": "CREMA-D",
                "model": "SVM",
                "method": "CORAL + Hybrid",
                "feature_type": "wav2vec2 + MFCC",
                "pooling_method": "mean",
                "evaluation_type": "cross_domain",
                "alpha": float(alpha),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "confusion_matrix": confusion.tolist(),
            }
        )
        print(f"{alpha:<4.2f}  | {accuracy:.4f}   | {macro_f1:.4f}")

    save_result(
        "phase11_svm_method.json",
        {
            "phase": "phase11_svm_method",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "SVM",
            "method": "CORAL + Hybrid",
            "feature": "wav2vec2 + MFCC",
            "pooling": "mean",
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
