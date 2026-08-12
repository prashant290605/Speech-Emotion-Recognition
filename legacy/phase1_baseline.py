from __future__ import annotations

from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from progress_utils import log_step
from results_utils import append_experiment_record, save_result
from ravdess_preprocessing import build_cremad_dataset, build_dataset
from wav2vec2_cross_dataset_eval import (
    filter_common_labels,
    filter_to_seen_labels,
    format_ravdess_dataset_for_split,
    train_model,
    evaluate_model,
)

from phase0_baseline import prepare_cached_wav2vec_features


def main() -> None:
    log_step("[phase1] Loading datasets")
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)
    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)

    cremad = filter_common_labels(cremad_raw)
    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))

    log_step("[phase1] Preparing wav2vec2 features")
    X_train, y_train_raw = prepare_cached_wav2vec_features(cremad, pooling_mode="mean")
    X_test, y_test_raw = prepare_cached_wav2vec_features(ravdess, pooling_mode="mean")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw)
    seen_labels = set(label_encoder.classes_)

    X_train, y_train_raw = filter_to_seen_labels(X_train, y_train_raw, seen_labels)
    X_test, y_test_raw = filter_to_seen_labels(X_test, y_test_raw, seen_labels)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    log_step("[phase1] Training Logistic Regression")
    model = train_model(X_train, y_train)
    log_step("[phase1] Evaluating")
    accuracy, macro_f1, confusion = evaluate_model(model, X_test, y_test)

    save_result(
        "phase1_baseline.json",
        {
            "phase": "phase1_baseline",
            "train_dataset": "CREMA-D",
            "test_dataset": "RAVDESS",
            "model": "Logistic Regression",
            "feature": "wav2vec2",
            "pooling": "mean",
            "accuracy": float(accuracy),
            "f1_score": float(macro_f1),
        },
    )
    append_experiment_record(
        {
            "phase": "phase1_baseline",
            "train_dataset": "CREMA-D",
            "test_dataset": "RAVDESS",
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

    print("===== PHASE 1 BASELINE =====")
    print("Train: CREMA-D → Test: RAVDESS")
    print("Model: Logistic Regression")
    print("Feature: wav2vec2 (mean pooling)")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
