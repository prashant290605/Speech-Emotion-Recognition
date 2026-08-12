from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from coral import coral_align
from phase0_baseline import prepare_cached_wav2vec_features
from phase6_gaa_groupwise_alpha import (
    N_CLUSTERS,
    cluster_feature_dimensions,
    compute_group_alphas,
    prepare_mfcc_features_silent,
)
from progress_utils import log_step
from results_utils import append_experiment_record, save_result
from ravdess_preprocessing import build_cremad_dataset, build_dataset
from wav2vec2_cross_dataset_eval import (
    evaluate_model,
    filter_common_labels,
    filter_to_seen_labels,
    format_ravdess_dataset_for_split,
    set_seed,
    split_dataset_by_speaker,
    train_model,
)


def format_cremad_dataset_for_split(dataset: list[dict]) -> list[dict]:
    out: list[dict] = []
    for sample in dataset:
        row = dict(sample)
        row["path"] = sample.get("path", sample.get("file_path"))
        row["speaker"] = sample["speaker_id"]
        out.append(row)
    return out


def run_gaa(
    train_dataset: list[dict], test_dataset: list[dict]
) -> tuple[float, float, np.ndarray, dict[str, float]]:
    X_w2v_train, y_train_raw = prepare_cached_wav2vec_features(
        train_dataset, pooling_mode="mean"
    )
    X_w2v_test, y_test_raw = prepare_cached_wav2vec_features(
        test_dataset, pooling_mode="mean"
    )

    X_mfcc_train, y_train_mfcc_raw = prepare_mfcc_features_silent(
        train_dataset, pooling_mode="mean"
    )
    X_mfcc_test, y_test_mfcc_raw = prepare_mfcc_features_silent(
        test_dataset, pooling_mode="mean"
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

    X_w2v_aligned = coral_align(X_w2v_train, X_w2v_test)
    group_labels = cluster_feature_dimensions(X_w2v_train)
    alpha_vector, alpha_summary = compute_group_alphas(
        X_w2v_train, X_w2v_test, group_labels
    )

    if alpha_vector.shape[0] != X_w2v_train.shape[1]:
        raise RuntimeError("Group-wise alpha vector dimension mismatch.")

    X_w2v_final = X_w2v_train.copy()
    for group_id in range(N_CLUSTERS):
        idx = np.where(group_labels == group_id)[0]
        if idx.size == 0:
            continue
        alpha_g = alpha_vector[idx][0]
        X_w2v_final[:, idx] = (
            alpha_g * X_w2v_aligned[:, idx] + (1.0 - alpha_g) * X_w2v_train[:, idx]
        )

    X_train_final = np.concatenate([X_w2v_final, X_mfcc_train], axis=1)
    X_test_final = np.concatenate([X_w2v_test, X_mfcc_test], axis=1)

    model = train_model(X_train_final, y_train)
    accuracy, macro_f1, confusion = evaluate_model(model, X_test_final, y_test)
    return accuracy, macro_f1, confusion, alpha_summary


def main() -> None:
    set_seed()
    log_step("[phase9] Loading datasets and creating speaker splits")

    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(format_cremad_dataset_for_split(cremad_raw))

    ravdess_train, ravdess_test = split_dataset_by_speaker(ravdess, test_size=0.2, seed=42)
    cremad_train, cremad_test = split_dataset_by_speaker(cremad, test_size=0.2, seed=42)

    log_step("[phase9] Running GAA for RAVDESS")
    ravdess_accuracy, ravdess_f1, ravdess_confusion, ravdess_alpha = run_gaa(ravdess_train, ravdess_test)
    log_step("[phase9] Running GAA for CREMA-D")
    cremad_accuracy, cremad_f1, cremad_confusion, cremad_alpha = run_gaa(cremad_train, cremad_test)

    save_result(
        "phase9_gaa_in_domain.json",
        {
            "phase": "phase9_gaa_in_domain",
            "model": "Logistic Regression",
            "method": "Group-wise Adaptive Alignment (GAA)",
            "datasets": {
                "RAVDESS_to_RAVDESS": {
                    "accuracy": float(ravdess_accuracy),
                    "f1_score": float(ravdess_f1),
                    "confusion_matrix": ravdess_confusion.tolist(),
                    "alpha_summary": ravdess_alpha,
                },
                "CREMA-D_to_CREMA-D": {
                    "accuracy": float(cremad_accuracy),
                    "f1_score": float(cremad_f1),
                    "confusion_matrix": cremad_confusion.tolist(),
                    "alpha_summary": cremad_alpha,
                },
            },
        },
    )
    append_experiment_record(
        {
            "phase": "phase9_gaa_in_domain",
            "train_dataset": "RAVDESS",
            "test_dataset": "RAVDESS",
            "model": "Logistic Regression",
            "method": "Group-wise Adaptive Alignment (GAA)",
            "feature_type": "wav2vec2 + MFCC",
            "pooling_method": "mean",
            "evaluation_type": "in_domain",
            "accuracy": float(ravdess_accuracy),
            "macro_f1": float(ravdess_f1),
            "confusion_matrix": ravdess_confusion.tolist(),
        }
    )
    append_experiment_record(
        {
            "phase": "phase9_gaa_in_domain",
            "train_dataset": "CREMA-D",
            "test_dataset": "CREMA-D",
            "model": "Logistic Regression",
            "method": "Group-wise Adaptive Alignment (GAA)",
            "feature_type": "wav2vec2 + MFCC",
            "pooling_method": "mean",
            "evaluation_type": "in_domain",
            "accuracy": float(cremad_accuracy),
            "macro_f1": float(cremad_f1),
            "confusion_matrix": cremad_confusion.tolist(),
        }
    )

    print("===== PHASE 9 (GAA IN-DOMAIN) =====")
    print()
    print("Dataset: RAVDESS → RAVDESS")
    print()
    print(f"Accuracy: {ravdess_accuracy:.4f}")
    print(f"F1 Score: {ravdess_f1:.4f}")
    print()
    print("Dataset: CREMA-D → CREMA-D")
    print()
    print(f"Accuracy: {cremad_accuracy:.4f}")
    print(f"F1 Score: {cremad_f1:.4f}")


if __name__ == "__main__":
    main()
