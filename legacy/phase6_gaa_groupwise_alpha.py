from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder

from coral import coral_align
from phase0_baseline import prepare_cached_wav2vec_features
from progress_utils import log_step
from results_utils import append_experiment_record, save_result
from ravdess_preprocessing import build_cremad_dataset, build_dataset
from wav2vec2_cross_dataset_eval import (
    evaluate_model,
    filter_common_labels,
    filter_to_seen_labels,
    format_ravdess_dataset_for_split,
    prepare_mfcc_features,
    set_seed,
    train_model,
)


N_CLUSTERS = 8


def prepare_mfcc_features_silent(
    dataset: list[dict], pooling_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_mfcc_features(dataset, pooling_mode=pooling_mode)


def cluster_feature_dimensions(X_train: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="precomputed",
        linkage="average",
    )
    return clustering.fit_predict(distance)


def compute_group_alphas(
    X_source: np.ndarray, X_target: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    alphas = np.zeros(labels.shape[0], dtype=np.float32)
    group_values: list[float] = []

    for group_id in range(N_CLUSTERS):
        idx = np.where(labels == group_id)[0]
        if idx.size == 0:
            continue

        mu_s = X_source[:, idx].mean(axis=0)
        mu_t = X_target[:, idx].mean(axis=0)
        d_g = float(np.linalg.norm(mu_s - mu_t, ord=2))
        alpha_g = np.clip(d_g / (d_g + 1e-6), 0.0, 1.0)

        alphas[idx] = alpha_g
        group_values.append(float(alpha_g))

    if not group_values:
        raise RuntimeError("No group-wise alpha values were computed.")

    summary = {
        "min": float(np.min(group_values)),
        "max": float(np.max(group_values)),
        "mean": float(np.mean(group_values)),
    }
    return alphas, summary


def main() -> None:
    set_seed()
    log_step("[phase6] Loading datasets")

    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(cremad_raw)

    log_step("[phase6] Preparing wav2vec2 features")
    X_w2v_train, y_train_raw = prepare_cached_wav2vec_features(
        ravdess, pooling_mode="mean"
    )
    X_w2v_test, y_test_raw = prepare_cached_wav2vec_features(
        cremad, pooling_mode="mean"
    )

    log_step("[phase6] Preparing MFCC features")
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

    log_step("[phase6] Applying CORAL and group clustering")
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

    log_step("[phase6] Training Logistic Regression")
    model = train_model(X_train_final, y_train)
    accuracy, macro_f1, confusion = evaluate_model(model, X_test_final, y_test)

    save_result(
        "phase6_gaa_groupwise_alpha.json",
        {
            "phase": "phase6_gaa_groupwise_alpha",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "Logistic Regression",
            "method": "Group-wise Adaptive Alignment (GAA)",
            "feature": "wav2vec2 + MFCC",
            "pooling": "mean",
            "n_clusters": N_CLUSTERS,
            "alpha_summary": alpha_summary,
            "accuracy": float(accuracy),
            "f1_score": float(macro_f1),
        },
    )
    append_experiment_record(
        {
            "phase": "phase6_gaa_groupwise_alpha",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "Logistic Regression",
            "method": "Group-wise Adaptive Alignment (GAA)",
            "feature_type": "wav2vec2 + MFCC",
            "pooling_method": "mean",
            "evaluation_type": "cross_domain",
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "confusion_matrix": confusion.tolist(),
        }
    )

    print("===== PHASE 6 (GAA - GROUP-WISE ALPHA) =====")
    print()
    print("Model: Logistic Regression")
    print("Method: Group-wise Adaptive Alignment (GAA)")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
