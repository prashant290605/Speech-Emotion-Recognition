from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    filter_common_labels,
    filter_to_seen_labels,
    format_ravdess_dataset_for_split,
    set_seed,
)


def train_mlp_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(256,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size=32,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float, list[list[int]]]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, macro_f1, confusion.tolist()


def main() -> None:
    set_seed()
    log_step("[phase7] Loading datasets")

    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(cremad_raw)

    log_step("[phase7] Preparing wav2vec2 features")
    X_w2v_train, y_train_raw = prepare_cached_wav2vec_features(
        ravdess, pooling_mode="mean"
    )
    X_w2v_test, y_test_raw = prepare_cached_wav2vec_features(
        cremad, pooling_mode="mean"
    )

    log_step("[phase7] Preparing MFCC features")
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

    log_step("[phase7] Applying CORAL and group clustering")
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

    log_step("[phase7] Training MLP")
    model = train_mlp_model(X_train_final, y_train)
    accuracy, macro_f1, confusion = evaluate_model(model, X_test_final, y_test)

    save_result(
        "phase7_mlp_gaa.json",
        {
            "phase": "phase7_mlp_gaa",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "MLPClassifier",
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
            "phase": "phase7_mlp_gaa",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "MLPClassifier",
            "method": "Group-wise Adaptive Alignment (GAA)",
            "feature_type": "wav2vec2 + MFCC",
            "pooling_method": "mean",
            "evaluation_type": "cross_domain",
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "confusion_matrix": confusion,
        }
    )

    print("===== PHASE 7 (MLP + GAA) =====")
    print()
    print("Model: MLPClassifier")
    print("Method: Group-wise Adaptive Alignment (GAA)")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
