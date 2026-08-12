from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    set_seed,
    split_dataset_by_speaker,
)


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def prepare_mfcc_features_silent(dataset: list[dict], pooling_mode: str) -> tuple[np.ndarray, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_mfcc_features(dataset, pooling_mode=pooling_mode)


def format_cremad_dataset_for_split(dataset: list[dict]) -> list[dict]:
    out: list[dict] = []
    for sample in dataset:
        row = dict(sample)
        row["path"] = sample.get("path", sample.get("file_path"))
        row["speaker"] = sample["speaker_id"]
        out.append(row)
    return out


def train_lr_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


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
) -> tuple[float, float, np.ndarray]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, macro_f1, confusion


def prepare_split_features(
    train_dataset: list[dict],
    test_dataset: list[dict],
) -> dict[str, np.ndarray]:
    X_w2v_train, y_train_raw = prepare_cached_wav2vec_features(train_dataset, pooling_mode="mean")
    X_w2v_test, y_test_raw = prepare_cached_wav2vec_features(test_dataset, pooling_mode="mean")

    X_mfcc_train, y_train_mfcc_raw = prepare_mfcc_features_silent(train_dataset, pooling_mode="mean")
    X_mfcc_test, y_test_mfcc_raw = prepare_mfcc_features_silent(test_dataset, pooling_mode="mean")

    if not np.array_equal(y_train_raw, y_train_mfcc_raw):
        raise RuntimeError("Train label mismatch between wav2vec2 and MFCC.")
    if not np.array_equal(y_test_raw, y_test_mfcc_raw):
        raise RuntimeError("Test label mismatch between wav2vec2 and MFCC.")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw)
    seen_labels = set(label_encoder.classes_)

    X_w2v_train, y_train_raw = filter_to_seen_labels(X_w2v_train, y_train_raw, seen_labels)
    X_w2v_test, y_test_raw = filter_to_seen_labels(X_w2v_test, y_test_raw, seen_labels)
    X_mfcc_train, y_train_mfcc_raw = filter_to_seen_labels(X_mfcc_train, y_train_mfcc_raw, seen_labels)
    X_mfcc_test, y_test_mfcc_raw = filter_to_seen_labels(X_mfcc_test, y_test_mfcc_raw, seen_labels)

    if not np.array_equal(y_train_raw, y_train_mfcc_raw):
        raise RuntimeError("Train label mismatch after filtering.")
    if not np.array_equal(y_test_raw, y_test_mfcc_raw):
        raise RuntimeError("Test label mismatch after filtering.")

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    return {
        "X_w2v_train": X_w2v_train,
        "X_w2v_test": X_w2v_test,
        "X_mfcc_train": X_mfcc_train,
        "X_mfcc_test": X_mfcc_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def run_baseline(
    features: dict[str, np.ndarray],
    trainer: Callable[[np.ndarray, np.ndarray], Pipeline],
) -> tuple[float, float, np.ndarray]:
    model = trainer(features["X_w2v_train"], features["y_train"])
    return evaluate_model(model, features["X_w2v_test"], features["y_test"])


def run_hybrid(
    features: dict[str, np.ndarray],
    trainer: Callable[[np.ndarray, np.ndarray], Pipeline],
) -> tuple[float, float, np.ndarray]:
    X_train = np.concatenate([features["X_w2v_train"], features["X_mfcc_train"]], axis=1)
    X_test = np.concatenate([features["X_w2v_test"], features["X_mfcc_test"]], axis=1)
    model = trainer(X_train, features["y_train"])
    return evaluate_model(model, X_test, features["y_test"])


def run_method_best_alpha(
    features: dict[str, np.ndarray],
    trainer: Callable[[np.ndarray, np.ndarray], Pipeline],
) -> tuple[float, float, np.ndarray]:
    X_w2v_aligned = coral_align(features["X_w2v_train"], features["X_w2v_test"])

    best_metrics: tuple[float, float, np.ndarray] | None = None
    for alpha in ALPHAS:
        X_w2v_final = alpha * X_w2v_aligned + (1.0 - alpha) * features["X_w2v_train"]
        X_train = np.concatenate([X_w2v_final, features["X_mfcc_train"]], axis=1)
        X_test = np.concatenate([features["X_w2v_test"], features["X_mfcc_test"]], axis=1)
        model = trainer(X_train, features["y_train"])
        metrics = evaluate_model(model, X_test, features["y_test"])
        if best_metrics is None or metrics[1] > best_metrics[1]:
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Alpha sweep did not produce any results.")
    return best_metrics


def collect_dataset_results(
    features: dict[str, np.ndarray]
) -> dict[str, tuple[float, float, np.ndarray]]:
    lr_baseline = run_baseline(features, train_lr_model)
    lr_hybrid = run_hybrid(features, train_lr_model)
    lr_method = run_method_best_alpha(features, train_lr_model)

    mlp_baseline = run_baseline(features, train_mlp_model)
    mlp_hybrid = run_hybrid(features, train_mlp_model)
    mlp_method = run_method_best_alpha(features, train_mlp_model)
    svm_baseline = run_baseline(features, train_svm_model)
    svm_hybrid = run_hybrid(features, train_svm_model)
    svm_method = run_method_best_alpha(features, train_svm_model)

    return {
        "LR_Baseline": lr_baseline,
        "LR_Hybrid": lr_hybrid,
        "LR_Method": lr_method,
        "MLP_Baseline": mlp_baseline,
        "MLP_Hybrid": mlp_hybrid,
        "MLP_Method": mlp_method,
        "SVM_Baseline": svm_baseline,
        "SVM_Hybrid": svm_hybrid,
        "SVM_Method": svm_method,
    }


def print_dataset_results(title: str, rows: dict[str, tuple[float, float, np.ndarray]]) -> None:
    print(f"Dataset: {title}")
    print("Method rows are analysis only: best F1 over alpha sweep [0.0, 0.25, 0.5, 0.75, 1.0]")
    print()
    print("Model | Setup    | Accuracy | F1")
    print("------------------------------------")
    print(f"LR    | Baseline | {rows['LR_Baseline'][0]:.4f}   | {rows['LR_Baseline'][1]:.4f}")
    print(f"LR    | Hybrid   | {rows['LR_Hybrid'][0]:.4f}   | {rows['LR_Hybrid'][1]:.4f}")
    print(f"LR    | Method   | {rows['LR_Method'][0]:.4f}   | {rows['LR_Method'][1]:.4f}")
    print(f"MLP   | Baseline | {rows['MLP_Baseline'][0]:.4f}   | {rows['MLP_Baseline'][1]:.4f}")
    print(f"MLP   | Hybrid   | {rows['MLP_Hybrid'][0]:.4f}   | {rows['MLP_Hybrid'][1]:.4f}")
    print(f"MLP   | Method   | {rows['MLP_Method'][0]:.4f}   | {rows['MLP_Method'][1]:.4f}")
    print(f"SVM   | Baseline | {rows['SVM_Baseline'][0]:.4f}   | {rows['SVM_Baseline'][1]:.4f}")
    print(f"SVM   | Hybrid   | {rows['SVM_Hybrid'][0]:.4f}   | {rows['SVM_Hybrid'][1]:.4f}")
    print(f"SVM   | Method   | {rows['SVM_Method'][0]:.4f}   | {rows['SVM_Method'][1]:.4f}")
    print()


def main() -> None:
    set_seed()
    log_step("[in-domain] Loading datasets and creating speaker splits")

    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(format_cremad_dataset_for_split(cremad_raw))

    ravdess_train, ravdess_test = split_dataset_by_speaker(ravdess, test_size=0.2, seed=42)
    cremad_train, cremad_test = split_dataset_by_speaker(cremad, test_size=0.2, seed=42)

    log_step("[in-domain] Preparing RAVDESS feature sets")
    ravdess_features = prepare_split_features(ravdess_train, ravdess_test)
    log_step("[in-domain] Preparing CREMA-D feature sets")
    cremad_features = prepare_split_features(cremad_train, cremad_test)
    log_step("[in-domain] Running model matrix for RAVDESS")
    ravdess_rows = collect_dataset_results(ravdess_features)
    log_step("[in-domain] Running model matrix for CREMA-D")
    cremad_rows = collect_dataset_results(cremad_features)

    save_result(
        "in_domain_results.json",
        {
            "phase": "in_domain_results",
            "analysis_note": "Method rows are analysis only: best F1 over alpha sweep [0.0, 0.25, 0.5, 0.75, 1.0]",
            "datasets": {
                "RAVDESS_to_RAVDESS": {
                    key: {"accuracy": float(value[0]), "f1_score": float(value[1])}
                    | {"confusion_matrix": value[2].tolist()}
                    for key, value in ravdess_rows.items()
                },
                "CREMA-D_to_CREMA-D": {
                    key: {"accuracy": float(value[0]), "f1_score": float(value[1])}
                    | {"confusion_matrix": value[2].tolist()}
                    for key, value in cremad_rows.items()
                },
            },
        },
    )
    for key, value in ravdess_rows.items():
        model_name, method_name = key.split("_", maxsplit=1)
        append_experiment_record(
            {
                "phase": "in_domain_results",
                "train_dataset": "RAVDESS",
                "test_dataset": "RAVDESS",
                "model": "Logistic Regression"
                if model_name == "LR"
                else ("MLPClassifier" if model_name == "MLP" else "SVM"),
                "method": method_name,
                "feature_type": "wav2vec2"
                if method_name == "Baseline"
                else "wav2vec2 + MFCC",
                "pooling_method": "mean",
                "evaluation_type": "in_domain",
                "analysis_only": method_name == "Method",
                "accuracy": float(value[0]),
                "macro_f1": float(value[1]),
                "confusion_matrix": value[2].tolist(),
            }
        )
    for key, value in cremad_rows.items():
        model_name, method_name = key.split("_", maxsplit=1)
        append_experiment_record(
            {
                "phase": "in_domain_results",
                "train_dataset": "CREMA-D",
                "test_dataset": "CREMA-D",
                "model": "Logistic Regression"
                if model_name == "LR"
                else ("MLPClassifier" if model_name == "MLP" else "SVM"),
                "method": method_name,
                "feature_type": "wav2vec2"
                if method_name == "Baseline"
                else "wav2vec2 + MFCC",
                "pooling_method": "mean",
                "evaluation_type": "in_domain",
                "analysis_only": method_name == "Method",
                "accuracy": float(value[0]),
                "macro_f1": float(value[1]),
                "confusion_matrix": value[2].tolist(),
            }
        )

    print("===== IN-DOMAIN RESULTS =====")
    print()
    print_dataset_results("RAVDESS → RAVDESS", ravdess_rows)
    print_dataset_results("CREMA-D → CREMA-D", cremad_rows)


if __name__ == "__main__":
    main()
