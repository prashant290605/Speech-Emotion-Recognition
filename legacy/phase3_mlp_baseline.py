from __future__ import annotations

from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from phase0_baseline import prepare_cached_wav2vec_features
from progress_utils import log_step
from results_utils import append_experiment_record, save_result
from ravdess_preprocessing import build_cremad_dataset, build_dataset
from wav2vec2_cross_dataset_eval import (
    filter_common_labels,
    filter_to_seen_labels,
    format_ravdess_dataset_for_split,
    set_seed,
)


def train_mlp_model(X_train, y_train) -> Pipeline:
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


def evaluate_model(model: Pipeline, X_test, y_test) -> tuple[float, float, list[list[int]]]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, macro_f1, confusion.tolist()


def main() -> None:
    set_seed()
    log_step("[phase3] Loading datasets")

    ravdess_raw = build_dataset(Path("Radvess"), load_audio_data=False)
    cremad_raw = build_cremad_dataset(Path("Crema D"), load_audio_data=False)

    ravdess = filter_common_labels(format_ravdess_dataset_for_split(ravdess_raw))
    cremad = filter_common_labels(cremad_raw)

    log_step("[phase3] Preparing wav2vec2 features")
    X_train, y_train_raw = prepare_cached_wav2vec_features(ravdess, pooling_mode="mean")
    X_test, y_test_raw = prepare_cached_wav2vec_features(cremad, pooling_mode="mean")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw)
    seen_labels = set(label_encoder.classes_)

    X_train, y_train_raw = filter_to_seen_labels(X_train, y_train_raw, seen_labels)
    X_test, y_test_raw = filter_to_seen_labels(X_test, y_test_raw, seen_labels)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    log_step("[phase3] Training MLP")
    model = train_mlp_model(X_train, y_train)
    log_step("[phase3] Evaluating")
    accuracy, macro_f1, confusion = evaluate_model(model, X_test, y_test)

    save_result(
        "phase3_mlp_baseline.json",
        {
            "phase": "phase3_mlp_baseline",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "MLPClassifier",
            "feature": "wav2vec2",
            "pooling": "mean",
            "accuracy": float(accuracy),
            "f1_score": float(macro_f1),
        },
    )
    append_experiment_record(
        {
            "phase": "phase3_mlp_baseline",
            "train_dataset": "RAVDESS",
            "test_dataset": "CREMA-D",
            "model": "MLPClassifier",
            "method": "Baseline",
            "feature_type": "wav2vec2",
            "pooling_method": "mean",
            "evaluation_type": "cross_domain",
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "confusion_matrix": confusion,
        }
    )

    print("===== PHASE 3 (MLP BASELINE) =====")
    print("Train: RAVDESS → Test: CREMA-D")
    print("Model: MLPClassifier")
    print("Feature: wav2vec2 (mean pooling)")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
