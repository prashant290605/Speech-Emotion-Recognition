from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from python_speech_features import mfcc as psf_mfcc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ravdess_preprocessing import build_cremad_dataset, build_dataset


RANDOM_SEED = 42
N_MFCC = 13
COMMON_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def filter_common_labels(dataset: List[Dict]) -> List[Dict]:
    """Keep only samples in the shared RAVDESS/CREMA-D label space."""
    allowed = set(COMMON_LABELS)
    return [sample for sample in dataset if sample["label"] in allowed]


def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Extract MFCC features with fixed n_mfcc."""
    features = psf_mfcc(signal, samplerate=sr, numcep=n_mfcc)
    return features.astype(np.float32)


def pool_features(mfcc: np.ndarray) -> np.ndarray:
    """Mean+std pooling to convert variable-length MFCC to fixed-size vector."""
    mean_vec = np.mean(mfcc, axis=0)
    std_vec = np.std(mfcc, axis=0)
    return np.concatenate([mean_vec, std_vec]).astype(np.float32)


def prepare_features(dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and raw labels y from dataset records."""
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for sample in dataset:
        mfcc = extract_mfcc(sample["signal"], sample["sample_rate"])
        pooled = pool_features(mfcc)
        X_list.append(pooled)
        y_list.append(sample["label"])

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    return X, y


def train_on_ravdess() -> Tuple[Pipeline, LabelEncoder, int]:
    """Train model on filtered RAVDESS only."""
    ravdess_dataset = build_dataset(Path("Radvess"), load_audio_data=True)
    ravdess_dataset = filter_common_labels(ravdess_dataset)

    X_train, y_train_raw = prepare_features(ravdess_dataset)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, random_state=RANDOM_SEED, solver="lbfgs"
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    return model, label_encoder, len(ravdess_dataset)


def test_on_cremad(
    model: Pipeline, label_encoder: LabelEncoder
) -> Tuple[int, float, float, np.ndarray]:
    """Evaluate trained model on filtered CREMA-D only."""
    cremad_dataset = build_cremad_dataset(Path("Crema D"), load_audio_data=True)
    cremad_dataset = filter_common_labels(cremad_dataset)

    X_test, y_test_raw = prepare_features(cremad_dataset)
    y_test = label_encoder.transform(y_test_raw)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    return len(cremad_dataset), acc, macro_f1, cm


def main() -> None:
    model, label_encoder, n_train = train_on_ravdess()
    n_test, acc, macro_f1, cm = test_on_cremad(model, label_encoder)

    print(f"Number of training samples (RAVDESS): {n_train}")
    print(f"Number of testing samples (CREMA-D): {n_test}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Label order:")
    print(list(label_encoder.classes_))


if __name__ == "__main__":
    main()
