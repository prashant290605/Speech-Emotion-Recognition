from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from python_speech_features import mfcc as psf_mfcc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ravdess_preprocessing import build_dataset


RANDOM_SEED = 42
N_MFCC = 13


def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract MFCC features using python_speech_features.

    Returns an array with shape (num_frames, n_mfcc).
    """
    features = psf_mfcc(signal, samplerate=sr, numcep=n_mfcc)
    return features.astype(np.float32)


def pool_features(mfcc: np.ndarray, mode: str = "mean_std") -> np.ndarray:
    """
    Pool variable-length MFCC into a fixed-size vector.

    mode="mean" -> [mean]
    mode="mean_std" -> [mean, std]
    """
    mean_vec = np.mean(mfcc, axis=0)

    if mode == "mean":
        return mean_vec.astype(np.float32)

    if mode == "mean_std":
        std_vec = np.std(mfcc, axis=0)
        return np.concatenate([mean_vec, std_vec]).astype(np.float32)

    raise ValueError(f"Unsupported pooling mode: {mode}")


def prepare_features(
    dataset: List[Dict], pooling_mode: str = "mean_std"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and raw label vector y from dataset records.
    """
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for item in dataset:
        mfcc = extract_mfcc(item["signal"], item["sample_rate"])
        pooled = pool_features(mfcc, mode=pooling_mode)
        X_list.append(pooled)
        y_list.append(item["label"])

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    return X, y


def split_by_speaker(
    dataset: List[Dict],
    train_ratio: float = 0.8,
    random_seed: int = RANDOM_SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Speaker-independent split: 80% speakers for train, 20% for test.
    """
    speaker_ids = sorted({item["speaker_id"] for item in dataset})
    rng = np.random.default_rng(random_seed)
    shuffled_speakers = speaker_ids.copy()
    rng.shuffle(shuffled_speakers)

    n_train_speakers = int(len(shuffled_speakers) * train_ratio)
    train_speakers = set(shuffled_speakers[:n_train_speakers])
    test_speakers = set(shuffled_speakers[n_train_speakers:])

    if train_speakers & test_speakers:
        raise RuntimeError("Speaker overlap detected between train and test splits.")

    train_data = [item for item in dataset if item["speaker_id"] in train_speakers]
    test_data = [item for item in dataset if item["speaker_id"] in test_speakers]
    return train_data, test_data


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Train baseline model (Logistic Regression).
    """
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
    return model


def evaluate_model(
    model: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate model with accuracy, macro F1, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    return acc, macro_f1, cm


def main() -> None:
    root_path = Path("Radvess")

    # Load full dataset with preprocessed waveforms from the existing pipeline.
    dataset = build_dataset(root_path, load_audio_data=True)

    # Speaker-independent split (critical for SER generalization).
    train_data, test_data = split_by_speaker(dataset)

    # Feature extraction and pooling (default uses mean+std concatenation).
    X_train, y_train_raw = prepare_features(train_data, pooling_mode="mean_std")
    X_test, y_test_raw = prepare_features(test_data, pooling_mode="mean_std")

    # Label encoding.
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    # Train and evaluate baseline model.
    model = train_model(X_train, y_train)
    accuracy, macro_f1, cm = evaluate_model(model, X_test, y_test)

    train_speakers = sorted({item["speaker_id"] for item in train_data})
    test_speakers = sorted({item["speaker_id"] for item in test_data})
    overlap = set(train_speakers) & set(test_speakers)

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of testing samples: {len(test_data)}")
    print(f"Number of training speakers: {len(train_speakers)}")
    print(f"Number of testing speakers: {len(test_speakers)}")
    print(f"Speaker overlap count: {len(overlap)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Label order:")
    print(list(label_encoder.classes_))


if __name__ == "__main__":
    main()
