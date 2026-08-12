from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model
