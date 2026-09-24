"""Classifier families under an equal tuning budget.

Five families -- ``logreg``, ``svm_linear``, ``svm_rbf``, ``mlp``,
``transformer`` -- each searched with **exactly the same number of trials**.

That equality is the point of this module. The original study ran logistic
regression and SVM at library defaults while the neural models got a training
loop, then compared them; the trial count is recorded on every row and a test
asserts every family consumed the identical budget, so the asymmetry cannot
return unnoticed.

Two further rules, both enforced here rather than by convention:

**Selection is on ``source_val`` and nothing else.** Every ``fit_and_select``
call receives a validation split and never sees target data at all. The target
score is computed afterwards, by the caller, from a model that has already been
chosen.

**No standardisation inside the classifier.** The obvious thing is a
``StandardScaler`` in front of every sklearn model, and the original did exactly
that -- but standardisation is the ``zscore`` rung of the alignment ladder. Doing
it here too would make the ``none`` rung unmeasurable and the ``zscore`` rung a
no-op, silently collapsing two conditions the paper reports as distinct.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import macro_f1

__all__ = [
    "FAMILIES",
    "SKLEARN_FAMILIES",
    "TORCH_FAMILIES",
    "TrialRecord",
    "SelectionResult",
    "sample_params",
    "fit_and_select",
    "supports_layer_agg",
    "NotConverged",
]

SKLEARN_FAMILIES = ("logreg", "svm_linear", "svm_rbf")
TORCH_FAMILIES = ("mlp", "transformer")
FAMILIES = SKLEARN_FAMILIES + TORCH_FAMILIES


def supports_layer_agg(family: str, layer_agg: str) -> bool:
    """``weighted`` needs a model with trainable parameters to learn the weights.

    A closed-form sklearn classifier has none, so that combination is not
    enumerated rather than being silently approximated by something else.
    """
    if layer_agg != "weighted":
        return True
    return family in TORCH_FAMILIES


# --------------------------------------------------------------------------
# Search spaces
# --------------------------------------------------------------------------
def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def sample_params(family: str, rng: np.random.Generator, config) -> Dict[str, Any]:
    """One random draw from a family's search space.

    Random search rather than a grid: with a fixed budget it covers each
    dimension at the full budget resolution instead of the grid's per-axis
    fraction, and it makes "the same number of trials" mean the same thing for
    families whose spaces have different dimensionality.
    """
    if family == "logreg":
        # max_iter is deliberately NOT drawn here. It was, from
        # [1000, 2000, 5000], and that was a defect: a convergence budget is not
        # a model choice, and searching it lets a trial win selection by having
        # stopped early rather than by fitting better. It is now a fixed cap
        # (classifiers.sklearn_max_iter) and convergence is asserted.
        return {
            "C": _log_uniform(rng, 1e-4, 1e4),
            "class_weight": rng.choice([None, "balanced"]),
        }
    if family == "svm_linear":
        return {
            "C": _log_uniform(rng, 1e-4, 1e4),
            "class_weight": rng.choice([None, "balanced"]),
        }
    if family == "svm_rbf":
        return {
            "C": _log_uniform(rng, 1e-3, 1e4),
            "gamma": _log_uniform(rng, 1e-6, 1e0),
            "class_weight": rng.choice([None, "balanced"]),
        }
    if family == "mlp":
        return {
            "hidden_dim": int(rng.choice([64, 128, 256, 512])),
            "depth": int(rng.choice([1, 2, 3])),
            "dropout": float(rng.choice([0.0, 0.1, 0.3, 0.5])),
            "lr": _log_uniform(rng, 1e-5, 1e-2),
            "weight_decay": _log_uniform(rng, 1e-6, 1e-1),
            "batch_size": int(rng.choice([32, 64, 128])),
        }
    if family == "transformer":
        heads = int(rng.choice([1, 2, 4]))
        # d_model must be divisible by the head count.
        d_model = int(rng.choice([64, 128, 256])) // heads * heads
        return {
            "d_model": max(d_model, heads),
            "depth": int(rng.choice([1, 2])),
            "heads": heads,
            "dropout": float(rng.choice([0.0, 0.1, 0.3])),
            "lr": _log_uniform(rng, 1e-5, 1e-2),
            "batch_size": int(rng.choice([32, 64])),
        }
    raise ValueError(f"unknown family {family!r}; expected one of {list(FAMILIES)}")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class TrialRecord:
    index: int
    params: Dict[str, Any]
    source_val_macro_f1: float
    epochs_run: Optional[int] = None
    failed: Optional[str] = None
    solver_n_iter: Optional[int] = None


@dataclass
class SelectionResult:
    family: str
    layer_agg: str
    best_params: Dict[str, Any]
    best_source_val_macro_f1: float
    n_trials: int
    trials: List[TrialRecord]
    predict: Callable[[np.ndarray], np.ndarray]
    epochs_run: Optional[int] = None
    # Iterations the selected sklearn solver used. None for the torch families,
    # which report epochs_run instead.
    solver_n_iter: Optional[int] = None

    def as_hyperparams(self) -> Dict[str, Any]:
        """What lands in ``hyperparams_json``."""
        return {
            "family": self.family,
            "layer_agg": self.layer_agg,
            "selected": _jsonable(self.best_params),
            "n_search_trials": self.n_trials,
            "source_val_macro_f1": self.best_source_val_macro_f1,
            "epochs_run": self.epochs_run,
            "n_failed_trials": sum(1 for t in self.trials if t.failed),
            "n_not_converged": sum(
                1 for t in self.trials if t.failed and "max_iter" in t.failed
            ),
            "solver_n_iter": self.solver_n_iter,
            "selection_surface": "source_val",
        }


def _jsonable(params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in params.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, np.str_):
            out[key] = str(value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------
# sklearn families
# --------------------------------------------------------------------------
class NotConverged(RuntimeError):
    """An iterative solver hit its cap instead of converging.

    Raised rather than warned about. A non-converged fit is not a valid
    configuration: its score reflects where the optimiser stopped, not what the
    model can do, and an undertuned simple baseline compared against a trained
    neural model is the exact defect this rebuild exists to correct. The trial
    loop records it as a failed trial and scores it -inf, so it cannot be
    selected and cannot buy the family a retry.
    """


def _build_sklearn(family: str, params: Dict[str, Any], seed: int, config):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC

    class_weight = params.get("class_weight")
    class_weight = None if class_weight in (None, "None") else str(class_weight)
    max_iter = config.classifiers.sklearn_max_iter

    if family == "logreg":
        return LogisticRegression(
            C=params["C"],
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=seed,
        )
    if family == "svm_linear":
        return LinearSVC(
            C=params["C"], class_weight=class_weight, random_state=seed, max_iter=max_iter
        )
    if family == "svm_rbf":
        return SVC(
            C=params["C"],
            gamma=params["gamma"],
            kernel="rbf",
            class_weight=class_weight,
            random_state=seed,
        )
    raise ValueError(family)


def _assert_converged(model, family: str) -> Optional[int]:
    """Iterations the fitted solver used, raising if it hit the cap.

    ``n_iter_`` is per-class for the one-vs-rest solvers, so the maximum over
    classes is what matters -- one class failing to converge is enough to make
    the fit invalid. Models that expose no iteration count (libsvm's SVC on
    older sklearn) return None rather than a fabricated number.
    """
    raw = getattr(model, "n_iter_", None)
    if raw is None:
        return None
    used = int(np.max(np.asarray(raw)))
    cap = getattr(model, "max_iter", None)
    # libsvm's SVC uses max_iter=-1 to mean "no limit". Comparing against that
    # would make every SVC fit look non-converged, which is how this guard
    # failed every svm_rbf trial the first time it was written.
    if cap is not None and int(cap) > 0 and used >= int(cap):
        raise NotConverged(
            f"{family}: solver hit max_iter={cap} without converging "
            f"(n_iter={used}). Raise classifiers.sklearn_max_iter."
        )
    return used


# --------------------------------------------------------------------------
# torch families
# --------------------------------------------------------------------------
def _make_torch_model(family: str, params, n_classes: int, input_shape, n_layers, config):
    """Build an MLP or Transformer head, optionally with learnable layer weights.

    ``input_shape`` is the per-sample shape after loading:
        (d,)              a single layer, MLP
        (L, d)            all layers, MLP with learnable weighting
        (S, d)            a segment sequence, Transformer
        (L, S, d)         all layers x segments, Transformer with weighting
    """
    import torch
    from torch import nn

    class LayerWeights(nn.Module):
        """Softmax over the ``L`` cached hidden states, learned with the head.

        These are the classifier's parameters, not a cache-time constant --
        which is why ``aggregate_layers('weighted')`` returns the unreduced
        stack.
        """

        def __init__(self, n: int) -> None:
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(n))

        def forward(self, x):  # x: (B, L, ...)
            w = torch.softmax(self.logits, dim=0)
            return (x * w.view(1, -1, *([1] * (x.dim() - 2)))).sum(dim=1)

    weighted = len(input_shape) in (2, 3) and input_shape[0] == n_layers

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weighting = LayerWeights(n_layers) if weighted else None
            feature_dim = input_shape[-1]
            sequence = (
                len(input_shape) == 3
                or (len(input_shape) == 2 and not weighted)
            )
            self.sequence = sequence and family == "transformer"

            if family == "mlp":
                layers: List[nn.Module] = []
                in_dim = feature_dim
                for _ in range(params["depth"]):
                    layers += [
                        nn.Linear(in_dim, params["hidden_dim"]),
                        nn.ReLU(),
                        nn.Dropout(params["dropout"]),
                    ]
                    in_dim = params["hidden_dim"]
                layers.append(nn.Linear(in_dim, n_classes))
                self.net = nn.Sequential(*layers)
            else:
                self.project = nn.Linear(feature_dim, params["d_model"])
                encoder = nn.TransformerEncoderLayer(
                    d_model=params["d_model"],
                    nhead=params["heads"],
                    dim_feedforward=params["d_model"] * 2,
                    dropout=params["dropout"],
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder, num_layers=params["depth"])
                self.out = nn.Linear(params["d_model"], n_classes)

        def forward(self, x):
            if self.weighting is not None:
                x = self.weighting(x)
            if family == "mlp":
                if x.dim() == 3:  # segments present; mean over time
                    x = x.mean(dim=1)
                return self.net(x)
            if x.dim() == 2:  # no sequence axis; treat as length 1
                x = x.unsqueeze(1)
            encoded = self.encoder(self.project(x))
            return self.out(encoded.mean(dim=1))

    return Head()


def _fit_torch(
    family, params, X_train, y_train, X_val, y_val, class_names, config, seed, n_layers
):
    """Train with early stopping on ``source_val``. Never a fixed epoch count."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    n_classes = len(class_names)
    lookup = {name: i for i, name in enumerate(class_names)}

    xt = torch.tensor(np.asarray(X_train, dtype=np.float32))
    yt = torch.tensor([lookup[v] for v in y_train], dtype=torch.long)
    xv = torch.tensor(np.asarray(X_val, dtype=np.float32))

    model = _make_torch_model(
        family, params, n_classes, X_train.shape[1:], n_layers, config
    )
    optimiser = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 0.0)
    )
    criterion = nn.CrossEntropyLoss()

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    patience = config.classifiers.early_stopping_patience
    since_improved = 0
    batch = params.get("batch_size", 64)
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, config.classifiers.max_epochs + 1):
        model.train()
        order = torch.randperm(xt.shape[0], generator=generator)
        for start in range(0, xt.shape[0], batch):
            idx = order[start : start + batch]
            optimiser.zero_grad()
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            predictions = model(xv).argmax(dim=1).numpy()
        score = macro_f1(
            list(y_val), [class_names[i] for i in predictions], class_names
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.tensor(np.asarray(X, dtype=np.float32))).argmax(dim=1)
        return np.asarray([class_names[i] for i in out.numpy()])

    return predict, float(best_score), best_epoch


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def fit_and_select(
    family: str,
    X_train: np.ndarray,
    y_train: Sequence[str],
    X_val: np.ndarray,
    y_val: Sequence[str],
    class_names: Sequence[str],
    config,
    *,
    layer_agg: str = "last",
    seed: int = 0,
) -> SelectionResult:
    """Search a family's space and return the configuration best on source_val.

    Consumes exactly ``config.classifiers.search_budget`` trials, identical for
    every family. A trial that fails to fit is recorded and scored -inf rather
    than being retried, so a family cannot quietly buy itself extra attempts.

    No target data reaches this function.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if not supports_layer_agg(family, layer_agg):
        raise ValueError(
            f"layer_agg='weighted' needs learnable parameters; {family} has none. "
            f"Torch families: {list(TORCH_FAMILIES)}"
        )

    budget = config.classifiers.search_budget
    rng = np.random.default_rng(seed)
    class_names = list(class_names)

    trials: List[TrialRecord] = []
    best: Optional[Tuple[float, Dict[str, Any], Callable, Optional[int], Optional[int]]] = None

    for index in range(budget):
        params = sample_params(family, rng, config)
        try:
            if family in SKLEARN_FAMILIES:
                model = _build_sklearn(family, params, seed, config)
                model.fit(X_train, list(y_train))
                # Before the score is read, so a non-converged fit can never be
                # scored, ranked, or selected.
                n_iter = _assert_converged(model, family)
                predictions = model.predict(X_val)
                score = macro_f1(list(y_val), list(predictions), class_names)
                predict = model.predict
                epochs = None
            else:
                predict, score, epochs = _fit_torch(
                    family, params, X_train, y_train, X_val, y_val,
                    class_names, config, seed + index, config.features.n_layers,
                )
                n_iter = None
            trials.append(TrialRecord(index, _jsonable(params), score, epochs, None, n_iter))
        except Exception as exc:  # noqa: BLE001 - a failed trial is data
            trials.append(
                TrialRecord(index, _jsonable(params), float("-inf"), None, str(exc)[:200])
            )
            continue

        if best is None or score > best[0]:
            best = (score, params, predict, epochs, n_iter)

    if best is None:
        raise RuntimeError(
            f"{family}: all {budget} trials failed; see the trial records for why"
        )

    return SelectionResult(
        family=family,
        layer_agg=layer_agg,
        best_params=_jsonable(best[1]),
        best_source_val_macro_f1=float(best[0]),
        n_trials=budget,
        trials=trials,
        predict=best[2],
        epochs_run=best[3],
        solver_n_iter=best[4],
    )
