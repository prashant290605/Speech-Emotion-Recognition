"""The alignment ladder.

Ordered by which moments each rung matches, so the ablation reads as a series
rather than a set of arbitrary points:

    none         nothing
    zscore       per-dimension 1st + 2nd, no cross-terms
    mean_shift   1st only  (= the linear-kernel MMD minimiser, see below)
    coral        1st + 2nd with full covariance
    mkmmd_diag   all moments, but only per-dimension scaling (768 parameters)
    mkmmd_full   all moments, full linear map (~590k parameters)

**`mean_shift` is the linear-kernel MMD minimiser.** Under a linear kernel,
MMD²(X, Y) reduces to ‖μ_X − μ_Y‖², and the translation minimising it is exactly
``X + (μ_Y − μ_X)``. The original study implemented precisely this and reported
it as "MMD". Keeping it as its own rung measures what that column was actually
worth, and there is deliberately **no alias** from any MMD name to it.

Contract every rung obeys:

* ``fit`` takes float64. Caches are float16; :func:`ser.numerics.require_float64`
  raises rather than silently upcasting, so a caller that fitted in half
  precision finds out.
* ``fitted_on_indices`` records every utterance the object was shown. The Phase 2
  assertion ``assert_alignment_blind_to_target_test`` checks it against
  ``target_test``, and rejects an object that does not expose it at all.
* ``diagnostics`` carries covariance conditioning onto the run row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from .mmd import fit_affine_mmd, marginal_mmd
from .numerics import (
    CovarianceDiagnostics,
    covariance,
    diagnose,
    inverse_sqrt_psd,
    ledoit_wolf_covariance,
    require_float64,
    shrink,
    sqrt_psd,
)

__all__ = [
    "Alignment",
    "NoAlignment",
    "ZScoreAlignment",
    "MeanShiftAlignment",
    "CoralAlignment",
    "MKMMDAlignment",
    "build_alignment",
    "LADDER",
]

LADDER = ("none", "zscore", "mean_shift", "coral", "mkmmd_diag", "mkmmd_full")

SOURCE = "source"
TARGET = "target"


class Alignment:
    """Base class. Subclasses implement ``_fit`` and ``transform``."""

    name = "base"

    def __init__(self) -> None:
        self.fitted_on_indices: Set[str] = set()
        self.diagnostics: Dict[str, object] = {}
        self._fitted = False

    # -- contract ----------------------------------------------------------
    def fit(
        self,
        X_source: np.ndarray,
        X_target_adapt: np.ndarray,
        target_adapt_indices: Iterable[str],
        source_indices: Iterable[str] = (),
    ) -> "Alignment":
        """Fit on source and target-adapt only.

        ``source_indices`` extends the brief's signature so ``fitted_on_indices``
        records the *whole* set the object saw, not just the target half. The
        leakage assertion then checks both that ``target_test`` is absent and
        that nothing outside the split leaked in.
        """
        require_float64(X_source, f"{self.name}.X_source")
        require_float64(X_target_adapt, f"{self.name}.X_target_adapt")

        self.fitted_on_indices = set(target_adapt_indices) | set(source_indices)
        self._fit(X_source, X_target_adapt)
        self._fitted = True
        return self

    def _fit(self, X_source: np.ndarray, X_target_adapt: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, X: np.ndarray, domain: str = SOURCE) -> np.ndarray:
        """Apply the alignment.

        ``domain`` extends the brief's single-argument signature because
        ``zscore`` is definitionally two maps -- each corpus standardised by its
        own statistics -- and a one-argument transform would have to guess which
        side it was handed. Guessing is the class of implicit behaviour this
        rebuild removes. Rungs that only touch the source ignore it.
        """
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: transform called before fit")

    def row_fields(self) -> Dict[str, Optional[float]]:
        """Conditioning columns for the result row."""
        return {
            "cov_condition_number": self.diagnostics.get("condition_number"),
            "cov_effective_rank": self.diagnostics.get("effective_rank"),
        }


class NoAlignment(Alignment):
    """Identity. The control the whole ladder is measured against."""

    name = "none"

    def _fit(self, X_source, X_target_adapt) -> None:
        return None

    def transform(self, X: np.ndarray, domain: str = SOURCE) -> np.ndarray:
        self._check_fitted()
        return np.asarray(X, dtype=np.float64)


class ZScoreAlignment(Alignment):
    """Per-corpus standardisation. No covariance, no cross-terms.

    The control that decides whether CORAL's gains were ever real: if z-scoring
    recovers most of the improvement, the covariance matching was doing little.
    """

    name = "zscore"

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.stats: Dict[str, Dict[str, np.ndarray]] = {}

    def _fit(self, X_source, X_target_adapt) -> None:
        for key, X in ((SOURCE, X_source), (TARGET, X_target_adapt)):
            std = X.std(axis=0)
            # A constant dimension has no scale to normalise; dividing by ~0
            # would manufacture enormous values from numerical noise.
            std = np.where(std < self.eps, 1.0, std)
            self.stats[key] = {"mean": X.mean(axis=0), "std": std}
        self.diagnostics = {
            "n_constant_dims_source": int(
                (X_source.std(axis=0) < self.eps).sum()
            ),
            "n_constant_dims_target": int(
                (X_target_adapt.std(axis=0) < self.eps).sum()
            ),
        }

    def transform(self, X: np.ndarray, domain: str = SOURCE) -> np.ndarray:
        self._check_fitted()
        if domain not in self.stats:
            raise ValueError(f"{self.name}: unknown domain {domain!r}")
        stats = self.stats[domain]
        return (np.asarray(X, dtype=np.float64) - stats["mean"]) / stats["std"]


class MeanShiftAlignment(Alignment):
    """``X_src + (μ_tgt − μ_src)``. First moment only.

    This is exactly the minimiser of linear-kernel MMD, since that kernel
    reduces MMD² to ‖μ_s − μ_t‖². It is what the original study shipped under
    the name "MMD"; it is kept here as its own rung so the difference between it
    and a real multi-kernel MMD is measured rather than asserted.
    """

    name = "mean_shift"

    def __init__(self) -> None:
        super().__init__()
        self.offset: Optional[np.ndarray] = None

    def _fit(self, X_source, X_target_adapt) -> None:
        self.offset = X_target_adapt.mean(axis=0) - X_source.mean(axis=0)
        self.diagnostics = {"offset_norm": float(np.linalg.norm(self.offset))}

    def transform(self, X: np.ndarray, domain: str = SOURCE) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return X + self.offset if domain == SOURCE else X


class CoralAlignment(Alignment):
    """Whiten the source covariance, recolour with the target's.

    Regularisation is **mandatory**: d=768 against ~1000 source-train samples
    gives a covariance of rank at most n−1, so the unregularised whitening step
    is not merely ill-conditioned but singular. Two variants:

    * ``shrinkage`` — ``Cov + eps · trace(Cov)/d · I`` with ``eps`` searched on
      the grid and recorded in the run row.
    * ``ledoit_wolf`` — analytic, parameter-free.

    Conditioning of both covariances is measured after regularisation and
    recorded. A matrix still singular at that point raises rather than
    pseudo-inverting.
    """

    name = "coral"

    def __init__(self, *, eps: Optional[float] = None, ledoit_wolf: bool = False) -> None:
        super().__init__()
        if ledoit_wolf and eps is not None:
            raise ValueError("choose either a shrinkage eps or Ledoit-Wolf, not both")
        if not ledoit_wolf and eps is None:
            raise ValueError(
                "CORAL requires regularisation: pass eps, or ledoit_wolf=True. "
                "An unregularised 768-dimensional covariance from ~1000 samples "
                "is singular."
            )
        self.eps = eps
        self.ledoit_wolf = ledoit_wolf
        self.mean_source: Optional[np.ndarray] = None
        self.mean_target: Optional[np.ndarray] = None
        self.transform_matrix: Optional[np.ndarray] = None

    def _regularised(self, X: np.ndarray, label: str):
        raw = covariance(X, name=label)
        raw_diagnostics = diagnose(raw, n_samples=X.shape[0])

        if self.ledoit_wolf:
            regularised, shrinkage = ledoit_wolf_covariance(X, name=label)
            extra = {"ledoit_wolf_shrinkage": shrinkage}
        else:
            regularised = shrink(raw, self.eps)
            extra = {"eps": self.eps}

        return regularised, raw_diagnostics, diagnose(regularised, X.shape[0]), extra

    def _fit(self, X_source, X_target_adapt) -> None:
        self.mean_source = X_source.mean(axis=0)
        self.mean_target = X_target_adapt.mean(axis=0)

        cov_s, raw_s, reg_s, extra_s = self._regularised(X_source, "source covariance")
        cov_t, raw_t, reg_t, _ = self._regularised(X_target_adapt, "target covariance")

        whiten = inverse_sqrt_psd(cov_s, label="source covariance")
        recolour = sqrt_psd(cov_t, label="target covariance")
        self.transform_matrix = whiten @ recolour

        self.diagnostics = {
            **extra_s,
            "variant": "ledoit_wolf" if self.ledoit_wolf else "shrinkage",
            # Reported conditioning is the worst of the two, after
            # regularisation -- the number that governs whether whitening was
            # numerically meaningful.
            "condition_number": max(reg_s.condition_number, reg_t.condition_number),
            "effective_rank": raw_s.effective_rank,
            "source_raw": raw_s.as_dict(),
            "target_raw": raw_t.as_dict(),
            "source_regularised": reg_s.as_dict(),
            "target_regularised": reg_t.as_dict(),
        }

    def transform(self, X: np.ndarray, domain: str = SOURCE) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if domain != SOURCE:
            return X
        return (X - self.mean_source) @ self.transform_matrix + self.mean_target


class MKMMDAlignment(Alignment):
    """Affine map fitted by minimising multi-kernel MMD.

    ``diagonal=True`` restricts W to a per-dimension scaling: 768 parameters
    instead of ~590k. If it matches the full map, the alignment was only ever
    rescaling individual dimensions, which is a finding about how much structure
    the method uses.
    """

    def __init__(self, *, lam: float, diagonal: bool, config, seed: int = 0) -> None:
        super().__init__()
        self.name = "mkmmd_diag" if diagonal else "mkmmd_full"
        self.lam = lam
        self.diagonal = diagonal
        self.config = config
        self.seed = seed
        self.result = None

    def coral_warm_start(self, coral: "CoralAlignment"):
        """(W, b) equivalent to a fitted CORAL, for warm starting.

        CORAL computes ``(x - mu_s) @ M + mu_t``; the affine form is
        ``x @ W.T + b``. So ``W = M.T`` **and** ``b = mu_t - mu_s @ M`` -- the
        bias term is not optional, and omitting it initialises somewhere that is
        not CORAL at all.
        """
        M = coral.transform_matrix
        return M.T.copy(), (coral.mean_target - coral.mean_source @ M).copy()

    def _fit(self, X_source, X_target_adapt) -> None:
        W_init = b_init = None
        warm_start = "identity"

        if self.config.alignment.mmd_warm_start == "coral":
            if self.diagonal:
                # The diagonal analogue of CORAL: per-dimension moment matching,
                # w_j = sigma_t,j / sigma_s,j and b_j = mu_t,j - w_j * mu_s,j.
                # CORAL's dense solution cannot be projected onto a diagonal
                # without becoming a different transform, but this is the same
                # idea within the diagonal family -- and without it the diagonal
                # rung starts at the identity and, at a step size scaled for 768
                # parameters, cannot travel far enough in the step budget.
                std_source = X_source.std(axis=0)
                std_source = np.where(std_source < 1e-12, 1.0, std_source)
                W_init = X_target_adapt.std(axis=0) / std_source
                b_init = X_target_adapt.mean(axis=0) - W_init * X_source.mean(axis=0)
                warm_start = "diagonal_moment_match"
            else:
                coral = CoralAlignment(
                    eps=min(self.config.alignment.coral_shrinkage)
                ).fit(X_source, X_target_adapt, (), ())
                W_init, b_init = self.coral_warm_start(coral)
                warm_start = "coral"

        self.result = fit_affine_mmd(
            X_source,
            X_target_adapt,
            self.config,
            lam=self.lam,
            diagonal=self.diagonal,
            seed=self.seed,
            W_init=W_init,
            b_init=b_init,
        )
        self.diagnostics = {
            "lambda": self.lam,
            "diagonal": self.diagonal,
            "warm_start": warm_start,
            "learning_rate": self.result.learning_rate,
            "converged": self.result.converged,
            "final_grad_norm": self.result.final_grad_norm,
            "bandwidth": self.result.bandwidth,
            "steps": self.result.steps,
            "initial_mmd2": self.result.initial_mmd2,
            "final_mmd2": self.result.final_mmd2,
            "initial_objective": self.result.initial_objective,
            "final_objective": self.result.final_objective,
            "W_deviation_from_identity": float(
                np.linalg.norm(
                    self.result.W - (np.ones_like(self.result.W) if self.diagonal
                                     else np.eye(self.result.W.shape[0]))
                )
            ),
        }

    def transform(self, X: np.ndarray, domain: str = SOURCE) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return self.result.transform(X) if domain == SOURCE else X


def build_alignment(
    method: str,
    config,
    *,
    eps: Optional[float] = None,
    ledoit_wolf: bool = False,
    lam: Optional[float] = None,
    seed: int = 0,
) -> Alignment:
    """Construct a rung by name. Unknown names raise rather than defaulting."""
    if method == "none":
        return NoAlignment()
    if method == "zscore":
        return ZScoreAlignment()
    if method == "mean_shift":
        return MeanShiftAlignment()
    if method == "coral":
        return CoralAlignment(eps=eps, ledoit_wolf=ledoit_wolf)
    if method in ("mkmmd_diag", "mkmmd_full"):
        if lam is None:
            raise ValueError(f"{method} requires a lambda; it is a searched axis")
        return MKMMDAlignment(
            lam=lam, diagonal=(method == "mkmmd_diag"), config=config, seed=seed
        )
    raise ValueError(f"unknown alignment {method!r}; the ladder is {list(LADDER)}")
