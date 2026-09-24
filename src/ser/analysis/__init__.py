"""Analysis-only code. Nothing here may be imported by the pipeline.

This package exists to satisfy amendment **A10**. The conditional-shift
diagnostic computes ``MMD(X_src | y=k, X_tgt | y=k)``, which requires **target
test labels** by construction. That is legitimate as post-hoc analysis and
illegitimate anywhere near fitting or selection -- it is exactly the leak Phase 2
exists to prevent, reintroduced under a respectable name.

The containment is four things, and :func:`assert_conditional_shift_firewall`
checks the ones a machine can check:

1. It lives here, and `alignment`, `classifiers` and `run_grid` do not import
   this package -- verified by reading their source, not by convention.
2. It is never written into an artifact the pipeline reads. The frozen result
   schema has no field for it, and adding one would need a ``SCHEMA_VERSION``
   bump that fails every existing row. That mechanical guard is load-bearing:
   do not weaken it by adding a "diagnostics" column.
3. It is never an input to configuration selection, including Stage 1 axis
   pruning. Not machine-checkable; enforced by the fact that the pruning code
   cannot import this package without failing (1).
4. Values below ``shift.conditional_mmd_min_support`` are reported as undefined
   rather than as a number, and per-class n accompanies every defined value --
   enforced in :func:`ser.analysis.shift.conditional_mmd_by_class`.
"""

from __future__ import annotations

import inspect
from typing import List

__all__ = ["assert_conditional_shift_firewall", "FirewallViolation"]

# Modules that make or select fitted objects. None of them may reach this code.
PIPELINE_MODULES = ("ser.alignment", "ser.classifiers", "ser.run_grid", "ser.blending")


class FirewallViolation(AssertionError):
    """The conditional-shift diagnostic has leaked into the pipeline."""


def assert_conditional_shift_firewall() -> None:
    """Raise unless the A10 containment still holds.

    Called by the analysis entry point and by the test suite. An assertion
    rather than a comment, because A10 requires one.
    """
    import importlib

    from ..utils.results import FIELD_NAMES

    problems: List[str] = []

    # (1) No pipeline module may import this package.
    for name in PIPELINE_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover - a missing module is not a leak
            continue
        try:
            source = inspect.getsource(module)
        except OSError:  # pragma: no cover
            continue
        for marker in ("ser.analysis", "from .analysis", "from ..analysis"):
            if marker in source:
                problems.append(
                    f"{name} references {marker!r}: the conditional-shift "
                    "diagnostic reads target labels and must not be reachable "
                    "from fitting or selection"
                )

    # (2) No result-schema field may carry it into a pipeline-readable artifact.
    leaked = [f for f in FIELD_NAMES if "conditional" in f.lower()]
    if leaked:
        problems.append(
            f"result schema exposes {leaked}: a target-label-derived quantity "
            "must not be written where the pipeline can read it back"
        )

    if problems:
        raise FirewallViolation(
            "A10 conditional-shift firewall violated:\n  - " + "\n  - ".join(problems)
        )
