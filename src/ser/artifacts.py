"""What the published analysis reads, and which frozen experiment it reads.

One module so that "what does a clean clone need?" has a single answer that can
be checked, rather than being distributed across the tools as glob patterns and
string literals. Three things live here:

**The tracked analysis inputs.** Every artifact the manuscript's tables and
figures are generated from, with a note on what it feeds. :func:`missing` turns
that list into a check the build wrapper can run before doing any work, so a
reviewer with an incomplete checkout is told what is missing instead of reading
a traceback from inside a plotting routine.

**The layer-sweep loader.** The sweep used to be read straight out of
``results/shards/``, which is gitignored, so one manuscript subsection was not
reproducible from a checkout. :func:`read_layer_sweep` prefers the tracked
canonical artifact and falls back to the shards only when it is absent, saying
which it used.

**The publication freeze tag.** ``grid-freeze-v3`` was hardcoded in two
generators. It is still the default and the published build still resolves to
it, but it is now a named default with an override, so a later prospective
experiment can be analysed without editing the plotting code and, more
importantly, without any path by which its rows could be averaged in with the
frozen grid's. :func:`select_freeze_tag` is deliberately strict: it refuses an
unknown tag rather than silently returning an empty selection, because an
empty selection downstream looks like a filter working correctly.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import repo_root

__all__ = [
    "PUBLICATION_FREEZE_TAG",
    "LAYER_SWEEP",
    "LAYER_SWEEP_PROVENANCE",
    "LAYER_SWEEP_SHARDS",
    "EPS_PROBE",
    "EPS_PROBE_PROVENANCE",
    "EPS_PROBE_SOURCES",
    "REQUIRED_ARTIFACTS",
    "UnknownFreezeTag",
    "missing",
    "read_layer_sweep",
    "read_eps_probe",
    "select_freeze_tag",
    "filter_by_freeze_tag",
]

# The frozen confirmatory grid this manuscript reports. Changing this constant
# changes which experiment the paper is about, which is why it is a constant
# with a name rather than a literal in a plotting script.
PUBLICATION_FREEZE_TAG = "grid-freeze-v3"

LAYER_SWEEP = "results/layer_sweep_v2.jsonl"
LAYER_SWEEP_PROVENANCE = "results/layer_sweep_v2.provenance.json"
LAYER_SWEEP_SHARDS = "results/shards/sweep2_*.jsonl"

# The CORAL shrinkage probe has the same history as the sweep: 35 of its 120
# rows were tracked and the other 85 lived only in results/shards/, so the
# epsilon-asymptote table could not be regenerated from a checkout either.
EPS_PROBE = "results/eps_asymptote_full.jsonl"
EPS_PROBE_PROVENANCE = "results/eps_asymptote_full.provenance.json"
EPS_PROBE_SOURCES = ("results/eps_*.jsonl", "results/shards/eps_*.jsonl")

# Tracked inputs, and what each one is for. Ordered as the analysis consumes
# them. Anything that is NOT here either is not needed for the manuscript or is
# documented as unavailable without the raw corpora.
REQUIRED_ARTIFACTS: Sequence[tuple[str, str]] = (
    ("results/runs.jsonl", "frozen confirmatory grid: headline, ladder, audit"),
    ("results/layer_sweep_v2.jsonl", "13-layer sweep: frame-dependence table and figure"),
    ("results/speaker_confusions.jsonl.gz", "per-speaker confusions: cluster bootstraps"),
    ("results/eps_asymptote_full.jsonl", "CORAL shrinkage probe, all 120 rows"),
    ("results/phase9_shift.jsonl", "shift decomposition"),
    ("results/phase9_reference_geometry_controls_v2.jsonl", "fixed-reference MMD diagnostics"),
    ("results/calm_dropped_sensitivity.jsonl", "calm-drop label control"),
    ("results/calm_dropped_diagnostics.jsonl", "calm-drop MMD diagnostics"),
    ("results/neutral_excluded_sensitivity_v2.jsonl", "neutral-exclusion label control"),
    ("results/neutral_excluded_diagnostics.jsonl", "neutral-exclusion MMD diagnostics"),
    ("results/stage2_calibration.jsonl", "stage 2 calibration probe"),
    ("data/manifest_portable.csv", "portable scientific manifest: speakers and labels"),
    ("configs/FROZEN_LEDGER.sha256", "expected digest of the frozen ledger"),
    ("configs/default.yaml", "frozen experimental configuration"),
)


class UnknownFreezeTag(ValueError):
    """A freeze tag was requested that no row in the source carries."""


def missing(root: Optional[Path] = None) -> List[tuple[str, str]]:
    """Required artifacts that are absent, as ``(path, purpose)`` pairs."""
    base = root or repo_root()
    return [(path, why) for path, why in REQUIRED_ARTIFACTS if not (base / path).exists()]


def read_layer_sweep(root: Optional[Path] = None) -> tuple[List[Dict[str, Any]], str]:
    """Return ``(rows, source_description)`` for the 13-layer sweep.

    Prefers the tracked canonical artifact. Falls back to the gitignored shards
    when it is absent, because a working copy that has the shards but has not
    run the merge should still be able to generate figures -- but the caller is
    told which source was used, so a report cannot silently be built from
    something a reviewer does not have.
    """
    base = root or repo_root()
    canonical = base / LAYER_SWEEP
    if canonical.exists():
        rows = [json.loads(line) for line in
                canonical.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows, LAYER_SWEEP

    unique: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(str(base / LAYER_SWEEP_SHARDS))):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    unique[row["run_id"]] = row
    if not unique:
        raise FileNotFoundError(
            f"no layer sweep found: neither {LAYER_SWEEP} nor {LAYER_SWEEP_SHARDS}. "
            "Run `python tools/merge_layer_sweep.py` if the shards are present."
        )
    return list(unique.values()), f"{LAYER_SWEEP_SHARDS} (untracked fallback)"


def select_freeze_tag(rows: Iterable[Dict[str, Any]], tag: Optional[str] = None) -> str:
    """Resolve the freeze tag to analyse, refusing an absent one.

    ``None`` means the published default. An explicit tag that no row carries
    raises rather than returning an empty selection: downstream, an empty
    selection is indistinguishable from a filter that worked, and the first
    symptom would be a plot with no points rather than an error.
    """
    wanted = PUBLICATION_FREEZE_TAG if tag is None else tag
    present = {row.get("freeze_tag") for row in rows}
    if wanted not in present:
        raise UnknownFreezeTag(
            f"no rows carry freeze_tag {wanted!r}. Present: "
            f"{sorted(t for t in present if t is not None)}. "
            "Refusing to continue with an empty selection."
        )
    return wanted


def filter_by_freeze_tag(rows: Sequence[Dict[str, Any]],
                         tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """Rows of exactly one freeze tag. Never a union of several.

    The single-tag restriction is the point. Two freeze tags in one summary
    would be two different experimental configurations averaged together under
    one number, which is the failure the config freeze exists to prevent; doing
    it at analysis time instead of run time would not make it acceptable.
    """
    resolved = select_freeze_tag(rows, tag)
    return [row for row in rows if row.get("freeze_tag") == resolved]


def read_eps_probe(root: Optional[Path] = None) -> tuple[List[Dict[str, Any]], str]:
    """Return ``(rows, source_description)`` for the CORAL shrinkage probe.

    Prefers the tracked canonical artifact; falls back to the union of the
    partially-tracked ledger and the gitignored shards, which is what the
    figure code used to do unconditionally.
    """
    base = root or repo_root()
    canonical = base / EPS_PROBE
    if canonical.exists():
        rows = [json.loads(line) for line in
                canonical.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows, EPS_PROBE

    unique: Dict[str, Dict[str, Any]] = {}
    for pattern in EPS_PROBE_SOURCES:
        for path in sorted(glob.glob(str(base / pattern))):
            if Path(path).name == Path(EPS_PROBE).name:
                continue
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        unique[row["run_id"]] = row
    if not unique:
        raise FileNotFoundError(
            f"no epsilon probe found: neither {EPS_PROBE} nor "
            f"{EPS_PROBE_SOURCES}. Run `python tools/merge_eps_probe.py` if the "
            "shards are present."
        )
    return list(unique.values()), " + ".join(EPS_PROBE_SOURCES) + " (untracked fallback)"
