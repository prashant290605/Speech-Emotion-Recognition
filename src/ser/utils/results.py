"""The frozen result schema and its append-only writer.

Every completed run -- baseline, grid run, or failure -- is one JSON object on
one line of ``results/runs.jsonl``. Every table and figure in the paper is
generated from this file and nothing else. No number is ever typed by hand.

The schema is FROZEN. Adding, removing, or retyping a field invalidates the
grid, because rows written before the change would no longer validate. If a
later phase genuinely needs a new column, bump ``SCHEMA_VERSION`` and write a
migration -- do not quietly widen the schema.

Design notes:

* Metric columns are nullable so a crashed run can still be recorded with
  ``status="failed"`` and a traceback. A silent skip is worse than a recorded
  failure: it leaves a hole in the grid that nobody can see.
* Structured values (per-class F1, confusion matrix, hyperparameters, library
  versions) are stored as JSON *strings* rather than nested objects, so the file
  loads into a flat dataframe with no column explosion and no ragged nesting.
* ``run_id`` is a deterministic function of the experimental coordinates, which
  is what makes the Phase 7 runner resumable: it can rebuild the completed set
  by reading ids off disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

from .runmeta import hash_payload

__all__ = [
    "SCHEMA_VERSION",
    "SMOKE_CORPUS",
    "FIELDS",
    "FIELD_NAMES",
    "RUN_ID_FIELDS",
    "VALID_STATUSES",
    "SchemaError",
    "is_smoke_row",
    "schema_as_markdown",
    "make_run_id",
    "new_row",
    "validate_row",
    "append_row",
    "read_rows",
    "count_rows",
    "completed_run_ids",
]

# v2 (2026-08-10): added label_map_hash and split_spec_hash, both run_id
# coordinates. Without them a changed label decision or split ratio leaves
# run_id unchanged, and a Phase 7 resume silently merges runs scored against
# different label spaces. Bumped before any experimental run existed.
#
# v3 (2026-08-15): added cov_condition_number and cov_effective_rank. With
# d=768 and ~1000 source-train samples, every covariance this project forms is
# rank-deficient, so how badly conditioned it was is part of what a result
# means -- not a debugging detail. Explicit columns rather than a corner of
# hyperparams_json, because "which runs were near-singular?" has to be
# answerable by filtering. Bumped before any alignment run existed; the 60
# baseline rows were regenerated.
#
# v4 (2026-08-15): config_hash REMOVED from RUN_ID_FIELDS (still recorded);
# feature_spec_hash and search_spec_hash added as coordinates in its place;
# marginal_mmd_raw, marginal_mmd_normalised and n_search_trials added.
# config_hash as a coordinate meant editing any unrelated config section
# orphaned every completed run -- observed for real in Phase 5.
SCHEMA_VERSION = 4

VALID_STATUSES = ("ok", "failed")

# `ser smoke` writes a synthetic row to the real results file to prove the
# writer and schema work on the production path. The row is tagged with this
# reserved corpus name so analysis can exclude it mechanically -- filtering by
# convention ("ignore anything that looks like a test") is how fake numbers
# reach a table.
SMOKE_CORPUS = "smoke"


@dataclass(frozen=True)
class Field:
    name: str
    types: tuple[type, ...]
    nullable: bool
    doc: str


def _f(name: str, types, nullable: bool, doc: str) -> Field:
    return Field(name, types if isinstance(types, tuple) else (types,), nullable, doc)


# --------------------------------------------------------------------------
# THE FROZEN SCHEMA
# --------------------------------------------------------------------------
FIELDS: tuple[Field, ...] = (
    # -- identity and provenance -------------------------------------------
    _f("schema_version", int, False, "Result schema version; bump on any change."),
    _f("run_id", str, False, "Deterministic id over the experimental coordinates."),
    _f("git_sha", str, False, "Commit that produced the row, or 'unknown'."),
    _f("git_dirty", bool, False, "True if the working tree had uncommitted changes."),
    _f("config_hash", str, False, "sha256 of the whole config. Recorded, NOT a run_id coordinate."),
    _f("label_map_hash", str, False, "Hash of the resolved label mapping."),
    _f("split_spec_hash", str, False, "Hash of the split specification."),
    _f("feature_spec_hash", str, False, "Hash of the feature extraction spec."),
    _f("search_spec_hash", str, False, "Hash of the searched space and reported statistics."),
    _f("timestamp", str, False, "ISO-8601 UTC completion time."),
    _f("hostname", str, False, "Machine that ran it."),
    _f("lib_versions_json", str, False, "JSON map of tracked library versions."),
    _f("seed", int, False, "Seed passed to set_all_seeds for this run."),
    # -- data ---------------------------------------------------------------
    _f("source_corpus", str, False, "Corpus supplying labelled training data."),
    _f("target_corpus", str, False, "Corpus evaluated on. Equal to source for in-domain."),
    _f("n_classes", int, False, "Size of the label space actually used."),
    _f("class_names", list, False, "Ordered class names; indexes confusion_json."),
    # -- features -----------------------------------------------------------
    _f("backbone", str, False, "hubert | wav2vec2 | wavlm | mfcc."),
    _f("layer_agg", str, False, "last | layer | mean | weighted | n/a."),
    _f("layer_index", int, True, "Layer used when layer_agg='layer', else null."),
    _f("feature_branch", str, False, "ssl | mfcc | fused."),
    # -- alignment and blending ---------------------------------------------
    _f("alignment", str, False, "none | zscore | coral | mmd."),
    _f("blending", str, False, "none | scalar | gaa."),
    _f("blend_alpha", (float, int), True, "Scalar alpha, or null for none/gaa."),
    _f("n_groups", int, True, "Group count when blending='gaa', else null."),
    # -- classifier ---------------------------------------------------------
    _f("classifier", str, False, "logreg | svm | mlp | transformer | baseline_*."),
    _f("hyperparams_json", str, False, "JSON of the config selected on source_val."),
    # -- numerical conditioning (null when no covariance was formed) --------
    _f(
        "cov_condition_number",
        (float, int),
        True,
        "Worst condition number over covariances formed, after regularisation.",
    ),
    _f(
        "cov_effective_rank",
        (float, int),
        True,
        "Spectral-entropy effective rank of the source covariance (Roy & Vetterli).",
    ),
    _f(
        "n_search_trials",
        int,
        True,
        "Hyperparameter configurations evaluated on source_val. Equal across families.",
    ),
    # -- covariate shift, at a bandwidth fixed once on the unaligned pair ----
    _f(
        "marginal_mmd_raw",
        (float, int),
        True,
        "Marginal MMD^2 between aligned source and target.",
    ),
    _f(
        "marginal_mmd_normalised",
        (float, int),
        True,
        "marginal_mmd_raw / typical same-distribution MMD^2. Scale-invariant.",
    ),
    # -- splits -------------------------------------------------------------
    _f("split_id", str, False, "Identifies the speaker-disjoint split realisation."),
    _f("n_train", int, False, "Utterances in source_train."),
    _f("n_val", int, False, "Utterances in source_val."),
    _f("n_target_adapt", int, False, "Utterances alignment was allowed to see."),
    _f("n_target_test", int, False, "Utterances scored on."),
    # -- metrics (null iff status='failed') ---------------------------------
    _f("macro_f1", (float, int), True, "Primary metric, on target_test."),
    _f("accuracy", (float, int), True, "Accuracy on target_test."),
    _f("uar", (float, int), True, "Unweighted average recall on target_test."),
    _f("per_class_f1_json", str, True, "JSON map class name -> F1."),
    _f("confusion_json", str, True, "JSON nested list, rows=true, cols=predicted."),
    # -- floors -------------------------------------------------------------
    _f("chance_macro_f1", (float, int), True, "Uniform-random macro-F1 floor."),
    _f("majority_macro_f1", (float, int), True, "Majority-class collapse floor."),
    _f("prior_matched_macro_f1", (float, int), True, "Source-prior-sampling floor."),
    # -- selection ----------------------------------------------------------
    _f(
        "selection_source_val_macro_f1",
        (float, int),
        True,
        "source_val macro-F1 of the selected config. NEVER a target quantity.",
    ),
    # -- execution ----------------------------------------------------------
    _f("wall_seconds", (float, int), False, "Wall-clock seconds for the run."),
    _f("status", str, False, "ok | failed."),
    _f("error", str, True, "Traceback when status='failed', else null."),
)

FIELD_NAMES: tuple[str, ...] = tuple(field.name for field in FIELDS)

_FIELDS_BY_NAME: Dict[str, Field] = {field.name: field for field in FIELDS}

# Coordinates that define "the same run". Deliberately excludes hyperparameters
# (searched inside a run on source_val, so they are an output not a coordinate),
# metrics, and provenance (which vary between reruns of an identical config).
#
# The four facet hashes replace config_hash, which was too coarse to be a
# coordinate: it changes when ANY key changes, so editing an unrelated section
# orphaned 60 completed baseline rows in Phase 5. Each facet pins one part of
# the semantics that actually determines what a row means, so an edit
# invalidates only the runs it can affect. ser.config.Config.FACET_SECTIONS
# maps them to config sections, and a test asserts every config key is either
# covered by a facet or explicitly declared inert.
#
# Note on gaa: per-group alphas are selected on source_val inside the run, so
# they live in hyperparams_json, not here. Only the scalar blend_alpha axis is a
# coordinate.
RUN_ID_FIELDS: tuple[str, ...] = (
    # config_hash is deliberately ABSENT. See SCHEMA_VERSION v4 and PHASES.md A6.
    "label_map_hash",
    "split_spec_hash",
    "feature_spec_hash",
    "search_spec_hash",
    "seed",
    "source_corpus",
    "target_corpus",
    "backbone",
    "layer_agg",
    "layer_index",
    "feature_branch",
    "alignment",
    "blending",
    "blend_alpha",
    "n_groups",
    "classifier",
    "split_id",
)


class SchemaError(ValueError):
    """A row does not conform to the frozen schema."""


def make_run_id(coords: Dict[str, Any]) -> str:
    """Deterministic 16-hex-char id over :data:`RUN_ID_FIELDS`.

    Two invocations with the same coordinates produce the same id on any
    machine, which is what lets the Phase 7 runner skip completed work after a
    kill. Missing coordinates are an error, not a default -- a silently defaulted
    coordinate would collide two genuinely different runs onto one id.
    """
    missing = [name for name in RUN_ID_FIELDS if name not in coords]
    if missing:
        raise SchemaError(f"make_run_id missing coordinates: {missing}")
    payload = {name: coords[name] for name in RUN_ID_FIELDS}
    return hash_payload(payload)[:16]


def new_row(**values: Any) -> Dict[str, Any]:
    """Build a schema-shaped row.

    Every field is present. Nullable fields default to ``None``; non-nullable
    fields must be supplied. ``schema_version`` and ``status`` are filled in.
    The result is validated before it is returned, so a malformed row fails at
    construction rather than at write time.
    """
    values.setdefault("schema_version", SCHEMA_VERSION)
    values.setdefault("status", "ok")

    unknown = sorted(set(values) - set(FIELD_NAMES))
    if unknown:
        raise SchemaError(
            f"unknown field(s) {unknown}; the schema is frozen -- see results.py"
        )

    row = {name: values.get(name) for name in FIELD_NAMES}
    validate_row(row)
    return row


def validate_row(row: Dict[str, Any]) -> None:
    """Raise :class:`SchemaError` unless ``row`` conforms exactly."""
    if not isinstance(row, dict):
        raise SchemaError(f"row must be a dict, got {type(row).__name__}")

    missing = sorted(set(FIELD_NAMES) - set(row))
    if missing:
        raise SchemaError(f"missing field(s): {missing}")

    extra = sorted(set(row) - set(FIELD_NAMES))
    if extra:
        raise SchemaError(f"unexpected field(s): {extra}")

    for name, value in row.items():
        field = _FIELDS_BY_NAME[name]
        if value is None:
            if not field.nullable:
                raise SchemaError(f"field '{name}' is not nullable")
            continue
        if not _type_ok(value, field.types):
            expected = "|".join(t.__name__ for t in field.types)
            raise SchemaError(
                f"field '{name}' expected {expected}, got "
                f"{type(value).__name__} ({value!r})"
            )

    if row["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {row['schema_version']} != {SCHEMA_VERSION}"
        )

    if row["status"] not in VALID_STATUSES:
        raise SchemaError(f"status must be one of {VALID_STATUSES}, got {row['status']!r}")

    if row["status"] == "failed" and not row["error"]:
        raise SchemaError("status='failed' requires a non-empty 'error'")

    if row["status"] == "ok" and row["macro_f1"] is None:
        raise SchemaError("status='ok' requires a macro_f1")

    if not all(isinstance(name, str) for name in row["class_names"]):
        raise SchemaError("class_names must be a list of strings")

    if len(row["class_names"]) != row["n_classes"]:
        raise SchemaError(
            f"n_classes={row['n_classes']} does not match "
            f"len(class_names)={len(row['class_names'])}"
        )


def _type_ok(value: Any, types: tuple[type, ...]) -> bool:
    # bool is a subclass of int in Python; an int column must not silently
    # accept True. Only allow a bool where bool is explicitly declared.
    if isinstance(value, bool) and bool not in types:
        return False
    return isinstance(value, types)


def append_row(path: str | os.PathLike, row: Dict[str, Any], *, validate: bool = True) -> None:
    """Append one validated row to a JSONL file.

    Append-only by construction: the file is opened in ``"a"`` mode and is never
    read, rewritten, or truncated by this module. Concurrent writers are a
    Phase 7 concern (file locking); a single writer is safe here because one
    ``write`` of a sub-4KiB line in append mode is atomic on both POSIX and NTFS.
    """
    if validate:
        validate_row(row)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps({name: row[name] for name in FIELD_NAMES}, separators=(",", ":"))
    if "\n" in line:  # pragma: no cover - json.dumps escapes newlines
        raise SchemaError("serialised row contains a newline")

    with open(target, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: str | os.PathLike, *, validate: bool = False) -> Iterator[Dict[str, Any]]:
    """Yield rows from a JSONL file. Missing file yields nothing."""
    target = Path(path)
    if not target.exists():
        return

    with open(target, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{target}:{line_no} is not valid JSON: {exc}") from exc
            if validate:
                try:
                    validate_row(row)
                except SchemaError as exc:
                    raise SchemaError(f"{target}:{line_no} {exc}") from exc
            yield row


def is_smoke_row(row: Dict[str, Any]) -> bool:
    """True for synthetic rows written by `ser smoke`. Excluded from analysis."""
    return row.get("source_corpus") == SMOKE_CORPUS or row.get("target_corpus") == SMOKE_CORPUS


def count_rows(path: str | os.PathLike) -> int:
    return sum(1 for _ in read_rows(path))


def completed_run_ids(path: str | os.PathLike) -> set[str]:
    """Run ids already on disk. Used by the Phase 7 runner to resume."""
    return {row["run_id"] for row in read_rows(path) if "run_id" in row}


def schema_as_markdown() -> str:
    """Render the frozen schema as a table, for the reproducibility appendix."""
    lines = [
        f"Result schema version {SCHEMA_VERSION} ({len(FIELDS)} fields)",
        "",
        "| field | type | nullable | meaning |",
        "|---|---|---|---|",
    ]
    for field in FIELDS:
        types = " \\| ".join(t.__name__ for t in field.types)
        lines.append(
            f"| `{field.name}` | {types} | {'yes' if field.nullable else 'no'} | {field.doc} |"
        )
    return "\n".join(lines)


def field_names() -> Sequence[str]:
    return FIELD_NAMES
