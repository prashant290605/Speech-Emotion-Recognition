"""Dataclass-backed configuration loaded from YAML.

Two rules this module exists to enforce:

1. Every experimental value lives in a config file, never in a script. A result
   is reproducible from ``(config file, seed)`` and nothing else.
2. No magic strings. Loading is *strict*: an unknown key, a missing key, or a
   value outside its allowed set raises at load time rather than silently
   falling back to a default. A typo in a config must fail the run, not quietly
   change the experiment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .utils.runmeta import hash_payload

__all__ = ["Config", "ConfigError", "load_config", "repo_root"]

DEFAULT_CONFIG_PATH = "configs/default.yaml"


class ConfigError(ValueError):
    """The config file is malformed, incomplete, or internally inconsistent."""


def repo_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    results_path: str


@dataclass(frozen=True)
class PathsConfig:
    raw_ravdess: str
    raw_cremad: str
    raw_iemocap: str
    manifest: str
    cache_dir: str
    reports_dir: str
    figures_dir: str


@dataclass(frozen=True)
class LabelsConfig:
    """Label spaces and the corpus-specific mapping decisions.

    The entries in :data:`DECISION_FIELDS` are paper-level decisions, not code
    defaults: each one materially changes class support and therefore every
    downstream number. A ``null`` means undecided, and
    :meth:`Config.require_decision` halts rather than guessing.

    ``label_map_version`` is bumped by hand whenever the *semantics* of
    ``ser.labels.map_label`` change in a way the config keys do not express.
    It feeds :attr:`Config.label_map_hash`, which is a ``run_id`` coordinate --
    so a mapping change forces new run ids instead of silently merging
    incompatible runs during a Phase 7 resume.
    """

    # Decisions a human must make. Anything else in this section is machinery.
    DECISION_FIELDS = (
        "iemocap_label_source",
        "iemocap_excited_to_happy",
        "iemocap_frustrated",
        "iemocap_subsets",
        "iemocap_record_subset",
        "ravdess_calm_to_neutral",
    )

    # How an IEMOCAP utterance gets its categorical label. This determines the
    # counts, so it determines the priors, so it determines the entire shift
    # analysis -- it is not a detail. It lives in this section specifically so
    # that it falls inside label_map_hash.
    LABEL_SOURCES = (
        "majority_vote_discard_disagreement",
        "majority_vote_with_self_assessment",
        "any_annotator",
        "self_assessment",
    )

    label_map_version: str
    spaces: Dict[str, List[str]]
    space_for_iemocap_pairs: str
    space_for_other_pairs: str
    min_class_support_warn: int
    iemocap_label_source: Optional[str] = None
    iemocap_excited_to_happy: Optional[bool] = None
    iemocap_frustrated: Optional[str] = None  # "drop" | "merge_angry"
    iemocap_subsets: Optional[str] = None  # "scripted" | "improvised" | "both"
    iemocap_record_subset: Optional[bool] = None
    ravdess_calm_to_neutral: Optional[bool] = None

    def __post_init__(self) -> None:
        for name, classes in self.spaces.items():
            if sorted(classes) != sorted(set(classes)):
                raise ConfigError(f"labels.spaces.{name} contains duplicates")
            if list(classes) != sorted(classes):
                raise ConfigError(
                    f"labels.spaces.{name} must be sorted; class order is the "
                    "index order of every confusion matrix in the paper"
                )
        for key in ("space_for_iemocap_pairs", "space_for_other_pairs"):
            space = getattr(self, key)
            if space not in self.spaces:
                raise ConfigError(
                    f"labels.{key}='{space}' is not a key of labels.spaces "
                    f"({sorted(self.spaces)})"
                )
        if self.iemocap_subsets not in (None, "scripted", "improvised", "both"):
            raise ConfigError(
                "labels.iemocap_subsets must be null, 'scripted', 'improvised', or 'both'"
            )
        if self.iemocap_frustrated not in (None, "drop", "merge_angry"):
            raise ConfigError(
                "labels.iemocap_frustrated must be null, 'drop', or 'merge_angry'"
            )
        if self.iemocap_label_source not in (None, *self.LABEL_SOURCES):
            raise ConfigError(
                f"labels.iemocap_label_source must be null or one of "
                f"{list(self.LABEL_SOURCES)}"
            )


@dataclass(frozen=True)
class SplitsConfig:
    split_spec_version: str
    source_train_ratio: float
    target_adapt_ratio: float
    iemocap_split_unit: str
    seeds: List[int]

    def __post_init__(self) -> None:
        for name in ("source_train_ratio", "target_adapt_ratio"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"splits.{name} must be strictly between 0 and 1")
        if self.iemocap_split_unit not in ("session", "speaker"):
            raise ConfigError("splits.iemocap_split_unit must be 'session' or 'speaker'")
        if len(self.seeds) != len(set(self.seeds)):
            raise ConfigError("splits.seeds contains duplicates")
        if len(self.seeds) < 5:
            raise ConfigError("splits.seeds must contain at least 5 seeds")


@dataclass(frozen=True)
class FeaturesConfig:
    feature_version: str
    sample_rate: int
    mono: bool
    peak_normalise: bool
    backbones: Dict[str, str]
    n_layers: int
    hidden_dim: int
    storage_dtype: str
    segment_pooling_enabled: bool
    n_segments: int
    mfcc_n_coefficients: int
    mfcc_deltas: bool
    mfcc_pooling: List[str]

    def __post_init__(self) -> None:
        if self.storage_dtype not in ("float16", "float32"):
            raise ConfigError("features.storage_dtype must be float16 or float32")
        for pooling in self.mfcc_pooling:
            if pooling not in ("mean", "std"):
                raise ConfigError("features.mfcc_pooling entries must be 'mean' or 'std'")


@dataclass(frozen=True)
class AlignmentConfig:
    """The alignment ladder, ordered by which moments each condition matches.

        zscore      per-dimension 1st + 2nd, no cross-terms
        mean_shift  1st            (= MMD under a linear kernel)
        coral       1st + 2nd, with covariance
        mmd         all, via a sum of RBF kernels

    ``mean_shift`` is a first-class condition, not a synonym for ``mmd``. The
    original study's ``"mmd"`` was ``X_src + (mu_tgt - mu_src)``, which is
    exactly the minimiser of linear-kernel MMD -- a degenerate special case of
    what it claimed. Keeping both rungs measures what that column was worth.
    There is deliberately no alias from ``mmd`` to the mean shift: an alias is
    how the original misstatement would survive into v2.

    ``mmd_*`` fully specifies the operator. The paper must state the operator,
    kernel family, bandwidth rule, and optimisation budget -- these fields are
    that statement, and the paper text is generated from them.
    """

    methods: List[str]
    coral_eps: float
    coral_eps_sensitivity: List[float]
    mmd_kernel: str
    mmd_bandwidth_multipliers: List[float]
    mmd_fit_bias: bool
    mmd_identity_penalty: float
    mmd_steps: int
    mmd_learning_rate: float

    # Ordered by moments matched. Used to order the ablation table.
    LADDER = ("none", "zscore", "mean_shift", "coral", "mmd")

    def __post_init__(self) -> None:
        unknown = sorted(set(self.methods) - set(self.LADDER))
        if unknown:
            raise ConfigError(
                f"alignment.methods contains unknown method(s): {unknown}. "
                f"The ladder is {list(self.LADDER)}."
            )
        if self.coral_eps <= 0:
            raise ConfigError("alignment.coral_eps must be positive")
        if len(self.coral_eps_sensitivity) != 3:
            raise ConfigError(
                "alignment.coral_eps_sensitivity must list exactly 3 values; "
                "the paper reports epsilon sensitivity at three settings"
            )
        if self.mmd_kernel != "gaussian_multikernel":
            raise ConfigError("alignment.mmd_kernel must be 'gaussian_multikernel'")
        if self.mmd_identity_penalty < 0:
            raise ConfigError("alignment.mmd_identity_penalty must be non-negative")

    def ladder_order(self) -> List[str]:
        """Configured methods, in ladder order."""
        return [method for method in self.LADDER if method in self.methods]


@dataclass(frozen=True)
class BlendingConfig:
    modes: List[str]
    alpha_grid: List[float]
    n_groups: int

    def __post_init__(self) -> None:
        allowed = {"none", "scalar", "gaa"}
        unknown = sorted(set(self.modes) - allowed)
        if unknown:
            raise ConfigError(f"blending.modes contains unknown mode(s): {unknown}")
        if not all(0.0 <= a <= 1.0 for a in self.alpha_grid):
            raise ConfigError("blending.alpha_grid values must lie in [0, 1]")
        if self.n_groups < 2:
            raise ConfigError("blending.n_groups must be at least 2")


@dataclass(frozen=True)
class ClassifiersConfig:
    families: List[str]
    search_budget: int
    early_stopping_patience: int
    layer_agg_options: List[str]
    layer_candidates: List[int]

    def __post_init__(self) -> None:
        allowed = {"logreg", "svm", "mlp", "transformer"}
        unknown = sorted(set(self.families) - allowed)
        if unknown:
            raise ConfigError(f"classifiers.families contains unknown family: {unknown}")
        if self.search_budget < 1:
            raise ConfigError("classifiers.search_budget must be at least 1")
        agg_allowed = {"last", "layer", "mean", "weighted"}
        unknown_agg = sorted(set(self.layer_agg_options) - agg_allowed)
        if unknown_agg:
            raise ConfigError(f"classifiers.layer_agg_options unknown: {unknown_agg}")


@dataclass(frozen=True)
class GridConfig:
    corpora: List[str]
    feature_branches: List[str]
    include_iemocap_subset_pair: bool

    KNOWN_CORPORA = ("ravdess", "cremad", "iemocap")

    def __post_init__(self) -> None:
        unknown_corpora = sorted(set(self.corpora) - set(self.KNOWN_CORPORA))
        if unknown_corpora:
            raise ConfigError(
                f"grid.corpora contains unknown corpus/corpora: {unknown_corpora}"
            )
        allowed = {"ssl", "mfcc", "fused"}
        unknown = sorted(set(self.feature_branches) - allowed)
        if unknown:
            raise ConfigError(f"grid.feature_branches unknown: {unknown}")
        if self.include_iemocap_subset_pair and "iemocap" not in self.corpora:
            raise ConfigError(
                "grid.include_iemocap_subset_pair requires 'iemocap' in grid.corpora"
            )


@dataclass(frozen=True)
class BaselinesConfig:
    n_random_draws: int


@dataclass(frozen=True)
class StatsConfig:
    bootstrap_resamples: int
    ci_level: float
    correction: str

    def __post_init__(self) -> None:
        if not 0.0 < self.ci_level < 1.0:
            raise ConfigError("stats.ci_level must lie strictly between 0 and 1")
        if self.correction != "holm-bonferroni":
            raise ConfigError("stats.correction must be 'holm-bonferroni'")


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    paths: PathsConfig
    labels: LabelsConfig
    splits: SplitsConfig
    features: FeaturesConfig
    alignment: AlignmentConfig
    blending: BlendingConfig
    classifiers: ClassifiersConfig
    grid: GridConfig
    baselines: BaselinesConfig
    stats: StatsConfig
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)
    source_path: Optional[str] = None

    # -- provenance --------------------------------------------------------
    @property
    def config_hash(self) -> str:
        """sha256 over the canonicalised raw config. Recorded on every row."""
        return hash_payload(self.raw)

    @property
    def label_map_hash(self) -> str:
        """sha256 over the resolved label mapping. A ``run_id`` coordinate.

        Without this, changing a mapping decision mid-project (say
        ``iemocap_frustrated``) leaves ``run_id`` unchanged, and a Phase 7
        resume silently merges runs scored against different label spaces --
        a corruption nothing downstream could detect.
        """
        return hash_payload(self.raw["labels"])[:16]

    @property
    def split_spec_hash(self) -> str:
        """sha256 over the split specification. A ``run_id`` coordinate.

        Same argument as :attr:`label_map_hash`: a changed ratio or split unit
        must produce new run ids rather than colliding with old ones.
        """
        return hash_payload(self.raw["splits"])[:16]

    @property
    def seed(self) -> int:
        return self.project.seed

    # -- path resolution ---------------------------------------------------
    def resolve(self, relative: str) -> Path:
        """Resolve a config path against the repo root.

        Config files hold repository-relative paths so a config is portable;
        this is the single place they become absolute.
        """
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else repo_root() / candidate

    @property
    def results_path(self) -> Path:
        return self.resolve(self.project.results_path)

    # -- open decisions ----------------------------------------------------
    def require_decision(self, name: str) -> Any:
        """Fetch a label decision, refusing to proceed if it is unmade.

        Rule 3 of the rebuild is "if a design decision is genuinely ambiguous,
        STOP and ask". This is that rule expressed in code: an unmade decision
        halts the pipeline instead of quietly resolving to a default that would
        then be silently baked into every downstream number.
        """
        if name not in LabelsConfig.DECISION_FIELDS:
            raise ConfigError(
                f"'{name}' is not a labels decision; "
                f"expected one of {list(LabelsConfig.DECISION_FIELDS)}"
            )
        value = getattr(self.labels, name)
        if value is None:
            raise ConfigError(
                f"labels.{name} is undecided (null in {self.source_path}). "
                "This is a paper-level decision: set it explicitly in the config "
                "and record the rationale in PROGRESS.md before continuing."
            )
        return value

    def undecided(self) -> List[str]:
        """Names of label decisions still null. Reported by `ser inventory`."""
        return [
            name
            for name in LabelsConfig.DECISION_FIELDS
            if getattr(self.labels, name) is None
        ]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
_SECTIONS: Dict[str, type] = {
    "project": ProjectConfig,
    "paths": PathsConfig,
    "labels": LabelsConfig,
    "splits": SplitsConfig,
    "features": FeaturesConfig,
    "alignment": AlignmentConfig,
    "blending": BlendingConfig,
    "classifiers": ClassifiersConfig,
    "grid": GridConfig,
    "baselines": BaselinesConfig,
    "stats": StatsConfig,
}


def load_config(path: str | Path | None = None) -> Config:
    """Load and strictly validate a YAML config."""
    config_path = Path(path) if path is not None else repo_root() / DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    missing = sorted(set(_SECTIONS) - set(raw))
    if missing:
        raise ConfigError(f"{config_path} missing section(s): {missing}")
    extra = sorted(set(raw) - set(_SECTIONS))
    if extra:
        raise ConfigError(f"{config_path} has unknown section(s): {extra}")

    sections = {
        name: _build(cls, raw[name], name) for name, cls in _SECTIONS.items()
    }

    config = Config(raw=raw, source_path=str(config_path), **sections)
    _validate_cross_section(config)
    return config


def _build(cls: type, values: Any, section: str):
    """Instantiate a section dataclass, rejecting unknown and missing keys."""
    if not isinstance(values, dict):
        raise ConfigError(f"section '{section}' must be a mapping")

    known = {f.name for f in dataclasses.fields(cls)}
    required = {
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }

    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(f"section '{section}' has unknown key(s): {unknown}")
    missing = sorted(required - set(values))
    if missing:
        raise ConfigError(f"section '{section}' missing key(s): {missing}")

    try:
        return cls(**values)
    except ConfigError:
        raise
    except TypeError as exc:
        raise ConfigError(f"section '{section}': {exc}") from exc


def _validate_cross_section(config: Config) -> None:
    """Consistency checks that span sections."""
    if config.project.seed < 0:
        raise ConfigError("project.seed must be non-negative")

    for space, classes in config.labels.spaces.items():
        if not classes:
            raise ConfigError(f"labels.spaces.{space} is empty")

    if config.grid.include_iemocap_subset_pair and not config.labels.iemocap_record_subset:
        raise ConfigError(
            "grid.include_iemocap_subset_pair requires labels.iemocap_record_subset: "
            "the improvised/scripted probe cannot be built without the per-utterance "
            "subset recorded in the manifest"
        )

    for layer in config.classifiers.layer_candidates:
        if not 0 <= layer < config.features.n_layers:
            raise ConfigError(
                f"classifiers.layer_candidates contains {layer}, outside "
                f"[0, {config.features.n_layers})"
            )

    if config.features.segment_pooling_enabled is False and "transformer" in config.classifiers.families:
        raise ConfigError(
            "classifiers.families includes 'transformer' but "
            "features.segment_pooling_enabled is false. The transformer baseline "
            "requires the segment-pooled cache to be a genuine sequence model "
            "(Decision A). Either enable segment pooling or drop the family."
        )
