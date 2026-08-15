"""Label mapping: raw corpus annotation -> a class in a label space, or excluded.

:func:`map_label` is a **pure function**. Same inputs, same output, no side
effects, no I/O, no config lookup of its own -- every decision that could change
its behaviour arrives as an explicit :class:`LabelPolicy`. That is what makes the
Phase 2 purity assertion meaningful and what lets ``label_map_hash`` stand for
the mapping actually applied.

Two failure modes it refuses to have:

* **Silent exclusion.** A raw label the corpus is not known to contain raises
  rather than returning ``None``. ``None`` means "deliberately excluded"; it must
  never also mean "unrecognised".
* **Implicit policy.** There is no default policy. A caller must state the
  decisions, so no run can quietly adopt one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

__all__ = [
    "LabelPolicy",
    "map_label",
    "raw_labels_for",
    "RAVDESS_EMOTION_CODES",
    "CREMAD_EMOTION_CODES",
    "IEMOCAP_RAW_LABELS",
    "KNOWN_CORPORA",
]

KNOWN_CORPORA = ("ravdess", "cremad", "iemocap")

# RAVDESS encodes emotion as the third filename field.
RAVDESS_EMOTION_CODES: Mapping[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised",
}

# CREMA-D encodes emotion as the third underscore-separated filename field.
CREMAD_EMOTION_CODES: Mapping[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Categorical labels appearing in IEMOCAP's EmoEvaluation files.
IEMOCAP_RAW_LABELS: Tuple[str, ...] = (
    "angry",
    "disgust",
    "excited",
    "fear",
    "frustrated",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "xxx",  # no majority agreement
)

_RAW_LABELS = {
    "ravdess": tuple(sorted(set(RAVDESS_EMOTION_CODES.values()))),
    "cremad": tuple(sorted(set(CREMAD_EMOTION_CODES.values()))),
    "iemocap": IEMOCAP_RAW_LABELS,
}


def raw_labels_for(corpus: str) -> Tuple[str, ...]:
    """Every raw label string the corpus is known to produce.

    The table-driven label test enumerates these, so a corpus emitting something
    outside this set fails loudly rather than being silently dropped.
    """
    corpus = corpus.strip().lower()
    if corpus not in _RAW_LABELS:
        raise ValueError(f"unknown corpus {corpus!r}; expected one of {KNOWN_CORPORA}")
    return _RAW_LABELS[corpus]


@dataclass(frozen=True)
class LabelPolicy:
    """The A-series label decisions, resolved.

    Frozen and hashable so it can be compared and logged. Built from config by
    :meth:`from_config`, which refuses to proceed while any decision is unmade.
    """

    spaces: Mapping[str, Tuple[str, ...]]
    iemocap_label_source: str
    iemocap_excited_to_happy: bool
    iemocap_frustrated: str  # "drop" | "merge_angry"
    ravdess_calm_to_neutral: bool

    @classmethod
    def from_config(cls, config) -> "LabelPolicy":
        """Build from a :class:`ser.config.Config`.

        Uses ``require_decision`` throughout, so an unmade decision halts here
        rather than silently resolving to a default.
        """
        return cls(
            spaces={name: tuple(classes) for name, classes in config.labels.spaces.items()},
            iemocap_label_source=config.require_decision("iemocap_label_source"),
            iemocap_excited_to_happy=config.require_decision("iemocap_excited_to_happy"),
            iemocap_frustrated=config.require_decision("iemocap_frustrated"),
            ravdess_calm_to_neutral=config.require_decision("ravdess_calm_to_neutral"),
        )

    def space(self, label_space: str) -> Tuple[str, ...]:
        if label_space not in self.spaces:
            raise ValueError(
                f"unknown label space {label_space!r}; have {sorted(self.spaces)}"
            )
        return self.spaces[label_space]


def map_label(
    corpus: str,
    original_label: str,
    label_space: str,
    policy: LabelPolicy,
) -> Optional[str]:
    """Map a raw corpus label into ``label_space``.

    Returns the class name, or ``None`` when the utterance is deliberately
    excluded from this label space.

    Raises:
        ValueError: if ``corpus`` is unknown, ``label_space`` is unknown, or
            ``original_label`` is not a label the corpus is known to produce.
            An unrecognised label is a bug in the parser or a change in the
            corpus, and either way must not be silently dropped.
    """
    corpus_key = corpus.strip().lower()
    known = raw_labels_for(corpus_key)  # validates the corpus
    classes = policy.space(label_space)  # validates the space

    raw = original_label.strip().lower()
    if raw not in known:
        raise ValueError(
            f"{corpus_key}: unrecognised raw label {original_label!r}. "
            f"Known: {list(known)}. Refusing to guess -- an unrecognised label "
            "must not be silently excluded."
        )

    mapped = _apply_corpus_rules(corpus_key, raw, policy)
    if mapped is None:
        return None

    # A class not present in the requested space is excluded. This is what
    # removes IEMOCAP fear and disgust under the 4-class space, and RAVDESS
    # surprised under the 6-class space.
    return mapped if mapped in classes else None


def _apply_corpus_rules(corpus: str, raw: str, policy: LabelPolicy) -> Optional[str]:
    """Corpus-specific merges and drops, before the label space filters."""
    if corpus == "ravdess":
        if raw == "calm":
            return "neutral" if policy.ravdess_calm_to_neutral else "calm"
        return raw

    if corpus == "cremad":
        return raw

    # IEMOCAP
    if raw in ("xxx", "other", "surprised"):
        # 'xxx' is the no-majority-agreement marker. Under
        # majority_vote_discard_disagreement these utterances have no label.
        return None
    if raw == "excited":
        return "happy" if policy.iemocap_excited_to_happy else "excited"
    if raw == "frustrated":
        return "angry" if policy.iemocap_frustrated == "merge_angry" else None
    return raw
