"""Dataset statistics, and the A8 prior-shift verification.

Every number here is derived from ``data/manifest.csv``. Nothing is hardcoded
except the published expectations this module exists to check *against*.

The load-bearing part is :data:`EXPECTED_PRIOR_KL`. Amendment A8 concluded that
prior shift is near-zero across every corpus pair, and that conclusion reframed
the whole of Phase 9. It was computed from published counts, not from data on
disk. This module recomputes it from the manifest and **halts** if the two
disagree -- because if the real priors are not what A8 assumed, the reframe needs
revisiting before anything is built on top of it.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .manifest import ManifestRow, read_manifest

__all__ = [
    "EXPECTED_PRIOR_KL",
    "KL_TOLERANCE",
    "PriorMismatch",
    "class_counts",
    "prior_vector",
    "kl_divergence",
    "js_distance",
    "pairwise_prior_shift",
    "run_dataset_stats",
]

# Amendment A8's predicted corpus-level prior KL, in nats, computed from
# published class counts. Verified here against the manifest.
#
# Not a config value: a guard whose threshold the user can edit is not a guard.
EXPECTED_PRIOR_KL: Dict[Tuple[str, str], float] = {
    ("ravdess", "cremad"): 0.0252,
    ("cremad", "ravdess"): 0.0224,
    ("iemocap", "ravdess"): 0.0148,
    ("ravdess", "iemocap"): 0.0139,
    ("iemocap", "cremad"): 0.0336,
    ("cremad", "iemocap"): 0.0335,
}

# Absolute tolerance in nats. The expectations came from the same published
# counts the manifest should reproduce, so agreement should be near-exact;
# this allows for rounding in the published figures only.
KL_TOLERANCE = 0.002


class PriorMismatch(RuntimeError):
    """Manifest priors disagree materially with the A8 prediction."""


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def _label_of(row: ManifestRow, label_space: str) -> str:
    value = row.label_six if label_space == "six" else row.label_four
    return value


def class_counts(
    rows: Sequence[ManifestRow], corpus: str, label_space: str
) -> Counter:
    """Utterances per class for one corpus under one label space.

    Excluded utterances (empty mapped label) are not counted.
    """
    return Counter(
        _label_of(row, label_space)
        for row in rows
        if row.corpus == corpus and _label_of(row, label_space)
    )


def prior_vector(
    counts: Counter, classes: Sequence[str]
) -> Optional[List[float]]:
    """Class prior over ``classes``. None when the corpus has no support."""
    total = sum(counts.get(name, 0) for name in classes)
    if total == 0:
        return None
    return [counts.get(name, 0) / total for name in classes]


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) in nats.

    A zero in q where p is positive is genuinely infinite; it is returned as
    ``inf`` rather than smoothed, because smoothing would hide a class that is
    absent from the target entirely -- which is a finding, not a nuisance.
    """
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            continue
        if qi == 0.0:
            return math.inf
        total += pi * math.log(pi / qi)
    return total


def js_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon distance (the metric: sqrt of the divergence)."""
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return math.sqrt(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def space_for_pair(config, source: str, target: str) -> str:
    """Label space a pair is evaluated in (A4)."""
    if "iemocap" in (source, target):
        return config.labels.space_for_iemocap_pairs
    return config.labels.space_for_other_pairs


def pairwise_prior_shift(
    rows: Sequence[ManifestRow], config, corpora: Sequence[str]
) -> List[dict]:
    """Corpus-level prior KL and JS for every ordered pair."""
    results = []
    for source, target in permutations(corpora, 2):
        label_space = space_for_pair(config, source, target)
        classes = config.labels.spaces[label_space]
        p = prior_vector(class_counts(rows, source, label_space), classes)
        q = prior_vector(class_counts(rows, target, label_space), classes)
        if p is None or q is None:
            continue
        results.append(
            {
                "source": source,
                "target": target,
                "label_space": label_space,
                "n_classes": len(classes),
                "kl": kl_divergence(p, q),
                "js": js_distance(p, q),
                "expected_kl": EXPECTED_PRIOR_KL.get((source, target)),
            }
        )
    return results


def verify_against_a8(shifts: Sequence[dict]) -> List[str]:
    """Return a list of disagreements with the A8 prediction. Empty is good."""
    problems = []
    for shift in shifts:
        expected = shift["expected_kl"]
        if expected is None:
            continue
        delta = abs(shift["kl"] - expected)
        if delta > KL_TOLERANCE:
            problems.append(
                f"{shift['source']} -> {shift['target']}: manifest KL "
                f"{shift['kl']:.4f} vs A8 prediction {expected:.4f} "
                f"(delta {delta:.4f} > tolerance {KL_TOLERANCE})"
            )
    return problems


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def run_dataset_stats(config, corpora: Sequence[str]) -> int:
    rows = read_manifest(config.resolve(config.paths.manifest))
    present = [name for name in corpora if any(r.corpus == name for r in rows)]
    missing = [name for name in corpora if name not in present]

    reports_dir = config.resolve(config.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    shifts = pairwise_prior_shift(rows, config, present)
    problems = verify_against_a8(shifts)

    markdown = _render(config, rows, present, missing, shifts, problems)
    (reports_dir / "dataset_stats.md").write_text(markdown, encoding="utf-8")
    _write_csv(config, rows, present, reports_dir / "dataset_stats.csv")

    print(f"corpora in manifest: {', '.join(present) or 'none'}")
    if missing:
        print(f"not yet acquired:    {', '.join(missing)}")
    print()
    print("corpus-level prior shift (A9: this is the corpus-level check;")
    print("split-level KL per seed is a Phase 8 quantity)")
    for shift in shifts:
        expected = shift["expected_kl"]
        verdict = "—"
        if expected is not None:
            verdict = "OK" if abs(shift["kl"] - expected) <= KL_TOLERANCE else "MISMATCH"
        print(
            f"  {shift['source']:>8} -> {shift['target']:<8} K={shift['n_classes']} "
            f"KL={shift['kl']:.4f}  JS={shift['js']:.4f}  "
            f"A8={expected if expected is not None else '—'}  {verdict}"
        )
    print()
    print(f"wrote {reports_dir / 'dataset_stats.md'}")

    if problems:
        print("\nHALTED: manifest priors disagree with amendment A8:", flush=True)
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nA8's reframe of Phase 9 rests on these numbers. Do not proceed until "
            "the disagreement is understood."
        )
        return 1
    return 0


def _render(config, rows, present, missing, shifts, problems) -> str:
    lines: List[str] = ["# Dataset statistics", ""]
    lines.append(
        "Every number is derived from `data/manifest.csv`. Regenerate with "
        "`ser dataset-stats`."
    )
    lines.append("")
    if missing:
        lines.append(
            f"> **Partial.** Not yet acquired: {', '.join(missing)}. "
            "Statistics cover only the corpora present."
        )
        lines.append("")

    lines.append("## Per corpus")
    lines.append("")
    lines.append("| corpus | speakers | utterances | total hours | mean duration (s) |")
    lines.append("|---|---|---|---|---|")
    for corpus in present:
        subset = [r for r in rows if r.corpus == corpus]
        durations = [r.duration_s for r in subset]
        lines.append(
            f"| {corpus} | {len({r.speaker_id for r in subset})} | {len(subset)} | "
            f"{sum(durations)/3600:.2f} | {sum(durations)/len(durations):.2f} |"
        )
    lines.append("")

    for label_space in sorted(config.labels.spaces):
        classes = config.labels.spaces[label_space]
        lines.append(f"## Label space `{label_space}` ({len(classes)} classes)")
        lines.append("")
        lines.append("| corpus | " + " | ".join(classes) + " | total | excluded |")
        lines.append("|---" * (len(classes) + 3) + "|")
        for corpus in present:
            counts = class_counts(rows, corpus, label_space)
            total = sum(counts.values())
            n_corpus = sum(1 for r in rows if r.corpus == corpus)
            cells = []
            for name in classes:
                count = counts.get(name, 0)
                flag = " ⚠️" if 0 < count < config.labels.min_class_support_warn else ""
                cells.append(f"{count}{flag}" if count else "0 ⚠️")
            lines.append(
                f"| {corpus} | " + " | ".join(cells) + f" | {total} | {n_corpus - total} |"
            )
        lines.append("")
        lines.append("Class prior:")
        lines.append("")
        lines.append("| corpus | " + " | ".join(classes) + " |")
        lines.append("|---" * (len(classes) + 1) + "|")
        for corpus in present:
            prior = prior_vector(class_counts(rows, corpus, label_space), classes)
            if prior is None:
                continue
            lines.append(f"| {corpus} | " + " | ".join(f"{p:.3f}" for p in prior) + " |")
        lines.append("")
        lines.append(
            f"⚠️ marks a class with fewer than {config.labels.min_class_support_warn} "
            "utterances after mapping."
        )
        lines.append("")

    lines.append("## Corpus-level prior shift")
    lines.append("")
    lines.append(
        "Amendment **A9**: these are *corpus-level* priors. The quantity the "
        "analysis rests on is split-level KL per pair per seed, computed from the "
        "realised partitions — a Phase 8 deliverable. This table is the "
        "data-integrity check against the published counts that **A8** used."
    )
    lines.append("")
    lines.append("| source | target | K | KL (nats) | JS | A8 predicted | agrees |")
    lines.append("|---|---|---|---|---|---|---|")
    for shift in shifts:
        expected = shift["expected_kl"]
        agrees = "—"
        if expected is not None:
            agrees = "yes" if abs(shift["kl"] - expected) <= KL_TOLERANCE else "**NO**"
        lines.append(
            f"| {shift['source']} | {shift['target']} | {shift['n_classes']} | "
            f"{shift['kl']:.4f} | {shift['js']:.4f} | "
            f"{expected if expected is not None else '—'} | {agrees} |"
        )
    lines.append("")
    if problems:
        lines.append("> ⚠️ **HALT.** " + " ".join(problems))
    else:
        lines.append(
            f"All computed pairs agree with A8 within {KL_TOLERANCE} nats. The "
            "near-zero prior shift that reframed Phase 9 is confirmed against real data."
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(config, rows, present, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corpus", "label_space", "class", "count", "share", "below_warn"])
        for label_space in sorted(config.labels.spaces):
            classes = config.labels.spaces[label_space]
            for corpus in present:
                counts = class_counts(rows, corpus, label_space)
                total = sum(counts.values())
                for name in classes:
                    count = counts.get(name, 0)
                    writer.writerow(
                        [
                            corpus,
                            label_space,
                            name,
                            count,
                            f"{count/total:.6f}" if total else "",
                            int(count < config.labels.min_class_support_warn),
                        ]
                    )
