# Cross-Corpus Speech Emotion Recognition

Cross-corpus SER over RAVDESS, CREMA-D, and IEMOCAP with self-supervised speech
representations (HuBERT Base, wav2vec 2.0 Base, WavLM Base), feature alignment,
and several classifier heads.

> ## ⚠️ Results under revision — do not cite
>
> **No performance numbers are currently published in this repository. Any
> figures you may have seen in an earlier version of this README, or in the draft
> manuscript, should not be cited or relied on.**
>
> An audit of the original pipeline found methodological defects serious enough
> to invalidate the reported results:
>
> - Feature alignment was fitted on the target **test** set, not just a target
>   adaptation split — transductive leakage into every aligned condition.
> - Model selection was performed on target-test scores, with no source-side
>   validation split in existence.
> - The condition reported as **MMD** was implemented as a plain mean shift,
>   `X_src + (μ_tgt − μ_src)` — no kernel, no learned map, no optimisation.
> - SSL features used final-layer pooling only.
> - A single hardcoded seed, so no run-to-run variance was ever measured.
> - No chance or prior-matched floors, against which several reported numbers
>   need re-reading.
>
> A full rebuild is underway. Corrected results will be published here when it
> completes. Until then, please treat this repository as code under active
> revision rather than as an accompaniment to any published claim.

## Where the rebuild lives

Active work is on the [`rebuild`](../../tree/rebuild) branch:

- [PHASES.md](../../blob/rebuild/PHASES.md) — the twelve-phase rebuild plan
- [PROGRESS.md](../../blob/rebuild/PROGRESS.md) — running log: the full audit of
  the original pipeline, every decision made, and what is deferred
- [`legacy/`](../../tree/rebuild/legacy) — the original pipeline, preserved
  untouched for provenance

**This branch (`main`) is the pre-revision code**, kept so the published state
remains inspectable. It is not the entry point for new work and its results are
the ones described above.

## What the rebuild changes

- Speaker-disjoint splits with an explicit `target_adapt` / `target_test`
  separation. Alignment may see `target_adapt` only; `target_test` is touched
  once, at scoring time, and this is asserted in tests rather than in prose.
- All hidden layers cached, so layer aggregation becomes a searchable condition
  rather than an unexamined default.
- Two reporting protocols, both published: **validated** (configuration selected
  on source-side validation — what a practitioner could achieve without target
  labels) and **oracle** (grid maximum on target test). The gap between them is
  reported as a result in its own right.
- An ordered alignment ladder — identity, z-score, mean shift, CORAL, MK-MMD — by
  moments matched, rather than two arbitrary points of comparison.
- Chance, majority-class, and prior-matched floors on every table.
- Five seeds minimum, with paired significance tests.
- Every result row carries its git SHA, config hash, label-map hash, split hash,
  and library versions. Every table and figure is generated from a single
  append-only results file by script; no number is typed by hand.

## Data

Raw corpora are licence-restricted and are not distributed here. RAVDESS and
CREMA-D are public downloads; IEMOCAP requires a signed agreement with USC.

## License

No licence is currently granted. Treat this as research code accompanying work in
progress.
