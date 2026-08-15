# Cross-Corpus Speech Emotion Recognition

Cross-corpus SER over RAVDESS, CREMA-D, and IEMOCAP with self-supervised speech
representations (HuBERT Base, wav2vec 2.0 Base, WavLM Base), feature alignment,
and several classifier heads.

> ## ⚠️ Results under revision
>
> **This repository is mid-rebuild. No performance numbers are currently
> published here, and any figures you may have seen in an earlier version of this
> README or in the draft manuscript should not be cited or relied on.**
>
> An audit of the original pipeline (recorded in [PROGRESS.md](PROGRESS.md))
> found methodological defects serious enough to invalidate the reported results:
>
> - Feature alignment was fitted on the target **test** set, not just the target
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
> Corrected results will be published here when the rebuild completes. Until
> then, please treat this repository as code under active revision rather than as
> an accompaniment to any published claim.

## Status

The rebuild is organised as twelve phases; see [PHASES.md](PHASES.md) for the
plan and [PROGRESS.md](PROGRESS.md) for the running log of what has been done,
decided, and deferred.

| Phase | | Status |
|---|---|---|
| 0 | Scaffold and reproducibility spine | complete |
| 1 | Reference integrity checker | script complete; manual DOI resolution outstanding |
| 2 | Manifest, label map, splits, leakage tests | complete for {RAVDESS, CREMA-D}; IEMOCAP pending |
| 3 | Feature extraction and caching | complete for {RAVDESS, CREMA-D}; IEMOCAP pending |
| 4 | Metrics and trivial baselines | not started |
| 5 | Alignment and blending | not started |
| 6 | Classifiers with equal-budget search | not started |
| 7 | Grid runner | not started |
| 8 | Selection protocol and headline tables | not started |
| 9 | Shift decomposition: label, covariate, conditional | not started |
| 10 | Per-class analysis and figures | not started |
| 11 | Release packaging and LaTeX tables | not started |

The original pipeline is preserved untouched under [`legacy/`](legacy/) for
traceability. It is **not** the entry point and should not be run for new work.

## What the rebuild changes

- Speaker-disjoint splits with an explicit `target_adapt` / `target_test`
  separation. Alignment may see `target_adapt` only; `target_test` is touched
  once, at scoring time. Alignment objects record the indices they were fitted
  on, and this is asserted, not asserted-in-prose.
- All hidden layers cached, so layer aggregation becomes a searchable condition
  rather than an unexamined default.
- Two reporting protocols, both published: **validated** (configuration selected
  on source-side validation — what a practitioner could achieve without target
  labels) and **oracle** (grid maximum on target test). The gap between them is
  reported as a result in its own right.
- An ordered alignment ladder — identity, z-score, mean shift, CORAL, MK-MMD —
  by moments matched, rather than two arbitrary points of comparison.
- Chance, majority-class, and prior-matched floors on every table.
- Five seeds minimum, with paired significance tests.
- Every result row carries its git SHA, config hash, label-map hash, split hash,
  and library versions. Every table and figure is generated from
  `results/runs.jsonl` by script; no number is typed by hand.

## Environment

Python 3.12. Fully pinned; see [requirements.txt](requirements.txt).

```bash
python -m pip install -r requirements.txt && python -m pip install -e .
```

## Usage

```bash
ser --help
```

`ser inventory` reports repository state, configuration, and open decisions.
`ser smoke` exercises the result-writing path end to end. Commands belonging to
phases that are not yet built exit with the phase number that owns them.

On Windows, or without `make`:

```bash
PYTHONPATH=src python -m ser.cli inventory
```

Every experimental value lives in [`configs/default.yaml`](configs/default.yaml).
Configuration loading is strict — an unknown or missing key is an error, never a
silent default.

## Tests

```bash
python -m pytest
```

## Data

Raw corpora are licence-restricted and are not distributed here. RAVDESS and
CREMA-D are public downloads; IEMOCAP requires a signed agreement with USC. Point
`paths.raw_*` in the config at local copies.

## Paper

The manuscript sources are under [`legacy/`](legacy/) and are being rewritten
alongside the code. They contain the pre-revision results described above.

## License

No licence is currently granted. Treat this as research code accompanying work in
progress.
