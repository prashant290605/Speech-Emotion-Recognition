# Cross-Corpus Speech Emotion Recognition

A reproducibility rebuild of cross-corpus SER over RAVDESS and CREMA-D with
self-supervised speech representations (HuBERT Base, wav2vec 2.0 Base, WavLM
Base), an ordered feature-alignment ladder, and five classifier families.

> **The rebuilt pipeline, results, figures and tables live on the [`rebuild`](https://github.com/prashant290605/Speech-Emotion-Recognition/tree/rebuild) branch.**
> This branch (`main`) still holds the original pre-revision code, kept for
> traceability. Every link below points into `rebuild`.

**Every number below regenerates from `results/` by script. Nothing is typed by
hand.** The full set, with intervals, floors and run filters, is in
[reports/RESULTS.md](https://github.com/prashant290605/Speech-Emotion-Recognition/blob/rebuild/reports/RESULTS.md).

---

## Headline

Target macro-F1 of the configuration a practitioner could actually select —
chosen on **source-side validation only**, never on target labels — against the
best configuration present in the grid.

| pair | validated | oracle (upper bound) | chance floor |
|---|---|---|---|
| RAVDESS → CREMA-D | **0.3156** [0.1880, 0.4432] | 0.4555 [0.4326, 0.4785] | 0.1665 |
| CREMA-D → RAVDESS | **0.4994** [0.4592, 0.5395] | 0.5680 [0.5148, 0.6212] | 0.1656 |

Three findings, each with its own report:

**1. Alignment buys one step, then nothing.** Moving from unaligned features to
*any* aligned condition is worth +0.12 to +0.20 macro-F1 (all 14 pre-registered
comparisons survive Holm correction). Among the five aligned rungs the largest
difference is 0.0151 / 0.0437 — an order of magnitude smaller, and **not ordered
by discrepancy**. `mkmmd_full` achieves the lowest discrepancy in both
geometries and is never the best rung. Per-dimension z-scoring matches
everything more elaborate that was tried.

**2. Alignment removes the wrong term.** The three-way shift decomposition shows
label shift is negligible (KL ≈ 0.022 nats). Alignment drives the *marginal*
discrepancy down to 0.010× while the *conditional* term falls only to 0.07×; the
conditional/marginal ratio rises from 0.14 to 0.92. What remains after alignment
is conditional shift, which no marginal alignment can touch. A label-shift
correction, as the near-zero KL predicts, **hurts in 238 of 240 cases**.

**3. Reporting "MMD reduction" is not well posed.** Measured in each rung's own
geometry the discrepancy-transfer correlation is ρ = **−0.444** [−0.505, −0.383];
measured in a fixed reference frame it is **+0.142** [+0.062, +0.221]. Opposite
signs, neither interval covering zero, same features and same target scores. The
CORAL shrinkage asymptote supplies a case where the correct answer is known
analytically, and the reference frame is the one that tracks it.

Retractions and null results are reported alongside these in
[RESULTS.md §8–9](https://github.com/prashant290605/Speech-Emotion-Recognition/blob/rebuild/reports/RESULTS.md) — five claims made during this rebuild
were later withdrawn or narrowed, and they are listed so they cannot be reused
by accident.

---

## Reproducing this from a clean clone

Everything except feature extraction runs from the committed result files.

```bash
git clone https://github.com/prashant290605/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
git checkout rebuild
python -m pip install -r requirements.txt && python -m pip install -e .
python -m pytest                      # 429 tests
```

Regenerate every table, figure and report:

```bash
PYTHONPATH=src python tools/phase10_per_class.py    # per-class analysis
PYTHONPATH=src python tools/make_figures.py         # 7 figures -> figures/
PYTHONPATH=src python tools/make_tables.py          # 7 LaTeX tables -> tables/
PYTHONPATH=src python tools/make_results_doc.py     # reports/RESULTS.md
```

Re-running the experiments themselves needs the raw corpora (see **Data**) and
about 62 hours of CPU; `tools/launch_stage2.ps1` is the entry point.

## Provenance and the frozen config

| artifact | rows | failures |
|---|---|---|
| `results/runs.jsonl` — the designed grid | 5424 | 0 |
| 13-layer sweep (`results/shards/sweep2_*.jsonl`) | 2340 | 0 |
| eps asymptote probe | 120 | 0 |
| shift decomposition (`results/phase9_shift.jsonl`) | 120 | 0 |

Zero failures across 8004 runs and zero non-converged trials of 99,720.

Experimental configuration is **frozen against a git tag** and the runner
refuses to start if the working config has drifted from it. Three tags exist:
`grid-freeze-v1` (Stage 0 gate), `grid-freeze-v2` (Stage 1 screening),
`grid-freeze-v3` (Stage 2 and everything reported here). Each row records its
tag, its git SHA, four config facet hashes, and library versions.

Every `run_id` is a deterministic hash over 19 coordinates, and all 5424
recompute from their own recorded columns. Two unplanned determinism checks came
free: 23 sweep cells recomputed by a duplicate worker were bit-identical, and
151 sweep runs sharing coordinates with grid rows agree on all 57 non-volatile
fields.

## Leakage assertions

The original pipeline fitted alignment on the target **test** set and selected
models on target-test scores. Both are prevented mechanically here, not by
convention:

- Splits are **speaker-disjoint**, with `target_adapt` and `target_test`
  separated. Alignment may see `target_adapt` only.
- Every fitted alignment object records the utterance ids it was fitted on, and
  `assert_alignment_blind_to_target_test` runs on the **real fitted object in
  every run** — not on a mock, not once at startup.
- `fit_and_select` receives a source validation split and never receives target
  data at all. The target score is computed afterwards, by the caller, from a
  model already chosen.
- No `StandardScaler` inside any classifier, because standardisation *is* the
  `zscore` rung — doing it silently would collapse two conditions the paper
  reports as distinct.
- Axis pruning between stages was scored on `source_val` only.

## The A10 firewall

The conditional-shift diagnostic computes `MMD(X_src | y=k, X_tgt | y=k)`, which
requires **target test labels** by construction. That is legitimate as post-hoc
analysis and illegitimate anywhere near fitting or selection. Containment is
executable, not documentary:

- It lives in `src/ser/analysis/` and nothing else may import it.
  `assert_conditional_shift_firewall()` reads the source of `alignment`,
  `classifiers`, `run_grid` and `blending` to confirm none of them can reach it.
- The frozen result schema has **no field** for it, and the assertion checks
  that too. It is written only to `results/phase9_shift.jsonl`.
- Values below `shift.conditional_mmd_min_support` (50) are reported as
  undefined rather than as numbers, and per-class *n* accompanies every value.
- `tests/test_analysis_shift.py` simulates the exact regression A10 warns about
  — a "diagnostics" column carrying the conditional term — and confirms the
  assertion fires.

## Status

Twelve phases; see [PHASES.md](https://github.com/prashant290605/Speech-Emotion-Recognition/blob/rebuild/PHASES.md) for the plan and
[PROGRESS.md](https://github.com/prashant290605/Speech-Emotion-Recognition/blob/rebuild/PROGRESS.md) for the running log.

| Phase | | Status |
|---|---|---|
| 0 | Scaffold and reproducibility spine | complete |
| 1 | Reference integrity checker | script complete; 5 DOIs outstanding |
| 2 | Manifest, label map, splits, leakage tests | complete for {RAVDESS, CREMA-D} |
| 3 | Feature extraction and caching | complete for {RAVDESS, CREMA-D} |
| 4 | Metrics and trivial baselines | complete |
| 5 | Alignment and blending | complete |
| 6 | Classifiers with equal-budget search | complete |
| 7 | Grid runner (Stage 0/1/2) | complete — 4986 Stage 2 runs |
| 8 | Selection protocol and headline tables | complete |
| 9 | Shift decomposition | complete |
| 10 | Per-class analysis and figures | complete |
| 11 | Release packaging and LaTeX tables | complete |

IEMOCAP is **not** included: its licence requires a signed agreement with a
faculty signatory, which was not obtained. Every claim here is over two corpora
in both directions, and the manuscript says so.

The original pipeline is preserved untouched under [`legacy/`](https://github.com/prashant290605/Speech-Emotion-Recognition/tree/rebuild/legacy) for
traceability. It is **not** the entry point and should not be run for new work.

## What the rebuild changes

- Speaker-disjoint splits with an explicit `target_adapt` / `target_test`
  separation, asserted rather than asserted-in-prose.
- All 13 hidden layers cached, so layer aggregation is a searchable condition
  rather than an unexamined default.
- Two reporting protocols, both published: **validated** and **oracle**. The gap
  between them is a result in its own right.
- An ordered alignment ladder — identity, z-score, mean shift, CORAL, MK-MMD —
  by moments matched, rather than two arbitrary comparisons. The condition the
  original reported as "MMD" was a plain mean shift; it is kept as its own
  named rung, `mean_shift`, so that column's real value is measured.
- Chance, majority-class and prior-matched floors on every table and figure.
- Five seeds, paired cluster bootstrap over target-test speakers **and** seeds.
- A pre-registered primary comparison family, declared in code before any of its
  numbers were computed, Holm-corrected within that family.
- Equal tuning budget: 20 random-search trials per family, asserted identical.
  `max_iter` is a fixed convergence budget, not a searched hyperparameter, and
  convergence is asserted rather than warned about.
- Matched-n reverse direction, so a reported transfer asymmetry cannot be a
  6× training-set size difference in disguise.

## Environment

Python 3.12. Fully pinned; see [requirements.txt](https://github.com/prashant290605/Speech-Emotion-Recognition/blob/rebuild/requirements.txt).

```bash
python -m pip install -r requirements.txt && python -m pip install -e .
```

## Usage

```bash
ser --help
```

`ser inventory` reports repository state, configuration and open decisions.
On Windows, or without `make`:

```bash
PYTHONPATH=src python -m ser.cli inventory
```

Every experimental value lives in [`configs/default.yaml`](https://github.com/prashant290605/Speech-Emotion-Recognition/blob/rebuild/configs/default.yaml).
Configuration loading is strict — an unknown or missing key is an error, never a
silent default.

## Tests

```bash
python -m pytest
```

## Data

Raw corpora are licence-restricted and are not distributed here. RAVDESS and
CREMA-D are public downloads; IEMOCAP requires a signed agreement with USC.
Point `paths.raw_*` in the config at local copies. Feature caches and raw audio
are gitignored; `results/runs.jsonl` **is** committed, because it is the
provenance record every table and figure is generated from.

## Paper

The manuscript is written from [reports/RESULTS.md](https://github.com/prashant290605/Speech-Emotion-Recognition/blob/rebuild/reports/RESULTS.md), with
tables from `tables/*.tex` and figures from `figures/*.pdf`. Pre-revision
sources under [`legacy/`](https://github.com/prashant290605/Speech-Emotion-Recognition/tree/rebuild/legacy) contain the results this rebuild corrects and
should not be cited.

## License

No licence is currently granted. Treat this as research code accompanying work
in progress.
