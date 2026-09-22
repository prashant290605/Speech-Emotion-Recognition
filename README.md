# What source validation cannot see

Repository for *"What source validation cannot see: an affine adaptation audit
for cross-corpus speech emotion recognition"* (manuscript in preparation for
*Speech Communication*).

The paper asks one question: **can a source-side validation protocol be
completely leakage-free and still be structurally unable to distinguish an
adaptation operation that changes target predictions?** For source translations
with stationary-kernel classifiers, the answer is exact, and the frozen
RAVDESS/CREMA-D runs in this repository contain a direct test of it.

Work happens on the `rebuild` branch. `main` is the pre-rebuild history.

---

## The central result

Translating every source training and validation vector by a fixed offset
preserves their kernel matrices, so an RBF-SVM's source-validation predictions
are unchanged, while the target decision function becomes `f(x - delta)` and
target predictions can change.

A retrospective audit of the frozen ledger finds **exact equality of the
selected validation score and of the selected hyperparameters in all 30 matched
RBF-SVM cells per direction**. Logistic regression, a linear SVM and an MLP are
reported alongside as implementation contrasts that fall outside the
proposition's fitting assumptions; they do not retain exact numerical equality,
which is what those assumptions predict.

```bash
PYTHONPATH=src python tools/audit_translation.py
```

That command needs only `results/runs.jsonl` and
`configs/audit_translation.yaml`, both committed. It reads the ledger and
writes `reports/translation_audit.{json,md}` and `tables/translation_audit.tex`.

Three kinds of evidence are kept distinct throughout, and should not be
conflated when reading the manuscript:

| | what it is | status |
|---|---|---|
| **The frozen experiment** | 4986 confirmatory runs under `grid-freeze-v3`, enumerated and run before any of its numbers were computed | pre-specified |
| **The retrospective audit** | the translation analysis above, designed after the runs existed | retrospective, and labelled as such in the paper |
| **Sensitivity analyses** | calm-drop and neutral-exclusion label controls, MMD measurement-protocol checks | controls, separately frozen under their own tags |

---

## Reproducing the analysis from a clean clone

**Everything the manuscript reports regenerates from committed files.** No raw
audio, no SSL feature caches and no per-utterance prediction files are needed.

```bash
git clone https://github.com/prashant290605/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition && git checkout rebuild
python -m pip install -r requirements.txt && python -m pip install -e .
python -m pytest
```

Verify inputs without generating anything:

```bash
python tools/build_paper.py --check-only
```

Regenerate every report, table and figure:

```bash
python tools/build_paper.py --analysis
```

Regenerate and compile the PDF (needs Tectonic on `PATH`; the script will not
download one):

```bash
python tools/build_paper.py
```

The build runs in stages and stops at the first failure: frozen-ledger digest,
tracked artifacts present, reports, tables and figures, translation audit,
`tools/check_paper.py`, `tools/check_number_trace.py`, LaTeX compile, then a log
inspection that fails on any undefined reference or citation. Output goes to
`output/paper/`.

### Tracked analysis inputs

Listed in `ser.artifacts.REQUIRED_ARTIFACTS`, which is what `--check-only`
verifies.

| artifact | what it carries |
|---|---|
| `results/runs.jsonl` | the 5424-row result ledger, of which 4986 are the frozen grid |
| `results/layer_sweep_v2.jsonl` | the 2340-run 13-layer sweep, packaged from its shards |
| `results/speaker_confusions.jsonl.gz` | per-speaker confusion counts: the sufficient statistic for every cluster bootstrap |
| `data/manifest_portable.csv` | corpus, speaker, session, labels and durations, with corpus-relative audio paths |
| `results/eps_asymptote_full.jsonl` | the 120-run CORAL shrinkage probe, packaged from its sources |
| `results/phase9_*.jsonl`, sensitivity ledgers | shift decomposition and label controls |
| `configs/default.yaml`, `configs/FROZEN_LEDGER.sha256` | the frozen configuration and the ledger's expected digest |

Four of these are derived artifacts, each built by a script that verifies its
own inputs and records a provenance sidecar. They exist because the analyses
that depend on them previously required gitignored, machine-local state:

```bash
python tools/merge_layer_sweep.py        # results/shards/sweep2_*.jsonl -> layer_sweep_v2.jsonl
python tools/merge_eps_probe.py          # results/shards/eps_*.jsonl     -> eps_asymptote_full.jsonl
python tools/make_portable_manifest.py   # data/manifest.csv              -> data/manifest_portable.csv
python tools/make_speaker_confusions.py  # results/predictions/           -> speaker_confusions.jsonl.gz
```

Each has a `--check`/`--verify` mode that rebuilds in memory and compares
against what is committed, so a reviewer can confirm the derivation without
holding the source data. `make_speaker_confusions.py --verify` additionally
proves exact equivalence: per-speaker tensors identical, pooled confusions
identical, zero difference in macro-F1, per-class F1, and a full paired
bootstrap under identical seeds, across all 5364 runs with stored predictions.

### What still requires the raw corpora

| task | needs |
|---|---|
| Building `data/manifest.csv` from scratch | RAVDESS and CREMA-D audio |
| Feature extraction (`ser extract`) | audio, plus GPU/CPU time and ~4.4 GB of cache |
| Re-running the grid (`ser run-grid`) | the feature caches; about 250 hours of CPU |
| Utterance-level analysis (McNemar, per-clip inspection) | `results/predictions/`, which is not tracked |

RAVDESS and CREMA-D are public downloads but are **not redistributed here**.
Point `paths.raw_*` at local copies. IEMOCAP is **not** included: its licence
requires a signed agreement with a faculty signatory, which was not obtained.
Every claim in the paper is over two corpora in both directions, and the
manuscript says so.

---

## The frozen experiment

The configuration is frozen against a git tag and the runner refuses to start if
the working config has drifted. Three tags exist: `grid-freeze-v1` (Stage 0
gate), `grid-freeze-v2` (Stage 1 screening), `grid-freeze-v3` (Stage 2 and
everything the paper reports). Every row records its tag, its git SHA, four
config facet hashes and library versions.

Each `run_id` is a deterministic hash over 19 experimental coordinates, and all
5424 recompute from their own recorded columns. `config_hash` is deliberately
not one of the coordinates.

The ledger itself is also pinned. `configs/FROZEN_LEDGER.sha256` holds the
expected SHA256 of `results/runs.jsonl`, and the retrospective audit and the
build wrapper both refuse to run when the file no longer matches:

```bash
python -c "import sys; sys.path.insert(0,'src'); from ser.freeze import assert_ledger_unchanged; print(assert_ledger_unchanged())"
```

This is a detector, not a lock. It does not make the file read-only and it does
not restrict any new result file; it makes an accidental overwrite of the
historical ledger loud instead of silent.

### Analysing a different frozen experiment

`ser.artifacts.PUBLICATION_FREEZE_TAG` names the tag the published build
resolves to. `filter_by_freeze_tag` selects exactly one tag and raises on one
that no row carries, so there is no code path that unions two frozen
configurations into a single summary, and none that silently returns an empty
selection.

---

## Leakage assertions

The original pipeline fitted alignment on the target **test** set and selected
models on target-test scores. Both are prevented mechanically, not by
convention:

- Splits are speaker-disjoint, with `target_adapt` and `target_test` separated.
  Alignment may see `target_adapt` only.
- Every fitted alignment records the utterance ids it was fitted on, and
  `assert_alignment_blind_to_target_test` runs on the **real fitted object in
  every run**.
- `fit_and_select` receives a source validation split and never receives target
  data at all. The target score is computed afterwards, from a model already
  chosen.
- No `StandardScaler` inside any classifier, because standardisation *is* the
  `zscore` rung.
- The conditional-shift diagnostic reads target labels and is firewalled into
  `src/ser/analysis/`; an executable assertion reads the source of the
  alignment, classifier, grid-runner and blending modules to confirm none can
  reach it.

---

## Tests

```bash
python -m pytest
```

Tests marked `ledger` read committed repository artifacts rather than synthetic
fixtures. They are part of the default run; `-m "not ledger"` deselects them.

Coverage worth naming: Proposition 1 is certified end to end against a real
`sklearn.svm.SVC`, the translation audit is pinned to the committed ledger by
digest, the compact bootstrap artifact is proved equivalent to the predictions
it replaces, and the portable manifest is checked for absolute-path leakage.

---

## Documentation

- [`docs/inference_estimands.md`](docs/inference_estimands.md) — what each
  manuscript table estimates, which interval procedure it uses, and why. Read
  this before comparing two tables' intervals: the paper uses two estimators
  deliberately.
- [`PHASES.md`](PHASES.md) — the rebuild plan.
- [`PROGRESS.md`](PROGRESS.md) — the running log.
- [`paper/BUILD_REPORT.md`](paper/BUILD_REPORT.md) — the current build's checks
  and outstanding warnings.
- [`reports/RESULTS.md`](reports/RESULTS.md) — every number the manuscript
  cites, with intervals, baselines and run filters.

The original pipeline is preserved untouched under [`legacy/`](legacy/) for
traceability. It is **not** the entry point and should not be cited.

---

## Environment

Python 3.12, fully pinned; see [requirements.txt](requirements.txt).

```bash
python -m pip install -r requirements.txt && python -m pip install -e .
```

`ser inventory` reports repository state, configuration and open decisions.
Every experimental value lives in [`configs/default.yaml`](configs/default.yaml);
configuration loading is strict, so an unknown or missing key is an error rather
than a silent default.

## Licence

No licence is currently granted. Treat this as research code accompanying work
in progress.
