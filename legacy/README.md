# Speech Emotion Recognition

Research repository for **cross-corpus Speech Emotion Recognition (SER)** with:

- hybrid **SSL + MFCC** representations
- configurable **alignment** (`none`, `mmd`, `coral`)
- configurable **blending** (`none`, `scalar`, `fwaa`, `gaa`)
- multiple **classifier heads** (`logreg`, `svm`, `mlp`, `aplin`, `transformer`)
- **in-domain** and **cross-domain** evaluation on RAVDESS, CREMA-D, and IEMOCAP

This repository is prepared for research presentation, academic reproducibility, and public code review. Raw datasets are **not** included.

## Repository Status

- Main research pipeline: [`strict_modular_ser.py`](./strict_modular_ser.py)
- Full experiment sweep: [`full_run.py`](./full_run.py)
- Legacy / earlier experiment scripts are retained for traceability and comparison.

## Project Structure

```text
.
├── strict_modular_ser.py        # Main modular SER pipeline
├── full_run.py                  # Exhaustive experiment runner
├── phase*.py                    # Phase-based experimental scripts
├── run_all_phases.py            # Sequential runner for phase scripts
├── progress_utils.py            # Logging helpers
├── results_utils.py             # Result serialization helpers
├── svm_utils.py                 # SVM experiment utilities
├── coral.py                     # CORAL alignment utilities
├── ravdess_preprocessing.py     # Dataset preprocessing helpers
├── baseline_ser_mfcc.py         # Legacy MFCC baseline
├── cross_dataset_eval.py        # Legacy cross-dataset baseline
├── wav2vec2_cross_dataset_eval.py
│                                # Earlier wav2vec2 pipeline
├── SER_Report.tex               # Current research paper source
├── cross_corpus_ser_paper.tex   # Older paper draft
├── requirements.txt             # Reproducible Python dependencies
└── README.md
```

## Environment

- Recommended Python: **3.10 or 3.11**
- PyTorch + Transformers are required for SSL backbones.
- `matplotlib` is used for confusion matrix export.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset Preparation

Datasets must be downloaded manually and kept **outside version control**.

Expected local directories:

```text
Radvess/
Crema D/
IEMOCAP/
MAED/     # optional; local support exists but not required for the main paper sweep
```

### Notes

- The repository `.gitignore` excludes all raw corpora and large media.
- Do **not** commit dataset folders, extracted features, or cached model artifacts.
- If you want a single common root, place datasets in the repository root as above.

## Main Pipeline

The main pipeline implemented in [`strict_modular_ser.py`](./strict_modular_ser.py) follows:

```text
audio
├── SSL branch   -> extract_ssl -> align_ssl -> blend_ssl
├── MFCC branch  -> extract_mfcc
└── fusion       -> concat([SSL_final ; MFCC]) -> classifier
```

Key design constraints:

- alignment is applied **only** to SSL features
- MFCC features are **never aligned**
- fusion happens **after** SSL blending
- the classifier sees only the concatenated hybrid representation

## Running Experiments

### Quick single experiment

```bash
python - <<'PY'
from strict_modular_ser import run_experiment

config = {
    "src_dataset": "ravdess",
    "tgt_dataset": "crema",
    "backbone": "hubert",
    "alignment": "mmd",
    "blending": "gaa",
    "alpha": None,
    "classifier": "logreg",
}

print(run_experiment(config))
PY
```

### Fast debug run

```bash
python - <<'PY'
from strict_modular_ser import run_experiment

config = {
    "src_dataset": "ravdess",
    "tgt_dataset": "crema",
    "backbone": "hubert",
    "alignment": "coral",
    "blending": "gaa",
    "alpha": None,
    "classifier": "transformer",
    "ultra_debug": True,
}

print(run_experiment(config))
PY
```

### Full sweep

```bash
python full_run.py
```

The full sweep covers:

- in-domain and cross-domain pairs for `ravdess`, `crema`, `iemocap`
- plain baseline runs
- alignment / blending / classifier combinations

## Current Results Snapshot

The repository includes logged full-run results in local `results/results.json` during experimentation. The strongest observed cross-domain settings from the current run history are:

| Source -> Target | Backbone | Alignment | Blending | Classifier | Macro-F1 |
|---|---|---|---|---|---:|
| ravdess -> crema | hubert | mmd | gaa | logreg | 0.4111 |
| ravdess -> crema | hubert | mmd | scalar (`alpha=0.5`) | logreg | 0.3995 |
| ravdess -> crema | hubert | mmd | none | mlp | 0.3775 |
| ravdess -> crema | hubert | mmd | none | logreg | 0.3730 |
| crema -> ravdess | hubert | mmd | gaa | svm | 0.3715 |

Observed aggregate trends from the recorded run matrix:

- **HuBERT** is the strongest backbone on average.
- **CORAL** and **MMD** both help on average, but gains are pair-dependent.
- **GAA** appears in several top configurations, but is not uniformly best in average-case analysis.
- **Logistic regression** remains highly competitive despite stronger nonlinear heads being available.

## Outputs

The main pipeline writes experiment artifacts locally under:

```text
results/
└── confusion/
feature_cache/
```

These are intentionally ignored by Git.

## Reproducibility Notes

- Set a consistent random seed where exposed by the scripts.
- Cached features are separated from source code and excluded from version control.
- The project includes explicit dependency versions in [`requirements.txt`](./requirements.txt).
- The main modular pipeline supports reproducible configuration dictionaries for experiments.

## Legacy Scripts

Several scripts remain in the repository because they document earlier experimental phases:

- `phase0_baseline.py` to `phase13_svm_gaa_reverse.py`
- `wav2vec2_cross_dataset_eval.py`
- `baseline_ser_mfcc.py`
- `cross_dataset_eval.py`

These are kept for research traceability, not because they are the preferred public entry points. For new work, use:

- [`strict_modular_ser.py`](./strict_modular_ser.py)
- [`full_run.py`](./full_run.py)

## Paper

- Current report: [`SER_Report.tex`](./SER_Report.tex)
- Earlier draft retained: [`cross_corpus_ser_paper.tex`](./cross_corpus_ser_paper.tex)

## What Is Not Committed

The public repository should **not** contain:

- raw datasets
- extracted features
- model checkpoints
- logs / outputs / cache
- virtual environments
- notebook checkpoints
- local secrets or `.env` files

## License / Usage

Add a project license before public release if you want explicit reuse permissions. Until then, treat the repository as research code accompanying ongoing work.
