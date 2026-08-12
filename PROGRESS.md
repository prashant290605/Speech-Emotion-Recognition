# PROGRESS

Running log of the cross-corpus SER reproducibility rebuild. One dated entry per
phase. Each entry lists files created, files modified, tests added, decisions
made, and anything deferred.

Phase briefs live in [PHASES.md](PHASES.md). Sessions do not share memory: this
file plus PHASES.md is the entire handover between them.

---

## 2026-08-10 — Phase 0 addendum: decisions settled, schema v2

Status: **complete**. `pytest` → 85 passed. Decisions below are now encoded in
`configs/default.yaml` and mirrored into PHASES.md as amendments A1–A6, which
override the phase text wherever they conflict.

### Decisions

**A1 — Decision A: fix the Transformer.** Confirmed. 8-segment cache retained;
the pooled-vector-reshaped-into-pseudo-tokens variant stays unconfigurable.
Expectation set for the paper: the Transformer will probably lose to the MLP at
these data sizes, and that gets reported. A fair weak baseline is publishable; a
strawman is not.

**A2 — Decision B: implement the ladder, keep the mean shift.**
`X_src + (μ_tgt − μ_src)` is exactly the minimiser of linear-kernel MMD, since
linear-kernel MMD reduces to ‖μ_s − μ_t‖². The original therefore implemented a
degenerate special case of what it claimed, which converts the finding into an
ordered ablation rather than a correction:

| Condition | Moments matched |
|---|---|
| `zscore` | per-dimension 1st + 2nd, no cross-terms |
| `mean_shift` (= linear-kernel MMD) | 1st |
| `coral` | 1st + 2nd, with covariance |
| `mmd` | all, via RBF |

MK-MMD is specified as an affine map fit by minimising MMD, not a divergence
gestured at: learn `W, b` minimising
`MMD²_k(W·X_src + b, X_target_adapt) + λ‖W − I‖²_F`, `k` a sum of RBF kernels at
`{0.25, 0.5, 1, 2, 4} × σ_median`, fitted on `source_train` and `target_adapt`
only. λ is `alignment.mmd_identity_penalty` (0.1).

No alias from `mmd` to `mean_shift` exists, and a test asserts both are
independently nameable — the alias is how the original misstatement would survive
into v2.

**A3 — label decisions.** `iemocap_excited_to_happy: true` (standard convention;
happy ~595 → ~1636). `iemocap_frustrated: drop` — **not** merged into angry:
frustration (~1849) against anger (~1103) would make `angry` ~30% of retained
IEMOCAP, manufacturing the label-prior skew the thesis then claims to discover,
and it has no counterpart in RAVDESS or CREMA-D. `iemocap_subsets: both` with
`iemocap_record_subset: true`, so improvised-only is a free robustness check
later. `ravdess_calm_to_neutral: true` — calm exists only in RAVDESS and cannot
transfer; merging gives neutral 288 vs 192, dropping would give 96 vs 192.

**Manuscript correction required:** RAVDESS is balanced at 8 classes, **not** at
the 6-class intersection. The "RAVDESS is exactly balanced" line in the current
draft is an easy factual hit in review. The KL analysis uses empirical priors, so
the thesis is unaffected.

**A4 — IEMOCAP pairs run as canonical 4-class.** Disgust ≈ 2 utterances, fear
≈ 40; across 10 speakers that is ~4 fear utterances per speaker-disjoint fold, so
macro-F1 collapse is guaranteed by construction on exactly the pairs the draft
attributes to "structural mismatch". The `five` space is replaced by
`four: [angry, happy, neutral, sad]`. Phase 2 still reports fear and disgust
support explicitly.

**Consequence, and it is a real one:** chance floors now differ by pair (0.167 for
RAVDESS↔CREMA-D, 0.250 for IEMOCAP pairs), so any statistic averaging macro-F1
across pairs is ill-defined, not merely uninformative. The "mean cross-domain
macro-F1" headline is removed from Phase 8; reporting is per pair against
per-pair chance.

**A6 — the full factorial is dead.** Ladder (5) × α (5) × layer aggregation (3) ×
5 seeds on top of the existing axes is six figures of runs. Phases 6/7 become a
screening pass on one pair and one backbone to prune axes, then a reduced
factorial with seeds. Pruning is scored on `source_val`, never target test.

### Schema v2 (A5)

`label_map_hash` and `split_spec_hash` added, both non-nullable, both `run_id`
coordinates. 42 fields. Rationale: `config_hash` alone is too coarse in one
direction (changes when an unrelated key changes) and potentially too fine in the
other; without these, changing `iemocap_frustrated` mid-project would leave
`run_id` unchanged and a Phase 7 resume would silently merge runs scored against
different label spaces, with nothing downstream able to detect it.

`labels.label_map_version` and `splits.split_spec_version` are hand-bumped when
mapping or split *semantics* change in ways the config keys do not express.

`blend_alpha` was already a `run_id` coordinate, so the α axis is covered. Per-group
`gaa` alphas are selected on `source_val` inside a run, so they are an output and
live in `hyperparams_json` — documented in `RUN_ID_FIELDS`.

**One deletion under `results/`.** `results/runs.jsonl` held exactly one synthetic
schema-v1 smoke row written earlier the same day; it could not validate under v2.
It was removed and regenerated. No experimental data existed at any point. Noting
it because the standing rules forbid deleting under `results/`.

### Files modified

```
configs/default.yaml         label spaces six/four, five decisions, the ladder,
                             MMD affine spec, label_map_version, split_spec_version
src/ser/config.py            LabelsConfig restructured (DECISION_FIELDS, enum for
                             iemocap_frustrated, space_for_* references),
                             AlignmentConfig.LADDER + ladder_order(),
                             Config.label_map_hash / split_spec_hash
src/ser/utils/results.py     SCHEMA_VERSION 1 -> 2, two new fields, RUN_ID_FIELDS
src/ser/cli.py               smoke and inventory carry the new coordinates
PHASES.md                    AMENDMENTS A1-A6 + inline notes in Phases 2,4,5,6,7,8
README.md                    created at root (see below)
tests/                       updated and extended, 76 -> 85
```

### Tests added

Nine new, 85 total. Notable: the ladder is ordered by moments matched and
`ladder_order()` is insensitive to config ordering; `mean_shift` and `mmd` are
independently nameable and `mmd_mean_shift` is rejected; every label decision is
made and `iemocap_frustrated != "merge_angry"`; `label_map_hash` moves when a
mapping decision, the map version, or a label space changes, and `split_spec_hash`
when a ratio or split unit changes; both are insensitive to unrelated config
changes while `config_hash` is not; both are asserted present in `RUN_ID_FIELDS`.

### Public README

Root `README.md` created. The previous README (now `legacy/README.md`) advertised
a results table headed by `0.4111` and the aggregate claim that "CORAL and MMD
both help on average" — from a pipeline that fitted alignment on the target test
set and whose MMD was a mean shift. The results table and all trend claims are
gone, replaced with a prominent "results under revision" notice that names the
specific defects, plus a phase status table.

### Deferred / still open

- **Not pushed.** Committed locally to branch `rebuild`; `main` still matches the
  public five-commit tree. Pushing the corrected README is a separate decision.
- **IEMOCAP licence.** `data/raw/` is empty. RAVDESS and CREMA-D are immediate
  downloads; IEMOCAP is a signed USC agreement with a multi-week turnaround and
  is the critical-path item for Phase 2. Status unconfirmed.
- The ~1849 / ~1103 / ~595 / ~1636 / ~40 / ~2 IEMOCAP counts above are literature
  estimates. Phase 2 must verify every one against the manifest and correct these
  entries if they differ.

---

## 2026-08-10 — Phase 0: Scaffold and reproducibility spine

Status: **complete**. `pytest` → 76 passed. `ser smoke` writes and revalidates
one row in `results/runs.jsonl`.

### Inventory (first action)

The working directory was empty. The original study was cloned from
`https://github.com/prashant290605/Speech-Emotion-Recognition` (HEAD `c2f7738`,
5 commits, 730 KiB) and its git history retained as this repository's history.
All 33 original files were moved **untouched** to `legacy/`.

**What exists:** 30 Python modules, `SER_Report.tex`, `cross_corpus_ser_paper.tex`,
`Speach_Emotion_Recognition.pdf`, `requirements.txt`.

**What does not exist:** no cached features, no `results/` artefacts, no logs, no
`.bib` file, no raw audio. **There are no original result files.** The only record
of the original numbers is the tables inside the two `.tex` files and the PDF. The
rebuild therefore cannot diff against original per-run outputs; comparisons to the
original are limited to published table values.

Entry point is `legacy/strict_modular_ser.py` (40 KiB); the `legacy/phase*.py`
scripts are one-off drivers from the original development sequence and are
unrelated to the phase numbering in PHASES.md.

### Findings from reading the original code

These change what later phases must do. Line references are into `legacy/`.

1. **`"mmd"` was a mean shift, not MMD.** `strict_modular_ser.py:507-511`:
   `X_aligned = X_src + (mu_tgt - mu_src)`. No kernel, no learned map, no
   optimisation, no bandwidth. The paper's claim that MMD "yields an aligned
   representation" does not describe this code. **Resolves Decision B's premise;
   the decision itself is still open** — see Open questions.

2. **Transductive leakage into alignment.** `strict_modular_ser.py:959-972`
   builds `X_ssl_tgt_all = vstack([tgt X_ssl_train, tgt X_ssl_test])` and passes
   it to `align_ssl` as the target. Every CORAL/mean-shift/SA transform in the
   original was fitted on the target **test** set. This is precisely what the
   Phase 2 test 3 and the Phase 5 `fitted_on_indices` assertion exist to prevent.

3. **Final-layer-only pooling confirmed.** `strict_modular_ser.py:475-476` uses
   `outputs.last_hidden_state` and mean-pools it. Phase 3's all-layer cache is the
   fix; keeping `last` as a Phase 6 condition quantifies the cost.

4. **One hardcoded seed, and it is re-applied per run.** `RANDOM_SEED = 42`
   (`strict_modular_ser.py:36`); `_train_torch_classifier` calls
   `torch.manual_seed(42)` on entry, so every torch run shares an initialisation
   and there is no seed variance anywhere in the study.

5. **No validation set existed.** `train_classifier(X_train, y_train, X_test,
   y_test, ...)` passes the target test set straight into training-time code and
   scores on it. There was no `source_val`, so no non-target selection surface
   was even available.

6. **The paper's training description does not match its code.**
   `SER_Report.tex:159` says "Adam, learning rate 1e-3, fixed mini-batches, and
   eight epochs". `_train_torch_classifier:712-717` runs **8 full-batch gradient
   steps** — no mini-batching, no epoch loop, no early stopping. Eight updates
   total on the whole training set.

7. **Transformer strawman confirmed.** `_TransformerHead:641-669` pads the pooled
   768-d vector and reshapes it to 16 tokens × 48 dims, `nhead=1`, 2 layers.
   Attention runs over an arbitrary partition of feature dimensions, not time.
   Decision A's premise holds.

8. **Two divergent CORAL implementations.** `coral.py::coral_align` rescales the
   transformed source so its covariance trace matches the target's;
   `strict_modular_ser.py::align_ssl` (the one `run_experiment` calls) does not.
   `coral.py` appears to be dead code. Which one produced the published numbers is
   not recoverable from the repo — assume `align_ssl`.

9. **Blending α was never searched.** `blend_ssl:574-608`: `fwaa` and `gaa` derive
   α from the magnitude of `|X_aligned − X_orig|`, and `gaa` hardcodes
   `n_groups = 16` with contiguous dimension slices, not k-means. Only `scalar`
   took an α argument. The rebuild's "α selected on `source_val`" is a change in
   kind, not a fix to the selection surface — note this in the paper.

10. **A fourth alignment method exists in code:** `sa` (subspace alignment via
    PCA, `align_ssl:513-530`). Not mentioned in PHASES.md. Not carried forward
    unless asked.

11. **IEMOCAP speaker IDs are probably wrong.** `_parse_iemocap_annotations:347`
    sets `speaker_id = utterance_id.split("_")[0]`, e.g. `Ses01F` — that is the
    session plus *lead actor* tag of the dialogue, not the speaker of the
    utterance. Both actors in a session speak in both its `F` and `M` dialogues,
    so the same physical speaker can land on both sides of a "speaker-disjoint"
    split. Phase 2 must verify this against the real corpus. It is an independent
    argument for the session-level split PHASES.md already mandates.

12. **A fourth corpus is wired in.** `_parse_mead_records` handles MEAD. Out of
    scope for the rebuild.

13. **Label decisions the original made** (`_normalize_label:147-202`), recorded
    for Phase 2, not adopted: IEMOCAP `excited`→`happy`; IEMOCAP
    `frustrated`→`angry`; RAVDESS `calm`→`neutral`; `surprised`/`contempt`/`xxx`/
    `oth` dropped; `disgust` dropped in 5-class; IEMOCAP glob
    `Session*/dialog/EmoEvaluation/*.txt` takes **both** scripted and improvised.
    `frustrated`→`angry` is aggressive and materially inflates the `angry` prior.

14. **The 972 figure is in the manuscript.** `SER_Report.tex:159` states the grid
    "yield[s] 972 runs", matching the duplicate-enumeration problem PHASES.md
    describes.

### Files created

```
pyproject.toml                          packaging + pytest config (pythonpath=src)
requirements.txt                        fully pinned lock, verified on this machine
.gitignore                              raw audio and caches out; runs.jsonl in
Makefile                                thin wrapper; delegates to the ser CLI
PHASES.md                               the phase plan, copied in verbatim
PROGRESS.md                             this file
configs/default.yaml                    every experimental value
src/ser/__init__.py
src/ser/config.py                       dataclass config, strict loading
src/ser/cli.py                          `ser` entrypoint, stub per later phase
src/ser/utils/__init__.py
src/ser/utils/seeding.py                set_all_seeds
src/ser/utils/runmeta.py                provenance capture
src/ser/utils/results.py                frozen schema + append-only JSONL writer
tests/test_smoke.py
tests/test_config.py
tests/test_results_schema.py
tests/test_seeding_and_runmeta.py
data/{raw,cache}/  reports/  figures/  results/
```

### Files modified

None. `legacy/` is byte-identical to the upstream clone.

### Tests added

76 tests, all passing.

- `test_smoke.py` (4) — Phase 0 acceptance: full round trip config → seed →
  provenance → schema → JSONL → re-read; append-only behaviour; unbuilt-phase
  commands exit 2; `inventory` runs.
- `test_config.py` (34) — strict loading (unknown/missing key and section),
  every range check, config-hash stability under reformatting and sensitivity to
  content, the shipped default matching the values PHASES.md specifies, and the
  open-decision mechanism.
- `test_results_schema.py` (25) — every field from the brief present, rejection of
  unknown/missing/mistyped/null-in-non-nullable values, `bool` not accepted as
  `int`, `status="failed"` rows accepted with null metrics, `run_id` determinism
  and sensitivity to *every* coordinate, append round trip, corrupt-line reporting.
- `test_seeding_and_runmeta.py` (13) — seed determinism for `random`/`numpy`/
  `torch`, cuDNN flags, invalid seeds rejected, hash order-independence, provenance
  fields populated, and "no git repo" recorded as `unknown`/dirty rather than faked.

### Decisions made

- **Repository identity.** The original clone's git history was kept rather than
  starting fresh, so `git log` reaches the code that produced the published
  numbers and `runmeta` SHAs are meaningful across the whole rebuild.

- **Four fields added to the brief's schema, before anything was run.** All four
  are required by the brief's own text and are cheaper to add now than after the
  grid: `git_dirty`, `hostname`, `lib_versions_json` (Phase 0 says "every result
  row carries this" of the full runmeta), and `status` + `error` (Phase 7 requires
  `status="failed"` rows with a traceback). Also `schema_version`, so a future
  change is detectable rather than silent. Metric columns are nullable so a
  failed run is recordable. Total 40 fields.

- **`run_id` is a deterministic hash of 14 experimental coordinates**
  (`RUN_ID_FIELDS`), excluding hyperparameters, metrics, and provenance. This is
  what makes Phase 7 resumable. A test asserts no coordinate is inert.

- **`ser smoke` writes to the real `results/runs.jsonl`** as the acceptance
  criterion requires, tagged with the reserved corpus name `smoke`.
  `results.is_smoke_row()` lets Phase 8 exclude it mechanically rather than by
  convention. There is currently **1 smoke row** in `results/runs.jsonl`.

- **The CLI is canonical, the Makefile delegates.** `make` is not installed on
  this machine; every target is `PYTHONPATH=src python -m ser.cli <command>`.

- **Open label decisions are `null` in the config and halt the pipeline.**
  `Config.require_decision()` raises while a decision is unmade rather than
  falling back to a default. Rule 3 expressed in code.

- **Config encodes Decision A as "fix"** (`segment_pooling_enabled: true`,
  `transformer` in families), per the plan's recommendation, and a cross-section
  validator makes "Transformer without the segment cache" unconfigurable. Awaiting
  confirmation — see Open questions.

- **Pins were taken from the versions verified on this machine**, not from
  `legacy/requirements.txt`. Backbone checkpoints match the original
  (`facebook/hubert-base-ls960`, `facebook/wav2vec2-base`, `microsoft/wavlm-base`)
  so the `last`-layer condition remains a fair reproduction of the original.

### Deferred

- **Concurrent-write file locking** on `results/runs.jsonl`. `append_row` is
  append-only and fsynced, which is safe for one writer; the lock belongs to
  Phase 7 with the parallel runner.
- **`layer_agg` spec strings** (`mean:a-b`) are named in the schema docstring but
  the parser is Phase 3 (`features/aggregate.py`).
- **Legacy `sa` alignment and MEAD corpus** — present in `legacy/`, not carried
  forward.
- **No `.bib` exists** for Phase 1. `tools/check_refs.py` will have to parse the
  reference list out of `legacy/cross_corpus_ser_paper.tex` or a `.bib` must be
  produced first.

### Open questions for the next session

1. **Decision A — confirm.** Config currently assumes *fix* (8-segment cache,
   Transformer retained). Say if you want *drop* instead; it is a two-line config
   change plus deleting the Phase 3 segment deliverable.

2. **Decision B — now a real choice, not a guess.** The original "MMD" was a mean
   shift (finding 1). Three options, and the paper must state which:
   (a) implement the proper multi-kernel MMD from PHASES.md and report it as MMD,
   noting the original operator was different;
   (b) implement it *and* keep the mean shift as a separate named condition, which
   costs one extra grid axis and directly measures what the original's "MMD"
   column was actually worth;
   (c) rename the original's operator to `mean_shift` and drop MMD entirely.
   Recommend (b) — it is the honest reading and turns a misstatement into a
   measured result.

3. **Phase 2 label decisions** (`iemocap_excited_to_happy`,
   `iemocap_frustrated_to_angry`, `iemocap_subsets`, `ravdess_calm_to_neutral`).
   All four are `null` and will halt Phase 2. Finding 13 records what the original
   did; `frustrated`→`angry` in particular deserves a deliberate decision rather
   than inheritance.

4. **Raw corpora are not present.** Phase 2 needs RAVDESS, CREMA-D, and IEMOCAP
   under `data/raw/`. Paths are configurable in `configs/default.yaml`.
