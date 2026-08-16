# Cross-Corpus SER Paper: Phased Rebuild Plan

Twelve phases, 0 through 11. Each phase is a self-contained brief you paste into a
fresh Claude Code session. Do not run two phases in one session.

Reference audit (Track R) runs in parallel and is manual. It does not block the code track.

---

## Standing preamble

Paste this at the top of **every** phase. Claude Code sessions do not share memory,
so this is what keeps each session inside its lane.

```
CONTEXT
Project: reproducibility rebuild of a cross-corpus speech emotion recognition study.
Corpora: RAVDESS, CREMA-D, IEMOCAP. Backbones: HuBERT Base, wav2vec 2.0 Base, WavLM Base.
The original version of this study had methodological problems we are systematically
fixing: test-set model selection, final-layer-only SSL pooling, no seeds, no chance
baselines, unspecified alignment operator. The rebuild must be defensible to a
journal reviewer.

RULES FOR THIS SESSION
1. Read PHASES.md and PROGRESS.md before doing anything.
2. Implement ONLY the phase specified below. Do not implement future phases,
   do not add "helpful" extras, do not refactor code outside this phase's scope.
3. If a design decision is genuinely ambiguous, STOP and ask. Do not guess and
   proceed.
4. Every result must be reproducible from a config file plus a seed. No values
   hardcoded in scripts.
5. Run `pytest` before declaring the phase done. All tests must pass.
6. At the end, append a dated entry to PROGRESS.md listing: files created, files
   modified, tests added, decisions made, and anything you deferred.
7. Do not delete or overwrite anything under data/cache/ or results/.
```

---

## AMENDMENTS — settled 2026-08-10, after the Phase 0 inventory

**These override the phase text below wherever they conflict. Read this section
before the phase you are implementing.** Rationale for each is in PROGRESS.md.

### A1. Decision A — Transformer: **fix**, not drop

8-segment cache retained; the Transformer operates on an `(8, D)` sequence. The
pooled-vector-reshaped-into-pseudo-tokens variant is *unconfigurable*, not merely
discouraged (`config.py` raises if `transformer` is requested without the segment
cache).

Expect the Transformer to lose to the MLP at these data sizes. That is fine and
gets reported. The original's problem was never that it underperformed — it was
that the discussion admitted the architecture was meaningless and shipped it.
A fair weak baseline is publishable; a strawman is not.

### A2. Decision B — MMD: implement the ladder, keep the mean shift

`X_src + (μ_tgt − μ_src)` is exactly the transform minimising MMD under a
**linear kernel** (linear-kernel MMD reduces to ‖μ_s − μ_t‖²). The original
implemented a degenerate special case of what it claimed. Report both as one
ordered ablation:

| Condition | Moments matched |
|---|---|
| `zscore` | per-dimension 1st + 2nd, no cross-terms |
| `mean_shift` (= linear-kernel MMD) | 1st |
| `coral` | 1st + 2nd, with covariance |
| `mmd` | all, via RBF |

Flat performance across the whole ladder is direct evidence the shift is not
covariate — an ordered series, not two arbitrary points.

MK-MMD is specified as an **affine map fit by minimising MMD**, not a divergence
gestured at. Learn `W, b` minimising

> `MMD²_k(W·X_src + b, X_target_adapt) + λ‖W − I‖²_F`

with `k` a sum of RBF kernels at bandwidths `{0.25, 0.5, 1, 2, 4} × σ_median`.
Fitted on `source_train` and `target_adapt` **only**.

**Name the mean-shift condition `mean_shift`. Do not alias `mmd` to it, ever** —
the alias is how the misstatement survives into v2.

### A3. Label decisions

| Decision | Value |
|---|---|
| `iemocap_excited_to_happy` | **merge** — standard convention; happy ~595 → ~1636 |
| `iemocap_frustrated` | **drop** — do *not* merge into angry |
| `iemocap_subsets` | **both**, subset recorded per utterance |
| `ravdess_calm_to_neutral` | **merge** |

Frustration (~1849 utterances vs anger's ~1103) merged into anger would make
`angry` ~30% of retained IEMOCAP — *manufacturing* the label-prior skew the
thesis then claims to discover. It is also absent from RAVDESS and CREMA-D.

RAVDESS is balanced at 8 classes, **not** at the 6-class intersection. The
"RAVDESS is exactly balanced" line in the current draft is wrong and must be
corrected.

### A4. IEMOCAP pairs are 4-class, and the cross-pair mean headline is dead

IEMOCAP disgust ≈ 2 utterances, fear ≈ 40 — about 4 fear utterances per
speaker-disjoint fold, so macro-F1 collapse is guaranteed by construction on
exactly the pairs the draft attributes to "structural mismatch". Label spaces are
now:

- `six`: angry, disgust, fear, happy, neutral, sad — RAVDESS ↔ CREMA-D
- `four`: angry, happy, neutral, sad — any pair involving IEMOCAP

Phase 2 still **reports** IEMOCAP fear and disgust support; it does not silently
drop them.

**Consequence:** chance floors differ by pair (0.167 vs 0.250). Any statistic
averaging macro-F1 across pairs is now not merely uninformative but ill-defined.
**Report per pair against per-pair chance.** This kills the "mean cross-domain
macro-F1" table in Phase 8 (see A6).

### A5. `run_id` carries the label map and split spec

Schema v2 adds `label_map_hash` and `split_spec_hash` as `run_id` coordinates.
Without them, changing a label decision mid-project leaves `run_id` unchanged and
a Phase 7 resume silently merges incompatible runs. `blend_alpha` is already a
coordinate; per-group `gaa` alphas are selected on `source_val` inside a run and
live in `hyperparams_json`.

Bump `labels.label_map_version` / `splits.split_spec_version` whenever the
*semantics* of the mapping or split code change in a way the config keys do not
express.

### A7. The IEMOCAP annotation rule is an explicit config key

The canonical counts — neutral 1708, frustrated 1849, angry 1103, sad 1084,
excited 1041, happy 595, fear 40, disgust 2, plus 2507 no-agreement — assume
**majority vote over the three categorical annotators, discarding
no-agreement**. Any-annotator labels, or folding in self-assessment, change every
count substantially, therefore every prior, therefore the entire shift analysis.

That rule is now `labels.iemocap_label_source`, inside `label_map_hash`. It was
previously an implicit property of the parsing code — a silent axis that could
change without moving the hash, which is exactly the failure mode the hash exists
to prevent.

### A8. The label-shift thesis is dead. Phase 9 becomes a shift decomposition.

**This is the most important amendment. Read it before Phase 9.**

The settled label decisions produce these priors:

| Corpus (4-class) | angry | happy | neutral | sad |
|---|---|---|---|---|
| IEMOCAP | .199 | .296 | .309 | .196 |
| RAVDESS | .222 | .222 | .333 | .222 |
| CREMA-D | .259 | .259 | .222 | .259 |

Pairwise KL over all six cross-domain pairs spans **0.0139 to 0.0336 nats**
(6-class RAVDESS↔CREMA-D: 0.0252). Verified from published counts; Phase 2 must
re-verify from the manifest.

The original premise — IEMOCAP heavily skewed, RAVDESS balanced — was true of
*raw* IEMOCAP. Dropping frustration and cutting fear removed almost all of the
skew. **The decisions that make the label space defensible are the same decisions
that dissolve the effect Phase 9 was built to find.** A Spearman correlation over
six points spanning 0.02 nats is not underpowered; it is undefined.

This cannot be fixed by restoring frustration. Merging it into anger would
manufacture the skew and then discover it, which is worse than having no thesis.

**The reframe.** Stop asking which shift dominates. Decompose all three, on
features that are cached anyway:

- **Label shift** — KL/JS between priors. Now measurably near-zero. That is a
  *result*: it eliminates the hypothesis cleanly and predicts that BBSE- and
  EM-style prior correction cannot help. Report it as an eliminated explanation,
  not as a null finding.
- **Covariate shift** — MMD between marginal source and target features. What
  CORAL and MK-MMD actually minimise; the A2 ladder measures how much each rung
  removes.
- **Conditional shift** — class-conditional MMD, `MMD(X_src | y=k, X_tgt | y=k)`
  per class, measured before *and* after alignment.

**The claim:** alignment fails on cross-corpus SER because it minimises marginal
discrepancy while class-conditional discrepancy is what moves the decision
boundary — evidenced by the ladder showing marginal discrepancy shrinking
monotonically while transfer macro-F1 does not. This holds whichever way the
numbers land, needs no new data, and is an explanation rather than a table.

**The mechanism-isolating experiment.** Treat IEMOCAP-improvised and
IEMOCAP-scripted as two corpora. Same speakers, same label space, near-identical
priors — so label shift and covariate shift are both structurally near zero and
only elicitation style differs. Degradation across that boundary is conditional
shift with the confounds held fixed, which none of the cited comparisons achieve.
Config: `grid.include_iemocap_subset_pair`. Costs one extra pair, and depends on
`labels.iemocap_record_subset`, already set.

This experiment exists only if IEMOCAP is in hand — it is the strongest argument
for waiting on the licence rather than substituting a smaller acted corpus.

### A9. KL is measured at split level, not corpus level

The 0.0139–0.0336 figures in A8 are **whole-corpus** priors. The quantity that
actually governs a run is the divergence between the realised `source_train`
prior and the realised `target_test` prior *after* speaker-disjoint splitting.

IEMOCAP has five sessions and ten speakers, and per-session emotion
distributions are not uniform, so a session-disjoint fold can move class
proportions by several points — and moves them differently per seed. Split-level
KL will therefore be **larger than corpus-level KL and will vary across the five
seeds**.

> **MEASURED 2026-08-12 — the prediction is wrong for RAVDESS and CREMA-D.**
> Over all 20 splits (4 pairs × 5 seeds), split-level KL sits within 0.0003 nats
> of the corpus-level figure, and in-domain pairs come out at ~0:
>
> | pair | corpus-level | split-level mean | spread over 5 seeds |
> |---|---|---|---|
> | ravdess → cremad | 0.0252 | 0.0251 | 0.0003 |
> | cremad → ravdess | 0.0224 | 0.0224 | 0.0000 |
> | ravdess → ravdess | 0.0000 | 0.0000 | 0.0000 |
> | cremad → cremad | 0.0000 | 0.0000 | 0.0000 |
>
> The cause is structural, not accidental: both are **acted corpora with a fixed
> per-actor recording protocol** — every RAVDESS actor records the same 60
> trials, every CREMA-D actor the same sentence × emotion grid — so class
> proportions are invariant under any speaker-disjoint partition. RAVDESS is
> exactly 0 by construction.
>
> A9's *reasoning* was specifically about IEMOCAP's non-uniform per-session
> distributions. That remains plausible and **remains untested** — it must be
> re-measured when IEMOCAP arrives. What is now known is that it does not
> generalise to the acted corpora.
>
> **The procedural requirement below still stands.** Reporting split-level and
> corpus-level side by side is what let this be checked at all; it simply turns
> out they coincide here. Do not drop the split-level column on the strength of
> two corpora.

This does not rescue the label-shift thesis and must not be used to. But "label
shift is near zero" asserted at corpus level and tested at split level is a
mismatch a careful reviewer will find.

- **Phase 2's halt-and-report guard compares corpus level to corpus level.** That
  is a data-integrity check against the published counts, and it stays that way.
- **Phase 8 reports split-level KL per pair per seed**, as mean and range,
  computed from the realised partitions. That is the number the analysis uses.

### A10. The conditional-shift diagnostic reads target test labels — firewall it

`MMD(X_src | y=k, X_tgt | y=k)` requires target labels by construction. That is
legitimate as post-hoc analysis and illegitimate anywhere near fitting or
selection. It is exactly the shape of the leak Phase 2 exists to prevent,
reintroduced under a respectable name. Required containment:

1. Computed **only** in the analysis layer (`src/ser/analysis/`), never in
   `alignment`, `classifiers`, or `run_grid`.
2. **Never written into any artifact the pipeline reads.** The frozen result
   schema has no field for it, and adding one would require a `SCHEMA_VERSION`
   bump and fail every existing row — that mechanical guard is load-bearing, so
   do not weaken it by adding a "diagnostics" column.
3. **Never an input to configuration selection**, under any framing, including
   axis pruning in the A6 screening pass.
4. Covered by an explicit assertion stating the above, not a comment.
5. Minimum support: `shift.conditional_mmd_min_support` (50). Below it the value
   is reported as undefined rather than as a number. **Per-class n is reported
   alongside every defined value.**

### A6. The full factorial is dead — Phases 6/7 must be staged

The ladder (5 alignments) × α (5) × layer aggregation (3) × 5 seeds on top of the
existing axes is six figures of runs. Required design:

1. **Screening pass** on one pair, one backbone, to prune axes.
2. **Reduced factorial** with seeds over what survives.

Pruning is scored on the **source-side validation split, never on target test** —
otherwise the screening pass reinvents the exact leak Phase 2 exists to prevent.
A designed ablation is a contribution; an exhaustive sweep is a table.

> **RESOLVED in schema v4 — `config_hash` is no longer a `run_id` coordinate.**
>
> It was, and that was wrong: **any** edit to `configs/default.yaml` — even to a
> section the run never reads — changed every `run_id` and orphaned every
> completed run. Observed for real in Phase 5, where editing the `alignment`
> section made the 60 baseline rows re-run rather than resume.
>
> `run_id` now uses **four facet hashes** instead, each covering the part of the
> config that actually determines what a run computes:
>
> | facet | covers | so an edit invalidates |
> |---|---|---|
> | `label_map_hash` | `labels` | runs whose labels could differ |
> | `split_spec_hash` | `splits` (minus `seeds`) | runs whose partitions could differ |
> | `feature_spec_hash` | `features` | runs whose features could differ |
> | `search_spec_hash` | `alignment`, `blending`, `classifiers`, `baselines`, `stats` | runs whose selected hyperparameter could differ |
>
> `splits.seeds` is excluded because the per-run `seed` is already its own
> coordinate — otherwise adding a sixth seed would invalidate the five that
> already ran. `config_hash` is still **recorded** on every row for provenance.
>
> **Every config key must be classified.** `Config.classify_config_key` maps each
> key to a facet, to a dedicated coordinate, or to `INERT_CONFIG_KEYS`
> (`project`, `paths`, `grid`, `shift` — naming, locations, which runs get
> enumerated, and post-hoc analysis parameters). A test walks the whole config
> and fails on any key that is in none of the three, because an unclassified key
> can change a result without changing `run_id`. This is the mirror of the
> existing test that no `run_id` coordinate is inert.
>
> **The freeze is now mechanical (schema v5).** The facets make an unrelated edit
> harmless, but an edit *within* a facet still invalidates that facet's runs, and
> `search_spec_hash` is deliberately broad. Worse: with `config_hash` no longer a
> coordinate, a mid-grid edit no longer *orphans* completed runs — it silently
> produces rows that are not comparable to earlier ones under the same ids. So:
>
> - `configs/FROZEN` holds a git tag name; the tagged commit's
>   `configs/default.yaml` is the frozen config.
> - `ser.freeze.assert_config_frozen` compares the **parsed** working config
>   against it, so reformatting and comments are not drift but a value change is.
> - **The Phase 7 runner refuses to start** when the config is unfrozen or has
>   drifted. A convention would not survive a two-week run.
> - `freeze_tag` is recorded on every row — but is deliberately **not** a `run_id`
>   coordinate, so re-freezing does not invalidate completed work.
>
> Freeze, screen, freeze, run.
>
> A `run_id` coordinate change cannot be migrated — old ids were computed over a
> different field set, so carrying them forward would assert an equivalence that
> does not hold. `tools/migrate_results.py` refuses and says so.

---

## Phase map

| Phase | Name | Compute | Blocks |
|---|---|---|---|
| 0 | Scaffold and reproducibility spine | none | everything |
| 1 | Reference integrity checker | none | nothing |
| 2 | Manifest, label map, splits, leakage tests, dataset stats | light | 3 onward |
| 3 | Feature extraction and caching | heavy, one time | 5 onward |
| 4 | Metrics and trivial baselines | none | 7 onward |
| 5 | Alignment and blending module | light | 7 |
| 6 | Classifier module with equal-budget search | light | 7 |
| 7 | Grid runner | heavy | 8 onward |
| 8 | Selection protocol and headline tables | none | 9 onward |
| 9 | Shift decomposition: label, covariate, conditional | light | 10 |
| 10 | Per-class analysis and figures | none | 11 |
| 11 | Release packaging and LaTeX table generation | none | submission |

---

# PHASE 0 — Scaffold and reproducibility spine

**Goal.** Build the skeleton every later phase writes into. No science in this phase.

**First action: inventory.** Before writing anything, search the repo and any
attached directories for the original study's code, cached features, and result
files. Report what exists in PROGRESS.md. If original code exists, do not delete
it; move it to `legacy/` untouched.

**Deliverables.**

- `pyproject.toml` or `requirements.txt` with fully pinned versions.
- `src/ser/utils/seeding.py`: one `set_all_seeds(seed)` that sets python `random`,
  `numpy`, `torch`, `torch.cuda`, and enables deterministic cuDNN.
- `src/ser/utils/runmeta.py`: captures git SHA, dirty-tree flag, config hash,
  library versions, hostname, timestamp. Every result row carries this.
- `src/ser/config.py`: dataclass-backed config loaded from YAML. No magic strings.
- `configs/default.yaml`.
- `src/ser/utils/results.py`: append-only JSONL writer with a frozen schema.
- `tests/` with pytest configured and one smoke test that passes.
- `PHASES.md` (copy this file in) and `PROGRESS.md` (empty, with a header).
- `Makefile` or a `ser` CLI entrypoint with stub commands for each later phase.

**Frozen result schema.** Get this right now, because changing it later means
re-running the grid. One row per completed run:

```
run_id, git_sha, config_hash, timestamp, seed,
source_corpus, target_corpus, n_classes, class_names,
backbone, layer_agg, layer_index, feature_branch,
alignment, blending, blend_alpha, n_groups,
classifier, hyperparams_json,
split_id, n_train, n_val, n_target_adapt, n_target_test,
macro_f1, accuracy, uar, per_class_f1_json, confusion_json,
chance_macro_f1, majority_macro_f1, prior_matched_macro_f1,
selection_source_val_macro_f1,
wall_seconds
```

**Acceptance.** `pytest` passes. `make smoke` writes one dummy row to
`results/runs.jsonl` and the schema validates. `PROGRESS.md` lists what legacy
assets were found.

**Do not.** Touch audio, install torch models, write any classifier or alignment code.

---

# PHASE 1 — Reference integrity checker

**Goal.** A script that flags broken and likely-fabricated citations. You verify by
hand; the script only narrows the search.

**Deliverables.**

> **Source located.** There is no `.bib`. The reference list is a
> `thebibliography` environment at `legacy/SER_Report.tex:346-433` with exactly
> **17 `\bibitem` entries**, each in a regular
> `key / authors / ``title'' / \textit{venue}, vol, no, pp, year` layout — fully
> parseable, so the missing `.bib` is not a blocker.
>
> Two notes for the duplicate-title check, from inspecting the source:
> - Match titles **case- and whitespace-insensitively**. `[16] w2vprosody2023`
>   differs from `[6] naderi2023cross` only by `wav2vec2` vs `Wav2Vec2`; an exact
>   match would miss it.
> - Also flag **duplicate venue+volume+issue+page**, independent of title.
>   `[17] li2023cross` and `[7] fu2023cross` both claim Entropy 25(1):124 — the
>   same article coordinates with a different author list, which is a stronger
>   signal than the title match alone.

- `tools/check_refs.py`: reads the `.bib` (or parses the reference list from the
  `.tex`), queries the Crossref REST API by title, and for each entry reports:
  matched DOI, Crossref author surnames, Crossref volume/pages/year, and a flag
  when the bib authors do not overlap the Crossref authors.
- Second check: flag any reference whose title matches another reference's title.
  Duplicate titles with different author lists are the fabrication signature.
- Third check: flag any reference never cited in the body `.tex`.
- `refs_report.md`: table of every reference with status
  `VERIFIED / AUTHOR-MISMATCH / VOLUME-MISMATCH / DUPLICATE-TITLE / UNCITED / NOT-FOUND`.

**Known findings to confirm.** These three are already established and the script
should reproduce them:

- `[9]` DistilHuBERT domain adaptation: bib says Jafari/Shahin/Alavi, vol 187.
  Actual is Naeeni and Nasersharif, Computers in Biology and Medicine 2025, vol 194,
  p. 110510.
- `[16]` duplicates the title of `[6]` (Naderi and Nasersharif, KBS 2023, 277:110814)
  with an invented author list. Delete.
- `[17]` duplicates the title of `[7]` (Fu, Zhuang, Wang, Huang, Duan, Entropy
  2023, 25(1):124) with an invented author list. Delete.

**Acceptance.** `refs_report.md` covers every reference. Every non-VERIFIED entry
has been opened manually on the publisher landing page and corrected in the `.bib`.
Re-running the script yields all VERIFIED.

**Do not.** Auto-edit the `.bib`. The script reports; you fix.

---

# PHASE 2 — Manifest, label map, splits, leakage tests, dataset stats

**Goal.** One canonical description of the data, plus the assertions that make every
later claim about splits mechanically checkable.

**Deliverables.**

`data/manifest.csv`, one row per audio file, columns:
`corpus, file_path, utterance_id, speaker_id, session_id, original_label, duration_s, sample_rate, sha256`.
Build it by walking the raw corpora. Never hardcode counts.

The IEMOCAP label source is `labels.iemocap_label_source` (A7), currently
`majority_vote_discard_disagreement`. Implement that rule explicitly and record
the discarded no-agreement count in `dataset_stats.md`; do not let the annotation
rule be an emergent property of the parsing code.

`src/ser/labels.py`: label mapping as a **pure function**
`map_label(corpus, original_label, label_space) -> str | None`, where `None` means
excluded. Two label spaces:

- `six`: angry, disgust, fear, happy, neutral, sad (RAVDESS and CREMA-D pairs)
- ~~`five`: angry, fear, happy, neutral, sad~~ → **superseded by A4**:
  `four`: angry, happy, neutral, sad (any pair involving IEMOCAP)

**Amendment A3 settles the label decisions** — `excited`→`happy` merge,
`frustrated` dropped (not merged into angry), both IEMOCAP subsets with the
subset recorded per utterance, RAVDESS `calm`→`neutral` merge. They are set in
`configs/default.yaml`; do not re-ask, and do not silently change them (the
config keys feed `label_map_hash`, a `run_id` coordinate — see A5).

The manifest gains a `subset` column for IEMOCAP (`scripted` / `improvised`), so
improvised-only is available later as a free robustness check.

`src/ser/splits.py`: speaker-disjoint splits, deterministic given a seed.

- Source corpus splits into `source_train` and `source_val` by speaker.
- Target corpus splits into `target_adapt` and `target_test` by speaker.
- `target_adapt` is the **only** target data any alignment method may see.
- `target_test` is touched exactly once, at scoring time.
- For IEMOCAP use session-level splits, not speaker-level, since sessions are the
  standard unit.

`tests/test_leakage.py`, all four as hard assertions:

1. No speaker ID appears in more than one split within a corpus.
2. No `utterance_id` appears in both a source split and a target split for any pair.
3. `target_test` indices never appear in any fitted alignment object. Implement this
   by having alignment objects record the index set they were fitted on, and assert
   the intersection with `target_test` is empty.
4. `map_label` is pure: same inputs give same output, has no side effects, and every
   raw label in the manifest either maps to a class in the space or explicitly to `None`.

`tests/test_labelmap.py`: table-driven test covering every raw label string present
in the manifest for all three corpora.

`reports/dataset_stats.md` plus `reports/dataset_stats.csv`:

- Per corpus: speaker count, utterance count, total hours, mean utterance duration.
- Per corpus per class, for both label spaces: utterance count and share.
- Class prior vector for every corpus under every label space.
- Explicit flag on any class with fewer than 100 utterances after mapping.
- **Pairwise corpus-level prior KL and JS for all 9 pairs.** A8 predicts, from
  published counts, KL in 0.0139–0.0336 nats across the six cross-domain pairs.
  Phase 9's entire framing depends on this, so it is verified here, at manifest
  time, not assumed. If the manifest disagrees materially with the table in A8,
  **stop and report** — the reframe may need revisiting.
  This guard is **corpus level against corpus level** (A9): it is a data-integrity
  check against the published counts. The split-level quantity the analysis
  actually uses is a Phase 8 deliverable — do not conflate them here.
- Per-subset counts for IEMOCAP (`scripted` / `improvised`), since the A8
  mechanism-isolating pair depends on them.

**Expect this to surface a problem.** IEMOCAP's fear class is very small. If support
is under about 50 utterances, that class will collapse and it likely explains most of
the sub-chance results on IEMOCAP pairs. Report the number; do not silently drop the
class. The decision about whether to exclude it is a paper-level decision, not a
code-level one.

> **That decision has been made — see A4.** Fear (~40) and disgust (~2) are
> excluded from IEMOCAP pairs via the `four` label space. This phase must still
> **report** their support in `dataset_stats.md`, with the exclusion stated
> explicitly. Verify the ~40 / ~2 figures against the real corpus; they are
> estimates until the manifest exists.

**Acceptance.** `pytest tests/test_leakage.py tests/test_labelmap.py` passes.
`dataset_stats.md` renders and every number traces back to `manifest.csv`.

**Do not.** Extract features, load any model, or touch audio content beyond reading
duration and sample rate.

---

# PHASE 3 — Feature extraction and caching

**Goal.** Extract once, reuse forever. This is the expensive phase and the design
choice that makes layer selection free downstream.

**Key design point.** Cache **all hidden layers**, not just the last one. For a Base
model that is 13 states (CNN output plus 12 transformer layers). Storing all of them
means Phase 5 onward can try any single layer or any weighted sum with zero
re-extraction. This is what fixes the final-layer-pooling problem cheaply.

**Deliverables.**

- `src/ser/features/ssl.py`: for each backbone, mean-pool over time **per layer**,
  producing `(n_utterances, 13, 768)` stored as float16.
- Segment-pooled cache (only if you chose to fix the Transformer): for each
  utterance, split frames into 8 uniform segments and mean-pool each, producing
  `(n_utterances, 13, 8, 768)` float16. Store separately so it can be skipped.
- `src/ser/features/mfcc.py`: 13 base coefficients plus delta and delta-delta,
  mean-pooled and std-pooled, giving 78 dims. Store both; the paper can use mean only,
  but std-pooling is nearly free and often helps.
- Cache keyed by `sha256(manifest_rows) + backbone_name + feature_version`. Never
  overwrite a cache on a key hit.
- `src/ser/features/aggregate.py`: given the layer cache and a spec
  (`last`, `layer:k`, `mean:a-b`, `weighted`), return the pooled matrix. Weighted-sum
  weights are learnable parameters owned by the classifier, not baked into the cache.
- `tools/verify_cache.py`: asserts row count matches the manifest, no NaN or Inf,
  expected shapes, and that the utterance ordering matches the manifest ordering exactly.

**Preprocessing, fixed and documented.** Mono, resample to 16 kHz, peak normalise.
Same for every corpus and backbone. Record the exact torchaudio/transformers versions
in the cache metadata.

> **Implementation notes from the 2026-08-15 build.** Three things a later
> session will otherwise rediscover the hard way:
>
> - **Cache keys are per corpus**, not over the whole manifest. Same intent —
>   each cache is keyed by exactly the rows it covers, including each row's audio
>   sha256 — but adding IEMOCAP later then costs only IEMOCAP's extraction rather
>   than invalidating RAVDESS and CREMA-D. On CPU that is the difference between
>   an afternoon and a day.
>
> - **Batch size is 1 on purpose.** Batching needs padding, and a padded frame
>   reaching the mean corrupts the pooled vector silently, worst for the shortest
>   utterances. Masking would fix it, but `facebook/wav2vec2-base` is documented
>   as degrading under masked batched inference (pretrained without an attention
>   mask; its feature extractor sets `return_attention_mask=False`). Treating one
>   backbone differently would put an unmeasured confound into the backbone
>   comparison. Recover wall time with **process-level parallelism** — one
>   process per backbone, `--threads` split across cores — which changes nothing
>   numerically.
>
> - **OpenMP ordering, Windows/conda.** conda's numpy+MKL and pip's torch each
>   ship `libiomp5md.dll`. If torch's OpenMP initialises first, the first call
>   into librosa's MFCC path aborts the process with `OMP: Error #15`.
>   `ser.features.audio.warm_up_audio_stack()` runs the **full** MFCC path
>   (including `librosa.feature.delta`, which is separately linked through scipy
>   — a warm-up that skips it still aborts) before torch is imported. Import
>   ordering only, no numerical effect. Do **not** reach for
>   `KMP_DUPLICATE_LIB_OK=TRUE`: Intel documents it as able to "silently produce
>   incorrect results", which this project cannot accept.

**Acceptance.** `tools/verify_cache.py` passes for all three backbones. Cache sizes
and extraction wall time recorded in `PROGRESS.md`. Re-running extraction is a no-op.

**Do not.** Train anything. Do not fit alignment. Do not standardise features at
extraction time; standardisation is an experimental condition in Phase 5, not a
preprocessing default.

---

# PHASE 4 — Metrics and trivial baselines

**Goal.** Every table in the paper needs a floor. This phase builds it.

**Deliverables.**

- `src/ser/metrics.py`: macro-F1, accuracy, UAR (unweighted average recall),
  per-class F1, confusion matrix. Include UAR because most prior work reports it and
  the comparison table needs a shared axis.
- Three baselines in `src/ser/baselines.py`, each returning full metrics on
  `target_test`:
  1. `uniform_random`: analytic chance value plus an empirical estimate over 1000
     draws with a CI. For K=6 the analytic macro-F1 is ~0.167; **for K=4 it is
     0.25** (A4 replaced the 5-class space, so the ~0.20 figure is obsolete).
     Floors are therefore **pair-dependent**: every table carries the floor for
     that pair, and nothing averages macro-F1 across pairs of different K.
  2. `majority_class`: always predicts the most frequent source class. Expect
     macro-F1 near 0.05 for K=6. This is the collapse floor.
  3. `prior_matched_random`: samples labels from the source class prior. This is the
     most honest floor, because it is what a model that has learned nothing about the
     input but everything about the source prior would score.
- `src/ser/stats.py`:
  - Bootstrap CI over test utterances (2000 resamples) for any metric.
  - Wilcoxon signed-rank paired test across matched runs, for comparing two
    conditions over the set of (pair, seed) combinations.
  - Holm-Bonferroni correction helper, since you will run several comparisons.
- `tests/test_metrics.py`: verifies macro-F1 against sklearn on synthetic data,
  verifies the collapse case returns the expected analytic value, verifies the
  uniform-random empirical mean lands within CI of the analytic value.

**Acceptance.** Tests pass. Running baselines for all 9 corpus pairs writes rows to
`results/runs.jsonl` with `classifier="baseline_*"`.

**Do not.** Train real classifiers. Do not implement alignment.

---

# PHASE 5 — Alignment and blending module

**Goal.** A clean interface, an honest control condition, and a fully specified MMD.

**Interface.** Every alignment method implements:

```python
class Alignment:
    def fit(self, X_source, X_target_adapt, target_adapt_indices) -> Self
    def transform(self, X) -> np.ndarray
    fitted_on_indices: set   # asserted disjoint from target_test in tests
```

**Five conditions — the ladder of A2**, ordered by moments matched.

1. `none`: identity. Raw cached features.
2. `zscore`: per-corpus standardisation only, no covariance matching. **This is the
   control that decides whether your CORAL gains were ever real.** If z-scoring
   recovers most of CORAL's improvement, that is the paper's most interesting
   negative result.
3. `mean_shift`: `X_src + (μ_tgt − μ_src)`. First moment only, and exactly the
   minimiser of linear-kernel MMD. This is what the original study's `"mmd"`
   column actually was. **Never alias `mmd` to this.**
4. `coral`: whiten source covariance, recolour with target. Regularise with
   `C + eps*I`; the epsilon must be a config value, reported in the paper, and
   sensitivity to it checked at three values.
5. `mmd`: the affine map of A2 — learn `W, b` minimising
   `MMD²_k(W·X_src + b, X_target_adapt) + λ‖W − I‖²_F`, `k` a sum of RBF kernels
   at `{0.25, 0.5, 1, 2, 4} × σ_median`. Write the full spec into a docstring:
   kernel family, bandwidth rule, optimisation steps, learning rate, λ, and what
   is being learned. All are config values (`alignment.mmd_*`).

**Blending.**

- `none`
- `scalar`: single α. **α is selected on `source_val`, never on target test.**
  Search α over {0.0, 0.25, 0.5, 0.75, 1.0}.
- `gaa`: k-means over feature dimensions into g groups, per-group α_g, each selected
  on `source_val`. g is a config value.

Blending only applies when alignment is `mean_shift`, `coral`, or `mmd`. With
`none` or `zscore` the three blending modes are mathematically identical, so the
runner must not enumerate them. The original study generated 216 duplicate runs
this way and reported a grid size of 972 when only 756 were distinct.

Note for the paper: in the original, α was **never searched** — `fwaa` and `gaa`
derived it from `|X_aligned − X_orig|` magnitudes and only `scalar` took an α
argument. So the draft's Table 3 compares three blending modes at three different
unspecified α values. Selecting α on `source_val` is a change in kind, not a
correction to the selection surface; say so.

**Tests.**

- `transform` on `target_test` never causes `fitted_on_indices` to change.
- `fitted_on_indices ∩ target_test == ∅` for every method.
- CORAL with source equal to target is approximately identity.
- Blending with α=1.0 equals pure aligned; α=0.0 equals pure original.
- `zscore` output has per-corpus mean ≈ 0 and std ≈ 1.

**Acceptance.** Tests pass. A tiny end-to-end run on one corpus pair produces
sane numbers for all four alignment conditions.

**Do not.** Run the full grid. Do not tune α against target test data under any
framing.

---

# PHASE 6 — Classifier module with equal-budget search

**Goal.** Remove the asymmetry where the simple baseline runs at library defaults
while the neural model gets a real training loop.

**Rule for this phase.** Every classifier family gets the same hyperparameter search
budget, measured in number of configurations evaluated on `source_val`. Set the budget
in config (suggest 20 configurations per family). No family gets defaults.

**Families.**

- `logreg`: search C (log scale), class_weight (`None`, `balanced`), solver, max_iter.
- `svm`: search C and gamma (log scale), kernel, class_weight balanced.
- `mlp`: search hidden dim, depth, dropout, learning rate, weight decay. **Add early
  stopping on `source_val`.** The original ran a fixed 8 epochs with no early stopping
  and no validation set, so there was no evidence of convergence.
- `transformer` (only if you chose to fix it in Decision A): operates on the
  8-segment cache from Phase 3, so it sees genuine temporal structure. Search depth,
  heads, hidden dim, learning rate.

**Also implement `layer_agg` as a searchable option**, since Phase 3 cached every
layer:

- `last` (reproduces the original, keep it as a comparison condition)
- `layer:k` for k in a small candidate set around the middle layers
- `weighted` (learnable softmax weights over the 13 layers, trained jointly with the
  head)

Middle layers carry the paralinguistic signal in these models; the final layer skews
toward the pretraining objective. Including `last` as a condition lets the paper
quantify how much of the original study's weak numbers came from this one choice,
which turns a mistake into a reported finding.

**Selection always happens on `source_val`. Never on target.** This holds for the
Phase 7 screening pass too (A6): pruning an axis on target-test numbers would
reinvent the exact leak Phase 2 exists to prevent.

**Acceptance.** For one corpus pair and one backbone, all families run end to end,
each consumes exactly the configured budget, and the selected hyperparameters are
written into the result row's `hyperparams_json`.

**Do not.** Run the full grid. Do not add classifier families beyond this list.

---

# PHASE 7 — Grid runner

**Goal.** Execute the full experiment, resumably, with seeds.

**Deliverables.**

> **A6 applies here.** The full factorial is dead. This phase runs a screening
> pass on one pair and one backbone to prune axes, then a reduced factorial with
> seeds over what survives. Pruning is scored on `source_val`, **never** on
> target test. The acceptance criterion below therefore reads "row count equals
> the enumerated *reduced* configuration count times seeds"; record the pruning
> decisions and the resulting enumeration in PROGRESS.md.

- `src/ser/run_grid.py`: enumerates distinct configurations, skipping the
  blending duplicates identified in Phase 5. When
  `grid.include_iemocap_subset_pair` is set, IEMOCAP-improvised and
  IEMOCAP-scripted enumerate as an additional corpus pair in both directions
  (A8) — speaker-disjoint within IEMOCAP as usual.
- 5 seeds minimum per configuration. Seeds affect split assignment within the
  speaker-disjoint constraint, classifier init, and hyperparameter search order.
- Resumable: on start, read `results/runs.jsonl`, build the set of completed
  `run_id`s, skip them. Killing and restarting the job must lose at most one run.
- Append-only writes with a file lock. Never rewrite the results file.
- Progress logging with ETA, and a periodic checkpoint of how many runs remain.
- Failure handling: a crashed run writes a row with `status="failed"` and the
  traceback, then the runner continues. A silent skip is worse than a recorded failure.

**Sanity gates before the long job.** Run a `--smoke` mode over one corpus pair, one
backbone, one seed, all alignment and classifier conditions. Inspect the numbers.
If cross-domain macro-F1 is still below the chance value from Phase 4, stop and
diagnose rather than burning compute on the full grid.

**Acceptance.** Full grid completes. Row count equals the enumerated configuration
count times seeds, plus baselines. Zero rows with `status="failed"`. A second
invocation of the runner does nothing.

**Do not.** Analyse results. Do not build tables. That is Phase 8.

---

# PHASE 8 — Selection protocol and headline tables

**Goal.** This is where the paper's central methodological fix lands.

**Two reporting protocols, both reported.**

1. **Validated** (the honest number). For each (source, target, backbone), select
   the configuration with the best `source_val` macro-F1, then report that
   configuration's `target_test` macro-F1. This is what a practitioner could actually
   achieve without target labels.
2. **Oracle** (the upper bound). Max over the whole grid on `target_test`. Label it
   explicitly as an oracle. This is what the original Table 1 reported without saying so.

**The gap between them is a result.** Report it as its own table and discuss it. A
large gap means the field's habit of reporting grid maxima on target test data
systematically overstates cross-corpus transfer. That is a defensible contribution
and it costs you nothing extra to make.

**Deliverables.**

- `src/ser/analysis/select.py` implementing both protocols.
- Regenerated versions of every table from the original paper, under both protocols,
  each with 95% CIs and each carrying the chance and prior-matched floors as columns:
  - Top configurations (replaces original Table 1)
  - ~~Mean cross-domain macro-F1 by backbone and alignment~~ → **killed by A4.**
    Pairs no longer share a K, so a cross-pair mean is ill-defined, not merely
    uninformative. Replace with **per-pair macro-F1 by backbone and alignment,
    each against its own chance floor** (replaces Table 2).
  - Blending effects in aligned settings (replaces Table 3)
  - ~~Mean macro-F1 by classifier~~ → per-pair, same reason (replaces Table 4)
  - Backbone-specific in-domain vs cross-domain gap (replaces Table 5)
  - New: **split-level prior KL per pair per seed** (A9) — computed from the
    realised `source_train` and `target_test` partitions, reported as mean and
    range across the five seeds, next to the corpus-level figure from Phase 2.
    Expect it to be larger than corpus level and to vary by seed; that variance
    is the point, not noise to be averaged away.
  - New: validated vs oracle gap per pair
  - New: `last` vs `weighted` layer aggregation, quantifying the layer-pooling fix
  - New: **the alignment ladder** — `none` / `zscore` / `mean_shift` / `coral` /
    `mmd` in moment order (A2). Quantifies how much of "alignment" is just
    standardisation, and what the original's mislabelled `"mmd"` column was worth.
    A flat ladder is itself the label-shift evidence.
- Paired significance tests with Holm-Bonferroni for the headline comparisons.
- `reports/results.md` with all tables rendered.

**Acceptance.** Every table cell traces to `results/runs.jsonl` via a script. No
number is typed by hand anywhere. Every table with a comparison also carries the
chance floor.

**Do not.** Write paper prose. Do not make figures.

---

# PHASE 9 — Shift decomposition: label, covariate, conditional

> **Rewritten by A8.** The original brief hypothesised that transfer asymmetry is
> driven by label prior shift. That hypothesis is dead: after the settled label
> decisions, prior KL spans 0.0139–0.0336 nats across all six cross-domain pairs.
> Read A8 before implementing.

**Goal.** Give the paper an explanation instead of a grid dump.

**The claim.** Alignment fails on cross-corpus SER because it minimises *marginal*
discrepancy while *class-conditional* discrepancy is what moves the decision
boundary. The A2 ladder provides the evidence: marginal discrepancy shrinks
monotonically along it while transfer macro-F1 does not.

**Deliverables.**

- For all 9 pairs, decompose shift three ways:
  1. **Label shift.** `KL(prior_source || prior_target)` and symmetric
     Jensen-Shannon distance. Report **both** the corpus-level figure from
     Phase 2 and the split-level figure per pair per seed from Phase 8 (A9) —
     the split-level one is the quantity the analysis rests on, and asserting
     "near zero" at corpus level while testing at split level is the mismatch
     A9 exists to prevent. Expected near-zero at both. Report it as an
     *eliminated* explanation, with the numbers, and state the prediction it
     licenses: prior-correction methods cannot help here.
  2. **Covariate shift.** MMD between marginal source and target features, plus
     proxy A-distance from a domain discriminator. Measure at every rung of the
     ladder, before and after alignment.
  3. **Conditional shift.** Class-conditional MMD,
     `MMD(X_src | y=k, X_tgt | y=k)` for each class k, before and after
     alignment. This is the quantity the claim is about.
     **It reads target test labels, so A10's firewall is mandatory** — analysis
     layer only, never written into a pipeline-readable artifact, never an input
     to selection or to A6 axis pruning, covered by an explicit assertion, and
     reported as undefined below `shift.conditional_mmd_min_support` with
     per-class n always shown.
- **The joint plot that carries the paper:** marginal discrepancy on one axis,
  conditional discrepancy and validated macro-F1 on the other, across the ladder.
  If marginal falls while conditional and macro-F1 stay flat, the claim is
  demonstrated directly rather than inferred.
- **The mechanism-isolating pair (A8).** IEMOCAP-improvised → IEMOCAP-scripted
  and the reverse. Same speakers, same label space, near-identical priors: label
  and covariate shift structurally near zero, only elicitation style differs.
  Degradation here is conditional shift with the confounds held fixed. Enabled by
  `grid.include_iemocap_subset_pair`.
- Spearman correlation of each of the three measures against validated transfer
  macro-F1 across the 6 cross-domain pairs, with CIs. **Report the label-shift
  correlation as undefined-by-construction rather than as a weak result** — six
  points spanning 0.02 nats does not support a correlation coefficient, and
  presenting one would be the same overclaiming this rebuild exists to remove.
- **Prior-correction baseline, kept as a falsifiable prediction.** Implement EM
  prior estimation (Saerens-Latinne-Decaestecker): estimate target priors from
  the source-trained classifier's outputs on `target_adapt`, using no target
  labels, then reweight posteriors on `target_test`. BBSE is an acceptable
  alternative. A8 predicts it is inert here. If it *does* help despite near-zero
  prior KL, that falsifies the decomposition and must be investigated, not
  quietly reported.
- Comparison table: `none`, `zscore`, `mean_shift`, `coral`, `mmd`, `em_prior`,
  `coral+em_prior`, `mmd+em_prior`. Paired tests across pairs and seeds.
- `reports/shift_decomposition.md`.

**Read the outcome honestly.** The claim is designed to hold whichever way the
numbers land — that is why it is worth making. Do not tune until something wins.

**Acceptance.** All three decompositions and every comparison reproducible from a
single script. The EM implementation has a unit test on synthetic data with a
known prior shift. The conditional-MMD implementation has a unit test on synthetic
data with known conditional shift and zero marginal shift.

**Do not.** Selectively report. Every measure computed goes into the report,
including the ones that do not support the claim. Do not resurrect the label-shift
thesis by changing a label decision to restore skew.

---

# PHASE 10 — Per-class analysis and figures

**Goal.** Publication-quality figures and the per-class breakdown that costs nothing
and adds real value.

**Deliverables.**

- Per-class F1 across all cross-domain pairs, as a table and a heatmap. Expect anger
  and sadness to transfer and happiness, disgust and fear not to. That pattern is
  a genuine finding sitting in data you already have.
- Class-collapse diagnostic: for every run, the number of classes with zero predicted
  instances. Report the share of the grid that collapses. This explains the sub-chance
  aggregates directly.
- Rebuilt confusion matrices replacing the original Figure 2:
  - Class names on both axes, not integer indices.
  - Row-normalised, shared colour scale across all panels.
  - Vector output (PDF), legible at single-column width.
  - Panel captions naming the exact configuration and protocol.
- Scatter: prior-shift KL on x, validated transfer macro-F1 on y, one point per pair,
  labelled, with the fitted trend and the Spearman value.
- Bar chart with CIs comparing alignment conditions, with the chance line drawn in.
  Drawing the chance line is not optional; it is the single most informative mark on
  the figure.
- All figures generated by scripts in `src/ser/analysis/figures.py`, saved to
  `figures/`, regenerable with one command.

**Acceptance.** `make figures` regenerates every figure from results with no manual
steps. Every figure is legible when printed at journal column width.

**Do not.** Hand-edit any figure in an image editor. If it needs fixing, fix the script.

---

# PHASE 11 — Release packaging and LaTeX table generation

**Goal.** Make the artefact reviewable and make paper tables impossible to drift from
results.

**Deliverables.**

- `README.md`: exact commands to reproduce every table and figure from raw corpora,
  including expected wall time and disk requirements.
- Environment lock file. Verify a clean-machine install.
- `src/ser/analysis/latex.py`: emits every paper table as a `.tex` file directly from
  `results/runs.jsonl`. The paper `\input{}`s these. No number is ever typed into the
  manuscript by hand.
- Consolidate the Phase 2 and Phase 5 leakage assertions into a single
  `make verify` target. This is your "executable checks, not a form" requirement,
  and it goes in the paper as a reproducibility statement with the command name in it.
- Ready-to-paste manuscript sections:
  - Data availability statement
  - Code availability statement, with the archive DOI
  - Author contributions
  - Funding statement
  - Conflict of interest statement
  - Brief ethics note (affective inference on human subjects, public corpora, no new
    data collection)
- Archive the repo to Zenodo, get a DOI, put it in the paper.
- Anonymised variant of the repo for double-blind submission, if the target venue
  requires it.

**Acceptance.** A clean clone plus `make verify && make tables && make figures`
produces byte-identical tables and figures. Someone who is not you can follow the
README start to finish.

**Do not.** Write the manuscript prose. That is your job, not Claude Code's.

---

## After Phase 11

The rewrite is a human task. The structural changes the manuscript needs:

- New Section 2 Related Work, moved out of the Results section.
- Rebuilt comparison-with-prior-work table: replace the "reported performance" column
  with a protocol column (what is controlled, whether target labels are used, how
  selection is done). Comparing your macro-F1 against other papers' accuracy on
  different corpora makes you look far worse than you are and proves nothing.
- Abstract carrying actual numbers, including the validated-versus-oracle gap.
- Title sharpened to state the claim rather than list the components.
- Limitations section.
- Corresponding author email and ORCIDs.

Then run your venue-benchmark rule: pull Pastor et al. 2023 from Applied Sciences,
run the same surface test on it that you would run on your own paper, and fix
whatever the comparison exposes before you submit.
