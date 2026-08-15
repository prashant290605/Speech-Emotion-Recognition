# PROGRESS

Running log of the cross-corpus SER reproducibility rebuild. One dated entry per
phase. Each entry lists files created, files modified, tests added, decisions
made, and anything deferred.

Phase briefs live in [PHASES.md](PHASES.md). Sessions do not share memory: this
file plus PHASES.md is the entire handover between them.

---

## 2026-08-12 — Session 2: corpus acquisition, manifest, prior verification

Status: **complete**. `pytest` → 190 passed. IEMOCAP untouched; no features extracted.

### Acquired

| corpus | source | size | files | speakers |
|---|---|---|---|---|
| RAVDESS speech | Zenodo record 1188976, `Audio_Speech_Actors_01-24.zip` | 208,468,073 B | 1440 wav | 24 actors |
| CREMA-D | GitHub `CheyneyComputerScience/CREMA-D`, `AudioWAV/` | 578 MB | 7442 wav | 91 actors |

RAVDESS zip `sha256 = 5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657`,
`testzip()` clean, 1440 wav entries.

CREMA-D is a **Git LFS** repository — the tree is 15 MB of pointers, so a plain
clone yields no audio. Acquired with `GIT_LFS_SKIP_SMUDGE=1`, a sparse checkout
of `AudioWAV`, then `git lfs pull --include="AudioWAV/**"`. Verified zero files
under 1 KB, i.e. no unresolved pointers left masquerading as audio. The temporary
clone was deleted after the audio was moved (its `.git/lfs` held a second 578 MB
copy).

Both counts match the published sizes **exactly** — 1440/24 and 7442/91 — so the
count guard passed without needing its tolerance.

### Manifest

`data/manifest.csv`, 8882 rows, 12 columns (the Phase 2 set plus `subset` for
IEMOCAP later and one mapped-label column per space):

```
corpus, file_path, utterance_id, speaker_id, session_id, subset,
original_label, label_six, label_four, duration_s, sample_rate, sha256
```

RAVDESS 1.48 h, CREMA-D 5.26 h. Duration and sample rate come from the file
header; audio content is read only for the integrity hash.

Two design points worth keeping:

- **One mapped-label column per label space**, not a single "mapped label".
  Which space applies depends on the *pair*, so a single column would be
  ambiguous the moment IEMOCAP arrives.
- **`CORPUS_EXPECTATIONS` is a module constant, not config.** A verification
  threshold a user can edit is not a verification. A partial download halts the
  build rather than silently shrinking the experiment.
- RAVDESS **song** (`channel == 02`) is explicitly skipped. It is a separate
  Zenodo download, and letting it in would inflate counts past the guard.

### A8 verified against real data

The near-zero prior shift that reframed all of Phase 9 was computed from
*published counts*. Recomputed from the manifest:

| source → target | K | KL (nats) | JS | A8 predicted | agrees |
|---|---|---|---|---|---|
| ravdess → cremad | 6 | 0.0252 | 0.0769 | 0.0252 | yes |
| cremad → ravdess | 6 | 0.0224 | 0.0769 | 0.0224 | yes |

Agreement to four decimal places. Per-class counts reproduce the published
figures exactly: RAVDESS 192 each for angry/disgust/fear/happy/sad and 288
neutral (96 + 192 calm), 1248 in the 6-class space with 192 surprised excluded;
CREMA-D 1271 each and 1087 neutral, all 7442 retained.

**The "RAVDESS is exactly balanced" correction is now demonstrated, not asserted:**
priors are 0.231 neutral against 0.154 for every other class. There is a test
named for it.

`ser dataset-stats` exits non-zero and prints HALT if any pair disagrees with A8
by more than 0.002 nats. Per A9 this is the **corpus-level** integrity check;
split-level KL per seed remains a Phase 8 quantity.

### Files created / modified

```
src/ser/labels.py          pure map_label + LabelPolicy (no default policy)
src/ser/manifest.py        parsers, build, count guard, CSV round trip
src/ser/datastats.py       counts, priors, KL/JS, the A8 guard, report
src/ser/cli.py             `ser manifest`, `ser dataset-stats`
tests/test_labelmap.py     table-driven over every raw label x space x corpus
tests/test_manifest.py     building, guards, divergence maths, real-data checks
data/manifest.csv          8882 rows (gitignored: derived, and paths are local)
reports/dataset_stats.{md,csv}
```

`map_label` takes an explicit `LabelPolicy` rather than reading config itself —
that is what makes the purity assertion meaningful and stops any run from
quietly adopting a default. An unrecognised raw label **raises** instead of
returning `None`, so `None` can only ever mean "deliberately excluded".

### Deferred

- IEMOCAP: untouched, as instructed. Its manifest parser is not written; the
  `subset` and `session_id` columns exist and are empty.
- No features extracted (Phase 3).
- `data/manifest.csv` is gitignored: it is derived, and `file_path` holds
  machine-local absolute paths. Regenerate with `ser manifest`.

---

## 2026-08-12 — Phase 1: reference integrity checker

Status: **script complete; manual resolution outstanding.** `pytest` → 107 passed.

The deliverables are done. The *acceptance criterion* is not met and cannot be
met by code: it requires every non-clean entry to be opened on the publisher
landing page by a human and corrected in the `.tex`. Five entries are waiting.
`ser check-refs` exits non-zero until they are clean, so it can gate a release
check later.

### Result

17 references parsed from `legacy/SER_Report.tex:346-433`; 15 distinct keys
cited in the body.

| Tier | Count | Entries |
|---|---|---|
| C. probable fabrication | 3 | [9] `jafari2025feature`, [16] `w2vprosody2023`, [17] `li2023cross` |
| B. needs manual resolution | 2 | [11] `baevski2020wav2vec`, [15] `gretton2012kernel` |
| A. confirmed correct | 12 | — |

**All three known findings reproduced**, and the diagnosis is sharper than the
brief's:

- **[9]** — title, venue, page (110510) and year all match the Crossref record
  exactly. Only the authors and the volume are wrong: entry claims Jafari,
  Shahin, Alavi, vol. 187; the record is Naeeni and Nasersharif, vol. 194.
  DOI `10.1016/j.compbiomed.2025.110510`. Correct in place, do not delete.
- **[16]** — duplicates [6] `naderi2023cross` case-insensitively, disjoint
  authors, uncited. Delete.
- **[17]** — duplicates [7] `fu2023cross` *and* claims the identical article
  slot (Entropy 25(1):124), disjoint authors, uncited. Delete.

The two uncited entries are exactly [16] and [17] — the citation check
independently corroborates the duplicate check, with no shared inputs.

### Two defects found by inspecting the first run's output

Both were caught because the first run produced results that contradicted known
facts, and both would have wasted the manual pass or, worse, corrupted it.

**1. Weak Crossref matches produced false fabrication signals.** The first run
flagged [11] `baevski2020wav2vec` and [15] `gretton2012kernel` as
AUTHOR-MISMATCH + VOLUME-MISMATCH. Both are real, heavily-cited papers. Crossref
had matched them to *different* papers at 0.76 and 0.83 title similarity
("ccc-wav2vec 2.0…", "A composite kernel two-sample test"), and the checker then
compared authors and volumes against the wrong record.

Fixed: metadata is compared **only** above a 0.90 title similarity. Below that
the entry gets `NOT-IN-CROSSREF` and the report states explicitly that absence
from Crossref is not evidence of fabrication — NeurIPS, JMLR, Interspeech and
arXiv-only work are routinely unindexed. The landing link for such an entry is a
Crossref *search*, never the wrong paper's DOI.

**2. Duplicate groups condemned the genuine entry alongside the fake.** [6] and
[7] were placed in tier C purely for being duplicated. [6] resolves to
`10.1016/j.knosys.2023.110814` with matching authors, volume and pages — it is
the real citation, and [16] is the impostor.

Fixed: Crossref resolution now runs for every entry *before* the duplicate check,
so the duplicate check can ask which member is independently corroborated. The
corroborated member gets an informational `DUPLICATED-BY-OTHER` and stays tier A
with a note naming the entry to delete. If no member is corroborated, both go to
manual. Both fixes are covered by named regression tests.

### Files created / modified

```
src/ser/refs.py            parser, Crossref client, four checks, report renderer
tools/check_refs.py        thin CLI wrapper (the Phase 1 deliverable path)
src/ser/cli.py             `ser check-refs` implemented; removed from PENDING
tests/test_check_refs.py   17 tests, no network — Crossref is stubbed
reports/refs_report.md     the report
.gitignore                 .cache/ (Crossref responses; regenerable)
README.md                  Phase 1 status
```

`reports/refs_report.md` opens with an **Actions** table: one row per outstanding
entry, one sentence of instruction, one link. Crossref responses are cached, so
`ser check-refs --offline` re-renders with no network.

### Deferred

- **The manual pass is yours.** Five landing pages. The script does not and will
  not edit the bibliography.
- **Two `.tex` files carry bibliographies.** Only `SER_Report.tex` was audited;
  `cross_corpus_ser_paper.tex` is the older draft and has no
  `thebibliography` block. If it is revived, audit it too.
- The cache is not committed, so a fresh clone needs one online run to reproduce
  the report.

---

## 2026-08-10 — Phase 0 addendum 3: two corrections and two firewalls

Status: **complete**. `pytest` → 90 passed. Amendments A9–A10 added.

### Corrections to earlier entries in this file

**1. The KL figures in addendum 2 are corpus-level, and the analysis needs
split-level (A9).** The 0.0139–0.0336 range is computed from whole-corpus
priors. The quantity that governs a run is the divergence between the realised
`source_train` prior and the realised `target_test` prior *after* speaker-disjoint
splitting. IEMOCAP has five sessions and ten speakers with non-uniform
per-session emotion distributions, so a session-disjoint fold moves class
proportions by several points, differently per seed. Split-level KL will be
larger than corpus-level and will vary across the five seeds.

This does not rescue the label-shift thesis and must not be used to. But
asserting "near zero" at corpus level while testing at split level is a mismatch
a reviewer will find. Phase 2's halt-and-report guard stays corpus-level against
corpus-level (a data-integrity check against published counts); Phase 8 gains
split-level KL per pair per seed, as mean and range over realised partitions.

**2. "No `.bib` exists" (Phase 0 entry, Deferred) overstated the Phase 1
blocker.** There is no `.bib`, but `legacy/SER_Report.tex:346-433` contains a
`thebibliography` environment with exactly 17 `\bibitem` entries in a regular
parseable layout. Phase 1 is not blocked on producing a `.bib` first.

All three known findings are confirmed present in that source:

| ref | key | claimed | problem |
|---|---|---|---|
| [9] | `jafari2025feature` | Jafari/Shahin/Alavi, Comput. Biol. Med. **187**, 110510, 2025 | real paper is Naeeni & Nasersharif, vol **194** — page matches, volume and authors do not |
| [16] | `w2vprosody2023` | Ploszaj/Tarnowski/Jedrzejczak, KBS **275**:110676 | title duplicates [6] `naderi2023cross` (KBS 277:110814); differs only by `wav2vec2` vs `Wav2Vec2` |
| [17] | `li2023cross` | Y. Li/J. Wu/X. Liu/H. Meng, Entropy 25(1):124 | **same venue, volume, issue and page** as [7] `fu2023cross` — identical article coordinates, invented author list |

Two implications written into Phase 1: title matching must be case- and
whitespace-insensitive (or [16] is missed), and duplicate venue+volume+issue+page
should be flagged independently of title (it is what makes [17] unambiguous).

**3. A merge of `rebuild` into `main` is not a fast-forward.** Stated incorrectly
in the previous session summary. `main` carries `2215976` (the README withdrawal),
which is not in `rebuild`'s ancestry, so the branches have diverged. The
`README.md` conflict is unavoidable whenever the merge happens; resolve in favour
of `rebuild`'s version and delete `main`'s variant at that moment. **Do not rename
or delete the `rebuild` branch while `main`'s README is public** — its links to
the revision notice are relative and point there.

### A10 — firewalling the conditional-shift diagnostic

`MMD(X_src | y=k, X_tgt | y=k)` requires target test labels by construction.
Legitimate post hoc, illegitimate anywhere near fitting or selection — it is the
Phase 2 leak reintroduced under a respectable name. Containment written into
PHASES.md: analysis layer only; never written into a pipeline-readable artifact;
never an input to selection, including A6 axis pruning; covered by an explicit
assertion rather than a comment; reported as undefined below
`shift.conditional_mmd_min_support` (50), with per-class n always shown.

Worth recording: **the frozen result schema is itself part of this firewall.** It
has no field for a conditional-shift quantity, and adding one requires a
`SCHEMA_VERSION` bump that invalidates every existing row. That mechanical guard
should not be weakened by adding a general-purpose "diagnostics" column.

### Files modified

```
configs/default.yaml     new `shift` section: conditional_mmd_min_support: 50
src/ser/config.py        ShiftConfig
PHASES.md                A9, A10; Phase 1 source located + two matching notes;
                         Phase 2 guard scoped to corpus level; Phase 8 gains
                         split-level KL; Phase 9 label-shift and conditional-shift
                         bullets updated
tests/test_config.py     +1 case, +1 assertion (90 total)
```

### Not blocked on IEMOCAP

Calling IEMOCAP "the whole critical path" was too pessimistic. RAVDESS and
CREMA-D are immediate downloads and cover two of the six cross-domain pairs, the
full 6-class label space, and **every piece of machinery**: splits, leakage
assertions, 13-layer caching, the four-rung ladder, the α axis, the smoke gate.
Only the four IEMOCAP pairs and the A8 subset probe genuinely block.

A two-corpus config is a two-line edit and is verified to load:
`grid.corpora: [ravdess, cremad]` with `grid.include_iemocap_subset_pair: false`
(the validator requires the latter, since the probe needs IEMOCAP).

Phase 2 should start on `{ravdess, cremad}` without waiting. If the smoke gate
fires on that pair, better to learn it now than a month later with IEMOCAP
finally in hand.

---

## 2026-08-10 — Phase 0 addendum 2: the label-shift thesis is dead

Status: **complete**. `pytest` → 89 passed. Recorded as PHASES.md amendments
A7–A8; Phase 9 rewritten.

### The finding

The published IEMOCAP counts are not estimates — they are the canonical
majority-agreement figures: neutral 1708, frustrated 1849, angry 1103, sad 1084,
excited 1041, happy 595, fear 40, disgust 2, plus 2507 no-agreement. Applying the
settled A3/A4 decisions and computing priors:

| Corpus (4-class) | angry | happy | neutral | sad | n |
|---|---|---|---|---|---|
| IEMOCAP | .199 | .296 | .309 | .196 | 5531 |
| RAVDESS | .222 | .222 | .333 | .222 | 864 |
| CREMA-D | .259 | .259 | .222 | .259 | 4900 |

Pairwise KL, all six cross-domain pairs:

| pair | K | KL | JS |
|---|---|---|---|
| iemocap → ravdess | 4 | 0.0148 | 0.0598 |
| ravdess → iemocap | 4 | 0.0139 | 0.0598 |
| iemocap → cremad | 4 | 0.0336 | 0.0914 |
| cremad → iemocap | 4 | 0.0335 | 0.0914 |
| ravdess → cremad | 6 | 0.0252 | 0.0769 |
| cremad → ravdess | 6 | 0.0224 | 0.0769 |

**Prior shift is near-zero everywhere.** The original premise — IEMOCAP heavily
skewed, RAVDESS balanced — held of *raw* IEMOCAP. Dropping frustration (A3) and
cutting fear (A4) removed almost all of the skew. The decisions that make the
label space defensible are the same decisions that dissolve the effect Phase 9
was built to detect. A Spearman correlation over six points spanning 0.02 nats is
not underpowered; it is undefined.

This is not fixable by restoring frustration: merging it into anger would
manufacture the skew and then discover it.

### The reframe (A8)

Phase 9 becomes a three-way decomposition rather than a single-hypothesis test:
label shift (KL/JS — now an *eliminated* explanation, reported as such, with the
prediction that prior-correction methods are inert), covariate shift (marginal
MMD, plus proxy A-distance, measured at every rung of the A2 ladder), and
conditional shift (class-conditional MMD before and after alignment).

The claim: alignment fails because it minimises marginal discrepancy while
class-conditional discrepancy moves the decision boundary — evidenced by the
ladder showing marginal discrepancy shrinking monotonically while transfer
macro-F1 does not. Holds whichever way the numbers land, needs no new data.

EM/BBSE prior correction is retained, reframed as a **falsifiable prediction**:
A8 says it must be inert. If it helps despite near-zero prior KL, the
decomposition is wrong and that must be investigated, not quietly reported.

### The mechanism-isolating experiment

IEMOCAP-improvised ↔ IEMOCAP-scripted as two corpora. Same speakers, same label
space, near-identical priors, so label and covariate shift are structurally near
zero and only elicitation style differs. Degradation there is conditional shift
with the confounds held fixed — something none of the cited comparisons achieve.
Costs one extra pair; `grid.include_iemocap_subset_pair: true`, with a
cross-section validator requiring `labels.iemocap_record_subset`.

**This is the strongest argument for waiting on the IEMOCAP licence rather than
substituting EmoDB/SAVEE.** The experiment does not exist without the corpus, and
both substitutes are acted, which would cost the spontaneous-vs-acted axis
entirely.

### The annotation rule is now explicit (A7)

`labels.iemocap_label_source: majority_vote_discard_disagreement`, inside
`label_map_hash`. The counts above depend entirely on it; any-annotator or
self-assessment labels change every count, every prior, and the whole analysis.
It was previously an implicit property of the parsing code — a silent axis that
could change without moving the hash, the exact failure mode the hash exists to
prevent.

### Files modified

```
configs/default.yaml     iemocap_label_source, grid.include_iemocap_subset_pair
src/ser/config.py        LABEL_SOURCES enum, subset-pair validators; corpus
                         validation moved into GridConfig so its error precedes
                         the subset-pair check
PHASES.md                A7, A8; Phase 9 rewritten; Phase 2 gains pairwise-KL
                         verification and per-subset counts; Phase 7 enumerates
                         the subset pair; phase map renamed
README.md                Phase 9 renamed
legacy/README.md         superseded banner + inline warning above the results
                         table (annotated, not edited: the original text is
                         intact below the banner)
tests/test_config.py     +4 tests (89 total)
```

### Deferred / still open

- The priors above are computed from published counts. **Phase 2 must re-verify
  from the manifest and stop if it disagrees materially** — A8's framing depends
  on it. The verification is written into the Phase 2 deliverables.
- IEMOCAP licence: it is an institutional agreement with SAIL requiring a
  signatory with authority to bind the recipient organisation, so it needs the
  faculty co-author. Licence and co-author are one item, not two. Open question
  for the group: did the original semester's work run under a signed licence? If
  one is already held, reuse it. If the data came informally, regularise before
  publishing — clause 5 requires citation and clause 6 asks licensees to discuss
  planned evaluations with SAIL prior to public reporting.
- Contingency if the licence stalls past four weeks: EmoDB and SAVEE as
  substitute corpora. Both acted, so the spontaneous-vs-acted axis and the A8
  probe are lost. Fallback, not plan.

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
