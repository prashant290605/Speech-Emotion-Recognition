# PROGRESS

Running log of the cross-corpus SER reproducibility rebuild. One dated entry per
phase. Each entry lists files created, files modified, tests added, decisions
made, and anything deferred.

Phase briefs live in [PHASES.md](PHASES.md). Sessions do not share memory: this
file plus PHASES.md is the entire handover between them.

---

## FINDING (b) — the discrepancy–transfer relationship has no sign until the measurement geometry is fixed

**Promote to the paper. This is the methodological contribution, and it is the
one no reviewer can attribute to our implementation, because it is a statement
about the measurement rather than about any method we wrote.**

The same 13 layers, the same target scores, two ways of measuring marginal
discrepancy. Spearman ρ against target macro-F1:

| backbone | ρ(effect, target) — rung's own geometry | ρ(effect, target) — fixed ZCA reference |
|---|---|---|
| hubert | **−0.769** | **+0.692** |
| wav2vec2 | +0.445 | +0.341 |
| wavlm | **−0.654** | **+0.670** |

**The two frames disagree in sign on two of three backbones.** Measured in each
representation's own geometry, less discrepancy goes with better transfer —
the result the domain-adaptation literature assumes. Measured in a fixed
reference frame derived once from unaligned source_train, *more* discrepancy
does.

Neither frame is picked out by theory:

* The **per-rung geometry** uses a median-heuristic bandwidth computed on the
  features being compared, so it adapts to whatever scale each representation
  happens to be in. That adaptation is what makes rungs comparable at all, and
  it is also what makes the statistic partly a measure of the frame rather than
  of the overlap.
* The **fixed reference geometry** removes that degree of freedom, but at the
  cost of privileging one arbitrary frame — and it is far more sensitive to
  uncorrected mean offsets, because ZCA-whitening by the source covariance
  leaves a translation fully visible.

The consequence for the field is direct: **reporting "our method reduces MMD by
X%" is not a well-posed claim** unless the geometry the MMD was measured in is
stated, and essentially nobody states it. Two honest researchers measuring the
same alignment can report opposite relationships between discrepancy and
transfer.

What the paper must do: **report both frames side by side, always, and never
one alone.** Every table carrying an effect size carries both columns. Where
they disagree, that disagreement is the finding, not a defect to be resolved by
picking the more convenient one.

This also reframes the project's own headline claim. It is not "alignment does
not help" — that is a weak null that a reviewer can attribute to our CORAL or
our MK-MMD. It is: **the standard practice of reporting marginal discrepancy
reduction as evidence of successful adaptation is not well-posed**, and our
ladder is the demonstration.

Caveat: measured on the layer sweep (2 seeds, one pair, logreg, rung `none`).
Stage 2 supplies seeds and backbone coverage. Recorded now because it changes
what Stage 2's tables must contain, not because it is confirmed.

---

## FINDING (a) — in-domain-optimal depth is 4–5 layers shallower than transfer-optimal depth

**Promote to the paper.** Supersedes the earlier two-point `layer:6`-vs-`last`
finding, which the full sweep partly refutes — see "What changed" below.

The practical consequence, stated plainly: **anyone selecting an SSL layer on
in-domain validation — which is what the SUPERB-style probing literature
does — systematically picks the wrong depth for cross-corpus transfer.**

Full 13-layer sweep, 3 backbones × 13 layers × 2 seeds, logreg, rung `none`,
ravdess→cremad. Tables in [reports/layer_sweep.md](reports/layer_sweep.md).

| backbone | argmax `source_val` | argmax target | gap | cost of choosing depth on `source_val` |
|---|---|---|---|---|
| hubert | 6 | 11 | 5 | **−0.1436** |
| wav2vec2 | 5 | 9 | 4 | **−0.1405** |
| wavlm | 6 | 10 | 4 | **−0.1282** |

The layer that is best in-domain is consistently 4–5 layers shallower than the
layer that transfers best, in every backbone, and selecting depth the way the
literature does — on in-domain validation — costs 0.13–0.14 macro-F1 across
corpora.

Why this is the strongest thing in the project:

* **It does not involve the alignment ladder at all.** No CORAL, no MK-MMD, no
  warm start, no optimiser. Every methodological question about the MMD
  implementation — saturation, geometry, the fallback, the step size — is
  irrelevant to it. It is 13 cached representations and one classifier.
* **It is a curve across three backbones, not a two-point contrast**, and the
  gap has the same sign and nearly the same magnitude in all three.
* **The SUPERB-style layer-probing literature does not report it**, because that
  literature is almost entirely in-domain — which is exactly the regime where
  the shallower layer wins, as our own `source_val` column shows.
* **It has a direct practical consequence**: the learnable `weighted`
  aggregation avoids having to choose, and is the retrospective justification
  for caching all 13 layers in Phase 3.

### The discrepancy explanation does not work, and that is itself a finding

Spearman ρ over the 13 layers:

| backbone | ρ(effect own, target) | ρ(effect reference, target) | ρ(source_val, target) |
|---|---|---|---|
| hubert | −0.769 | **+0.692** | +0.104 |
| wav2vec2 | +0.445 | +0.341 | +0.396 |
| wavlm | −0.654 | **+0.670** | +0.527 |

**The two geometries disagree in sign on two of three backbones.** Measured in
each rung's own frame, less discrepancy goes with better transfer; measured in
the fixed ZCA reference frame, more discrepancy does — same features, same
layers, same target scores. Any claim of the form "lower MMD implies better
transfer" must name its geometry and defend it, and neither frame is picked out
by theory. This caution applies to the alignment ladder too.

### What changed

The earlier version of this finding claimed `layer:6` has **lower** discrepancy
than `last` *and* transfers worse — a dose-response contradiction. Under the
corrected fixed-bandwidth MMD:

* **Survives:** `layer:6` scores higher in-domain in all three backbones.
* **Survives, weakened:** it transfers worse on hubert (−0.122) and wavlm
  (−0.076), but on wav2vec2 the two are tied (+0.005).
* **Refuted:** "lower discrepancy". `layer:6` has *higher* own-geometry
  discrepancy than `last` on hubert (954 vs 894) and wav2vec2 (1658 vs 1540),
  and lower only on wavlm (1088 vs 1171). The 3.76× / 1.93× figures the original
  claim rested on came from the pre-correction per-rung-bandwidth measurement.

The dose-response framing is therefore withdrawn. The depth-divergence curve
replaces it and is the stronger claim.

Caveat carried forward: two seeds, one pair, one classifier, pre-selection.
Stage 2 supplies the seeds; nothing here enters the paper on this evidence
alone.

---

## 2026-08-17 — grid-freeze-v3: four corrections before Stage 2 launches

Four changes required after the Stage 1 analysis was accepted. All four move
`run_id` coordinates, so this is a deliberate break with `grid-freeze-v2`:
Stage 1's 372 rows stay valid as measurements under their own tag, and no
longer resume. Schema v9 adds columns only, so they migrate.

### 1. The CORAL eps grid was mis-centred — extended to 1.0 and 10.0

Stage 1 found `source_val` monotone increasing in eps with **1e-1, the grid
maximum, winning 18/18 conditions**. That says the search stopped too early,
not that 1e-1 is optimal, and "the selected hyperparameter sat at the edge of
its range" is a reviewer critique that costs a revision cycle. Launching 60+
hours against a grid whose optimum lies outside it would mean CORAL is never
evaluated at its best setting.

Surviving grid is now `[1e-4, 1e-2, 1e-1, 1.0, 10.0, Ledoit-Wolf]`. 1e-3 stays
pruned — the original rationale (interior, bracketed, adds no shape) is
unaffected by extending the top end.

### 2. `blend_alpha` — decided: Stage 2 varies it

The axis was unscreened, so the choice was to vary it or drop blending from the
paper. **Varying it**, in a dedicated screening arm: 288 runs, the two rungs
that move features most (`coral`, `mkmmd_full`), one backbone, two seeds,
α ∈ {0, 0.25, 0.5, 0.75}. α=1.0 is not re-enumerated because it is exactly the
main grid's `blending="none"` cell — enumerating it twice is how the original
counted 972 runs when only 756 were distinct.

Kept rather than dropped because the original study **never searched α at all**
— its Table 3 compares three blending modes at three unspecified α values — so
measuring what α does converts a documented criticism into a result.

**Found while implementing it: blending was enumerable but never applied.**
`BLENDABLE_ALIGNMENTS` was imported into `run_grid` and unused; `execute_run`
had no blending step. A blending arm on the old code would have written 288
rows labelled with a `blend_alpha` whose features were never blended, and
nothing would have caught it. Scalar blending is now implemented in the run
path, applied *before* the effect size is measured so the recorded discrepancy
describes what the classifier actually consumed.

Verified end to end on real features, not just unit-tested: α=0.0 reproduces
the unaligned run to floating-point equality (target 0.3700 both), α=1.0
reproduces the pure-aligned run (0.3943 both), α=0.5 differs from both
(0.3181). `gaa` now **raises** rather than silently no-opping — it needs
k-means over feature dimensions and a nested per-group α selection this grid
does not have, and is explicitly out of Stage 2 scope.

### 3. LogReg convergence — the cap was a searched hyperparameter

The real defect was worse than the symptom. `max_iter` was **drawn from
[1000, 2000, 5000] as part of the random search**, so a trial could win
selection by having stopped early rather than by fitting better. A convergence
budget is not a model choice.

* `max_iter` removed from the search space; fixed at `sklearn_max_iter: 20000`.
* Sized from measurement: worst observed case is lbfgs at C=1e4 on 5972
  CREMA-D utterances, **3529 iterations**. At the matched-n size of 988 the
  worst case is 313.
* Convergence is **asserted** — `NotConverged` is raised, the trial loop records
  it as a failed trial scored −inf, so a non-converged fit can never be scored,
  ranked, or selected. It also cannot buy the family extra attempts, which
  keeps the equal-budget contract intact.
* `solver_n_iter` recorded on every row.

**Bug I introduced and caught in the same pass:** the first version compared
against `max_iter` unconditionally, and libsvm's `SVC` uses `max_iter=-1` to
mean "no limit" — so every `svm_rbf` trial was marked non-converged and every
svm_rbf run failed. Caught by the existing budget tests. Guarded by
`test_unlimited_solvers_are_not_reported_as_non_converged`.

### 4. Matched-n reverse arm instead of dropping the reverse direction

The right call, and not only cheaper: **direction was confounded with a 6×
difference in training data.** CREMA-D contributes 5972 source-train utterances
against RAVDESS's 988, so any asymmetry the paper reported could have been
sample size.

`splits.matched_source_train` caps cross-corpus `source_train` to the smaller
direction's size, per seed. The cap is **derived, never hard-coded**: both
directions' natural sizes are computed and the minimum taken, so the smaller
direction is untouched and the larger comes down to meet it.

* Stratified by class using largest-remainder allocation, so the label prior is
  preserved to within one utterance per class. A uniform subsample would shift
  the prior and confound the very comparison the cap exists to enable.
* Within the existing speaker-disjoint split — utterances are **removed, never
  moved**, so every leakage guarantee still holds. Verified: CREMA-D drops
  5972 → 988 across 73 speakers with no utterance or speaker overlap with any
  other role.
* Seeded and deterministic; recorded in `split_spec_hash` and on the row as
  `source_train_n` / `source_train_cap`.
* Reported as **"matched-n"**. Full-n reverse remains available as a separate
  later arm with `matched_source_train: false`.

**Measured effect on cost: the reverse direction went from ~16× to ~1.1×.**
Per-family measured ratios (`results/stage2_calibration.jsonl`): logreg
0.99/1.18, svm_linear 0.57/1.45, svm_rbf 0.92/1.07, mlp 1.20/2.07 (cheap
rungs / MK-MMD), median **1.12×**.

### Stage 2 — re-projected, fits, NOT launched

**4986 runs, 67.3 h wall at 4 shards**, projected from real Stage 1 medians
with the measured per-family reverse ratios applied rather than one flat
factor — a flat factor is wrong by 2× across families.

| family | runs | CPU h | wall h |
|---|---|---|---|
| mlp | 1626 | 150.0 | 37.5 |
| transformer | 108 | 60.8 | 15.2 |
| logreg | 1084 | 21.6 | 5.4 |
| svm_linear | 1084 | 20.7 | 5.2 |
| svm_rbf | 1084 | 16.2 | 4.1 |
| **total** | **4986** | **269.3** | **67.3** |

The first projection came in at **79.3 h**, over the 72 h ceiling. Trimmed the
transformer arm further, as instructed, **not** the direction: the transformer
now runs the primary direction only. Chosen over cutting it to one seed, which
would have left the arm with no spread to form an interval from — the whole
point of calling it a reduced-seed arm. Its rungs, backbones and aggregations
stay at full width, and direction is neither protected nor an inner grid.

Harness re-verified on the new design: 4986 enumerated ids **all unique**,
shard balance 1244/1233/1232/1277, resume after kill exact (6 written → 6
skipped on restart), merge refuses non-disjoint shards with **exit 2**.
397 tests pass.

### Files

Modified: `configs/default.yaml` (coral_shrinkage, sklearn_max_iter,
matched_source_train, split_spec_version v2), `configs/FROZEN` →
`grid-freeze-v3`, `src/ser/config.py`, `src/ser/splits.py` (matched-n),
`src/ser/classifiers.py` (convergence), `src/ser/run_grid.py` (blending in the
run path, extended eps, blending arm, transformer direction trim),
`src/ser/utils/results.py` (schema v9), `tools/migrate_results.py`,
`tools/project_stage2.py` (measured per-family ratios), `tests/test_grid.py`,
`tests/test_results_schema.py`.

Created: `tests/test_matched_n.py` (14 tests),
`results/stage2_calibration.jsonl`.

### Open

* **Full-n reverse arm** as a separate later run, to report matched-n against
  full-n and show what the size confound was worth.
* **`gaa` blending** remains unimplemented and out of scope. It raises rather
  than no-ops, so it cannot reach a table by accident.
* Two findings above ((a) depth divergence, (b) frame dependence) are recorded
  on Stage 1 evidence and do not enter the paper until Stage 2 confirms them.

---

## 2026-08-17 — Stage 1 screening analysis, pruning, and Stage 2 configuration

Stage 1 ran to completion on the user's machine. Full analysis in
[reports/stage1_analysis.md](reports/stage1_analysis.md); the `source_val`-only
pruning surface is in [reports/stage1_screening.md](reports/stage1_screening.md);
the layer sweep is in [reports/layer_sweep.md](reports/layer_sweep.md).

### Integrity

**438 rows: 60 baselines, 6 Stage 0, 372 Stage 1. Zero failures.** No
tracebacks, so no failure pattern to characterise. 372 = 360 screening + 12
transformer probe, exactly as enumerated.

Two columns carry nulls and **both are correct, not defects**:
`marginal_mmd_reference` is null in the 6 Stage 0 rows that predate the column;
`mmd_fallback_fired` is null in 159 rows — those 6 plus every rung with no warm
start to revert to (coral 96, none 21, zscore 18, mean_shift 18 = 153). The
MK-MMD rungs, where it is meaningful, have none. Aggregations filter rather than
impute. An earlier `statistics.fmean` crash on these nulls was the analysis code
missing a filter, not bad data.

Measured against projection: 25.1 CPU hours, ~6.3 h wall across 4 shards.
Per-family contention ratios 1.8×–4.7× over the solo projection. The transformer
probe is 12 of 372 runs and 20% of CPU time.

### Pruning — inner grids only

**Protected and untouched:** the alignment ladder, layer aggregation, backbone.
Screening showed the pooled ladder to be nearly flat on target and it is still
kept whole.

* **`alignment_eps`: dropped `1e-3`.** `source_val` is monotone increasing in
  eps (0.5592 → 0.6523) and eps=1e-1 is the argmax in **18/18** conditions.
  Monotone-to-the-boundary means the grid is mis-centred, not that the other
  values are dead, so only the interior, bracketed, shape-free point was cut.
  Ledoit-Wolf retained despite never winning — it lands at 0.6096, and "the
  standard automatic choice does not find the good regime" is a result.
  **Probe agrees 3/3** on the direction; it does not test 1e-3 itself, and that
  gap is recorded rather than glossed.
* **`alignment_lambda`: dropped `10.0` and `100.0`.** `source_val` is flat
  (spread 0.0016 diag / 0.0054 full) so the argmax is noise; the cut is made on
  mechanism instead — at λ ≥ 1 `mkmmd_full` reverts to its CORAL warm start in
  14/18 runs, so those cells are CORAL with extra wall time rather than a sixth
  rung. λ=1.0 kept as the anchor where the fallback saturates.
  **Probe cannot adjudicate** — it ran `mkmmd_full` at λ=0.001 only. Stated
  plainly rather than implied.
* **`blend_alpha`: NOT pruned — it was never screened.** All 372 Stage 1 runs
  ran `blending="none"`. There is no evidence about it either way, and 11
  blending settings would multiply Stage 2 by ~11. Needs its own screening pass.
  Pinned by `test_stage1_blending_axis_was_never_screened`.
* **Classifier ranges: NOT pruned.** Narrowing one family's range would break
  the equal-budget contract — 20 trials would buy more effective coverage for
  the narrowed family.

### Required analyses

1. **CORAL `source_val` vs eps — it recovers.** 0.5592 → 0.6523, i.e. **56% of
   the 0.166 cost**, while target moves 0.0096 across the whole range. So the
   cost is **substantially a statement about the regularisation, not about
   CORAL** — but not entirely: 0.089 remains at the strongest shrinkage, and the
   surface is still climbing at the grid edge, so 56% is a lower bound.
   Extending the grid means editing a frozen config and orphaning all 372 rows;
   **flagged, not done.**
2. **Effect size vs target — observation only.** The step off `none` is worth
   +0.10; a further 140× reduction in discrepancy is worth 0.011, i.e. noise.
   But **the pooled average hides a large interaction**: alignment is worth
   +0.042 at `last`, **+0.168 at `layer:6`**, +0.095 at `weighted`, sign-stable
   in all four screening families and reproduced in the independent transformer
   probe. "The ladder does nothing" is an artefact of averaging over depth.
3. **Fallback rate: `mkmmd_diag` 27.8%, `mkmmd_full` 61.3%.** Any table with an
   `mkmmd_full` row must say so — in the majority of those cells the rung is
   CORAL. The diag rate is 5/18 at *every* lambda, confirming it is
   feature-dependent, not budget-dependent.
4. **Full 13-layer sweep — the headline.** See the rewritten FINDING at the top
   of this file. In-domain-optimal depth is 4–5 layers shallower than
   transfer-optimal depth in all three backbones, costing 0.13–0.14 macro-F1.
   The sweep reproduces the Stage 1 grid at the `none` rung to within RNG noise
   (0.3700 vs 0.3698), which validates the harness independently.

**The hypothesis the sweep was meant to test did not hold.** A monotone curve —
discrepancy falling toward mid-stack with target falling alongside — is not what
the data shows. Target does not track discrepancy in any backbone, and the two
geometries **disagree in sign on two of three**. That is a stronger result than
the monotone curve would have been, and it forces every "lower MMD ⇒ better
transfer" claim to name its geometry.

### Stage 2 — configured, verified, NOT launched

**2133 runs, 33.0 h wall at 4 shards**, projected from measured Stage 1 medians
by `tools/project_stage2.py`. 276 runs already complete and resume rather than
recompute (verified by dry run).

Reduced factorial: 3 backbones, 5 seeds, both aggregations (three for torch
families), 6 rungs, pruned inner grids. Transformer is an explicitly reduced arm
— 2 seeds, one inner-grid setting per rung — reduced **only** on seeds and inner
grids, never on a protected axis, and **must be reported with wider intervals
and not pooled** with the five-seed families.

**The reverse direction does not fit and is a decision for the supervisor.**
`cremad→ravdess` has 5972 source-train utterances against 988; measured logreg
fit cost is ~16× (4.1 s → 35.6 s at C=1; 5.1 s → 84.8 s at C=100, where it hits
the `max_iter` cap). The MMD side is not the problem — it subsamples and costs
0.7 s either way. A full mirror is 264–528 h against a 72 h budget. Options and
costs are tabulated in the analysis; recommendation is forward-only now, reverse
as a separate reduced arm later once `tools/calibrate_stage2.py` has measured
per-family factors. Direction is neither protected nor an inner grid, so this is
scoping, not pruning.

### Files

Created: `tools/layer_sweep.py`, `tools/layer_sweep_report.py`,
`tools/project_stage2.py`, `tools/calibrate_stage2.py`,
`tools/launch_stage2.ps1`, `reports/stage1_analysis.md`,
`reports/layer_sweep.md`, `results/layer_sweep.jsonl`.

Modified: `src/ser/run_grid.py` (STAGE2_SURVIVING, `stage2_surviving`,
`_enumerate_stage2`, `_stage2_variants`, `surviving` threaded through
`run_grid`), `tests/test_grid.py` (+9 Stage 2 tests), `PROGRESS.md`.

Generated: `reports/stage1_screening.md` (the runner writes it at the end of an
unsharded run; the sharded launch never reached that path).

### Two process notes

* **The calibration was stopped deliberately.** `tools/calibrate_stage2.py` was
  measuring per-family reverse-direction factors; its SVM cells are O(n²) at
  n=5972 and would have run for hours on a question that forward-only Stage 2
  makes moot. The decisive number came from direct fit timing instead. The tool
  remains, and should be run before any reverse-arm launch.
* **A restarted sweep worker double-computed 23 wavlm cells.** I misread a
  process check and started a second run while the first was alive. All 23
  duplicates agree to floating-point equality — an unplanned determinism check.
  The report generator now collapses duplicates and *raises* if any pair
  disagrees. Only wavlm layer 1 shifted (0.1132 → 0.1167); no conclusion moved.

### Open

* **Extend the CORAL eps grid upward?** Needs a new freeze tag; orphans 372 rows
  for resume purposes (they stay valid as data).
* **Blending axis unscreened.** Needs a small dedicated pass before it can enter
  any factorial.
* **Reverse direction.** Scope decision above.
* **Logreg hits `max_iter` with CREMA-D as source** at high C. Capped trials lose
  selection on `source_val`, so this costs wall time rather than correctness —
  but it must be stated wherever reverse-direction numbers appear. Raising
  `max_iter` would edit the frozen config.

---

## 2026-08-16 — Stage 1 pre-flight, handed off for launch

### The transformer's alignment was applied in the wrong space — fixed

`_flatten` collapsed everything but the sample axis, so alignment was fitted in
the *flattened* space rather than the 768-dimensional feature space:

| condition | alignment dim | CORAL covariance |
|---|---|---|
| sklearn / MLP `last` | 768 | 768² |
| transformer `last` | 6144 | 0.30 GB each, 512× the eigendecomposition |
| MLP `weighted` | 9984 | 0.8 GB each |
| transformer `weighted` | 79872 | **~51 GB — would have OOM'd** |

Stage 1 contains `mlp × weighted × coral`, so this would have failed hours in.
It was also wrong on its own terms: a 6144-dimensional covariance from ~988
utterances has rank at most 987, and the conditioning would have differed per
family, making the alignment rung mean something different for the transformer
than for everything else.

**Fixed** by viewing the input as `(-1, D)` — every 768-d vector is one
observation — so every family aligns in the same space with the same
conditioning, and the map is fitted on exactly the distribution of vectors it is
applied to. Discrepancy is measured on a separate `_mmd_view` that mean-pools to
one vector per **utterance**, so the transformer's effect size is comparable to
every other family's rather than being computed over 8× as many points from a
within-utterance distribution.

**Recorded asymmetry, as requested.** Even after the fix the transformer is not
in quite the same position as the other families: its map is fitted on the
segment marginal, while sklearn and MLP fit on the utterance-level pooled vector.
Those are different distributions — segments have higher variance than their
own mean — and no single choice makes them identical, because the two families
consume different representations by construction. The map is at least fitted
and applied on the *same* distribution within each family, which the previous
code did not achieve for either.

### Stage 2 projection — the transformer arm exceeds the ceiling alone

Projected from measured per-family timings, under a reduced factorial of 4 pairs
(both directions plus in-domain) × 3 SSL backbones + MFCC × 5 seeds:

| surviving alignments | aggregations | sklearn | mlp | transformer | total |
|---|---|---|---|---|---|
| 2 | 1 | 6.4 | 9.0 | 50.0 | 65.4 |
| 3 | 2 | 16.7 | 23.6 | **131.4** | **171.6** |
| 4 | 3 | 31.8 | 44.9 | 250.2 | 326.9 |

At the expected scenario (3 alignments, 2 aggregations) the total is **172 h**
and the transformer arm alone is **131 h — 77% of the grid and well past the
72 h ceiling.** Even the most aggressive pruning leaves it at 50 h.

**Proposed trim, for a decision before Stage 2:**

| trim to the transformer arm | arm | grid total |
|---|---|---|
| none | 131 h | 172 h |
| 2 seeds instead of 5 | 53 h | 93 h |
| cross-domain pairs only | 66 h | 106 h |
| **2 seeds + cross-domain only** | **26 h** | **66 h** |
| 2 seeds + cross-domain + hubert only | 8 h | 48 h |

Recommended: **2 seeds, cross-domain pairs only** for the transformer, with
sklearn and MLP at the full 5 seeds and 4 pairs. That keeps the headline claim —
which is about alignment and rests on all families — at full statistical
strength, and reduces only a baseline whose role is to not be a strawman. It
must then be reported as such: the Transformer arm ran at reduced seed count for
compute reasons and its intervals are correspondingly wider.

### Scope of the launch

**Screening without the transformer: 360 runs.** At measured timings the
transformer was 80% of Stage 1, Stage 2 re-runs it anyway, and every other axis's
pruning decision is available without it.

**Stratified transformer probe: 12 runs** — both ends of the ladder (`none`,
`mkmmd_full`), every layer aggregation (`last`, `layer:6`, `weighted`), and both
epsilon extremes (1e-4, 1e-1). Purpose is to check that pruning decided on
sklearn + MLP transfers to the one family consuming segments. **If any pruning
decision would flip for the transformer, that axis is not pruned.**

372 runs total, 372 unique `run_id`s.

### Launch harness

Four shards by `run_id` hash (79 / 102 / 93 / 98), each with capped BLAS threads,
its own results file, and its own log with a timestamped heartbeat. Sharding is
by hash rather than by list slice so a half-finished shard stays resumable across
any code edit that reorders the enumeration.

**No pipes anywhere** — output is redirected straight to files, because buffered
pipes have swallowed output three times on this project.

Verified before hand-off:

| check | result |
|---|---|
| 372 runs → unique `run_id`s | **372/372 OK** |
| resume after a mid-run kill | **7 enumerated, 4 already complete, 3 executed, no duplicates, no lost work** |
| merge preserves uniqueness | **OK** |
| merge refuses conflicting duplicates | **refuses, exit 2** |
| config frozen | `grid-freeze-v2`, matches |

Not launched. A 9–13 h job cannot survive a Claude Code session, and a permission
prompt at hour three would stall it silently.

---

## 2026-08-16 — Items A–D, and the Stage 1 pre-flight

### ITEM A — the MK-MMD fallback is feature-dependent, not budget-dependent

Set out to raise the budget; the measurement said the budget was not the
problem.

| features | warm start | fitted, 200 steps | fitted, 500 steps | fell back |
|---|---|---|---|---|
| `last` | 3.76 | **3.45** | 3.69 | never |
| `layer:6` | 1.93 | 1.93 | 1.93 | always |

On `last` the optimiser beats its CORAL warm start at every budget tried (100,
200, 300, 500) and never reverts. On `layer:6` it reverts at every budget —
because the warm start is *already* at 1.93× the same-distribution null. There is
essentially nothing left to gain, so no budget can find any.

**The finding is not "the budget was wrong". It is that MK-MMD has nothing to add
once CORAL has taken the discrepancy to within about 2× of the noise floor.**
That is a statement about the method's practical value on this data, and it is
now measurable rather than anecdotal: `mmd_fallback_fired` is a column, and the
Stage 1 report computes the rate per rung. Any table containing `mkmmd_*` must
state that rate.

Budget set to the validated **200 steps** — as good or better than 500 at half
the cost. Config re-frozen at **`grid-freeze-v2`**.

### ITEM B — fixed reference geometry

One ZCA map derived **once** from the unaligned `source_train` covariance
(shrinkage 1e-2, chosen because an effective rank near 57 of 768 would otherwise
make the reference frame itself noise-dominated), applied identically to every
rung's output before the MMD. Recorded as `marginal_mmd_reference` beside the
per-rung-geometry figure.

Being one fixed linear map it cannot undo any rung's alignment, and being the
same for all rungs it removes the *choice of frame* as a degree of freedom.
Stated honestly in the docstring: it does **not** equalise the anisotropy each
rung produces. The coarse ladder claim (hundreds-fold vs single-digit) survives
either statistic; the fine ordering is what this column exists to arbitrate.

### ITEM C — Stage 0 framing corrected

The dose-response reading has been retracted in place. The claim is now only
what survives error bars: **target macro-F1 is flat across 226× of marginal
discrepancy.** No ordering may be asserted among rungs from a single seed, and
`zscore` / `mkmmd_diag` / `mean_shift` are explicitly a tie band.

### ITEM D — CORAL's source_val cost is a question, not a conclusion

The Stage 1 report tabulates `source_val` against `alignment_eps` for CORAL,
with the note that if it recovers at larger eps while target stays flat, the
0.166 cost is a property of the **regularisation** and not of CORAL — and the
paper has to say which.

### Stage 1 pre-flight

Run **before** launching, as required:

| check | result |
|---|---|
| `run_id` uniqueness across the full enumeration | **480 runs → 480 ids, OK** |
| predictions written for every completed run | **6/6, 0 missing, OK** |
| config frozen | `grid-freeze-v2`, matches |

**Projected wall time: 46.6 h** (480 runs, 350 s/run mean). Under the 72 h
ceiling, but the distribution is lopsided:

| family | hours | share | runs |
|---|---|---|---|
| transformer | 37.3 | **80.0%** | 120 |
| mlp | 6.5 | 14.0% | 120 |
| logreg | 1.1 | 2.3% | 80 |
| svm_linear | 0.9 | 2.0% | 80 |
| svm_rbf | 0.8 | 1.7% | 80 |

Flagged rather than silently accepted: a *screening* pass spending four fifths of
its budget on one family is poor economics, since Stage 2 re-runs that family
anyway and the pruning decisions for every other axis are available from the
remaining 9.3 h. Dropping the Transformer from screening only would cut Stage 1
to ~9 h. That is a call for a human, not for the runner.

---

## 2026-08-16 — Phase 7 Stage 0: the smoke gate

Status: **PASS.** 6 runs, 0 failed, 4.3 min (43 s/run). Every rung clears its
pair's own chance floor. `ravdess → cremad`, seed 0, HuBERT, `last`, logreg.

| alignment | source_val | target | chance | effect size |
|---|---|---|---|---|
| none | 0.7580 | 0.3674 | 0.1665 | **834.8** |
| zscore | 0.7447 | **0.3864** | 0.1665 | 36.4 |
| mean_shift | 0.7580 | 0.3840 | 0.1665 | 45.2 |
| coral | 0.5921 | 0.3798 | 0.1665 | **3.8** |
| mkmmd_diag | 0.7559 | **0.3864** | 0.1665 | 39.4 |
| mkmmd_full | 0.6090 | 0.3809 | 0.1665 | **3.7** |

### This is the A8 question, answered in miniature

**Target macro-F1 is flat across 226× of marginal discrepancy** — 834.8 down to
3.7, with every rung between 0.3674 and 0.3864.

That is the entire claim, and it is the only one this data supports.

**CORRECTED 2026-08-16.** An earlier version of this entry read the 0.019 spread
as a dose-response curve and asserted that "the first, cheapest step buys the
entire gain" (`none → zscore` +0.019, then `zscore → coral` −0.007). That
over-reads the data. On **one seed**, at 6-class macro-F1 over ~1500 target
utterances, 0.019 and 0.007 are both well inside what a seed change will move,
and there is no variance estimate here at all. Any internal structure attributed
to that spread is an argument five seeds could dismantle.

Nothing may be claimed about the *ordering* of rungs from Stage 0. In particular
`zscore` (36.4), `mkmmd_diag` (39.4) and `mean_shift` (45.2) are **within the
tie band** of the effect-size metric (see Item B below) and must not be ranked
against each other. Stage 2, with five seeds, decides whether any step is real.

What does survive: the flatness itself. A 226-fold reduction in marginal
discrepancy produced no change in transfer that is distinguishable from noise.

Note also what CORAL costs on the source side: `source_val` drops from 0.758 to
0.592. Matching the target's covariance actively degrades source-domain fit, and
buys nothing on target. `mkmmd_full` shows the same pattern (0.609), which is
expected since its fallback makes it CORAL.

This is one pair, one backbone, one seed and one classifier, so it is a
direction rather than a result. But it is the direction A8 predicted, measured on
a dose-response axis rather than two arbitrary points, and it is what Stage 1 is
now designed to test properly.

### A bug the tests caught before the grid ran

`test_every_enumerated_run_has_a_distinct_run_id` failed on Stage 1's
enumeration: **480 runs collapsed onto 144 `run_id`s.** `alignment_eps` and
`alignment_lambda` were neither recorded nor coordinates, so all four CORAL
epsilons and all six MK-MMD lambdas shared one id. A resume would have skipped
three of every four CORAL runs *and reported the grid complete*.

Schema **v7** adds both as columns and as `run_id` coordinates. Since this
redefines identity it cannot be migrated — `tools/migrate_results.py` refuses
across it — so results were archived and the 60 baseline rows regenerated.

Also fixed: the effect size is now computed for **every** rung including `none`.
It was skipped when no alignment was fitted, which left a hole in the
covariate-shift column at exactly the reference point the ladder is read against.

### The runner

`ser run-grid --stage N`, resumable, append-only, one prediction file per run.

- **Refuses to start** unless the working config matches `grid-freeze-v1`.
  Verified: the guard is exercised in `tests/test_grid.py`, and the freeze
  survives schema changes because the schema is not part of the config.
- A crashed run writes `status="failed"` with its traceback and the runner
  continues; a silent skip leaves a hole nobody can see.
- Every run records per-class precision/recall/F1/support, the confusion matrix,
  collapse count, selected hyperparameters, trial count, `source_val` and target
  scores **separately**, epochs to early stop, the alignment effect size, wall
  time, and the per-utterance predictions.
- `assert_alignment_blind_to_target_test` runs against the real fitted object on
  **every** run, not just in tests.

Stage 2 enumeration deliberately raises until Stage 1 has recorded its pruning
decisions — the surviving axes are an input a human has to supply.

---

## 2026-08-16 — Items 1–3: MK-MMD invariants, bandwidth robustness, config freeze

### ITEM 1 — `mkmmd_diag`, and an invariant that turns out not to hold

**The warm start was already correct.** It computes exactly
`W = σ_tgt/σ_src`, `b = μ_tgt − W·μ_src`, the diagonal moment match as specified.
So the violation was not a wrong initialisation.

**A fallback now enforces the invariant that *is* valid.** After fitting, the
warm-start transform and the fitted transform are evaluated on the same data at
the same bandwidth, and the better one is kept
(`alignment.mmd_fallback_to_warm_start`). Shipping a fitted map that is worse
than its own starting point would report the optimiser's failure as the method's
performance. When the fallback fires it is recorded in `diagnostics` and printed
in the report — "the optimiser did not beat its own starting point" is a fact
about the method, not something to hide.

**The proposed invariant `mkmmd_diag ≤ zscore` is not valid, and measurement
says so.** The reasoning was that the diagonal warm start differs from z-scoring
only by a global affine map, which a scale-invariant effect size cannot see.
Measured on `ravdess → cremad`:

| transform | effect size |
|---|---|
| `zscore` | **33.4** |
| diagonal moment match (the warm start itself) | **47.5** |
| `mkmmd_diag` fitted | 45.4 |

The warm start is **already worse than z-score before any optimisation**, so no
optimiser could have satisfied the invariant. The cause is geometric:

* `zscore` rescales **both** domains to isotropic unit variance.
* `mkmmd_diag` rescales **only the source**, onto the target's per-dimension
  variances — which stay anisotropic. Measured spread of target per-dimension
  std on this pair: **17.8×** (0.0144 to 0.2564).

An isotropic RBF kernel is invariant to a *global* scale but not to
*per-dimension* reweighting, so the effect size cannot equate the two. They are
different families and neither contains the other: a source-only diagonal map
cannot reach z-score's geometry because it cannot touch the target.

Note the fitted diagonal (45.4) *does* improve on its warm start (47.5), so
invariant (a) holds for it. It is `mkmmd_full` whose fallback fires.

### ITEM 2 — bandwidth robustness

Every rung measured at {0.25, 0.5, 1, 2, 4} × its own median heuristic, with the
saturation diagnostic at each. Effect size (lower is closer to
same-distribution):

| rung | 0.25× | 0.5× | 1× | 2× | 4× |
|---|---|---|---|---|---|
| none | 950.0 | 1034.1 | 1059.3 | 1215.2 | 1724.4 |
| zscore | 39.8 | 34.7 | 33.4 | 31.3 | 12.6 |
| mean_shift | 73.2 | 67.3 | 65.7 | 64.6 | 35.0 |
| coral (eps=1e-4) | 2.6 | 2.0 | 1.9 | 1.4 | −0.7 |
| coral (Ledoit-Wolf) | 5.6 | 4.9 | 4.7 | 3.8 | 0.2 |
| mkmmd_diag | 51.9 | 46.8 | 45.4 | 44.1 | 23.1 |
| mkmmd_full | 2.6 | 2.0 | 1.9 | 1.4 | −0.7 |

**The ordering is stable across all five bandwidths** — every rung has an
identical rank vector:

| rung | ranks at 0.25× / 0.5× / 1× / 2× / 4× |
|---|---|
| coral (eps=1e-4) | 1, 1, 1, 1, 1 |
| mkmmd_full | 1, 1, 1, 1, 1 |
| coral (Ledoit-Wolf) | 3, 3, 3, 3, 3 |
| zscore | 4, 4, 4, 4, 4 |
| mkmmd_diag | 5, 5, 5, 5, 5 |
| mean_shift | 6, 6, 6, 6, 6 |
| none | 7, 7, 7, 7, 7 |

So the ordering is a property of the data, not of one bandwidth choice — which
is what had to be shown before this table goes in a paper.

The first run reported `coral` and `mkmmd_full` swapping ranks 1↔2. That was an
artefact of the ranking code, not the data: `mkmmd_full` had fallen back to its
CORAL warm start, so the two are **the same transform** and their values are
identical to every digit. Ranking now treats values within 1% as tied, and they
share rank 1.

**`mkmmd_full`'s fallback fires under the default budget.** At 500 steps,
step-norm 0.01 and batch 256 the optimiser does not beat its CORAL warm start, so
the warm start is what gets reported — and the report says so explicitly rather
than presenting it as a fitted result. The earlier standalone diagnostic that did
beat CORAL (1.2× vs 2.0×) used 200 steps at a hand-set lr=1e-5 and measured
against the *fixed* bandwidth; under the per-rung bandwidth and the derived step
size it does not reproduce. **`mkmmd_full` is therefore currently CORAL by
another name, and must be labelled that way in any table until the optimiser
budget is revisited.**

Negative effect sizes at 4× are expected and not a bug: the unbiased MMD²
estimator is signed, and a value at or below the same-distribution null means the
two samples are statistically indistinguishable at this sample size.

### ITEM 3 — the config freeze, made mechanical

`configs/FROZEN` holds a git tag name; the tagged commit's
`configs/default.yaml` is the frozen config. `ser.freeze.assert_config_frozen`
compares the **parsed** working config against it, so reformatting and comment
changes are not drift while a value change is. The Phase 7 runner refuses to
start when the config is unfrozen or has drifted.

This matters more since schema v4 than it did before: with `config_hash` no
longer a `run_id` coordinate, a mid-grid edit no longer *orphans* completed runs —
it silently produces rows that are not comparable to earlier ones **under the
same ids**. The freeze is what closes that gap.

Schema **v5** adds `freeze_tag`, recorded on every row but deliberately **not** a
`run_id` coordinate, so re-freezing does not invalidate completed work. Migrated
v4 → v5 in place; 60 rows, 50 fields, all `run_id`s preserved.

---

## 2026-08-16 — Corrections 1–3, effective rank, and Phase 6

### CORRECTION 1 — the MMD diagnostic, and a Phase 5 claim overturned

**Phase 5's headline finding was wrong.** That entry reported "z-scoring alone
removes 97.6% of marginal MMD, more than CORAL". It does not. The number was an
artefact of re-estimating the RBF bandwidth per rung.

Two changes, both required, and the second only became visible after the first:

**Bandwidth fixed once** on the unaligned pair (2.1209) and reused for every
rung. Re-estimating per rung is scale-dependent — a rung that shrinks the
features shrinks the median pairwise distance, the kernel widens to compensate,
and the reported MMD falls with nothing having moved closer.

**But a fixed bandwidth is invalid for a rung that changes the feature scale.**
Measured directly: z-scoring 768 dimensions moves the median pairwise distance
from 1.52 to 35.57 — **16.8× the fixed bandwidth** — at which point the kernel
value at the widest multiplier is **1.5e-4**. Everything reads as infinitely far
apart and MMD² collapses toward zero regardless of overlap. Under the fixed
bandwidth z-score appeared to reach 0.000579, better than CORAL. That is pure
saturation.

The statistic that is genuinely scale-invariant is the **effect size**: MMD at a
bandwidth appropriate to the transformed data, divided by the same-distribution
MMD of that same transformed data. Both move together under rescaling, so the
ratio cannot be manufactured. A saturation diagnostic is reported alongside so
the artefact is visible rather than silent.

| rung | MMD²@fixed | saturation | **effect size** | cond | eff. rank |
|---|---|---|---|---|---|
| none | 0.863064 | 0.96 | 982.6 | — | — |
| zscore | 0.000579 | **1.8e-04 ⚠ SATURATED** | **31.3** | — | — |
| mean_shift | 0.053219 | 0.98 | 61.0 | — | — |
| coral (eps=1e-4) | 0.001813 | 0.97 | **2.0** | 1.82e+06 | 57.2 |
| coral (eps=1e-3) | 0.001985 | 0.97 | 2.2 | 1.82e+05 | 57.2 |
| coral (eps=1e-2) | 0.003260 | 0.97 | 3.6 | 1.82e+04 | 57.2 |
| coral (eps=1e-1) | 0.010437 | 0.98 | 11.9 | 1.82e+03 | 57.2 |
| coral (Ledoit-Wolf) | 0.004497 | 0.97 | 4.8 | 4.40e+04 | 57.2 |
| mkmmd_diag (λ=1e-3) | 0.050789 | 0.97 | 56.8 | — | — |
| mkmmd_full (λ=1e-3) | 0.017434 | 0.97 | 20.8 | — | — |

**Corrected conclusion: CORAL is decisively the best rung** at 2.0× the
same-distribution null — nearly indistinguishable from sampling noise. z-score is
31.3×, roughly 15× worse than CORAL, not better than it. The A8 argument is
unaffected in structure (it turns on whether transfer macro-F1 tracks *any* of
this), but the specific Phase 5 sentence must not reach the paper.

**Deviation from the brief, deliberate.** The normalisation denominator is the
mean **absolute** null MMD², not the signed mean. With the unbiased estimator the
signed mean is zero by construction: measured at −3.5e-4 against a spread of
9.2e-4, so the literal ratio would be unstable and would flip sign. The absolute
mean (8.8e-4) agrees with the spread to within 5%; both are recorded.

### CORRECTION 2 — MK-MMD does not converge, root cause identified

At λ=0, full-W MK-MMD reaches **0.0113** against CORAL's **0.0018**. It does not
beat CORAL, and it should: CORAL's W is in the feasible set, so the MMD optimum
is at least as good. Diagnostics as requested:

| condition | steps | final MMD² | ×null | final grad norm | converged | objective trace |
|---|---|---|---|---|---|---|
| identity, lr 1e-3, batch 256 | 500 | 0.011335 | 12.9 | 4.60e-01 | **False** | 0.8463 → 0.0153 |
| identity, lr 1e-3, batch 256 | 1500 | 0.012240 | 13.9 | 4.89e-01 | False | — |
| identity, lr 1e-3, batch 256 | 3000 | 0.015899 | 18.1 | 4.91e-01 | False | — |
| identity, lr 1e-2, batch 256 | 500 | 0.106171 | 120.9 | 9.91e-01 | False | — |
| identity, lr 5e-2, batch 256 | 1000 | 2.696547 | 3070.0 | 5.51e-06 | False | — |
| identity, lr 1e-3, **full batch** | 300 | 0.008509 | 9.7 | — | — | 0.8561 → 0.0122 |
| **CORAL warm start**, full batch | 100 | 0.004001 | 4.6 | — | — | **0.00400 → 0.00580** |
| **CORAL warm start**, full batch | 300 | 0.005053 | 5.8 | — | — | **0.00400 → 0.00848** |

More steps make it *worse*. A higher learning rate collapses it entirely (lr=5e-2
lands at 2.70, worse than no alignment at all, with a gradient of 5.5e-6 — the
kernel saturated and the gradient vanished). Full batch helps only marginally.

**The decisive test.** Warm-starting from CORAL — verified to reproduce CORAL
exactly, max|diff| 9.8e-15 — the optimiser's own objective **increases**, 0.00400
→ 0.00580. It is not converging slowly; it is failing to descend from a good
solution.

Root cause: **Adam normalises per parameter**, so a single step at lr=1e-3 across
768×768 parameters moves ‖ΔW‖_F ≈ √590000 × 1e-3 ≈ 0.77 — an enormous
displacement from a near-optimal point. The learning rate is not scaled to the
parameter count.

An earlier version of this diagnostic was itself wrong: CORAL is
`(x − μ_s)·M + μ_t`, and initialising `W = Mᵀ` with `b = 0` drops both mean terms,
starting at objective 4.59 rather than CORAL's 0.0040. `MKMMDAlignment.coral_warm_start`
now returns both, with an assertion that it reproduces CORAL.

**Status: diagnosed, not yet fixed.** `mkmmd_diag` and `mkmmd_full` are **not
validated for Phase 7**. Phase 6 does not use them — it runs classifiers on
unaligned features — so this blocks Phase 7, not Phase 6.

### CORRECTION 3 — `config_hash` removed from `run_id`

Replaced by four facet hashes, each covering the part of the config that
determines what a run computes: `label_map_hash` (labels), `split_spec_hash`
(splits minus `seeds`), `feature_spec_hash` (features), `search_spec_hash`
(alignment, blending, classifiers, baselines, stats). `config_hash` is still
recorded. `splits.seeds` is excluded deliberately — the per-run `seed` is already
a coordinate, so hashing the list would make adding a sixth seed invalidate the
five that already ran.

`Config.classify_config_key` maps every key to a facet, a coordinate, or
`INERT_CONFIG_KEYS` (`project`, `paths`, `grid`, `shift`) and **raises** on
anything else. A test walks the whole config and fails on any unclassified key —
the mirror of the existing test that no `run_id` coordinate is inert.

Schema **v4**: `feature_spec_hash`, `search_spec_hash`, `n_search_trials`,
`marginal_mmd_raw`, `marginal_mmd_normalised`. 47 fields.

A `run_id` coordinate change **cannot be migrated** — old ids were computed over
a different field set, so carrying them forward would assert an equivalence that
does not hold. `tools/migrate_results.py` now refuses across such a version and
says so; the v3 file was archived and the 60 baseline rows regenerated.

### Effective rank per backbone per layer

**Estimator: spectral entropy** (Roy & Vetterli 2007), `exp(-Σ pᵢ log pᵢ)` over
the normalised eigenvalue spectrum — the estimator behind the
`cov_effective_rank` column. Participation ratio `(Σλ)²/Σλ²` is reported
alongside in `reports/effective_rank.md`.

Nominal dimension 768 throughout; source_train of each in-domain split, seed 0:

| corpus | backbone | L0 | L2 | L4 | L6 | L8 | L10 | L12 |
|---|---|---|---|---|---|---|---|---|
| ravdess | hubert | 18.0 | 28.1 | 35.9 | 47.3 | 44.8 | 41.0 | 28.9 |
| ravdess | wav2vec2 | 22.9 | 26.5 | 31.0 | 38.0 | 34.2 | 29.2 | 20.9 |
| ravdess | wavlm | 17.1 | 26.8 | 41.0 | 48.4 | 46.1 | 40.0 | 26.3 |
| cremad | hubert | 12.4 | 28.5 | 33.1 | 50.3 | 60.6 | 53.8 | 32.5 |
| cremad | wav2vec2 | 15.3 | 23.0 | 27.8 | 37.9 | 30.2 | 20.8 | 15.6 |
| cremad | wavlm | 12.2 | 28.3 | 49.3 | 61.9 | 62.0 | 49.2 | 31.7 |

Range **11.4 to 62.0 of 768** — never above 8% of nominal dimension. Effective
rank rises through the stack, peaks at layers 6–8, and falls again at layer 12,
which is the same shape the layer-aggregation argument predicts: the middle
layers carry the richest representation and the final layer narrows toward the
pretraining objective. wav2vec2 is consistently lowest and bottoms out at 11.4
(cremad, L11).

This is why CORAL's shrinkage is load-bearing rather than cosmetic: it whitens a
matrix whose usable rank is a few percent of its nominal size.

### CORRECTION 2, continued — fixed and verified

The learning rate was the whole problem, and the fix is principled rather than
tuned. Config now specifies **`mmd_step_norm`** — the target Frobenius norm of a
single optimiser step — and the learning rate is derived as
`step_norm / sqrt(n_parameters)`:

| variant | parameters | derived lr |
|---|---|---|
| `mkmmd_full` | 589,824 | 1.30e-05 |
| `mkmmd_diag` | 768 | 3.61e-04 |

1.30e-05 is exactly the value that worked in the diagnostic. One step now means
the same displacement for both variants instead of being 768× larger for one.

Both variants are also **warm started**: `mkmmd_full` at the fitted CORAL
solution, `mkmmd_diag` at per-dimension moment matching
(`w = σ_t/σ_s`, `b = μ_t − w·μ_s`) — the diagonal-family analogue, since CORAL's
dense solution cannot be projected onto a diagonal without becoming a different
transform.

**Result: `mkmmd_full` now reaches 1.2× null, beating CORAL's 2.0×** — exactly
the ordering the brief said must hold, since CORAL's W is in the feasible set.
`mkmmd_diag` improved from 173.3× to 38.4×.

Final ladder by effect size (lower is closer to same-distribution):

| rung | effect size | | rung | effect size |
|---|---|---|---|---|
| **mkmmd_full** | **1.2** | | zscore | 31.3 ⚠ saturated |
| coral (1e-4) | 2.0 | | mkmmd_diag | 38.4 |
| coral (1e-3) | 2.2 | | mean_shift | 61.0 |
| coral (1e-2) | 3.6 | | none | 982.6 |
| coral (Ledoit-Wolf) | 4.8 | | | |
| coral (1e-1) | 11.9 | | | |

The ordering is now monotone in expressiveness, which is what a ladder should
look like and what the previous configuration did not produce.

**Residual caveat, recorded rather than papered over.** On small synthetic data
the diagonal optimiser can still degrade its own warm start, so a test asserting
`final_objective < initial_objective` was wrong twice over: it compares minibatch
estimates on different batches, and with a warm start the initial value is
already good. The test now asserts the meaningful invariant — the fitted
transform beats no alignment on the *evaluated* MMD. `mkmmd_full` is validated
for Phase 7; `mkmmd_diag` is usable but has not been shown to improve on its own
warm start, which should be checked before its numbers carry weight.

### PHASE 6 — classifiers, layer aggregation, equal budget

Five families, **20 trials each, identical**, all selection on `source_val`, no
target data reaching any fitting or selection step. `ravdess → cremad`, seed 0,
HuBERT, K=6.

| family | layer agg | trials | source_val | target | epochs | sec |
|---|---|---|---|---|---|---|
| logreg | last | 20 | 0.7580 | 0.3674 | — | 45 |
| logreg | layer:6 | 20 | 0.8110 | 0.2190 | — | 22 |
| svm_linear | last | 20 | 0.7383 | 0.3178 | — | 39 |
| svm_linear | layer:6 | 20 | 0.8043 | 0.1987 | — | 17 |
| svm_rbf | last | 20 | 0.7345 | 0.3326 | — | 21 |
| svm_rbf | layer:6 | 20 | 0.8270 | 0.2871 | — | 22 |
| mlp | last | 20 | 0.7997 | 0.3833 | 47 | 194 |
| mlp | layer:6 | 20 | 0.8453 | 0.2445 | 16 | 215 |
| mlp | weighted | 20 | 0.8152 | 0.3742 | 34 | 138 |
| transformer | last | 20 | 0.8225 | **0.4137** | 19 | 1365 |
| transformer | layer:6 | 20 | 0.8614 | 0.3353 | 36 | 724 |
| transformer | weighted | 20 | **0.8894** | 0.3737 | 28 | 1229 |

Chance at K=6 is 0.167, so every condition clears its floor comfortably —
unlike the original study's sub-chance aggregates.

**The finding: `source_val` and target disagree systematically.** In every one of
the five families, `layer:6` scores *higher* on `source_val` than `last` and
*lower* on target. The middle layers fit the source better and transfer worse.
This inverts the usual "middle layers carry the paralinguistic signal" reasoning
for the cross-corpus case — it holds in-domain, and reverses across corpora.

Consequence: **selection on `source_val` does not pick the best transferring
condition.** Over all twelve conditions the validated protocol picks
`transformer/weighted` (source_val 0.8894, the highest) whose target is 0.3737;
the oracle is `transformer/last` at 0.4137.

**Validated 0.3737 vs oracle 0.4137 — gap +0.0400 macro-F1, about 11%
relative** — with a *correct* selection protocol, on one pair with one backbone.
This is the Phase 8 validated-vs-oracle result appearing early, and it is not an
artefact of a bad selection rule; it is what an honest selection rule costs.

*(An earlier version of this entry, committed before the final condition
finished, reported the validated pick as `transformer/layer:6` and the gap as
+0.078. `transformer/weighted` then landed with a higher `source_val`, which
moves the validated pick and narrows the gap. The numbers above are the complete
twelve-condition result.)*

**The learnable weighting avoids the mid-layer trap.** `weighted` beats
`layer:6` on target in both families that can use it — MLP 0.3742 vs 0.2445,
Transformer 0.3737 vs 0.3353 — while also scoring highest on `source_val`. A
learned softmax over all 13 states recovers most of what committing to a fixed
middle layer throws away, which is the concrete argument for caching every layer
in Phase 3.

**Two expectations of mine that the data refuted.** I predicted the Transformer
would lose to the MLP at these data sizes — it is the best condition on target
(0.4137 vs 0.3833) and on `source_val`. And I expected middle layers to help;
they help only in-domain.

Early stopping is real: 16–47 epochs against a 200-epoch cap, different per
condition, never a fixed count.

**Equal budget is enforced, not asserted.** `n_search_trials` is recorded on
every row, and a test builds every family and asserts the trial counts are
identical. A failed trial is recorded and scored −∞ rather than retried, so a
fragile family cannot quietly buy extra attempts.

`weighted` is offered only to the torch families: the softmax over 13 layers is a
classifier parameter, and a closed-form sklearn model has none to learn it with.
Requesting it on logreg raises rather than silently substituting something else.

**No standardisation inside any classifier.** The obvious `StandardScaler` in
front of every sklearn model — which the original had — would make the `zscore`
rung a no-op and the `none` rung unmeasurable, silently collapsing two conditions
the paper reports as distinct.

---

## 2026-08-15 — Phase 5: the alignment ladder

Status: **complete for `{ravdess, cremad}`.** `pytest` → 316 passed. End-to-end
sanity run over all six rungs on `ravdess → cremad`, every rung passing the
Phase 2 leakage assertion against a real fitted object.

### The ladder, measured

`ravdess → cremad`, seed 0, HuBERT `layer:6`. `source_train` is **(988, 768)** —
n < d, so the covariance is rank-deficient by construction, exactly as the brief
warned.

| rung | marginal MMD² after | reduction | cond. number | effective rank |
|---|---|---|---|---|
| *(before)* | 0.863419 | — | — | — |
| none | 0.863419 | 0.0% | — | — |
| zscore | 0.020602 | **97.6%** | — | — |
| mean_shift | 0.053957 | 93.8% | — | — |
| coral (eps=1e-4) | 0.001843 | **99.8%** | 1.82e+06 | 57.2 |
| coral (eps=1e-3) | 0.002016 | 99.8% | 1.82e+05 | 57.2 |
| coral (eps=1e-2) | 0.003303 | 99.6% | 1.82e+04 | 57.2 |
| coral (eps=1e-1) | 0.010546 | 98.8% | 1.82e+03 | 57.2 |
| coral (Ledoit-Wolf) | 0.004548 | 99.5% | 4.40e+04 | 57.2 |
| mkmmd_diag (λ=1e-3) | 0.051186 | 94.1% | — | — |
| mkmmd_diag (λ=1) | 0.080392 | 90.7% | — | — |
| mkmmd_diag (λ=100) | 0.081528 | 90.6% | — | — |
| mkmmd_full (λ=1e-3) | 0.017479 | 98.0% | — | — |
| mkmmd_full (λ=1) | 0.058600 | 93.2% | — | — |
| mkmmd_full (λ=100) | 0.080016 | 90.7% | — | — |

**Three findings, all provisional until a classifier exists (Phase 6).**

*Effective rank is 57.2 out of 768.* The source covariance concentrates
essentially all of its energy in ~57 directions. This is far more extreme than
"n < d" alone implies and it reframes what CORAL is doing: whitening a matrix
whose usable rank is 7% of its nominal dimension. It is also the strongest
argument yet that regularisation is not a numerical nicety here.

*z-scoring alone removes 97.6% of marginal MMD* — more than `mean_shift`, and
more than either MK-MMD variant. If transfer macro-F1 does not track this, the
paper's central claim (A8: alignment minimises marginal discrepancy while
conditional discrepancy moves the boundary) has its cleanest possible evidence.

*MK-MMD underperforms CORAL on the very objective it optimises.* Expected, and
worth stating plainly: it is minibatched at 256 with 500 steps and anchored by
λ‖W−I‖²_F, while CORAL solves second-moment matching in closed form. λ behaves
exactly as designed — larger λ means less MMD reduction and a W closer to
identity, so the rung degrades towards `none` rather than towards noise.

### Numerics

`src/ser/numerics.py` is the load-bearing module.

- **float64 or refuse.** `require_float64` raises rather than silently upcasting,
  so a caller that handed over a float16 cache slice and believed it was fitting
  in double precision finds out. Asserted at the entry to every `fit`; two
  parametrised tests confirm float16 and float32 are both rejected.
- **Conditioning is measured, not assumed.** Condition number, entropy-based
  effective rank (Roy & Vetterli), numerical rank, and eigenvalue extremes are
  computed for every covariance and the first two land on the run row as
  explicit columns.
- **Singular means fail, not pseudo-invert.** A regularised covariance worse
  than 1e12 raises `SingularCovariance`. Silently pseudo-inverting would produce
  a number that looks like a result.
- **Shrinkage is scale-aware:** `Cov + eps · trace(Cov)/d · I`. The original
  added a fixed `1e-5 · I` regardless of feature scale, so the same nominal
  epsilon meant something different for every backbone and layer. Anchoring to
  the mean eigenvalue makes `eps` comparable across the grid — there is a test
  asserting the shrinkage ratio is invariant to feature scaling.

### CORAL and MK-MMD specifics

CORAL cannot be constructed unregularised — `CoralAlignment()` with no `eps` and
no `ledoit_wolf` raises. `eps` is a searched axis (1e-4 … 1e-1) recorded in the
run row, and Ledoit-Wolf is a parameter-free second variant whose analytic
shrinkage lands between eps=1e-3 and 1e-2 here.

MK-MMD learns `W, b` minimising `MMD²_k(WX+b, X_tgt) + λ‖W−I‖²_F` with the RBF
sum at `{0.25,0.5,1,2,4}×σ_median`. λ spans 1e-3…1e2. W initialises at the
identity, so step 0 is exactly the `none` rung and any improvement is
attributable to optimisation rather than initialisation. The diagonal variant is
a separate rung with 768 parameters against ~590k.

### Two traps caught

**YAML 1.1 numeric parsing.** `1.0e1` parses as the *string* `"1.0e1"` because
YAML only recognises scientific notation when the exponent carries a sign, while
`1.0e-1` parses as a float. The λ grid was silently half strings until the
validator rejected it. Rewritten as plain decimals with a comment, and the whole
config scanned for other instances (none).

**`config_hash` is a `run_id` coordinate, so any config edit orphans every run.**
Editing the `alignment` section made the 60 baseline rows re-run rather than
resume; `results/runs.jsonl` now holds three `config_hash` generations, 121 rows,
zero collisions. The append-only discipline held and nothing was corrupted, but
at grid scale this is the difference between resuming and restarting. Written
into PHASES.md A6: **freeze the config before the grid, and filter analysis by
`config_hash`.**

### Schema v3 and a real migration

`cov_condition_number` and `cov_effective_rank` added as explicit columns rather
than a corner of `hyperparams_json`, because "which runs were near-singular?" has
to be answerable by filtering. 44 fields.

`tools/migrate_results.py` migrates v2 → v3 in place with a backup, adding both
fields as null (no pre-v3 row formed a covariance, so null is truthful rather
than a placeholder). Regenerating would also have worked for cheap baseline rows,
but it would have replaced provenance — new timestamps, new git SHA — for rows
whose numbers did not change. All 61 `run_id`s preserved.

### Files created / modified

```
src/ser/numerics.py        dtype discipline, conditioning, shrinkage, PSD roots
src/ser/mmd.py             multi-kernel MMD + the affine map that minimises it
src/ser/alignment.py       six rungs behind one contract
src/ser/blending.py        scalar and group-wise blending + enumeration rule
src/ser/alignrun.py        `ser align-check`
src/ser/features/load.py   split ids -> cache rows, by id and never by position
src/ser/utils/results.py   schema v3
tools/migrate_results.py   v2 -> v3
configs/default.yaml       six rungs, searched eps and lambda grids
tests/test_alignment.py    49 tests
reports/alignment_check.md
```

### Deferred

- **No selection.** Choosing `eps`, `λ`, or `α` on `source_val` needs a
  classifier: Phase 6. Nothing here is tuned.
- Blending transforms and the enumeration rule are implemented and tested, but α
  selection is Phase 6/7.
- `marginal_mmd` before/after is collected per rung and is the covariate-shift
  column Phase 9 needs; conditional-shift MMD stays behind the A10 firewall and
  is not computed here.

---

## 2026-08-15 — Phase 4: metrics, chance floors, and statistics

Status: **complete for `{ravdess, cremad}`.** `pytest` → 264 passed.
`results/runs.jsonl` holds **60 baseline rows** (3 floors × 4 pairs × 5 seeds),
all schema-valid; re-running is idempotent (0 written, 60 skipped).

### The sample-rate guard is now structural

RAVDESS ships at 48 kHz and CREMA-D at 16 kHz; every backbone expects 16 kHz.
Feeding 48 kHz audio to the model would **not error** — it would silently encode
speech running at a third of its true rate and produce plausible features. Timing
evidence said resampling was correct; it is now enforced in three places:

- `assert_target_sample_rate()` — the shared check.
- `load_audio()` — asserts after resampling, and refuses outright if
  `features.sample_rate` is unset, since loading at native rate would mix 48 kHz
  and 16 kHz audio in one feature space.
- `SSLExtractor.encode()` — rejects any rate other than the checkpoint's own
  `sampling_rate`, so a config change cannot quietly feed a model the wrong rate.

Four tests, including one that writes a real 48 kHz file and asserts it resamples
to 16 kHz with duration preserved, and one asserting `encode` raises on 48 kHz.

### Floors, computed per pair and per seed

All three are computed against the **realised `target_test` label distribution**,
never an assumed uniform one.

| pair | K | uniform | analytic | majority | stratified |
|---|---|---|---|---|---|
| ravdess → ravdess | 6 | 0.1629–0.1662 | 0.1656 | 0.0625 | 0.1649–0.1672 |
| ravdess → cremad | 6 | 0.1663–0.1669 | 0.1665 | 0.0424–0.0426 | 0.1644–0.1648 |
| cremad → ravdess | 6 | 0.1646–0.1665 | 0.1656 | 0.0444 | 0.1636–0.1650 |
| cremad → cremad | 6 | 0.1660–0.1664 | 0.1665 | 0.0486–0.0487 | 0.1662–0.1669 |

Closed forms were derived rather than assumed, and each is checked against a
1000-draw empirical estimate with a 95% CI:

- uniform: `F1_k = 2·p_k·(1/K) / (p_k + 1/K)`, which reduces to `1/K` for a
  balanced target — 0.167 at K=6, 0.250 at K=4.
- majority: only the predicted class scores, `F1_m = 2·p_m/(p_m+1)`, everything
  else 0 — ~0.048 at K=6, the collapse floor.
- stratified: `F1_k = 2·p_k·q_k / (p_k + q_k)` with `q` the source-train prior.

The analytic values are a ratio of expectations, not the expectation of a ratio,
so agreement with the empirical mean is a real check rather than a tautology. A
test asserts the analytic value falls inside the empirical CI.

**Two observations worth carrying into the paper.**

*Majority depends on the target, not just K.* The same constant predictor scores
0.0625 against a RAVDESS target but 0.0425 against CREMA-D, because the source
majority class is `neutral` — 23% of RAVDESS but 15% of CREMA-D. A single global
"majority floor" would be wrong for half the grid.

*Stratified ≈ uniform here*, because both corpora are near-uniform in the 6-class
space. That is a direct consequence of the near-zero prior shift established in
A8, showing up independently in a second measurement.

### Metrics

`src/ser/metrics.py`: macro-F1, accuracy, UAR, per-class F1, confusion matrix,
and the class-collapse count. Verified against sklearn (`f1_score`,
`accuracy_score`, `balanced_accuracy_score`, `confusion_matrix`) on synthetic
data.

Every function scores over **all** `class_names`, so a class the model never
predicts contributes 0 rather than vanishing from the average — dropping
unpredicted classes is exactly what makes a collapsed model look competent. UAR
is the one deliberate exception: it averages recall only over classes present in
`y_true`, since a class absent from the evaluation set has no recall to average
and counting it as 0 would penalise the split rather than the model.

### Statistics

`src/ser/stats.py`: percentile bootstrap CI over **test utterances** (not runs),
paired Wilcoxon signed-rank across matched (pair, seed) observations, and
Holm-Bonferroni with monotone adjusted p-values. Wilcoxon returns p=1 on
identical inputs rather than raising, which scipy does.

### Files created / modified

```
src/ser/metrics.py             macro-F1, accuracy, UAR, per-class, confusion, collapse
src/ser/baselines.py           three floors, analytic + empirical with CIs
src/ser/stats.py               bootstrap CI, Wilcoxon, Holm-Bonferroni
src/ser/baselinerun.py         per pair/seed -> results/runs.jsonl + report
src/ser/features/audio.py      assert_target_sample_rate, stricter load_audio
src/ser/features/ssl.py        expected_sample_rate + encode guard
src/ser/cli.py                 `ser baselines`
tests/test_metrics.py          26 tests
reports/baselines.md
```

### Deferred

- IEMOCAP pairs (K=4, chance 0.250) will be added when the corpus arrives; the
  code is already pair-generic and needs no change.
- No classifier is trained (Phase 6) and no alignment is fitted (Phase 5).

---

## 2026-08-15 — Phase 3: feature extraction and caching

Status: **complete for `{ravdess, cremad}`.** `pytest` → 238 passed.
`ser verify-cache` → **8/8 caches OK, 4.79 GB**. Re-running `ser extract` is a
confirmed no-op (0.0 min, every unit reported `cached`).

### What was extracted

All **13 hidden states** per utterance, not just the last — the design point that
makes layer aggregation a free experimental condition downstream instead of an
unexamined default. Mean-pooled and 8-segment-pooled come from a **single forward
pass** and are stored as separate arrays, so the segment cache stays skippable.

| corpus | backbone | n | wall | s/utt | size |
|---|---|---|---|---|---|
| ravdess | hubert | 1440 | 39.3 min | 1.637 | 258.8 MB |
| ravdess | wav2vec2 | 1440 | 39.9 min | 1.662 | 258.8 MB |
| ravdess | wavlm | 1440 | 42.2 min | 1.759 | 258.8 MB |
| ravdess | mfcc | 1440 | 1.0 min | 0.043 | 0.5 MB |
| cremad | hubert | 7442 | 159.4 min | 1.285 | 1337.6 MB |
| cremad | wav2vec2 | 7442 | 158.2 min | 1.275 | 1337.6 MB |
| cremad | wavlm | 7442 | 160.0 min | 1.290 | 1337.6 MB |
| cremad | mfcc | 7442 | 4.0 min | 0.032 | 2.5 MB |
| **total** | | | **604 min = 10.07 CPU-h** | | **4.79 GB** |

Wall clock was **~3.4 h**, not 10, because the three backbones ran as parallel
processes. RAVDESS costs more per utterance (1.64–1.76 s) than CREMA-D
(1.28–1.29 s) simply because its clips are longer — 3.7 s mean against 2.5 s.

Shapes: `layers (n, 13, 768)` and `segments (n, 13, 8, 768)`, float16;
`mfcc (n, 78)` float32. **No GPU on this machine** — torch is CPU-only, so these
are CPU numbers and a CUDA box would be far faster.

### Three implementation decisions

**Batch size 1, deliberately.** Batching needs padding, and a padded frame that
reaches the mean corrupts the pooled vector silently, worst for the shortest
utterances. Masking fixes that in principle, but `facebook/wav2vec2-base` is
documented as degrading under masked batched inference — it was pretrained
without an attention mask and its feature extractor sets
`return_attention_mask=False`. Treating one backbone differently from the other
two would put an unmeasured confound directly into the backbone comparison.
Wall time was recovered with process-level parallelism instead, which has no
numerical consequences at all.

**Cache keys are per corpus**, a deliberate deviation from the brief's
`sha256(manifest_rows)` over the whole manifest. The intent is unchanged — a
cache is keyed by exactly the rows it covers, including each row's audio sha256 —
but adding IEMOCAP later then costs only IEMOCAP's extraction instead of
invalidating 4.79 GB. Writes are atomic through a staging directory, so a killed
run leaves nothing or a whole cache, never a half-written one that would later
read as valid. A key hit is never overwritten.

**`weighted` returns the unreduced stack.** Layer weights are learnable
parameters owned by the classifier (Phase 6); baking them into the cache would
turn them into a preprocessing constant and remove the very thing being measured.

### A platform trap, resolved without touching numerics

Extraction aborted with `OMP: Error #15` — conda's numpy+MKL and pip's torch each
ship `libiomp5md.dll`. The documented workaround, `KMP_DUPLICATE_LIB_OK=TRUE`, is
described by Intel as able to "silently produce incorrect results", which is not
a trade this project can make.

Traced instead: the clash only occurs when torch's OpenMP initialises **before**
librosa's; the reverse order coexists fine, including MFCC calls made after torch
is loaded. `warm_up_audio_stack()` now initialises librosa first.

The first fix still aborted, because it warmed only `librosa.feature.mfcc` and
not `librosa.feature.delta`, which routes through scipy and links its own OpenMP
separately. Warming the **full** path fixed it. Import ordering only — no
numerical effect. Recorded in PHASES.md Phase 3.

### Verification

`tools/verify_cache.py` / `ser verify-cache` asserts row count, finiteness,
shapes, and that **utterance ordering matches the manifest exactly**. Ordering is
the silent failure: every split and label lookup assumes row *i* of the cache is
row *i* of the corpus, and a reordering would misalign features and labels
everywhere while still producing plausible numbers.

Four further tests run against the real caches and catch what shape and
finiteness checks cannot — features that are well-formed but wrong:

- **The 13 layers are distinct states.** Storing one hidden state 13 times would
  pass every shape assertion while making the entire layer axis meaningless.
  Measured: adjacent layers differ by 0.04–0.17, and layer norms grow with depth
  (HuBERT 2.3 → 3.0 → 5.5).
- **`mean(segments)` reconstructs `layers` to 0.0005** (float16 rounding), which
  is what proves both poolings came from the same forward pass.
- Two backbones differ (HuBERT vs wav2vec2: 0.107).
- Cache row order equals manifest row order.

### Files created

```
src/ser/features/__init__.py
src/ser/features/audio.py      fixed preprocessing + the OpenMP warm-up
src/ser/features/cache.py      per-corpus keys, atomic writes, metadata
src/ser/features/ssl.py        per-layer mean + segment pooling, one pass
src/ser/features/mfcc.py       78-dim, documented slice layout
src/ser/features/aggregate.py  last | layer:k | mean:a-b | weighted
src/ser/features/extract.py    driver, skip-on-hit
src/ser/features/verify.py     the four assertions
tools/verify_cache.py
tests/test_features.py         33 tests
src/ser/cli.py                 `ser extract`, `ser verify-cache`
```

### Deferred

- IEMOCAP: not extracted, not acquired. Per-corpus keys mean it will cost only
  its own extraction.
- `data/cache/` is gitignored — 4.79 GB of derived data. Rebuild with
  `ser extract`; expect ~3.4 h wall on a 16-core CPU box, minutes on a GPU.
- Standardisation is deliberately **not** applied at extraction time. It is a
  Phase 5 experimental condition (`zscore`), and baking it in would make the
  `none` rung of the ladder unmeasurable.

---

## 2026-08-12 — Session 3: Phase 2 splits and the leakage assertion suite

Status: **complete for `{ravdess, cremad}`.** `pytest` → 205 passed.
`ser splits` → 20 splits (4 pairs × 5 seeds), **all leakage assertions pass**.

### The leakage suite, built before anything else

Three of the brief's four assertions are in `src/ser/leakage.py`; the fourth
(`map_label` purity) was asserted in Session 2. All run over *every* pair and
seed, not a sample.

1. **Speaker disjointness** — no speaker or session in two roles of a corpus.
2. **No utterance on both sides** — nothing in a source split and a target split.
3. **`target_test` never reaches a fitted alignment.** This is a **contract on
   Phase 5**, not an option: `assert_alignment_blind_to_target_test` rejects an
   object that has no `fitted_on_indices` at all, so an alignment that cannot be
   checked cannot be used. It also rejects fitting on anything outside the split.
4. `map_label` purity — `tests/test_labelmap.py`.

Each assertion has a paired test proving it **fails** on a violating input. An
assertion never observed to fail proves nothing, and this suite exists precisely
because the original study's equivalent checks did not exist.

### In-domain pairs needed a four-way partition

Running the source and target splits independently over one corpus would place
the same speakers in `source_train` and `target_test`, so every in-domain number
would silently report training data. Instead an in-domain pair divides the corpus
into a source side and a target side first (`splits.in_domain_source_ratio`, new
config key), then carves the roles out within each side. Tested explicitly, with
a companion test that the assertion catches the naive version.

Splits are keyed on `(seed, corpus, side)` rather than a shared RNG stream, so a
corpus gets the **same source-side partition in every pair where it is the
source**. Results stay comparable across targets, and adding a pair does not
perturb existing ones. There is a test for that too.

### A9 measured — and the prediction is wrong for these corpora

A9 predicted split-level prior KL would be *larger* than corpus-level and would
*vary across seeds*. Measured over all 20 splits:

| pair | corpus-level | split-level mean | spread over 5 seeds |
|---|---|---|---|
| ravdess → cremad | 0.0252 | 0.0251 | 0.0003 |
| cremad → ravdess | 0.0224 | 0.0224 | 0.0000 |
| ravdess → ravdess | 0.0000 | 0.0000 | 0.0000 |
| cremad → cremad | 0.0000 | 0.0000 | 0.0000 |

Verified as real, not a degenerate split: group sets are disjoint, sizes differ
by seed, and CREMA-D's in-domain KL is 1e-6 rather than exactly 0.

The cause is structural. Both are **acted corpora with a fixed per-actor
recording protocol** — every RAVDESS actor records the same 60 trials, every
CREMA-D actor the same sentence × emotion grid — so class proportions are
invariant under any speaker-disjoint partition. RAVDESS is exactly 0 by
construction.

A9's *reasoning* was about IEMOCAP's non-uniform per-session distributions, which
remains plausible and **remains untested**. What is now known is that it does not
generalise to the acted corpora. PHASES.md A9 has been annotated with the
measurement, and the procedural requirement — report both columns — is kept,
since reporting both is what made this checkable.

**For the thesis:** prior shift on these pairs is not merely near-zero at corpus
level, it is *invariant to splitting*. The label-shift explanation is dead for
RAVDESS↔CREMA-D in a stronger sense than A8 established.

### Files created / modified

```
src/ser/splits.py        Split/PairSplit, deterministic speaker-disjoint splits,
                         four-way in-domain partition
src/ser/leakage.py       the assertion suite + the Phase 5 alignment contract
src/ser/splitreport.py   `ser splits` report, split-level priors per seed
src/ser/config.py        splits.in_domain_source_ratio
configs/default.yaml     same
src/ser/cli.py           `ser splits`
tests/test_leakage.py    15 tests
reports/splits.md
```

### Deferred

- IEMOCAP: session-level splitting is implemented and config-driven
  (`grouping_key_for`) but **untested against data** — no IEMOCAP on disk.
- The A9 seed-variance question is open until IEMOCAP lands.
- Phase 2's `reports/dataset_stats.md` currently covers two corpora; it must be
  regenerated when IEMOCAP arrives.

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
