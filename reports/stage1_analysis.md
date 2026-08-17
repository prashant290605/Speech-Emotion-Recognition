# Stage 1 — screening analysis

Companion to [stage1_screening.md](stage1_screening.md). That file is
`source_val` only, because it is the surface every pruning decision was made on.
This file contains target scores as well, all of them labelled as
**observations, not results** — Stage 1 is two seeds, one pair, one backbone,
and pre-selection. No claim below goes into the paper on Stage 1 evidence.

---

## 1. Integrity

| | |
|---|---|
| rows in `results/runs.jsonl` | 438 |
| Phase 4 baselines | 60 |
| Stage 0 smoke gate (`grid-freeze-v1`) | 6 |
| Stage 1 (`grid-freeze-v2`) | 372 |
| **failed** | **0** |

372 = 360 screening (4 families) + 12 transformer probe. Every enumerated run
committed a row with `status="ok"`. No tracebacks, so there is no systematic
failure pattern to report.

### Nulls, and why they are correct

Two columns contain nulls. Neither is a defect:

| column | null rows | why |
|---|---|---|
| `marginal_mmd_reference` | 6 | exactly the Stage 0 rows, which predate the column (Item B) |
| `mmd_fallback_fired` | 159 | 6 Stage 0, plus every rung with no warm start to fall back to |

The 153 Stage 1 nulls in `mmd_fallback_fired` are `coral` 96, `none` 21,
`zscore` 18, `mean_shift` 18. Those are precisely the rungs that do not run an
optimiser, so "did it revert to the warm start" is undefined for them. The
MK-MMD rungs, where the column is meaningful, have **no** nulls. Aggregations
over these columns must filter, not impute.

### Timings, measured against projection

Per-run wall seconds under 4-shard contention. The projection was made from
solo runs, so the ratio is contention, not a modelling error.

| family | projected (solo) | measured (4 shards) | ratio |
|---|---|---|---|
| logreg | 34 s | 84 s | 2.5× |
| svm_linear | 28 s | 133 s | 4.7× |
| svm_rbf | 22 s | 65 s | 3.0× |
| mlp | 179 s | 323 s | 1.8× |
| transformer | 1086 s | 2389 s | 2.2× |

Per condition, median seconds:

| family | `last` | `layer:6` | `weighted` |
|---|---|---|---|
| logreg | 56 | 87 | — |
| svm_linear | 75 | 66 | — |
| svm_rbf | 53 | 55 | — |
| mlp | 236 | 230 | 309 |
| transformer | 3374 | 1171 | 1633 |

**Total 25.1 CPU hours, ~6.3 h wall across 4 shards.** The transformer probe is
12 of 372 runs and 20% of the CPU time; that ratio is what forces the Stage 2
transformer arm to be reduced.

---

## 2. Pruning decisions

All decisions on `source_val`, within each `(seed, classifier, layer_agg)`
condition, then cross-checked against the 12 transformer probe runs.

### Protected — not pruned, by instruction

The alignment ladder, layer aggregation, and backbone are kept at full width
regardless of what screening shows. Screening did in fact show the ladder to be
nearly flat on target when pooled (§3), and it is still kept whole: it is the
dose-response axis the central claim rests on, and a dose-response axis with
rungs removed is not one.

### `alignment_eps` (CORAL) — dropped `1e-3`

Win rate as `source_val` argmax within its own condition:

| eps | wins |
|---|---|
| 1e-1 | **18/18 (100%)** |
| 1e-4 | 0/18 |
| 1e-3 | 0/18 |
| 1e-2 | 0/18 |
| Ledoit-Wolf | 0/18 |

`source_val` is **monotone increasing in eps** — 0.5592, 0.5713, 0.6032,
0.6523 — so only the largest value is ever selected. That is not a licence to
keep only the winner: a surface monotone to the boundary means the grid is
mis-centred, and the retained points are what make the shape reportable.

Dropped **1e-3** only. It is interior, bracketed by 1e-4 and 1e-2, and its
`source_val` sits between theirs, so it carries no shape the neighbours do not.
Retained: 1e-4 (the endpoint the "unregularised CORAL" reading rests on), 1e-2,
1e-1, and Ledoit-Wolf.

Ledoit-Wolf is retained despite never winning, because "does the standard
automatic shrinkage find the good regime?" is a real question and the answer
here is **no** — it lands at 0.6096, nearer 1e-2 than 1e-1.

**Transformer cross-check:** the probe tested 1e-4 and 1e-1 at all three
aggregations. eps=1e-1 is higher on `source_val` in **3/3** (0.7453→0.7994,
0.8014→0.8411, 0.8350→0.8361). The direction agrees. The probe does not test
1e-3 itself, so it cannot exclude a dip there for the transformer; given the
monotone trend at both endpoints and 1e-3's interior position, the risk is
low — but it is an inference, not a measurement, and is recorded as such.

### `alignment_lambda` (MK-MMD) — dropped `10.0` and `100.0`

`source_val` is **flat** in lambda: spread 0.0016 across the whole grid for
`mkmmd_diag` (0.7418–0.7434) and 0.0054 for `mkmmd_full` (0.5605–0.5659). The
argmax is scattered accordingly (diag: 11/17/33/22/11/6%; full:
22/39/11/11/11/6%), which is what a flat surface with random tie-breaking looks
like. **No lambda is distinguishable on score.**

So the drop is made on mechanism instead. `mkmmd_full`'s fallback rate rises
with lambda:

| lambda | 0.001 | 0.01 | 0.1 | 1.0 | 10 | 100 |
|---|---|---|---|---|---|---|
| `mkmmd_full` reverted to warm start | 8/18 | 8/18 | 10/18 | 14/18 | 14/18 | 14/18 |
| `mkmmd_diag` reverted to warm start | 5/18 | 5/18 | 5/18 | 5/18 | 5/18 | 5/18 |

At λ ≥ 1 the ‖W−I‖²_F penalty dominates, the optimiser cannot improve on its
CORAL warm start, and the cell reverts to it in 14 of 18 runs. Those cells are
not a sixth rung — they are CORAL with extra wall time. Retained 0.001, 0.01,
0.1, and 1.0, keeping 1.0 as the anchor that shows where the fallback
saturates.

**Transformer cross-check: NOT AVAILABLE.** The probe ran `mkmmd_full` at
λ=0.001 only. It cannot adjudicate any lambda decision. The drop rests on
sklearn + MLP plus the fallback mechanism, and that limitation is stated rather
than hidden. Nothing in the probe contradicts it.

### `blend_alpha` — NOT PRUNED, because it was never screened

Stage 1 ran `blending="none"` on all 372 runs. The blending axis
(`alpha_grid` = [0.0, 0.25, 0.5, 0.75, 1.0], `modes` = [none, scalar, gaa]) was
never varied, so **there is no Stage 1 evidence about it and it cannot be
pruned on this evidence.** It is also not affordable at full width in Stage 2:
11 distinct blending settings would multiply a 33 h grid by roughly 11.

Recorded as an open item, not silently skipped. It needs its own small
screening pass before it can enter any factorial. A test
(`test_stage1_blending_axis_was_never_screened`) pins this so a later reader
cannot assume it was screened because every other inner grid was.

### Classifier hyperparameter ranges — NOT PRUNED

The equal-tuning-budget contract is 20 random-search trials per (family,
condition), asserted identical across all five families. Narrowing a family's
range on screening evidence would break that symmetry — the budget would buy
more effective coverage for the narrowed family. Left at full width
deliberately.

One observation for Stage 2 rather than a pruning decision: with CREMA-D as
source (5972 training utterances) logreg at C=100 hits the `max_iter=1000` cap
(84.8 s, `n_iter=1000`) while at 988 it converges in 287 iterations. Capped
trials score worse on `source_val` and lose selection, so this costs wall time
rather than correctness — but it should be stated wherever reverse-direction
numbers are reported.

---

## 3. Required analysis — CORAL `source_val` against `alignment_eps`

The Item D question was: is CORAL's 0.166 `source_val` cost a statement about
CORAL, or about the regularisation? **It is substantially about the
regularisation.**

| eps | source_val | effect (own) | effect (reference) | target |
|---|---|---|---|---|
| 1e-4 | 0.5592 | 5.11 | 10.29 | 0.3982 |
| 1e-3 | 0.5713 | 5.72 | 13.05 | 0.3930 |
| 1e-2 | 0.6032 | 8.57 | 32.42 | 0.3916 |
| 1e-1 | **0.6523** | 19.88 | 120.76 | 0.4012 |
| Ledoit-Wolf | 0.6096 | 9.79 | 44.11 | 0.3894 |
| *(none)* | *0.7421* | *925.38* | *31438.35* | *0.2948* |
| *(zscore)* | *0.7473* | *33.11* | *276.99* | *0.3963* |

`source_val` recovers from 0.5592 to 0.6523 — **0.093 of the 0.166 gap, about
56% — while target moves 0.0096 across the whole range** (0.3894 to 0.4012,
i.e. nothing).

The paper must therefore say: **most of CORAL's source-domain cost is a
shrinkage artefact, not a property of CORAL.** It is not all of it — 0.089
remains at the strongest shrinkage tested — and the surface is still climbing
at the grid boundary, so the true recoverable fraction is a lower bound.

Extending the eps grid upward would settle it, but that requires editing
`configs/default.yaml`, which is frozen at `grid-freeze-v2`. Doing so changes
`search_spec_hash` and orphans all 372 Stage 1 rows. **Not done unilaterally —
flagged for decision.** Stage 1's rows would remain valid as a subset under a
new tag; they would simply no longer resume.

---

## 4. Required analysis — effect size against target macro-F1

> **OBSERVATION, NOT A RESULT.** Two seeds, pre-selection, one pair, one
> backbone. No claim from these target numbers goes into the paper.

| rung | n | effect (own) | effect (reference) | source_val | target |
|---|---|---|---|---|---|
| none | 18 | 925.38 | 31438.35 | 0.7421 | 0.2948 |
| zscore | 18 | 33.11 | 276.99 | 0.7473 | 0.3963 |
| mean_shift | 18 | 54.42 | 15685.61 | 0.7389 | 0.4053 |
| coral | 90 | 9.82 | 44.13 | 0.5991 | 0.3947 |
| mkmmd_diag | 108 | 38.40 | 900.99 | 0.7425 | 0.4043 |
| mkmmd_full | 108 | 6.63 | 21.41 | 0.5622 | 0.3956 |

Two things to note, and one trap.

**The step off `none` is the only step that moves target.** `none` → anything
is +0.10. Across the further 140× reduction in own-geometry discrepancy
(925 → 6.6), target spans 0.3947 to 0.4053 — a range of 0.011, which at two
seeds is noise.

**The two geometries rank the rungs differently.** `mean_shift` is 54× null in
its own frame and 15686× in the reference frame; `mkmmd_diag` is 38× and 901×.
The reference frame is far more sensitive to uncorrected mean offsets, because
ZCA-whitening by the source covariance leaves a translation fully visible while
a per-rung median bandwidth adapts to it. Neither column is the truth. They
answer different questions and the paper must report both and rank on neither
alone.

**The trap: the pooled ladder average hides a large interaction.** Splitting the
same 360 runs by layer aggregation:

| aggregation | n | target, `none` | target, aligned | gain from alignment |
|---|---|---|---|---|
| `last` (layer 12) | 160 | 0.3405 | 0.3823 | **+0.042** |
| `layer:6` | 160 | 0.2375 | 0.4056 | **+0.168** |
| `weighted` | 40 | 0.3413 | 0.4360 | **+0.095** |

Alignment is worth four times as much at mid-stack as at the top layer, and the
sign is stable in every family (logreg +0.19, svm_linear +0.20, svm_rbf +0.15,
mlp +0.12 at `layer:6`; and at `last`, mlp is **−0.009** and the transformer
probe **−0.017**). The independent 12-run transformer probe reproduces the
pattern (`last` −0.017, `layer:6` +0.084, `weighted` +0.052).

So "the ladder does nothing" is an artefact of averaging over depth. What the
ladder does depends on which representation it is applied to. This is the
hypothesis Stage 2 exists to test, and it is why the layer-aggregation axis is
worth protecting.

---

## 5. Required analysis — `mmd_fallback_fired` per rung

| rung | runs | fired | rate |
|---|---|---|---|
| `mkmmd_diag` | 108 | 30 | **27.8%** |
| `mkmmd_full` | 111 | 68 | **61.3%** |

`mkmmd_full` reverts to its CORAL warm start in the majority of runs. **Any
table containing an `mkmmd_full` row has to say this**, because in those cells
the rung is not MK-MMD — it is CORAL, and reporting it as a distinct sixth rung
would overstate what was actually fitted.

The diag rate is exactly 5/18 at every lambda, i.e. entirely independent of the
regularisation strength. It is feature-dependent, consistent with the earlier
finding that the diagonal warm start is already at 1.93× null on `layer:6` and
so cannot be improved on there, while on `last` it never fires.

---

## 6. Required analysis — the full 13-layer sweep

Full tables in [layer_sweep.md](layer_sweep.md). 78 cells (3 backbones × 13
layers × 2 seeds), logreg, rung `none`, from the existing cache — ~24 min
total.

The append-only sweep file holds 101 rows, not 78: a restarted worker
recomputed 23 wavlm cells. **All 23 agreed to floating-point equality with the
originals**, which is an unplanned determinism check on the whole path from
cache load through random search to scoring. The report generator collapses
duplicates and *raises* rather than averaging if any pair disagrees, since a
disagreement would be non-determinism, not a duplicate.

**Harness validation first.** At the `none` rung the sweep reproduces the Stage
1 grid to within RNG noise: hubert seed 0, layer 12 gives 0.3700 here against
0.3698 in the grid; layer 6 gives 0.2192 against 0.2162. Different code path,
same numbers.

### The result

| backbone | argmax `source_val` | argmax target | gap | cost of choosing depth on `source_val` |
|---|---|---|---|---|
| hubert | 6 | 11 | 5 | **−0.1436** |
| wav2vec2 | 5 | 9 | 4 | **−0.1405** |
| wavlm | 6 | 10 | 4 | **−0.1282** |

**The depth that maximises in-domain validation is 4–5 layers shallower than
the depth that maximises cross-corpus transfer, in all three backbones, and
choosing on `source_val` costs 0.13–0.14 macro-F1.**

### The hypothesis this was meant to test did not hold

The expectation was a monotone curve — discrepancy falling toward the middle of
the stack with target macro-F1 falling with it. That is not what the data shows.
Discrepancy does fall toward mid-stack for hubert and wavlm, but is flat-to-
rising for wav2vec2, and target does not track it in any backbone:

| backbone | ρ(effect own, target) | ρ(effect reference, target) | ρ(source_val, target) |
|---|---|---|---|
| hubert | **−0.769** | **+0.692** | +0.104 |
| wav2vec2 | +0.445 | +0.341 | +0.396 |
| wavlm | **−0.654** | **+0.670** | +0.527 |

**The two discrepancy columns disagree in sign on two of three backbones.** In
each rung's own geometry, less discrepancy goes with better transfer; in the
fixed ZCA reference frame, more discrepancy does. Same features, same layers,
same target scores — opposite conclusions from the choice of measurement frame
alone. Any claim of the form "lower MMD implies better transfer" must name its
geometry and defend it, and neither frame is singled out by theory.

That is a stronger and more useful finding than the monotone curve would have
been, and it is a caution that applies to the alignment ladder as well.

### Consequence for the recorded `layer:6` finding

The PROGRESS.md finding — "`layer:6` has lower discrepancy than `last` and
transfers worse" — **half survives and half does not.**

| backbone | source_val @6 | @12 | target @6 | @12 | effect @6 | @12 |
|---|---|---|---|---|---|---|
| hubert | 0.7774 | 0.7036 | 0.2163 | 0.3383 | 954 | 894 |
| wav2vec2 | 0.7452 | 0.6340 | 0.1533 | 0.1482 | 1658 | 1540 |
| wavlm | 0.7734 | 0.6900 | 0.1760 | 0.2524 | 1088 | 1171 |

* **Survives:** `layer:6` scores higher in-domain in all three backbones.
* **Survives, weakened:** it transfers worse in hubert (−0.122) and wavlm
  (−0.076) — but on wav2vec2 the two are tied (+0.005).
* **Does not survive:** "lower discrepancy". Under the corrected fixed-bandwidth
  MMD, `layer:6` has *higher* own-geometry discrepancy than `last` on hubert and
  wav2vec2, and lower only on wavlm. The dose-response contradiction the finding
  was built on does not reproduce.

The finding block has been rewritten accordingly. The curve replaces the
two-point contrast, and the claim it supports is the `source_val`/target depth
divergence, not a discrepancy argument.

---

## 7. Stage 2 configuration

### Design

| axis | Stage 1 | Stage 2 |
|---|---|---|
| direction | ravdess→cremad | **ravdess→cremad** (see below) |
| backbone | hubert | **hubert, wav2vec2, wavlm** |
| seeds | 0, 1 | **0–4** (transformer 0, 1) |
| ladder | 6 rungs | 6 rungs (protected) |
| layer_agg | last, layer:6, weighted | unchanged (protected) |
| coral eps | 5 | **4** (dropped 1e-3) |
| mkmmd lambda | 6 | **4** (dropped 10, 100) |
| blending | none | none (never screened) |

**2133 runs, 33.0 h wall at 4 shards**, projected from measured Stage 1
per-cell medians. 276 of those runs are already complete — the Stage 1 cells
that survived pruning resume rather than recompute, verified by dry run.

| family | runs | CPU hours | wall hours |
|---|---|---|---|
| transformer | 108 | 60.8 | 15.2 |
| mlp | 675 | 47.1 | 11.8 |
| logreg | 450 | 8.7 | 2.2 |
| svm_linear | 450 | 8.6 | 2.1 |
| svm_rbf | 450 | 6.9 | 1.7 |
| **total** | **2133** | **132.1** | **33.0** |

The transformer arm is 5% of the runs and 46% of the CPU. It is reduced to 2
seeds and one inner-grid setting per rung (`coral` at eps=1e-1, MK-MMD at
λ=0.01) — reductions on seeds and inner grids only, never on a protected axis.
It keeps all six rungs, all three backbones and all three aggregations, and
**must be reported as a reduced-seed arm with wider intervals and not pooled
with the five-seed families.**

### The reverse direction does not fit, and this is a decision to make

`cremad→ravdess` has 5972 source-train utterances against 988 — 6.0×. Measured
single-fit cost (hubert, `last`, logreg): 4.1 s → 35.6 s at C=1, 5.1 s → 84.8 s
at C=100. **~16× per fit.** The MMD side is not the problem — it subsamples and
costs 0.7 s in both directions.

| reverse-direction option | runs | wall hours (factor 8 / 16) |
|---|---|---|
| full mirror (3 backbones, 5 seeds, + transformer) | 2133 | 264 / **528** |
| hubert only, 5 seeds, + transformer | 711 | 88 / 176 |
| hubert only, 2 seeds, no transformer | 270 | 19 / **38** |
| hubert only, 2 seeds, logreg + mlp | 150 | 15 / 30 |

A full mirror is impossible against a 72 h budget by a factor of 7. Forward-only
Stage 2 (33 h) plus a reduced reverse arm (19–38 h) totals 52–71 h and fits,
but only just, and the 8×–16× spread is the uncertainty.

**Recommendation: launch forward-only Stage 2 now (33 h, fits with certainty),
and treat the reverse arm as a separate later launch** once
`tools/calibrate_stage2.py` has produced measured per-family factors. Transfer
asymmetry is a real finding and worth having; it is not worth risking the arm
every table depends on. The direction axis is neither protected nor an inner
grid, so this is a scoping call, not a pruning decision — flagged for the
supervisor rather than taken silently.

### Harness

Same design as Stage 1, verified the same way:

* `tools/launch_stage2.ps1` — 4 shards, 4 BLAS threads each, base64-encoded
  inner script, `Start-Process` with file redirection, no pipes anywhere.
* Two passes per shard, sklearn + MLP **before** transformer, so an interrupted
  run still leaves a complete reportable arm rather than five partial ones.
* Hash-based sharding (`int(run_id[:8], 16) % n_shards`), so shard membership
  survives any enumeration reordering.
* 4266 enumerated ids for the two-direction design, 2133 for one — **all
  unique**, asserted in `test_stage2_run_ids_are_distinct`.
* Resume verified by dry run: 276 already-complete rows skipped.
* `tools/merge_shards.py` refuses (exit 2) on any run_id appearing in two shards
  with differing content.
