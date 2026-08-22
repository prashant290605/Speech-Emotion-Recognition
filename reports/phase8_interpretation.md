# Phase 8 pass 2 — interpretation

Every number referenced here is in [phase8_tables.md](phase8_tables.md). Nothing
new is computed in this file.

---

## 1. Primary comparisons: survives or does not survive

Fourteen tests, declared before computation, Holm-corrected across the family.
All fourteen have Holm-adjusted p < 0.007, which is the bootstrap's resolution
floor rather than a measured value.

| id | comparison | ravdess→cremad | cremad→ravdess |
|---|---|---|---|
| L1 | zscore − none | **survives** (+0.1310) | **survives** (+0.1961) |
| L2 | mean_shift − none | **survives** (+0.1306) | **survives** (+0.1817) |
| L3 | coral − none | **survives** (+0.1359) | **survives** (+0.1833) |
| L4 | mkmmd_diag − none | **survives** (+0.1208) | **survives** (+0.1852) |
| L5 | mkmmd_full − none | **survives** (+0.1251) | **survives** (+0.1524) |
| A1 | layer:6 − last | **survives** (+0.0640) | **survives** (+0.1439) |
| A2 | weighted − last | **survives** (+0.0597) | **survives** (+0.1096) |

14 of 14 survive.

---

## 2. Is target macro-F1 flat across the alignment ladder?

**No. And the ladder is not a dose-response either. It is a step followed by a
plateau.**

Both halves of that need stating separately, because the project has been wrong
about each of them at different times.

**The step is real and large.** Moving from `none` to *any* aligned rung is
worth +0.12 to +0.20 macro-F1, in both directions, at 5 seeds, Holm-corrected.
This is not a null result and must not be written as one.

**The plateau is real and the ladder does not climb it.** Among the five aligned
rungs the largest difference is 0.0151 forward and 0.0437 reverse — roughly an
order of magnitude smaller than the step. Some of those intervals exclude zero,
so the plateau is not perfectly flat; but its variation is not ordered by
discrepancy, which is what the ladder was built to test:

* `mkmmd_full` achieves by far the lowest discrepancy in **both** frames
  (12.04 own / 23.25 reference, forward) — 4× lower than `zscore` in its own
  geometry and 20× lower in the reference frame.
* It is not the best rung anywhere. Forward it is indistinguishable from
  `zscore` (−0.0059 [−0.0139, +0.0021]); reverse it is the **worst** aligned
  rung (−0.0437 [−0.0559, −0.0320] against `zscore`).
* `zscore` — per-dimension centring and scaling, the cheapest rung on the
  ladder — is at or above every more sophisticated rung in both directions.

So the honest statement is: **almost all of the benefit attributed to
distribution alignment in this literature is obtained by z-scoring, and the
additional moments matched by CORAL and MK-MMD buy nothing measurable — while
reducing marginal discrepancy by one to two orders of magnitude.** The
dose-response prediction fails not because target performance is flat, but
because it saturates immediately and then declines slightly as discrepancy
keeps falling.

That is a stronger and more specific claim than "alignment does not help", and
it survives at full seed count with paired intervals.

---

## 3. Selection is the bigger problem

Not a primary comparison, so no significance is claimed. The intervals are
reported and the pattern is described.

The validated configuration reaches 0.3156 [0.1880, 0.4432] forward against an
oracle 0.4555 [0.4326, 0.4785]; a gap of 0.1399 [0.0231, 0.2568]. Reverse the
gap is 0.0687 [0.0090, 0.1283].

The mechanism is visible in the tables: **`source_val` is nearly blind to the
ladder.** Forward, `none` scores 0.7290 on `source_val` and `zscore` 0.7322 — a
0.003 separation on the selection surface for a 0.13 separation on target. The
selection therefore picks `none` on 2 of 5 seeds forward, returning target
0.2803 and 0.1508, which is what makes the validated interval four times wider
than the oracle's.

This matters for how the paper frames its own contribution: an alignment step
that helps by 0.13 is worth little if the standard model-selection protocol
cannot see that it helps.

---

## 4. What does NOT replicate

Five items. Two of them weaken findings this project previously promoted.

### 4.1 Stage 0: "target macro-F1 is flat across 226× of marginal discrepancy" — REFUTED

The single-seed Stage 0 reading was wrong, as was already suspected when it was
retracted once for over-reading. At full seed count the alignment step is
+0.0836 forward and +0.1280 reverse **at `last`**, the exact condition Stage 0
measured. The flatness was an artefact of one seed.

### 4.2 FINDING (b), frame dependence — DOES NOT REPLICATE on the ladder axis

This is the important one, because I promoted it to the top of PROGRESS.md as
"the methodological contribution".

The layer sweep found ρ(effect, target) = −0.77 in the rung's own geometry and
**+0.69** in the reference frame — opposite signs on two of three backbones.
Re-measured across the six ladder rungs at full seed count, the two frames
**agree**: −0.200 / −0.200 forward, and −0.371 / −0.086 reverse. Same sign in
both directions.

The finding is therefore supported by **one axis, not two**. The 13-layer sweep
that produced it was a Stage 1 artefact (2 seeds, logreg, one pair, one rung)
and **was not re-run at Stage 2**, so it is neither confirmed nor refuted — it
is simply still resting on the evidence it always rested on, which is weaker
than the framing it was given.

The claim must be demoted accordingly: frame dependence is a real observation on
the layer axis, not an established general property. Promoting it to the paper's
headline methodological contribution is not currently supported. Re-running the
sweep at full seed count across the ladder is the experiment that would settle
it, and it is cheap.

### 4.3 MK-MMD fallback rate — DOES NOT REPLICATE

`mkmmd_diag` reverted to its warm start in 27.8% of Stage 1 runs and **64.9%**
of Stage 2 runs. Stage 1 further reported the diag rate as *exactly* 5/18 at
every λ — "entirely independent of the regularisation strength". At Stage 2 it
rises with λ, 55.9% → 68.9%. Both the level and the claimed λ-independence fail.

The `mkmmd_full` rate roughly holds (61.3% → 55.2%) and its strong λ dependence
replicates (43.0% → 87.0%).

The consequence for the paper is unchanged and now larger: in the majority of
`mkmmd_*` cells the fitted map is the CORAL warm start, so those rows must not be
presented as a distinct rung without saying so.

### 4.4 The extended eps grid did not fix the mis-centring

Stage 1 found `source_val` monotone in eps with the grid maximum (1e-1) winning
18/18, and the grid was extended to 1.0 and 10.0 at `grid-freeze-v3` for exactly
that reason. At Stage 2 the surface is **still monotone and the argmax is again
the grid maximum**, now 10.

Extending the grid moved the number without fixing the problem. The reviewer
critique that prompted the change — "the selected hyperparameter sat on the edge
of the search range" — still applies. Either the grid needs extending again
until it turns over, or the paper states plainly that CORAL's `source_val` is
monotone in shrinkage over four orders of magnitude and that the shrinkage is
doing regularisation work unrelated to domain alignment. The second reading is
better supported: at eps=10 the covariance correction is almost entirely
suppressed, and `source_val` is highest there.

### 4.5 Depth divergence — NOT TESTED at full seed count

The 13-layer curve is carried into the tables unchanged from Stage 1 and is
still 2 seeds, one classifier, one pair, one rung. It is neither confirmed nor
refuted.

What *is* tested: the aggregation axis at full seed count agrees with its shape.
`source_val` prefers `weighted` in both directions while target prefers
`layer:6` — **they disagree, in both directions**. And A1/A2 confirm that
`last`, the aggregation the original study used, is the worst of the three by
+0.06 to +0.14.

So the *claim* "selecting depth on in-domain validation picks the wrong depth
for transfer" holds on the axis Stage 2 measured. The specific "4–5 layers
shallower" figure does not have Stage 2 support and should be reported as a
Stage 1 observation with its seed count attached.

---

## 5. What does replicate

* **The alignment × aggregation interaction.** Stage 1 gains of +0.042 / +0.168
  / +0.095 (last / layer / weighted) come back as +0.0836 / +0.1742 / +0.0973.
  Ordering preserved, magnitudes close except at `last`, which roughly doubled.
* **λ is flat on `source_val`.** Spread 0.0066 (diag) and 0.0196 (full), still
  an order of magnitude below every other axis. The Stage 1 pruning decision
  that dropped λ=10 and 100 on mechanism rather than score was sound.
* **`mkmmd_full`'s λ-dependent fallback**, as above.
* **Zero failures, zero non-converged trials** across 4986 runs and 99,720
  search trials, at a solver cap of 20000 with a maximum observed 10793.

---

## 6. Consequences for the manuscript

1. The central result is **not** "alignment does not help". It is that the
   benefit saturates at the cheapest rung and does not track discrepancy
   thereafter. Write it that way.
2. `zscore` is the recommendation that falls out of the ladder, and it is worth
   stating that a per-dimension linear rescaling matches everything more
   elaborate that was tried.
3. Frame dependence gets demoted from headline to observation, with its
   two-seed provenance stated, unless the sweep is re-run.
4. Every `mkmmd_*` number carries its fallback rate.
5. The validated-vs-oracle gap and the `source_val` blindness to the ladder
   belong in the paper as a result, not a limitation — they are the strongest
   evidence that the field's selection protocol is the weak link.
