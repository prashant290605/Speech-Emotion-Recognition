# Phase 9 pass 2 — interpretation

Numbers in [phase9_tables.md](phase9_tables.md). Nothing new is computed here.

---

## 1. The decomposition

| term | measured as | ravdess→cremad | cremad→ravdess |
|---|---|---|---|
| **label shift** | KL(P_tgt ‖ P_src) between realised priors | 0.0224 nats (0.032 bits) | 0.0254 nats (0.037 bits) |
| **covariate shift** | marginal MMD / null, unaligned | 1496× | 949× |
| **conditional shift** | class-conditional MMD / null, unaligned | 215× | 276× |

**Label shift is negligible and is not what makes this problem hard.** Total
variation between the priors is 0.085, and the entire difference sits in one
class: RAVDESS carries 0.231 neutral against CREMA-D's 0.146, which is a direct
consequence of the `calm → neutral` mapping decision recorded in Phase 2. Every
other class differs by under 0.02.

## 2. The result that matters: alignment removes the wrong term

Taking `none` as the baseline and the strongest rung as the endpoint:

| direction | agg | marginal falls to | conditional falls to | conditional/marginal at `none` | at `mkmmd_full` |
|---|---|---|---|---|---|
| ravdess→cremad | last | **0.010×** | 0.07× | 0.14 | **0.92** |
| ravdess→cremad | layer:6 | 0.013× | 0.05× | 0.19 | 0.68 |
| cremad→ravdess | last | 0.042× | 0.08× | 0.29 | 0.53 |
| cremad→ravdess | layer:6 | 0.069× | 0.09× | 0.28 | 0.36 |

Alignment cuts the marginal discrepancy by up to **100×** and the conditional
discrepancy by only **14×** over the same rungs. The consequence is the last two
columns: before alignment the conditional term is a seventh of the marginal one;
after the strongest rung the two are nearly equal.

**So the residual discrepancy after alignment is essentially all conditional.**
That is the mechanism behind Phase 8's central finding. The ladder saturates
after the cheapest rung not because discrepancy stops falling — it keeps falling,
by two orders of magnitude — but because the part that keeps falling is the part
that was never limiting performance. P(x|y) still differs between corpora, and
no marginal alignment can touch it.

This is the joint claim PHASES.md asked for, demonstrated directly rather than
inferred: marginal falls, conditional does not follow it down, and macro-F1
tracks the conditional term rather than the marginal one.

## 3. The falsifiable test passes

Near-zero prior KL predicts that a label-shift correction cannot help. It does
not merely fail to help — **it hurts in 238 of 240 (record, estimator) pairs**,
by 0.013 to 0.24 macro-F1, with intervals excluding zero in nearly all cells.

That is the right sign for a correction estimated from a shift that is not
there: BBSE and EM both re-weight the posterior by a ratio estimated from noise,
and the reweighting moves decisions away from a decision rule that was already
better calibrated than the estimate. Both estimators recover a *planted* prior
shift to within 0.05 in `tests/test_analysis_shift.py`, so this is a result
about the corpora and not a broken implementation.

The decomposition therefore survives the test designed to break it. Had BBSE
helped at a KL of 0.02 nats, the decomposition would have been wrong and the
right move would have been to investigate, not to report the gain.

One asymmetry worth carrying: EM does much less damage than BBSE under `zscore`
(−0.021 against −0.077 forward), and much less under `coral` and `mkmmd_full` at
`last` (−0.019 and −0.023 against −0.163 and −0.205). EM is anchored by the
source prior and converges near it when the evidence is weak; BBSE inverts a
confusion matrix and amplifies its estimation error. Neither helps.

## 4. Per-class structure

The class-conditional term is not uniform, and the ordering is stable across
rungs (ravdess→cremad, `last`, unaligned → `mkmmd_full`):

| class | unaligned | best rung | retained |
|---|---|---|---|
| angry | 147× | 31× | **21%** |
| sad | 189× | 19× | 10% |
| neutral | 259× | 16× | 6% |
| disgust | 293× | 8× | 3% |
| fear | 230× | 5× | **2%** |
| happy | 172× | 6× | 3% |

`angry` retains an order of magnitude more conditional shift than `fear` after
the same alignment. Anger is the class where the two corpora disagree most about
what the emotion sounds like — which is consistent with RAVDESS being acted at
fixed intensity levels and CREMA-D being crowd-elicited — and it is the class a
marginal alignment is least able to help.

All six classes clear the minimum support of 50 in both directions, so no value
above is undefined. Per-class n is in the tables.

## 5. What this does and does not license

**Does:** the paper can now say that cross-corpus SER between these two corpora
is a conditional-shift problem, that the alignment literature addresses the
marginal term, and that this is why the ladder saturates. That is a claim about
the problem, not about our implementation of any method, so it does not depend
on the CORAL or MK-MMD details a reviewer could dispute.

**Does not:** two corpora, one backbone, one classifier. The conditional term is
measured with the same MMD machinery whose frame-dependence is unresolved (see
the sweep), so the *absolute* conditional numbers inherit that caveat. What is
robust is the **ratio** — conditional falling far more slowly than marginal —
because both are measured the same way on the same sets, so a frame effect
largely cancels.

The A10 firewall held: the conditional term was computed in `ser.analysis`,
written only to `results/phase9_shift.jsonl`, and never to the result schema or
to any input of fitting or selection. The assertion runs before the experiment
and before this report.
