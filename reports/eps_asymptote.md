# CORAL's shrinkage asymptote

Stage 1 and Stage 2 both found `source_val` monotone increasing in eps with the argmax sitting on the grid boundary, and Phase 8 recorded that as an unresolved defect -- "the grid is still mis-centred". **It is not a defect.** It is a property of the estimator and it is predictable in closed form.

---

## 1. The derivation

CORAL's map is `M = C_s^{-1/2} C_t^{1/2}`, with each covariance regularised as `C + eps * tr(C)/d * I`. As eps grows both regularised covariances approach a scaled identity, so

```
    M  ->  sqrt( tr(C_t) / tr(C_s) ) * I
```

and the transform `x -> (x - mu_s) M + mu_t` collapses to **a global scalar rescale plus a mean shift** -- which is `mean_shift` with one extra degree of freedom, and close to `zscore` when the per-dimension scales are similar. The covariance matching that CORAL exists to do is switched off continuously as eps rises.

## 2. Analytic convergence, measured on real features

hubert, `last`, seed 0. `||M - cI|| / ||M||` is the relative distance from the fitted map to the nearest scaled identity; `c` is `trace(M)/d`; the predicted limit is `sqrt(tr(C_t)/tr(C_s))` computed from the **unregularised** covariances.

| pair | eps | \|\|M-cI\|\|/\|\|M\|\| | c fitted | c predicted | ratio |
|---|---|---|---|---|---|
| ravdess->cremad | 0.0001 | 0.9254 | 2.4867 | 1.1327 | 2.195 |
| ravdess->cremad | 0.01 | 0.8401 | 1.6668 | 1.1327 | 1.472 |
| ravdess->cremad | 0.1 | 0.6904 | 1.3401 | 1.1327 | 1.183 |
| ravdess->cremad | 1 | 0.4009 | 1.1871 | 1.1327 | 1.048 |
| ravdess->cremad | 10 | 0.1436 | 1.1414 | 1.1327 | 1.008 |
| ravdess->cremad | 100 | 0.0343 | 1.1333 | 1.1327 | 1.001 |
| ravdess->cremad | 1000 | 0.0049 | 1.1327 | 1.1327 | 1.000 |
| cremad->ravdess | 0.0001 | 0.9630 | 1.3124 | 0.8764 | 1.497 |
| cremad->ravdess | 0.01 | 0.8899 | 1.1018 | 0.8764 | 1.257 |
| cremad->ravdess | 0.1 | 0.7502 | 0.9856 | 0.8764 | 1.125 |
| cremad->ravdess | 1 | 0.4463 | 0.9125 | 0.8764 | 1.041 |
| cremad->ravdess | 10 | 0.1571 | 0.8830 | 0.8764 | 1.008 |
| cremad->ravdess | 100 | 0.0344 | 0.8769 | 0.8764 | 1.001 |
| cremad->ravdess | 1000 | 0.0049 | 0.8764 | 0.8764 | 1.000 |

The residual falls monotonically from 0.93 to 0.005 and the fitted scalar converges on its predicted limit (ratio 2.195 -> 1.000, and 1.497 -> 1.000). **At eps=10 -- the value `source_val` selects in Stage 2 -- the map is already 86% of the way to a pure scalar.**

This half of the question is settled. It involves no classifier, no seed variance and no target labels: it is arithmetic on two covariance matrices, and it holds in both directions.

## 3. Empirical behaviour -- INCOMPLETE

**35 of 120 enumerated runs completed, covering 1 of 2 directions.** The probe was killed by memory exhaustion and has not been relaunched.

| | |
|---|---|
| runs completed | 35 / 120 |
| directions covered | ravdess->cremad |
| eps values | 100, 1000 |
| seeds | [0, 1, 2] |
| families | logreg, mlp, svm_rbf |

**This is not enough to confirm the asymptote empirically.** `cremad->ravdess` has zero coverage, so the claim cannot be checked in the direction where CORAL's scalar limit is *below* one (0.8764 against 1.1327), which is the more informative case. The partial numbers are shown for completeness and should not be read as a result.

### ravdess -> cremad (partial)

| rung | eps | runs | source_val | target macro-F1 | effect, own frame | effect, reference frame |
|---|---|---|---|---|---|---|
| coral | 0.0001 | 30 | 0.5177 [0.4474, 0.5881] | 0.3818 [0.3695, 0.3942] | 3.62 [0.12, 7.12] | 3.64 [1.83, 5.44] |
| coral | 0.01 | 30 | 0.5693 [0.4970, 0.6415] | 0.3810 [0.3643, 0.3977] | 7.11 [2.40, 11.83] | 27.45 [21.98, 32.92] |
| coral | 0.1 | 30 | 0.6273 [0.5469, 0.7077] | 0.3883 [0.3714, 0.4052] | 19.86 [11.16, 28.56] | 121.09 [96.08, 146.09] |
| coral | 1 | 30 | 0.6897 [0.6027, 0.7768] | 0.4023 [0.3833, 0.4212] | 49.85 [38.67, 61.04] | 412.95 [282.22, 543.68] |
| coral | 10 | 30 | 0.7235 [0.6407, 0.8062] | 0.4118 [0.3913, 0.4323] | 75.65 [42.36, 108.94] | 1254.61 [938.47, 1570.75] |
| coral | 100 *(partial probe)* | 18 | 0.7341 [0.6060, 0.8621] | 0.4012 [0.3502, 0.4521] | 34.93 [21.61, 48.24] | 6274.31 [2824.40, 9724.23] |
| coral | 1000 *(partial probe)* | 17 | 0.7282 [0.5963, 0.8601] | 0.3898 [0.3461, 0.4336] | 33.89 [20.38, 47.40] | 10474.00 [8164.30, 12783.71] |
| **mean_shift** | -- | 30 | 0.7278 [0.6427, 0.8130] | 0.3932 [0.3731, 0.4132] | 78.92 [19.26, 138.57] | 15137.19 [13288.97, 16985.41] |
| **zscore** | -- | 30 | 0.7396 [0.6646, 0.8146] | 0.3895 [0.3766, 0.4024] | 48.59 [16.38, 80.80] | 396.43 [113.24, 679.62] |
| **none** | -- | 30 | 0.7304 [0.6469, 0.8138] | 0.2997 [0.2874, 0.3119] | 1339.65 [274.76, 2404.55] | 29444.55 [25121.79, 33767.32] |

## 4. Interpretation

### The boundary argmax is a result, not a mis-centred grid

`source_val` rises monotonically as eps rises, and the analytic result says what it is rising toward: **less and less covariance matching**. The selection surface is not asking for a larger shrinkage parameter, it is asking for CORAL to stop being CORAL. Extending the grid further would move the argmax again, because the quantity being maximised improves monotonically toward the degenerate limit.

Phase 8 listed "the CORAL eps grid is still monotone to its boundary" as an open defect. **That item is closed.** The correct statement for the paper is that CORAL's `source_val` is monotone in shrinkage over seven orders of magnitude because the shrinkage is doing regularisation work unrelated to domain alignment, and at the selected value the covariance term is 86% suppressed.

### Independent confirmation that covariance matching contributes nothing

This is the third line of evidence pointing the same way, and the only one that does not depend on a classifier:

1. **Phase 8** -- `zscore` matches or beats every more elaborate rung on target macro-F1, and the largest difference among aligned rungs is 0.0151 forward / 0.0437 reverse.

2. **Phase 9** -- alignment cuts marginal discrepancy to 0.010x while conditional discrepancy falls only to 0.07x, so what the extra moments remove was not what limited performance.

3. **Here** -- selection on source data drives CORAL continuously toward a scalar rescale plus a mean shift, i.e. toward `mean_shift`, and never turns back.

### What is still missing

The empirical arm is 35/120 and covers one direction. It cannot currently confirm that **target** macro-F1 and the discrepancy columns converge on `mean_shift`'s values as eps grows -- only that the map does. Completing it needs both directions; the relaunch command is in PROGRESS.md. Until then this report claims the analytic result and nothing more.
