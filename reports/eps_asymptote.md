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

## 3. Empirical behaviour

**120 of 120 enumerated runs completed, covering 2 of 2 directions.**

| | |
|---|---|
| runs completed | 120 / 120 |
| unique run_ids | 120 |
| duplicate rows collapsed | 0 |
| directions covered | cremad->ravdess, ravdess->cremad |
| eps values | 100, 1000 |
| seeds | [0, 1, 2, 3, 4] |
| families | logreg, mlp, svm_rbf |
| aggregations | last, layer |
| non-ok status | 0 |

Assembled from `results/eps_asymptote.jsonl` (35 rows), `results/shards/eps_fwd.jsonl` (25 rows), `results/shards/eps_rev.jsonl` (60 rows), deduplicated by `run_id`. A row recorded twice with *different* values raises rather than being collapsed -- that would be an identity bug, not a duplicate.

### cremad -> ravdess

| rung | eps | runs | source_val | target macro-F1 | effect, own frame | effect, reference frame |
|---|---|---|---|---|---|---|
| coral | 0.0001 | 30 | 0.4542 [0.4419, 0.4664] | 0.4324 [0.4056, 0.4592] | 4.05 [2.10, 6.00] | 1.93 [1.06, 2.80] |
| coral | 0.01 | 30 | 0.5008 [0.4934, 0.5081] | 0.4139 [0.3726, 0.4552] | 7.02 [5.07, 8.96] | 13.20 [8.97, 17.43] |
| coral | 0.1 | 30 | 0.5636 [0.5558, 0.5714] | 0.4061 [0.3737, 0.4384] | 21.68 [16.74, 26.63] | 80.98 [58.05, 103.91] |
| coral | 1 | 30 | 0.6257 [0.6141, 0.6374] | 0.4322 [0.3909, 0.4734] | 67.32 [54.41, 80.23] | 344.02 [292.77, 395.26] |
| coral | 10 | 30 | 0.6490 [0.6313, 0.6668] | 0.4665 [0.4321, 0.5010] | 104.80 [55.08, 154.52] | 1327.74 [1004.75, 1650.74] |
| coral | 100 *(probe)* | 30 | 0.6517 [0.6345, 0.6689] | 0.4749 [0.4361, 0.5136] | 49.37 [39.65, 59.08] | 7401.52 [5393.58, 9409.45] |
| coral | 1000 *(probe)* | 30 | 0.6505 [0.6343, 0.6667] | 0.4695 [0.4329, 0.5061] | 41.04 [30.90, 51.19] | 12832.37 [8741.15, 16923.58] |
| **mean_shift** | -- | 30 | 0.6517 [0.6340, 0.6695] | 0.4634 [0.4265, 0.5003] | 56.57 [45.26, 67.87] | 10296.87 [6219.19, 14374.54] |
| **zscore** | -- | 30 | 0.6530 [0.6353, 0.6706] | 0.4708 [0.4238, 0.5178] | 33.97 [26.03, 41.91] | 249.54 [209.58, 289.49] |
| **none** | -- | 30 | 0.6519 [0.6333, 0.6705] | 0.3074 [0.2696, 0.3452] | 938.56 [698.88, 1178.25] | 27465.53 [16911.94, 38019.12] |

### ravdess -> cremad

| rung | eps | runs | source_val | target macro-F1 | effect, own frame | effect, reference frame |
|---|---|---|---|---|---|---|
| coral | 0.0001 | 30 | 0.5177 [0.4474, 0.5881] | 0.3818 [0.3695, 0.3942] | 3.62 [0.12, 7.12] | 3.64 [1.83, 5.44] |
| coral | 0.01 | 30 | 0.5693 [0.4970, 0.6415] | 0.3810 [0.3643, 0.3977] | 7.11 [2.40, 11.83] | 27.45 [21.98, 32.92] |
| coral | 0.1 | 30 | 0.6273 [0.5469, 0.7077] | 0.3883 [0.3714, 0.4052] | 19.86 [11.16, 28.56] | 121.09 [96.08, 146.09] |
| coral | 1 | 30 | 0.6897 [0.6027, 0.7768] | 0.4023 [0.3833, 0.4212] | 49.85 [38.67, 61.04] | 412.95 [282.22, 543.68] |
| coral | 10 | 30 | 0.7235 [0.6407, 0.8062] | 0.4118 [0.3913, 0.4323] | 75.65 [42.36, 108.94] | 1254.61 [938.47, 1570.75] |
| coral | 100 *(probe)* | 30 | 0.7276 [0.6449, 0.8103] | 0.3992 [0.3792, 0.4193] | 64.62 [12.65, 116.59] | 6471.74 [5194.24, 7749.24] |
| coral | 1000 *(probe)* | 30 | 0.7282 [0.6455, 0.8110] | 0.3879 [0.3720, 0.4039] | 57.85 [8.06, 107.65] | 10976.21 [9703.88, 12248.54] |
| **mean_shift** | -- | 30 | 0.7278 [0.6427, 0.8130] | 0.3932 [0.3731, 0.4132] | 78.92 [19.26, 138.57] | 15137.19 [13288.97, 16985.41] |
| **zscore** | -- | 30 | 0.7396 [0.6646, 0.8146] | 0.3895 [0.3766, 0.4024] | 48.59 [16.38, 80.80] | 396.43 [113.24, 679.62] |
| **none** | -- | 30 | 0.7304 [0.6469, 0.8138] | 0.2997 [0.2874, 0.3119] | 1339.65 [274.76, 2404.55] | 29444.55 [25121.79, 33767.32] |

## 4. Interpretation

### The boundary argmax is a result, not a mis-centred grid

`source_val` rises with eps, and the analytic result says what it is rising toward: **less and less covariance matching**. The selection surface is not asking for a larger shrinkage parameter, it is asking for CORAL to stop being CORAL.

When this section was first written the empirical arm was incomplete, and it predicted that extending the grid would simply move the argmax again. **The completed probe falsifies that prediction**, in the direction that strengthens the result: the rise is the approach to an asymptote, not an unbounded climb. Past eps=10 the increments collapse to noise and in `cremad->ravdess` the argmax is interior. The wrong prediction is left visible rather than edited away, because it was written down before the data arrived.

Phase 8 listed "the CORAL eps grid is still monotone to its boundary" as an open defect. **That item is closed.** The correct statement for the paper is that CORAL's `source_val` increases with shrinkage over seven orders of magnitude and then saturates, because the shrinkage is doing regularisation work unrelated to domain alignment -- and at the value Stage 2 selects the covariance term is already 86% suppressed.

### Independent confirmation that covariance matching contributes nothing

This is the third line of evidence pointing the same way, and the only one that does not depend on a classifier:

1. **Phase 8** -- `zscore` matches or beats every more elaborate rung on target macro-F1, and the largest difference among aligned rungs is 0.0151 forward / 0.0437 reverse.

2. **Phase 9** -- alignment cuts marginal discrepancy to 0.010x while conditional discrepancy falls only to 0.07x, so what the extra moments remove was not what limited performance.

3. **Here** -- selection on source data drives CORAL toward a scalar rescale plus a mean shift, and its `source_val` and target scores converge on `mean_shift`'s. The elaborate rung is selected into being the simple one.

### The empirical arm confirms it

Every figure below is computed from the completed 120-run probe; none is typed in.

| direction | quantity | CORAL at largest eps | `mean_shift` | difference |
|---|---|---|---|---|
| crem->ravd | source_val (eps=1000) | 0.6505 | 0.6517 | -0.0012 |
| crem->ravd | target macro-F1 (eps=1000) | 0.4695 | 0.4634 | +0.0061 |
| ravd->crem | source_val (eps=1000) | 0.7282 | 0.7278 | +0.0004 |
| ravd->crem | target macro-F1 (eps=1000) | 0.3879 | 0.3932 | -0.0052 |

`source_val` and target macro-F1 at the largest shrinkage land within 0.0061 of `mean_shift` on every row -- the rung converges on the behaviour the derivation predicts, not merely on the matrix form. The reference-frame effect size converges the same way, to the 10^4 order `mean_shift` sits at rather than `zscore`'s 10^2, which is what distinguishes "scalar rescale plus mean shift" from "per-dimension standardisation".

**The surface also stops climbing.** In `cremad->ravdess` the `source_val` argmax is now **interior** (eps=100 at 0.6517, against 0.6505 at eps=1000); in `ravdess->cremad` it is still nominally at the boundary but the last decade buys +0.0006. The monotone-to-the-edge behaviour that prompted this whole investigation was the approach to an asymptote, and the asymptote has now been reached in both directions.

One honest caveat: the **own-frame** effect size does not converge on `mean_shift`'s value (57.9 against 78.9 forward, 41.0 against 56.6 reverse). That is expected rather than contradictory -- the own-frame statistic uses a per-rung median bandwidth, so the extra global scalar that distinguishes degenerate CORAL from `mean_shift` also shifts the bandwidth that normalises it. The reference frame, which is fixed across rungs, is the column that can answer this question, and it does.
