# Phase 8 pass 1 — tables

Numbers only. No interpretation appears in this file; pass 2 lives in `reports/phase8_interpretation.md`.

Every target figure is a mean over seeds with a 95% interval from a **paired cluster bootstrap** that resamples target_test speakers and seeds together. Discrepancy columns use a t-interval over seeds instead: they are properties of a fitted map, not of a prediction, so they have no per-utterance form to bootstrap.

---

## 0. Integrity

| | |
|---|---|
| total rows | 5424 |
| unique run_ids | 5424 |
| status=ok | 5424 |
| status=failed | 0 |
| Phase 4 baselines (no freeze tag) | 60 |
| Stage 0 (grid-freeze-v1) | 6 |
| Stage 1 (grid-freeze-v2) | 372 |
| Stage 2 (grid-freeze-v3) | 4986 |
| schema versions present | [9] |

### Convergence

| family | runs with a solver count | min | median | max | cap | at cap |
|---|---|---|---|---|---|---|
| logreg | 1084 | 16 | 253 | 1094 | 20000 | 0 |
| svm_linear | 1084 | 5 | 14 | 437 | 20000 | 0 |
| svm_rbf | 1084 | 313 | 1408 | 10793 | 20000 | 0 |

NotConverged trials: **0**. Failed trials of any kind: **0** of 99720 attempted.

### MK-MMD fallback

| rung | runs | reverted to warm start | rate |
|---|---|---|---|
| mkmmd_diag | 1098 | 713 | 64.9% |
| mkmmd_full | 1242 | 686 | 55.2% |

| rung | lambda | reverted | rate |
|---|---|---|---|
| mkmmd_diag | 0.001 | 151/270 | 55.9% |
| mkmmd_diag | 0.01 | 194/288 | 67.4% |
| mkmmd_diag | 0.1 | 182/270 | 67.4% |
| mkmmd_diag | 1 | 186/270 | 68.9% |
| mkmmd_full | 0.001 | 116/270 | 43.0% |
| mkmmd_full | 0.01 | 162/432 | 37.5% |
| mkmmd_full | 0.1 | 173/270 | 64.1% |
| mkmmd_full | 1 | 235/270 | 87.0% |

### Cost against projection

| | projected | actual |
|---|---|---|
| CPU hours | 269.3 | 249.5 |
| wall hours at 4 shards | 67.3 | 62.4 |

| family | runs | CPU hours | share |
|---|---|---|---|
| logreg | 1084 | 11.7 | 4.7% |
| mlp | 1626 | 95.6 | 38.3% |
| svm_linear | 1084 | 96.5 | 38.7% |
| svm_rbf | 1084 | 15.7 | 6.3% |
| transformer | 108 | 30.1 | 12.0% |

## 1. Validated vs oracle

Validated = the configuration with the best `source_val` in that (pair, seed), scored on target. Oracle = the best target score in the same grid slice. **The oracle column is an upper bound that no protocol can reach; it is not a result.**

### full grid

| pair | n seeds | validated | oracle | gap | chance | majority |
|---|---|---|---|---|---|---|
| ravdess->cremad | 5 | 0.3156 [0.1880, 0.4432] | 0.4555 [0.4326, 0.4785] | 0.1399 [0.0231, 0.2568] | 0.1665 | 0.0425 |
| cremad->ravdess | 5 | 0.4994 [0.4592, 0.5395] | 0.5680 [0.5148, 0.6212] | 0.0687 [0.0090, 0.1283] | 0.1656 | 0.0444 |

### sklearn + MLP only (all 5 seeds)

| pair | n seeds | validated | oracle | gap | chance | majority |
|---|---|---|---|---|---|---|
| ravdess->cremad | 5 | 0.3054 [0.1830, 0.4277] | 0.4555 [0.4326, 0.4785] | 0.1502 [0.0360, 0.2643] | 0.1665 | 0.0425 |
| cremad->ravdess | 5 | 0.4994 [0.4592, 0.5395] | 0.5680 [0.5148, 0.6212] | 0.0687 [0.0090, 0.1283] | 0.1656 | 0.0444 |

## Primary comparisons

Declared in `tools/phase8_tables.py` before any result was computed. Paired cluster bootstrap over target_test speakers and seeds, 2000 replicates, Holm-corrected across all 14 tests.

| id | comparison | pair | seeds | conditions | difference in target macro-F1 [95% CI] | p | Holm p |
|---|---|---|---|---|---|---|---|
| L1 | zscore - none | ravdess->cremad | 5 | 153 | +0.1310 [+0.1174, +0.1422] | <0.0005 | <0.0070 |
| L1 | zscore - none | cremad->ravdess | 5 | 135 | +0.1961 [+0.1501, +0.2449] | <0.0005 | <0.0070 |
| L2 | mean_shift - none | ravdess->cremad | 5 | 153 | +0.1306 [+0.1159, +0.1439] | <0.0005 | <0.0070 |
| L2 | mean_shift - none | cremad->ravdess | 5 | 135 | +0.1817 [+0.1354, +0.2302] | <0.0005 | <0.0070 |
| L3 | coral - none | ravdess->cremad | 5 | 153 | +0.1359 [+0.1253, +0.1476] | <0.0005 | <0.0070 |
| L3 | coral - none | cremad->ravdess | 5 | 135 | +0.1833 [+0.1373, +0.2293] | <0.0005 | <0.0070 |
| L4 | mkmmd_diag - none | ravdess->cremad | 5 | 153 | +0.1208 [+0.1079, +0.1330] | <0.0005 | <0.0070 |
| L4 | mkmmd_diag - none | cremad->ravdess | 5 | 135 | +0.1852 [+0.1406, +0.2317] | <0.0005 | <0.0070 |
| L5 | mkmmd_full - none | ravdess->cremad | 5 | 153 | +0.1251 [+0.1149, +0.1351] | <0.0005 | <0.0070 |
| L5 | mkmmd_full - none | cremad->ravdess | 5 | 135 | +0.1524 [+0.1057, +0.1991] | <0.0005 | <0.0070 |
| A1 | layer - last | ravdess->cremad | 5 | 126 | +0.0640 [+0.0541, +0.0748] | <0.0005 | <0.0070 |
| A1 | layer - last | cremad->ravdess | 5 | 90 | +0.1439 [+0.1140, +0.1746] | <0.0005 | <0.0070 |
| A2 | weighted - last | ravdess->cremad | 5 | 126 | +0.0597 [+0.0504, +0.0702] | <0.0005 | <0.0070 |
| A2 | weighted - last | cremad->ravdess | 5 | 90 | +0.1096 [+0.0888, +0.1326] | <0.0005 | <0.0070 |

## 2. Alignment ladder

Target macro-F1 by rung, with **both** discrepancy columns. The two frames are reported together throughout; neither is presented alone.

### ravdess -> cremad

| rung | runs | target macro-F1 [95% CI] | effect size, own geometry | effect size, reference frame |
|---|---|---|---|---|
| none | 153 | 0.2289 [0.2133, 0.2444] | 1432.93 [1332.32, 1533.54] | 36905.40 [34506.36, 39304.45] |
| zscore | 153 | 0.3599 [0.3483, 0.3726] | 51.79 [47.94, 55.65] | 480.85 [439.16, 522.53] |
| mean_shift | 153 | 0.3595 [0.3442, 0.3762] | 104.05 [92.70, 115.41] | 19641.52 [18457.74, 20825.30] |
| coral | 828 | 0.3648 [0.3480, 0.3831] | 34.84 [32.39, 37.28] | 311.15 [275.50, 346.79] |
| mkmmd_diag | 558 | 0.3497 [0.3347, 0.3663] | 67.34 [64.00, 70.67] | 1057.77 [985.01, 1130.53] |
| mkmmd_full | 558 | 0.3540 [0.3383, 0.3696] | 12.04 [10.40, 13.68] | 23.25 [20.07, 26.42] |
| *chance floor* | | *0.1665* | | |
| *majority floor* | | *0.0425* | | |

### cremad -> ravdess

| rung | runs | target macro-F1 [95% CI] | effect size, own geometry | effect size, reference frame |
|---|---|---|---|---|
| none | 135 | 0.2361 [0.1981, 0.2694] | 1045.24 [1001.86, 1088.62] | 23610.38 [21402.46, 25818.29] |
| zscore | 135 | 0.4321 [0.4070, 0.4547] | 41.00 [38.50, 43.50] | 331.64 [307.77, 355.51] |
| mean_shift | 135 | 0.4178 [0.3931, 0.4386] | 76.60 [68.53, 84.67] | 8574.50 [7690.06, 9458.94] |
| coral | 810 | 0.4193 [0.3968, 0.4394] | 46.55 [42.75, 50.36] | 309.77 [272.52, 347.02] |
| mkmmd_diag | 540 | 0.4212 [0.3984, 0.4417] | 46.37 [44.95, 47.79] | 830.54 [790.82, 870.26] |
| mkmmd_full | 540 | 0.3885 [0.3653, 0.4099] | 10.71 [9.20, 12.21] | 17.71 [14.66, 20.75] |
| *chance floor* | | *0.1656* | | |
| *majority floor* | | *0.0444* | | |

## 3. Layer aggregation

### ravdess -> cremad

| aggregation | runs | families | target macro-F1 [95% CI] |
|---|---|---|---|
| last | 1056 | logreg,mlp,svm_linear,svm_rbf,transformer | 0.3060 [0.2913, 0.3197] |
| layer | 1056 | logreg,mlp,svm_linear,svm_rbf,transformer | 0.3706 [0.3496, 0.3929] |
| weighted | 291 | mlp,transformer | 0.3582 [0.3249, 0.3913] |

### cremad -> ravdess

| aggregation | runs | families | target macro-F1 [95% CI] |
|---|---|---|---|
| last | 1020 | logreg,mlp,svm_linear,svm_rbf | 0.3187 [0.2952, 0.3369] |
| layer | 1020 | logreg,mlp,svm_linear,svm_rbf | 0.4587 [0.4299, 0.4861] |
| weighted | 255 | mlp | 0.4788 [0.4482, 0.5056] |

### 13-layer curve (Stage 1 artefact, logreg, rung `none`, 2 seeds)

Carried forward unchanged from Stage 1. Stage 2 did not re-run the sweep, so this is **2 seeds and one classifier**, not full seed count.

| layer | hubert source_val / target | wav2vec2 source_val / target | wavlm source_val / target |
|---|---|---|---|
| 0 | 0.560 / **0.077** | 0.582 / **0.061** | 0.554 / **0.063** |
| 1 | 0.668 / **0.217** | 0.655 / **0.151** | 0.651 / **0.117** |
| 2 | 0.688 / **0.182** | 0.700 / **0.157** | 0.685 / **0.169** |
| 3 | 0.718 / **0.248** | 0.717 / **0.099** | 0.682 / **0.202** |
| 4 | 0.745 / **0.239** | 0.743 / **0.099** | 0.724 / **0.169** |
| 5 | 0.758 / **0.210** | 0.755 / **0.162** | 0.735 / **0.209** |
| 6 | 0.777 / **0.216** | 0.745 / **0.153** | 0.773 / **0.176** |
| 7 | 0.775 / **0.302** | 0.747 / **0.233** | 0.765 / **0.252** |
| 8 | 0.747 / **0.314** | 0.728 / **0.236** | 0.727 / **0.273** |
| 9 | 0.740 / **0.312** | 0.717 / **0.302** | 0.725 / **0.258** |
| 10 | 0.731 / **0.315** | 0.727 / **0.263** | 0.725 / **0.304** |
| 11 | 0.714 / **0.360** | 0.645 / **0.175** | 0.735 / **0.274** |
| 12 | 0.704 / **0.338** | 0.634 / **0.148** | 0.690 / **0.252** |

| backbone | argmax source_val | argmax target | gap |
|---|---|---|---|
| hubert | 6 | 11 | 5 |
| wav2vec2 | 5 | 9 | 4 |
| wavlm | 6 | 10 | 4 |

## 4. Classifier family

**The transformer is a reduced arm**: 2 seeds, primary direction only, one inner-grid setting per rung. Its interval is wider for that reason and its row is not comparable to the five-seed families.

### ravdess -> cremad

| family | runs | seeds | target macro-F1 [95% CI] |
|---|---|---|---|
| logreg | 510 | 5 | 0.3530 [0.3372, 0.3706] |
| svm_linear | 510 | 5 | 0.3322 [0.3188, 0.3465] |
| svm_rbf | 510 | 5 | 0.3418 [0.3251, 0.3617] |
| mlp | 765 | 5 | 0.3440 [0.3135, 0.3760] |
| transformer *(reduced arm)* | 108 | 2 | 0.3387 [0.3221, 0.3561] |

### cremad -> ravdess

| family | runs | seeds | target macro-F1 [95% CI] |
|---|---|---|---|
| logreg | 510 | 5 | 0.4065 [0.3785, 0.4288] |
| svm_linear | 510 | 5 | 0.3827 [0.3516, 0.4087] |
| svm_rbf | 510 | 5 | 0.3749 [0.3469, 0.4048] |
| mlp | 765 | 5 | 0.4201 [0.3923, 0.4417] |
| transformer | 0 | 0 | not run in this direction |

## 5. Direction (matched-n)

**Both directions are matched-n**: cross-corpus `source_train` is capped to the smaller direction's size, so CREMA-D contributes 988 training utterances rather than 5972. Any asymmetry below is therefore not a training-set size effect. Full-n reverse has not been run.

| direction | runs | source_train n | target_test n | seeds | target macro-F1 [95% CI] | chance | majority |
|---|---|---|---|---|---|---|---|
| ravdess->cremad (matched-n) | 2295 | 988-988 (cap None) | 3677-3690 | 5 | 0.3429 [0.3265, 0.3627] | 0.1665 | 0.0425 |
| cremad->ravdess (matched-n) | 2295 | 988-988 (cap 988) | 624-624 | 5 | 0.3987 [0.3793, 0.4155] | 0.1656 | 0.0444 |

Excludes the transformer, which ran the primary direction only and would otherwise weight one side of this comparison.

## 6. Blending (the alpha arm)

288 runs. `scalar` mode only; `gaa` is not implemented and was not run.

alpha=0 discards the alignment entirely, so the two rungs' alpha=0 rows are different `run_id`s computed over identical features. They agree to every reported digit, which is an end-to-end check on the blending path -- and 18 runs per direction of duplicated compute.

The alpha=1.00 row is drawn from the main grid at **exactly the inner-grid setting the arm used** (coral eps=0.1, mkmmd_full lambda=0.01) and the same backbone, seeds and families -- not pooled over the main grid's other eps and lambda values, which would make it a different experiment.

### ravdess -> cremad

| alignment | alpha | runs | target macro-F1 [95% CI] |
|---|---|---|---|
| coral | 0.00 | 18 | 0.2987 [0.2746, 0.3188] |
| coral | 0.25 | 18 | 0.3287 [0.3105, 0.3433] |
| coral | 0.50 | 18 | 0.3586 [0.3301, 0.3870] |
| coral | 0.75 | 18 | 0.3907 [0.3644, 0.4161] |
| coral | 1.00 *(= blending none)* | 18 | 0.4012 [0.3802, 0.4186] |
| mkmmd_full | 0.00 | 18 | 0.2987 [0.2746, 0.3188] |
| mkmmd_full | 0.25 | 18 | 0.3283 [0.3076, 0.3492] |
| mkmmd_full | 0.50 | 18 | 0.3536 [0.3259, 0.3796] |
| mkmmd_full | 0.75 | 18 | 0.3869 [0.3619, 0.4089] |
| mkmmd_full | 1.00 *(= blending none)* | 18 | 0.3963 [0.3779, 0.4115] |

### cremad -> ravdess

| alignment | alpha | runs | target macro-F1 [95% CI] |
|---|---|---|---|
| coral | 0.00 | 18 | 0.3154 [0.2513, 0.3868] |
| coral | 0.25 | 18 | 0.3013 [0.2442, 0.3619] |
| coral | 0.50 | 18 | 0.3024 [0.2504, 0.3546] |
| coral | 0.75 | 18 | 0.3349 [0.2937, 0.3754] |
| coral | 1.00 *(= blending none)* | 18 | 0.3833 [0.3522, 0.4225] |
| mkmmd_full | 0.00 | 18 | 0.3154 [0.2513, 0.3868] |
| mkmmd_full | 0.25 | 18 | 0.2959 [0.2403, 0.3540] |
| mkmmd_full | 0.50 | 18 | 0.3050 [0.2523, 0.3581] |
| mkmmd_full | 0.75 | 18 | 0.3344 [0.2866, 0.3785] |
| mkmmd_full | 1.00 *(= blending none)* | 18 | 0.4027 [0.3690, 0.4448] |

## 7. Per-class F1, from the stored predictions

Computed by summing the per-speaker confusion matrices of the **validated** configuration in each (pair, seed), then taking per-class F1 of the total. Interval is over seeds.

### ravdess -> cremad

| class | support | F1 [95% CI over seeds] |
|---|---|---|
| angry | 630 | 0.4097 [0.1947, 0.6247] |
| disgust | 630 | 0.3799 [0.3009, 0.4588] |
| fear | 630 | 0.3108 [0.1675, 0.4542] |
| happy | 630 | 0.2881 [0.0932, 0.4829] |
| neutral | 540 | 0.2390 [0.1128, 0.3653] |
| sad | 630 | 0.2661 [0.1065, 0.4257] |
| **macro** | 3690 | **0.3156 [0.1880, 0.4432]** |

### cremad -> ravdess

| class | support | F1 [95% CI over seeds] |
|---|---|---|
| angry | 96 | 0.5673 [0.4660, 0.6687] |
| disgust | 96 | 0.5776 [0.5289, 0.6263] |
| fear | 96 | 0.6777 [0.6467, 0.7086] |
| happy | 96 | 0.3493 [0.2428, 0.4557] |
| neutral | 144 | 0.5832 [0.4995, 0.6670] |
| sad | 96 | 0.2410 [0.1773, 0.3047] |
| **macro** | 624 | **0.4994 [0.4592, 0.5395]** |

## 8. Secondary — differences among the aligned rungs

**Not in the primary family. Intervals only; no significance is claimed or implied for any row here.** The primary tests compare each rung against `none`; this table is the remaining question of whether the rungs differ from *each other*, which is what "flat across the ladder" refers to. 500 replicates.

### ravdess -> cremad

| comparison | conditions | difference in target macro-F1 [95% CI] |
|---|---|---|
| mean_shift - zscore | 153 | -0.0004 [-0.0066, +0.0064] |
| coral - zscore | 153 | +0.0049 [-0.0032, +0.0143] |
| mkmmd_diag - zscore | 153 | -0.0103 [-0.0165, -0.0039] |
| mkmmd_full - zscore | 153 | -0.0059 [-0.0139, +0.0021] |
| coral - mean_shift | 153 | +0.0053 [-0.0021, +0.0122] |
| mkmmd_diag - mean_shift | 153 | -0.0098 [-0.0122, -0.0078] |
| mkmmd_full - mean_shift | 153 | -0.0055 [-0.0161, +0.0041] |
| mkmmd_diag - coral | 153 | -0.0151 [-0.0213, -0.0079] |
| mkmmd_full - coral | 153 | -0.0108 [-0.0175, -0.0049] |
| mkmmd_full - mkmmd_diag | 153 | +0.0043 [-0.0052, +0.0133] |

Largest absolute difference among aligned rungs: **mkmmd_diag - coral = -0.0151 [-0.0213, -0.0079]**.

### cremad -> ravdess

| comparison | conditions | difference in target macro-F1 [95% CI] |
|---|---|---|
| mean_shift - zscore | 135 | -0.0144 [-0.0265, -0.0036] |
| coral - zscore | 135 | -0.0128 [-0.0247, -0.0027] |
| mkmmd_diag - zscore | 135 | -0.0109 [-0.0212, -0.0008] |
| mkmmd_full - zscore | 135 | -0.0437 [-0.0559, -0.0320] |
| coral - mean_shift | 135 | +0.0016 [-0.0087, +0.0120] |
| mkmmd_diag - mean_shift | 135 | +0.0034 [-0.0015, +0.0080] |
| mkmmd_full - mean_shift | 135 | -0.0293 [-0.0441, -0.0140] |
| mkmmd_diag - coral | 135 | +0.0019 [-0.0070, +0.0112] |
| mkmmd_full - coral | 135 | -0.0309 [-0.0402, -0.0221] |
| mkmmd_full - mkmmd_diag | 135 | -0.0327 [-0.0457, -0.0194] |

Largest absolute difference among aligned rungs: **mkmmd_full - zscore = -0.0437 [-0.0559, -0.0320]**.

## 9. Stage 1 observations, re-measured

The Stage 1 figure is quoted from `reports/stage1_analysis.md` (2 seeds, hubert, ravdess->cremad, pre-selection). The Stage 2 column is the same quantity at full seed count. **Verdicts are in pass 2, not here.**

### CORAL `source_val` against eps (the mis-centred grid)

| eps | Stage 1 source_val | Stage 2 source_val | Stage 2 target macro-F1 |
|---|---|---|---|
| 0.0001 | 0.5592 | 0.5025 | 0.3726 |
| 0.01 | 0.6032 | 0.5529 | 0.3679 |
| 0.1 | 0.6523 | 0.6174 | 0.3708 |
| 1 | not in grid | 0.6597 | 0.3805 |
| 10 | not in grid | 0.6782 | 0.3913 |
| ledoit-wolf | 0.6096 | 0.5594 | 0.3676 |

`source_val` argmax over the numeric eps grid: **10** (grid maximum is 10).

### Alignment gain by layer aggregation (the interaction)

| pair | aggregation | target, `none` | target, aligned | gain | Stage 1 gain |
|---|---|---|---|---|---|
| ravdess->cremad | last | 0.2339 | 0.3175 | +0.0836 | +0.042 |
| ravdess->cremad | layer | 0.2116 | 0.3858 | +0.1742 | +0.168 |
| ravdess->cremad | weighted | 0.2824 | 0.3797 | +0.0973 | +0.095 |
| cremad->ravdess | last | 0.2103 | 0.3382 | +0.1280 | not measured |
| cremad->ravdess | layer | 0.2533 | 0.4578 | +0.2046 | not measured |
| cremad->ravdess | weighted | 0.2704 | 0.4502 | +0.1798 | not measured |

### Frame dependence across the ladder

Spearman rho over the six rung means, target macro-F1 against each discrepancy column. Six points, so this is a coarse re-measurement of the layer-sweep finding on a different axis, not a replication of it.

| pair | rho(own geometry, target) | rho(reference frame, target) |
|---|---|---|
| ravdess->cremad | -0.200 | -0.200 |
| cremad->ravdess | -0.371 | -0.086 |

### MK-MMD lambda on `source_val`

| rung | lambda | source_val | target macro-F1 | fallback rate |
|---|---|---|---|---|
| mkmmd_diag | 0.001 | 0.6794 | 0.3863 | 55.9% |
| mkmmd_diag | 0.01 | 0.6857 | 0.3848 | 67.4% |
| mkmmd_diag | 0.1 | 0.6791 | 0.3863 | 67.4% |
| mkmmd_diag | 1 | 0.6794 | 0.3860 | 68.9% |
| mkmmd_diag | **spread** | **0.0066** | | |
| mkmmd_full | 0.001 | 0.5079 | 0.3713 | 43.0% |
| mkmmd_full | 0.01 | 0.5242 | 0.3725 | 43.8% |
| mkmmd_full | 0.1 | 0.5082 | 0.3708 | 64.1% |
| mkmmd_full | 1 | 0.5047 | 0.3715 | 87.0% |
| mkmmd_full | **spread** | **0.0196** | | |
