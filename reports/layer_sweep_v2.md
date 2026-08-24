# 13-layer sweep, full seed count

Replaces the Stage 1 sweep (2 seeds, one classifier, one pair, rung `none`). Two claims rested entirely on that and are settled here.

**Numbers first; interpretation is section 5 and nothing before it.**

---

## 0. Coverage and what is excluded

1560 runs analysed: 13 layers x 4 rungs x 3 backbones x 2 directions x 5 seeds, logreg. Zero failures, 0 duplicate rows, 0 unparsable lines.

| rung | cells | status |
|---|---|---|
| none | 390/390 | **included** |
| zscore | 390/390 | **included** |
| mean_shift | 390/390 | **included** |
| coral | 390/390 | **included** |
| mkmmd_diag | 254/390 | **EXCLUDED** (136 missing) |
| mkmmd_full | 157/390 | **EXCLUDED** (233 missing) |

**mkmmd_diag, mkmmd_full are excluded** — 369 of 780 cells were still running when this report was generated. A partially covered rung would contribute correlations computed over different layer subsets in different seeds, which is a mechanism for manufacturing a sign flip rather than measuring one. Section 5 states what their absence can and cannot change.

## 1. Frame dependence

Spearman rho between a layer's discrepancy and its target macro-F1, computed **within each seed** across the 13 layers, then averaged over seeds with a t-interval. A sign that is not stable across seeds cannot look stable here.

A cell counts as a disagreement only when **both** intervals exclude zero and fall on opposite sides. Two opposite-signed point estimates whose intervals straddle zero are noise.

| direction | backbone | rung | rho (own geometry) | rho (reference frame) | disagree |
|---|---|---|---|---|---|
| crem->ravd | hubert | none | -0.324 [-0.988, +0.339] | -0.023 [-0.715, +0.669] | no |
| crem->ravd | hubert | zscore | -0.718 [-0.892, -0.544] | +0.457 [+0.203, +0.712] | **YES** |
| crem->ravd | hubert | mean_shift | +0.090 [-0.564, +0.744] | -0.236 [-0.773, +0.301] | no |
| crem->ravd | hubert | coral | -0.591 [-0.858, -0.324] | -0.462 [-0.797, -0.126] | no |
| crem->ravd | wav2vec2 | none | +0.286 [+0.007, +0.564] | -0.010 [-0.363, +0.344] | no |
| crem->ravd | wav2vec2 | zscore | -0.424 [-0.767, -0.081] | +0.543 [+0.335, +0.750] | **YES** |
| crem->ravd | wav2vec2 | mean_shift | -0.453 [-0.578, -0.327] | +0.162 [-0.380, +0.703] | no |
| crem->ravd | wav2vec2 | coral | -0.545 [-0.903, -0.187] | -0.514 [-0.814, -0.215] | no |
| crem->ravd | wavlm | none | -0.233 [-0.915, +0.450] | -0.032 [-0.283, +0.220] | no |
| crem->ravd | wavlm | zscore | -0.707 [-1.061, -0.352] | +0.645 [+0.451, +0.839] | **YES** |
| crem->ravd | wavlm | mean_shift | -0.491 [-0.863, -0.119] | -0.082 [-0.352, +0.187] | no |
| crem->ravd | wavlm | coral | -0.588 [-0.802, -0.374] | -0.507 [-0.741, -0.272] | no |
| ravd->crem | hubert | none | -0.362 [-1.116, +0.392] | +0.330 [+0.127, +0.533] | no |
| ravd->crem | hubert | zscore | -0.615 [-0.979, -0.252] | +0.569 [+0.267, +0.871] | **YES** |
| ravd->crem | hubert | mean_shift | -0.074 [-0.733, +0.586] | +0.451 [+0.259, +0.642] | no |
| ravd->crem | hubert | coral | -0.247 [-0.458, -0.037] | +0.498 [+0.358, +0.637] | **YES** |
| ravd->crem | wav2vec2 | none | +0.053 [-0.696, +0.802] | +0.226 [-0.030, +0.483] | no |
| ravd->crem | wav2vec2 | zscore | -0.554 [-1.013, -0.095] | +0.766 [+0.674, +0.858] | **YES** |
| ravd->crem | wav2vec2 | mean_shift | +0.004 [-0.536, +0.545] | +0.181 [-0.122, +0.485] | no |
| ravd->crem | wav2vec2 | coral | -0.713 [-0.858, -0.568] | +0.196 [-0.037, +0.428] | no |
| ravd->crem | wavlm | none | -0.208 [-0.937, +0.522] | +0.226 [-0.145, +0.598] | no |
| ravd->crem | wavlm | zscore | -0.599 [-1.076, -0.122] | +0.432 [+0.315, +0.548] | **YES** |
| ravd->crem | wavlm | mean_shift | -0.424 [-1.066, +0.218] | +0.192 [-0.033, +0.418] | no |
| ravd->crem | wavlm | coral | -0.548 [-0.843, -0.254] | +0.169 [-0.146, +0.484] | no |

**Sign disagreements: 7 of 24 cells.**

Pooled over every cell and seed:

| frame | mean rho | 95% interval | n |
|---|---|---|---|
| own geometry | -0.374 | [-0.454, -0.295] | 120 |
| reference frame | +0.174 | [+0.099, +0.249] | 120 |

## 2. Depth divergence

Argmax layer on `source_val` against argmax layer on target macro-F1, computed per seed. A positive gap means the transfer-optimal layer is **deeper** than the in-domain-optimal one, which is the Stage 1 claim.

**The cost column is non-negative by construction** -- it is `best target score - target score of the source_val pick`, and the first term is a maximum. On its own it proves nothing. It is therefore reported against the cost of picking a layer uniformly at random from the same 13, which is the honest null: if `source_val` costs as much as a coin flip, it carries no information about depth.

| direction | backbone | rung | argmax val (median) | argmax target (median) | gap (layers) | cost, source_val pick | cost, random pick |
|---|---|---|---|---|---|---|---|
| crem->ravd | hubert | none | 7 | 11 | +2.800 [-0.031, +5.631] | +0.0830 [-0.0070, +0.1729] | +0.1272 |
| crem->ravd | hubert | zscore | 6 | 6 | +0.400 [-1.015, +1.815] | +0.0183 [-0.0179, +0.0545] | +0.1101 |
| crem->ravd | hubert | mean_shift | 7 | 6 | +0.200 [-3.245, +3.645] | +0.0143 [-0.0006, +0.0292] | +0.0953 |
| crem->ravd | hubert | coral | 12 | 7 | -2.800 [-5.344, -0.256] | +0.0983 [+0.0556, +0.1410] | +0.0969 |
| crem->ravd | wav2vec2 | none | 8 | 5 | -3.800 [-10.094, +2.494] | +0.0432 [+0.0049, +0.0816] | +0.0760 |
| crem->ravd | wav2vec2 | zscore | 7 | 5 | -1.800 [-3.419, -0.181] | +0.0410 [-0.0051, +0.0871] | +0.1233 |
| crem->ravd | wav2vec2 | mean_shift | 8 | 5 | -3.000 [-3.878, -2.122] | +0.0443 [+0.0171, +0.0715] | +0.1116 |
| crem->ravd | wav2vec2 | coral | 9 | 6 | -2.400 [-3.510, -1.290] | +0.0566 [-0.0013, +0.1145] | +0.1000 |
| crem->ravd | wavlm | none | 7 | 10 | +1.600 [-2.573, +5.773] | +0.0847 [+0.0182, +0.1511] | +0.1040 |
| crem->ravd | wavlm | zscore | 6 | 6 | -0.400 [-1.510, +0.710] | +0.0150 [-0.0042, +0.0341] | +0.1084 |
| crem->ravd | wavlm | mean_shift | 7 | 7 | +0.000 [-1.241, +1.241] | +0.0197 [+0.0031, +0.0363] | +0.1064 |
| crem->ravd | wavlm | coral | 12 | 6 | -5.600 [-6.280, -4.920] | +0.0936 [+0.0418, +0.1455] | +0.0866 |
| ravd->crem | hubert | none | 6 | 11 | +4.200 [+1.979, +6.421] | +0.1148 [+0.0438, +0.1859] | +0.1087 |
| ravd->crem | hubert | zscore | 6 | 8 | +1.800 [-0.240, +3.840] | +0.0446 [+0.0073, +0.0819] | +0.0689 |
| ravd->crem | hubert | mean_shift | 7 | 8 | +0.800 [-1.240, +2.840] | +0.0376 [+0.0166, +0.0585] | +0.0690 |
| ravd->crem | hubert | coral | 5 | 7 | +1.800 [-2.795, +6.395] | +0.0680 [+0.0206, +0.1154] | +0.0670 |
| ravd->crem | wav2vec2 | none | 5 | 9 | +4.200 [+2.359, +6.041] | +0.1404 [+0.0890, +0.1918] | +0.1122 |
| ravd->crem | wav2vec2 | zscore | 6 | 6 | +0.400 [-2.716, +3.516] | +0.0302 [-0.0080, +0.0685] | +0.0760 |
| ravd->crem | wav2vec2 | mean_shift | 5 | 7 | +1.800 [-0.421, +4.021] | +0.0576 [+0.0115, +0.1038] | +0.0544 |
| ravd->crem | wav2vec2 | coral | 7 | 8 | +0.800 [-0.819, +2.419] | +0.0139 [-0.0046, +0.0323] | +0.0602 |
| ravd->crem | wavlm | none | 6 | 10 | +2.400 [-2.042, +6.842] | +0.0881 [+0.0424, +0.1339] | +0.0695 |
| ravd->crem | wavlm | zscore | 7 | 6 | -1.800 [-4.188, +0.588] | +0.0451 [-0.0033, +0.0935] | +0.0964 |
| ravd->crem | wavlm | mean_shift | 6 | 6 | +0.400 [-1.015, +1.815] | +0.0120 [-0.0028, +0.0269] | +0.0719 |
| ravd->crem | wavlm | coral | 8 | 6 | -2.000 [-5.400, +1.400] | +0.0285 [+0.0012, +0.0558] | +0.0724 |

**Cells whose gap interval excludes zero and is positive: 2 of 24.**

Pooled gap +0.00 layers [-0.57, +0.57] over 120 (cell, seed) observations; pooled cost of selecting depth on `source_val` +0.0539 [+0.0454, +0.0624] macro-F1, against +0.0905 [+0.0858, +0.0953] for a layer picked at random.

Paired difference, `source_val` minus random: **-0.0366 [-0.0448, -0.0285]**. Negative means selecting depth on `source_val` beats a coin flip; an interval covering zero means it does not.

## 3. Depth divergence at rung `none`, per backbone

The Stage 1 condition, now at 5 seeds and both directions.

| direction | backbone | gap (layers) | cost |
|---|---|---|---|
| crem->ravd | hubert | +2.800 [-0.031, +5.631] | +0.0830 [-0.0070, +0.1729] |
| crem->ravd | wav2vec2 | -3.800 [-10.094, +2.494] | +0.0432 [+0.0049, +0.0816] |
| crem->ravd | wavlm | +1.600 [-2.573, +5.773] | +0.0847 [+0.0182, +0.1511] |
| ravd->crem | hubert | +4.200 [+1.979, +6.421] | +0.1148 [+0.0438, +0.1859] |
| ravd->crem | wav2vec2 | +4.200 [+2.359, +6.041] | +0.1404 [+0.0890, +0.1918] |
| ravd->crem | wavlm | +2.400 [-2.042, +6.842] | +0.0881 [+0.0424, +0.1339] |

## 4. The curves, rung `none`, mean over 5 seeds

### cremad -> ravdess

| layer | hubert val / target | wav2vec2 val / target | wavlm val / target |
|---|---|---|---|
| 0 | 0.523 / **0.162** | 0.532 / **0.142** | 0.514 / **0.161** |
| 1 | 0.565 / **0.206** | 0.568 / **0.117** | 0.547 / **0.205** |
| 2 | 0.578 / **0.206** | 0.596 / **0.120** | 0.563 / **0.189** |
| 3 | 0.601 / **0.210** | 0.615 / **0.101** | 0.586 / **0.175** |
| 4 | 0.612 / **0.154** | 0.626 / **0.104** | 0.619 / **0.234** |
| 5 | 0.646 / **0.218** | 0.651 / **0.153** | 0.653 / **0.263** |
| 6 | 0.662 / **0.285** | 0.654 / **0.163** | 0.669 / **0.240** |
| 7 | 0.665 / **0.279** | 0.657 / **0.151** | 0.667 / **0.238** |
| 8 | 0.658 / **0.301** | 0.659 / **0.164** | 0.655 / **0.246** |
| 9 | 0.660 / **0.317** | 0.660 / **0.149** | 0.649 / **0.245** |
| 10 | 0.658 / **0.299** | 0.645 / **0.108** | 0.645 / **0.249** |
| 11 | 0.658 / **0.294** | 0.584 / **0.088** | 0.641 / **0.296** |
| 12 | 0.643 / **0.319** | 0.553 / **0.082** | 0.631 / **0.247** |

### ravdess -> cremad

| layer | hubert val / target | wav2vec2 val / target | wavlm val / target |
|---|---|---|---|
| 0 | 0.568 / **0.096** | 0.573 / **0.066** | 0.551 / **0.075** |
| 1 | 0.659 / **0.204** | 0.653 / **0.167** | 0.639 / **0.136** |
| 2 | 0.685 / **0.188** | 0.685 / **0.155** | 0.680 / **0.181** |
| 3 | 0.697 / **0.248** | 0.725 / **0.100** | 0.693 / **0.220** |
| 4 | 0.733 / **0.274** | 0.729 / **0.091** | 0.718 / **0.189** |
| 5 | 0.735 / **0.226** | 0.739 / **0.161** | 0.715 / **0.224** |
| 6 | 0.761 / **0.251** | 0.742 / **0.146** | 0.764 / **0.177** |
| 7 | 0.756 / **0.310** | 0.734 / **0.198** | 0.738 / **0.246** |
| 8 | 0.737 / **0.316** | 0.724 / **0.231** | 0.720 / **0.255** |
| 9 | 0.727 / **0.310** | 0.711 / **0.279** | 0.713 / **0.243** |
| 10 | 0.733 / **0.306** | 0.698 / **0.240** | 0.717 / **0.271** |
| 11 | 0.718 / **0.362** | 0.642 / **0.185** | 0.719 / **0.247** |
| 12 | 0.688 / **0.329** | 0.617 / **0.160** | 0.705 / **0.215** |

## 4b. Merge safety: run_ids shared with the Stage 2 grid

The sweep fixes `layer_agg='layer'`, so its **layer 6** cells have the same 19 run_id coordinates as the Stage 2 grid's `layer:6` cells and therefore the same ids. 151 ids are shared, all of them at layer 6.

| | |
|---|---|
| shared run_ids | 151 |
| non-volatile fields compared | 57 |
| fields with any mismatch | 0 |
| total mismatching values | 0 |

**Zero mismatches across 8607 compared values.** These runs were executed weeks apart, in different processes, under different launchers, on a machine that was swapping for part of the time -- and produced bit-identical metrics, confusion matrices and selected hyperparameters. That is a determinism check the project did not plan for and it is the second one to come free (the first was 23 duplicated wavlm cells in the Stage 1 sweep).

**Decision: the sweep stays a separate artifact and is NOT merged into `results/runs.jsonl`.** Phase 8 established that every row in the provenance record is one the Stage 2 enumeration produces, and verified it (`recorded but NOT enumerated: 0`). Merging would put 1820 rows at layers Stage 2 never enumerates into that file and break the invariant, for no gain: the analysis reads the shard files directly, and the shared cells are already present. The eps probe stays separate for the same reason.

## 5. Interpretation

### Frame dependence: **CONFIRMED**

Pooled over all 120 (cell, seed) correlations, the two geometries have **opposite signs and neither interval covers zero**: own -0.374 [-0.454, -0.295] against reference +0.174 [+0.099, +0.249]. 7 of 24 individual cells disagree by the stricter per-cell test.

Disagreeing cells by rung: `coral` 1, `zscore` 6.

**But it does not replicate where it was found.** At rung `none` -- the Stage 1 condition -- 0 of 6 cells disagree. The Stage 1 observation was made on 2 seeds at `none`, and at 5 seeds that specific measurement is noise: every `none` interval covers zero in at least one frame. The effect is real, and it lives on the aligned rungs rather than the unaligned one.

So the claim the paper can make is narrower and better specified than the one PROGRESS.md carried: *after per-dimension standardisation, the measured relationship between marginal discrepancy and transfer reverses sign depending on the geometry the discrepancy is measured in*. That is still a statement about the measurement rather than about our implementation, and it is now backed by 5 seeds, 3 backbones and both directions.

### Depth divergence: **DOES NOT REPLICATE as stated**

Pooled over the same cells the gap is +0.00 layers [-0.57, +0.57] -- exactly no systematic offset -- and only 2 of 24 cells have a gap interval that excludes zero and is positive.

What survives is much narrower. At rung `none` in the **ravdess->cremad** direction the gap is positive in all three backbones (+4.2, +4.2, +2.4 layers, two of three excluding zero), which is the Stage 1 condition and does replicate. In the reverse direction it does not (wav2vec2 gives -3.8), and under alignment it collapses or reverses (coral, cremad->ravdess, wavlm: -5.6).

The stronger framing -- that selecting depth on in-domain validation *systematically picks the wrong depth* -- is contradicted outright. Against a random layer, `source_val` selection is better by -0.0366 [-0.0448, -0.0285] macro-F1. It is an imperfect criterion, not an anti-correlated one, and the 0.054 it leaves on the table has to be read against the 0.091 a coin flip leaves.

**The '4-5 layers shallower, in all three backbones' claim should come out of the paper.** What can replace it is the conditional version: in the forward direction, unaligned, the transfer-optimal layer sits 2 to 4 layers deeper than the in-domain-optimal one.

### What the excluded rungs can and cannot change

`mkmmd_diag`, `mkmmd_full` are missing 369 cells.

**Cannot change either verdict at rung `none`.** That rung is 390/390 complete, so the Stage 1 replication attempt -- which is the only thing either claim originally rested on -- is fully answered.

**Cannot overturn frame dependence.** The pooled disagreement already has both intervals clear of zero on four rungs; two further rungs can add disagreeing cells or neutral ones, but cannot make existing opposite-signed intervals overlap.

**Could refine the rate and the depth verdict slightly.** The disagreement rate 7/24 is over four rungs and would be recomputed over six. For depth, both MK-MMD rungs are alignment rungs and CORAL -- the closest analogue already measured -- shows mixed gaps from -5.6 to +1.8, so they are unlikely to move a pooled +0.00 into a systematic positive. That is an expectation, not a measurement, and this section should be regenerated when the remaining cells land.
