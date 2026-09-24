# Full 13-layer sweep

`ravdess` -> `cremad`, logreg, alignment rung `none`, seeds [0, 1], 78 cells. Every number is from the existing feature cache; no extraction and no alignment search was run.

The rung is fixed at `none` deliberately. This measures the *intrinsic* discrepancy and transferability of each layer; varying the rung too would confound depth with alignment.

**Not a result.** Two seeds, one pair, one classifier, no significance testing, and the layer axis is not corrected for multiplicity. Stage 2 carries the seeds.

## hubert

| layer | effect size (own) | effect size (reference) | source_val | target macro-F1 | target sd |
|---|---|---|---|---|---|
| 0 | 1570 | 21528 | 0.5599 | **0.0771** | 0.0252 |
| 1 | 2031 | 54833 | 0.6683 | **0.2172** | 0.0195 |
| 2 | 1917 | 28986 | 0.6881 | **0.1817** | 0.0204 |
| 3 | 1315 | 68118 | 0.7183 | **0.2480** | 0.0202 |
| 4 | 1097 | 23450 | 0.7449 | **0.2392** | 0.0240 |
| 5 | 985 | 27461 | 0.7577 | **0.2095** | 0.0126 |
| 6 | 954 | 22928 | 0.7774 | **0.2163** | 0.0029 |
| 7 | 902 | 30090 | 0.7753 | **0.3016** | 0.0127 |
| 8 | 832 | 66678 | 0.7466 | **0.3139** | 0.0362 |
| 9 | 788 | 52713 | 0.7396 | **0.3120** | 0.0299 |
| 10 | 801 | 45244 | 0.7308 | **0.3153** | 0.0102 |
| 11 | 842 | 78972 | 0.7145 | **0.3598** | 0.0064 |
| 12 (`last`) | 894 | 30871 | 0.7036 | **0.3383** | 0.0316 |

`source_val` peaks at **layer 6** (0.7774); target macro-F1 peaks at **layer 11** (0.3598). **They disagree** -- selecting depth on `source_val` would cost 0.1436 macro-F1 on target.

## wav2vec2

| layer | effect size (own) | effect size (reference) | source_val | target macro-F1 | target sd |
|---|---|---|---|---|---|
| 0 | 1692 | 33883 | 0.5825 | **0.0614** | 0.0092 |
| 1 | 2003 | 104311 | 0.6549 | **0.1508** | 0.0353 |
| 2 | 1579 | 51596 | 0.7001 | **0.1573** | 0.0626 |
| 3 | 1535 | 45193 | 0.7172 | **0.0989** | 0.0061 |
| 4 | 1695 | 36886 | 0.7431 | **0.0994** | 0.0065 |
| 5 | 1570 | 34826 | 0.7552 | **0.1617** | 0.0031 |
| 6 | 1658 | 37665 | 0.7452 | **0.1533** | 0.0270 |
| 7 | 1753 | 55671 | 0.7468 | **0.2325** | 0.0128 |
| 8 | 1916 | 46755 | 0.7284 | **0.2357** | 0.0299 |
| 9 | 1998 | 40955 | 0.7173 | **0.3022** | 0.0423 |
| 10 | 2040 | 76454 | 0.7266 | **0.2630** | 0.0584 |
| 11 | 1335 | 154412 | 0.6452 | **0.1749** | 0.0320 |
| 12 (`last`) | 1540 | 60656 | 0.6340 | **0.1482** | 0.0099 |

`source_val` peaks at **layer 5** (0.7552); target macro-F1 peaks at **layer 9** (0.3022). **They disagree** -- selecting depth on `source_val` would cost 0.1405 macro-F1 on target.

## wavlm

| layer | effect size (own) | effect size (reference) | source_val | target macro-F1 | target sd |
|---|---|---|---|---|---|
| 0 | 1720 | 15480 | 0.5539 | **0.0627** | 0.0076 |
| 1 | 1723 | 29433 | 0.6512 | **0.1167** | 0.0105 |
| 2 | 1969 | 24209 | 0.6854 | **0.1693** | 0.0252 |
| 3 | 1333 | 39713 | 0.6824 | **0.2023** | 0.0241 |
| 4 | 1089 | 25401 | 0.7242 | **0.1689** | 0.0335 |
| 5 | 1308 | 23639 | 0.7355 | **0.2094** | 0.0100 |
| 6 | 1088 | 40256 | 0.7734 | **0.1760** | 0.0360 |
| 7 | 990 | 26592 | 0.7645 | **0.2519** | 0.0540 |
| 8 | 1074 | 35194 | 0.7267 | **0.2728** | 0.0159 |
| 9 | 992 | 37183 | 0.7247 | **0.2579** | 0.0670 |
| 10 | 1103 | 46812 | 0.7253 | **0.3043** | 0.0074 |
| 11 | 907 | 56959 | 0.7350 | **0.2739** | 0.0220 |
| 12 (`last`) | 1171 | 39932 | 0.6900 | **0.2524** | 0.0346 |

`source_val` peaks at **layer 6** (0.7734); target macro-F1 peaks at **layer 10** (0.3043). **They disagree** -- selecting depth on `source_val` would cost 0.1282 macro-F1 on target.

## Across backbones

| backbone | argmax source_val | argmax target | gap | cost of choosing on source_val |
|---|---|---|---|---|
| hubert | 6 | 11 | 5 | +0.1436 |
| wav2vec2 | 5 | 9 | 4 | +0.1405 |
| wavlm | 6 | 10 | 4 | +0.1282 |

The depth that maximises in-domain validation is **4 to 5 layers shallower** than the depth that maximises cross-corpus transfer, in every backbone, and the gap is worth 0.13 to 0.14 macro-F1.

### Does discrepancy predict transfer across depth?

Spearman rho over the 13 layers, per backbone.

| backbone | rho(effect own, target) | rho(effect reference, target) | rho(source_val, target) |
|---|---|---|---|
| hubert | -0.769 | +0.692 | +0.104 |
| wav2vec2 | +0.445 | +0.341 | +0.396 |
| wavlm | -0.654 | +0.670 | +0.527 |

**The two discrepancy columns disagree in sign on two of three backbones.** Measured in each rung's own geometry, less discrepancy goes with better transfer; measured in the fixed ZCA reference frame, more discrepancy does. Same features, same layers, same target scores -- opposite conclusions from the choice of frame alone. Any claim of the form "lower MMD implies better transfer" has to name its geometry and defend it, and neither frame is picked out by theory.
