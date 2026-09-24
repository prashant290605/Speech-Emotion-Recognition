# Stage 1 screening

**Every figure here is `source_val` or a diagnostic. Target scores are deliberately excluded** — pruning an axis on target performance is the leak Phase 2 exists to prevent, moved one level up.

## alignment

| alignment | n | mean source_val | min | max |
|---|---|---|---|---|
| coral | 97 | 0.6121 | 0.4074 | 0.8411 |
| mean_shift | 19 | 0.7399 | 0.6223 | 0.8457 |
| mkmmd_diag | 109 | 0.7426 | 0.6240 | 0.8453 |
| mkmmd_full | 112 | 0.5691 | 0.4074 | 0.8369 |
| none | 22 | 0.7572 | 0.6223 | 0.8614 |
| zscore | 19 | 0.7472 | 0.6271 | 0.8492 |

## classifier

| classifier | n | mean source_val | min | max |
|---|---|---|---|---|
| logreg | 86 | 0.6503 | 0.4786 | 0.8419 |
| mlp | 120 | 0.6874 | 0.4711 | 0.8492 |
| svm_linear | 80 | 0.6541 | 0.5105 | 0.8133 |
| svm_rbf | 80 | 0.6051 | 0.4074 | 0.8270 |
| transformer | 12 | 0.8181 | 0.7453 | 0.8614 |

## layer_agg

| layer_agg | n | mean source_val | min | max |
|---|---|---|---|---|
| last | 170 | 0.6299 | 0.4074 | 0.8220 |
| layer | 164 | 0.6663 | 0.4192 | 0.8614 |
| weighted | 44 | 0.7410 | 0.6288 | 0.8608 |

## CORAL: source_val against shrinkage epsilon

CORAL cost 0.166 of `source_val` at eps=1e-4 in Stage 0. At an effective rank near 57 of 768, weak shrinkage amplifies hundreds of near-null directions, so that cost may be a property of the regularisation rather than of CORAL. **If `source_val` recovers at larger eps while target stays flat, the paper must say so.**

| eps | n | mean source_val | mean effect size |
|---|---|---|---|
| 0.0001 | 22 | 0.5927 | 8.31 |
| 0.001 | 18 | 0.5713 | 5.72 |
| 0.01 | 18 | 0.6032 | 8.57 |
| 0.1 | 21 | 0.6770 | 23.11 |
| ledoit-wolf | 18 | 0.6096 | 9.79 |

## MK-MMD fallback rate

How often the optimiser failed to beat its own warm start. A rung that reverts most of the time is its warm start wearing a different label, and any table containing it must state this rate.

| rung | n | fallback fired | rate |
|---|---|---|---|
| mkmmd_diag | 109 | 30 | 27.5% |
| mkmmd_full | 112 | 68 | 60.7% |

## Pruning decisions

_To be filled in by hand before Stage 2, with the rationale. An axis is pruned only on the evidence above._

