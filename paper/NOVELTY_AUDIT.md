# Research and novelty audit

Date: 2026-09-09. Scope: a retrospective analysis of frozen runs, not a new
training campaign. No cached features or experimental result rows are changed.

## Review verdict

The previous manuscript has a reproducible case study but an unfocused claim.
Five observations do not constitute five independent scientific contributions.
Failure of marginal alignment, difficulty of unsupervised model selection, and
kernel dependence of discrepancy are established problems. A two-corpus grid
cannot establish conditional shift as the cause of transfer failure.

The strongest available contribution is a specific, testable limitation:
**source validation can be exactly invariant to an adaptation operation that
changes target predictions.** Source translations and stationary-kernel SVMs
provide an analytically tractable case. The existing ledger can test the
predicted equality without fitting another classifier. Affine factorisation
then distinguishes this exact result from the separate, non-invariant z-score
comparison. This is an audit of a selection criterion, not a new recogniser.

## Closest primary sources

| Source | Already established | Boundary for this paper |
|---|---|---|
| [Zhao et al., ICML 2019](https://proceedings.mlr.press/v97/zhao19a.html) | Invariant marginal representations and low source error need not yield successful adaptation. | Do not claim discovery of the general failure of marginal alignment. |
| [Sun et al., AAAI 2016](https://doi.org/10.1609/aaai.v30i1.10306) | CORAL and feature/weight transformation equivalence. | Affine reparameterisation is an organising identity, not a new learning algorithm. |
| [Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html) | Representation comparison depends on the transformations to which the measure is invariant; unrestricted affine invariance can itself discard information. | Do not prescribe an invariant metric as universally correct. |
| [Hu et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f50cebc22663df45ce619645bfabb3b3-Abstract-Datasets_and_Benchmarks_Track.html) | Model-selection reliability across UDA methods and settings; an ensemble selector with a worst-case guarantee. | Do not claim all unsupervised selectors fail or that no alternatives exist. |
| [Gretton et al., JMLR 2012](https://www.jmlr.org/papers/v13/gretton12a.html) | MMD as a kernel-dependent two-sample statistic. | A kernel pullback is an explanation of the audit, not a newly invented discrepancy. |

This targeted search does not certify priority for a theorem or guarantee
publication. The defensible originality is the precisely delimited analytical
question, its executable audit, and its manifestation in frozen SER runs.

## Errors to repair before presenting the contribution

1. The moment ladder is not nested. Domain-wise z-score also changes common
   classifier coordinates; scalar regularisation searches do not remove this.
2. A source-fitted scaler shared by both domains does not make `none` equivalent
   to domain-wise `zscore`.
3. The covariance rank bound does not imply singularity when n exceeds d.
4. Regularised CORAL matches regularised, not necessarily empirical, covariances.
5. Mean matching uniquely chooses the offset within translations, not within
   all affine maps. The squared empirical-mean identity is not the unbiased
   finite-sample linear-kernel estimator.
6. The MK-MMD anchor penalises W, not b. Its ideal large-penalty limit is not
   necessarily identity. The actual finite-step algorithm has no such guarantee.
7. Fallback rejects a map using an unpenalised finite-sample MMD check. It is not
   proof that the penalised training objective failed to improve.
8. Adaptive MMD scale invariance requires scaling both domains, not just source.
9. Table ladder performance uses selected cells but discrepancy averages use
   candidate rows. They cannot be interpreted as matched estimands.
10. Correlations pooled across shared seeds/cells are descriptive; a t interval
    treating those cells as independent is not a generalisation guarantee.
11. Raw RAVDESS is not balanced over all eight labels: neutral has 96 items.
12. Small prior KL does not predict that prior correction must hurt. Neither
    this observation nor class-conditional MMD identifies a causal mechanism.

## Scope of the new evidence

Join `none` and `mean_shift` by corpus direction, seed, backbone, layer,
classifier, label-map, split, feature and search specifications. Require one
row per arm. Report every eligible match, including non-equalities. Keep
logistic regression as a solver-sensitive contrast. Do not infer independent
replications from multiple cells sharing a seed. Do not run a significance
test on rounded equality or use target scores to select the matched cells.

The mathematical argument requires matching hyperparameter candidates and
translation-equivariant fitting/tie-breaking. It does not apply to penalised
intercepts, arbitrary neural training, or `none` versus `zscore`.
