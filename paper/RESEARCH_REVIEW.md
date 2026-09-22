# Review of the analytical rebuild

2026-09-22. This supplements, rather than silently replaces, the earlier
anticipated-objections register.

## Strongest remaining objections

1. **The kernel identity is elementary.** Correct. The paper does not claim a
   new kernel theory or algorithm. Its contribution is the explicit criterion
   failure, the independently reproducible frozen-run audit, and the boundary
   between this exact result and the non-invariant z-score comparison. A venue
   can still judge that contribution too narrow; wording cannot remove that risk.
2. **The research question is retrospective.** Correct and disclosed in the
   introduction, results and limitations. No claim of preregistration is made
   for the translation audit. All eligible pairs, including logistic-regression
   non-equalities, are retained.
3. **Only two acted English corpora.** The empirical scope remains limited.
   The theorem has assumptions, not a claim about frequency in all SER systems.
   No generalisation to spontaneous or multilingual emotion recognition follows.
4. **The ledger stores winners, not every trial's validation predictions.**
   The empirical audit verifies the winning score and selected parameters only.
   The full validation-surface statement is analytical, not empirically certified.
5. **More cells do not imply independent replication.** Matched cells share seeds,
   representations and speakers. The new table reports descriptive differences
   without an independence-based significance test.
6. **Ideal invariance need not survive numerical fitting.** Logistic regression
   makes this limitation visible. The stationary-kernel proposition conditions
   on a fitting rule determined by invariant matrices and consistent tie-breaking.
7. **No remedy for target-risk selection is evaluated.** The audit identifies a
   limitation, not a replacement selector. DEV and contemporary selection work
   are cited; their failure is not inferred from this result.
8. **The alignment grid confounds transport and classifier geometry.** Correct.
   The affine factorisation now states this explicitly. Current scores evaluate
   pipelines; they cannot causally identify which matched moments explain gains.
9. **MK-MMD fallback limits interpretation.** The main text distinguishes its
   unpenalised acceptance check from its penalised objective. Diagnostics from
   the rejected attempted fit are not treated as final-map convergence evidence.
10. **Absolute scores and the Transformer arm are not competitive evidence.**
    No state-of-the-art claim is made. The reduced-seed arm is supporting context,
    not the source of the new central result.

## Corrections made without training

- Removed the claim that covariance dimension/sample count forces singularity.
- Distinguished regularised covariance matching from empirical matching.
- Restricted the uniqueness of mean-shift minimisation to translations.
- Removed the false claim that an unpenalised offset vanishes at large MK-MMD
  regularisation.
- Distinguished a source-fitted shared scaler from domain-wise z-score.
- Removed pooled independence-based correlation intervals from the main table;
  its previously reported means are unchanged.
- Corrected the discrepancy table's averaging description and removed the
  claimed causal interpretation of juxtaposed, differently selected summaries.
- Removed the false balance statement for eight-label RAVDESS.
- Kept all stored experimental results and cached features unchanged.

## Author sign-off still required

The authors should review the proof assumptions and contribution scope, verify
the retained citation spot checks, and confirm author metadata and declarations.
The new central contribution is stronger and narrower than the earlier grid
narrative. It is not a guarantee of novelty priority or journal acceptance.
