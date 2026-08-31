# Manuscript Coherence Audit

Date: 2026-08-30
Scope: `paper/main.tex`, all eight manuscript sections, `paper/highlights.txt`,
the generated tables, figures and `reports/RESULTS.md`.

## Findings Fixed

1. The title edit had duplicated the phrase "in cross-corpus speech emotion
   recognition". The title now contains it once.
2. `mkmmd_diag` and `mkmmd_full` had both been described as falling back to
   CORAL. The diagonal rung falls back to a diagonal moment match; the full
   rung falls back to CORAL. The Methods, Results, Discussion and generated
   results document now agree.
3. The manuscript treated the ratio of marginal and class-conditional MMD as
   an additive decomposition and as evidence directly about $P(y \mid x)$. It
   is neither. Every section now calls it a class-conditional diagnostic over
   $P(x \mid y)$ and states that it does not identify a causal mechanism.
4. Claims about source-side validation are axis-specific. In the forward
   direction it can miss the alignment choice; the recorded random-layer
   comparison shows it is informative on layer depth. The Results, Discussion,
   Conclusion and Retractions now state both facts together.
5. The Related Work description of per-dimension standardisation now matches
   Methods: it matches first and second moments per dimension, not only first
   moments.
6. Claims that performance was flat after alignment were narrowed. The text
   reports the observed gain from moving off `none` and only claims that no
   more complex rung is shown to improve on `zscore`.
7. The title, abstract, highlights, contribution list and conclusion now carry
   the same five contributions: the first-rung result; the marginal versus
   class-conditional diagnostic; the CORAL asymptote; the forward alignment
   selection failure; and frame-dependent discrepancy--transfer correlation.
8. `fig:decomposition` and `fig:confusion` were not explicitly referenced.
   Both are now cited in the Results. The per-class and confusion captions now
   identify the source-side-validation protocol used to select each seed's
   configuration.
9. The AI declaration now distinguishes automatic BibTeX-audit results from the
   remaining publisher-page spot checks. It does not claim a manual verification
   that has not occurred.

## Terminology Ledger

| Term | Required meaning |
|---|---|
| `none`, `zscore`, `mean_shift`, `coral`, `mkmmd_diag`, `mkmmd_full` | The six ordered alignment rungs. `mean_shift` is never called MMD. |
| validated | The configuration with highest `source_val`, scored once on target test. |
| oracle | The best target-test score in the completed grid; an upper bound, not an available protocol. |
| forward / reverse | RAVDESS to CREMA-D / CREMA-D to RAVDESS, respectively. |
| class-conditional diagnostic | MMD within observed labels, measuring $P(x \mid y)$; not additive with marginal MMD and not a direct estimate of $P(y \mid x)$. |
| source-side selection failure | The forward alignment result only; it must not be generalised to layer-depth selection. |

## Mechanical Checks

`python tools/check_paper.py` verifies all inputs, graphics, cited keys,
cross-references, balanced environments, control characters, bibliography
placeholders and that every figure/table label is referenced. It currently
finds 59 labels and no structural problem.

`python tools/check_number_trace.py` reads manuscript outcome numbers and
matches them to `reports/RESULTS.md`, allowing ordinary display rounding. The
generated source was extended to emit the MK-MMD fallback rates and the exact
values in the epsilon, frame and split/floor tables. Final result: **782
outcome-number occurrences traced; zero untraced outcome numbers**.

Fixed design and corpus-description values in Methods and `tables/corpora.tex`
are reported separately by the checker. They are inputs derived from the
manifest or frozen configuration rather than results outcomes; their list is
retained in the command output for review (**54 occurrences** in this pass).

## Remaining Human Work

No scientific wording decision remains from this pass. All citation placeholders
have been replaced, every citation key resolves, and the BibTeX audit reports no
probable-fabrication finding. Eight records that Crossref does not reliably index
remain on the author checklist for publisher-page spot checks.

The final PDF typesetting review, author block, ORCIDs, funding confirmation,
Zenodo archive and submission-portal steps remain human hand-offs.
