# Research rebuild delivery checks

2026-09-22

## Deliverables

- `output/pdf/Speech_Communication_research_rebuild_20260922.pdf`: 21 pages.
- `output/pdf/Speech_Communication_provenance_20260922.pdf`: one page.
- `output/overleaf/20260922-final/Speech_Communication_submission_20260922.zip`:
  self-contained official CAS double-column sources, 38 files, 24 TeX files,
  seven vector-PDF figures, bibliography, highlights and template assets.

## Checks completed

- `pytest` on the affine audit, number tracing, figure/LaTeX invariants,
  bibliography parser and existing global z-score bounds: 44 passed.
- `tools/check_paper.py`: 35 cited bibliography entries, no placeholders and
  no structural problems.
- `tools/check_number_trace.py`: all scanned outcome numbers trace to the
  generated reports. The translation report is a separate retrospective source;
  this check is not a substitute for reviewing what each number means.
- `tools/verify_overleaf_package.py`: passed on the delivered ZIP/staging pair.
- Local Tectonic 0.15.0 compile, using official CAS v2.4 with BibTeX: completed.
- All article pages were rendered for layout review; revised proof, audit table,
  corpus table, plots, author metadata, references and supplement were checked.
  No unresolved `??` markers or text spans outside the page were detected.
- The abstract has 196 whitespace-delimited words; all four highlights satisfy
  the 85-character limit.
- No em dashes or triple-hyphen em-dash commands remain in the article sources.
- The original ledger SHA256 remains
  `51b8ff64d500d77d16c047430802645fb7bf86bea351258d3d03cee7381b1407`.
  No training run was executed and no result row or cached feature was edited.

## Warnings and remaining checks

- The compiler emits nonfatal underfull-box warnings, a title-layout box warning,
  and a small headline-table box warning. Visual inspection found no clipping.
  The local Tectonic/fontconfig setup reports a fontconfig warning and substitutes
  an unavailable italic typewriter shape. Rendered mathematical glyphs are intact.
- BibTeX warns about absent page ranges for Hu et al. and Adam. No page range
  was invented. The DOI/proceedings or OpenReview link remains in each record.
- Crossref confirms 23 of 35 records. Four additional PMLR records were verified
  on primary proceedings pages. Eight older publisher-page checks remain listed
  in `CITATIONS_NEEDED.md`.
- The full legacy test suite was not rerun; this pass used the 44 focused tests.
- Authors must approve the retrospective contribution, proof assumptions,
  metadata, declarations and final Overleaf output. Zenodo archiving and portal
  submission are not completed by this build.

## Interpretation corrections carried into the PDF

The source-translation result is not claimed for z-score or arbitrary neural
training. Affine reparameterisation is not claimed as new linear algebra.
The full-grid headline and ladder retain available two-seed Transformer cells;
the new translation audit and matched-direction comparison exclude that arm.
Candidate-level discrepancy and source-selected performance summaries are
distinguished. Shared-cell correlation means are descriptive. These limitations
are part of the scientific account, not conditions hidden outside the paper.
