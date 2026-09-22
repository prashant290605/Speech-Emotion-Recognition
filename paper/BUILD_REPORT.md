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


---

# Phase 1 and 2 additions

2026-09-22, after the delivery recorded above. No experiment was rerun, no
result row edited and no cached feature touched. The ledger SHA256 is unchanged
at `51b8ff64d500d77d16c047430802645fb7bf86bea351258d3d03cee7381b1407`.

## What changed in the build

The build is now one command, `python tools/build_paper.py`, running nine
stages and stopping at the first failure: frozen-ledger digest, tracked
artifacts present, reports, tables and figures, translation audit,
`check_paper.py`, `check_number_trace.py`, Tectonic compile, then a log
inspection that fails on any undefined reference or citation. `--check-only`
verifies inputs without generating; `--analysis` regenerates without
compiling. It does not download a compiler: a missing Tectonic is reported
with where to obtain it.

## Reproducibility inputs

Three derived artifacts are now tracked, so the analyses that needed
machine-local state no longer do:

- `results/layer_sweep_v2.jsonl` (2340 rows). 2160 of these previously lived
  only in gitignored `results/shards/`, which made the frame-dependence
  subsection unreproducible from a checkout.
- `data/manifest_portable.csv` (8882 rows). Scientific metadata with
  corpus-relative audio paths; the absolute-path manifest stays local.
- `results/speaker_confusions.jsonl.gz` (5364 runs, 2.9 MB). The sufficient
  statistic for every cluster-bootstrap interval, replacing 7730 gitignored
  prediction files totalling 81 MB.

Each has a provenance sidecar and a `--check`/`--verify` mode. Equivalence for
the third was proved across all 5364 runs: identical per-speaker tensors,
identical pooled confusions, zero difference in macro-F1, per-class F1, and a
full paired bootstrap under identical seeds.

**No published number changed.** All 13 generated tables and all 14 figures are
byte-identical before and after the refactor.

## Test count

549 collected, 549 passed, 0 failed, 0 skipped, 1m41s. Phase 1 took the suite
from 463 to 494; Phase 2 added the artifact, manifest, freeze-tag and
generated-table tests. Tests marked `ledger` read committed repository
artifacts; `-m "not ledger"` deselects them.

## Corrections carried into the PDF

The consistency-check counts in Section 6.2 were stale. The sweep/grid
agreement is 180 shared run ids over 57 non-volatile fields, 10260 values, zero
mismatches; the manuscript previously said 151 and 8607, which was the count
before the per-backbone sweep shards were added. `reports/RESULTS.md` had also
gone stale relative to its own generator. Both numbers are now derived by
`tools/make_results_doc.py` rather than asserted in prose, and the check passes
more strongly than was claimed, not less.

Methods now states why the source-selected summary in Table 2 uses a
t-interval over seeds while fixed-arm contrasts use the paired cluster
bootstrap, and what the oracle column is and is not. No statistic changed.

## Compile

`python tools/build_paper.py` produces `output/paper/Speech_Communication.pdf`:
22 pages, 0 undefined references, 0 undefined citations, 2 overfull hboxes.

A latent defect surfaced on the first scripted end-to-end build and is fixed.
`tools/report_reference_mmd_diagnostics.py` promoted a single-column float to
`table*` by string surgery; once `ser.latex.table` began emitting `table*`
directly, the promotion became a no-op and the trailing `rsplit` stopped
matching, appending a second closing tag. The committed
`tables/decomposition.tex` predated the change, so the manuscript still
compiled from the stale file while its generator produced LaTeX that did not.
The regenerated table is now byte-identical to the committed one, and
`tests/test_figures_and_latex.py` checks every generated table for balanced
environments.

## Known warnings, unchanged

The title-block overfull box and the `headline.tex` overfull box (4.59 pt) are
pre-existing and were present in the delivery above. The local Tectonic setup
still reports a fontconfig warning. No new warning was introduced: the audit
table's panel headings use bold rather than italic specifically to avoid
a font substitution at 9 pt that an earlier draft introduced.

