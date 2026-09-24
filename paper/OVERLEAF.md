# Overleaf Package

Create the upload-ready project from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/make_overleaf_package.ps1
```

The command creates a dated ZIP in `dist/`. In Overleaf, select **New Project**,
then **Upload Project**, and choose that ZIP. The project root contains
`main.tex`, so no main-file setting is required. All manuscript source files
and figures are at the project root. The single `thumbnails/` subdirectory is
an official CAS asset required to render the corresponding-author email icon.

The package includes only the files needed to compile the manuscript:

```
main.tex
refs.bib
highlights.txt
cas-dc.cls
cas-common.sty
cas-model2-names.bst
thumbnails/cas-email.jpeg
*.tex (sections and tables)
*.pdf (figures)
OVERLEAF.md
supplementary.tex
```

Overleaf should run the required LaTeX/BibTeX passes automatically. The expected
engine is pdfLaTeX with BibTeX. The archive carries Elsevier's official CAS
double-column template (cas-dc v2.4, 2024-05-04), its required common style,
and the CAS author-date bibliography style, so it does not depend on an
installed template version.

Before submission, check the compiled PDF for table overflow, figure legibility
at column width, complete reference rendering, the 250-word abstract limit, and
the separate highlights file. The package contains no citation placeholders; see
`reports/refs_report_submission.md` in the repository for the BibTeX audit and
the final publisher-page spot-check list.

## Analytical rebuild (2026-09-22)

The main file is still `main.tex`. The new proof section is
`selection_theory.tex` and its frozen-run audit table is
`translation_audit.tex`; both are included automatically. The archive contains
35 bibliography entries and no unresolved citation placeholders. The current
reference-check report is `reports/refs_report_research_rebuild.md` in the
repository; Crossref absence is not a verdict that a proceedings paper is
invalid. See `paper/CITATIONS_NEEDED.md` for remaining author spot checks.

Inspect the equations in Section 3, the exact-equality audit table, the author
ORCIDs and email, and all full-width tables. The proof and matched-row audit
are retrospective and do not add training runs. Confirm that the final
manuscript states this scope before submitting. Build `supplementary.tex` separately for the supplementary PDF, which now
carries the robustness controls, extended diagnostics and the provenance
ledger of withdrawn claims.
