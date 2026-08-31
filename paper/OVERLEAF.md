# Overleaf Package

Create the upload-ready project from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/make_overleaf_package.ps1
```

The command creates a dated ZIP in `dist/`. In Overleaf, select **New Project**,
then **Upload Project**, and choose that ZIP. The project root contains
`main.tex`, so no main-file setting is required. The ZIP is intentionally flat:
the same archive is suitable for Elsevier Editorial Manager, which does not
process LaTeX submissions containing subfolders.

The package includes only the files needed to compile the manuscript:

```
main.tex
refs.bib
highlights.txt
elsarticle.cls
elsarticle-num.bst
*.tex (sections and tables)
*.pdf (figures)
OVERLEAF.md
```

Overleaf should run the required LaTeX/BibTeX passes automatically. The expected
engine is pdfLaTeX with BibTeX. The archive carries Elsevier's official
`elsarticle` class (v3.3, 2020-11-20) and numbered bibliography style, so it
does not depend on an installed template version.

Before submission, check the compiled PDF for table overflow, figure legibility
at column width, complete reference rendering, and the journal's abstract-length
requirement. The package contains no citation placeholders; see
`reports/refs_report_submission.md` in the repository for the BibTeX audit and
the final publisher-page spot-check list.
