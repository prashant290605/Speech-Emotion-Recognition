# Speech Communication Submission Checklist

**Target:** *Speech Communication* (Elsevier)  
**Article type:** Original Research Article  
**Last reviewed:** 2026-08-31

Status values: **MET** means the repository contains the item; **VERIFY** means
it needs a final PDF or portal check; **PENDING** is an author action; and
**BLOCKED** depends on a later author hand-off.

## Submission requirements

| Requirement | Status | Evidence or required action |
|---|---|---|
| Scope fit | **VERIFY** | The study is a cross-corpus speech evaluation and assessment-methodology paper. Confirm the live journal scope in the submission portal. |
| Originality, exclusive submission and co-author approval | **PENDING** | Corresponding author confirms in the portal and cover letter. |
| Review model and anonymity | **PENDING** | Confirm the journal's live review setting before upload; the manuscript is currently identified. |
| Final author order, affiliations and corresponding author | **BLOCKED** | Hand-off H. Add postal address and explicit corresponding-author marker. |
| ORCIDs | **BLOCKED** | Hand-off H. |
| CRediT contributions | **VERIFY** | Present in `paper/sections/backmatter.tex`; update after author list is final. |
| Funding and competing-interest declarations | **VERIFY** | No-funding and no-conflict statements are present; author must confirm they remain true. |
| Ethics and corpus licence compliance | **PENDING** | Confirm permissions for RAVDESS and CREMA-D, permitted research use, and institutional requirements. |
| Data and code availability | **PENDING** | Text is present; replace the repository URL with the verified Zenodo DOI before submission. Do not archive third-party raw corpora. |
| Generative-AI disclosure | **VERIFY** | Detailed disclosure is in `backmatter.tex`; confirm current Elsevier policy and portal requirement. |
| English-language proofread | **PENDING** | Complete after the compiled-PDF review. |
| Abstract, keywords, main sections, captions and editable tables | **VERIFY** | Source is complete; validate in compiled PDF. Six keywords are present. |
| Acknowledgements | **PENDING** | Fill or remove the empty section in `backmatter.tex`. |
| Bibliography integrity | **VERIFY** | 29 cited records, zero placeholders, zero probable-fabrication findings in `reports/refs_report_submission.md`. Complete the eight publisher-page spot checks in `CITATIONS_NEEDED.md`. |
| Figures and tables | **VERIFY** | Seven vector PDF figures and nine editable LaTeX tables are packaged. Check legibility and overflow in the PDF. |
| LaTeX source and bibliography style | **MET** | Official Elsevier `elsarticle` v3.3 and `elsarticle-num.bst` are vendored for the upload package. |
| Highlights | **VERIFY** | Five highlights in `paper/highlights.txt`; check the journal's live character rule. |
| Suggested reviewers | **PENDING** | Hand-off H; use institutional emails and exclude conflicts/recent collaborators. |
| Cover letter | **PENDING** | Hand-off H, after author block and Zenodo DOI are final. |
| Copyright/licence agreement | **PENDING** | Completed through Elsevier after acceptance. |

## Artefact checks

| Item | Status | Evidence |
|---|---|---|
| Structural manuscript validation | **MET** | `python tools/check_paper.py` validates inputs, graphics, citations, labels and environments. |
| Outcome-number trace | **MET** | `python tools/check_number_trace.py` traces manuscript outcomes to `reports/RESULTS.md`. |
| Reference audit | **VERIFY** | `python tools/check_refs.py --tex paper/main.tex --bib paper/refs.bib --out reports/refs_report_submission.md`. |
| Python test suite | **VERIFY** | Focused reference tests pass. The full suite is blocked here by a Windows pytest temporary-directory/process-permission issue, not a reported assertion failure; rerun in the archived environment. |
| Hostile-review pass | **MET** | `paper/ANTICIPATED_OBJECTIONS.md`; no new experiment or number was added. |
| Flat Overleaf / Editorial Manager archive | **PENDING** | Build with `powershell -ExecutionPolicy Bypass -File tools/make_overleaf_package.ps1`, then inspect the ZIP and compile in Overleaf. |
| Clean compile and PDF defect resolution | **VERIFY** | A local Tectonic compile produced a 21-page PDF after table-width and float-placement corrections. Confirm the final journal PDF in Overleaf. |
| Zenodo archive and DOI | **BLOCKED** | Hand-off H. |

## Author hand-off

1. Complete the eight publisher-page citation spot checks recorded in
   `paper/CITATIONS_NEEDED.md`.
2. Upload the generated ZIP to Overleaf and inspect the compiled PDF for
   reference rendering, table overflow, figure legibility and abstract length.
3. Finalise author block, ORCIDs, acknowledgements, funding, conflict,
   affiliations, corresponding-author address, code archive DOI and cover letter.
4. Re-check the journal's current submission requirements in the portal before
   final upload.

## Update log

| Date | Work completed |
|---|---|
| 2026-08-30 | Phase A checklist created. |
| 2026-08-30 | Phase B coherence and number-trace review completed. |
| 2026-08-31 | Review-driven language, interval, unequal-grid and run-ledger corrections completed without new experiments. |
| 2026-08-31 | BibTeX records integrated, audit extended to BibTeX/included sections, inherited `pastor2023cross` author mismatch repaired, and `pasad2021layer` updated to the ASRU proceedings record. |
| 2026-08-31 | Official Elsevier class assets vendored and flat upload packaging prepared. |
| 2026-08-31 | Manuscript reduced from 43 to 21 compiled pages. The local PDF and flat upload archive were rebuilt after a visual review of all retained tables and figures. |
