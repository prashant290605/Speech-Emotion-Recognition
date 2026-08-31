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
| Review model and anonymity | **MET** | The guide specifies single-anonymized review; the manuscript includes author and affiliation details. |
| Final author order, affiliations and corresponding author | **VERIFY** | Prashant Singh and Pranav Singh share the IIT Ropar affiliation and equal-contribution note. Prashant Singh is the corresponding author. Confirm the final legal names and contact details in the portal. |
| ORCIDs | **BLOCKED** | Hand-off H. Add each verified identifier and remove the empty-ORCID suppression in `paper/main.tex`. |
| CRediT contributions | **VERIFY** | Present in `paper/sections/backmatter.tex`; update after author list is final. |
| Funding and competing-interest declarations | **VERIFY** | No-funding and no-conflict statements are present; author must confirm they remain true. |
| Ethics and corpus licence compliance | **PENDING** | Confirm permissions for RAVDESS and CREMA-D, permitted research use, and institutional requirements. |
| Data and code availability | **PENDING** | Text is present; replace the repository URL with the verified Zenodo DOI before submission. Do not archive third-party raw corpora. |
| Generative-AI disclosure | **VERIFY** | Detailed disclosure is in `backmatter.tex`; confirm current Elsevier policy and portal requirement. |
| English-language proofread | **PENDING** | Complete after the compiled-PDF review. |
| Abstract, keywords, main sections, captions and editable tables | **VERIFY** | The abstract is 179 words, six keywords are present, sections are numbered, and tables are editable. Validate in the compiled PDF. |
| Acknowledgements | **MET** | No acknowledgements section is present. Add one only if an author confirms a required acknowledgement. |
| Bibliography integrity | **VERIFY** | 29 cited records, zero placeholders, zero probable-fabrication findings in `reports/refs_report_submission.md`. Complete the eight publisher-page spot checks in `CITATIONS_NEEDED.md`. |
| Figures and tables | **VERIFY** | Seven vector PDF figures and nine editable LaTeX tables are packaged. Check legibility and overflow in the PDF. |
| LaTeX source and bibliography style | **MET** | Elsevier CAS single-column v2.4, `cas-common.sty`, and `cas-model2-names.bst` are vendored for the upload package. |
| Highlights | **MET** | Five separate highlights are in `paper/highlights.txt`; each is at most 85 characters including the bullet. |
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
| Self-contained Overleaf archive | **MET** | `output/overleaf/cas-v2-authors/Speech_Communication_submission_20260831.zip` passed static path, asset, graphics, citation, and bibliography checks. The only nested path is the official CAS email-icon asset. |
| Clean compile and PDF defect resolution | **VERIFY** | A local Tectonic compile of the CAS package produced an 11-page PDF. All pages were visually inspected. Confirm the final journal PDF in Overleaf. |
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
| 2026-08-31 | Replaced the prior class with the supplied official Elsevier CAS single-column template (cas-sc v2.4). Applied the guide's author-date citation style, abstract limit, separate highlights file, and end-of-manuscript generative-AI declaration requirement. |
| 2026-08-31 | Rebuilt the self-contained CAS archive and PDF. The package includes the official CAS email-icon asset at its required relative path and passed static validation. The 11-page CAS PDF was visually inspected. |
| 2026-09-01 | Added Pranav Singh as the second author with the shared IIT Ropar affiliation, email address, equal-contribution note, and matching CRediT roles. Rebuilt and visually checked the 11-page CAS PDF. |
