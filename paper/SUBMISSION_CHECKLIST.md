# Speech Communication Submission Checklist

**Target:** *Speech Communication* (Elsevier)  
**Article type:** Original Research Article  
**Last reviewed:** 2026-09-22

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
| ORCIDs | **VERIFY** | Both identifiers are present in `paper/main.tex`; authors must confirm them. |
| CRediT contributions | **VERIFY** | Present in `paper/sections/backmatter.tex`; update after author list is final. |
| Funding and competing-interest declarations | **VERIFY** | No-funding and no-conflict statements are present; author must confirm they remain true. |
| Ethics and corpus licence compliance | **PENDING** | Confirm permissions for RAVDESS and CREMA-D, permitted research use, and institutional requirements. |
| Data and code availability | **VERIFY** | Text includes the verified public GitHub repository URL. Add a Zenodo DOI after the archival release; do not archive third-party raw corpora. |
| Generative-AI disclosure | **MET** | `backmatter.tex` names OpenAI ChatGPT and OpenAI Codex, states their drafting and software-assistance roles, and records author review and responsibility. Confirm the portal field before final upload. |
| English-language proofread | **VERIFY** | The compiled 21-page article was reviewed for readability and layout. Authors must complete the final domain proofread before portal submission. |
| Abstract, keywords, main sections, captions and editable tables | **VERIFY** | The current abstract contains 196 whitespace-delimited words, six keywords are present, and numbered sections and editable tables compile. Confirm the portal word count and final Overleaf output. |
| Acknowledgements | **MET** | No acknowledgements section is present. Add one only if an author confirms a required acknowledgement. |
| Bibliography integrity | **VERIFY** | 35 cited records, no placeholders or probable-fabrication flags; 23 Crossref confirmations. Four further PMLR additions were checked on primary pages. Eight older publisher spot checks remain. See `CITATIONS_NEEDED.md`. |
| Figures and tables | **VERIFY** | Seven vector-PDF figures and thirteen editable tables compile. Page overviews and detailed checks of revised pages found no clipping. Confirm legibility in Overleaf. |
| LaTeX source and bibliography style | **MET** | Elsevier CAS double-column v2.4, `cas-common.sty`, and `cas-model2-names.bst` are vendored for the upload package. |
| Highlights | **MET** | Four highlights, each within 85 characters including the bullet. |
| Suggested reviewers | **PENDING** | Hand-off H; use institutional emails and exclude conflicts/recent collaborators. |
| Cover letter | **PENDING** | Hand-off H, after author block and Zenodo DOI are final. |
| Copyright/licence agreement | **PENDING** | Completed through Elsevier after acceptance. |

## Artefact checks

| Item | Status | Evidence |
|---|---|---|
| Structural manuscript validation | **MET** | `python tools/check_paper.py` validates inputs, graphics, citations, labels and environments. |
| Outcome-number trace | **MET** | `python tools/check_number_trace.py` traces outcomes to generated reports, including the separate retrospective translation audit. |
| Reference audit | **VERIFY** | Current output: `reports/refs_report_current.md` (article plus supplement); automatic flags and primary-page verification are distinguished in the citation register. |
| Python test suite | **VERIFY** | All 44 focused audit, number-trace, figure/table, reference and global-bound tests pass. The complete legacy suite was not rerun in this no-training pass. Its earlier long-running CPU test remains a release check. |
| Hostile-review pass | **MET** | `paper/RESEARCH_REVIEW.md` updates the earlier objections for the new analytical claim. No training experiment was added; retrospective summaries are explicitly identified. |
| Self-contained Overleaf archive | **MET** | `output/overleaf/20260922-final/Speech_Communication_submission_20260922.zip`: 38 files, 24 TeX files, seven vector-PDF figures. Package structure, relative paths, citations and required bibliography fields pass. |
| Local compile and PDF inspection | **MET** | Official CAS double-column sources compile to 21 pages, plus a one-page provenance supplement. No unresolved citations/references or off-page text spans were detected. Remaining nonfatal TeX warnings are recorded in `paper/BUILD_REPORT.md`. Overleaf confirmation remains a human check. |
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

## Research rebuild status (2026-09-22)

The September 3 PDF and archive are superseded drafts. The manuscript now
centres the exact translation-selection audit, with a separate affine lemma
and explicit retrospective scope. The local compile, visual inspection and
archive validation are complete. Scientific author sign-off and the human
submission requirements above remain pending.
The read-only translation audit and all 44 focused tests pass. It adds retrospective
summaries of existing scores, not training runs. The manuscript must distinguish
this post-hoc question from the original pre-specified comparison family.
Five new related-work records were checked against primary proceedings pages;
this does not close the earlier eight-record manual spot-check list. Author
ORCIDs are present in the current source and still require author confirmation.
Zenodo, final declarations and portal submission remain human hand-offs.

## Update log

| Date | Work completed |
|---|---|
| 2026-08-30 | Phase A checklist created. |
| 2026-08-30 | Phase B coherence and number-trace review completed. |
| 2026-08-31 | Review-driven language, interval, unequal-grid and run-ledger corrections completed without new experiments. |
| 2026-08-31 | BibTeX records integrated, audit extended to BibTeX/included sections, inherited `pastor2023cross` author mismatch repaired, and `pasad2021layer` updated to the ASRU proceedings record. |
| 2026-08-31 | Official Elsevier class assets vendored and flat upload packaging prepared. |
| 2026-08-31 | An intermediate condensation reduced the manuscript from 43 to 21 compiled pages. This intermediate version was superseded after the 2026-09-01 content audit. |
| 2026-08-31 | Replaced the prior class with the supplied official Elsevier CAS single-column template (cas-sc v2.4). Applied the guide's author-date citation style, abstract limit, separate highlights file, and end-of-manuscript generative-AI declaration requirement. |
| 2026-08-31 | Rebuilt the self-contained CAS archive and PDF. The package includes the official CAS email-icon asset at its required relative path and passed static validation. The 11-page CAS PDF was visually inspected. |
| 2026-09-01 | Added Pranav Singh as the second author with the shared IIT Ropar affiliation, email address, equal-contribution note, and matching CRediT roles. Rebuilt and visually checked the 11-page CAS PDF. |
| 2026-09-01 | Audited the earlier condensation against the complete manuscript. Restored 6,178 words of scientific narrative, all detailed methods and results text, and the reproducibility section. No result, experiment, or numerical claim was added. |
| 2026-09-01 | Confirmed that published Speech Communication articles use a two-column production layout. Switched the manuscript to the supplied official Elsevier CAS double-column class (`cas-dc` v2.4). |
| 2026-09-01 | Rebuilt the final flat CAS archive. Static package validation, manuscript structure validation, outcome-number tracing, focused citation tests, and a visual review of all 19 PDF pages passed. |
| 2026-09-01 | Reviewer-revision Phase 1 completed: bounded the paper to a RAVDESS--CREMA-D case study, replaced `cost of selection` with `oracle gap`, replaced `chance floor` with `chance baseline`, limited MK-MMD claims to the evaluated affine maps and budget, removed the prospective DOI statement, and made the OpenAI ChatGPT/Codex disclosure specific. |
| 2026-09-01 | Reviewer-revision Phase 2 completed: replaced the conditional-to-marginal ratio with raw and separately normalised diagnostics, specified the speaker-and-seed paired cluster bootstrap, added a tested median-heuristic scaling proposition, and documented RBF saturation under a globally fixed source-defined bandwidth. |
| 2026-09-01 | Reviewer-revision Phase 3 completed: ran and integrated a pre-specified 40-run RAVDESS-calm-drop control. The generated five-seed table shows the alignment improvement without merging `calm` into `neutral`; it is explicitly scoped to HuBERT final-layer logistic regression and four control rungs. |
| 2026-09-01 | Reviewer-revision Phase 4 completed: added the separately hashed five-class neutral-exclusion control, completed both mapping-specific MMD diagnostic ledgers, and reported them as descriptive adaptive-geometry results rather than a conditional-shift test. |
| 2026-09-02 | Reviewer-revision Phase 6 completed: added verified current related work, moved the detailed provenance ledger to a one-page supplement, repaired generated-table escaping, compiled and visually reviewed the 20-page article and one-page supplement, and created the delivery archive. |
| 2026-09-02 | Review-precision corrections completed: repaired the abstract and measurement-protocol wording, added paired label-control intervals from stored predictions, marked uncorrected cell counts descriptive, rendered saturated MMD values in scientific notation, regenerated the affected figures and tables, restored full-width tables, and visually reviewed the revised 21-page PDF. |
| 2026-09-02 | Global multiplicity correction completed: the z-score claim now uses one eight-contrast family and the two label-harmonisation controls use one twelve-contrast family. Oracle, baseline, availability, and discrepancy-column wording were also clarified. |
| 2026-09-03 | Final reviewer wording pass completed: calibrated the conclusion's z-score statement, made the Methods oracle definition match Table 3, and narrowed an overbroad layer-selection sentence. No result, table, figure, or experiment changed. |
