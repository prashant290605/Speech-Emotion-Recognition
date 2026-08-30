# Speech Communication Submission Checklist

**Target:** *Speech Communication* (Elsevier)  
**Article type:** Original Research Article  
**Last reviewed:** 2026-08-30  
**Source of requirements:** [Elsevier Guide for Authors -- Your Paper Your Way](https://www.elsevier.com/en-gb/subject/next/guide-for-authors), accessed 2026-08-30. The journal-specific ScienceDirect guide must be rechecked in the submission portal because automated access returned 403.

Status values: **MET** means the repository currently contains the required item;
**PENDING** requires a human action or a later phase; **BLOCKED** cannot be
completed until an explicit dependency is available; **VERIFY** means the
manuscript appears to satisfy the requirement but must be checked in the
compiled PDF or submission portal.

## Journal and submission requirements

| Requirement | Status | Evidence or required action |
|---|---|---|
| In-scope original research article | **VERIFY** | The paper addresses cross-corpus speech emotion recognition and speech representation evaluation. Confirm scope fit against the live journal page before submission. |
| Originality, exclusive submission, approval by all authors and responsible authorities | **PENDING** | Corresponding author must confirm in the portal and cover letter. Do not submit while the author list or institutional approval is unsettled. |
| Peer-review mode and anonymity | **PENDING** | The current generic guide says single-anonymized review; the manuscript contains author identity. Confirm Speech Communication's live setting before upload. |
| Corresponding author designated with email and full postal address | **PENDING** | `main.tex` has an email and institution/country, but no postal address or explicit corresponding-author marker. Resolve with the author block. |
| Definitive author list, order, affiliations and contact details | **PENDING** | Author/co-author decision belongs to Hand-off H. |
| ORCIDs recorded where available | **PENDING** | Add verified ORCID identifiers to the author block after author list is final. |
| CRediT contribution statement | **MET, pending author-list update** | Present in `sections/backmatter.tex`; revise if a co-author is added. |
| Conflict-of-interest declaration generated/uploaded through Elsevier tool | **PENDING** | Manuscript statement exists; corresponding author must create and upload Elsevier's required declaration file. |
| Funding statement and funder role | **MET** | No-funding statement is present in `sections/backmatter.tex`. Confirm it remains true after co-author review. |
| Ethics / human-data compliance | **VERIFY** | Ethics statement and data-availability text are present. Confirm corpus licences, permitted research use, and any institutional requirements before submission. |
| Data availability statement | **MET, pending archive link** | Third-party-corpus restriction and reproducibility route are stated. Replace the repository-only code link with the Zenodo archive DOI before submission. |
| Research code/data sharing where appropriate | **PENDING** | Create the Zenodo archive and test the archived artefact; raw third-party corpora must not be redistributed. |
| Generative-AI declaration conforms to current publisher policy | **VERIFY** | A detailed declaration is present. Corresponding author must check current Elsevier policy and any required portal disclosure immediately before submission. |
| Good, consistent English and final proofreading | **PENDING** | Human proofread after compiled-PDF review. Use one English variant consistently. |
| Essential article elements: abstract, keywords, introduction, methods, results, conclusion, captions and tables | **MET, pending compile** | All are present in source; verify rendering in Phase G. |
| Stand-alone factual abstract; no references; define non-standard abbreviations if retained | **PENDING** | No citations appear in the abstract. Phase B checks terminology; compiled output must be read once. |
| Maximum six keywords | **MET** | Six keywords in `main.tex`. |
| Acknowledgements placed before references | **PENDING** | Placeholder section is present; author must fill it or delete the empty section in Hand-off H. |
| References complete and internally consistent; DOI encouraged | **BLOCKED** | 23 citation records are pending verified metadata; see citation register below. |
| Figures numbered, cited, captioned and supplied in an accepted format | **MET, pending compiled-PDF review** | Six self-generated PDF figures are in `figures/`; captions and references are checked in Phase B/E. |
| Tables editable, numbered, cited, captioned, without vertical rules | **MET, pending compiled-PDF review** | Nine generated LaTeX tables use `booktabs`; check overflow in Phase G. |
| Figure artwork usable at submission quality | **VERIFY** | Source figures are vector PDFs. Check print/column-width legibility in Phase G. |
| Permissions for third-party material | **MET** | All figures and tables are generated by this project; no third-party artwork is included. Recheck if any item is added. |
| LaTeX source prepared with `elsarticle` and BibTeX | **MET, pending clean compile** | `main.tex` uses `elsarticle`; Phase E/G must verify a clean compile. |
| Editable source files and all figures supplied at submission | **PENDING** | Assemble and test the Overleaf/package contents in Phase E. |
| Optional graphical abstract | **PENDING: decision** | Not present. Decide whether to omit or prepare a separate graphical abstract; it is encouraged, not required by the current generic guide. |
| Highlights | **VERIFY** | Five draft highlights exist in `paper/highlights.txt`; confirm Speech Communication's live requirement and character limits before portal upload. |
| Suggested reviewers with institutional email addresses and no recent collaboration/conflict | **PENDING** | Prepare in Hand-off H; do not nominate editorial-board members or recent collaborators. |
| Cover letter | **PENDING** | Prepare in Hand-off H after the author block, archive DOI and declaration status are final. |
| Copyright/licence agreement | **PENDING** | Completed only after acceptance through Elsevier. |

## Citation register -- 23 records, all pending human verification

No placeholder is to be converted into a bibliography entry until the author
has checked the publisher/proceedings landing page. The records marked
**text decision** deliberately may result in a manuscript edit rather than a
new citation; Phase D governs that work.

| ID | Manuscript location | Record or claim to verify | State |
|---|---|---|---|
| M1 | §3.4 | Ledoit--Wolf covariance estimator | **PENDING -- publisher record** |
| M2 | §3.4 | Kingma--Ba Adam | **PENDING -- choose/cite canonical version** |
| M3 | §3.7 | Clustered-data bootstrap | **PENDING -- choose source actually relied on** |
| M4 | §3.7 | Holm sequentially rejective procedure | **PENDING -- publisher record** |
| R1 | §4.3; RW11 | BBSE / black-box label-shift correction | **PENDING -- publisher record; one key reused** |
| R2 | §4.3 | Saerens--Latinne--Decaestecker EM adjustment | **PENDING -- publisher record** |
| RW1 | §2.1 | Cross-corpus SER heterogeneity survey/meta-analysis | **PENDING -- source must support the stated heterogeneity** |
| RW2 | §2.2 | Covariate shift and importance weighting | **PENDING -- source must state the used assumption** |
| RW3 | §2.2; §4.7 | Hypothesis-class-dependent domain-adaptation bound | **PENDING -- candidate supplied; verify divergence wording** |
| RW4 | §2.2 | Closed-form CORAL | **PENDING -- candidate supplied; distinct from deep CORAL** |
| RW5 | §2.2 | Single-kernel MMD domain adaptation | **PENDING -- publisher record** |
| RW6 | §2.2 | Multi-kernel MMD domain adaptation | **PENDING -- formulation must match the paper** |
| RW7 | §2.2 | Adversarial domain adaptation | **PENDING -- canonical formulation** |
| RW8 | §2.2 | Full-ladder ablation practice | **PENDING -- text decision if unsupported** |
| RW9 | §2.2 | Median-heuristic bandwidth selection | **PENDING -- source should establish or analyse the heuristic** |
| RW10 | §2.2 | Conditional/target shift | **PENDING -- formulation must match the decomposition** |
| RW12 | §2.3 | Learned SSL layer weighting | **PENDING -- downstream speech protocol** |
| RW13 | §2.3 | SSL layer-wise probing | **PENDING -- intermediate-layer claim** |
| RW14 | §2.3 | Paralinguistic layer-wise probing | **PENDING -- text decision if no appropriate source** |
| RW15 | §2.4 | Transfer-aware model selection | **PENDING -- establish named alternatives** |
| RW16 | §2.4 | Target-domain tuning / oracle distinction | **PENDING -- text decision if critique unsupported** |
| RW17 | §2.4 | Chance-level or permutation floors | **PENDING -- metric-reporting source** |
| RW18 | §2.4 | Speaker-independent SER evaluation | **PENDING -- SSL speaker evidence or split-inflation evidence** |

## Existing bibliography entries requiring manual venue confirmation

| Key | Expected venue | Status | Required human action |
|---|---|---|---|
| `baevski2020wav2vec` | NeurIPS | **PENDING -- manual venue confirmation** | Verify authors, title, proceedings metadata and year against the proceedings/publisher record. |
| `gretton2012kernel` | JMLR | **PENDING -- manual venue confirmation** | Verify authors, title, volume, pages and year against the journal record. |

The remaining 13 current bibliography entries are marked cleared by the prior
reference audit, but all 15 entries must still pass `tools/check_refs.py` and
`tools/check_paper.py` after Phase D changes.

## Project completion gates

| Gate | Status | Owner / phase |
|---|---|---|
| Whole-manuscript coherence and RESULTS.md trace | **PENDING** | Phase B |
| Hostile-review objections and text-only responses | **PENDING** | Phase C |
| Verified citation records integrated; no placeholders remain | **BLOCKED** | Phase D, after Hand-off F |
| Self-contained Overleaf package and static validation | **PENDING** | Phase E |
| Clean Overleaf compile and PDF defect resolution | **BLOCKED** | Hand-off G |
| Final author block, affiliations, ORCIDs and corresponding author | **BLOCKED** | Hand-off H |
| Funding, acknowledgements, competing-interest upload and ethics confirmation | **BLOCKED** | Hand-off H |
| Zenodo archive DOI and archived-artifact verification | **BLOCKED** | Hand-off H |
| Cover letter, reviewers and portal submission | **BLOCKED** | Hand-off H |

## Phase update log

| Date | Phase | Update |
|---|---|---|
| 2026-08-30 | A | Checklist created from the manuscript tree, `CITATIONS_NEEDED.md`, `refs.bib`, and the current Elsevier generic Guide for Authors. Citation integration is intentionally **PENDING**. |
