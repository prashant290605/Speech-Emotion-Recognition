# Response to Phyo Thet Yee's manuscript review

2026-10-03. Every comment from the review notes and every highlighted
annotation in the reviewed PDF, with the change made. Section numbers refer to
the revised article (15 pages; previously 17). No experiment was rerun and no
reported number changed.

## Authorship

| Comment | Change |
|---|---|
| Offer authorship to Shweta Jain, the main course instructor, with permission | Shweta Jain (shwetajain@iitrpr.ac.in, IIT Ropar) added as fourth author with CRediT role Supervision. **Shweta Jain's agreement to be listed still needs to be confirmed before submission.** |

## Abstract

| Comment | Change |
|---|---|
| Present the method briefly; include SER background, research gap, problem, method or contributions, gains and a concluding sentence | Rewritten in that order (236 words). |
| Gains in one or two sentences, near the end, after the method | One sentence near the end: equal validation scores and hyperparameters in all 30 matched RBF-SVM cells per direction, while mean shift raises target macro-F1 by 0.1342 and 0.1969. |
| No limitations in the abstract | Removed. |
| "What does frozen RAVDESS and CREMA-D mean?" | The abstract now says "4,986 completed runs". The introduction defines frozen: the configuration and result ledger were fixed under a version tag and checksum before the audit was designed. |

## Introduction

| Comment | Change |
|---|---|
| "This paper asks" | Removed. The problem is stated as "This work therefore addresses two problems". |
| No formula in the introduction | The formula was removed; the introduction contains no mathematics. |
| "What is and is not new." | Replaced by a paragraph stating what is established and where the novelty lies. |
| Write contributions as a sentence | "Our contributions are as follows:" followed by the list. |
| Introduction as a detailed abstract: gap, motivation, problem, method overview, contributions | Restructured into motivation, research gap, problem, method overview, key findings, contributions and a roadmap. |

## Writing throughout

| Comment | Change |
|---|---|
| Do not use "here" | No occurrence remains in the article ("in this work" and similar instead). |
| "need not reduce" | "does not necessarily reduce"; no "need not" remains. |
| "Features are row vectors. All maps are fitted ..." | Replaced with the suggested sentence (Section 3). |
| Wh-type headings | All replaced. Section 3: "Identifiability under source-side validation"; "Z-score standardisation beyond moment matching"; 3.3 "Measurement geometry of discrepancy comparisons"; 5.2 "Classifiers outside the proposition's assumptions"; 5.3 "Practical consequences of source-side selection"; 5.6 "Cases where source validation remains informative"; 6.2 "Scope of the affine factorisation"; 6.3 "Practical implications of the executable audit"; 6.4 "Limitations of the audit"; and the paragraph headings "Acceptance check and fallback", "Unequal inner grids in the ladder summary", "Interval for the source-selected summary", "Interpretation of the oracle column". |
| Semicolons joining clauses | Split into separate sentences; none remains in the article prose. |
| Do not repeat table numbers in paragraphs | Removed. Paragraphs now state which setting outperforms which and cite the table. Numbers that appear in no table, such as the 0.7304 against 0.7322 near-tie and the post hoc bounds, are kept. |
| Long and weak highlighted sentences | All split or rewritten (details below). |

## Highlighted sentences

| Location in reviewed PDF | Change |
|---|---|
| p. 4, matched-n sentence too long | Split into five sentences (Section 4.2). |
| p. 4, "reported here" | "reported in this work". |
| p. 5, "Derived from data/manifest.csv" | Removed from the paper. The provenance stays as a non-printed source comment in the table file. |
| p. 5, "frozen feature extractors" | "used as feature extractors". |
| p. 5, "is what the earlier version of this work used" | Removed. |
| p. 5, "Section 5.6 shows ... smaller in this study" | Split into two sentences. |
| p. 5, segment-pooling sentence | Split and rewritten. |
| p. 5, batch-size sentence | Split into three sentences. The unused MFCC branch sentence was also removed. |
| p. 6, "Fallback, and why it must be reported" | Renamed "Acceptance check and fallback". |
| p. 6, "These rates qualify every MK-MMD comparison ..." | Rewritten as three sentences. |
| p. 6, MMD effect-size sentence | Rewritten: the two reported quantities are introduced one at a time. |
| p. 6, reference-basis sentence with a semicolon | Split. |
| p. 7, "The matched-direction comparison in supplementary Section S3 excludes it" | Rewritten as "The Transformer arm is excluded from the matched-direction comparison (Supplementary Section S3) and, by construction, from the translation audit." The suggested "We exclude the matched-direction comparison ..." would have said that the comparison itself is excluded, which is not the meaning, so the subject was made explicit instead. |
| p. 7, "The earlier version of this work did this." | Removed. |
| p. 7, "no figure in this paper averages macro-F1 across pairs" | "Because the chance baseline differs between the two transfer directions, macro-F1 is always reported separately for each direction and is never averaged across directions." |
| p. 7, "oracle" | "The oracle column in Table 4 reports ...". |
| p. 8, Table 2 caption | "Split sizes (ranges over five seeds) and macro-F1 baselines for each transfer direction." The explanation moved to Sections 4.2 and 4.8. |
| p. 9, Table 3 caption | "Retrospective translation audit of matched none/mean_shift pairs." The column definitions and panel reading moved to Section 5.1. |

The other main-text table captions were shortened the same way, and their
explanatory notes moved into the text.

## Tables and figures

| Comment | Change |
|---|---|
| Table 1 short forms | Caption: "Speakers (spk), utterances (utts) and durations (dur) describe the raw speech subsets." It also notes that class names are abbreviated to four letters. |
| Add a pipeline figure | New Figure 1 on page 2: the experimental grid (corpora and splits, SSL features, alignment ladder, classifiers, model selection, evaluation) and the retrospective audit (ledger, matching, test of Proposition 1, target consequence, affine factorisation). |

## Structure and length

| Comment | Change |
|---|---|
| Discussion can be removed or shortened | Shortened from five subsections to four (951 to 724 words including limitations). |
| Shorten the limitations | "What it does not give them" and "Limitations" merged into "Limitations of the audit", three paragraphs. |
| Reproducibility not needed in the main paper | Moved to Supplementary Section S10, with a pointer from Code availability. |
| Shorten the conclusion | One paragraph (269 to 138 words). |
| Line numbers | Continuous line numbers added to the article and supplement. Remove `\linenumbers` for the camera-ready version. |
| Page limit | The journal's guide could not be opened from the build environment, so no limit was confirmed. The article is now 15 pages and should be checked against the portal's requirements. |
| Paper too long, remove unimportant parts | 17 to 15 pages including the new figure. Duplicated caveats, repeated numbers and the unused MFCC sentence were removed. |

## Not changed

- The paper title, "What source validation cannot see: ...", is still a
  wh-type phrase. The review did not mark it, so it was left for the authors to
  decide.
