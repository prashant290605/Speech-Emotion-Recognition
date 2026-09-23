# Prospective source-translation replication — protocol v1

**Status: DRAFT. Not frozen. Not executed.**

IEMOCAP is not present on this machine, so this protocol cannot be frozen and
the experiment cannot be run. The implementation, the protocol and the tests
are complete; what is missing is the corpus. The freeze procedure is at the end
of this document and must be carried out as a deliberate, dated act — not
backdated, and not implied by this draft existing.

**No prospective result has been generated. No target macro-F1 has been
computed for any IEMOCAP-involving cell, because no such cell exists.**

---

## Why this experiment

The article's central empirical evidence has three weaknesses, and this
experiment is aimed at exactly them:

1. the audit is **retrospective** — it was designed after the ledger it reads
   was frozen;
2. the evidence uses **two corpora**, RAVDESS and CREMA-D;
3. both are **acted English** speech.

A prospectively specified replication on a corpus pair that played no part in
formulating the audit addresses (1) directly and (2) partially. IEMOCAP
addresses (3) in the way that matters: it is dyadic conversational speech, part
improvised, recorded in sessions rather than as isolated prompts. Its value is
not that it is a third dataset; it is that the elicitation design is different
in kind.

## The question

> Does the exact source-validation invariance predicted by Proposition 1 appear
> again on a corpus pair that was not used to formulate the current audit?

## Primary hypothesis

For matched RBF-SVM `none` vs `mean_shift` cells satisfying Proposition 1's
assumptions, translating source training and source validation together leaves
the source-validation predictions — and therefore the selection criterion —
invariant, while the target decision function is permitted to change.

**The primary endpoint is the validation symmetry, not any target outcome.**

## What is explicitly *not* predicted

Proposition 1 says nothing about the sign of the target-performance difference.
All three of the following are consistent with it, and none of them would
falsify it:

- `mean_shift` improves target macro-F1;
- `mean_shift` degrades target macro-F1;
- target predictions do not change at all.

The third case would mean the criterion was still blind, but that this corpus
realisation did not make the blindness consequential. That is a weaker result,
not a contradicted one, and it will be reported as such.

Logistic regression is included as a **finite-solver implementation contrast**.
Exact equality is *not* predicted for it. If it nevertheless appears, it is
reported as an observed numerical fact and Proposition 1 is *not* extended to
cover it.

## Design

| | |
|---|---|
| Directions | CREMA-D → IEMOCAP and IEMOCAP → CREMA-D |
| Label space | four classes: angry, happy, neutral, sad |
| Label policy | unchanged from `configs/default.yaml`: `excited`→`happy`, `frustrated`/`xxx`/`other`/`surprised` excluded |
| Grouping | CREMA-D by speaker, IEMOCAP by **session** |
| Seeds | 0, 1, 2, 3, 4 |
| Backbones | HuBERT Base, wav2vec 2.0 Base, WavLM Base — **final layer, mean pooling only** |
| Alignments | `none`, `mean_shift` — nothing else |
| Classifiers | RBF SVM (primary), logistic regression (contrast) |
| Search | 20 candidates, generated once per (classifier, seed), shared byte-for-byte by both arms |
| Runs | 120 |
| Candidate evaluations | 2400 |

CREMA-D is chosen over RAVDESS because it has 91 speakers against 24; the
prospective evidence should not rest again on the smallest corpus here.

No z-score, CORAL, MK-MMD or blending. No layer sweep, no learned aggregation,
no transformer head, no MFCC. This is a criterion replication, and every axis
that does not bear on the criterion is fixed.

### The one open decision, and its pre-committed rule

Improvised speech is the sharper contrast with acted CREMA-D, but only if it
supports a session-disjoint four-class design. Rather than choose after seeing
counts, the rule is fixed now:

> Use **improvised only** if, after the four-class mapping, every class has
> ≥ 150 utterances, every class appears in all five sessions, and every class
> has ≥ 20 utterances in every session. Otherwise use **both subsets**.

The threshold of 150 is set so that a 50/50 adapt/test split leaves roughly 75
utterances per class for testing. The branch taken, and the counts it was taken
on, are recorded in the provenance file at freeze time. Counting utterances by
class is metadata inspection and does not expose any model outcome.

## Candidate surface

The retrospective audit could only check the *winning* candidate, because the
frozen ledger stores the winner's validation score and not the rest of the
search surface. That is a real limitation of the existing evidence and this
experiment removes it.

Before execution, the exact candidate list is generated for every
(classifier, seed). The `none` and `mean_shift` arms of a matched cell receive
**the same candidates, in the same order, with the same random state**. For
every candidate the run records:

- the candidate index and its hyperparameters (`C`, `gamma`, …);
- the source-validation macro-F1;
- a digest of the validation predictions;
- a digest of the validation decision-function values.

The theorem can then be audited at two levels:

- **A. candidate-level** — every candidate's validation predictions must match
  across arms, not merely the winner's;
- **B. selected-model** — the selected hyperparameters and winning score must
  match.

## Numerical equality, fixed in advance

The mathematics is exact; floating-point software need not be. Two levels are
pre-specified:

1. **Stored-score exact equality.** Predicted validation *labels* and stored
   validation macro-F1 are compared with exact equality. For the RBF SVM,
   predicted labels are expected to be identical.
2. **Numerical function equivalence.** Decision-function and kernel values use
   `atol = 1e-12`, the tolerance already justified in
   `tests/test_proposition1_source_translation.py`: about three orders of
   magnitude above the ~4e-15 drift that `rbf_kernel`'s squared-distance
   expansion introduces, and far below any decision margin.

**This tolerance is not to be revised after seeing prospective differences.**
If a stored validation macro-F1 differs, that is something to investigate — an
assumption violation or an implementation problem — not something to round
away.

## Result isolation

Prospective results never enter `results/runs.jsonl`. They go to
`results/prospective_translation_v1/`, with their own runs ledger, trial
surface, provenance, and audit. The runner refuses to write to the historical
ledger, and a test asserts that refusal. The historical ledger's digest is
verified before and after.

## Statistical reporting

The primary result is an **audit, not a significance test**. It reports counts:
matched cells, candidate comparisons, exact-equality counts. No p-value is
invented for an algebraic identity.

The secondary target comparison is paired within matched cells and reported
descriptively, with counts of positive, zero and negative differences.
Cluster-bootstrap intervals use the existing machinery, clustered on the target
grouping unit. When IEMOCAP is the target there are only two or three session
clusters, so those intervals are weak and will be labelled as such; this does
not affect the primary endpoint, which uses no interval.

## Prospectivity record

At freeze time the provenance file records: the protocol config SHA256, the
freeze commit, the freeze tag, the manifest SHA256, the feature-cache hashes,
the candidate-list hashes, the run identifiers, the result-ledger SHA256, and
the exact execution command.

**What had been inspected before this protocol was written:**

- the repository's existing IEMOCAP label policy, split policy and config keys;
- the published structure of the IEMOCAP release (directory layout, annotation
  file format, utterance-id grammar, emotion codes);
- the retrospective RAVDESS/CREMA-D audit results, which are already published
  in the article.

**What code had been run:** the IEMOCAP manifest parser against synthetic
fixtures only; the existing test suite.

**What had *not* been inspected:** any IEMOCAP audio, any IEMOCAP annotation
file, any IEMOCAP feature, any model trained on IEMOCAP, and any target
macro-F1 for any cell involving IEMOCAP. None of these exist on this machine.

## If the result is surprising

Fixed in advance, so that no reading of the outcome can be rationalised later:

- **RBF validation scores not invariant** → stop; check whether the fitting
  assumptions actually hold; inspect numerics; report the failure.
- **Equality holds, `mean_shift` hurts target performance** → report it. The
  blind spot is demonstrated at least as forcefully.
- **Equality holds, target predictions unchanged** → report it. The criterion
  is still invariant; this realisation was simply not consequential.
- **Logistic regression unexpectedly exact** → report as observed; do not
  extend Proposition 1.
- **Prospective and retrospective disagree** → report both; do not tune the
  design until they agree.

## Freeze procedure

To be carried out only when IEMOCAP is available, and before the experiment is
run:

1. Place the release at `data/raw/IEMOCAP` (see the data-access requirement in
   the session report).
2. `python tools/prospective_counts.py` — evaluates the subset decision rule on
   metadata and writes the counts. No model is fitted.
3. Record the chosen branch and counts in the protocol (`iemocap_subset.decided`).
4. `python -m pytest` — the full suite must pass.
5. `python tools/run_prospective_translation.py --check` — verifies the
   protocol, the candidate surface and the output isolation without running.
6. Commit: `Freeze prospective source-translation replication protocol`.
7. Tag `prospective-translation-v1` at that commit. Do not backdate. Do not
   touch `grid-freeze-v3`.
8. Only then run the experiment.
