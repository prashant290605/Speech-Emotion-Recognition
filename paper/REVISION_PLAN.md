# Major Revision Plan

**Target:** *Speech Communication*  
**Created:** 2026-09-01  
**Evidence source:** reviewer-style assessment supplied by the authors on
2026-09-01.

This plan addresses every actionable issue in that assessment. It separates
changes that can be verified from the current repository from experiments that
require new licensed data. No result will be claimed until it is produced by a
versioned configuration and recorded in an append-only result file.

## Phase 1: Claim, terminology, and compliance correction

**Status:** Complete on 2026-09-01. The manuscript now bounds every finding to
the RAVDESS--CREMA-D case study, uses `oracle gap` and `chance baseline`,
limits MK-MMD conclusions to the evaluated affine maps and optimisation budget,
and includes tool-specific AI disclosures. Validation passed with
`tools/check_paper.py`, `tools/check_number_trace.py`, and `pytest`.

**Purpose:** make every current claim match the existing two-corpus evidence.

| Review issue | Deliverable | Acceptance criterion |
|---|---|---|
| Two-corpus scope is too narrow for field-wide advice | Narrow the title, abstract, highlights, discussion, and conclusion to a controlled RAVDESS--CREMA-D case study until the third-corpus confirmation exists. | No sentence generalises a two-corpus result to all cross-corpus SER. |
| MK-MMD overclaim | Replace claims about successful higher-order matching with the evaluated affine-map formulation and report the observed fallback rate as an optimisation limitation. | Abstract, Results, Discussion, captions, and Conclusion agree. |
| "Cost of selection" | Use `oracle gap` or `selection opportunity gap`; state that it includes target-set maximisation and is not a causal cost. | All terminology is consistent. |
| "Chance floor" | Use `chance baseline` except where a mathematical lower bound is actually meant. | No misleading occurrence remains. |
| AI disclosure and DOI placeholder | Name OpenAI ChatGPT and Codex, their limited drafting/software role, and author oversight. Remove the prospective DOI sentence; leave the DOI as a checklist requirement until an archive exists. | Disclosure names tools, purpose, and oversight. No fabricated DOI appears. |

## Phase 2: Diagnostic and frame-dependence formalisation

**Status:** Complete on 2026-09-01. The paper now separates raw MMD$^2$ from
null-scaled effect sizes, removes the conditional-to-marginal ratio, states and
tests the median-heuristic scaling proposition, and specifies the paired
speaker-and-seed bootstrap. A 40-record, five-seed control audit in a fixed
source-defined geometry found RBF saturation after z-scoring. The manuscript
reports this as a limitation of a globally fixed bandwidth, not as a closeness
result.

**Purpose:** preserve the frame-dependence result while removing claims that its
diagnostics cannot support.

| Review issue | Deliverable | Acceptance criterion |
|---|---|---|
| Conditional-to-marginal ratio is over-interpreted | Demote the ratio to a descriptive secondary value. Promote raw MMD-squared and normalised trajectories under the fixed reference geometry. | The main text says only that the marginal diagnostic declines faster. |
| Own-frame MMD is called wrong or unidentified | Use `not invariant to measurement geometry` and `not comparable under a common geometry`. | No unsupported "wrong" or "unidentified" language remains. |
| Median-heuristic mechanism is implicit | Add a proposition and proof: global scaling scales the median bandwidth by the same factor, leaving an RBF kernel value unchanged. | Proposition is algebraically correct, cited to the implemented median rule, and has a unit test. |
| Bootstrap description is incomplete | State resampling units, seed handling, and the paired cluster-bootstrap algorithm in Methods and the supplement. | An independent reader can reproduce the interval construction. |

## Phase 3: Calm-drop performance control

**Status:** Complete on 2026-09-01. This focused control removes RAVDESS
\texttt{calm} without altering any target-test role. Its 40 rows are in a
separate append-only ledger and its table is generated from that ledger.

**Purpose:** test whether the observed alignment performance improvement
requires the RAVDESS \texttt{calm} merge.

The experiment was committed before execution. It uses existing RAVDESS/CREMA-D
feature caches and never re-extracts audio.

| Variant | Mapping | Question |
|---|---|---|
| `calm_merged_six` | Current `calm -> neutral` mapping | Existing reference analysis. |
| `calm_dropped_six` | Drop RAVDESS calm; retain neutral in both corpora | Does removing the mixture preserve the performance result? |

The immutable sensitivity specification has its own label-map hash, split hash,
result path and seed list. It evaluates `none`, `zscore`, `mean_shift`, and
`coral` ($\varepsilon=10.0$) with HuBERT final-layer features and logistic
regression in both directions over five speaker-disjoint seeds. Model selection
remains source-side only.

**Acceptance criterion:** passed. Unit tests prove the new mapping changes the
label-map hash, keeps splits speaker-disjoint, and creates 40 distinct run
identities. Every aligned rung exceeded `none` in the generated five-seed table;
the manuscript records the scope of this result without treating it as a full
grid.

## Phase 4: Label-harmonisation diagnostic extension

**Purpose:** address the residual reviewer concern about the class-conditional
diagnostic rather than only the target-score ladder.

The extension will use two separately versioned controls: `calm_dropped_six`
and `neutral_excluded_five`, which also drops RAVDESS `calm`. It will recompute
only the frozen HuBERT/logistic-regression control rungs, raw MMD$^2$, and
separately null-scaled conditional values. It will not restore the invalid
conditional-to-marginal ratio or claim that a fixed RBF bandwidth is a general
closeness scale. Every result remains in a separate ledger and is integrated
only at the strength it supports.

**Acceptance criterion:** both controls are complete, their labels and run IDs
are unit-tested, and the manuscript distinguishes a performance robustness
result from a class-conditional diagnostic robustness result.

## Phase 5: Third-corpus confirmatory study

**Purpose:** test the bounded case-study result on a corpus with a different
recording and elicitation design.

The planned corpus is IEMOCAP under its institutional SAIL licence. The
repository already contains its mapping policy, four-class label space,
session-level split logic, and cache-key support. The raw corpus and signed
licence are absent from this workspace, so no IEMOCAP result can be generated
or claimed in the current environment.

When the licensed corpus is supplied, run the pre-specified reduced grid:
`none`, `zscore`, `coral`, and `mkmmd_full`; HuBERT and WavLM; five seeds;
source-side selection; both IEMOCAP transfer directions available from the
manifest. Report per-pair results against the four-class chance baseline. Do
not pool them with the six-class RAVDESS/CREMA-D score.

**Acceptance criterion:** manifest and cache checks pass, the reduced grid is
resumable and leakage-tested, and the manuscript scope is widened only if the
generated results support the same qualified finding.

## Phase 6: Positioning, supplement, and final release

**Purpose:** make the revised article current, conventional, and reviewable.

| Review issue | Deliverable | Acceptance criterion |
|---|---|---|
| Recent related work | Verify publisher records for the 2025 *Speech Communication* semi-supervised adaptation article and the 2025 DistilHuBERT paper; add accurate BibTeX records and a scope paragraph. | Every new record passes the reference checker. |
| Research-history ledger is too long for the main article | Move the detailed withdrawn-claim ledger to supplementary material; keep a compact statement of frozen protocol and provenance in the article. | Main text remains self-contained and the supplement is referenced. |
| Submission readiness | Regenerate tables, figures, PDF, Overleaf ZIP, checklist, and reference report. | `check_paper.py`, number tracing, tests, package verification, and a visual PDF review pass. |

## Execution status

| Phase | Status | Notes |
|---|---|---|
| 1 | Complete | Text and compliance corrections use existing evidence only. |
| 2 | Complete | Measurement-frame claims now match the diagnostic protocol. |
| 3 | Complete | Calm-drop performance control is generated and integrated. |
| 4 | Pending | Uses existing caches; no external data required. |
| 5 | Waiting for licensed IEMOCAP | Raw data are absent; this is an external dependency, not a reason to claim results. |
| 6 | Pending | Runs after the scientific changes and source verification. |
