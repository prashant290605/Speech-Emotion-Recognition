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

## Phase 3: Calm-label sensitivity implementation

**Purpose:** remove the label-harmonisation confound from the core diagnostic.

The experiment is pre-specified before execution. It uses the existing
RAVDESS/CREMA-D feature caches and never re-extracts audio.

| Variant | Mapping | Question |
|---|---|---|
| `calm_merged_six` | Current `calm -> neutral` mapping | Existing reference analysis. |
| `calm_dropped_six` | Drop RAVDESS calm; retain neutral in both corpora | Does removing the mixture preserve the ladder result? |
| `neutral_excluded_five` | Drop neutral in both corpora and drop RAVDESS calm | Does the result survive without the affected class? |

The runner will use a new immutable sensitivity specification with its own
label-map hash, split hash, result path, and seed list. It will evaluate the
pre-specified six-rung ladder in both directions over five speaker-disjoint
seeds. Model selection remains source-side only. The primary analysis will
report each rung's macro-F1, raw MMD-squared, own-geometry normalised MMD,
fixed-geometry normalised MMD, and per-class conditional values with supports.

**Acceptance criterion:** unit tests prove each variant has the expected labels,
no target-test leakage, and distinct run identities. The analysis is produced
from an append-only result file and is either integrated into the manuscript or
reported honestly as not supporting the previous interpretation.

## Phase 4: Calm-label sensitivity execution and integration

**Purpose:** run the Phase 3 design and revise the paper from its generated
report.

**Acceptance criterion:** all planned runs finish without recorded failures;
the report, table, and figure are generated by scripts; every resulting
manuscript number traces to the sensitivity result file. The text will state
whether the robustness analysis supports, narrows, or overturns the diagnostic.

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
| 1 | In progress | Text and compliance corrections use existing evidence only. |
| 2 | Pending | Depends on Phase 1 terminology being stable. |
| 3 | Pending | Uses existing caches; no external data required. |
| 4 | Pending | Runs only after Phase 3 tests and frozen configuration. |
| 5 | Waiting for licensed IEMOCAP | Raw data are absent; this is an external dependency, not a reason to claim results. |
| 6 | Pending | Runs after the scientific changes and source verification. |
