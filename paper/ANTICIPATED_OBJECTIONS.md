# Anticipated Reviewer Objections

**Target:** *Speech Communication*
**Status:** 2026-08-31 text-only review pass. No experiment or numerical result
was added in response to these objections.

| Rank | Objection | Already answered in manuscript | Can be answered without new experiments? | Honest response |
|---|---|---|---|---|
| 1 | Two directions between one corpus pair are not an independent demonstration of a general cross-corpus phenomenon. | `Discussion` - Limitations; `Conclusion`. | Yes, by restricting the scope. | The paper is a controlled RAVDESS--CREMA-D case study, not a claim of universality. A third, distinct corpus would be the strongest future test. |
| 2 | The absolute transfer scores are modest and should not be read as a deployable SER system. | `Discussion` - Limitations; `Backmatter` - Ethics statement. | Yes. | The manuscript explicitly presents a diagnostic evaluation, not a recommended deployment pipeline. |
| 3 | Both corpora are English and acted, so the result may not transfer to spontaneous or multilingual speech. | `Discussion` - Limitations. | Yes, by limiting language and elicitation claims. | The limitation remains; no existing result tests spontaneous or multilingual transfer. |
| 4 | Target adaptation could conceal target-test leakage. | `Methods` - splits and leakage assertions; `Reproducibility` - provenance machinery. | Yes. | Alignment sees only `target_adapt`; `target_test` is scored once and executable assertions enforce the separation. |
| 5 | Source validation fails to select alignment only because the forward source-validation split is smaller. | `Results` - Direction; `Discussion` - Limitations; `Results` - selection. | Partly. | Source-training size is matched, but source-validation and target-adaptation sizes differ. The text no longer attributes the asymmetry to a single cause; the proposed matched-validation robustness test is not run. |
| 6 | The alignment ladder is unfair because CORAL and MK-MMD have more inner hyperparameters. | `Methods` - unequal inner grids; Table `tab:ladder`. | Yes. | Each fixed cell first selects its own inner setting on `source_val`; target intervals use only those selected predictions. Candidate-row counts are shown only to disclose grid size. |
| 7 | The class-conditional MMD ratio is presented as evidence about conditional shift or as an additive decomposition. | `Methods` - class-conditional diagnostic; `Discussion` - direct test. | Yes. | The text now states it measures `P(x|y)`, is descriptive, and does not estimate `P(y|x)` or identify a causal mechanism. |
| 8 | Small prior KL and failed BBSE/EM do not prove that label shift is absent. | `Results` - label-shift correction; `Discussion` - label-shift interpretation. | Yes. | The paper only reports small observed prior differences and failure of the tested standard corrections; their assumptions can fail under class-conditional shift. |
| 9 | The discrepancy--transfer association is over-interpreted because it changes under a different MMD geometry. | `Results` - Frame dependence; `Methods` - measurement geometry; `Discussion`. | Yes. | The claim is deliberately limited to non-invariance under the chosen measurement geometry. It does not designate one frame as universally correct. |
| 10 | The Transformer arm receives fewer seeds and cannot support a broad classifier-family conclusion. | `Discussion` - Limitations; Table `tab:classifier`. | Yes. | The two-seed primary-direction Transformer arm is reported separately and is never pooled with the five-seed families. It is a limited baseline, not a headline comparison. |

## Text changes completed in this pass

- Restricted all general claims to the observed corpus pair and protocol.
- Replaced `pre-registered` with `pre-specified before the frozen confirmatory
  grid`.
- Distinguished the 4,986-run frozen confirmatory grid from the 5,424-row
  retained result ledger.
- Stated the per-cell source-validation selection rule behind the unequal-grid
  alignment table.
- Reframed the CORAL limit as a scalar-rescaled mean shift and the geometry
  conclusion as non-invariance, not identification or frame superiority.
- Replaced literal citation placeholders with bibliography records and retained
  an explicit audit/spot-check register.
