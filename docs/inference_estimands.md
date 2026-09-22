# Inference estimands, by table

What each manuscript table estimates, how its uncertainty is computed, and why
that choice rather than another. Written because the paper uses two different
interval procedures and a reader comparing two adjacent tables is otherwise
comparing two different variance decompositions without being told.

This is a developer/reviewer reference. It describes the code as it stands; it
does not propose changes. The statistical procedures themselves are unchanged.

## The two estimators

**Paired cluster bootstrap** — `ser.phase8.paired_cluster_bootstrap`, and its
single-arm form `cluster_bootstrap`. Each of 2000 replicates draws the five
seed identifiers with replacement, then draws that seed's target-test speaker
identifiers with replacement, keeping every utterance of a selected speaker.
Both arms of a contrast are scored on the *same* draws, so the sampling noise
common to them cancels. Resampling unit: **(seed, speaker)**. It never
resamples utterances independently, because errors cluster hard by talker and
an utterance bootstrap would report intervals several times too narrow.

**Seed t-interval** — `ser.phase8.seed_interval`. Mean plus or minus
`t * sd / sqrt(n)` over at most five values, with tabulated critical values for
`n` in 2..5. Resampling unit: **seed only**. Used where the quantity has no
per-utterance form to bootstrap, either because it is a property of a fitted
map rather than of a prediction, or because the arm itself changes between
seeds.

## Requirement that decides between them

The paired cluster bootstrap needs a **fixed arm**: a condition that can be
recomputed on an arbitrary resampled set of speakers within an arbitrary
resampled set of seeds. A rung, a layer aggregation and a classifier family
are all fixed arms. Two things in this paper are not:

1. *Discrepancy quantities.* `marginal_mmd_*` describes a fitted alignment map,
   not a set of per-utterance predictions, so there is nothing to resample over
   speakers.
2. *The source-selected summary.* Its per-seed value is the target macro-F1 of
   whichever configuration source-side validation chose **on that seed**, and
   the chosen configuration differs across seeds. Forward, the protocol selects
   `none` on two of five seeds and an aligned rung on the other three. There is
   no single arm to hold fixed; each seed contributes one complete deployment
   decision.

## Table map

| Table | Estimand | Method | Resampling unit | Paired | Generator |
|---|---|---|---|---|---|
| `tab:translation-audit` | Per-cell exact equality counts; mean paired target difference | **None** — counts and descriptive means | n/a | Cells are matched pairs, but no interval or test is reported | `tools/audit_translation.py` |
| `tab:headline` | Target macro-F1 of the source-selected configuration; of the grid maximum; their difference | Seed t-interval | Seed | No | `tools/make_tables.py::table_headline` |
| `tab:ladder` (target column) | Mean target macro-F1 of a rung, inner setting selected on `source_val` | Cluster bootstrap, 2000 replicates | Seed + target-test speaker | Single-arm | `tools/make_tables.py::table_ladder` |
| `tab:ladder` (discrepancy columns) | Mean discrepancy over **candidate** rows | Means only, no interval | n/a | No | same |
| `tab:primary` | 14 pre-specified contrasts in target macro-F1 | Paired cluster bootstrap + Holm | Seed + target-test speaker | Yes | `tools/phase8_tables.py` → `reports/phase8_primary.json` |
| `tab:perclass` | Per-class F1 of the validated configuration | Paired cluster bootstrap | Seed + target-test speaker | Yes | `tools/phase10_per_class.py` → `reports/per_class.json` |
| `tab:calm-sensitivity` | Aligned-minus-`none` under the calm-drop label map | Paired cluster bootstrap + 12-contrast one-sided Bonferroni bound | Seed + target-test speaker | Yes | `tools/report_calm_sensitivity.py` |
| `tab:neutral-excluded-sensitivity` | Same, five-class variant | Paired cluster bootstrap + shared Bonferroni family | Seed + target-test speaker | Yes | same |
| `tab:decomposition` | Marginal and class-conditional MMD-squared in a fixed reference geometry | Seed t-interval | Seed | No | `tools/report_reference_mmd_diagnostics.py` |
| `tab:label-harmonisation-diagnostics` | Same, per label mapping | Seed t-interval | Seed | No | `tools/report_label_sensitivity_diagnostics.py` |
| `tab:eps` | Map residual and scores across CORAL shrinkage | Seed t-interval | Seed | No | `tools/eps_asymptote_report.py` |
| `tab:frames` | Spearman rho between layer discrepancy and layer transfer, per protocol | Descriptive means, **no intervals** | n/a | No | `tools/make_tables.py::table_frames` |
| `tab:corpora`, `tab:floors` | Design constants and analytic baselines | None — not outcomes | n/a | n/a | `tools/make_tables.py` |

## Notes that matter for reading the tables

**`tab:headline` is not comparable to `tab:primary`.** The former carries
between-seed variability of an end-to-end selection protocol and no within-seed
speaker resampling; the latter carries both sources for a fixed contrast. A
wider interval in the former is not evidence of a noisier measurement of the
same thing. Methods, "Why the source-selected summary uses a different
interval", states this in the manuscript.

**The oracle column is a maximum.** The maximum of a set of noisy scores is
biased upward, so the oracle column is an optimistic summary of this target set
rather than an estimate of achievable performance. The validated-to-oracle
difference is descriptive and is not the causal effect of source-side
selection: it also contains target-set maximisation, finite-sample variation
and the size of the search space.

**`tab:ladder` juxtaposes two averaging protocols.** Target scores use the
`source_val`-selected inner setting within each cell; discrepancy columns
average every candidate row. They are different estimands and their
juxtaposition is descriptive, not a matched map-level correlation.

**`tab:frames` correlations are descriptive by decision.** Cells share seeds
and feature sets, so pooled independence-based intervals were removed rather
than reported; the claim is a protocol-level sign difference, not a
population-level estimate.

**Where a claim is null, a bound is reported rather than a test.** Failing to
reject a null is not evidence for it, so the among-rung comparisons use
one-sided bootstrap upper bounds under an eight-contrast Bonferroni family
(`tools/report_global_zscore_bound.py`), not p-values.

**Multiplicity.** Holm correction is applied across exactly the 14
pre-specified primary contrasts. Comparisons outside that family are reported
as such and corrected within their own set, with both raw and corrected values
given.

## Unused machinery

`ser.stats` provides `bootstrap_ci` (percentile bootstrap over test
utterances), `wilcoxon_signed_rank` and `holm_bonferroni`. These are tested but
**no current manuscript table uses them**; the utterance bootstrap in
particular is deliberately not used, for the clustering reason above. They are
retained because earlier phases used them and the provenance ledger refers to
results that were computed with them.
