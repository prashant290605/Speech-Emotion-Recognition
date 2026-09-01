# Calm Dropped Six sensitivity

This pre-specified sensitivity drops RAVDESS \texttt{calm} rather than merging it into \texttt{neutral}. It reuses cached hubert last-layer features, runs logreg, and scores 5 speaker-disjoint seeds in each transfer direction. It is a robustness control, not a replacement full classifier grid.

| direction | alignment | target macro-F1 | source-val macro-F1 | chance baseline | n train | n target test |
|---|---|---|---|---|---|---|
| ravdess->cremad | none | 0.2704 [0.2255, 0.3153] | 0.7064 [0.6169, 0.7959] | 0.1665 | 836 | 3677 |
| ravdess->cremad | zscore | 0.3692 [0.3556, 0.3828] | 0.7110 [0.6389, 0.7831] | 0.1665 | 836 | 3677 |
| ravdess->cremad | mean_shift | 0.3619 [0.3471, 0.3766] | 0.7040 [0.6214, 0.7867] | 0.1665 | 836 | 3677 |
| ravdess->cremad | coral | 0.3763 [0.3608, 0.3918] | 0.7077 [0.6156, 0.7998] | 0.1665 | 836 | 3677 |
| cremad->ravdess | none | 0.3122 [0.2471, 0.3773] | 0.6346 [0.6190, 0.6502] | 0.1645 | 836 | 528 |
| cremad->ravdess | zscore | 0.4243 [0.3593, 0.4893] | 0.6305 [0.6132, 0.6478] | 0.1645 | 836 | 528 |
| cremad->ravdess | mean_shift | 0.4454 [0.3861, 0.5046] | 0.6348 [0.6186, 0.6509] | 0.1645 | 836 | 528 |
| cremad->ravdess | coral | 0.4511 [0.4172, 0.4849] | 0.6314 [0.6133, 0.6495] | 0.1645 | 836 | 528 |

## Target-score differences from `none`

- ravdess->cremad: `zscore` minus `none` = +0.0988; `mean_shift` minus `none` = +0.0914; `coral` minus `none` = +0.1059.
- cremad->ravdess: `zscore` minus `none` = +0.1121; `mean_shift` minus `none` = +0.1331; `coral` minus `none` = +0.1388.
