# Neutral Excluded Five sensitivity

This pre-specified sensitivity drops RAVDESS \texttt{calm} and excludes \texttt{neutral} from both corpora. It reuses cached hubert last-layer features, runs logreg, and scores 5 speaker-disjoint seeds in each transfer direction. It is a robustness control, not a replacement full classifier grid.

| direction | alignment | target macro-F1 | source-val macro-F1 | chance baseline | n train | n target test |
|---|---|---|---|---|---|---|
| ravdess->cremad | none | 0.3616 [0.3237, 0.3995] | 0.7389 [0.6463, 0.8315] | 0.2000 | 760 | 3140 |
| ravdess->cremad | zscore | 0.4562 [0.4383, 0.4741] | 0.7378 [0.6551, 0.8205] | 0.2000 | 760 | 3140 |
| ravdess->cremad | mean_shift | 0.4333 [0.4116, 0.4551] | 0.7408 [0.6476, 0.8340] | 0.2000 | 760 | 3140 |
| ravdess->cremad | coral | 0.4228 [0.4065, 0.4392] | 0.7335 [0.6341, 0.8329] | 0.2000 | 760 | 3140 |
| cremad->ravdess | none | 0.3246 [0.2856, 0.3635] | 0.6481 [0.6266, 0.6696] | 0.2000 | 760 | 480 |
| cremad->ravdess | zscore | 0.4560 [0.4062, 0.5058] | 0.6406 [0.6096, 0.6716] | 0.2000 | 760 | 480 |
| cremad->ravdess | mean_shift | 0.4556 [0.4144, 0.4968] | 0.6475 [0.6260, 0.6690] | 0.2000 | 760 | 480 |
| cremad->ravdess | coral | 0.4943 [0.4553, 0.5333] | 0.6487 [0.6250, 0.6725] | 0.2000 | 760 | 480 |

## Paired target-score differences from `none`

Each interval is a paired cluster bootstrap over target-test speakers and seeds, 2000 replicates.

- ravdess->cremad: `zscore` minus `none` = +0.0946 [+0.0756, +0.1120]; `mean_shift` minus `none` = +0.0717 [+0.0508, +0.0943]; `coral` minus `none` = +0.0612 [+0.0394, +0.0845].
- cremad->ravdess: `zscore` minus `none` = +0.1314 [+0.0677, +0.1935]; `mean_shift` minus `none` = +0.1310 [+0.0712, +0.1890]; `coral` minus `none` = +0.1697 [+0.1282, +0.2112].
