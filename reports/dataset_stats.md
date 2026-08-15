# Dataset statistics

Every number is derived from `data/manifest.csv`. Regenerate with `ser dataset-stats`.

## Per corpus

| corpus | speakers | utterances | total hours | mean duration (s) |
|---|---|---|---|---|
| ravdess | 24 | 1440 | 1.48 | 3.70 |
| cremad | 91 | 7442 | 5.26 | 2.54 |

## Label space `four` (4 classes)

| corpus | angry | happy | neutral | sad | total | excluded |
|---|---|---|---|---|---|---|
| ravdess | 192 | 192 | 288 | 192 | 864 | 576 |
| cremad | 1271 | 1271 | 1087 | 1271 | 4900 | 2542 |

Class prior:

| corpus | angry | happy | neutral | sad |
|---|---|---|---|---|
| ravdess | 0.222 | 0.222 | 0.333 | 0.222 |
| cremad | 0.259 | 0.259 | 0.222 | 0.259 |

⚠️ marks a class with fewer than 100 utterances after mapping.

## Label space `six` (6 classes)

| corpus | angry | disgust | fear | happy | neutral | sad | total | excluded |
|---|---|---|---|---|---|---|---|---|
| ravdess | 192 | 192 | 192 | 192 | 288 | 192 | 1248 | 192 |
| cremad | 1271 | 1271 | 1271 | 1271 | 1087 | 1271 | 7442 | 0 |

Class prior:

| corpus | angry | disgust | fear | happy | neutral | sad |
|---|---|---|---|---|---|---|
| ravdess | 0.154 | 0.154 | 0.154 | 0.154 | 0.231 | 0.154 |
| cremad | 0.171 | 0.171 | 0.171 | 0.171 | 0.146 | 0.171 |

⚠️ marks a class with fewer than 100 utterances after mapping.

## Corpus-level prior shift

Amendment **A9**: these are *corpus-level* priors. The quantity the analysis rests on is split-level KL per pair per seed, computed from the realised partitions — a Phase 8 deliverable. This table is the data-integrity check against the published counts that **A8** used.

| source | target | K | KL (nats) | JS | A8 predicted | agrees |
|---|---|---|---|---|---|---|
| ravdess | cremad | 6 | 0.0252 | 0.0769 | 0.0252 | yes |
| cremad | ravdess | 6 | 0.0224 | 0.0769 | 0.0224 | yes |

All computed pairs agree with A8 within 0.002 nats. The near-zero prior shift that reframed Phase 9 is confirmed against real data.
