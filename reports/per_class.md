# Phase 10 — per-class transfer

Computed from the stored per-utterance predictions of the **validated** configuration in each (pair, seed): the run with the best `source_val`, which is what the protocol would actually deploy. Precision, recall and F1 all come from the same confusion matrix, so they cannot disagree.

Intervals are a paired cluster bootstrap over target_test speakers and seeds, 2000 replicates -- the same scheme as Phase 8. Per-class support is small, so an utterance-level interval would be badly over-confident here.

---

## ravdess -> cremad

Validated configuration per seed, 5 seeds. Chance floor 0.1665, majority floor 0.0425.

| class | support | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| angry | 3144 | 2305 | 0.5206 [0.4764, 0.5862] | 0.3817 [0.2001, 0.5163] | 0.4404 [0.2974, 0.5145] |
| disgust | 3144 | 5046 | 0.2943 [0.2164, 0.5236] | 0.4723 [0.2937, 0.7436] | 0.3626 [0.3249, 0.4328] |
| fear | 3144 | 3278 | 0.3270 [0.2700, 0.4467] | 0.3410 [0.1743, 0.4918] | 0.3339 [0.2460, 0.3733] |
| happy | 3144 | 2690 | 0.3565 [0.3395, 0.3877] | 0.3050 [0.1317, 0.4612] | 0.3288 [0.1952, 0.3993] |
| neutral | 2691 | 1386 | 0.3752 [0.3315, 0.4344] | 0.1932 [0.1041, 0.2751] | 0.2551 [0.1652, 0.3149] |
| sad | 3144 | 3706 | 0.2588 [0.1870, 0.3494] | 0.3050 [0.1660, 0.4220] | 0.2800 [0.1897, 0.3687] |
| **macro** | 18411 | 18411 | | | **0.3335 [0.2550, 0.3856]** |

Weakest to strongest: `neutral` 0.255, `sad` 0.280, `happy` 0.329, `fear` 0.334, `disgust` 0.363, `angry` 0.440.

No class collapsed to zero predictions.

## cremad -> ravdess

Validated configuration per seed, 5 seeds. Chance floor 0.1656, majority floor 0.0444.

| class | support | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| angry | 480 | 365 | 0.6685 [0.5944, 0.7586] | 0.5083 [0.3708, 0.6688] | 0.5775 [0.4827, 0.6659] |
| disgust | 480 | 829 | 0.4536 [0.3899, 0.5356] | 0.7833 [0.6687, 0.8771] | 0.5745 [0.5246, 0.6283] |
| fear | 480 | 606 | 0.6073 [0.5399, 0.6836] | 0.7667 [0.6624, 0.8625] | 0.6777 [0.6298, 0.7241] |
| happy | 480 | 421 | 0.3872 [0.3264, 0.4773] | 0.3396 [0.2167, 0.4688] | 0.3618 [0.2759, 0.4304] |
| neutral | 720 | 566 | 0.6678 [0.6119, 0.7394] | 0.5250 [0.4208, 0.6375] | 0.5879 [0.5155, 0.6518] |
| sad | 480 | 333 | 0.2943 [0.2262, 0.3808] | 0.2042 [0.1396, 0.2729] | 0.2411 [0.1779, 0.3028] |
| **macro** | 3120 | 3120 | | | **0.5034 [0.4675, 0.5361]** |

Weakest to strongest: `sad` 0.241, `happy` 0.362, `disgust` 0.574, `angry` 0.578, `neutral` 0.588, `fear` 0.678.

No class collapsed to zero predictions.

## Consistency across directions

A class that transfers badly in both directions is a property of the label, not of one corpus being the source.

| class | F1 ravdess->cremad | F1 cremad->ravdess | both below macro |
|---|---|---|---|
| angry | 0.4404 | 0.5775 | no |
| disgust | 0.3626 | 0.5745 | no |
| fear | 0.3339 | 0.6777 | no |
| happy | 0.3288 | 0.3618 | **yes** |
| neutral | 0.2551 | 0.5879 | no |
| sad | 0.2800 | 0.2411 | **yes** |
