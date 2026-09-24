# Fixed-reference MMD diagnostics

The ZCA basis and RBF bandwidth are each derived from unaligned source-train features once per pair, seed, backbone and layer aggregation. Both are then held fixed across the selected control rungs. Raw MMD-squared and its source half-split normalisation are reported together. Z-scoring drives this globally fixed RBF kernel into saturation, producing near-zero values; those cells demonstrate that a fixed bandwidth is not a generally usable closeness scale for a scale-changing map. Conditional values have class-specific null scales, so no conditional-to-marginal ratio is reported.

## `layer_agg=last`

| direction | rung | raw marginal MMD2 | marginal MMD2 / null | mean raw conditional MMD2 | conditional / class null |
|---|---|---|---|---|---|
| cremad->ravdess | none | 1.81440 [1.67031, 1.95849] | 4840.95 [3372.75, 6309.15] | 1.85915 [1.73668, 1.98162] | 2797.53 [2482.78, 3112.29] |
| cremad->ravdess | zscore | 1.1514e-08 [-1.4537e-08, 3.7565e-08] | 3.63 [-4.87, 12.12] | 0.00015 [0.00004, 0.00025] | 67723.10 [-15270.83, 150717.03] |
| cremad->ravdess | mean_shift | 0.81690 [0.72773, 0.90607] | 2192.84 [1471.61, 2914.07] | 0.86069 [0.80231, 0.91906] | 1287.33 [1226.12, 1348.54] |
| cremad->ravdess | coral | 0.05285 [0.03522, 0.07048] | 14.71 [6.80, 22.62] | 0.08583 [0.05804, 0.11361] | 16.87 [12.13, 21.62] |
| ravdess->cremad | none | 1.90057 [1.85136, 1.94977] | 4867.53 [3696.33, 6038.73] | 1.94566 [1.91446, 1.97687] | 2938.02 [2461.50, 3414.54] |
| ravdess->cremad | zscore | 5.0926e-10 [-1.7794e-10, 1.1965e-09] | 0.04 [-9.1476e-03, 0.08] | 2.2939e-08 [1.0624e-08, 3.5255e-08] | 0.87 [-0.44, 2.17] |
| ravdess->cremad | mean_shift | 1.10464 [1.07169, 1.13759] | 2835.39 [2109.07, 3561.71] | 1.15727 [1.14579, 1.16875] | 1718.16 [1461.56, 1974.76] |
| ravdess->cremad | coral | 0.01999 [0.01099, 0.02900] | 13.27 [-3.28, 29.82] | 0.04996 [0.04801, 0.05192] | 18.39 [10.29, 26.49] |

