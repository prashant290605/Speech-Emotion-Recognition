# Calm Dropped Six diagnostics

This report uses the median_pairwise_distance_source_target rule separately for every rung and seed. The reported raw and null-scaled values are therefore descriptive adaptive-geometry diagnostics. They are not comparable under one fixed RBF scale, and no conditional-to-marginal ratio is reported.

| direction | rung | raw marginal MMD2 | marginal MMD2 / null | mean raw conditional MMD2 | mean conditional / class null |
|---|---|---|---|---|---|
| ravdess->cremad | none | 1.0514 [0.9970, 1.1059] | 293.7573 [168.4475, 419.0670] | 1.1947 [1.1630, 1.2263] | 182.2469 [150.8918, 213.6020] |
| ravdess->cremad | zscore | 0.0419 [0.0366, 0.0472] | 12.6942 [7.0997, 18.2886] | 0.1406 [0.1355, 0.1457] | 22.7165 [18.4647, 26.9683] |
| ravdess->cremad | mean_shift | 0.0601 [0.0496, 0.0706] | 16.4700 [10.2955, 22.6444] | 0.1736 [0.1656, 0.1817] | 23.8307 [20.7896, 26.8718] |
| ravdess->cremad | coral | 0.0510 [0.0359, 0.0661] | 24.7329 [17.4045, 32.0612] | 0.1241 [0.1191, 0.1292] | 24.3758 [16.9777, 31.7740] |
| cremad->ravdess | none | 1.0632 [1.0038, 1.1226] | 302.5719 [265.5559, 339.5879] | 1.1631 [1.0988, 1.2275] | 216.7180 [166.7292, 266.7069] |
| cremad->ravdess | zscore | 0.0692 [0.0597, 0.0788] | 28.8847 [5.4577, 52.3117] | 0.1591 [0.1381, 0.1802] | 32.2102 [24.9949, 39.4255] |
| cremad->ravdess | mean_shift | 0.0862 [0.0766, 0.0958] | 24.9244 [18.3272, 31.5216] | 0.1888 [0.1633, 0.2143] | 32.8345 [27.2209, 38.4481] |
| cremad->ravdess | coral | 0.1017 [0.0607, 0.1428] | 42.3364 [28.4996, 56.1732] | 0.1457 [0.1207, 0.1708] | 38.5202 [34.8020, 42.2383] |
