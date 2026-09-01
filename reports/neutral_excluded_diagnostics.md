# Neutral Excluded Five diagnostics

This report uses the median_pairwise_distance_source_target rule separately for every rung and seed. The reported raw and null-scaled values are therefore descriptive adaptive-geometry diagnostics. They are not comparable under one fixed RBF scale, and no conditional-to-marginal ratio is reported.

| direction | rung | raw marginal MMD2 | marginal MMD2 / null | mean raw conditional MMD2 | mean conditional / class null |
|---|---|---|---|---|---|
| ravdess->cremad | none | 1.0190 [0.9760, 1.0619] | 238.9549 [142.0787, 335.8311] | 1.1443 [1.1108, 1.1779] | 205.3247 [160.2070, 250.4424] |
| ravdess->cremad | zscore | 0.0413 [0.0365, 0.0461] | 14.2315 [4.3226, 24.1404] | 0.1270 [0.1214, 0.1327] | 24.4733 [17.7742, 31.1724] |
| ravdess->cremad | mean_shift | 0.0646 [0.0542, 0.0751] | 15.7141 [6.8770, 24.5511] | 0.1573 [0.1495, 0.1651] | 26.0699 [21.0449, 31.0949] |
| ravdess->cremad | coral | 0.0514 [0.0407, 0.0621] | 18.4716 [11.5418, 25.4015] | 0.1076 [0.1039, 0.1114] | 24.7140 [14.1466, 35.2814] |
| cremad->ravdess | none | 1.0386 [0.9978, 1.0794] | 330.3610 [215.0701, 445.6518] | 1.1563 [1.0926, 1.2201] | 244.7265 [160.8459, 328.6071] |
| cremad->ravdess | zscore | 0.0669 [0.0552, 0.0785] | 24.1710 [15.9200, 32.4221] | 0.1593 [0.1388, 0.1797] | 37.4563 [25.1877, 49.7249] |
| cremad->ravdess | mean_shift | 0.0960 [0.0814, 0.1106] | 30.1804 [21.2535, 39.1073] | 0.1899 [0.1670, 0.2128] | 41.5844 [22.7841, 60.3848] |
| cremad->ravdess | coral | 0.0916 [0.0783, 0.1048] | 40.5493 [22.9606, 58.1381] | 0.1494 [0.1232, 0.1756] | 39.2821 [24.5139, 54.0503] |
