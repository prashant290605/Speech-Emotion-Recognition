# Source-translation audit

Retrospective analysis of frozen runs. No training.
Stored-score equality is exact, not rounded. The full trial surface is not stored.
Rows share seeds and are not independent experimental replications.

`exact prediction` marks the families whose fitting rule satisfies Proposition 1's assumptions. The remaining families are reported as scope evidence for a boundary the manuscript states in advance; they are not counterexamples to the proposition.

Input SHA256: `51b8ff64d500d77d16c047430802645fb7bf86bea351258d3d03cee7381b1407`
Config SHA256: `ee002f864e3c2bde0867e1f43dcce646323694c93ad43695fa7135baed32f550`

| Direction | Classifier | Exact prediction | Pairs | Equal validation | Equal hyperparameters | Both equal | Positive target differences | Mean target difference | Chance |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cremad -> ravdess | RBF SVM | yes | 30 | 30 | 30 | 30 | 30 | +0.1969 | 0.1656 |
| ravdess -> cremad | RBF SVM | yes | 30 | 30 | 30 | 30 | 30 | +0.1342 | 0.1665 |
| cremad -> ravdess | Logistic | no | 30 | 0 | 22 | 0 | 30 | +0.1881 | 0.1656 |
| ravdess -> cremad | Logistic | no | 30 | 3 | 12 | 2 | 30 | +0.1442 | 0.1665 |
| cremad -> ravdess | Linear SVM | no | 30 | 0 | 19 | 0 | 30 | +0.1776 | 0.1656 |
| ravdess -> cremad | Linear SVM | no | 30 | 6 | 15 | 4 | 30 | +0.1456 | 0.1665 |
| cremad -> ravdess | MLP | no | 45 | 0 | 24 | 0 | 45 | +0.1701 | 0.1656 |
| ravdess -> cremad | MLP | no | 45 | 0 | 12 | 0 | 45 | +0.1287 | 0.1665 |
