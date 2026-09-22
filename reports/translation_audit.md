# Source-translation audit

Retrospective analysis of frozen runs. No training.
Stored-score equality is exact, not rounded. The full trial surface is not stored.
Rows share seeds and are not independent experimental replications.

Input SHA256: `51b8ff64d500d77d16c047430802645fb7bf86bea351258d3d03cee7381b1407`
Config SHA256: `7641dbe58068920580ba010e47e7c612ae257843c42fe481300953c7e65fbb8d`

| Direction | Classifier | Pairs | Equal validation | Equal hyperparameters | Both equal | Positive target differences | Mean target difference | Chance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cremad -> ravdess | logreg | 30 | 0 | 22 | 0 | 30 | +0.1881 | 0.1656 |
| cremad -> ravdess | svm_rbf | 30 | 30 | 30 | 30 | 30 | +0.1969 | 0.1656 |
| ravdess -> cremad | logreg | 30 | 3 | 12 | 2 | 30 | +0.1442 | 0.1665 |
| ravdess -> cremad | svm_rbf | 30 | 30 | 30 | 30 | 30 | +0.1342 | 0.1665 |
