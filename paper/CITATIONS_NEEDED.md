# Citation Verification Register

**Status: 2026-08-31**

The submission manuscript has **29 cited bibliography records and zero
`[CITE: ...]` placeholders**. `python tools/check_paper.py` checks that every
citation key resolves and that no bibliography entry is uncited.

`reports/refs_report_submission.md` is the machine-generated audit for
`paper/main.tex` and `paper/refs.bib`. Its latest run reports zero
probable-fabrication findings and 21 Crossref-confirmed records. The remaining
eight records need a final publisher-page spot check because their venues are
not reliably indexed by Crossref:

| Key | Canonical source to inspect before portal submission |
|---|---|
| `baevski2020wav2vec` | NeurIPS 2020 proceedings |
| `gretton2012kernel` | Journal of Machine Learning Research, volume 13 |
| `kingma2015adam` | ICLR/OpenReview record |
| `holm1979sequential` | Scandinavian Journal of Statistics, volume 6 |
| `lipton2018bbse` | PMLR volume 80 |
| `long2015dan` | PMLR volume 37 |
| `ganin2016dann` | Journal of Machine Learning Research, volume 17 |
| `cawley2010selection` | Journal of Machine Learning Research, volume 11 |

## Repairs made during submission preparation

- `pastor2023cross` had the correct DOI, volume, issue, pages and year but an
  unrelated author list. It now records Pastor, Ribas, Ortega, Miguel and
  Lleida, as returned by DOI `10.3390/app13169062`.
- `pasad2021layer` now cites the ASRU proceedings version with DOI
  `10.1109/ASRU51503.2021.9688093` and pages 914--921 rather than an arXiv
  placeholder.
- The two duplicate fabricated v1 records, `w2vprosody2023` and `li2023cross`,
  remain deleted. The fabricated `jafari2025feature` record remains replaced by
  the corrected `naeeni2025feature` record.

This file remains named for compatibility with `tools/check_paper.py`; it is a
verification register, not a list of unresolved manuscript placeholders.
