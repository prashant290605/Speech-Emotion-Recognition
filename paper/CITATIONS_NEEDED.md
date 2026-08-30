# Citations needed

Every `[CITE: ...]` placeholder in the manuscript, listed for you to fill.

**Why placeholders rather than plausible-looking references.** The v1 submission
carried three fabricated bibliography entries — real papers attributed to
invented author lists — and that is the single defect that made it
unpublishable. A draft with honest gaps is correct; a draft with invented
authors is not. No entry is added to `refs.bib` unless it has been read.

Fill each row by opening the source, then add the entry to `paper/refs.bib` and
replace the `[CITE: ...]` text with a `\cite{key}`. Re-run
`python tools/check_paper.py` afterwards — it fails if a placeholder is left in
the text but missing from this file.

---

## Method

| § | placeholder | what is needed | why it is cited |
|---|---|---|---|
| 3.4 | Ledoit and Wolf, well-conditioned estimator for large-dimensional covariance matrices | Ledoit, O. and Wolf, M., *Journal of Multivariate Analysis* 88(2), 2004, 365–411. DOI `10.1016/S0047-259X(03)00096-4` | The parameter-free CORAL shrinkage variant |
| 3.4 | Kingma and Ba, Adam | Kingma, D. P. and Ba, J., *Adam: A Method for Stochastic Optimization*, ICLR 2015. arXiv:1412.6980 | Optimiser for the MK-MMD rungs |
| 3.7 | cluster bootstrap / bootstrap for clustered data | A standard reference. Candidates: Field & Welsh (2007) *JRSS-B* on bootstrapping clustered data; or Cameron, Gelbach & Miller (2008) *Review of Economics and Statistics* on cluster-robust inference. Pick whichever the venue's readership will recognise | Justifies resampling speakers rather than utterances |
| 3.7 | Holm 1979 | Holm, S., *A simple sequentially rejective multiple test procedure*, Scandinavian Journal of Statistics 6(2), 1979, 65–70 | Multiple-comparison correction over the pre-registered family |

---

## Still to be written

Related Work is expected to add substantially to this list. Sections not yet
drafted: Introduction, Related Work, Results, Discussion, Reproducibility,
Conclusion, back matter.

---

## Bibliography entries needing manual DOI verification

Not placeholders — these are in `refs.bib` and are cited — but the Phase 1
checker could not confirm them because Crossref does not index the venue. That
is normal for NeurIPS and JMLR and is **not** evidence of a problem. Both need
one click each before submission:

| key | what to verify | link |
|---|---|---|
| `baevski2020wav2vec` | Authors, volume 33, pages 12449–12460, year 2020 | [Crossref search](https://search.crossref.org/search/works?q=wav2vec+2.0%3A+A+framework+for+self-supervised+learning+of+speech+representations&from_ui=yes) |
| `gretton2012kernel` | Authors, JMLR vol. 13, pages 723–773, year 2012 | [Crossref search](https://search.crossref.org/search/works?q=A+kernel+two-sample+test&from_ui=yes) |

Three further entries were resolved during Phase 1 and need no action, but the
history matters if a reviewer asks:

- `naeeni2025feature` — the v1 entry `jafari2025feature` claimed
  "M. Jafari, F. Shahin, and A. Alavi", vol. 187. The DOI record is **Naeeni
  and Nasersharif, vol. 194**. `refs.bib` uses the corrected record and the v1
  author list appears nowhere.
- `w2vprosody2023` — **deleted.** Duplicated `naderi2023cross` under a disjoint
  author list; never cited.
- `li2023cross` — **deleted.** Duplicated `fu2023cross` under a disjoint author
  list, claiming the same volume, issue and pages; never cited.
