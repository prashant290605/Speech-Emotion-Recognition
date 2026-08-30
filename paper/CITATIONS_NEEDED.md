# Citations needed

Every `[CITE: ...]` placeholder in the manuscript, listed for you to fill.

**Why placeholders rather than plausible-looking references.** The v1
submission carried three fabricated bibliography entries — real papers
attributed to invented author lists — and that is the single defect that made
it unpublishable. A draft with honest gaps is correct; a draft with invented
authors is not. No entry is added to `refs.bib` unless it has been read.

**This file deliberately does not give you author lists, volumes or page
numbers.** An earlier version of it did, generated from model memory, which is
the same failure mode one step removed: an unverified author list is no safer
for being in a checklist than in a bibliography. Each row below identifies the
work well enough to find it and says what to verify once you have the
publisher's page open. Take the metadata from that page, not from here and not
from any assistant.

Procedure per row: open the publisher or proceedings page, copy authors,
venue, volume, pages and year from it, add the entry to `paper/refs.bib`, and
replace the `[CITE: ...]` text with `\cite{key}`. Then re-run
`python tools/check_paper.py`, which fails if a placeholder is left in the text
but missing from this file.

---

## Method

| § | placeholder in the text | how to find it | what to verify |
|---|---|---|---|
| 3.4 | Ledoit and Wolf, well-conditioned estimator for large-dimensional covariance matrices | Search the exact title; it is a *Journal of Multivariate Analysis* paper from the early 2000s | authors, volume, issue, pages, year |
| 3.4 | Kingma and Ba, Adam: a method for stochastic optimization | Search the exact title; ICLR, mid-2010s. Decide whether to cite the conference version or the arXiv preprint and be consistent with the venue's convention | authors, venue, year, and which version you are citing |
| 3.7 | a cluster-bootstrap / clustered-data bootstrap reference | **Cite whichever source you actually rely on**, not a canonical-sounding one. The claim being supported is that resampling clusters (here, speakers) rather than individual observations is the correct unit when observations within a cluster are correlated | authors, venue, volume, pages, year — and that the paper actually supports the claim as stated |
| 3.7 | Holm, a simple sequentially rejective multiple test procedure | Search the exact title; *Scandinavian Journal of Statistics*, late 1970s | authors, volume, issue, pages, year |

Three of these four are standard enough that you will recognise the record
immediately; the cluster-bootstrap one is the one to think about, because the
right citation depends on which justification you want to lean on.

---

## Results

| § | placeholder in the text | how to find it | what to verify |
|---|---|---|---|
| 4.3 | black-box shift estimation, and the EM procedure for adjusting classifier outputs to new priors — **two separate sources** | The first is the label-shift correction that estimates target priors by inverting a source confusion matrix; the second is the expectation-maximisation procedure for re-estimating class priors from a classifier's outputs on unlabelled data, from the neural-computation literature of the early 2000s. Our implementation is described in Section 3; cite the methods it implements | authors, venue, volume, pages, year for each; and that each source describes the estimator we actually implemented rather than a later variant |
| 4.7 | reporting practice: work that justifies an alignment method by the discrepancy it removes without stating the measurement geometry | This one is a **negative claim about the literature**, so it needs either a survey that characterises reporting practice or a small set of representative papers you have actually read. It supports the sentence saying the kernel bandwidth and basis are not customarily reported | that the sources really do omit the frame, and that they are representative rather than cherry-picked. **If you cannot support the claim at that strength, weaken the sentence in `results.tex` rather than stretching the citation** --- the well-posedness result does not depend on it |

On 4.3: both estimators are named in the text. If you decide to cite a survey
or a later formulation instead of the originals, change the sentence in
`results.tex` to match what is being cited.

On 4.7: this is the only placeholder in the manuscript whose sentence should be
*rewritten* if the source turns out not to support it. The others are ordinary
attributions.

---

## Still to be written

Sections not yet drafted: Introduction, Discussion, Reproducibility, Conclusion,
back matter. Discussion is expected to add a small number of further
placeholders; the others should add none, since they restate results rather than
positioning them against other work.

---

## Bibliography entries needing manual DOI verification

Not placeholders — these are already in `refs.bib` and cited — but the Phase 1
checker could not confirm them because Crossref does not index the venue. That
is normal for NeurIPS and JMLR and is **not** evidence of a problem. Each needs
one click before submission, and the metadata currently in `refs.bib` for them
should be treated as unverified until you have checked it against the source.

| key | venue it should be | link to verify against |
|---|---|---|
| `baevski2020wav2vec` | NeurIPS | [Crossref search](https://search.crossref.org/search/works?q=wav2vec+2.0%3A+A+framework+for+self-supervised+learning+of+speech+representations&from_ui=yes) |
| `gretton2012kernel` | JMLR | [Crossref search](https://search.crossref.org/search/works?q=A+kernel+two-sample+test&from_ui=yes) |

Three further entries were resolved during Phase 1 and need no action, but the
history matters if a reviewer asks:

- `naeeni2025feature` — the v1 entry `jafari2025feature` claimed an author list
  and volume that the DOI record contradicts. `refs.bib` now carries the record
  from the DOI, under a different key so the fabricated key cannot be cited by
  accident. Verify against
  [10.1016/j.compbiomed.2025.110510](https://doi.org/10.1016/j.compbiomed.2025.110510).
- `w2vprosody2023` — **deleted.** Duplicated `naderi2023cross` under a disjoint
  author list; never cited.
- `li2023cross` — **deleted.** Duplicated `fu2023cross` under a disjoint author
  list, claiming the same volume, issue and pages; never cited.

`tools/check_paper.py` fails if any of those three keys reappears in
`refs.bib`.
