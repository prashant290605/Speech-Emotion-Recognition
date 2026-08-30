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

On 4.3: both estimators are named in the text. If you decide to cite a survey
or a later formulation instead of the originals, change the sentence in
`results.tex` to match what is being cited.

The 4.7 placeholder that stood here has been removed, not filled. It asked for
evidence that the literature does not state its measurement geometry -- a
negative claim about the literature, which cannot be verified without reading
it. The sentence is now a recommendation (state the geometry you measured in),
which needs no citation and cannot be falsified by one paper that happened to
state its frame.

---

## Candidate records supplied for verification (NOT yet in refs.bib)

Four records have been supplied by the supervisor with enough detail to check.
**None has been added to `refs.bib`.** They go in only after the publisher page
has been opened and the metadata taken from it, exactly as with every other
entry -- a record handed over in a message is still a record nobody in this
repository has verified.

| for | record to check | where to check it | what to watch |
|---|---|---|---|
| RW #3, and the divergence wording in 4.7 | Ben-David, Blitzer, Crammer, Kulesza, Pereira, Wortman Vaughan, "A theory of learning from different domains", *Machine Learning* 79(1--2):151--175, 2010 | [10.1007/s10994-009-5152-4](https://doi.org/10.1007/s10994-009-5152-4) | that the divergence really is **hypothesis-class-dependent**, defined over the symmetric difference hypothesis space, and estimable from finite unlabelled samples. Sections 2.2 and 4.7 have been rewritten on that basis and must be re-read against the paper |
| RW #4 | Sun, Feng, Saenko, "Return of Frustratingly Easy Domain Adaptation", *Proc. AAAI-16*, 2058--2065 | [10.1609/aaai.v30i1.10306](https://doi.org/10.1609/aaai.v30i1.10306) | that it is the **closed-form** second-order method our `coral` rung implements, and distinct from `sun2016deep` |
| RW #11 and Results 4.3 (BBSE) | Lipton, Wang, Smola, "Detecting and Correcting for Label Shift with Black Box Predictors", ICML 2018, PMLR vol. 80 | the PMLR proceedings page | **page numbers are disputed** -- dblp gives 3128--3136, PMLR-derived citations give 3122--3130. Take them from the PMLR page itself, not from a citation manager |
| Results 4.3 (EM) | Saerens, Latinne, Decaestecker, "Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure", *Neural Computation* 14(1):21--41, 2002 | the MIT Press page for the volume | the third author is **Decaestecker**. This repository already spells it correctly in `results.tex`, `PHASES.md` and `src/ser/analysis/shift.py`; a claim that it read "Decock" was checked and is not the case |

Filling these four closes RW #3, #4, #11 and the Results 4.3 row -- five
placeholders, including the one the well-posedness argument leans on.

---

## Related work

Eighteen placeholders, which is expected for a related-work section written
without source access. Nothing here is a guess at a reference: each row says
what claim needs support and what would count as supporting it.

Three of these rows say what to do **if the source does not exist or does not
support the claim at that strength** — 2.2 (ablation practice), 2.3 (layer-wise
probing for paralinguistics) and 2.4 (critique of target-domain tuning). Follow
that instruction rather than substituting the nearest thing you can find. None
of the paper's five contributions depends on any of these eighteen.

### 2.1 Cross-corpus speech emotion recognition

| # | claim it supports | what would count |
|---|---|---|
| 1 | reported gains are not commensurable across papers, because pair, baseline, label mapping and split policy all vary | a survey or meta-analysis of cross-corpus SER that actually documents this heterogeneity. A survey that merely lists methods does not support the claim |

### 2.2 Feature-space alignment under covariate shift

| # | claim it supports | what would count |
|---|---|---|
| 2 | the covariate-shift assumption and the importance-weighting correction that follows from it | the standard statement of covariate shift and reweighting. Verify it states the assumption in the form we use (marginal differs, conditional does not) |
| 3 | target error is bounded by source error plus a **classifier-induced** divergence, defined over the symmetric difference hypothesis space | the domain-adaptation generalisation bound; a candidate record is supplied above. **This is the load-bearing one for Section 4.7**, which now argues that a fixed-bandwidth MMD is not the quantity the bound is stated over. Verify the divergence is hypothesis-class-dependent in the version you cite; if it is not, Sections 2.2 and 4.7 both need rewording |
| 4 | CORAL in closed form (whiten source, recolour to target covariance) | the original CORAL paper, which is *not* `sun2016deep` — that is the deep variant, already in `refs.bib`. Both should be cited, and they are different papers |
| 5 | deep domain adaptation minimising a single-kernel MMD between hidden-layer distributions | the paper that introduced MMD as a hidden-layer adaptation loss |
| 6 | multi-kernel MMD as a domain-adaptation objective | the formulation our `mkmmd_diag` / `mkmmd_full` rungs implement. Verify the kernel family matches what Section 3 describes |
| 7 | adversarial domain adaptation via a gradient reversal layer or equivalent discriminator | the canonical formulation. Cited only to place the family; no claim is made about its performance |
| 8 | the full ladder is rarely run, so the share of the gain from the cheapest rung is seldom reported | an ablation-practice survey. **If no source supports this as a general claim, narrow the sentence in `related.tex` to the specific papers already cited in 2.1** rather than citing something weaker |
| 9 | kernel bandwidth is typically set by the median heuristic | the standard reference for the median heuristic. This is cited to establish common practice, so a source that *uses* it is weaker than one that *proposes or analyses* it |
| 10 | conditional or target shift, where the class-conditional distribution itself changes | the formulation matching our decomposition in Section 4.3 |
| 11 | label-shift correction by inverting a source confusion matrix | **the same source as the Results row above (BBSE)** — fill both together and cite the same key |

### 2.3 Self-supervised representations

| # | claim it supports | what would count |
|---|---|---|
| 12 | the learned layer-weighting convention for downstream speech tasks | the benchmark that established it. Verify the weighted-sum-over-layers protocol is actually specified there, since that is what we compare against |
| 13 | information in SSL speech encoders is not uniform across depth, and the most informative layers are not the last | a layer-wise probing analysis. Verify it reports the peak at intermediate layers, which is the part the sentence relies on |
| 14 | a comparable layer-wise analysis exists for paralinguistic targets | a probing analysis for emotion or another paralinguistic task. **If none exists, delete the clause.** Do not stretch a phonetic or word-level result to cover emotion — our own layer result would then be the evidence, and it is reported as such in Section 4.6 |

### 2.4 Protocol

| # | claim it supports | what would count |
|---|---|---|
| 15 | transfer-aware model selection exists: importance-weighted cross-validation, reverse or transfer cross-validation | one source per method, or one covering both. These are named as alternatives we did not use, so the citation only needs to establish they exist |
| 16 | reporting the grid maximum on the target is reporting an oracle, and the distinction is not always drawn | a methodological critique of tuning on the test domain. A general machine-learning statement is acceptable if none is specific to SER. **If the "not always drawn" half cannot be supported, drop that half** — our oracle column stands on its own |
| 17 | a classification metric reported without its chance level cannot be read | a source recommending explicit chance-level or permutation baselines |
| 18 | speaker-independent splits are necessary because speaker identity is recoverable from these representations | evidence that speaker identity is recoverable from SSL speech features, or that speaker-dependent splits inflate SER scores. Either supports the sentence; the first is the stronger form |

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
