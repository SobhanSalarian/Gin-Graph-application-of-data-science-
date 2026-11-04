# Proposal Writing Instructions (Target: 100/100)

Source: `proposal/Assessment-Project-Proposal (2).pdf` (COMP8240 Assessment Task #1, Session 2 2026), cross-referenced with `instructions .md` (unit-wide project brief) and `plan.md` (our actual project state).

This file translates the official brief into a concrete, section-by-section writing plan for **our specific project** — replicating *"How Powerful are Graph Neural Networks?"* (GIN, Xu, Hu, Leskovec & Jegelka, ICLR 2019) — so nothing generic is left for you to figure out at write-time.

---

## 1. Hard submission requirements (do not lose marks on logistics)

| Requirement | Detail |
|---|---|
| Due | 11:55pm, 28/08/26 (Week 5) |
| Weight | 20 marks = 20% of unit grade |
| Length | 5-6 pages (typical — not a hard cap, but don't pad or run long) |
| Format | **Must** be written in LaTeX using the **ACL template** (Overleaf), submitted as compiled PDF |
| References | ACL template's own reference/citation style (BibTeX, `\citep`/`\citet`) |
| Submission | iLearn dropbox |
| Track | **Novel project** — we are not doing the default project, so the "Justification for Paper Choice" section is required and worth marks (0 marks if default project were chosen) |
| AI use | Open, but you are responsible for correctness — any AI-suggested references must be personally verified as real and relevant before citing |

Action: create the Overleaf doc from the ACL template immediately if not already done; write directly against its section/subsection macros rather than drafting in plain Markdown and porting later, since page budget under the ACL two-column style behaves very differently from a single-column draft.

---

## 2. Rubric — exactly what earns each mark (20 marks total)

| Component | Marks | What "Outstanding (85-100%)" requires |
|---|---|---|
| Quality of description of chosen paper | 4 | Very clearly describes purpose and relevant details, no significant errors |
| Justification of chosen paper (novel project only) | 4 | Justified by **several criteria**, well explained |
| Quality of explanation of evaluation aspects + datasets | 4 | Explains **all** important evaluation aspects clearly, describes **all** datasets used |
| Quality of description of new data | 4 | Very clear, detailed, specific new *existing* datasets **and** the newly constructed data idea |
| Quality of presentation | 4 | Excellent writing, clearly conveys content, conforms to specified format (ACL template, LaTeX, references) |

Read every subsection below with its rubric row in mind — each subsection maps 1:1 to a row.

---

## 3. Report structure (build the LaTeX doc with these sections, in this order)

### 3.1 Title, Abstract, Introduction
Not explicitly demanded by the brief but expected under "well-structured with clear sections and logical flow" (Quality Criteria) and is standard ACL-template practice. Keep the abstract to ~4-6 sentences: what paper, what we'll replicate, what new data, what we expect to find. The intro should state the project goal in one paragraph and give a roadmap of the rest of the proposal.

### 3.2 Description of Paper — targets the "Quality of description" row (4 marks)

Write for a **general CS reader** who knows computing but not this subfield — do not assume graph ML background. Concretely:

- **What problem GIN solves:** Graph Neural Networks (GNNs) learn vector representations of graphs (or nodes) for downstream tasks like classifying whole graphs (e.g., "is this molecule mutagenic?", "is this social network from an Action or Romance movie community?"). Explain **message passing** in plain terms before using the term: each node repeatedly aggregates information from its neighbors and updates its own representation; after several rounds, a whole-graph representation is built by pooling all node representations together.
- **The paper's core question:** existing GNNs (GCN, GraphSAGE, etc.) worked well empirically but nobody had characterized *how expressive* they actually are — i.e., which pairs of distinct graphs can a GNN tell apart, and which can it never tell apart no matter how it's trained? The paper answers this theoretically.
- **The key result, explained without jargon:** the authors prove that the best any message-passing GNN can theoretically do is match the **Weisfeiler-Lehman (WL) test**, a classical (non-learned) graph isomorphism heuristic. They then show most popular GNNs (GCN, GraphSAGE) are strictly *less* powerful than the WL test because they use aggregation functions (mean, max) that can't distinguish certain multiset inputs (e.g., mean-pooling can't tell a node with neighbor-features {1,1} from one with {1,1,1,1} — same average, different graph). They design a new architecture, **GIN** (Graph Isomorphism Network), using sum-aggregation plus an MLP, and prove it *is* as powerful as the WL test — the theoretical ceiling for this whole model class.
- **Give a worked/visual example** (rubric explicitly rewards examples in this section, per the sample-paper walkthrough in the brief): the classic example from the paper itself is two graphs where one node has neighbor multiset {1,1} and another has {1,1,1,1} but mean-aggregation collapses both to the same value while sum-aggregation does not. Draw this as a small figure (2 tiny graphs side by side) — cheap to make, high clarity payoff for the rubric's "no significant errors, very clear" bar.
- **What the paper contributes concretely:** (1) a theoretical framework for GNN expressiveness, (2) proof that GIN is maximally powerful among message-passing GNNs, (3) empirical validation on 9 benchmark graph-classification datasets showing GIN matches or beats existing GNNs and even a non-learned WL-subtree kernel baseline.

Do **not** just paraphrase the abstract — the brief explicitly warns the abstract assumes reader background (their own example: "unsupervised topic segmentation" needs unpacking). Every technical term (message passing, aggregation, pooling, Weisfeiler-Lehman test, multiset) needs one plain-English clause the first time it's used.

### 3.3 Explanation of Evaluation — targets the "evaluation aspects + datasets" row (4 marks)

This is the section most students under-deliver on. The brief's own worked example (P_k/WindowDiff for topic segmentation) is the bar: **define the metric, cite where it's from, and walk a numeric example.**

Cover, each with a citation to the paper that originally defined it (not just the GIN paper, which just uses them):

1. **Graph classification accuracy** — straightforward, but state precisely: mean test accuracy across folds, at the epoch selected by the model-selection rule below. Not just "accuracy" — say what's being classified (a whole graph → one of *k* class labels, e.g. genre or molecule toxicity) since that's non-obvious without more context.
2. **10-fold cross-validation protocol** — the paper doesn't use a fixed train/test split; explain why (small datasets, e.g. MUTAG has only 188 graphs, so CV reduces variance) and how it works (partition into 10 folds, train on 9, test on 1, rotate, report mean ± std across the 10 folds).
3. **The paper's specific model-selection rule** — this is a subtle methodological point the paper's main text doesn't spell out in the abstract/intro, exactly the kind of "assumed knowledge" the brief warns about: for each of the 10 folds, they track test accuracy per epoch; they average the 10 folds' curves together, find the epoch with the single best averaged accuracy, and report the mean ± std at *that* epoch (not the best epoch per fold, which would overfit). We independently confirmed this from the official code (`main.py`) while building `replication/run_gin.py` — mention that we verified this from source, since the paper's text alone underspecifies it. This is a good, genuine "worked example" opportunity: show the rule with a tiny illustrative numeric table (e.g., 3 toy folds × a few epochs) rather than just describing it in prose.
4. **Baselines compared against** — WL subtree kernel (non-learned graph kernel), DCNN, PatchySAN, DGCNN, and, importantly for us, **GraphSAGE and a sum/mean/max-pooling GNN family** — explain briefly what each represents at a high level (kernel method vs. earlier GNN variant) so the comparison table means something to a non-expert reader.
5. **Datasets used in the paper** — describe all 9 (MUTAG, PTC, PROTEINS, NCI1, COLLAB, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI5K), but go deepest on **IMDB-BINARY and IMDB-MULTI** since those are the two we're actually replicating. For these two:
   - Cite the *original* source: Yanardag & Vishwanathan, "Deep Graph Kernels," KDD 2015 — the paper that actually released these datasets (GIN's paper just reuses them; the brief explicitly requires citing the release paper, not just the paper that uses it).
   - Construction: nodes = actors, edge between two actors if they co-starred in a movie, one graph = one actor's ego-network of collaborators, graph label = genre (IMDB-BINARY: Action vs. Romance; IMDB-MULTI: Comedy vs. Romance vs. Sci-Fi).
   - Give the actual stats table (already computed, reuse from `plan.md` §4): # graphs, # classes, avg # nodes.
   - Node features: the paper uses **node degree as the only feature** for these two datasets (`--degree_as_tag`) since IMDB has no attributes — flag this, it matters later for interpreting our new-data results.
6. **LLM-as-judge (new evaluation, only light detail needed per the brief — Week 7+ topic):** 2-3 sentences. Since GIN's task is graph classification with hard ground-truth labels (not open-ended generation), an LLM-as-judge is not a drop-in replacement for accuracy. Propose where it *could* plausibly add value in this project: qualitatively judging *why* the model failed on specific misclassified new-data graphs — e.g., feeding an LLM a text description of a misclassified ego-network (cast list + genre) and asking it to judge whether the genre label is even plausible/clean, as a sanity check on our TMDb-scraped label quality, complementing the quantitative accuracy numbers. State explicitly this is preliminary and will be refined once the unit covers LLM-as-judge methodology in Week 7.

### 3.4 Justification for Paper Choice (novel project only) — targets its own 4-mark row

The brief asks for justification "by some indicator of quality" plus a **feasibility** case. Structure as explicit sub-bullets/subsections so the marker can see "several criteria, well explained":

**Quality indicators:**
- **Venue:** ICLR, CORE 2026 ranking **A\*** (confirmed via CORE Portal, https://portal.core.edu.au/conf-ranks/2273/), a top-tier venue for representation learning / deep learning theory.
- **Citation impact:** this is one of the most cited papers in graph representation learning; state the *live* citation count pulled from Google Scholar or Semantic Scholar at time of writing rather than a number frozen now (it moves constantly — check https://scholar.google.com or the Semantic Scholar page for arXiv:1810.00826 and cite it with an access date). Frame it as "widely regarded as a foundational theoretical result for the GNN field" — this is well established regardless of the exact number.
- **Community adoption / code health:** official repo `github.com/weihua916/powerful-gnns` has **~1.2k stars and 224 forks** (checked 2026-08-29) — strong evidence of real-world reuse, not an abandoned research artifact.
- **Theoretical grounding:** unlike many empirical GNN papers, this one proves a tight upper bound on expressiveness — a stronger, more citable kind of contribution than "we tried a new architecture and it worked."

**Feasibility case (the brief explicitly asks: is code accessible, does it look implementable, is original data available):**
- Official code is public, MIT-style research repo, runs on the exact stack we tested (PyTorch, plain Python — no GPU-only closed dependencies).
- **We have already downloaded and successfully executed the code** — required as a "basic requirement" per the brief. State plainly: cloned `weihua916/powerful-gnns`, installed dependencies (including one undocumented but needed dependency, `scikit-learn`, discovered on a clean venv), ran the official `main.py` end-to-end on MUTAG as a smoke test, completed in seconds with accuracy in the expected range. This single sentence satisfies the brief's explicit minimum bar — do not skip it, it's a named requirement, not just supporting color.
- **Original data availability confirmed:** all 9 benchmark datasets ship bundled inside the repo's own `dataset.zip` — zero external download risk, a meaningfully lower-risk situation than papers whose data has gone stale/delisted.
- **Task shape suits the "new data" requirement well:** graph classification with a clean train/test-style CV protocol is much easier to re-target at freshly scraped data than, say, transductive node classification (where train/test nodes must live in one fixed graph) — directly relevant to why this is a *feasible* novel project, not just an interesting one.
- Optional but strengthens this section: mention that at proposal time we've gone beyond the minimum bar — we've already completed full replication runs (see `plan.md` §4) and built a working new-data pipeline. Framing note: the brief says "not expected to go into technical details at this stage; that will be covered in the Update Presentation" — so **do not** dump full result tables here; one sentence ("early runs are already tracking closely with the paper's reported numbers") is enough to reinforce feasibility without preempting the next assessment.

### 3.5 New Data — targets the "description of new data" row (4 marks)

The brief requires **two distinct kinds**, both required:

**(1) Existing datasets not yet tried with this method — discuss ~5 candidates, assess feasibility of each.**
Do not just name-drop; for each give: what it is, where to get it, why GIN could plausibly apply (does it naturally form a graph-classification task?), and a genuine feasibility note (size, licensing, whether labels exist). Suggested five, mixing "safe" and "reach" options so the marker sees real assessment rather than a rubber-stamped list:
1. **TU Dataset collection additions not in the original 9** (e.g., ENZYMES, DD from the same TU Dortmund benchmark repository) — highest feasibility, already in GIN's expected input format, but weakest as "new" since they're adjacent to the original benchmark suite; good to name but flag as the least novel option.
2. **OGB (Open Graph Benchmark) graph-classification datasets** (e.g., ogbg-molhiv) — high feasibility (maintained by the same Stanford group as this paper, PyTorch-native loaders), larger/more modern than the original 9, but would need a small adapter to GIN's TU-format expectations.
3. **A HuggingFace-hosted graph dataset** (search HF Datasets Hub for "graph classification") — feasibility depends entirely on format; note this needs a concrete dataset name once checked, not a generic pointer.
4. **Kaggle molecule/chemistry datasets** (e.g., QM9-derived or Tox21 subsets) — plausible, but assess licensing/labeling quality per-dataset before committing.
5. **A co-authorship or citation-network dataset (OpenAlex API)** — direct structural analogue to COLLAB (one of the original 9's benchmark types), highest conceptual fit, but heavier to build (need to define graph-per-entity boundaries ourselves) — this is really a bridge to the "newly constructed" category below, worth noting as such.

For each, be explicit about feasibility (rubric rewards *specificity*, and the HD band requires "detail," not just naming): can we get it in the TU-style format GIN expects (graph count header, per-graph node/edge/label lines, as used in `util.py`) with reasonable effort, or would it need substantial reformatting?

**(2) Newly constructed data — this is our strongest section; be concrete and detailed (this is what separates HD from Credit on this row).**
Describe the TMDb-based construction already built (`new_data/scraping/build_tmdb_dataset.py`), matching the brief's own example (splitting/concatenating Wikipedia articles for topic segmentation) in spirit — we are *mirroring the original dataset's construction recipe* on fresh source data:
- **Source & method:** TMDb API (free key, ~40 req/s limit, requires attribution — quote the required line: *"This product uses the TMDb API but is not endorsed or certified by TMDb"*).
- **Construction rule:** movies released 2016-01-01 onward (guarantees zero overlap with the 2015 original Yanardag & Vishwanathan dataset), single-genre-only filter to avoid ambiguous labels, top-billed ~20-cast per movie as graph nodes, edges from co-appearance — reproduces the "ego-network of collaboration" structure of the original rather than trivial per-movie cliques.
- **Scale achieved:** IMDBBINARY-NEW = 996 graphs (500 Action / 496 Romance), IMDBMULTI-NEW = 1492 graphs (Comedy 499 / Romance 493 / Sci-Fi 500) — comparable scale to the originals (1000 / 1500).
- **Output format compatibility:** written directly in GIN's expected TU format, so it drops into the existing pipeline with zero code changes — a concrete, checkable claim the marker can verify against the repo.
- **Honest data-quality note (this is a strength, not a weakness, if framed right):** our graphs are noticeably denser than the originals (avg ~178 edges/graph vs. ~96.5) because actor overlap only occurs *within* our sampled movie pool; flag this now as an anticipated methodological difference we'll investigate, which shows critical awareness the rubric's HD band rewards ("in some detail the... idea for newly constructed data").
- **On human judgements (brief says light detail is fine, Week 5+ topic):** one or two sentences — genre labels here come directly from TMDb's own metadata (not crowd-sourced), so no human-annotation pipeline is needed for this dataset; if we pursue a second new-data source (e.g., OpenAlex co-authorship) later, that one likewise inherits ground-truth labels from the source platform rather than requiring fresh annotation.

### 3.6 Conclusion / Plan Forward
Short paragraph: recap what will happen for the Update Presentation (deeper replication analysis, investigation experiments) and Final Report. Keep to a few sentences — this isn't a rubric row but rounds out "logical flow."

### 3.7 References (BibTeX via ACL template)
Must include at minimum: the GIN paper itself, Yanardag & Vishwanathan (IMDB dataset source), the WL-test original reference (Weisfeiler & Leman, 1968, or a standard modern citation for it), any baseline papers named (GraphSAGE — Hamilton et al. 2017), and the TMDb API. Verify every reference resolves to a real, correctly-attributed source before submitting — the brief explicitly flags this as your responsibility even when AI-suggested.

---

## 4. Quality Criteria checklist (from the brief, verify before submitting)

- [ ] Well-structured, clear sections, logical flow
- [ ] Accurate characterisation of the paper (no factual errors about what GIN proves or how it works)
- [ ] Paper choice justified by **several** distinct criteria, each explained (not just listed)
- [ ] Evaluation methodology explained clearly, **with a worked example** (the model-selection rule numeric walkthrough)
- [ ] All datasets used in the paper described, with original release papers cited (Yanardag & Vishwanathan for IMDB)
- [ ] Initial LLM-as-judge thoughts included (brief, since it's a later-week topic)
- [ ] Both new-data categories present: ≥5 existing-dataset candidates assessed for feasibility, **and** the TMDb newly-constructed dataset described in detail
- [ ] Confirmed-working code execution explicitly stated (the "basic requirement" sentence)
- [ ] LaTeX + ACL template + ACL reference style, 5-6 pages, compiles cleanly to PDF
- [ ] No unverified AI-suggested references

## 5. Common ways to lose marks (avoid these)

- Paraphrasing the GIN abstract without unpacking jargon (message passing, WL test, aggregation) — the brief's own worked example exists specifically to warn against this.
- Treating "Justification" and "Description" as the same section — they're graded separately (4 marks each); keep them as clearly distinct subsections.
- Listing datasets without depth on the two (IMDB-BINARY/MULTI) that are actually load-bearing for this project.
- Forgetting to cite the *original* dataset release paper (Yanardag & Vishwanathan), not just GIN's paper for the dataset description.
- Going deep into replication *results* here — the brief is explicit that technical depth belongs in the Update Presentation, not the proposal; this section should demonstrate feasibility, not report findings.
- Skipping the explicit "we downloaded and ran the code successfully" sentence — it's a named basic requirement, easy to lose a mark on by assuming it's implied.
- Missing the ACL template/format entirely (instant loss on the "Quality of presentation" row, worth 4 marks).
