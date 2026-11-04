# Slide Content: Project Update Presentation (COMP8240, Novel Project)

**Presenter:** Sobhan Salarian · **Paper:** Xu et al., *How Powerful are Graph Neural Networks?* (ICLR 2019)
**Format:** 5-minute MP4 (must be 4:30–5:30), face in a corner window throughout. **Due:** 11:55 pm, 9 Oct 2026.

Every number in this file was checked against the raw repo files on 2026-10-07 (`results/logs/*.json`, the dataset `.txt` files, `plan.md`, the scraper source).

---

## Instructions for the AI slide generator (read first)

1. **Use only the facts in this file.** Do not invent numbers, citations, logos, dataset names or results. If something looks missing, leave a placeholder; do not fill it in.
2. **12 slides, about 5 minutes.** Keep each slide sparse: a title, 3 to 5 short bullets or one table or one diagram. The detail goes in the speaker notes, not on the slide.
3. **Keep the bottom-right corner clear on every slide** (about 22% width × 28% height). The presenter's face-cam goes there.
4. Style: clean, academic, light background, one accent colour (e.g. deep blue) plus one warning colour (amber) used **only** for the "problem we found" content. Tables need large fonts (≥ 18 pt).
5. **Colour key for results:** paper numbers in grey, our numbers in the accent colour, new-data numbers in amber.
6. Speaker notes per slide = the script below (about 140 words per minute).
7. Do not put the logos of ICLR, TMDb, Macquarie or GitHub on any slide. Use text only.

---

## Timing plan (mapped to the brief's suggested structure)

| # | Slide | Time | Brief section | Rubric item |
|---|---|---|---|---|
| 1 | Title | 0:00–0:08 | Recap | Presentation |
| 2 | The paper and why we chose it | 0:08–0:30 | Recap: justification | Recap/Justification (6) |
| 3 | Original datasets | 0:30–1:15 | Recap: datasets | Recap/Justification (6) |
| 4 | Core idea: why sum beats mean/max | 1:15–1:40 | Recap: architecture | Replication (8) |
| 5 | System architecture and our pipeline | 1:40–2:10 | Recap: architecture, code availability | Replication (8) |
| 6 | Running the original code (live demo) | 2:10–2:45 | Recap: run code, explain output | Replication (8) |
| 7 | Replication: environment and what went wrong | 2:45–3:05 | Update: replication | Replication (8) |
| 8 | Replication results vs the paper | 3:05–3:30 | Update: replication | Replication (8) |
| 9 | New data: existing and constructed | 3:30–3:55 | Update: new data description | New Data: Description (8) |
| 10 | Constructed dataset: properties and complexity | 3:55–4:25 | Update: new data description | New Data: Description (8) |
| 11 | Construction progress and a problem we found | 4:25–4:50 | Update: progress | New Data: Progress (8) |
| 12 | Wrap-up and next steps | 4:50–5:05 | Wrap-up | Presentation (10) |

Recap ≈ 2:45 · Update ≈ 2:05 · Wrap-up ≈ 0:15 · **Total ≈ 5:05** (inside the 4:30–5:30 window).

---

## SLIDE 1: Title (0:00–0:08)

**On slide:**
- Title: **Replicating and Extending "How Powerful are Graph Neural Networks?"**
- Subtitle: Project Update · COMP8240 Applications of Data Science · Session 2, 2026
- Name: Sobhan Salarian · Macquarie University
- Small line: Novel project · Code & data: `github.com/SobhanSalarianCognitivo/Gin-Graph`

**Visual:** a small abstract graph (nodes and edges) as a background motif. No logos.

**Speaker notes (≈20 words):**
> Hi, I'm Sobhan. This is my project update on replicating the Graph Isomorphism Network paper, and testing it on new data I built.

---

## SLIDE 2: The paper and why we chose it (0:08–0:30)

**On slide:**
- **Paper:** *How Powerful are Graph Neural Networks?* Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
- **Venue:** ICLR 2019, **CORE A\***
- **Code:** official PyTorch repo `github.com/weihua916/powerful-gnns` (public, with all 9 benchmark datasets bundled)
- **Why this paper:**
  - It proves a ceiling: no message-passing GNN can beat the **Weisfeiler–Lehman (WL) test**, and **GIN reaches it**
  - Graph classification with 10-fold CV, so it is easy to point the same pipeline at new data
  - The IMDB benchmarks are built from public movie metadata, so we can rebuild them from scratch
  - Feasible: the code ran unmodified on modern PyTorch during the proposal stage

**Optional placeholder (fill in yourself, do not let the AI guess):** "Google Scholar citations: ____ (checked __/10/2026)". The proposal also recorded "over 1,000 GitHub stars and over 200 forks" for the official repo; recheck before quoting.

**Visual:** four short "reason" tiles (Theory · Venue · Feasible · Rebuildable data).

**Speaker notes (≈55 words):**
> As a reminder, the paper is by Xu, Hu, Leskovec and Jegelka, published at ICLR 2019, a CORE A-star venue. The official code is public on GitHub. I chose it because it doesn't just propose a model, it proves a limit on what graph neural networks can tell apart. Its task is also easy to re-run on new data.

---

## SLIDE 3: Original datasets (0:30–1:15)

**On slide:**
- The paper uses **9 benchmarks**: 4 bioinformatics (MUTAG, PTC, NCI1, PROTEINS) and 5 social (COLLAB, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI5K)
- **We focus on IMDB-BINARY and IMDB-MULTI.** Source: Yanardag & Vishwanathan, *Deep Graph Kernels*, KDD 2015
  - **Node** = an actor · **Edge** = two actors co-starred in a film · **Graph** = an actor's collaboration ego-network · **Label** = genre

| Dataset | Graphs | Classes | Labels | Avg nodes |
|---|---|---|---|---|
| IMDB-BINARY | 1,000 | 2 | Action vs Romance | 19.8 |
| IMDB-MULTI | 1,500 | 3 | Comedy vs Romance vs Sci-Fi | 13.0 |

- **Why the paper uses them:** no node attributes, so the model must rely on **graph structure alone** (node degree is the only input feature, via `--degree_as_tag`)
- **Why we chose these two:** they are the only ones we can rebuild with new data, which gives a clean chain: **paper → our replication → same pipeline on new data**. The other 7 were dropped on purpose (scope choice, not a compute limit).

**Visual:** a small sketch of an ego-network (one central actor, co-stars around them, some co-stars linked to each other).

**Speaker notes (≈95 words):**
> The paper evaluates on nine standard graph-classification benchmarks. I focus on the two IMDB datasets. They come from the Deep Graph Kernels paper at KDD 2015. Each graph is an actor's collaboration network: nodes are actors, edges mean two actors appeared in a film together, and the label is the genre. IMDB-BINARY is Action versus Romance with a thousand graphs; IMDB-MULTI adds Sci-Fi and Comedy, with fifteen hundred. The paper uses them because they have no node features, so the model can only use structure. I chose these two because I can rebuild them from new movie data, giving a direct comparison.

---

## SLIDE 4: Core idea, why sum beats mean and max (1:15–1:40)

**On slide:**
- A GNN updates each node by **aggregating its neighbours**, repeated over several layers, then **pools** all nodes into one graph vector
- **GIN update rule:**
  `h_v^(k) = MLP^(k)( (1 + ε^(k)) · h_v^(k−1) + Σ_{u ∈ N(v)} h_u^(k−1) )`
- **Mean** and **max** aggregation can't tell some different neighbourhoods apart. **Sum** can.
  - Example: 2 identical neighbours vs 4 identical neighbours → same mean, same max, **different sum**
- Paper's claim: GIN (sum + MLP) is **as powerful as the WL test**; GCN (mean) and GraphSAGE (max) are **strictly weaker**

**Visual:** reuse `proposal/figure1.png` (the mean-vs-sum neighbourhood illustration from the proposal).

**Speaker notes (≈60 words):**
> Quick idea of the method. A graph neural network updates each node by combining its neighbours' features. The paper shows the choice of combination matters. Mean or max can't tell two identical neighbours from four, but sum can. So GIN uses sum plus a small neural network, and the paper proves this matches the Weisfeiler-Lehman test, the theoretical ceiling.

---

## SLIDE 5: System architecture and our pipeline (1:40–2:10)

**On slide (diagram, left to right):**

```
TU-format text file → load graphs (networkx) → node feature = one-hot(degree)
        → 10-fold stratified CV (seed 0)
        → GIN: 5 layers incl. input (4 message-passing layers)
             each = sum aggregation + (1+ε)·self → 2-layer MLP (hidden 64) → BatchNorm → ReLU
             (GIN-0: ε fixed at 0)
        → readout: sum-pool nodes at EVERY layer → linear → dropout 0.5 → class scores
        → train: Adam, lr 0.01, halved every 50 epochs · 350 epochs × 50 iterations · batch 32
```

- **Official code:** `main.py`, `util.py`, `models/graphcnn.py`. Runs unmodified on Python 3.11 + PyTorch 2.x (the README only claims 0.4.1/1.0)
- **Our addition, `replication/run_gin.py`:** runs all 10 folds, keeps full learning curves, and applies the paper's model-selection rule
  - **Rule:** average the 10 folds' test-accuracy curves → pick the **single best epoch** → report **mean ± std across folds at that epoch**
  - Needed because `main.py` trains one fold at a time and its `--filename` output **overwrites each epoch**
- **GraphSAGE baseline = same code, 2 flags:** `--neighbor_pooling_type max --num_mlp_layers 1` (the paper's own "Max–1-layer" approximation)

**Visual:** horizontal flow diagram with 5 boxes (Data → Features → CV split → GIN model → Model selection). Highlight "run_gin.py (ours)" in the accent colour.

**Speaker notes (≈80 words):**
> Here's the system. The official code reads graphs from a plain text file and uses each node's degree as its feature. It splits the data into ten stratified folds. The model has four GIN layers with sum aggregation and a two-layer MLP, then sums the nodes at every layer to classify the graph. It trains for 350 epochs. The official script runs one fold at a time, so I wrote a driver that runs all ten folds and applies the paper's selection rule: average the curves, pick one best epoch, report mean and standard deviation there.

---

## SLIDE 6: Running the original code, live demo (2:10–2:45)

**On slide:** a screen recording or screenshot of the terminal, with 3 callout labels.

**Command to run on camera** (short run, CPU is fine):
```
cd replication/official-repo
python main.py --dataset IMDBBINARY --degree_as_tag --fold_idx 0 --epochs 5
```

**What the output shows (callouts):**
1. **Data loading:** `loading data` → `# classes: 2` → `# maximum node tag: …` → `# data: 1000`
2. **Each epoch:** a progress bar `epoch: N` over 50 batches, then
   `loss training: …` and `accuracy train: … test: …`
3. **The ε values** printed at the end of each epoch: all zeros, because this is GIN-0 (ε is fixed, not learned)

- Then show the **full-run result file** our driver writes, `results/logs/IMDBBINARY_cv_curves.json`, `"summary"` block:
  `best_epoch_0indexed: 12`, `mean_test_acc_at_best_epoch: 0.739`, `std: 0.027`, `per_fold_acc_at_best_epoch: [0.77, 0.74, 0.73, 0.75, 0.71, 0.75, 0.72, 0.74, 0.69, 0.79]`

**Recording tip:** record this run before the presentation and cut it to about 15 s. The numbers in a 5-epoch run will differ from the full 350-epoch result; say so.

**Speaker notes (≈75 words):**
> Here's the original code running on IMDB-BINARY, a short five-epoch run on one fold. First it loads the data: two classes, a thousand graphs. Each epoch trains fifty batches, then prints the training loss and the train and test accuracy. The zeros at the end are epsilon, fixed at zero in GIN-0. The full run is 350 epochs over all ten folds, and my driver writes this summary: best epoch, mean accuracy 73.9 percent, and each fold's accuracy.

---

## SLIDE 7: Replication, environment and what went wrong (2:45–3:05)

**On slide (timeline, 3 stages):**

| Stage | Hardware | Outcome |
|---|---|---|
| 1. Local laptop | Windows, CPU, Python 3.11, PyTorch 2.x | ✅ Code runs unmodified · ❌ about 3.5 h per IMDB dataset at full spec: too slow |
| 2. Google Colab | NVIDIA T4 GPU | ✅ GIN on all 4 datasets (about 50–56 min each) · ❌ GPU quota ran out during GraphSAGE |
| 3. Uni GPU server `aai-lab` | RTX 6000 Ada (49 GB), SSH + tmux | ✅ Full batch: GIN + GraphSAGE × 4 datasets in about 4 h 17 min |

**Problems found and fixed:**
- 🐛 **Bug in the official code:** GraphSAGE (max-pooling) path left an index tensor on the CPU while the model was on the GPU, so it **stalled for 12+ min** with no output. **Fix: one line** (`.to(self.device)`), no change to the maths
- `scikit-learn` missing from the official `requirements.txt` → installed in our setup script
- GIN and GraphSAGE results overwrote each other's output file → added a `--tag` option
- Max-pooling is slow by design (a single-threaded Python loop): about 2× slower than GIN on T4, about 5× on the RTX 6000 Ada. Not a bug.

**Speaker notes (≈60 words):**
> Did the code work? Yes, unmodified, on modern PyTorch. But on my laptop's CPU one dataset took about three and a half hours, so I moved to a Colab GPU, then to my university's GPU server when Colab's quota ran out. Along the way I found and fixed a one-line GPU bug in the official GraphSAGE path.

---

## SLIDE 8: Replication results vs the paper (3:05–3:30)

**On slide (main table, 10-fold CV accuracy %, mean ± std):**

| Dataset | Paper GIN-0 | **Our GIN (Colab T4)** | Our GIN (uni server re-run) | Paper GraphSAGE | **Our GraphSAGE** |
|---|---|---|---|---|---|
| IMDB-BINARY | 75.1 ± 5.1 | **74.4 ± 4.1** | 73.9 ± 2.7 | 72.3 ± 5.3 | **72.6 ± 3.0** |
| IMDB-MULTI | 52.3 ± 2.8 | **50.5 ± 2.3** | 50.6 ± 2.7 | 50.9 ± 2.2 | **50.9 ± 2.8** |

**Takeaways (3 bullets):**
- ✅ **Replication succeeded:** both GIN results are within one paper standard deviation
- ✅ **Reproducible across hardware:** a second run on a different GPU landed within 0.5 points
- ✅ **GraphSAGE matches the paper almost exactly** (72.6 vs 72.3; 50.9 vs 50.9). GIN beats GraphSAGE on IMDB-BINARY (+1.8), as the paper claims; on IMDB-MULTI they tie within noise (the paper's own gap, +1.4, is also within one std)

**Small footnote:** "Colab numbers from the notebook summary; per-epoch curves committed in the repo are from the uni-server run."

**Visual:** grouped bar chart, two groups (IMDB-BINARY, IMDB-MULTI), bars = Paper GIN, Our GIN (Colab), Paper GraphSAGE, Our GraphSAGE, with error bars.

**Speaker notes (≈65 words):**
> And the results. On IMDB-BINARY I got 74.4 percent against the paper's 75.1; on IMDB-MULTI 50.5 against 52.3. Both are within one standard deviation of the paper. A second run on different hardware landed within half a point. GraphSAGE matched the paper almost exactly, and GIN still beats it on IMDB-BINARY. So the replication on the original data worked.

---

## SLIDE 9: New data, existing and constructed (3:30–3:55)

**On slide, two columns:**

**Left, existing datasets not yet tried with GIN** (status: **selected, not yet run**)
- Candidates from the proposal: extra TU benchmarks (**ENZYMES**, **DD**), OGB `ogbg-molhiv`, HuggingFace **AIDS**, Kaggle Tox21/QM9, OpenAlex co-authorship
- **First pick: ENZYMES / DD.** Same TU text format, so they drop straight into the pipeline with **no adapter**

**Right, newly constructed dataset: TMDb "IMDB-style" graphs** (status: ✅ **built and run**)
- **Source:** The Movie Database (TMDb) API, with a free key
- **Recipe mirrored:** the original Deep Graph Kernels construction (actors, co-starring edges, genre labels)
- **New data only:** films released **2016 onward** (the originals were built in 2015, so there is no overlap)
- **Output:** two datasets, **IMDBBINARY-NEW** and **IMDBMULTI-NEW**, in the exact input format of the official code, so **zero code changes** were needed

**Footer (required):** *"This product uses the TMDb API but is not endorsed or certified by TMDb."*

> ⚠ **NOTE FOR SOBHAN (do not put this on the slide):** The brief requires at least **one existing** and **one constructed** dataset. As of 2026-10-07, **no existing dataset has been run**. If you run ENZYMES or DD before recording, replace the left column's status with the real result. If not, present it honestly as "selected, pipeline-ready, run is next". Do not imply it has been run.

**Speaker notes (≈60 words):**
> Now the new data. For an existing dataset, I've chosen ENZYMES or DD from the same benchmark family. They use the same file format, so no adapter is needed. That run is next. My main effort went into building a new dataset from scratch: I used the TMDb movie API to rebuild the IMDB benchmarks with films from 2016 onward.

---

## SLIDE 10: Constructed dataset, properties and complexity (3:55–4:25)

**On slide (properties table, original vs new):**

| Property | IMDB-BINARY (orig.) | **IMDBBINARY-NEW** | IMDB-MULTI (orig.) | **IMDBMULTI-NEW** |
|---|---|---|---|---|
| Graphs | 1,000 | **996** | 1,500 | **1,492** |
| Classes | Action 500 / Romance 500 | **Action 500 / Romance 496** | 500 / 500 / 500 | **Comedy 499 / Romance 493 / Sci-Fi 500** |
| Avg nodes (min–max) | 19.8 (12–136) | **19.2 (5–20)** | 13.0 (7–89) | **18.9 (5–20)** |
| Avg edges | 96.5 | **177.8** | 65.9 | **174.1** |
| Release years | ≤ 2015 | **2016 onward** | ≤ 2015 | **2016 onward** |

**Challenges and complexity (bullets):**
- **Ambiguous labels:** many films have several genres → kept only films with **exactly one** target genre
- **API limits:** throttled to ≥ 0.15 s between calls, with automatic retry on HTTP 429 (rate limit)
- **Variable cast lists:** capped at the top **20** billed actors; dropped films with **fewer than 5** (this is why we have 996 / 1,492, not 1,000 / 1,500)
- **Matching the original format exactly** (TU text format, degree as the node feature) so the comparison is fair

**Visual:** the table, plus 2 or 3 small sample film names as chips: *Spider-Man: Brand New Day* (Action), *Toy Story 5* (Comedy), *Moana* (Comedy), *Reminiscence* (Sci-Fi).

**Speaker notes (≈75 words):**
> Here are the properties. The new binary set has 996 graphs and the multi-class set 1,492, balanced across genres and close to the originals' sizes. Average node counts are similar, but notice that the new graphs have nearly twice as many edges. The main challenges were ambiguous multi-genre films, which I filtered out; API rate limits; and short cast lists, which explain the few missing graphs. I'll come back to that edge count.

---

## SLIDE 11: Construction progress, and a problem we found (4:25–4:50)

**On slide, top: pipeline strip (4 steps, all ✅)**
1. **Collection:** TMDb `/discover/movie` per genre (2016+, ≥ 20 votes, sorted by popularity, no adult titles) → `/movie/{id}/credits` per film. **500 candidate films per genre found**
2. **Preprocessing:** single-genre filter · top-20 cast · drop films with fewer than 5 actors · duplicate check (0 duplicates) · build edges · write TU format plus a metadata JSON (titles and actor names)
3. **Annotation:** no manual labelling. The label is **TMDb's curated genre**, kept unambiguous by the single-genre filter
4. **Run:** identical GIN and GraphSAGE pipeline, unmodified

**On slide, bottom: results and diagnosis (amber box)**

| Dataset | Our GIN on original | **GIN on new** | **GraphSAGE on new** | **Node-count-only ceiling** |
|---|---|---|---|---|
| Binary | 74.4 | **56.1** (−18.3) | **56.2** | **56.5** |
| Multi | 50.5 | **40.9** (−9.6) | **41.0** | **41.0** |

- ⚠ **Quality check found a flaw in our construction: 100% of the 2,488 new graphs are complete graphs (cliques).** (Originals: 13.9% and 52.6%)
- **Why:** each graph's nodes are *one film's cast*, and they all co-starred in that film, so every pair is already connected. The cross-film edges add nothing.
- **Effect:** the only information left is **cast size**. A classifier that uses only node count reaches 56.5% / 41.0%, which is **exactly** what GIN and GraphSAGE score
- **So the drop is caused by our data, not by GIN.** It also explains why GIN and GraphSAGE **tie**: with no structure to tell apart, extra expressive power cannot help

**Small footnote:** "Node-count-only ceiling = best accuracy achievable by predicting the majority label for each cast size (an in-sample upper bound). GIN/GraphSAGE on new data: Colab GIN run; GraphSAGE from the uni-server run."

**Visual:** two small graph drawings side by side: "original: ego-network with varied structure" vs "ours: K_20, every actor linked to every other".

**Speaker notes (≈90 words):**
> Progress: collection, cleaning and labelling are all done. Labels come from TMDb's own genre data. Running GIN on the new data, accuracy dropped sharply, to 56 and 41 percent. Rather than stop there, I checked my graphs, and found every single one is a complete graph. Each graph is one film's cast, and they all co-starred in that film, so everyone is connected to everyone. The only signal left is cast size, and a classifier using only node count scores exactly the same. So the drop comes from my construction, not from GIN.

---

## SLIDE 12: Wrap-up and next steps (4:50–5:05)

**On slide, two columns:**

**Done ✅**
- Official code running on 3 environments (CPU → Colab T4 → RTX 6000 Ada)
- GIN replicated within 1 std of the paper on IMDB-BINARY and IMDB-MULTI; reproduced on a second GPU
- GraphSAGE baseline matches the paper; a GPU bug in the official code fixed
- Two new TMDb datasets built (996 + 1,492 graphs) and run; a construction flaw diagnosed

**Next ➡**
1. **Rebuild the TMDb data as true actor ego-networks** (one graph per actor: their co-stars across films, linked if those co-stars worked together). Check the clique fraction and the node-count baseline **before** training, then re-run GIN and GraphSAGE
2. **Run an existing dataset** (ENZYMES or DD) through the same pipeline
3. **Investigation:** relate misclassifications to graph properties; a small toy test of graphs that mean/max can't separate but GIN can; optional GCN variant (`--neighbor_pooling_type average --num_mlp_layers 1`)
4. Final report and repo clean-up (README with the TMDb attribution)

**Footer:** `github.com/SobhanSalarianCognitivo/Gin-Graph` · *This product uses the TMDb API but is not endorsed or certified by TMDb.*

**Speaker notes (≈40 words):**
> To wrap up: the replication worked and matches the paper, and the new data is built. Next, I'll rebuild it as real ego-networks, run an existing benchmark like ENZYMES, and dig into where GIN succeeds and fails. Thank you.

---

## Appendix A: Fact sheet (for checking the generated slides)

**Paper and code**
- Xu, Hu, Leskovec, Jegelka. *How Powerful are Graph Neural Networks?* ICLR 2019 (CORE A\*). arXiv:1810.00826
- Code: `github.com/weihua916/powerful-gnns`. Our repo: `github.com/SobhanSalarianCognitivo/Gin-Graph` (public)
- IMDB datasets originally from Yanardag & Vishwanathan, *Deep Graph Kernels*, KDD 2015

**Hyperparameters (official defaults = paper):** 5 layers incl. input, 2-layer MLPs, hidden 64, final dropout 0.5, batch 32, 50 iterations per epoch, 350 epochs, Adam lr 0.01, StepLR ×0.5 every 50 epochs, sum graph pooling, sum neighbour pooling, ε fixed at 0 (GIN-0), `--degree_as_tag`, 10-fold StratifiedKFold with seed 0.

**All results (10-fold CV, %, mean ± std at the selected epoch; best epoch is 1-indexed)**

| Model | Dataset | Paper | Colab T4 | Uni server (raw JSON in repo) |
|---|---|---|---|---|
| GIN-0 | IMDB-BINARY | 75.1 ± 5.1 | 74.4 ± 4.1 (ep 80, ~50 min) | 73.9 ± 2.7 (ep 13, 17.8 min) |
| GIN-0 | IMDB-MULTI | 52.3 ± 2.8 | 50.5 ± 2.3 (ep 213, ~53 min) | 50.6 ± 2.7 (ep 230, 18.3 min) |
| GIN-0 | IMDBBINARY-NEW | — | 56.1 ± 2.2 (ep 136, ~53 min) | 55.9 ± 2.3 (ep 193, 18.2 min) |
| GIN-0 | IMDBMULTI-NEW | — | 40.9 ± 2.1 (ep 265, ~56 min) | 41.0 ± 2.3 (ep 265, 19.8 min) |
| GraphSAGE | IMDB-BINARY | 72.3 ± 5.3 | (quota hit) | 72.6 ± 3.0 (ep 14, 93.8 min) |
| GraphSAGE | IMDB-MULTI | 50.9 ± 2.2 | (quota hit) | 50.9 ± 2.8 (ep 271, 53.6 min) |
| GraphSAGE | IMDBBINARY-NEW | — | (quota hit) | 56.2 ± 2.3 (ep 181, 16.9 min) |
| GraphSAGE | IMDBMULTI-NEW | — | (quota hit) | 41.0 ± 2.2 (ep 273, 18.3 min) |

**New-data diagnostics (recomputed from the dataset files)**

| | IMDB-BINARY | BINARY-NEW | IMDB-MULTI | MULTI-NEW |
|---|---|---|---|---|
| Avg edge density | 0.521 | 1.000 | 0.772 | 1.000 |
| Complete graphs | 13.9% | 100% | 52.6% | 100% |
| Distinct degree sequences | 537 | 17 | 387 | 18 |
| Graphs with exactly 20 nodes | 3.6% | 87.3% | 1.9% | 82.9% |
| Node-count-only ceiling | 60.6% | 56.5% | 44.1% | 41.0% |
| Majority class | 50.0% | 50.2% | 33.3% | 33.5% |

- Films with fewer than 20 billed actors: binary, Romance 94 of 496 vs Action 32 of 500; multi, Romance 138, Sci-Fi 82, Comedy 35. This cast-size difference is the only signal the models can use.
- Missing graphs: binary −4 (all Romance); multi −8 (7 Romance, 1 Comedy). All were films with fewer than 5 billed actors.
- Full server batch: 2026-08-26, 04:28–08:45 UTC (≈ 4 h 17 min).

**Why GraphSAGE was much faster on the new data (17–18 min vs 54–94 min):** the max-pool path pads each node's neighbour list to the largest degree in the batch. The originals have graphs with up to 136 nodes; every new graph has at most 20. This is an inference from the code and has not been profiled. Mention it only if asked.

## Appendix B: What NOT to claim

- ❌ That an existing dataset (ENZYMES/DD/OGB/AIDS) has been run. It has not (as of 2026-10-07).
- ❌ That GIN "fails to generalise" on new data. The new graphs are all cliques, so the test is uninformative.
- ❌ That the new graphs are ego-networks. They are one-film cliques; the ego-network rebuild is the next step.
- ❌ Misclassification analysis, the WL toy experiment, GCN, OpenAlex, or LLM-as-judge results. None have been started.
- ❌ A specific citation count, unless you have just checked it yourself.
- ❌ That the Colab per-epoch curves are in the repo. Only the uni-server JSONs are committed.

## Appendix C: Recording checklist

- [ ] Record the slide 6 demo terminal run beforehand (5 epochs, fold 0, IMDBBINARY)
- [ ] Face-cam in the bottom-right corner for the whole video
- [ ] Final runtime between 4:30 and 5:30 (script ≈ 5:05 at a normal pace; cut slide 10's challenge bullets if you run long)
- [ ] TMDb attribution visible (slides 9 and 12)
- [ ] Citation count filled in or removed on slide 2
- [ ] Slide 9 left column updated if ENZYMES/DD has been run
- [ ] Export as MP4 and upload to the iLearn dropbox before 11:55 pm, 9 Oct 2026
