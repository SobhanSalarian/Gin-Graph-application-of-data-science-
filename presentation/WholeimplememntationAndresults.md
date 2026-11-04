# Whole Implementation and Results

What we proposed (`proposal/proposal.pdf`), what we actually built and ran, and what came out.
Built from the full repo: `plan.md`, git history, `replication/`, `new_data/`, `notebooks/`, `results/`, `docs/`. **Every number below was re-checked against the raw result files** (`results/logs/*.json`, dataset `.txt` files) on 2026-10-07, unless it is marked as coming only from `plan.md`.

---

## 0. TL;DR

| | Outcome |
|---|---|
| **Paper** | Xu, Hu, Leskovec & Jegelka, *How Powerful are Graph Neural Networks?*, ICLR 2019 (CORE A*). Code: `github.com/weihua916/powerful-gnns` |
| **Track** | Novel project |
| **Replication (original data)** | ✅ **Success.** IMDB-BINARY 74.4% (paper 75.1%), IMDB-MULTI 50.5% (paper 52.3%). Both are within one paper std. Independently re-run on a second GPU and landed within ~0.5 pt. |
| **Baseline (GraphSAGE)** | ✅ Reproduced the paper's GraphSAGE numbers almost exactly (72.6% vs 72.3%; 50.9% vs 50.9%). GIN > GraphSAGE on IMDB-BINARY, as the paper claims. |
| **New constructed data (TMDb)** | ✅ Built and run: **IMDBBINARY-NEW** (996 graphs) and **IMDBMULTI-NEW** (1,492 graphs), movies from 2016 onward, in the exact input format the official code reads. |
| **Result on new data** | Accuracy drops sharply: **56.1%** (binary) and **40.9%** (multi). GIN and GraphSAGE tie. |
| **⚠ Key finding (new, 2026-10-07)** | **Every graph in both new datasets is a complete graph (a clique)**: 100% of 2,488 graphs. The only information left for the model is the **number of nodes** (cast size). A classifier that uses only node count tops out at **56.5% / 41.0%**, which is exactly what GIN and GraphSAGE score. The drop comes from **a flaw in our graph construction**, not a failure of GIN's claim. |
| **Not done yet** | Existing-but-untried dataset (ENZYMES/DD/OGB/AIDS): **none run**. Misclassification analysis, the WL/isomorphism toy experiment, GCN variant, OpenAlex dataset and LLM-as-judge are all not started. |

---

## 1. Proposal vs. reality

| # | Proposed (proposal.pdf) | What actually happened | Status |
|---|---|---|---|
| 1 | Feasibility check: run the official code on MUTAG locally | Official repo runs unmodified on Python 3.11 / PyTorch 2.x CPU. MUTAG smoke test passed. (A full-spec MUTAG CV run was started, then paused and dropped when scope narrowed.) | ✅ |
| 2 | Confirm the model-selection rule from the official code | Confirmed from the official README. Implemented exactly in our `replication/run_gin.py` (§3.3) | ✅ |
| 3 | Replicate the paper on IMDB-BINARY and IMDB-MULTI | Done at full spec (350 epochs × 10 folds). Very close to the paper (§4) | ✅ |
| 4 | Compute: Colab GPU primary, uni GPU server fallback | Exactly that happened. Colab T4 for GIN; Colab's quota ran out mid-GraphSAGE, so we moved to the uni `aai-lab` RTX 6000 Ada (§2) | ✅ |
| 5 | New *constructed* dataset from the TMDb API, mirroring IMDB-BINARY/MULTI, about 1,000 / 1,500 graphs, films from 2016 on, single-target-genre filter, top-billed cast as nodes, cross-film collaboration edges → "ego-network, not one clique per film" | Scraper built and both datasets built at target scale. **However, the "not one clique per film" goal was not achieved: every graph is a clique** (§5.4) | ⚠ Built, but structurally degenerate |
| 6 | Run the identical, unmodified pipeline on the new data | Done, zero changes to model/training code (§6) | ✅ |
| 7 | (Time permitting) GraphSAGE-style baseline via the paper's own max-pooling, 1-layer MLP variant | Done on all 4 datasets. Along the way we found and fixed a GPU bug in the official code (§7) | ✅ |
| 8 | At least one *existing* dataset not yet tried with GIN (5 candidates: extra TU sets such as ENZYMES/DD, OGB ogbg-molhiv, HF AIDS, Kaggle Tox21/QM9, OpenAlex co-authorship) | **None run yet** | ❌ |
| 9 | Qualitative investigation of where/why the method fails, including misclassified examples | Not started. (The clique finding in §5.4 is the first real piece of this.) | ❌ |
| 10 | LLM-as-judge sanity check on label quality (tentative) | Not started | ❌ |
| 11 | Second new dataset via OpenAlex (stretch) | Not started | ❌ |

---

## 2. Environment and compute timeline

| Stage | Where | Spec | What ran |
|---|---|---|---|
| Phase A: feasibility (2026-08-24) | Local Windows laptop, CPU | Python 3.11.9, PyTorch 2.x CPU, numpy, networkx, tqdm, scikit-learn | Smoke tests; timing showed about 3.5 h per IMDB dataset at full spec, too slow for 4+ datasets |
| Phase B/D: GIN (2026-08-25) | **Google Colab, T4 GPU** | `notebooks/gin_colab_replication.ipynb` | GIN on all 4 datasets, about 50 to 56 min each |
| GraphSAGE attempt (2026-08-26) | Colab T4 | `notebooks/graphsage_colab_comparison.ipynb` | Hit the GPU stall bug (§7.2), fixed it, then **hit Colab's usage quota** |
| Full batch (2026-08-26, 04:28 to 08:45 UTC, ≈4 h 17 m) | **Uni server `aai-lab`, RTX 6000 Ada (49 GB)**, via SSH + tmux | `replication/setup_server.sh`, `replication/run_all_server.sh` | GIN **and** GraphSAGE × 4 datasets, all full spec |

Supporting infrastructure built:
- The repo was pushed to GitHub (`SobhanSalarianCognitivo/Gin-Graph`, public) so the server could `git clone` it.
- SSH key auth, a detached `tmux` session (`gnn`), and periodic progress snapshots (`results/server_progress.log`).
- `docs/uni-gpu-server-playbook.md` is a reusable how-to for the server.

The official code auto-detects CUDA, so **no code changes were needed to move CPU → GPU** (apart from the bug fix in §7.2, which only affects the max-pooling path).

---

## 3. Replication: implementation

### 3.1 System architecture (for the "explain the system" slide)

```
TU-format text file ──► util.load_data()            one networkx graph per sample;
(one graph per block)    + --degree_as_tag           node feature = one-hot(node degree)
                              │
                              ▼
            StratifiedKFold (10 folds, seed 0) ── fold_idx selects train/test split
                              │
                              ▼
   GraphCNN (models/graphcnn.py)
     5 layers (incl. input) × [ neighbour aggregation  ─►  MLP (2 layers, hidden 64) ─► BatchNorm ─► ReLU ]
        aggregation = SUM over neighbours + (1+ε)·self   (GIN-0: ε fixed at 0)
     readout: SUM-pool node embeddings at EVERY layer ─► linear ─► dropout 0.5 ─► class scores
                              │
                              ▼
   Train: Adam lr 0.01, StepLR ×0.5 every 50 epochs, 350 epochs × 50 iters, batch 32
   Output per epoch: training loss, train accuracy, test accuracy
```

- **GIN-0** = default flags (`--neighbor_pooling_type sum`, `--num_mlp_layers 2`, no `--learn_eps`).
- **GraphSAGE (paper's approximation, "Max–1-layer")** = the same code with `--neighbor_pooling_type max --num_mlp_layers 1`.
- The IMDB datasets have no node attributes, so `--degree_as_tag` is used throughout, as in the paper.

### 3.2 Our driver: `replication/run_gin.py`
The official `main.py` trains a single fold, and its `--filename` output **overwrites every epoch**, so it cannot hold a learning curve. Our driver:
1. Runs `main.py` for folds 0 to 9 as subprocesses, parses every epoch's `loss / train acc / test acc` from stdout, and checks that exactly 350 epochs were parsed.
2. Writes the curves incrementally to `results/logs/<dataset>[_tag]_cv_curves.json`, so partial progress survives interruptions.
3. Applies the paper's model-selection rule (below).
4. Takes `--tag` so GIN and GraphSAGE runs on the same dataset don't overwrite each other (this collision really happened, and was fixed in commit `ee6c3a3`).

### 3.3 Model-selection rule (verified, as promised in the proposal)
Average the 10 folds' test-accuracy curves epoch by epoch, pick the **single epoch** with the highest average, and report **mean ± std across folds at that epoch**. This avoids the optimistic bias of taking each fold's own best epoch.

### 3.4 Scope decision
Replication was **deliberately narrowed to IMDB-BINARY and IMDB-MULTI**. These are the only two datasets with matching new data, which gives a clean chain: paper → our replication → same pipeline on new data. The other 7 benchmarks were dropped on purpose, not because of a resource shortfall. (The rubric allows a subset as long as it is justified.)

---

## 4. Replication: results (original data, GIN-0)

| Dataset | #Graphs | #Classes | Avg nodes | **Paper GIN-0** | **Ours: Colab T4** (logged result) | Ours: uni server re-run |
|---|---|---|---|---|---|---|
| IMDB-BINARY | 1000 | 2 | 19.8 | 75.1 ± 5.1 | **74.4 ± 4.1** (epoch 80, ~50 min) | 73.9 ± 2.7 (epoch 13, 17.8 min) |
| IMDB-MULTI | 1500 | 3 | 13.0 | 52.3 ± 2.8 | **50.5 ± 2.3** (epoch 213, ~53 min) | 50.6 ± 2.7 (epoch 230, 18.3 min) |

- **Both are within one paper standard deviation.** Same code, same protocol, same hyperparameters: the replication succeeded.
- **Cross-hardware reproducibility:** the server re-run lands within 0.5 pt of Colab on both datasets.
- **Provenance caveat:** the Colab per-epoch JSONs were never committed. The Colab numbers survive only in `plan.md`, copied from the notebook's summary cell. The **raw curves in `results/logs/` are the server re-run.** If you need fully traceable numbers on a slide, quote the server run or show both.
- Best-epoch numbers are 1-indexed. On IMDB-BINARY the averaged curve is very flat, so the best epoch jumps around between runs (80 vs 13) while the accuracy barely moves.

Per-fold accuracy at the best epoch (server GIN): IMDB-BINARY 77, 74, 73, 75, 71, 75, 72, 74, 69, 79; IMDB-MULTI 52.0, 49.3, 54.0, 47.3, 48.7, 53.3, 46.7, 54.7, 48.7, 51.3.

---

## 5. New data

### 5.1 Existing datasets not yet tried with GIN: ❌ not done
The proposal listed five candidates (extra TU benchmarks such as ENZYMES/DD; OGB `ogbg-molhiv`; HuggingFace AIDS; Kaggle Tox21/QM9; OpenAlex co-authorship). **None has been run.** The update brief explicitly asks for "at least one existing + one you're constructing", so this is a visible gap. ENZYMES or DD is the cheapest fix: both are TU-format and need no adapter.

### 5.2 Newly constructed dataset: TMDb "IMDB-style" graphs
**Original recipe being mirrored:** Yanardag & Vishwanathan, *Deep Graph Kernels* (KDD 2015). Nodes are actors, an edge means two actors co-starred, the graph is an actor's collaboration **ego-network**, and the label is the genre. IMDB-BINARY is Action vs Romance; IMDB-MULTI is Comedy vs Romance vs Sci-Fi.

**Our pipeline:** `new_data/scraping/build_tmdb_dataset.py`

| Step | Implementation |
|---|---|
| **Collection** | TMDb API (free key, passed via the `TMDB_API_KEY` env var and never committed). `/discover/movie` per target genre, `primary_release_date ≥ 2016-01-01` (no overlap with the 2015 originals), `vote_count ≥ 20`, sorted by popularity, no adult titles; then `/movie/{id}/credits` per film. Throttled to ≥0.15 s between calls, with a 429 retry. |
| **Cleaning / filtering** | Keep only films tagged with **exactly one** of the target genres (no ambiguous labels). Take the top-billed cast in billing order, **capped at 20**. Drop films with fewer than 5 billed actors. Duplicate movie IDs are skipped (verified: 0 duplicates). |
| **Graph construction** | One graph per film, with that film's cast as the nodes. An edge connects two cast members if they co-appeared in *any* sampled film of that genre. |
| **Annotation (labels)** | No manual annotation: the label is TMDb's own curated genre metadata. A single-target-genre filter keeps the label unambiguous. |
| **Output** | TU-format text, exactly the format `util.load_data()` expects (tag 0, use `--degree_as_tag`), plus `*_metadata.json` with film title, label and actor names per graph, for EDA. Committed to `new_data/processed/`. |
| **Provenance** | Scrape logs in `results/logs_binary_scrape.log` and `results/logs_multi_scrape.log`. TMDb attribution: *"This product uses the TMDb API but is not endorsed or certified by TMDb."* |

### 5.3 Dataset properties (recomputed from the files)

| | IMDB-BINARY (orig.) | **IMDBBINARY-NEW** | IMDB-MULTI (orig.) | **IMDBMULTI-NEW** |
|---|---|---|---|---|
| Graphs | 1000 | **996** | 1500 | **1492** |
| Class counts | 500 / 500 | Action 500 / Romance 496 | 500 / 500 / 500 | Comedy 499 / Romance 493 / Sci-Fi 500 |
| Avg nodes (min to max) | 19.8 (12 to 136) | **19.2 (5 to 20)** | 13.0 (7 to 89) | **18.9 (5 to 20)** |
| Avg edges | 96.5 | **177.8** | 65.9 | **174.1** |
| Avg edge density | 0.521 | **1.000** | 0.772 | **1.000** |
| Complete graphs (cliques) | 13.9% | **100%** | 52.6% | **100%** |
| Distinct degree sequences | 537 | **17** | 387 | **18** |
| Graphs with exactly 20 nodes | n/a | 87.3% | n/a | 82.9% |
| Release years | ≤2015 | 2016 onward | ≤2015 | 2016 onward |

Sample graphs: *Spider-Man: Brand New Day* (Action), *Toy Story 5* (Comedy), *Moana* (Comedy), *Reminiscence* (Sci-Fi).

All 500 candidate films per genre were found. The shortfall comes from films with fewer than 5 billed actors being dropped: binary is missing 4 (all Romance); multi is missing 8 (7 Romance, 1 Comedy).

### 5.4 ⚠ Critical finding: every new graph is a clique

`plan.md` describes the new graphs as "near-clique". The actual numbers are stronger than that: **all 2,488 graphs are complete graphs.**

**Why:** the node set of each graph is *that one film's cast*. Every pair of cast members already co-starred in that film, so the "co-appeared in *any* sampled film" rule can only add edges that already exist. The cross-film collaboration edges never change anything. Each graph is therefore K_n, with n = cast size, where 20 is the cap.

**Consequence:** two cliques of the same size are isomorphic. The **only** feature that separates one graph from another is n. With `--degree_as_tag`, every node in K_n has degree n−1, so even the node features carry nothing beyond n.

**Evidence that the models learned only cast size:** a lookup table from node count to majority label, evaluated in-sample (an *upper bound* for any n-only classifier):

| Dataset | n-only ceiling | GIN | GraphSAGE | Majority class |
|---|---|---|---|---|
| IMDBBINARY-NEW | **56.5%** | 56.1% | 56.2% | 50.2% |
| IMDBMULTI-NEW | **41.0%** | 40.9% | 41.0% | 33.5% |

Both models sit right on that ceiling. The small signal comes from cast-size differences between genres: Romance films more often have fewer than 20 billed actors (94 of 496 Romance graphs vs 32 of 500 Action graphs have n < 20).

**What this means for the project's claim:**
- The drop on new data **does not show that GIN generalises poorly**. These graphs carry no topology for *any* graph classifier to use.
- It also explains why **GIN and GraphSAGE tie** on new data. With no structure to distinguish, extra expressive power cannot help. This is consistent with the paper's theory, but the test is uninformative.
- **Fix (next step):** follow the original recipe properly. Build **one graph per actor** (an ego-network): nodes are the actor plus everyone they co-starred with across the sampled films, and an edge connects two of those co-stars if they appeared in a film together. Label it by the genre of the films involved. This produces non-trivial structure, as in the originals (where only 13.9% / 52.6% of graphs are cliques). Alternatively, merge several films that share actors into one graph. Either way, check the clique fraction and n-only baseline before training.

This is also a strong presentation point. It shows we inspected our own data (preprocessing/QA) and dug into *why* results changed, which is what the rubric calls "digging into the original results".

---

## 6. GIN on new data: results

| Dataset | Ours: Colab GIN (logged) | Ours: server GIN re-run | Our original-data replication | Δ vs original |
|---|---|---|---|---|
| IMDBBINARY-NEW | **56.1 ± 2.2** (epoch 136, ~53 min) | 55.9 ± 2.3 (epoch 193, 18.2 min) | 74.4 | **−18.3 pt** |
| IMDBMULTI-NEW | **40.9 ± 2.1** (epoch 265, ~56 min) | 41.0 ± 2.3 (epoch 265, 19.8 min) | 50.5 | **−9.6 pt** |

Code, hyperparameters and protocol are identical between each original/new pair, so the gap isolates to the data. §5.4 shows *what* in the data: the new graphs are all cliques, leaving only node count as a signal.

---

## 7. GraphSAGE baseline

### 7.1 Setup
The paper itself approximates GraphSAGE inside its own architecture ("Max–1-layer"). We used the identical codebase with `--neighbor_pooling_type max --num_mlp_layers 1`: no new code, no new dependencies, and the same protocol as the paper's baseline.

### 7.2 Bug found and fixed in the official code
- **Symptom:** on GPU, the first GraphSAGE fold produced no output for 12+ minutes.
- **Root cause:** `models/graphcnn.py`, `__preprocess_neighbors_maxpool` returns `torch.LongTensor(padded_neighbor_list)` on the CPU, while `h` is on the GPU. Every iteration indexes a CUDA tensor with a CPU tensor. The sum/average path a few lines below already calls `.to(self.device)` correctly.
- **Fix (one line, no change to the computation):** `return torch.LongTensor(padded_neighbor_list).to(self.device)`. It is applied automatically by a patch cell in the Colab notebook (cell 1b) and by `setup_server.sh`. The patch is idempotent and fails loudly if upstream changes.

### 7.3 Performance characteristic (not a bug)
Even after the fix, max-pooling is much slower than GIN. Its neighbour list is built by a single-threaded Python loop every iteration, while GIN uses a GPU sparse matmul. It was about 2× slower on T4 and about 5× slower on the RTX 6000 Ada: the GPU part shrinks on faster hardware, the CPU loop doesn't. Confirmed via `nvidia-smi`/`top` (one CPU core pinned).

**Speed anomaly, likely explained:** GraphSAGE ran in about 17 to 18 min on the new datasets vs 54 to 94 min on the originals. The max-pool path pads every node's neighbour list to the **maximum degree in the batch**. The originals contain graphs with up to 136 nodes (max degree up to 135), while every new graph has at most 20 nodes (max degree 19). That makes the padded tensor and the loop far smaller. (This is an inference from the code and the stats in §5.3; it has not been profiled.)

### 7.4 Results (uni server, full spec)

| Dataset | Paper GraphSAGE | **Ours GraphSAGE** | Ours GIN (Colab / server) | GIN − GraphSAGE |
|---|---|---|---|---|
| IMDB-BINARY | 72.3 ± 5.3 | **72.6 ± 3.0** (epoch 14, 93.8 min) | 74.4 / 73.9 | +1.8 / +1.3 |
| IMDB-MULTI | 50.9 ± 2.2 | **50.9 ± 2.8** (epoch 271, 53.6 min) | 50.5 / 50.6 | −0.4 / −0.3 (tie) |
| IMDBBINARY-NEW | n/a | **56.2 ± 2.3** (epoch 181, 16.9 min) | 56.1 / 55.9 | ≈ 0 |
| IMDBMULTI-NEW | n/a | **41.0 ± 2.2** (epoch 273, 18.3 min) | 40.9 / 41.0 | ≈ 0 |

- **Original data:** our GraphSAGE matches the paper almost exactly. GIN beats GraphSAGE on IMDB-BINARY, consistent with the paper (paper gap +2.8). On IMDB-MULTI it is a tie within noise; the paper reports +1.4, also well inside one std.
- **New data:** an exact tie. Per-fold accuracies are nearly identical, fold by fold. This is expected given §5.4: on cliques both models can only read n.

---

## 8. Engineering issues hit and resolved

| Issue | Where | Resolution |
|---|---|---|
| `main.py --filename` overwrites each epoch, so it cannot hold a curve | Official code | Our own driver parses stdout per fold (`run_gin.py`) |
| GPU stall in the max-pooling path (CPU index tensor) | `models/graphcnn.py` | One-line `.to(self.device)` patch (§7.2) |
| GIN and GraphSAGE outputs overwrote each other | Our driver | `--tag` suffix on the output file |
| `scikit-learn` missing from the official `requirements.txt` (used by `util.py`) | Clean server venv | Installed explicitly in `setup_server.sh` |
| The 9 benchmark datasets ship inside `dataset.zip` | Official repo | Unzipped in `setup_server.sh` and the notebooks |
| CPU too slow (~3.5 h per dataset) | Local | Moved to Colab T4 |
| Colab GPU quota exhausted | Colab | Moved to the uni RTX 6000 Ada via SSH + tmux |
| Code only claims support for PyTorch 0.4.1/1.0 | Official README | Runs on PyTorch 2.x with only deprecation warnings |

---

## 9. Repository map

```
Assign1 temp4/
├── plan.md                          living project plan / decisions log
├── instructions .md                 unit's overall goal & assessment criteria
├── proposal/                        proposal.tex/.pdf, figures, bib, proposal rubric notes
├── presentation/
│   ├── Assessment-Project-Update (3).pdf
│   ├── requirement.md               requirements for this update video
│   └── WholeimplememntationAndresults.md   (this file)
├── replication/
│   ├── run_gin.py                   10-fold CV driver + paper's model-selection rule
│   ├── setup_server.sh              one-time uni-server env setup (+ GPU bug patch)
│   ├── run_all_server.sh            GIN + GraphSAGE × 4 datasets, full spec
│   └── official-repo/               (gitignored) clone of weihua916/powerful-gnns
├── new_data/
│   ├── scraping/build_tmdb_dataset.py   TMDb → TU-format graph datasets
│   └── processed/IMDB{BINARY,MULTI}-NEW/   *.txt (TU format) + *_metadata.json
├── notebooks/
│   ├── gin_colab_replication.ipynb
│   └── graphsage_colab_comparison.ipynb
├── results/
│   ├── logs/*_cv_curves.json        per-epoch curves for 10 folds + summary (server run)
│   ├── logs_{binary,multi}_scrape.log
│   └── server_progress.log
├── docs/uni-gpu-server-playbook.md
└── report/                          (empty, final report not started)
```

### How to reproduce
```bash
# new data (needs a TMDb key)
TMDB_API_KEY=... python new_data/scraping/build_tmdb_dataset.py --mode binary
TMDB_API_KEY=... python new_data/scraping/build_tmdb_dataset.py --mode multi
# GPU box
bash replication/setup_server.sh
bash replication/run_all_server.sh        # inside tmux
# single run
python replication/run_gin.py --dataset IMDBBINARY --epochs 350 --extra --degree_as_tag
python replication/run_gin.py --dataset IMDBBINARY --epochs 350 --tag graphsage \
       --extra --degree_as_tag --neighbor_pooling_type max --num_mlp_layers 1
```

---

## 10. Gaps against the update-presentation requirements

Cross-checked against `presentation/requirement.md`:

| Requirement | Status |
|---|---|
| Recap: venue, authors, code, justification | ✅ material exists (proposal) |
| Original datasets explained | ✅ |
| High-level architecture + **run the code live and explain the output** | ⚠ needs a short screen recording: e.g. a 5 to 10 epoch `main.py` run on IMDBBINARY, showing per-epoch loss and train/test accuracy |
| Replication status (environment, results vs paper, what did/didn't work) | ✅ (§2, §4, §7, §8) |
| **≥1 existing new dataset** | ❌ **not done**: quickest is ENZYMES or DD (TU format, no adapter needed) |
| ≥1 newly constructed dataset: source, properties, complexity, challenges | ✅ (§5.2 to §5.3) |
| Construction progress: collection, preprocessing, annotation | ✅ collection and preprocessing done; annotation = TMDb genre labels. ⚠ Be upfront about the clique flaw and the planned ego-network rebuild |
| What has *not* worked | ✅ the clique finding, the GPU bug, the Colab quota |
| Next steps | see §11 |

---

## 11. Next steps (priority order)

1. **Rebuild the TMDb datasets as true actor ego-networks** (§5.4). Check the clique fraction and n-only baseline first, then re-run GIN and GraphSAGE. This is what turns the new-data test into a meaningful one.
2. **Run one existing, untried dataset** (ENZYMES or DD) through the same pipeline.
3. **Investigation:** relate misclassifications to graph properties; the toy WL experiment (a pair of non-isomorphic graphs that mean/max pooling cannot separate but GIN can); optionally the GCN variant (`--neighbor_pooling_type average --num_mlp_layers 1`).
4. Keep raw Colab outputs, or standardise on the server numbers, so every reported figure traces back to a committed file.
5. Final report (`report/`), README with the TMDb attribution, and repo clean-up.
