# Project Plan — Replicating GIN (Graph Isomorphism Network)

Last updated: 2026-08-25

---

## 0. Project summary

- **Paper:** "How Powerful are Graph Neural Networks?" — Xu, Hu, Leskovec & Jegelka, ICLR 2019 (CORE A*). [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
- **Official code:** https://github.com/weihua916/powerful-gnns
- **Track:** Novel project (self-chosen paper, needs feasibility justification in the proposal)
- **Why this paper:** theoretically grounded (proves GIN matches the Weisfeiler-Lehman test, the expressiveness ceiling for message-passing GNNs), ICLR is CORE A*, task shape (graph classification, clean train/test split) is simpler to redo on new data than transductive node classification (e.g. GCN), and 3 of its benchmark datasets (IMDB, COLLAB, REDDIT) are built from scrapeable public metadata — giving an unusually clean "new data" story. Full write-up goes in `proposal/justification.md`.

## 1. Assignment requirements (recap, from `instructions .md`)

- 3 deliverables: proposal (20%), update presentation (40%), final report (40%)
- Novel project = no 80% cap, but must justify paper/code/data quality (venue rank, citations)
- Must show: (a) unit techniques/tools used, (b) real replication attempt on original data, (c) new data created + code that runs, (d) polished writeup
- GitHub repo is part of the submission (report, new data, generation scripts)

## 2. Setup

- [x] Local tooling confirmed: Python 3.11.9, pip 24.0, git 2.53.0, internet access to GitHub
- [x] Project folder structure created: `replication/`, `new_data/`, `report/`, `proposal/`, `presentation/`, `results/`, `notebooks/`
- [x] Local git repo initialized (not yet pushed to a remote)
- [x] Official GIN repo cloned into `replication/official-repo/`
- [x] Dependencies installed and verified: CPU PyTorch 2.13.0 + `requirements.txt` (tqdm, numpy, networkx). Code runs unmodified despite README only claiming to test PyTorch 0.4.1/1.0.0 — only 2 harmless deprecation warnings, no functional impact
- [x] Datasets confirmed available: all 9 benchmarks (MUTAG, PTC, PROTEINS, NCI1, COLLAB, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI5K) ship inside the repo's `dataset.zip` — no external download needed
- [x] Smoke test passed: short run on MUTAG completes end-to-end on CPU in seconds, accuracy already in the right ballpark
- [x] Training protocol confirmed from `main.py`: Adam optimizer, `StepLR` halving LR every 50 epochs, 350 epochs, 10-fold CV. Default CLI hyperparameters (5 layers, hidden_dim 64, dropout 0.5, batch 32, iters_per_epoch 50, lr 0.01) match the paper's setup
- [x] Found and worked around a quirk in the official code: `main.py`'s `--filename` output overwrites every epoch instead of appending, so it can't hold a full accuracy curve
- [x] Wrote `replication/run_gin.py` — our own driver script that runs all 10 folds, captures full per-epoch curves, and reproduces the paper's exact model-selection rule (average the 10 test-accuracy curves, pick the best epoch, report mean ± std across folds at that epoch)
- [x] Verified the paper's actual Table 1 results directly from the primary source (arXiv HTML render) — see results table in section 4
- [ ] Optional fallback: modern PyTorch Geometric reimplementation in `replication/modern-pyg/` — not needed unless the official repo path runs into trouble at full scale

## 3. Replication plan (Phase B)

- [ ] Run full 10-fold/350-epoch training on MUTAG
- [ ] Run PROTEINS, NCI1 (bioinformatics — similar graph sizes to MUTAG, comparable runtime expected)
- [ ] Run IMDB-BINARY, IMDB-MULTI, COLLAB (social — need `--degree_as_tag` flag, no node labels)
- [ ] Run REDDIT-BINARY, REDDIT-MULTI5K if compute allows (much larger graphs — may need a reduced-epoch or subsampled run, to be justified in the report if so)
- [ ] Log every dataset's result into the results table below
- [ ] Note any datasets/results that don't reproduce cleanly and why (resource limits, etc. — explicitly allowed by the assignment if justified)
- [ ] Investigation experiment: construct a small pair of non-isomorphic graphs that mean/max-pooling GNNs can't distinguish, show GIN can — a direct, hands-on test of the paper's theoretical claim

## 4. Replication results

Paper's Table 1 (GIN-0 variant — the simpler, better-performing of their two GIN variants), verified directly from the paper:

| Dataset | # Graphs | # Classes | Avg # Nodes | Paper GIN-0 acc. | Our result |
|---|---|---|---|---|---|
| MUTAG | 188 | 2 | 17.9 | 89.4 ± 5.6% | — |
| PTC | 344 | 2 | 25.5 | 64.6 ± 7.0% | — |
| PROTEINS | 1113 | 2 | 39.1 | 76.2 ± 2.8% | — |
| NCI1 | 4110 | 2 | 29.8 | 82.7 ± 1.7% | — |
| COLLAB | 5000 | 3 | 74.5 | 80.2 ± 1.9% | — |
| IMDB-BINARY | 1000 | 2 | 19.8 | 75.1 ± 5.1% | — |
| IMDB-MULTI | 1500 | 3 | 13.0 | 52.3 ± 2.8% | — |
| REDDIT-BINARY | 2000 | 2 | 429.6 | 92.4 ± 2.5% | — |
| REDDIT-MULTI5K | 5000 | 5 | 508.5 | 57.5 ± 1.5% | — |

Paper's headline claim: GIN-0 matches or beats every baseline (WL subtree kernel, DCNN, PatchySAN, DGCNN, GraphSAGE, weaker sum/mean/max-pooling GNN variants) on all 9 datasets.

## 5. New data creation (Phase C)

**Original construction (Yanardag & Vishwanathan, "Deep Graph Kernels", KDD 2015 — the source of IMDB-BINARY/MULTI, which GIN's paper reuses as a benchmark):** nodes = actors, edge between two actors if they've co-starred in a movie, one graph per collaboration ego-network, labeled by genre (IMDB-BINARY: Action vs. Romance; IMDB-MULTI: Comedy vs. Romance vs. Sci-Fi).

**Our mirrored construction, via the TMDb API:**
- [x] Checked TMDb API terms: free API key via account signup (instant, no approval wait), ~40 req/s soft rate limit, requires attribution ("This product uses the TMDb API but is not endorsed or certified by TMDb") — fine for an academic project, just need to include the attribution line in the report/README
- [x] Checked the GIN code's exact expected data format (from `util.py`) — plain text, one file per dataset: graph count header, then per graph `n_nodes graph_label` followed by one line per node (`tag degree neighbor1 neighbor2 ...`). Building the scraper to output this format directly means the new dataset drops straight into the existing pipeline for Phase D with **zero code changes**.
- [x] Decided construction rule: for each target genre, pull movies released **2016-01-01 onward** (safely after the 2015 original dataset was built, guaranteeing no overlap), keep only movies whose genre list contains exactly one target genre (avoids ambiguous multi-genre labels), take each movie's top-billed cast (~20 actors) as that graph's nodes, and connect two actors with an edge if they co-appeared in *any* sampled movie of that genre — this reproduces the "ego-network of collaboration" structure rather than a trivial per-movie clique.
- [x] Target scale: mirror the originals — ~500 movies/graphs per genre (1000 total for binary, 1500 for multi)
- [ ] **Blocked on a TMDb API key** — needs a free account signup at themoviedb.org (Settings → API), which only the user can do. Script is written and ready to run the moment a key is available.
- [x] Wrote `new_data/scraping/build_tmdb_dataset.py` — fetches genre-filtered movies + cast via TMDb, builds the collaboration graphs, writes both the TU-format dataset files (`IMDBBINARY-NEW`, `IMDBMULTI-NEW`) and a companion metadata JSON (movie titles/actor names per graph, for later EDA/report writeup)
- [ ] Run the scraper once a key is available, sanity-check output (graph count, average node count vs. original ~19.8/13.0)
- [ ] Stretch: second new dataset via OpenAlex (COLLAB-style co-authorship) and/or Reddit API (REDDIT-style), for a second independent test
- [ ] Document new data quality/limitations vs. the original (size, class balance, single-genre-only filtering reduces pool size, etc.)

## 6. Test GIN on new data (Phase D)

- [ ] Run the same GIN pipeline on the new dataset(s)
- [ ] Compare accuracy/behavior to the original IMDB/COLLAB results — does the paper's claim hold on new data?
- [ ] Dig into failure/success cases — explicit grading criterion, not optional polish

## 7. Deliverables (Phase E)

- [ ] Project proposal draft (`proposal/`) — paper justification, feasibility, plan
- [ ] Update presentation (`presentation/`)
- [ ] Final report (`report/`) — full writeup with all results, code, new data, limitations
- [ ] Clean up GitHub repo for submission (report, scripts, new data, README)

## 8. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-08-24 | Chose GIN over GCN and tabular candidates (XGBoost, CatBoost, SMOTE, SHAP, etc.) | Strongest theory-backed justification, cleanest "rebuild same recipe with new data" story via TMDb/OpenAlex/Reddit, CPU-feasible, task shape (graph classification, train/test split) simpler to redo than GCN's transductive node classification |
| 2026-08-24 | Skipping the modern PyTorch Geometric reimplementation for now | Official repo already runs cleanly on current PyTorch/Python — no need for a fallback unless something breaks later at full scale |

## 9. Open questions

- TMDb API terms of use for scraping (rate limits, attribution requirements) — not yet checked
- How large a REDDIT-BINARY/MULTI5K subset is CPU-feasible before training gets too slow — not yet tested at full scale

## 10. Next step

Run full 10-fold/350-epoch training on MUTAG (`cd replication && python run_gin.py --dataset MUTAG --epochs 350`), log the result, then move on to PROTEINS/NCI1.
