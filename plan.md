# Project Plan — Replicating GIN (Graph Isomorphism Network)

**Status legend:** ⬜ not started · 🔄 in progress · ✅ done · ⚠️ blocked/issue found

Last updated: 2026-08-24 (session 2)

---

## 0. Project summary

- **Paper:** "How Powerful are Graph Neural Networks?" — Xu, Hu, Leskovec & Jegelka, ICLR 2019 (CORE A*). [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
- **Official code:** https://github.com/weihua916/powerful-gnns
- **Track:** Novel project (self-chosen paper, needs feasibility justification in the proposal)
- **Why this paper** (full reasoning in `proposal/justification.md` once written): theoretically grounded (proves GIN = Weisfeiler-Lehman test expressiveness, the ceiling for message-passing GNNs), ICLR is CORE A*, task shape (graph classification) is simpler to redo on new data than transductive node classification, and 3 of its benchmark datasets (IMDB, COLLAB, REDDIT) are literally built from scrapeable public metadata — giving an unusually clean "new data" story.

## 1. Assignment requirements (recap, from `instructions .md`)

- 3 deliverables: proposal (20%), update presentation (40%), final report (40%)
- Novel project = no 80% cap, but must justify paper/code/data quality (venue rank, citations)
- Must show: (a) unit techniques/tools used, (b) real replication attempt on original data, (c) new data created + code that runs, (d) polished writeup
- GitHub repo is part of the submission (report, new data, generation scripts)

## 2. Phase plan

### Phase A — Environment & setup
- [x] Confirm local tooling: Python 3.11.9, pip 24.0, git 2.53.0, **no conda**, internet access to GitHub confirmed
- [x] Create project folder structure (`replication/`, `new_data/`, `report/`, `proposal/`, `presentation/`, `results/`, `notebooks/`)
- [x] Init local git repo (not yet pushed anywhere — will ask before creating/pushing to a remote). First commit: scaffolding + plan.md.
- [x] Clone official GIN repo into `replication/official-repo/`
- [x] Attempt to install official repo's dependencies — **result: no friction at all.** Installed current CPU PyTorch (2.13.0+cpu) via `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt` (tqdm, numpy, networkx). Despite the README only claiming to test PyTorch 0.4.1/1.0.0, the code runs unmodified on 2.13.0. Only 2 harmless deprecation warnings (`lr_scheduler.step()` ordering, `torch.sparse.FloatTensor` → should use `torch.sparse_coo_tensor`), no functional impact.
- [x] Smoke-tested: `python main.py --dataset MUTAG --epochs 3 --iters_per_epoch 20 --fold_idx 0` ran end-to-end on CPU. 3 epochs took ~1 second total (188 graphs, tiny). Test accuracy already 78.9% after just 3 epochs (paper reports ~86-90% with full training/hyperparameters) — strong sign a full run will reproduce the paper's ballpark.
- [x] Datasets: **no separate download needed** — `dataset.zip` (33MB) ships inside the official repo itself, containing all 9 benchmarks: MUTAG, PROTEINS, PTC, NCI1 (bioinformatics), COLLAB, IMDBBINARY, IMDBMULTI, REDDITBINARY, REDDITMULTI5K (social). Unzipped into `replication/official-repo/dataset/`.
- [ ] Set up modern fallback path in `replication/modern-pyg/` using PyTorch Geometric — **not needed for now**, official repo works as-is; keep this as a backup/comparison item, not a blocker.

### Phase B — Replicate original results
- [x] Datasets already available (shipped in repo, see Phase A)
- [x] Inspected `main.py`'s exact training protocol: Adam optimizer, `StepLR` halving the LR every 50 epochs, 350 epochs, 10-fold CV via `separate_data(graphs, seed, fold_idx)`. Default CLI hyperparameters (5 layers, hidden_dim 64, dropout 0.5, batch 32, iters_per_epoch 50, lr 0.01) match the paper's setup — README says these "are not the best performing" (paper does a small grid search per dataset), but they're a legitimate, representative point in that grid.
- [x] **Found a quirk in the official code**: `main.py`'s `--filename` output option opens the file in `'w'` mode *every epoch*, so it only ever holds the last epoch's numbers, not a full curve — can't be used to reproduce the paper's "average the 10 validation curves, pick best epoch" model-selection procedure as-is.
- [x] Wrote our own driver script `replication/run_gin.py` (ours, tracked in git — not part of the vendored repo) that: runs all 10 folds via subprocess, captures every epoch's (loss, train_acc, test_acc) from stdout, then reproduces the paper's exact selection rule — average the 10 test-accuracy curves epoch-by-epoch, pick the single epoch with the highest averaged accuracy, report mean ± std across folds at that epoch. Results saved incrementally to `results/logs/<dataset>_cv_curves.json` (survives interruption).
- [x] Timed 20 epochs / 1 fold on MUTAG: 30s → full 350-epoch × 10-fold run estimated ~80-90 min for MUTAG. Test acc already 89.5% after just 20 epochs (fold 0) — matches paper's reported ~86-90% ballpark.
- [ ] ⏸️ **Full MUTAG run started, then stopped by user before completion** (wanted to review everything so far before letting a long background run continue). Pipeline (`replication/run_gin.py`) is written, tested, and confirmed working — just needs `python run_gin.py --dataset MUTAG --epochs 350` re-run when ready to resume. Partial/incomplete output files were deleted so nothing half-finished is left lying around.
- [ ] Repeat for PROTEINS, NCI1 (bioinformatics — similar graph sizes to MUTAG, should be comparable speed)
- [ ] Repeat for IMDB-BINARY, IMDB-MULTI, COLLAB (social — need `--degree_as_tag` flag per README, since these have no node labels)
- [ ] REDDIT-BINARY/MULTI5K if compute allows — graphs are much bigger (100s-1000s of nodes vs. MUTAG's ~18), expect this to be the slow one; may need to justify a reduced-epoch or subsampled run
- [ ] Record accuracy per dataset in `## 6. Replication results` table below, compare against paper's reported numbers
- [ ] Note any datasets/results we couldn't reproduce and why (resource limits, code rot, etc.) — instructions explicitly allow partial replication if justified
- [ ] Bonus/investigation experiment: construct a small pair of non-isomorphic graphs that mean/max-pooling GNNs can't distinguish, show GIN can — direct test of the paper's theoretical claim

### Phase C — New data creation (scraping)
- [ ] Pick primary new-data source (leaning **TMDb API**, mirroring IMDB-BINARY/MULTI's "actor co-starring" construction) — confirm API terms/rate limits before building
- [ ] Write scraping script(s) in `new_data/scraping/`
- [ ] Build actor co-starring ego-network graphs, labeled by genre, from movies not in the original IMDB dataset
- [ ] (Stretch) Second new dataset via OpenAlex (COLLAB-style co-authorship) and/or Reddit API (REDDIT-style), for redundancy/robustness of the "does it generalize" story
- [ ] Document data quality/limitations vs. the original (dataset size, class balance, etc.)

### Phase D — Test GIN on new data
- [ ] Run same GIN pipeline (from Phase B) on the new dataset(s)
- [ ] Compare accuracy/behavior to original IMDB/COLLAB results — does the paper's claim hold?
- [ ] Dig into failure/success cases (this is an explicit grading criterion, not optional polish)

### Phase E — Deliverables
- [ ] Project proposal draft (`proposal/`) — paper justification, feasibility, plan
- [ ] Update presentation (`presentation/`)
- [ ] Final report (`report/`) — full writeup with all results, code, new data, limitations
- [ ] Clean up GitHub repo for submission (report, scripts, new data, README)

## 3. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-08-24 | Chose GIN over GCN/tabular candidates (XGBoost, CatBoost, SMOTE, SHAP, etc.) | Strongest theory-backed justification, cleanest "rebuild same recipe with new data" story via TMDb/OpenAlex/Reddit, CPU-feasible, task shape (graph classification, train/test split) simpler to redo than GCN's transductive node classification |
| 2026-08-24 | Project structure created locally; git initialized locally, no remote yet | Setting up scaffolding before writing any code; will confirm with user before doing anything remote (creating/pushing to GitHub) |
| 2026-08-24 | Skipping the "modern PyG reimplementation" fallback for now | Official repo already runs cleanly on current PyTorch/Python — no need for a fallback path unless something breaks later at full scale |

## 4. Open questions / things to verify

- ~~Does the official repo actually run under Python 3.11?~~ **Resolved: yes, runs unmodified on PyTorch 2.13 / Python 3.11.**
- TMDb API terms of use for scraping (rate limits, attribution requirements) — not yet checked
- How large a REDDIT-BINARY/MULTI5K subset is CPU-feasible before training gets too slow — not yet tested at full scale
- Exact paper hyperparameters per dataset (paper doesn't ship a config file, README says "refer to the paper") — need to pull from the paper's tables before running full replication, otherwise default args won't match reported numbers

## 5. Next immediate step

**Paused here at the user's request** — everything below is ready to run, nothing is currently executing. To resume: `cd replication && python run_gin.py --dataset MUTAG --epochs 350`, then log the result in the table below, then kick off PROTEINS/NCI1 next (expected similar runtime to MUTAG), then IMDB/COLLAB (with `--degree_as_tag`), then decide on REDDIT scope based on how slow a quick timing test shows it to be.

## 6. Replication results

Paper's Table 1 (GIN-0 variant, the simpler/better-performing of their two GIN variants), verified directly from the paper (via ar5iv HTML rendering of arXiv:1810.00826):

| Dataset | # Graphs | # Classes | Avg # Nodes | Paper GIN-0 acc. | Our result | Notes |
|---|---|---|---|---|---|---|
| MUTAG | 188 | 2 | 17.9 | 89.4 ± 5.6% | ⏸️ paused | run started, stopped before completion at user's request — pipeline verified working, ready to resume |
| PTC | 344 | 2 | 25.5 | 64.6 ± 7.0% | ⬜ not started | in dataset.zip, not yet planned above — add to Phase B if time allows |
| PROTEINS | 1113 | 2 | 39.1 | 76.2 ± 2.8% | ⬜ not started | |
| NCI1 | 4110 | 2 | 29.8 | 82.7 ± 1.7% | ⬜ not started | |
| COLLAB | 5000 | 3 | 74.5 | 80.2 ± 1.9% | ⬜ not started | needs `--degree_as_tag` |
| IMDB-BINARY | 1000 | 2 | 19.8 | 75.1 ± 5.1% | ⬜ not started | needs `--degree_as_tag` |
| IMDB-MULTI | 1500 | 3 | 13.0 | 52.3 ± 2.8% | ⬜ not started | needs `--degree_as_tag` |
| REDDIT-BINARY | 2000 | 2 | 429.6 | 92.4 ± 2.5% | ⬜ not started | large graphs, expect slow — avg 430 nodes vs. MUTAG's 18 |
| REDDIT-MULTI5K | 5000 | 5 | 508.5 | 57.5 ± 1.5% | ⬜ not started | large graphs, expect slow |

Paper's headline claim: GIN-0 matches or beats every baseline (WL subtree kernel, DCNN, PatchySAN, DGCNN, GraphSAGE, and weaker sum/mean/max-pooling GNN variants) on all 9 datasets.
