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

**Scope narrowed to IMDB-BINARY and IMDB-MULTI only** — these are the two original datasets we have matching new (scraped) data for, so replicating exactly these two gives a direct, apples-to-apples chain for Phase D: paper's original result → our replication of the original → our result on the new scraped data. The other 7 datasets (MUTAG, PTC, PROTEINS, NCI1, COLLAB, REDDIT-BINARY, REDDIT-MULTI5K) are dropped from the plan — no matching new data was built for them, so replicating them wouldn't feed into the new-data comparison that's the point of this project. This is a deliberate scoping choice, not a resource shortfall, and will be stated as such in the report.

**Compute plan:** local CPU timing showed IMDB-BINARY/MULTI take ~3.5 hrs each at full spec (350 epochs × 10 folds) — too slow to run locally for 4 datasets (2 original + 2 new). Moved execution to **Google Colab (T4 GPU)** instead, which the user has access to.

- [x] Confirmed the official `main.py` already auto-detects CUDA (`torch.device("cuda:0")` if available) — no code changes needed to run on GPU
- [x] Built `notebooks/gin_colab_replication.ipynb` — self-contained notebook: clones the official repo, installs deps, lets the user upload `IMDBBINARY-NEW.txt`/`IMDBMULTI-NEW.txt`, writes a Colab-local copy of the CV driver (same logic as `replication/run_gin.py`), then runs all 4 datasets (IMDB-BINARY, IMDB-MULTI, IMDBBINARY-NEW, IMDBMULTI-NEW) back to back at full spec, ending in one copy-pasteable summary block
- [x] User ran the notebook cell-by-cell on Colab T4, reported back the final summary block
- [x] Logged all 4 results (2 original + 2 new) into the results table below, compared against paper's reported numbers
- [ ] Investigation experiment: construct a small pair of non-isomorphic graphs that mean/max-pooling GNNs can't distinguish, show GIN can — a direct, hands-on test of the paper's theoretical claim

## 4. Replication results

Paper's Table 1 (GIN-0 variant — the simpler, better-performing of their two GIN variants), verified directly from the paper:

| Dataset | # Graphs | # Classes | Avg # Nodes | Paper GIN-0 acc. | Our result |
|---|---|---|---|---|---|
| IMDB-BINARY (original) | 1000 | 2 | 19.8 | 75.1 ± 5.1% | **74.4 ± 4.1%** (best epoch 80, ~50.0 min on T4) |
| IMDB-MULTI (original) | 1500 | 3 | 13.0 | 52.3 ± 2.8% | **50.5 ± 2.3%** (best epoch 213, ~52.9 min on T4) |
| IMDBBINARY-NEW (scraped) | 996 | 2 | 19.2 | n/a — new data | **56.1 ± 2.2%** (best epoch 136, 53.0 min on T4) |
| IMDBMULTI-NEW (scraped) | 1492 | 3 | 18.9 | n/a — new data | **40.9 ± 2.1%** (best epoch 265, 56.1 min on T4) |

**Result — replication succeeded, new-data test shows a real drop:**
- **Replication of the original datasets is a clean success.** IMDB-BINARY: 74.4% vs. paper's 75.1% (well within the paper's own ±5.1% std). IMDB-MULTI: 50.5% vs. paper's 52.3% (within ~1 std of the paper's ±2.8%). Same code, same protocol, same hyperparameters — reproduces the paper closely on both original benchmarks.
- **Applying the identical pipeline to our new scraped data shows a real, isolated drop**, now confirmed against *our own* replication (not just the paper's published number, which could differ for other reasons): IMDBBINARY-NEW is **18.3 points below** our own IMDB-BINARY replication (56.1% vs. 74.4%); IMDBMULTI-NEW is **9.6 points below** our own IMDB-MULTI replication (40.9% vs. 50.5%). Since the code, hyperparameters, and evaluation protocol are all identical between each original/new pair, this gap isolates to the *data itself*.
- **Leading explanation**, consistent with the data-quality note in section 5: our new graphs are far denser/closer-to-clique (avg ~178 edges vs. the originals' ~96.5) because edges only form from actor overlap *within* our sampled movie pool, so most movies come out as near-complete cliques on their 20-actor cast. A near-clique carries much less distinguishing topological structure than the originals' sparser ego-networks — and since node features here are only `--degree_as_tag` (no richer attributes), GIN has substantially less signal to work with. This is the natural next thing to dig into for Phase D/the final report — e.g. checking whether misclassifications correlate with how clique-like a given graph is.

Paper's headline claim: GIN-0 matches or beats every baseline (WL subtree kernel, DCNN, PatchySAN, DGCNN, GraphSAGE, weaker sum/mean/max-pooling GNN variants) on all 9 of its benchmark datasets — IMDB-BINARY and IMDB-MULTI included.

## 5. New data creation (Phase C)

**Original construction (Yanardag & Vishwanathan, "Deep Graph Kernels", KDD 2015 — the source of IMDB-BINARY/MULTI, which GIN's paper reuses as a benchmark):** nodes = actors, edge between two actors if they've co-starred in a movie, one graph per collaboration ego-network, labeled by genre (IMDB-BINARY: Action vs. Romance; IMDB-MULTI: Comedy vs. Romance vs. Sci-Fi).

**Our mirrored construction, via the TMDb API:**
- [x] Checked TMDb API terms: free API key via account signup (instant, no approval wait), ~40 req/s soft rate limit, requires attribution ("This product uses the TMDb API but is not endorsed or certified by TMDb") — fine for an academic project, just need to include the attribution line in the report/README
- [x] Checked the GIN code's exact expected data format (from `util.py`) — plain text, one file per dataset: graph count header, then per graph `n_nodes graph_label` followed by one line per node (`tag degree neighbor1 neighbor2 ...`). Building the scraper to output this format directly means the new dataset drops straight into the existing pipeline for Phase D with **zero code changes**.
- [x] Decided construction rule: for each target genre, pull movies released **2016-01-01 onward** (safely after the 2015 original dataset was built, guaranteeing no overlap), keep only movies whose genre list contains exactly one target genre (avoids ambiguous multi-genre labels), take each movie's top-billed cast (~20 actors) as that graph's nodes, and connect two actors with an edge if they co-appeared in *any* sampled movie of that genre — this reproduces the "ego-network of collaboration" structure rather than a trivial per-movie clique.
- [x] Target scale: mirror the originals — ~500 movies/graphs per genre (1000 total for binary, 1500 for multi)
- [x] TMDb API key obtained (kept out of git — passed via `TMDB_API_KEY` environment variable only, never committed)
- [x] Wrote `new_data/scraping/build_tmdb_dataset.py` — fetches genre-filtered movies + cast via TMDb, builds the collaboration graphs, writes both the TU-format dataset files (`IMDBBINARY-NEW`, `IMDBMULTI-NEW`) and a companion metadata JSON (movie titles/actor names per graph, for later EDA/report writeup)
- [x] Small test run (5 movies/genre) passed — correct TU-format output, sensible real movies/actors in metadata, graph structure as expected
- [x] `IMDBBINARY-NEW` scraped: **996 graphs** (500 Action, 496 Romance — a handful of Romance movies had <5 billed cast and were skipped), avg **19.2 nodes/graph** (original IMDB-BINARY: 19.8 — very close match), avg 177.8 edges/graph (original: 96.5 — ours is denser; likely because most graphs come out near a full clique at cast_size=20, since actor overlap *across* different sampled movies is fairly rare relative to same-movie co-billing. Noted as a data-quality difference to discuss in the report.)
- [x] `IMDBMULTI-NEW` scraped: **1492 graphs** (Comedy 499, Romance 493, Sci-Fi 500), avg **18.9 nodes/graph** (original IMDB-MULTI: 13.0 — ours runs bigger, since we standardized cast_size=20 across both binary and multi variants rather than letting node count vary by genre like the original), avg 174.1 edges/graph
- [x] Both new datasets committed to git under `new_data/processed/`
- [x] TMDb attribution requirement noted for the final report/README: "This product uses the TMDb API but is not endorsed or certified by TMDb"
- [ ] Stretch: second new dataset via OpenAlex (COLLAB-style co-authorship) and/or Reddit API (REDDIT-style), for a second independent test
- [ ] Document new data quality/limitations vs. the original (size, class balance, single-genre-only filtering reduces pool size, etc.)

## 6. Test GIN on new data (Phase D)

- [x] Same GIN pipeline runs on the new dataset(s) automatically as part of the Colab notebook (section 3) — no separate step needed, all 4 datasets run in one pass
- [x] Compared accuracy to our own IMDB-BINARY/IMDB-MULTI replication results (not just the paper's published numbers) — **the paper's claim does not fully hold on new data**: -18.3 points on binary, -9.6 points on multi, isolated to the data since code/hyperparameters/protocol are identical. See section 4 for the full result and leading explanation (graph density/clique structure).
- [ ] Dig into failure/success cases in more depth — explicit grading criterion, not optional polish. Next: check whether individual misclassified new-data graphs correlate with how clique-like they are (e.g. edge density per graph vs. correct/incorrect prediction)

## 7. Baseline comparison: GraphSAGE

**Why GraphSAGE:** the GIN paper's own Table 1 doesn't use GraphSAGE's original code — it approximates GraphSAGE *inside its own architecture* (quoting the paper: "max–1-layer correspond[s] to GraphSAGE, up to minor architecture modifications"). That means we can get this comparison from the exact same tested codebase, just with two flags changed (`--neighbor_pooling_type max --num_mlp_layers 1` instead of GIN's sum-pooling + multi-layer MLP) — zero new code, zero new dependencies, same training protocol.

Paper's Table 1 numbers for this comparison, for context on what to expect:

| Dataset | GIN-0 | GraphSAGE (Max–1-Layer) |
|---|---|---|
| IMDB-BINARY | 75.1 ± 5.1% | 72.3 ± 5.3% |
| IMDB-MULTI | 52.3 ± 2.8% | 50.9 ± 2.2% |

Question this answers: does GraphSAGE's ~2-3 point gap behind GIN on the original data hold, shrink, or vanish on our denser new-data graphs? If the density/clique explanation from section 4 is right, it might shrink — less structure overall could mean less room for GIN's extra expressiveness to matter.

- [x] Confirmed locally (small smoke-test runs, not full spec) that the flag combination runs cleanly on both an original dataset (IMDB-BINARY) and a new dataset (IMDBBINARY-NEW) — no errors, ready for a full Colab run
- [x] Built `notebooks/graphsage_colab_comparison.ipynb` — same structure as the GIN notebook (clone repo, upload new datasets, run all 4 datasets, print summary), driver script tags output filenames with `_graphsage` so results don't collide with the earlier GIN run's files
- [ ] User runs the notebook on Colab T4, reports back the final summary block
- [ ] Log all 4 GraphSAGE results into the table below
- [ ] Decide whether to also add GCN (Mean–1-Layer) — same marginal effort, one flag change (`--neighbor_pooling_type average` instead of `max`)

**GraphSAGE results:**

| Dataset | Paper GraphSAGE acc. | Our GIN result (section 4) | Our GraphSAGE result |
|---|---|---|---|
| IMDB-BINARY (original) | 72.3 ± 5.3% | 74.4 ± 4.1% | — |
| IMDB-MULTI (original) | 50.9 ± 2.2% | 50.5 ± 2.3% | — |
| IMDBBINARY-NEW (scraped) | n/a | 56.1 ± 2.2% | — |
| IMDBMULTI-NEW (scraped) | n/a | 40.9 ± 2.1% | — |

## 8. Deliverables (Phase E)

- [ ] Project proposal draft (`proposal/`) — paper justification, feasibility, plan
- [ ] Update presentation (`presentation/`)
- [ ] Final report (`report/`) — full writeup with all results, code, new data, limitations
- [ ] Clean up GitHub repo for submission (report, scripts, new data, README)

## 9. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-08-24 | Chose GIN over GCN and tabular candidates (XGBoost, CatBoost, SMOTE, SHAP, etc.) | Strongest theory-backed justification, cleanest "rebuild same recipe with new data" story via TMDb/OpenAlex/Reddit, CPU-feasible, task shape (graph classification, train/test split) simpler to redo than GCN's transductive node classification |
| 2026-08-24 | Skipping the modern PyTorch Geometric reimplementation for now | Official repo already runs cleanly on current PyTorch/Python — no need for a fallback unless something breaks later at full scale |
| 2026-08-25 | Narrowed Phase B replication to IMDB-BINARY and IMDB-MULTI only, dropped the other 7 benchmark datasets from the plan | Only these two have matching scraped new data (`IMDBBINARY-NEW`, `IMDBMULTI-NEW`), so replicating exactly these two directly feeds the Phase D comparison instead of spreading effort across datasets that don't connect to the new-data test |
| 2026-08-25 | Moved full-spec training runs (350 epochs × 10 folds) from local CPU to Google Colab (T4 GPU) via a self-contained notebook | Local CPU timing showed ~3.5 hrs/dataset × 4 datasets ≈ 6-7 hrs total — too slow; user has Colab GPU access, and the official code already supports CUDA with zero changes |
| 2026-08-25 | Added GraphSAGE as a second algorithm for comparison, implemented via the GIN paper's own architecture flags rather than a separate codebase | Zero new code/dependencies (lowest error risk), and it's the paper's own defined approximation — comparable methodology to how the paper itself reports the GraphSAGE baseline |

## 10. Open questions

- None currently blocking

## 11. Next step

Waiting on the GraphSAGE Colab run (`notebooks/graphsage_colab_comparison.ipynb`) — user to run cell-by-cell and report back the final summary block. After that's logged, remaining items:
1. Dig deeper into *why* accuracy drops on the new data — check whether misclassified new-data graphs correlate with edge density/clique-ness (data already available in the `_metadata.json` files + saved CV curves)
2. Run the investigation experiment: build a small pair of non-isomorphic graphs that mean/max-pooling GNNs can't tell apart, confirm GIN can — a direct test of the paper's theoretical claim
3. Decide on adding GCN as a third comparison point

After that: move to Phase E (proposal, presentation, report).
