# Project Plan — Replicating GIN (Graph Isomorphism Network)

**Status legend:** ⬜ not started · 🔄 in progress · ✅ done · ⚠️ blocked/issue found

Last updated: 2026-08-24

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
- [ ] Init local git repo (not yet pushed anywhere — will ask before creating/pushing to a remote)
- [ ] Clone official GIN repo into `replication/official-repo/`
- [ ] Attempt to install official repo's dependencies (expected friction: repo targets old PyTorch 0.4.1/1.0.0, we're on Python 3.11 which likely can't get wheels that old) — **document whatever happens, success or failure, as real replication evidence**
- [ ] Set up modern fallback path in `replication/modern-pyg/` using PyTorch Geometric (current PyTorch + PyG, Python 3.11-compatible)

### Phase B — Replicate original results
- [ ] Download/load TUDataset benchmarks: MUTAG, PROTEINS, NCI1 (bioinformatics) + IMDB-BINARY, IMDB-MULTI, COLLAB (social) at minimum; REDDIT-BINARY/MULTI5K if compute allows
- [ ] Run GIN training on each (official repo path and/or PyG path)
- [ ] Record accuracy per dataset, compare against paper's reported numbers
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
| 2026-08-24 | Project structure created locally; git not yet initialized/pushed | Setting up scaffolding before writing any code; will confirm before doing anything remote (GitHub push) |

## 4. Open questions / things to verify

- Does the official repo actually run under Python 3.11, or do we need an older Python side-by-side (e.g. via `venv` with an old CPython, if available) — **testing next**
- TMDb API terms of use for scraping (rate limits, attribution requirements) — not yet checked
- How large a REDDIT-BINARY/MULTI5K subset is CPU-feasible before training gets too slow

## 5. Next immediate step

Clone the official GIN repo and attempt a dependency install to see exactly what breaks under Python 3.11 — result will be logged below and this plan updated.
