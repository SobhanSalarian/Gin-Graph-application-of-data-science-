# Replicating and Extending *How Powerful are Graph Neural Networks?*

COMP8240 Applications of Data Science, Macquarie University, Session 2 2026. Novel project by Sobhan Salarian (sobhan.salarian@students.mq.edu.au).

This project replicates the graph classification results of Xu et al., ["How Powerful are Graph Neural Networks?"](https://arxiv.org/abs/1810.00826) (ICLR 2019), using the authors' [official code](https://github.com/weihua916/powerful-gnns), and then tests the same method on new data:

- **Replication** of GIN-0 and the paper's GraphSAGE approximation (Max-1-layer) on IMDB-BINARY and IMDB-MULTI.
- **A dataset we built** from The Movie Database (TMDb): actor ego-networks labelled by genre, from films released from 2016 onward.
- **An existing dataset** not used in the paper: ENZYMES.
- **An LLM-as-judge check** of the TMDb genre labels.

The full write-up is [`report/report.pdf`](report/report.pdf).

## Results

Test accuracy (%), mean ± standard deviation over 10 folds, at the epoch chosen by the paper's selection rule. The node-count baseline is a classifier that sees only the number of nodes; beating it requires information from graph structure.

| Dataset | GIN-0 | Max-1-layer | Node-count baseline |
|---|---|---|---|
| IMDB-BINARY (paper) | 75.1 ± 5.1 | 72.3 ± 5.3 | |
| IMDB-BINARY (ours) | 73.9 ± 2.7 | 72.6 ± 3.0 | 58.1 |
| IMDB-MULTI (paper) | 52.3 ± 2.8 | 50.9 ± 2.2 | |
| IMDB-MULTI (ours) | 50.6 ± 2.7 | 50.9 ± 2.8 | 40.1 |
| TMDb first scrape, binary (flawed, see below) | 55.9 ± 2.3 | 56.2 ± 2.3 | 56.0 |
| TMDb first scrape, multi (flawed) | 41.0 ± 2.3 | 41.0 ± 2.2 | 40.5 |
| TMDb ego-networks, binary | 74.4 ± 2.5 | 74.0 ± 1.9 | 57.7 |
| TMDb ego-networks, multi | 59.7 ± 4.4 | 57.6 ± 5.0 | 46.3 |
| ENZYMES | 48.5 ± 7.0 | 33.3 ± 5.6 | 23.5 |

- The replication falls within one of the paper's standard deviations on both datasets.
- Our first TMDb construction made every graph a clique, so node count was the only signal. We kept these results in the report because they show the flaw and how we found it.
- On the rebuilt ego-networks both models use structure and are tied. On ENZYMES, GIN clearly beats Max-1-layer.

## Repository layout

| Path | Contents |
|---|---|
| `report/` | Final report (`report.tex`, `report.pdf`), bibliography, and `make_figures.py` for the learning-curve figure |
| `replication/run_gin.py` | Our driver: runs all 10 folds of the official `main.py`, records every epoch, applies the paper's selection rule |
| `replication/setup_server.sh` | One-time setup on a Linux GPU machine: clones and patches the official code, creates a virtual environment, installs dependencies |
| `replication/run_all_server.sh` | Runs GIN and Max-1-layer on the given datasets |
| `replication/progress.py` | Shows folds done and time left for running jobs |
| `new_data/scraping/build_tmdb_dataset.py` | Collects films and credits from the TMDb API |
| `new_data/scraping/build_ego_networks.py` | Builds actor ego-network datasets from the collected films |
| `new_data/scraping/convert_tu_dortmund.py` | Converts a TU Dortmund dataset (ENZYMES) to the official code's format |
| `new_data/scraping/check_dataset.py` | Structure checks and the node-count baseline |
| `new_data/processed/` | All datasets in the official code's text format, with metadata |
| `evaluation/llm_judge.py` | LLM-as-judge check of the genre labels |
| `results/logs/` | Per-epoch, per-fold accuracy for every run, with summaries |
| `results/llm_judge/` | Every judge answer and the summary figures. Files ending `-NEW` are the first run, on the faulty scrape, which exposed the date bug |
| `results/scraping/` | Logs of each TMDb scrape |
| `notebooks/` | Earlier Google Colab notebooks used before moving to the GPU server |
| `plan.md` | Working log of the project, with dated decisions |

Dataset names in `new_data/processed/`:

| Name | What it is |
|---|---|
| `IMDBBINARY-EGO2016`, `IMDBMULTI-EGO2016` | Final ego-network datasets used in the report |
| `IMDBBINARY-2016`, `IMDBMULTI-2016` | One graph per film, from the corrected 2016+ scrape. Source of the ego-networks; as graphs they are cliques |
| `IMDBBINARY-NEW`, `IMDBMULTI-NEW` | First scrape: one clique per film, and the date filter was not applied. Kept to reproduce the flawed results |
| `ENZYMES` | ENZYMES converted from the TU Dortmund collection |

## How to run

**Requirements:** Python 3.11 or later, and a Linux machine with an NVIDIA GPU for the full runs. Each GIN run takes about 20 to 80 minutes on an RTX 6000 Ada. Max-1-layer is slower; see the report's Hardware paragraph.

```bash
git clone https://github.com/SobhanSalarian/Gin-Graph-application-of-data-science-.git assign1
cd assign1
bash replication/setup_server.sh
```

The setup script applies our one-line fix to the official code: in the max-pooling path, an index tensor was left on the CPU, which stalled GPU training.

**Run the experiments** (inside `tmux`, since runs take hours):

```bash
bash replication/run_all_server.sh                          # all datasets, GIN then Max-1-layer
bash replication/run_all_server.sh IMDBBINARY-EGO2016       # or a subset
python3 replication/progress.py                             # progress, from another terminal
```

ENZYMES uses its node labels, so it runs without `--degree_as_tag`:

```bash
source .venv/bin/activate
mkdir -p replication/official-repo/dataset/ENZYMES
cp new_data/processed/ENZYMES/ENZYMES.txt replication/official-repo/dataset/ENZYMES/
python replication/run_gin.py --dataset ENZYMES --epochs 350
python replication/run_gin.py --dataset ENZYMES --epochs 350 --tag graphsage \
    --extra --neighbor_pooling_type max --num_mlp_layers 1
```

For Max-1-layer on the dense ego-networks, a CPU run with several folds in parallel was faster than the GPU:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python replication/run_gin.py \
    --dataset IMDBBINARY-EGO2016 --epochs 350 --tag graphsage --jobs 10 \
    --extra --degree_as_tag --neighbor_pooling_type max --num_mlp_layers 1
```

Results are written to `results/logs/<dataset>[_graphsage]_cv_curves.json`.

**Rebuild the TMDb data** (needs your own TMDb API key, kept in an environment variable and never committed):

```bash
export TMDB_API_KEY=...
python new_data/scraping/build_tmdb_dataset.py --mode binary
python new_data/scraping/build_tmdb_dataset.py --mode multi
python new_data/scraping/build_ego_networks.py --mode binary
python new_data/scraping/build_ego_networks.py --mode multi
python new_data/scraping/check_dataset.py new_data/processed/IMDBBINARY-EGO2016/IMDBBINARY-EGO2016.txt
```

TMDb's catalogue changes over time, so a new scrape will not give exactly the same films.

**Rebuild ENZYMES:** download it from the [TUDataset collection](https://chrsmrrs.github.io/datasets/), unzip it, then run:

```bash
python new_data/scraping/convert_tu_dortmund.py path/to/ENZYMES new_data/processed/ENZYMES/ENZYMES.txt
```

**LLM judge** (needs the Claude Code CLI, logged in):

```bash
python evaluation/llm_judge.py --mode binary
python evaluation/llm_judge.py --mode multi
```

**Report:** run `python report/make_figures.py`, then in `report/` run `pdflatex report`, `bibtex report`, `pdflatex report` and `pdflatex report`.

## Licence and use declaration

The datasets and metadata derived from TMDb in this repository were created for non-commercial university coursework only. They must not be used for any commercial purpose. Any reuse, redistribution or publication beyond this assessment requires the appropriate licence and written permission from TMDb. No such licence is claimed here, and this repository does not grant one.

This product uses the TMDb API but is not endorsed or certified by TMDb.

The official GIN code is by Xu et al. and is cloned from its own repository at setup time, not redistributed here. ENZYMES is from the TUDataset collection (Morris et al., 2020), originally from Borgwardt et al. (2005).
