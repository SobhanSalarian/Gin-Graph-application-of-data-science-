#!/usr/bin/env bash
# Full-spec run of GIN + GraphSAGE across all 4 datasets, on the uni GPU
# server. Meant to be launched inside tmux so it survives SSH disconnects:
#
#   ssh you@uni-server
#   cd assign1
#   tmux new -s gnn
#   bash replication/run_all_server.sh          # inside the tmux session
#   # Ctrl-b d to detach; `tmux attach -t gnn` later to check progress
#
# Progress is also visible without attaching via: tail -f results/logs/run_all.log
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
source .venv/bin/activate

mkdir -p results/logs
LOG=results/logs/run_all.log
exec > >(tee -a "$LOG") 2>&1

echo "== $(date) : starting full run =="

# Link the scraped new-data files into the official repo's expected layout
# (dataset/<name>/<name>.txt) -- one-time, idempotent.
for name in IMDBBINARY-NEW IMDBMULTI-NEW; do
    dest="replication/official-repo/dataset/$name"
    mkdir -p "$dest"
    cp -f "new_data/processed/$name/$name.txt" "$dest/$name.txt"
done

DATASETS=(IMDBBINARY IMDBMULTI IMDBBINARY-NEW IMDBMULTI-NEW)

for ds in "${DATASETS[@]}"; do
    echo "== $(date) : GIN on $ds =="
    python replication/run_gin.py --dataset "$ds" --epochs 350 \
        --extra --degree_as_tag

    echo "== $(date) : GraphSAGE (max-pool, 1-layer MLP) on $ds =="
    python replication/run_gin.py --dataset "$ds" --epochs 350 --tag graphsage \
        --extra --degree_as_tag --neighbor_pooling_type max --num_mlp_layers 1
done

echo "== $(date) : all runs complete =="
echo "Results written to results/logs/*_cv_curves.json -- copy back with:"
echo "  scp -r you@uni-server:~/assign1/results/logs ./results/"
echo "or, if this checkout has git remote access: git add results/logs && git commit && git push"
