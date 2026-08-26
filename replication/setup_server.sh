#!/usr/bin/env bash
# One-time setup on the university GPU server. Run from the repo root after
# `git clone` (or rsync) has placed this project there:
#
#   ssh you@uni-server
#   git clone <your-repo-url> assign1 && cd assign1
#   bash replication/setup_server.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

echo "== GPU check =="
nvidia-smi || { echo "nvidia-smi not found -- is this a GPU node?"; exit 1; }

echo "== Cloning official GIN repo (gitignored, not part of this repo) =="
if [ ! -d replication/official-repo ]; then
    git clone https://github.com/weihua916/powerful-gnns replication/official-repo
fi

echo "== Applying GPU device-placement fix to max-pooling path (needed for GraphSAGE runs) =="
python3 - <<'PYEOF'
path = "replication/official-repo/models/graphcnn.py"
src = open(path).read()
old = "        return torch.LongTensor(padded_neighbor_list)"
new = "        return torch.LongTensor(padded_neighbor_list).to(self.device)"
if new in src:
    print("already patched, skipping")
elif old in src:
    open(path, "w").write(src.replace(old, new))
    print("patched:", old.strip(), "->", new.strip())
else:
    raise SystemExit("expected line not found -- official repo may have changed, needs a manual look")
PYEOF

echo "== Python env =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "== Installing PyTorch =="
echo "Check 'CUDA Version' in the nvidia-smi output above and adjust the"
echo "--index-url below to match (cu121 for CUDA 12.1+, cu118 for CUDA 11.8, etc)."
echo "See https://pytorch.org/get-started/locally/ for the current matrix."
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r replication/official-repo/requirements.txt
# scikit-learn isn't listed in the official repo's requirements.txt but
# util.py imports it (StratifiedKFold) -- only surfaced on a clean venv,
# since Colab and most local setups already have it preinstalled.
pip install scikit-learn

echo "== Sanity check =="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "Done. New scraped datasets (new_data/processed/) are already in git, so"
echo "they came along with the clone -- no upload step needed."
