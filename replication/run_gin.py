"""
Our own driver script (not part of the vendored official repo) that runs the
official main.py across all 10 CV folds for a given dataset, and reproduces
the paper's exact model-selection procedure:

  1. Run all 10 folds, recording the (loss, train_acc, test_acc) at every epoch.
  2. Average the 10 test-accuracy curves epoch-by-epoch -> one averaged curve.
  3. Pick the single epoch with the highest averaged accuracy.
  4. Report mean +/- std across folds AT that epoch.

This matches the "Cross-validation strategy in the paper" section of the
official repo's README. main.py's own --filename option is not used for this
because it overwrites the file every epoch instead of appending, so it can't
hold a full curve - we capture stdout per fold instead.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).parent / "official-repo"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "logs"

EPOCH_RE = re.compile(
    r"loss training: ([\d.]+)\naccuracy train: ([\d.]+) test: ([\d.]+)"
)


def run_fold(dataset, fold_idx, extra_args, epochs):
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--fold_idx", str(fold_idx),
        "--epochs", str(epochs),
    ] + extra_args
    proc = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fold {fold_idx} failed:\n{proc.stderr[-3000:]}")
    matches = EPOCH_RE.findall(proc.stdout)
    if len(matches) != epochs:
        raise RuntimeError(
            f"fold {fold_idx}: expected {epochs} epochs of output, parsed {len(matches)}. "
            f"stdout tail:\n{proc.stdout[-1000:]}"
        )
    return [
        {"loss": float(l), "train_acc": float(tr), "test_acc": float(te)}
        for l, tr, te in matches
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--tag", default="", help="suffix for the output filename, e.g. 'graphsage', so runs with different --extra flags on the same dataset don't overwrite each other")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"{args.dataset}{suffix}_cv_curves.json"

    all_folds = []
    t0 = time.time()
    for fold_idx in range(10):
        fstart = time.time()
        curve = run_fold(args.dataset, fold_idx, args.extra, args.epochs)
        all_folds.append(curve)
        print(
            f"[{args.dataset}] fold {fold_idx} done in {time.time() - fstart:.1f}s "
            f"(final test_acc={curve[-1]['test_acc']:.4f})",
            flush=True,
        )
        # write incrementally so partial progress survives if interrupted
        out_path.write_text(json.dumps({"dataset": args.dataset, "epochs": args.epochs,
                                         "extra_args": args.extra, "folds": all_folds}, indent=2))

    # paper's model-selection procedure
    n_epochs = args.epochs
    avg_curve = [
        sum(fold[e]["test_acc"] for fold in all_folds) / len(all_folds)
        for e in range(n_epochs)
    ]
    best_epoch = max(range(n_epochs), key=lambda e: avg_curve[e])
    accs_at_best = [fold[best_epoch]["test_acc"] for fold in all_folds]
    mean_acc = sum(accs_at_best) / len(accs_at_best)
    std_acc = (sum((a - mean_acc) ** 2 for a in accs_at_best) / len(accs_at_best)) ** 0.5

    summary = {
        "dataset": args.dataset,
        "epochs": n_epochs,
        "extra_args": args.extra,
        "best_epoch_0indexed": best_epoch,
        "mean_test_acc_at_best_epoch": mean_acc,
        "std_test_acc_at_best_epoch": std_acc,
        "per_fold_acc_at_best_epoch": accs_at_best,
        "total_runtime_sec": time.time() - t0,
    }
    out_path.write_text(json.dumps({"dataset": args.dataset, "epochs": args.epochs,
                                     "extra_args": args.extra, "folds": all_folds,
                                     "summary": summary}, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
