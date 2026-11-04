"""
Shows how far the server runs have got: folds done, time per fold, and time left.
Read-only, so it is safe to run while training. Each dataset's fold timings come
from the run logs, where the first 10 are GIN and the next 10 are GraphSAGE.

Usage:
    watch -n 10 python3 ~/assign1/replication/progress.py
"""
import re
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "results" / "logs"
LOGS = ["run_all.log", "run_enzymes.log"]
DATASETS = ["IMDBBINARY-EGO2016", "IMDBMULTI-EGO2016", "ENZYMES"]
MODELS = ["GIN", "GraphSAGE"]
FOLDS = 10
LINE_RE = re.compile(r"\[(.+?)\] fold (\d+) done in ([\d.]+)s")


def main():
    times = defaultdict(list)
    for name in LOGS:
        path = LOG_DIR / name
        if path.exists():
            for dataset, _, seconds in LINE_RE.findall(path.read_text()):
                times[dataset].append(float(seconds))

    print(f"{'job':<31}{'folds':>7}{'min/fold':>10}{'min left':>10}  status")
    total_left = 0.0
    unknown = 0
    for dataset in DATASETS:
        for i, model in enumerate(MODELS):
            done = times[dataset][i * FOLDS:(i + 1) * FOLDS]
            per_fold = sum(done) / len(done) / 60 if done else None
            if len(done) == FOLDS:
                status = "finished"
            elif done:
                status = "running"
            else:
                status = "no fold finished yet"
            if status == "finished":
                left = 0.0
            elif per_fold is not None:
                left = per_fold * (FOLDS - len(done))
            else:
                left = None
            if left is None:
                unknown += 1
            else:
                total_left += left
            print(f"{dataset + ' ' + model:<31}{len(done):>5}/{FOLDS}"
                  f"{per_fold if per_fold is not None else float('nan'):>10.1f}"
                  f"{left if left is not None else float('nan'):>10.1f}  {status}")
    note = f" + {unknown} job(s) with no timing yet" if unknown else ""
    print(f"\nKnown time left, if run one after another: {total_left:.0f} min{note}")
    print("Jobs run in two parallel sessions, so the real finish is sooner.")


if __name__ == "__main__":
    main()
