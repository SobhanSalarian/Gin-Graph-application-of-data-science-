"""
Structural checks for a TU-format dataset, run before spending GPU time on it:
density, share of cliques, distinct degree sequences, and the accuracy of a
classifier that sees only the node count. For the node-count classifier, the
in-sample majority lookup is an upper bound, and the 10-fold figure is the
fair estimate.

Usage:
    python check_dataset.py path/to/DATASET.txt [more paths ...]
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


def read_tu(path):
    with open(path) as f:
        lines = f.read().split("\n")
    graphs = []
    pos = 1
    for _ in range(int(lines[0])):
        n, label = map(int, lines[pos].split()[:2])
        pos += 1
        degrees = []
        for _ in range(n):
            degrees.append(int(lines[pos].split()[1]))
            pos += 1
        graphs.append({"n": n, "label": label, "degrees": degrees})
    return graphs


def node_count_accuracy(train, test):
    by_n = defaultdict(Counter)
    for g in train:
        by_n[g["n"]][g["label"]] += 1
    fallback = Counter(g["label"] for g in train).most_common(1)[0][0]
    hits = 0
    for g in test:
        pred = by_n[g["n"]].most_common(1)[0][0] if g["n"] in by_n else fallback
        hits += pred == g["label"]
    return hits / len(test)


def report(path):
    graphs = read_tu(path)
    labels = [g["label"] for g in graphs]
    counts = Counter(labels)
    densities = []
    cliques = 0
    for g in graphs:
        n, m = g["n"], sum(g["degrees"]) / 2
        possible = n * (n - 1) / 2
        densities.append(m / possible if possible else 0)
        cliques += m == possible
    seqs = {tuple(sorted(g["degrees"])) for g in graphs}
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    cv = [node_count_accuracy([graphs[i] for i in tr], [graphs[i] for i in te])
          for tr, te in folds.split(graphs, labels)]
    sizes = [g["n"] for g in graphs]

    print(f"\n{Path(path).stem}")
    print(f"  graphs {len(graphs)}, classes {dict(sorted(counts.items()))}")
    print(f"  nodes avg {sum(sizes) / len(sizes):.1f} (min {min(sizes)}, max {max(sizes)})")
    print(f"  avg density {sum(densities) / len(densities):.3f}")
    print(f"  cliques {100 * cliques / len(graphs):.1f}%")
    print(f"  distinct degree sequences {len(seqs)}")
    print(f"  majority class {100 * max(counts.values()) / len(graphs):.1f}%")
    print(f"  node-count only: in-sample ceiling {100 * node_count_accuracy(graphs, graphs):.1f}%, "
          f"10-fold {100 * sum(cv) / len(cv):.1f}%")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
