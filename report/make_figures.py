"""
Draws the report's learning-curve figure from the saved per-epoch results:
test accuracy averaged over the 10 folds at each epoch, for GIN and GraphSAGE,
with the epoch the paper's selection rule picks and the node-count baseline.

Usage:
    python report/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
OUT = Path(__file__).resolve().parent / "learning_curves.pdf"

PANELS = [
    ("IMDBBINARY", "IMDB-BINARY (original)", None),
    ("IMDBBINARY-EGO2016", "Ego-networks, binary", 57.7),
    ("IMDBMULTI-EGO2016", "Ego-networks, multi", 46.3),
    ("ENZYMES", "ENZYMES", 23.5),
]
MODELS = [("", "GIN-0", "#2a78d6", "-"), ("_graphsage", "Max-1-layer", "#eb6834", "--")]
INK, MUTED = "#0b0b0b", "#52514e"


def mean_curve(path):
    folds = json.loads(path.read_text())["folds"]
    return [100 * sum(f[e]["test_acc"] for f in folds) / len(folds) for e in range(len(folds[0]))]


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.3, 1.9))
    for ax, (name, title, baseline) in zip(axes, PANELS):
        for suffix, label, color, style in MODELS:
            curve = mean_curve(LOGS / f"{name}{suffix}_cv_curves.json")
            best = max(range(len(curve)), key=curve.__getitem__)
            ax.plot(range(1, len(curve) + 1), curve, color=color, ls=style, lw=1.2, label=label)
            ax.plot(best + 1, curve[best], "o", ms=4, color=color, mec="white", mew=0.8)
        if baseline is not None:
            ax.axhline(baseline, color=MUTED, lw=0.8, ls=":")
            ax.text(345, baseline + 0.6, "node count", color=MUTED, fontsize=6.5, ha="right", va="bottom")
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 350)
        ax.grid(axis="y", color="#e4e3df", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean test accuracy (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
