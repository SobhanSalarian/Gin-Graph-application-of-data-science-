"""
Builds actor ego-network datasets (IMDBBINARY-EGO2016, IMDBMULTI-EGO2016) from the films
already scraped by build_tmdb_dataset.py. No TMDb calls are needed: the input is
the *_metadata.json written by that script.

This replaces the one-graph-per-film construction, where every graph is a clique
because all of a film's cast co-appear in that film. It follows the recipe of
Yanardag & Vishwanathan ("Deep Graph Kernels", KDD 2015):

  - one collaboration graph per genre: actors are nodes, and two actors are
    linked if they co-appear in a sampled film of that genre
  - one graph per (actor, genre): the actor's ego-network in that collaboration
    graph, i.e. the actor, their co-stars, and all links among those co-stars
  - graph label = the genre

Actors with a single film in a genre are skipped, since their ego-network is
just that film's cast and so is a clique.

Actors are identified by TMDb id.

Usage:
    python build_ego_networks.py --mode binary
    python build_ego_networks.py --mode multi
"""
import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from build_tmdb_dataset import write_tu_format

OUT_DIR = Path(__file__).parent.parent / "processed"

SOURCE_NAMES = {
    "binary": "IMDBBINARY-2016",
    "multi": "IMDBMULTI-2016",
}
DATASET_NAMES = {
    "binary": "IMDBBINARY-EGO2016",
    "multi": "IMDBMULTI-EGO2016",
}


def build_ego_networks(films, min_films):
    """Returns one graph per actor with at least min_films films in this genre."""
    neighbors = defaultdict(set)
    film_count = defaultdict(int)
    for film in films:
        cast = set(film["actor_ids"])
        for actor in cast:
            film_count[actor] += 1
        for a, b in itertools.combinations(cast, 2):
            neighbors[a].add(b)
            neighbors[b].add(a)

    graphs = []
    for ego in sorted(film_count):
        if film_count[ego] < min_films:
            continue
        # ego goes first so it is always node 0
        nodes = [ego] + sorted(neighbors[ego])
        idx = {actor: i for i, actor in enumerate(nodes)}
        edges = []
        for a, b in itertools.combinations(nodes, 2):
            if b in neighbors[a]:
                edges.append((idx[a], idx[b]))
        graphs.append({
            "n_nodes": len(nodes),
            "edges": edges,
            "ego": ego,
            "n_films": film_count[ego],
        })
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "multi"], required=True)
    ap.add_argument("--graphs-per-genre", type=int, default=500)
    ap.add_argument("--min-films", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    source = SOURCE_NAMES[args.mode]
    meta_in = json.loads((OUT_DIR / source / f"{source}_metadata.json").read_text())
    label_order = meta_in["label_order"]

    rng = random.Random(args.seed)
    graphs = []
    available = {}
    for genre in label_order:
        films = [g for g in meta_in["graphs"] if g["label"] == genre]
        genre_graphs = build_ego_networks(films, args.min_films)
        available[genre] = len(genre_graphs)
        rng.shuffle(genre_graphs)
        for g in genre_graphs[:args.graphs_per_genre]:
            g["label"] = genre
            graphs.append(g)

    dataset_name = DATASET_NAMES[args.mode]
    out_dir = OUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tu_format(graphs, label_order, out_dir / f"{dataset_name}.txt")

    meta = {
        "dataset_name": dataset_name,
        "source_dataset": source,
        "label_order": label_order,
        "min_films": args.min_films,
        "seed": args.seed,
        "n_graphs": len(graphs),
        "egos_available_per_genre": available,
        "avg_nodes": sum(g["n_nodes"] for g in graphs) / len(graphs),
        "avg_edges": sum(len(g["edges"]) for g in graphs) / len(graphs),
        "label_counts": {name: sum(1 for g in graphs if g["label"] == name)
                         for name in label_order},
        "graphs": [
            {"ego": g["ego"], "label": g["label"], "n_films": g["n_films"],
             "n_nodes": g["n_nodes"]}
            for g in graphs
        ],
    }
    (out_dir / f"{dataset_name}_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(graphs)} graphs to {out_dir}")
    print(f"Egos available per genre: {available}")
    print(f"Avg nodes: {meta['avg_nodes']:.1f}, avg edges: {meta['avg_edges']:.1f}")
    print(f"Label counts: {meta['label_counts']}")


if __name__ == "__main__":
    main()
