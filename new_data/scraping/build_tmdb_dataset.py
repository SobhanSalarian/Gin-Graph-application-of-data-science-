"""
Builds new IMDB-BINARY / IMDB-MULTI-style datasets from the TMDb API, mirroring
the construction used by Yanardag & Vishwanathan ("Deep Graph Kernels", KDD
2015) for the original IMDB-BINARY/MULTI benchmarks that the GIN paper uses:

  - nodes = actors
  - edge between two actors if they co-starred in a sampled movie of that genre
  - one graph per movie (that movie's cast, plus any extra edges from actors
    also co-starring elsewhere in our sampled pool for that genre)
  - graph label = the movie's genre

Movies are restricted to release date >= 2016-01-01, safely after the 2015
paper's data collection, so this is guaranteed to be new, non-overlapping data.

Output: TU-Dataset text format (matching replication/official-repo/util.py's
load_data()), so the result drops straight into the existing GIN pipeline
with zero code changes - just point --dataset at it. Also writes a metadata
JSON (movie titles / actor names per graph) for later EDA / report writeup.

Requires a free TMDb API key: https://www.themoviedb.org/settings/api
Pass it via --api-key or the TMDB_API_KEY environment variable.

Usage:
    python build_tmdb_dataset.py --mode binary --api-key YOUR_KEY
    python build_tmdb_dataset.py --mode multi  --api-key YOUR_KEY
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://api.themoviedb.org/3"
OUT_DIR = Path(__file__).parent.parent / "processed"

GENRE_SETS = {
    "binary": ["Action", "Romance"],
    "multi": ["Comedy", "Romance", "Science Fiction"],
}
DATASET_NAMES = {
    "binary": "IMDBBINARY-2016",
    "multi": "IMDBMULTI-2016",
}


class TMDbClient:
    def __init__(self, api_key, min_delay=0.15):
        self.api_key = api_key
        self.min_delay = min_delay
        self._last_call = 0.0
        self.session = requests.Session()

    def get(self, path, **params):
        # simple politeness throttle; well under TMDb's ~40 req/s soft limit
        elapsed = time.time() - self._last_call
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        params["api_key"] = self.api_key
        resp = self.session.get(f"{API_BASE}{path}", params=params, timeout=15)
        self._last_call = time.time()
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "2"))
            print(f"  rate limited, sleeping {retry_after}s", file=sys.stderr)
            time.sleep(retry_after)
            return self.get(path, **params)
        resp.raise_for_status()
        return resp.json()

    def genre_map(self):
        data = self.get("/genre/movie/list")
        return {g["name"]: g["id"] for g in data["genres"]}

    def discover_movies(self, genre_id, page, min_date="2016-01-01"):
        return self.get(
            "/discover/movie",
            with_genres=genre_id,
            sort_by="popularity.desc",
            include_adult="false",
            # TMDb silently ignores misspelled filters, so these need the dot
            **{"vote_count.gte": 20, "primary_release_date.gte": min_date},
            page=page,
        )

    def credits(self, movie_id):
        return self.get(f"/movie/{movie_id}/credits")


def collect_movies_for_genre(client, genre_name, all_genre_ids, target_count,
                              cast_size, min_date):
    """Pull movies tagged with exactly one of our target genres (this one),
    to avoid a movie qualifying for two different labels."""
    target_id = all_genre_ids[genre_name]
    other_target_ids = {gid for name, gid in all_genre_ids.items()
                         if name in GENRE_SET_ALL and name != genre_name}

    movies = []
    seen_ids = set()
    page = 1
    while len(movies) < target_count and page <= 500:
        data = client.discover_movies(target_id, page, min_date)
        results = data.get("results", [])
        if not results:
            break
        for m in results:
            if m["id"] in seen_ids:
                continue
            seen_ids.add(m["id"])
            genre_ids = set(m.get("genre_ids", []))
            if genre_ids & other_target_ids:
                continue  # ambiguous: also tagged with another target genre
            movies.append(m)
            if len(movies) >= target_count:
                break
        page += 1
        if page > data.get("total_pages", 1):
            break

    print(f"  [{genre_name}] found {len(movies)} candidate movies "
          f"(wanted {target_count})", file=sys.stderr)

    # dates are ISO strings, so they compare correctly; a missing date also fails
    too_old = [m for m in movies if (m.get("release_date") or "") < min_date]
    if too_old:
        sys.exit(f"[{genre_name}] date filter failed: {len(too_old)} movies before "
                 f"{min_date}, e.g. {too_old[0].get('title')}")

    # fetch cast for each, keep only movies with enough billed actors
    movie_records = []
    for m in movies:
        try:
            cred = client.credits(m["id"])
        except requests.HTTPError as e:
            print(f"  skip {m['id']} ({m.get('title')}): {e}", file=sys.stderr)
            continue
        cast = sorted(cred.get("cast", []), key=lambda c: c.get("order", 999))
        cast = cast[:cast_size]
        if len(cast) < 5:
            continue
        movie_records.append({
            "movie_id": m["id"],
            "title": m.get("title"),
            "release_date": m.get("release_date"),
            "genre": genre_name,
            "actors": [{"id": c["id"], "name": c["name"]} for c in cast],
        })
    return movie_records


def build_graphs(movie_records):
    """One graph per movie. Edge between two actors if they co-appear in ANY
    sampled movie of this genre (not just this one movie) - this is what
    turns a trivial per-movie clique into a genuine collaboration ego-network,
    mirroring the structure of the original IMDB-BINARY/MULTI datasets."""
    # global collaboration edge set within this genre's sampled movies
    collab_edges = set()
    for rec in movie_records:
        actor_ids = [a["id"] for a in rec["actors"]]
        for a, b in itertools.combinations(sorted(actor_ids), 2):
            collab_edges.add((a, b))

    graphs = []
    for rec in movie_records:
        actor_ids = [a["id"] for a in rec["actors"]]
        idx = {aid: i for i, aid in enumerate(actor_ids)}
        edges = []
        for a, b in itertools.combinations(sorted(actor_ids), 2):
            if (a, b) in collab_edges:
                edges.append((idx[a], idx[b]))
        graphs.append({
            "n_nodes": len(actor_ids),
            "label": rec["genre"],
            "edges": edges,
            "movie_id": rec["movie_id"],
            "title": rec["title"],
            "release_date": rec["release_date"],
            "actor_ids": actor_ids,
            "actor_names": [a["name"] for a in rec["actors"]],
        })
    return graphs


def write_tu_format(graphs, label_order, out_path):
    label_to_int = {name: i for i, name in enumerate(label_order)}
    lines = [str(len(graphs))]
    for g in graphs:
        lines.append(f"{g['n_nodes']} {label_to_int[g['label']]}")
        neighbors = [[] for _ in range(g["n_nodes"])]
        for a, b in g["edges"]:
            neighbors[a].append(b)
            neighbors[b].append(a)
        for node in range(g["n_nodes"]):
            nbrs = neighbors[node]
            # tag=0 (no node labels available; use --degree_as_tag at train time)
            lines.append(f"0 {len(nbrs)} " + " ".join(str(n) for n in nbrs))
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "multi"], required=True)
    ap.add_argument("--api-key", default=os.environ.get("TMDB_API_KEY"))
    ap.add_argument("--movies-per-genre", type=int, default=500)
    ap.add_argument("--cast-size", type=int, default=20)
    ap.add_argument("--min-date", default="2016-01-01")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("No TMDb API key. Pass --api-key or set TMDB_API_KEY.")

    global GENRE_SET_ALL
    GENRE_SET_ALL = set(GENRE_SETS[args.mode])

    client = TMDbClient(args.api_key)
    all_genre_ids = client.genre_map()
    for name in GENRE_SET_ALL:
        if name not in all_genre_ids:
            sys.exit(f"Genre '{name}' not found in TMDb genre list: {list(all_genre_ids)}")

    all_records = []
    for genre_name in GENRE_SETS[args.mode]:
        recs = collect_movies_for_genre(
            client, genre_name, all_genre_ids,
            args.movies_per_genre, args.cast_size, args.min_date,
        )
        all_records.extend(recs)

    graphs = build_graphs(all_records)

    dataset_name = DATASET_NAMES[args.mode]
    out_dir = OUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tu_format(graphs, GENRE_SETS[args.mode], out_dir / f"{dataset_name}.txt")

    meta = {
        "dataset_name": dataset_name,
        "label_order": GENRE_SETS[args.mode],
        "min_release_date": args.min_date,
        "cast_size": args.cast_size,
        "n_graphs": len(graphs),
        "avg_nodes": sum(g["n_nodes"] for g in graphs) / len(graphs) if graphs else 0,
        "avg_edges": sum(len(g["edges"]) for g in graphs) / len(graphs) if graphs else 0,
        "label_counts": {name: sum(1 for g in graphs if g["label"] == name)
                          for name in GENRE_SETS[args.mode]},
        "graphs": [
            {"movie_id": g["movie_id"], "title": g["title"], "label": g["label"],
             "release_date": g["release_date"], "n_nodes": g["n_nodes"],
             "actor_ids": g["actor_ids"], "actor_names": g["actor_names"]}
            for g in graphs
        ],
    }
    (out_dir / f"{dataset_name}_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {len(graphs)} graphs to {out_dir}")
    print(f"Avg nodes: {meta['avg_nodes']:.1f}, avg edges: {meta['avg_edges']:.1f}")
    print(f"Label counts: {meta['label_counts']}")
    print(f"Release dates: {min(g['release_date'] for g in graphs)} to "
          f"{max(g['release_date'] for g in graphs)}")


if __name__ == "__main__":
    main()
