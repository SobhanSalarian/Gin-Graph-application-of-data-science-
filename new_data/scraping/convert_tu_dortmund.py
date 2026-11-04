"""
Converts a dataset from the TU Dortmund collection (https://chrsmrrs.github.io/datasets/)
into the single-file format read by the official GIN code's util.load_data().

TU Dortmund files, for a dataset DS:
  DS_A.txt               one "node_id, node_id" line per directed edge (1-indexed)
  DS_graph_indicator.txt line i = graph id of node i
  DS_graph_labels.txt    line i = class of graph i
  DS_node_labels.txt     line i = label of node i

Output: count of graphs, then for each graph "n_nodes label" followed by one
"tag degree neighbor1 neighbor2 ..." line per node.

Usage:
    python convert_tu_dortmund.py path/to/ENZYMES path/to/output/ENZYMES.txt
"""
import sys
from collections import defaultdict
from pathlib import Path


def read_ints(path):
    return [int(line) for line in path.read_text().split()]


def convert(src, out_path):
    name = src.name
    graph_of_node = read_ints(src / f"{name}_graph_indicator.txt")
    graph_labels = read_ints(src / f"{name}_graph_labels.txt")
    node_labels = read_ints(src / f"{name}_node_labels.txt")

    neighbors = defaultdict(set)
    for line in (src / f"{name}_A.txt").read_text().splitlines():
        a, b = (int(x) for x in line.split(","))
        if a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)

    nodes_of_graph = defaultdict(list)
    for node, graph in enumerate(graph_of_node, start=1):
        nodes_of_graph[graph].append(node)

    lines = [str(len(graph_labels))]
    for graph, label in enumerate(graph_labels, start=1):
        nodes = nodes_of_graph[graph]
        local = {node: i for i, node in enumerate(nodes)}
        lines.append(f"{len(nodes)} {label}")
        for node in nodes:
            nbrs = sorted(local[n] for n in neighbors[node])
            lines.append(" ".join(str(x) for x in [node_labels[node - 1], len(nbrs)] + nbrs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(graph_labels)} graphs to {out_path}")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
