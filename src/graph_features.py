"""
Speech-graph features: represent a transcript as a directed word-adjacency
graph (each word = node, edges connect consecutive spoken words) and extract
graph-theoretic metrics. Captures narrative structure/coherence, which
lexical and embedding features don't directly measure.

Reference technique: Mota et al., used originally for thought disorder,
later applied to Alzheimer's/dementia speech (narrative fragmentation,
word-level repetition/perseveration show up as graph-shape differences).
"""
import networkx as nx
import numpy as np


def build_speech_graph(text: str) -> nx.DiGraph:
    words = text.split()
    G = nx.DiGraph()
    for w in words:
        G.add_node(w)
    for i in range(len(words) - 1):
        u, v = words[i], words[i + 1]
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def graph_features(text: str) -> list:
    """Returns a fixed-length vector of graph metrics for one transcript."""
    G = build_speech_graph(text)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return [0.0] * 7

    avg_degree = n_edges / n_nodes if n_nodes else 0.0

    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    n_components = len(components)
    largest_cc_size = max(len(c) for c in components) if components else 0
    largest_cc_fraction = largest_cc_size / n_nodes if n_nodes else 0.0

    # Loops: edges where the same word directly repeats (self-loop-like repetition
    # after cleaning, or a 2-cycle A->B->A), a proxy for perseveration
    n_repeated_edges = sum(1 for _, _, d in G.edges(data=True) if d["weight"] > 1)

    # Diameter of the largest weakly connected component (undirected approx)
    if largest_cc_size > 1:
        largest_sub = undirected.subgraph(max(components, key=len))
        try:
            diameter = nx.diameter(largest_sub)
        except nx.NetworkXError:
            diameter = 0
    else:
        diameter = 0

    return [
        n_nodes,
        n_edges,
        avg_degree,
        n_components,
        largest_cc_fraction,
        n_repeated_edges,
        diameter,
    ]


GRAPH_FEATURE_NAMES = [
    "graph_n_nodes", "graph_n_edges", "graph_avg_degree",
    "graph_n_components", "graph_largest_cc_fraction",
    "graph_n_repeated_edges", "graph_diameter",
]


def graph_features_batch(texts) -> np.ndarray:
    return np.vstack([graph_features(t) for t in texts])
