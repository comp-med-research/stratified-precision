"""
Hetionet knowledge graph — local JSON, X-hop subgraph extraction, graph features.

Hetionet encodes biology as a heterogeneous network:
  nodes: Gene, Disease, Compound, Anatomy, Pathway, Side Effect, ...
  edges: Gene-associates-Disease, Gene-expresses-Anatomy, ...

The key insight for dynamic objectives: a 2-hop neighbourhood around
(BACE1, Alzheimer's) contains ~200 nodes that are almost entirely
neuronal anatomy + amyloid pathway + failed compounds.
A 2-hop neighbourhood around (CFTR, Cystic Fibrosis) has almost no
anatomy overlap and is dominated by CFTR-interacting proteins.
These different neighbourhood shapes become different Pareto objectives.

Download Hetionet JSON once:
    python -m stratified_precision.kg.hetionet download
"""

from __future__ import annotations

import bz2
import json
import os
from pathlib import Path
from typing import Optional
import urllib.request

import networkx as nx
import numpy as np
import pandas as pd

HETIONET_URL = (
    "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"
)
DEFAULT_CACHE = Path.home() / ".cache" / "stratified_precision" / "hetionet-v1.0.json"


# ---------------------------------------------------------------------------
# Graph loading & caching
# ---------------------------------------------------------------------------

def load_hetionet(cache_path: Path = DEFAULT_CACHE) -> nx.MultiDiGraph:
    """Load Hetionet from local cache, downloading if absent."""
    if not cache_path.exists():
        download_hetionet(cache_path)
    return _parse_hetionet_json(cache_path)


def download_hetionet(dest: Path = DEFAULT_CACHE):
    dest.parent.mkdir(parents=True, exist_ok=True)
    bz2_path = dest.with_suffix(".json.bz2")

    print(f"[Hetionet] Downloading to {bz2_path} (~200MB)...")
    urllib.request.urlretrieve(HETIONET_URL, bz2_path)

    print("[Hetionet] Decompressing...")
    with bz2.open(bz2_path, "rb") as f_in, open(dest, "wb") as f_out:
        f_out.write(f_in.read())
    bz2_path.unlink()
    print(f"[Hetionet] Saved to {dest}")


def _parse_hetionet_json(path: Path) -> nx.MultiDiGraph:
    with open(path) as f:
        raw = json.load(f)

    net = raw["network"]
    G = nx.MultiDiGraph()

    for node in net["nodes"]:
        node_id = (node["kind"], node["identifier"])
        G.add_node(node_id, kind=node["kind"], name=node["name"], identifier=node["identifier"])

    for edge in net["edges"]:
        src = tuple(edge["source_id"])
        tgt = tuple(edge["target_id"])
        G.add_edge(src, tgt, kind=edge["kind"])

    print(f"[Hetionet] Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# Node resolution
# ---------------------------------------------------------------------------

def find_gene_node(G: nx.MultiDiGraph, gene_symbol: str) -> Optional[tuple]:
    """Return the Hetionet node ID for a gene symbol, or None."""
    for node_id, data in G.nodes(data=True):
        if data.get("kind") == "Gene" and data.get("name", "").upper() == gene_symbol.upper():
            return node_id
    return None


def find_disease_node(G: nx.MultiDiGraph, disease_name: str) -> Optional[tuple]:
    """Fuzzy match a disease name to a Hetionet Disease node."""
    term = disease_name.lower()
    best_node = None
    best_score = 0
    for node_id, data in G.nodes(data=True):
        if data.get("kind") != "Disease":
            continue
        node_name = data.get("name", "").lower()
        # Simple overlap score
        overlap = sum(1 for w in term.split() if w in node_name)
        if overlap > best_score:
            best_score = overlap
            best_node = node_id
    return best_node


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_subgraph(
    G: nx.MultiDiGraph,
    gene_node: tuple,
    disease_node: Optional[tuple] = None,
    n_hops: int = 2,
    max_nodes: int = 500,
) -> nx.MultiDiGraph:
    """
    Extract an X-hop ego-network around the target gene, optionally
    filtered to paths that pass through (or near) the disease node.

    Returns a subgraph of G — same node/edge attributes.
    """
    # BFS up to n_hops from the gene node
    ego = nx.ego_graph(G.to_undirected(as_view=True), gene_node, radius=n_hops)

    # If disease node is known and reachable, keep only nodes on or near
    # paths between the gene and the disease to focus the neighbourhood.
    if disease_node and disease_node in ego:
        try:
            path_nodes = set(nx.shortest_path(ego, gene_node, disease_node))
            # Also keep 1-hop neighbours of path nodes
            extended = set(path_nodes)
            for n in path_nodes:
                extended.update(ego.neighbors(n))
            # But don't expand past max_nodes
            if len(extended) <= max_nodes:
                ego = ego.subgraph(extended)
        except nx.NetworkXNoPath:
            pass  # fall back to full ego-network

    # Trim to max_nodes if still too large (keep highest-degree nodes)
    if ego.number_of_nodes() > max_nodes:
        top = sorted(ego.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        ego = ego.subgraph([n for n, _ in top])

    return G.subgraph(ego.nodes())


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_graph_features(
    G: nx.MultiDiGraph,
    subgraph: nx.MultiDiGraph,
    gene_node: tuple,
    disease_node: Optional[tuple] = None,
) -> dict[str, float]:
    """
    Extract numeric features from a target's Hetionet neighbourhood.
    These become candidate Pareto objectives — the specific features vary
    by disease context because the neighbourhood composition varies.
    """
    ug = subgraph.to_undirected()
    features: dict[str, float] = {}

    # --- Topological features (always present) ---
    features["degree_in_subgraph"] = float(ug.degree(gene_node)) if gene_node in ug else 0.0
    features["subgraph_size"] = float(ug.number_of_nodes())
    features["subgraph_density"] = float(nx.density(ug))

    if disease_node and disease_node in ug:
        try:
            path_len = nx.shortest_path_length(ug, gene_node, disease_node)
            features["shortest_path_to_disease"] = float(path_len)
        except nx.NetworkXNoPath:
            features["shortest_path_to_disease"] = 99.0
    else:
        features["shortest_path_to_disease"] = 99.0

    # --- Neighbourhood composition features (disease-context-dependent) ---
    kind_counts: dict[str, int] = {}
    for node_id, data in subgraph.nodes(data=True):
        k = data.get("kind", "Unknown")
        kind_counts[k] = kind_counts.get(k, 0) + 1

    for kind, count in kind_counts.items():
        features[f"n_{kind.lower()}_neighbors"] = float(count)

    # --- Edge type diversity ---
    edge_kinds: set[str] = {data.get("kind", "") for _, _, data in subgraph.edges(data=True)}
    features["n_edge_types"] = float(len(edge_kinds))

    # --- Compound neighbours = existing drug competition / evidence ---
    features["n_compound_neighbors"] = float(kind_counts.get("Compound", 0))

    # --- Anatomy breadth = proxy for off-target expression risk ---
    features["n_anatomy_neighbors"] = float(kind_counts.get("Anatomy", 0))

    # --- Pathway coverage ---
    features["n_pathway_neighbors"] = float(kind_counts.get("Biological Process", 0)
                                             + kind_counts.get("Pathway", 0))

    # --- Normalise path distance (inverted: shorter = better target evidence) ---
    features["path_to_disease_score"] = 1.0 / max(features["shortest_path_to_disease"], 1.0)

    return features


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_hetionet()
