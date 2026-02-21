from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str) -> List[float]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return [float(x) for x in values]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_unit_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("Cannot average an empty vector collection.")
    arr = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("Averaged query vector has zero norm.")
    return arr / norm


def load_graph_payload(graph_path: str) -> Tuple[List[str], List[Tuple[str, str, float]], dict]:
    payload = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    nodes_obj = payload.get("nodes", {})
    edges_obj = payload.get("edges", [])
    if not isinstance(nodes_obj, dict):
        raise ValueError("Graph JSON must contain 'nodes' as an object.")
    if not isinstance(edges_obj, list):
        raise ValueError("Graph JSON must contain 'edges' as a list.")

    node_ids = [str(k) for k in nodes_obj.keys()]
    edges: List[Tuple[str, str, float]] = []
    for item in edges_obj:
        if not (isinstance(item, list) and len(item) == 3):
            continue
        src, dst, meta = item
        if not isinstance(meta, dict):
            meta = {}
        w = float(meta.get("weight", 1.0))
        edges.append((str(src), str(dst), w))
    return node_ids, edges, payload


def _load_vectors_as_dict(vec_payload: dict) -> Dict[str, np.ndarray]:
    vectors = vec_payload.get("vectors")
    if isinstance(vectors, dict):
        out: Dict[str, np.ndarray] = {}
        for k, v in vectors.items():
            out[str(k)] = np.asarray(v, dtype=float)
        return out
    if isinstance(vectors, list):
        return {str(i): np.asarray(v, dtype=float) for i, v in enumerate(vectors)}
    raise ValueError("Vectors JSON must have 'vectors' as a list or dict.")


def load_node_embeddings(vectors_path: str, node_ids: Iterable[str]) -> Dict[str, np.ndarray]:
    vec_payload = json.loads(Path(vectors_path).read_text(encoding="utf-8"))
    raw = _load_vectors_as_dict(vec_payload)
    embeddings: Dict[str, np.ndarray] = {}
    missing: List[str] = []

    for node_id in node_ids:
        key = str(node_id)
        if key in raw:
            embeddings[key] = raw[key]
        else:
            missing.append(key)

    if missing:
        preview = ",".join(missing[:10])
        raise ValueError(f"Missing vectors for {len(missing)} nodes (examples: {preview}).")
    return embeddings


def build_query_vector(query_vertices: Sequence[str], embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    missing = [v for v in query_vertices if v not in embeddings]
    if missing:
        raise ValueError(f"Query vertices missing from embeddings: {missing}")
    q_vecs = [embeddings[v] for v in query_vertices]
    return mean_unit_vector(q_vecs)


def rank_nodes_by_query_similarity(
    embeddings: Dict[str, np.ndarray],
    query_vector: np.ndarray,
) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    for node_id, vec in embeddings.items():
        scores.append((node_id, cosine(query_vector, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def build_undirected_graph(node_ids: Sequence[str], edges: Sequence[Tuple[str, str, float]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for src, dst, w in edges:
        if src == dst:
            continue
        if g.has_edge(src, dst):
            g[src][dst]["weight"] = max(float(g[src][dst]["weight"]), float(w))
        else:
            g.add_edge(src, dst, weight=float(w))
    return g


def weighted_graph_distance_rank(
    g: nx.Graph,
    seed_nodes: Sequence[str],
    embeddings: Dict[str, np.ndarray],
    query_vector: np.ndarray,
    max_distance: float,
    min_weight_eps: float = 1e-9,
) -> List[Tuple[str, float, float]]:
    dist_lists: Dict[str, List[float]] = {}

    def edge_cost(u: str, v: str, attrs: dict) -> float:
        w = float(attrs.get("weight", 1.0))
        w = max(w, min_weight_eps)
        # Larger weight => smaller distance.
        return 1.0 / w

    for s in seed_nodes:
        lengths = nx.single_source_dijkstra_path_length(g, s, weight=edge_cost)
        for node, d in lengths.items():
            dist_lists.setdefault(node, []).append(d)

    seed_set = set(seed_nodes)
    ranked: List[Tuple[str, float, float]] = []
    seed_count = len(seed_nodes)
    for node, dists in dist_lists.items():
        if node in seed_set:
            continue
        # Use average weighted shortest-path distance over all seeds.
        if len(dists) != seed_count:
            continue
        avg_d = float(sum(dists) / seed_count)
        if avg_d > max_distance:
            continue
        ranked.append((node, avg_d, cosine(query_vector, embeddings[node])))
    ranked.sort(key=lambda x: (x[1], -x[2], x[0]))
    return ranked


def effective_resistance(node_ids: Sequence[str], edges: Sequence[Tuple[str, str, float]]) -> Tuple[Dict[str, int], np.ndarray]:
    idx = {node: i for i, node in enumerate(node_ids)}
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for src, dst, weight in edges:
        if src not in idx or dst not in idx:
            continue
        i = idx[src]
        j = idx[dst]
        w = float(weight)
        rows.append(i)
        cols.append(j)
        vals.append(w)

    n = len(node_ids)
    a = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    a_und = a + a.T
    a_und.setdiag(0)
    a_und.eliminate_zeros()

    deg = np.asarray(a_und.sum(axis=1)).ravel()
    lap = sp.diags(deg) - a_und
    lap_pinv = np.linalg.pinv(lap.toarray())
    diag = np.diag(lap_pinv)
    r = diag[:, None] + diag[None, :] - 2.0 * lap_pinv
    return idx, r
