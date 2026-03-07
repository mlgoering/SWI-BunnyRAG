from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import plotly.io as pio
from plotly import graph_objects as go
from common import (
    DEFAULT_COSINE_BETA_ALPHA,
    DEFAULT_COSINE_BETA_CLIP_MAX,
    DEFAULT_COSINE_BETA_CLIP_MIN,
    DEFAULT_COSINE_BETA_KAPPA,
    DEFAULT_COSINE_BETA_OFFSET,
    DEFAULT_COSINE_BETA_SCALE,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    DEFAULT_QUERY_THETA0,
    DEFAULT_QUERY_THETA1,
    DEFAULT_QUERY_THETA2,
    DEFAULT_QUERY_THETA3,
    DEFAULT_TARGET_MEAN_WEIGHT,
    load_graph_payload as load_graph_payload_common,
    reweight_edges_by_mode,
)


VARIANT_SEED_OFFSETS = {"unit": 0, "random": 17, "cosine_beta": 37, "query_aware": 59}


def load_graph(graph_path: str) -> nx.Graph:
    payload = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    nodes_obj = payload.get("nodes", {})
    edges_obj = payload.get("edges", [])
    if not isinstance(nodes_obj, dict) or not isinstance(edges_obj, list):
        raise ValueError("Graph JSON must contain 'nodes' (dict) and 'edges' (list).")

    g = nx.Graph()
    for node_id in nodes_obj.keys():
        g.add_node(str(node_id))

    for item in edges_obj:
        if not (isinstance(item, list) and len(item) == 3):
            continue
        src, dst, meta = item
        src = str(src)
        dst = str(dst)
        if src == dst:
            continue
        weight = 1.0
        if isinstance(meta, dict):
            weight = float(meta.get("weight", 1.0))
        if g.has_edge(src, dst):
            g[src][dst]["weight"] = max(float(g[src][dst]["weight"]), weight)
        else:
            g.add_edge(src, dst, weight=weight)
    return g


def _graph_from_weighted_edges(
    node_ids: Sequence[str],
    edges: Sequence[Tuple[str, str, float]],
) -> nx.Graph:
    g = nx.Graph()
    for node_id in node_ids:
        g.add_node(str(node_id))
    for src, dst, weight in edges:
        src = str(src)
        dst = str(dst)
        if src == dst:
            continue
        w = float(weight)
        if g.has_edge(src, dst):
            g[src][dst]["weight"] = max(float(g[src][dst]["weight"]), w)
        else:
            g.add_edge(src, dst, weight=w)
    return g


def _build_variant_graphs(
    *,
    node_ids: Sequence[str],
    base_edges: Sequence[Tuple[str, str, float]],
    embeddings: Mapping[str, np.ndarray],
    query_vector: np.ndarray | None,
    variant_order: Sequence[str],
    meta: Mapping[str, object],
) -> Dict[str, nx.Graph]:
    raw_seed = meta.get("edge_weight_seed")
    if raw_seed is None:
        raw_seed = meta.get("query_seed", 0)
    base_seed = int(raw_seed) if raw_seed is not None else 0
    target_mean_weight = float(meta.get("target_mean_weight", DEFAULT_TARGET_MEAN_WEIGHT))
    min_weight = float(meta.get("min_weight", DEFAULT_MIN_WEIGHT))
    max_weight = float(meta.get("max_weight", DEFAULT_MAX_WEIGHT))
    cosine_beta_alpha = float(meta.get("cosine_beta_alpha", DEFAULT_COSINE_BETA_ALPHA))
    cosine_beta_kappa = float(meta.get("cosine_beta_kappa", DEFAULT_COSINE_BETA_KAPPA))
    cosine_beta_offset = float(meta.get("cosine_beta_offset", DEFAULT_COSINE_BETA_OFFSET))
    cosine_beta_scale = float(meta.get("cosine_beta_scale", DEFAULT_COSINE_BETA_SCALE))
    cosine_beta_clip_min = float(meta.get("cosine_beta_clip_min", DEFAULT_COSINE_BETA_CLIP_MIN))
    cosine_beta_clip_max = float(meta.get("cosine_beta_clip_max", DEFAULT_COSINE_BETA_CLIP_MAX))
    query_theta0 = float(meta.get("query_theta0", DEFAULT_QUERY_THETA0))
    query_theta1 = float(meta.get("query_theta1", DEFAULT_QUERY_THETA1))
    query_theta2 = float(meta.get("query_theta2", DEFAULT_QUERY_THETA2))
    query_theta3 = float(meta.get("query_theta3", DEFAULT_QUERY_THETA3))

    out: Dict[str, nx.Graph] = {}
    for variant in variant_order:
        mode = "random" if variant == "default" else str(variant)
        if mode not in {"unit", "random", "cosine_beta", "query_aware"}:
            continue
        if mode == "query_aware" and query_vector is None:
            continue
        seeded = base_seed + VARIANT_SEED_OFFSETS.get(mode, 0)
        weighted_edges = reweight_edges_by_mode(
            base_edges,
            embeddings,
            mode=mode,
            random_seed=seeded,
            query_vector=query_vector if mode == "query_aware" else None,
            cosine_beta_alpha=cosine_beta_alpha,
            cosine_beta_kappa=cosine_beta_kappa,
            cosine_beta_offset=cosine_beta_offset,
            cosine_beta_scale=cosine_beta_scale,
            cosine_beta_clip_min=cosine_beta_clip_min,
            cosine_beta_clip_max=cosine_beta_clip_max,
            query_theta0=query_theta0,
            query_theta1=query_theta1,
            query_theta2=query_theta2,
            query_theta3=query_theta3,
            target_mean_weight=target_mean_weight,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        out[variant] = _graph_from_weighted_edges(node_ids, weighted_edges)
    return out


def _is_floatish(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _lambda_label(raw_key: str) -> str:
    return f"\u03bb = {float(raw_key):+.3f}" if _is_floatish(raw_key) else f"\u03bb = {raw_key}"


def _mode_sort_key(label: str) -> Tuple[int, float, str]:
    if label == "GraphRAG":
        return (0, 0.0, label)
    if label.startswith("\u03bb = "):
        raw = label.replace("\u03bb = ", "")
        try:
            return (1, float(raw), label)
        except Exception:
            return (1, 0.0, label)
    return (2, 0.0, label)


def _stats_from_rows(rows: Sequence[Mapping[str, object]]) -> Tuple[float | None, float | None]:
    distances: List[float] = []
    sims: List[float] = []
    for row in rows:
        if row.get("query_similarity") is None:
            continue
        sim = float(row["query_similarity"])
        sims.append(sim)
        distances.append(1.0 - sim)
    avg_d = float(sum(distances) / len(distances)) if distances else None
    avg_s = float(sum(sims) / len(sims)) if sims else None
    return avg_d, avg_s


def _variant_color(variant: str) -> str:
    return {
        "random": "#00B8D9",
        "cosine_beta": "#FF8A00",
        "query_aware": "#00A86B",
        "default": "#00B8D9",
    }.get(variant, "#5B84FF")


def _load_vectors(vectors_path: str, node_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    payload = json.loads(Path(vectors_path).read_text(encoding="utf-8"))
    vectors = payload.get("vectors")
    if isinstance(vectors, dict):
        raw = {str(k): np.asarray(v, dtype=float) for k, v in vectors.items()}
    elif isinstance(vectors, list):
        raw = {str(i): np.asarray(v, dtype=float) for i, v in enumerate(vectors)}
    else:
        raise ValueError("Vectors JSON must have 'vectors' as a list or dict.")
    missing = [nid for nid in node_ids if nid not in raw]
    if missing:
        raise ValueError(f"Missing vectors for node IDs (examples: {missing[:5]}).")
    return {nid: raw[nid] for nid in node_ids}


def _project_3d(embeddings: Dict[str, np.ndarray], node_ids: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    arr = np.stack([embeddings[nid] for nid in node_ids], axis=0).astype(float)
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    if arr.shape[1] >= 3:
        _, _, v_t = np.linalg.svd(arr, full_matrices=False)
        projected = arr @ v_t[:3].T
    elif arr.shape[1] == 2:
        projected = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=float)], axis=1)
    elif arr.shape[1] == 1:
        projected = np.concatenate([arr, np.zeros((arr.shape[0], 2), dtype=float)], axis=1)
    else:
        projected = np.zeros((arr.shape[0], 3), dtype=float)
    return {nid: (float(projected[i, 0]), float(projected[i, 1]), float(projected[i, 2])) for i, nid in enumerate(node_ids)}


def _build_query_vector(
    *,
    embeddings: Dict[str, np.ndarray],
    query_mode: str,
    query_vertices: Sequence[str],
    query_random_points: int | None,
    query_seed: int | None,
    query_vector_space: str | None,
) -> np.ndarray | None:
    if query_mode == "vertex_subset" and query_vertices:
        vecs = [embeddings[str(v)] for v in query_vertices if str(v) in embeddings]
        if not vecs:
            return None
        q = np.mean(np.stack(vecs, axis=0), axis=0)
    elif query_random_points and query_seed is not None:
        dim = int(next(iter(embeddings.values())).shape[0])
        rng = np.random.default_rng(int(query_seed))
        if query_vector_space == "sphere":
            q = rng.normal(loc=0.0, scale=1.0, size=dim)
        else:
            q = rng.random(size=dim)
    else:
        return None
    norm = float(np.linalg.norm(q))
    return (q / norm) if norm > 0.0 else None


def load_variant_modes(
    *,
    variants_path: str | None,
    selected_nodes_path: str | None,
    graphrag_path: str | None,
) -> Tuple[
    List[str],
    Dict[str, Dict[str, Tuple[List[str], float | None, float | None]]],
    List[str],
    Dict[str, object],
]:
    if variants_path is not None:
        payload = json.loads(Path(variants_path).read_text(encoding="utf-8"))
        variants_obj = payload.get("variants", {})
        if not isinstance(variants_obj, dict):
            raise ValueError("Variants JSON must contain an object field 'variants'.")
        variant_order_raw = payload.get("variant_order", list(variants_obj.keys()))
        variant_order = [str(v) for v in variant_order_raw if str(v) in variants_obj]
        if not variant_order:
            variant_order = sorted(str(v) for v in variants_obj.keys())
        seed_nodes = [str(x) for x in payload.get("seed_nodes", [])]

        modes_by_variant: Dict[str, Dict[str, Tuple[List[str], float | None, float | None]]] = {}
        for variant in variant_order:
            vdata = variants_obj.get(variant, {})
            if not isinstance(vdata, dict):
                continue
            graphrag_obj = vdata.get("graphrag", {})
            bunny_obj = vdata.get("bunny_by_lambda", {})
            if not isinstance(graphrag_obj, dict) or not isinstance(bunny_obj, dict):
                continue

            modes: Dict[str, Tuple[List[str], float | None, float | None]] = {}
            graph_rows = graphrag_obj.get("nodes", [])
            typed_graph_rows = [r for r in graph_rows if isinstance(r, dict)] if isinstance(graph_rows, list) else []
            modes["GraphRAG"] = (
                [str(r["node_id"]) for r in typed_graph_rows if r.get("node_id") is not None],
                *_stats_from_rows(typed_graph_rows),
            )

            lambda_keys = list(bunny_obj.keys())
            try:
                lambda_keys.sort(key=lambda k: float(k))
            except Exception:
                lambda_keys.sort()
            for key in lambda_keys:
                rows = bunny_obj.get(key, [])
                typed_rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
                modes[_lambda_label(str(key))] = (
                    [str(r["node_id"]) for r in typed_rows if r.get("node_id") is not None],
                    *_stats_from_rows(typed_rows),
                )
            modes_by_variant[variant] = modes

        meta = {
            "query_mode": payload.get("query_mode"),
            "query_vertices": payload.get("query_vertices", []),
            "query_random_points": payload.get("query_random_points"),
            "query_seed": payload.get("query_seed"),
            "query_vector_space": payload.get("query_vector_space"),
            "vectors_path": payload.get("vectors_path"),
            "edge_weight_seed": payload.get("edge_weight_seed"),
            "cosine_beta_alpha": payload.get("cosine_beta_alpha"),
            "cosine_beta_kappa": payload.get("cosine_beta_kappa"),
            "cosine_beta_offset": payload.get("cosine_beta_offset"),
            "cosine_beta_scale": payload.get("cosine_beta_scale"),
            "cosine_beta_clip_min": payload.get("cosine_beta_clip_min"),
            "cosine_beta_clip_max": payload.get("cosine_beta_clip_max"),
            "query_theta0": payload.get("query_theta0"),
            "query_theta1": payload.get("query_theta1"),
            "query_theta2": payload.get("query_theta2"),
            "query_theta3": payload.get("query_theta3"),
            "target_mean_weight": payload.get("target_mean_weight"),
            "min_weight": payload.get("min_weight"),
            "max_weight": payload.get("max_weight"),
        }
        return variant_order, modes_by_variant, seed_nodes, meta

    if selected_nodes_path is None or graphrag_path is None:
        raise ValueError(
            "Provide either --variants-path OR both --selected-nodes-path and --graphrag-path."
        )

    selected_payload = json.loads(Path(selected_nodes_path).read_text(encoding="utf-8"))
    graphrag_payload = json.loads(Path(graphrag_path).read_text(encoding="utf-8"))
    seed_nodes = [str(x) for x in graphrag_payload.get("seed_nodes", [])]
    modes: Dict[str, Tuple[List[str], float | None, float | None]] = {}

    graph_rows = graphrag_payload.get("nodes", [])
    typed_graph_rows = [r for r in graph_rows if isinstance(r, dict)] if isinstance(graph_rows, list) else []
    modes["GraphRAG"] = (
        [str(r["node_id"]) for r in typed_graph_rows if r.get("node_id") is not None],
        *_stats_from_rows(typed_graph_rows),
    )

    lambda_keys = list(selected_payload.keys())
    try:
        lambda_keys.sort(key=lambda k: float(k))
    except Exception:
        lambda_keys.sort()
    for key in lambda_keys:
        rows = selected_payload.get(key, [])
        typed_rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        modes[_lambda_label(str(key))] = (
            [str(r["node_id"]) for r in typed_rows if r.get("node_id") is not None],
            *_stats_from_rows(typed_rows),
        )

    meta = {
        "query_mode": graphrag_payload.get("query_mode"),
        "query_vertices": graphrag_payload.get("query_vertices", []),
        "query_random_points": graphrag_payload.get("query_random_points"),
        "query_seed": graphrag_payload.get("query_seed"),
        "query_vector_space": graphrag_payload.get("query_vector_space"),
        "vectors_path": None,
        "edge_weight_seed": None,
        "cosine_beta_alpha": None,
        "cosine_beta_kappa": None,
        "cosine_beta_offset": None,
        "cosine_beta_scale": None,
        "cosine_beta_clip_min": None,
        "cosine_beta_clip_max": None,
        "query_theta0": None,
        "query_theta1": None,
        "query_theta2": None,
        "query_theta3": None,
        "target_mean_weight": None,
        "min_weight": None,
        "max_weight": None,
    }
    return ["default"], {"default": modes}, seed_nodes, meta


def clustered_layout(
    g: nx.Graph,
    inter_community_weight_scale: float = 0.1,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
    communities = list(nx.community.greedy_modularity_communities(g, weight="weight"))
    node_to_comm: Dict[str, int] = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[str(node)] = idx

    k_comm = max(1, len(communities))
    radius = 3.0
    centers: Dict[int, Tuple[float, float]] = {}
    for idx in range(k_comm):
        angle = 2.0 * math.pi * idx / k_comm
        centers[idx] = (radius * math.cos(angle), radius * math.sin(angle))

    rng = random.Random(42)
    init_pos: Dict[str, Tuple[float, float]] = {}
    for node in g.nodes():
        node = str(node)
        comm_idx = node_to_comm.get(node, 0)
        cx, cy = centers[comm_idx]
        init_pos[node] = (cx + rng.uniform(-0.35, 0.35), cy + rng.uniform(-0.35, 0.35))

    layout_g = nx.Graph()
    layout_g.add_nodes_from(g.nodes())
    for u, v, data in g.edges(data=True):
        u = str(u)
        v = str(v)
        base_w = float(data.get("weight", 1.0))
        if node_to_comm.get(u) != node_to_comm.get(v):
            layout_w = base_w * inter_community_weight_scale
        else:
            layout_w = base_w
        layout_g.add_edge(u, v, layout_weight=layout_w)

    pos = nx.spring_layout(
        layout_g,
        seed=42,
        pos=init_pos,
        weight="layout_weight",
        k=1.4 / math.sqrt(max(1, g.number_of_nodes())),
        iterations=350,
    )
    return {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}, node_to_comm


def build_graph_figure(
    *,
    g: nx.Graph,
    graph_pos: Dict[str, Tuple[float, float]],
    node_to_comm: Dict[str, int],
    modes: Mapping[str, Tuple[List[str], float | None, float | None]],
    seed_nodes: Sequence[str],
    title: str,
    selection_color: str,
    plot_height: int,
) -> go.Figure:
    mode_labels = sorted(modes.keys(), key=_mode_sort_key)
    if not mode_labels:
        raise ValueError("No mode data found for graph figure.")

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v in g.edges():
        x0, y0 = graph_pos[str(u)]
        x1, y1 = graph_pos[str(v)]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_color: List[str] = []
    node_size: List[float] = []
    for n in g.nodes():
        nid = str(n)
        x, y = graph_pos[nid]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"node={nid}<br>degree={g.degree(nid)}")
        cidx = node_to_comm.get(nid, 0) % len(palette)
        node_color.append(palette[cidx])
        node_size.append(8 + 1.2 * g.degree(nid))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.8, color="rgba(80,80,80,0.20)"),
            hoverinfo="skip",
            showlegend=False,
            name="edges",
        )
    )
    edge_trace_idx = len(fig.data) - 1
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=node_size, color=node_color, line=dict(width=0.0), opacity=0.25),
            text=node_text,
            hoverinfo="text",
            showlegend=False,
            name="nodes",
        )
    )
    colored_nodes_trace_idx = len(fig.data) - 1

    seed_x: List[float] = []
    seed_y: List[float] = []
    for node_id in seed_nodes:
        if node_id not in graph_pos:
            continue
        x, y = graph_pos[node_id]
        seed_x.append(x)
        seed_y.append(y)
    fig.add_trace(
        go.Scatter(
            x=seed_x,
            y=seed_y,
            mode="markers+text",
            marker=dict(size=16, color="#ff1f1f", line=dict(width=2.2, color="#8b0000"), opacity=1.0),
            text=[str(n) for n in seed_nodes if n in graph_pos],
            textposition="bottom center",
            hoverinfo="text",
            showlegend=False,
            name="seed_nodes",
        )
    )
    seed_trace_idx = len(fig.data) - 1
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=node_size, color="#8f9aa8", line=dict(width=0.0), opacity=0.30),
            text=node_text,
            hoverinfo="text",
            showlegend=False,
            visible=False,
            name="nodes_seed_only",
        )
    )
    seed_only_nodes_trace_idx = len(fig.data) - 1

    seed_set = set(seed_nodes)
    selection_trace_indices: List[int] = []
    for mode_label in mode_labels:
        selected, avg_distance, avg_similarity = modes[mode_label]
        selected_non_seed = [n for n in selected if n not in seed_set]
        xs: List[float] = []
        ys: List[float] = []
        texts: List[str] = []
        for node_id in selected_non_seed:
            if node_id not in graph_pos:
                continue
            x, y = graph_pos[node_id]
            xs.append(x)
            ys.append(y)
            txt = f"mode={mode_label}<br>node={node_id}"
            if avg_distance is not None:
                txt += f"<br>avg query distance={avg_distance:.4f}"
            if avg_similarity is not None:
                txt += f"<br>avg query similarity={avg_similarity:.4f}"
            texts.append(txt)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=16, color=selection_color, line=dict(width=2.0, color="#0f172a"), opacity=0.98),
                text=[str(n) for n in selected_non_seed if n in graph_pos],
                textposition="top center",
                hovertext=texts,
                hoverinfo="text",
                showlegend=False,
                visible=False,
                name=mode_label,
            )
        )
        selection_trace_indices.append(len(fig.data) - 1)

    mode_labels_with_seed_only = ["Seeds only", *mode_labels]

    steps = []
    for mode_label in mode_labels_with_seed_only:
        if mode_label == "Seeds only":
            vis = [False] * len(fig.data)
            vis[edge_trace_idx] = True
            vis[seed_only_nodes_trace_idx] = True
            vis[seed_trace_idx] = True
            steps.append(
                dict(
                    method="update",
                    label=mode_label,
                    args=[
                        {"visible": vis},
                        {"title.text": f"{title}<br><sup>{mode_label}</sup>"},
                    ],
                )
            )
            continue

        vis = [False] * len(fig.data)
        vis[edge_trace_idx] = True
        vis[colored_nodes_trace_idx] = True
        vis[seed_trace_idx] = True
        i = mode_labels.index(mode_label)
        vis[selection_trace_indices[i]] = True
        _, avg_distance, avg_similarity = modes[mode_label]
        d_suffix = f", avg dist={avg_distance:.4f}" if avg_distance is not None else ""
        s_suffix = f", avg sim={avg_similarity:.4f}" if avg_similarity is not None else ""
        steps.append(
            dict(
                method="update",
                label=mode_label,
                args=[
                    {"visible": vis},
                    {"title.text": f"{title}<br><sup>{mode_label}{d_suffix}{s_suffix}</sup>"},
                ],
            )
        )

    for idx, is_visible in enumerate(steps[0]["args"][0]["visible"]):
        fig.data[idx].visible = bool(is_visible)
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{mode_labels_with_seed_only[0]}</sup>"),
        sliders=[dict(active=0, currentvalue={"prefix": "Mode: "}, pad={"t": 35}, steps=steps)],
        margin=dict(l=10, r=10, t=80, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        height=plot_height,
    )
    return fig


def build_vector_figure(
    *,
    vector_pos_3d: Mapping[str, Tuple[float, float, float]],
    query_pos_3d: Tuple[float, float, float] | None,
    modes: Mapping[str, Tuple[List[str], float | None, float | None]],
    variant_label: str,
    selection_color: str,
    seed_nodes: Sequence[str],
    title: str,
    plot_height: int,
) -> go.Figure:
    mode_labels = sorted(modes.keys(), key=_mode_sort_key)
    if not mode_labels:
        raise ValueError("No mode data found for vector figure.")

    node_ids = list(vector_pos_3d.keys())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[vector_pos_3d[n][0] for n in node_ids],
            y=[vector_pos_3d[n][1] for n in node_ids],
            z=[vector_pos_3d[n][2] for n in node_ids],
            mode="markers",
            marker=dict(size=3.5, color="#6b6b6b", opacity=0.30),
            text=[f"node={n}" for n in node_ids],
            hoverinfo="text",
            showlegend=False,
            name="nodes",
        )
    )
    nodes_trace_idx = len(fig.data) - 1
    fig.add_trace(
        go.Scatter3d(
            x=[vector_pos_3d[n][0] for n in seed_nodes if n in vector_pos_3d],
            y=[vector_pos_3d[n][1] for n in seed_nodes if n in vector_pos_3d],
            z=[vector_pos_3d[n][2] for n in seed_nodes if n in vector_pos_3d],
            mode="markers+text",
            marker=dict(size=7, color="#ff1f1f", opacity=1.0, line=dict(width=1.5, color="#8b0000")),
            text=[str(n) for n in seed_nodes if n in vector_pos_3d],
            textposition="bottom center",
            hoverinfo="text",
            showlegend=False,
            name="seed_nodes",
        )
    )
    seed_trace_idx = len(fig.data) - 1
    has_query = query_pos_3d is not None
    query_trace_idx: int | None = None
    if has_query and query_pos_3d is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[query_pos_3d[0]],
                y=[query_pos_3d[1]],
                z=[query_pos_3d[2]],
                mode="markers",
                marker=dict(size=14, color="#E69F00", opacity=0.35, line=dict(width=1.2, color="#9c6500")),
                hoverinfo="skip",
                showlegend=False,
                visible=False,
                name="query",
            )
        )
        query_trace_idx = len(fig.data) - 1

    seed_set = set(seed_nodes)
    mode_trace_index: Dict[str, int] = {}
    for mode_label in mode_labels:
        selected = modes.get(mode_label, ([], None, None))[0]
        nodes = [n for n in selected if n not in seed_set and n in vector_pos_3d]
        fig.add_trace(
            go.Scatter3d(
                x=[vector_pos_3d[n][0] for n in nodes],
                y=[vector_pos_3d[n][1] for n in nodes],
                z=[vector_pos_3d[n][2] for n in nodes],
                mode="markers+text",
                marker=dict(size=6, color=selection_color, opacity=0.98, line=dict(width=1.0, color="#0f172a")),
                text=[str(n) for n in nodes],
                textposition="top center",
                hoverinfo="text",
                visible=False,
                showlegend=True,
                legendgroup=variant_label,
                name=variant_label,
            )
        )
        mode_trace_index[mode_label] = len(fig.data) - 1

    mode_labels_with_seed_only = ["Seeds only"]
    if query_trace_idx is not None:
        mode_labels_with_seed_only.append("Query")
    mode_labels_with_seed_only.extend(mode_labels)

    steps = []
    for mode_label in mode_labels_with_seed_only:
        if mode_label == "Seeds only":
            vis = [False] * len(fig.data)
            vis[nodes_trace_idx] = True
            vis[seed_trace_idx] = True
            steps.append(
                dict(
                    method="update",
                    label=mode_label,
                    args=[
                        {"visible": vis},
                        {"title.text": f"{title}<br><sup>{mode_label}</sup>"},
                    ],
                )
            )
            continue

        if mode_label == "Query":
            vis = [False] * len(fig.data)
            vis[nodes_trace_idx] = True
            vis[seed_trace_idx] = True
            if query_trace_idx is not None:
                vis[query_trace_idx] = True
            steps.append(
                dict(
                    method="update",
                    label=mode_label,
                    args=[
                        {"visible": vis},
                        {"title.text": f"{title}<br><sup>{mode_label}</sup>"},
                    ],
                )
            )
            continue

        vis = [False] * len(fig.data)
        vis[nodes_trace_idx] = True
        vis[seed_trace_idx] = True
        idx = mode_trace_index.get(mode_label)
        if idx is not None:
            vis[idx] = True
        steps.append(
            dict(
                method="update",
                label=mode_label,
                args=[
                    {"visible": vis},
                    {"title.text": f"{title}<br><sup>{mode_label}</sup>"},
                ],
            )
        )

    for idx, is_visible in enumerate(steps[0]["args"][0]["visible"]):
        fig.data[idx].visible = bool(is_visible)
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{mode_labels_with_seed_only[0]}</sup>"),
        sliders=[dict(active=0, currentvalue={"prefix": "Mode: "}, pad={"t": 35}, steps=steps)],
        margin=dict(l=10, r=10, t=80, b=10),
        scene=dict(
            dragmode="turntable",
            xaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False),
            aspectmode="data",
        ),
        showlegend=True,
        legend=dict(orientation="h"),
        height=plot_height,
    )
    return fig


def render_page(
    *,
    output_path: Path,
    sections: Sequence[Tuple[str, str, str]],
) -> None:
    buttons = []
    panes = []
    for i, (sid, label, content) in enumerate(sections):
        active = "active" if i == 0 else ""
        display = "block" if i == 0 else "none"
        buttons.append(f"<button class='view-btn {active}' onclick=\"showView('{sid}', this)\">{label}</button>")
        panes.append(f"<section id='{sid}' class='view-panel' style='display:{display};'>{content}</section>")

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Synthetic Lambda Sweep Visualization</title>
  <style>
    body {{ margin:0; padding:18px; background:#f4f6f8; color:#13233a; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    .shell {{ max-width:1600px; margin:0 auto; background:#fff; border:1px solid #d6dde5; border-radius:14px; overflow:hidden; }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; padding:14px; border-bottom:1px solid #d6dde5; background:linear-gradient(120deg,#fbfdff,#eef4fb); }}
    .view-btn {{ border:1px solid #b8c6d8; background:#fff; color:#183153; border-radius:999px; padding:8px 14px; cursor:pointer; }}
    .view-btn.active {{ border-color:#0a84ff; color:#0a84ff; box-shadow:inset 0 0 0 1px #0a84ff; background:#f0f7ff; }}
    .view-panel {{ padding:8px; }}
    .subtoolbar {{ display:flex; gap:8px; flex-wrap:wrap; padding:8px 8px 0 8px; }}
    .vector-variant-btn {{ border:1px solid #b8c6d8; background:#fff; color:#183153; border-radius:999px; padding:6px 12px; cursor:pointer; font-size:13px; }}
    .vector-variant-btn.active {{ border-color:#0a84ff; color:#0a84ff; box-shadow:inset 0 0 0 1px #0a84ff; background:#f0f7ff; }}
  </style>
</head>
<body>
  <div class='shell'>
    <div class='toolbar'>{''.join(buttons)}</div>
    {''.join(panes)}
  </div>
  <script>
    function showView(id, btn) {{
      document.querySelectorAll('.view-panel').forEach(p => p.style.display = (p.id===id ? 'block' : 'none'));
      document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      if (btn) btn.classList.add('active');
      setTimeout(() => {{
        if (window.Plotly) {{
          document.querySelectorAll('#'+id+' .plotly-graph-div').forEach(g => Plotly.Plots.resize(g));
        }}
      }}, 100);
    }}
    function showVectorVariant(containerId, panelId, btn) {{
      const root = document.getElementById(containerId);
      if (!root) return;
      root.querySelectorAll('.vector-variant-panel').forEach(p => p.style.display = (p.id===panelId ? 'block' : 'none'));
      root.querySelectorAll('.vector-variant-btn').forEach(b => b.classList.remove('active'));
      if (btn) btn.classList.add('active');
      setTimeout(() => {{
        if (window.Plotly) {{
          root.querySelectorAll('#'+panelId+' .plotly-graph-div').forEach(g => Plotly.Plots.resize(g));
        }}
      }}, 100);
    }}
  </script>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize synthetic lambda sweep with top-level view toggles: "
            "3D vector space, graph-random, graph-cosine_beta, graph-query_aware."
        )
    )
    parser.add_argument("--graph-path", required=True, help="Path to graph JSON.")
    parser.add_argument("--variants-path", default=None, help="Path to synthetic_lambda_sweep_variants.json (preferred).")
    parser.add_argument("--selected-nodes-path", default=None, help="Legacy path to synthetic_bunny_lambda_selected_nodes.json.")
    parser.add_argument("--graphrag-path", default=None, help="Legacy path to synthetic_graphrag_topk_selected_nodes.json.")
    parser.add_argument("--vectors-path", default=None, help="Path to vectors JSON for 3D vector view.")
    parser.add_argument("--output-html", default="synthetic_bunny/output/lambda_sweep_visualization.html", help="Output HTML path.")
    parser.add_argument("--plot-height", type=int, default=450, help="Plot height in pixels for each interactive view.")
    parser.add_argument("--cluster-separation", type=float, default=0.1, help="Lower values make communities separate more strongly in layout.")
    parser.add_argument("--title", default="Synthetic Variant + Lambda Visualization", help="Base title.")
    args = parser.parse_args()

    g = load_graph(args.graph_path)
    variant_order, modes_by_variant, seed_nodes, meta = load_variant_modes(
        variants_path=args.variants_path,
        selected_nodes_path=args.selected_nodes_path,
        graphrag_path=args.graphrag_path,
    )
    base_node_ids, base_edges, _ = load_graph_payload_common(args.graph_path)

    sections: List[Tuple[str, str, str]] = []

    vectors_path = args.vectors_path or (
        str(meta.get("vectors_path")) if isinstance(meta.get("vectors_path"), str) else None
    )
    embeddings: Dict[str, np.ndarray] | None = None
    query_vec: np.ndarray | None = None
    variant_graphs: Dict[str, nx.Graph] = {}
    if vectors_path:
        node_ids = [str(n) for n in g.nodes()]
        embeddings = _load_vectors(vectors_path, node_ids)
        vpos = _project_3d(embeddings, node_ids)
        query_vec = _build_query_vector(
            embeddings=embeddings,
            query_mode=str(meta.get("query_mode", "")),
            query_vertices=[str(x) for x in meta.get("query_vertices", [])] if isinstance(meta.get("query_vertices"), list) else [],
            query_random_points=int(meta["query_random_points"]) if meta.get("query_random_points") is not None else None,
            query_seed=int(meta["query_seed"]) if meta.get("query_seed") is not None else None,
            query_vector_space=str(meta["query_vector_space"]) if meta.get("query_vector_space") is not None else None,
        )
        qpos = None
        if query_vec is not None:
            arr = np.stack([embeddings[nid] for nid in node_ids], axis=0).astype(float)
            arr_centered = arr - np.mean(arr, axis=0, keepdims=True)
            if arr_centered.shape[1] >= 3:
                _, _, v_t = np.linalg.svd(arr_centered, full_matrices=False)
                basis = v_t[:3].T
                q_centered = query_vec - np.mean(arr, axis=0)
                qp = q_centered @ basis
                qpos = (float(qp[0]), float(qp[1]), float(qp[2]))

        vector_variants = [v for v in variant_order if v in modes_by_variant and modes_by_variant[v]]
        if not vector_variants:
            vector_variants = [next(iter(modes_by_variant.keys()))]

        vector_container_id = "vector-variant-container"
        vector_buttons: List[str] = []
        vector_panels: List[str] = []
        include_plotly_for_first = True

        for i, variant in enumerate(vector_variants):
            variant_label = "default" if variant == "default" else variant
            panel_id = f"vector-variant-panel-{i}"
            is_active = i == 0
            btn_active = "active" if is_active else ""
            panel_display = "block" if is_active else "none"
            vector_buttons.append(
                f"<button class='vector-variant-btn {btn_active}' "
                f"onclick=\"showVectorVariant('{vector_container_id}','{panel_id}', this)\">{variant_label}</button>"
            )

            fig_vector = build_vector_figure(
                vector_pos_3d=vpos,
                query_pos_3d=qpos,
                modes=modes_by_variant[variant],
                variant_label=variant_label,
                selection_color=_variant_color(variant),
                seed_nodes=seed_nodes,
                title=f"{args.title} - 3D Vector Space ({variant_label})",
                plot_height=args.plot_height,
            )
            fig_html = pio.to_html(
                fig_vector,
                full_html=False,
                include_plotlyjs=include_plotly_for_first,
                div_id=f"fig_vector_3d_{variant_label}_{i}",
            )
            include_plotly_for_first = False
            vector_panels.append(
                f"<div id='{panel_id}' class='vector-variant-panel' style='display:{panel_display};'>{fig_html}</div>"
            )

        vector_content = (
            f"<div id='{vector_container_id}'>"
            f"<div class='subtoolbar'>{''.join(vector_buttons)}</div>"
            f"{''.join(vector_panels)}"
            f"</div>"
        )
        sections.append(("view-vector", "3D Vector Space", vector_content))

    if args.variants_path is not None and embeddings is not None:
        variant_graphs = _build_variant_graphs(
            node_ids=base_node_ids,
            base_edges=base_edges,
            embeddings=embeddings,
            query_vector=query_vec,
            variant_order=variant_order,
            meta=meta,
        )

    variant_view_order: List[Tuple[str, str, str]] = []
    if "random" in modes_by_variant:
        variant_view_order.append(("random", "view-random", "Graph: random"))
    elif "default" in modes_by_variant:
        variant_view_order.append(("default", "view-random", "Graph"))
    if "cosine_beta" in modes_by_variant:
        variant_view_order.append(("cosine_beta", "view-cosine", "Graph: cosine_beta"))
    if "query_aware" in modes_by_variant:
        variant_view_order.append(("query_aware", "view-query", "Graph: query_aware"))

    for variant, sid, label in variant_view_order:
        graph_for_variant = variant_graphs.get(variant, g)
        graph_pos, node_to_comm = clustered_layout(
            graph_for_variant,
            inter_community_weight_scale=args.cluster_separation,
        )
        fig = build_graph_figure(
            g=graph_for_variant,
            graph_pos=graph_pos,
            node_to_comm=node_to_comm,
            modes=modes_by_variant[variant],
            seed_nodes=seed_nodes,
            title=f"{args.title} - {label}",
            selection_color=_variant_color(variant),
            plot_height=args.plot_height,
        )
        sections.append(
            (
                sid,
                label,
                pio.to_html(
                    fig,
                    full_html=False,
                    include_plotlyjs=(False if sections else True),
                    div_id=f"fig_{variant}",
                ),
            )
        )

    if not sections:
        raise ValueError("No views available to render.")

    # Ensure Plotly bundle included at least once.
    if all("plotly.js" not in html for _, _, html in sections):
        sid, label, html = sections[0]
        html = pio.to_html(go.Figure(), full_html=False, include_plotlyjs=True, div_id="fig_plotly_stub") + html
        sections[0] = (sid, label, html)

    out = Path(args.output_html)
    render_page(output_path=out, sections=sections)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
