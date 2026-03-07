from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_component_indices(raw: str) -> Tuple[int, int, int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--components must contain exactly three comma-separated integers.")
    # User-facing components are 1-based.
    zero_based = tuple(v - 1 for v in vals)
    if any(v < 0 for v in zero_based):
        raise ValueError("--components values must be >= 1.")
    return zero_based  # type: ignore[return-value]


def load_vectors(vectors_path: Path) -> Tuple[List[str], np.ndarray]:
    payload = json.loads(vectors_path.read_text(encoding="utf-8"))
    vectors = payload.get("vectors")
    if isinstance(vectors, list):
        ids = [str(i) for i in range(len(vectors))]
        arr = np.asarray(vectors, dtype=float)
        return ids, arr
    if isinstance(vectors, dict):
        ids = [str(k) for k in vectors.keys()]
        arr = np.asarray([vectors[k] for k in vectors.keys()], dtype=float)
        return ids, arr
    raise ValueError("Vectors JSON must contain 'vectors' as list or dict.")


def build_query_vector(dim: int, query_seed: int) -> np.ndarray:
    rng = np.random.default_rng(query_seed)
    query = rng.random(size=dim)
    norm = float(np.linalg.norm(query))
    if norm <= 0.0:
        raise ValueError("Query vector norm is zero.")
    return query / norm


def make_vector_figure(
    ids: Sequence[str],
    xyz: np.ndarray,
    query_xyz: np.ndarray,
    top3_ids: Sequence[str],
    selection_modes: Sequence[Tuple[str, List[str]]],
    title: str,
) -> go.Figure:
    node_color = "#7A7A7A"
    query_color = "#E69F00"
    top3_color = "#D55E00"

    top3_set = set(top3_ids)
    top3_mask = np.array([node_id in top3_set for node_id in ids], dtype=bool)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(size=3.5, color=node_color, opacity=0.62),
            hoverinfo="skip",
            showlegend=False,
            name="nodes",
        )
    )

    query_trace_indices: List[int] = []
    glow_sizes = [34, 26, 19, 13]
    glow_opacity = [0.05, 0.09, 0.16, 0.28]
    for size, alpha in zip(glow_sizes, glow_opacity):
        fig.add_trace(
            go.Scatter3d(
                x=[float(query_xyz[0])],
                y=[float(query_xyz[1])],
                z=[float(query_xyz[2])],
                mode="markers",
                marker=dict(size=size, color=query_color, opacity=alpha),
                hoverinfo="skip",
                visible=False,
                showlegend=False,
                name="query",
            )
        )
        query_trace_indices.append(len(fig.data) - 1)

    fig.add_trace(
        go.Scatter3d(
            x=[float(query_xyz[0])],
            y=[float(query_xyz[1])],
            z=[float(query_xyz[2])],
            mode="markers",
            marker=dict(size=8, color=query_color, opacity=0.95, line=dict(width=1.2, color="black")),
            hoverinfo="skip",
            visible=False,
            showlegend=False,
            name="query_core",
        )
    )
    query_trace_indices.append(len(fig.data) - 1)

    fig.add_trace(
        go.Scatter3d(
            x=xyz[top3_mask, 0],
            y=xyz[top3_mask, 1],
            z=xyz[top3_mask, 2],
            mode="markers+text",
            marker=dict(size=7.5, color=top3_color, opacity=0.98, line=dict(width=1.0, color="black")),
            text=[node_id for node_id in ids if node_id in top3_set],
            textposition="top center",
            textfont=dict(size=11, color="#111"),
            hoverinfo="skip",
            visible=False,
            showlegend=False,
            name="top3",
        )
    )
    top3_trace_index = len(fig.data) - 1

    id_to_idx = {node_id: i for i, node_id in enumerate(ids)}
    selection_trace_indices: List[int] = []
    for mode_label, selected_ids in selection_modes:
        sel_indices = [id_to_idx[node_id] for node_id in selected_ids if node_id in id_to_idx]
        if sel_indices:
            sel_xyz = xyz[np.asarray(sel_indices, dtype=int)]
            x_sel = sel_xyz[:, 0]
            y_sel = sel_xyz[:, 1]
            z_sel = sel_xyz[:, 2]
        else:
            x_sel = []
            y_sel = []
            z_sel = []
        fig.add_trace(
            go.Scatter3d(
                x=x_sel,
                y=y_sel,
                z=z_sel,
                mode="markers+text",
                marker=dict(size=8.5, color="#00E5FF", opacity=0.98, line=dict(width=1.1, color="#007F8C")),
                text=[node_id for node_id in selected_ids if node_id in id_to_idx],
                textposition="top center",
                textfont=dict(size=11, color="#003E47"),
                hoverinfo="skip",
                visible=False,
                showlegend=False,
                name=f"mode_{mode_label}",
            )
        )
        selection_trace_indices.append(len(fig.data) - 1)

    total = len(fig.data)

    def visibility(mode: int) -> List[bool]:
        v = [False] * total
        v[0] = True  # nodes always visible
        if mode >= 1:
            for idx in query_trace_indices:
                v[idx] = True
        if mode >= 2:
            v[top3_trace_index] = True
        return v

    def visibility_selection(trace_index: int) -> List[bool]:
        v = [False] * total
        v[0] = True  # nodes
        for idx in query_trace_indices:
            v[idx] = True  # query
        v[trace_index] = True  # selected top-k in cyan
        return v

    steps = [
        dict(
            method="update",
            label="Nodes",
            args=[{"visible": visibility(0)}, {"title.text": f"{title}<br><sup>Nodes only</sup>"}],
        ),
        dict(
            method="update",
            label="+ Query",
            args=[{"visible": visibility(1)}, {"title.text": f"{title}<br><sup>Nodes + query</sup>"}],
        ),
        dict(
            method="update",
            label="+ Query + Top-3",
            args=[
                {"visible": visibility(2)},
                {"title.text": f"{title}<br><sup>Nodes + query + closest nodes (476,150,438)</sup>"},
            ],
        ),
    ]
    for mode_label, trace_idx in zip(selection_modes, selection_trace_indices):
        label = mode_label[0]
        steps.append(
            dict(
                method="update",
                label=label,
                args=[
                    {"visible": visibility_selection(trace_idx)},
                    {"title.text": f"{title}<br><sup>{label}: top-k highlighted in cyan</sup>"},
                ],
            )
        )

    initial_visibility = visibility(2)
    for i, vis in enumerate(initial_visibility):
        fig.data[i].visible = vis

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>Nodes + query + closest nodes (476,150,438)</sup>"),
        sliders=[
            dict(
                active=2,
                currentvalue={"prefix": "View: "},
                pad={"t": 40},
                len=0.78,
                x=0.11,
                xanchor="left",
                y=0.0,
                ticklen=8,
                font=dict(size=12),
                steps=steps,
            )
        ],
        margin=dict(l=10, r=10, t=80, b=10),
        scene=dict(
            dragmode="turntable",
            xaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            aspectmode="data",
        ),
        height=900,
        showlegend=False,
    )
    return fig


def load_graph(graph_path: Path) -> nx.Graph:
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
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


def _is_floatish(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def load_selection_modes(
    selected_nodes_path: Path,
    graphrag_path: Path,
) -> Tuple[List[Tuple[str, List[str], float | None, float | None]], List[str]]:
    selected_payload = json.loads(selected_nodes_path.read_text(encoding="utf-8"))
    graphrag_payload = json.loads(graphrag_path.read_text(encoding="utf-8"))

    modes: List[Tuple[str, List[str], float | None, float | None]] = []
    seed_nodes = [str(x) for x in graphrag_payload.get("seed_nodes", [])]

    # Requested extra mode: show only seed nodes and no top-k selections.
    modes.append(("Seed-only", [], None, None))

    graphrag_nodes: List[str] = []
    graphrag_distances: List[float] = []
    graphrag_sims: List[float] = []
    for item in graphrag_payload.get("nodes", []):
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if node_id is None:
            continue
        graphrag_nodes.append(str(node_id))
        if item.get("query_similarity") is not None:
            sim = float(item["query_similarity"])
            graphrag_sims.append(sim)
            graphrag_distances.append(1.0 - sim)
    graphrag_avg_distance = (
        float(sum(graphrag_distances) / len(graphrag_distances)) if graphrag_distances else None
    )
    graphrag_avg_similarity = (
        float(sum(graphrag_sims) / len(graphrag_sims)) if graphrag_sims else None
    )
    modes.append(("GraphRAG", graphrag_nodes, graphrag_avg_distance, graphrag_avg_similarity))

    lambda_keys = list(selected_payload.keys())
    try:
        lambda_keys.sort(key=lambda k: float(k))
    except Exception:
        lambda_keys.sort()

    for key in lambda_keys:
        rows = selected_payload.get(key, [])
        node_ids: List[str] = []
        distances: List[float] = []
        sims: List[float] = []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("node_id") is not None:
                    node_ids.append(str(row["node_id"]))
                    if row.get("query_similarity") is not None:
                        sim = float(row["query_similarity"])
                        sims.append(sim)
                        distances.append(1.0 - sim)
        label = f"\u03bb = {float(key):+.3f}" if _is_floatish(key) else f"\u03bb = {key}"
        avg_distance = float(sum(distances) / len(distances)) if distances else None
        avg_similarity = float(sum(sims) / len(sims)) if sims else None
        modes.append((label, node_ids, avg_distance, avg_similarity))
    return modes, seed_nodes


def clustered_layout(
    g: nx.Graph,
    inter_community_weight_scale: float = 0.1,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
    communities = list(nx.community.greedy_modularity_communities(g))
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
    pos = {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    return pos, node_to_comm


def make_graph_figure(
    g: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    node_to_comm: Dict[str, int],
    modes: Sequence[Tuple[str, List[str], float | None, float | None]],
    seed_nodes: Sequence[str],
    title: str,
) -> go.Figure:
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
        x0, y0 = pos[str(u)]
        x1, y1 = pos[str(v)]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x: List[float] = []
    node_y: List[float] = []
    node_color: List[str] = []
    node_size: List[float] = []
    for n in g.nodes():
        nid = str(n)
        x, y = pos[nid]
        node_x.append(x)
        node_y.append(y)
        cidx = node_to_comm.get(nid, 0) % len(palette)
        node_color.append(palette[cidx])
        node_size.append(8 + 1.2 * g.degree(nid))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.8, color="rgba(80,80,80,0.25)"),
            hoverinfo="skip",
            showlegend=False,
            name="edges",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=node_size, color=node_color, line=dict(width=0.0), opacity=0.25),
            hoverinfo="skip",
            showlegend=False,
            name="nodes",
        )
    )

    seed_x: List[float] = []
    seed_y: List[float] = []
    for node_id in seed_nodes:
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        seed_x.append(x)
        seed_y.append(y)
    fig.add_trace(
        go.Scatter(
            x=seed_x,
            y=seed_y,
            mode="markers+text",
            marker=dict(
                size=19,
                color="#ff1f1f",
                line=dict(width=2.5, color="#8b0000"),
                opacity=1.0,
            ),
            text=[str(n) for n in seed_nodes if n in pos],
            textposition="bottom center",
            hoverinfo="skip",
            showlegend=False,
            name="seed_nodes",
            visible=True,
        )
    )

    seed_set = set(seed_nodes)
    mode_trace_indices: List[int] = []
    for mode_label, selected, _, _ in modes:
        xs: List[float] = []
        ys: List[float] = []
        selected_non_seed = [n for n in selected if n not in seed_set]
        for node_id in selected_non_seed:
            if node_id not in pos:
                continue
            x, y = pos[node_id]
            xs.append(x)
            ys.append(y)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(
                    size=18,
                    color="#00e5ff",
                    line=dict(width=2.5, color="#007f8c"),
                    opacity=1.0,
                ),
                text=[str(n) for n in selected_non_seed],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
                name=mode_label,
                visible=False,
            )
        )
        mode_trace_indices.append(len(fig.data) - 1)

    if mode_trace_indices:
        fig.data[mode_trace_indices[0]].visible = True

    steps = []
    for i, (mode_label, _, avg_distance, avg_similarity) in enumerate(modes):
        visible = [True, True, True] + [False] * len(mode_trace_indices)
        visible[3 + i] = True
        distance_suffix = f", avg dist={avg_distance:.4f}" if avg_distance is not None else ""
        similarity_suffix = (
            f", avg sim={avg_similarity:.4f}" if avg_similarity is not None else ""
        )
        steps.append(
            dict(
                method="update",
                label=f"{mode_label}{similarity_suffix}",
                args=[
                    {"visible": visible},
                    {"title.text": f"{title}<br><sup>Mode: {mode_label}{distance_suffix}{similarity_suffix}</sup>"},
                ],
            )
        )

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>Mode: {modes[0][0] if modes else 'N/A'}</sup>"),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Selection: "},
                pad={"t": 40},
                steps=steps,
            )
        ],
        margin=dict(l=10, r=10, t=80, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        height=900,
        showlegend=False,
    )
    return fig


def write_combined_html(vector_fig: go.Figure, graph_fig: go.Figure, output_html: Path) -> None:
    vector_html = pio.to_html(
        vector_fig,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
    graph_html = pio.to_html(
        graph_fig,
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive Embedding + Graph Views</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f8f8f8; }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 12px; }}
    .toolbar {{ display: flex; gap: 8px; margin-bottom: 10px; }}
    .toggle-btn {{
      border: 1px solid #666; background: #fff; color: #222; padding: 8px 12px; cursor: pointer;
    }}
    .toggle-btn.active {{ background: #222; color: #fff; }}
    .view {{ display: none; min-height: 86vh; }}
    .view.active {{ display: block; }}
    .view .plotly-graph-div {{ height: 86vh !important; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <button id="btn-vector" class="toggle-btn active" onclick="showView('vector')">Vector Plot</button>
      <button id="btn-graph" class="toggle-btn" onclick="showView('graph')">Graph Layout</button>
    </div>
    <div id="view-vector" class="view active">{vector_html}</div>
    <div id="view-graph" class="view">{graph_html}</div>
  </div>
  <script>
    function showView(which) {{
      const vector = document.getElementById('view-vector');
      const graph = document.getElementById('view-graph');
      const btnVector = document.getElementById('btn-vector');
      const btnGraph = document.getElementById('btn-graph');

      if (which === 'vector') {{
        vector.classList.add('active');
        graph.classList.remove('active');
        btnVector.classList.add('active');
        btnGraph.classList.remove('active');
      }} else {{
        graph.classList.add('active');
        vector.classList.remove('active');
        btnGraph.classList.add('active');
        btnVector.classList.remove('active');
      }}

      if (window.Plotly) {{
        const plots = document.querySelectorAll('.js-plotly-plot');
        plots.forEach((p) => window.Plotly.Plots.resize(p));
      }}
    }}
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an interactive 3D embedding HTML with slider modes for nodes/query/top-3."
    )
    parser.add_argument(
        "--vectors-path",
        default="synthetic_bunny/fixtures/seed42_n500_dim6_sp0025/random_spherical_vectors.json",
        help="Path to vectors JSON.",
    )
    parser.add_argument(
        "--graph-path",
        default="synthetic_bunny/fixtures/seed42_n500_dim6_sp0025/random_spherical_bunny_graph.json",
        help="Path to graph JSON.",
    )
    parser.add_argument(
        "--selected-nodes-path",
        default="synthetic_bunny/fixtures/seed42_n500_dim6_sp0025/lambda_sweep/synthetic_bunny_lambda_selected_nodes.json",
        help="Path to synthetic_bunny_lambda_selected_nodes.json.",
    )
    parser.add_argument(
        "--graphrag-path",
        default="synthetic_bunny/fixtures/seed42_n500_dim6_sp0025/lambda_sweep/synthetic_graphrag_topk_selected_nodes.json",
        help="Path to synthetic_graphrag_topk_selected_nodes.json.",
    )
    parser.add_argument(
        "--components",
        default="2,5,6",
        help="Three 1-based component indices to project, e.g. '2,5,6'.",
    )
    parser.add_argument(
        "--query-seed",
        type=int,
        default=17,
        help="Seed used to reproduce the random query vector.",
    )
    parser.add_argument(
        "--top3-ids",
        default="476,150,438",
        help="Comma-separated IDs to highlight as closest nodes.",
    )
    parser.add_argument(
        "--output-html",
        default="presentation/interactive_projection.html",
        help="Output interactive HTML path with vector/graph toggle.",
    )
    parser.add_argument(
        "--title",
        default="500-Node Embedding Projection",
        help="Figure title.",
    )
    args = parser.parse_args()

    vectors_path = Path(args.vectors_path)
    ids, arr = load_vectors(vectors_path)
    if arr.ndim != 2:
        raise ValueError("Vectors payload must produce a 2D array.")

    comp = parse_component_indices(args.components)
    if max(comp) >= arr.shape[1]:
        raise ValueError(
            f"Component index out of range for dim={arr.shape[1]}. Received {args.components}."
        )

    query = build_query_vector(arr.shape[1], args.query_seed)
    xyz = arr[:, comp]
    query_xyz = query[list(comp)]
    top3_ids = parse_csv_list(args.top3_ids)

    g = load_graph(Path(args.graph_path))
    modes, seed_nodes = load_selection_modes(
        selected_nodes_path=Path(args.selected_nodes_path),
        graphrag_path=Path(args.graphrag_path),
    )
    vector_selection_modes = [(label, node_ids) for label, node_ids, _, _ in modes if label != "Seed-only"]

    vector_fig = make_vector_figure(
        ids=ids,
        xyz=xyz,
        query_xyz=query_xyz,
        top3_ids=top3_ids,
        selection_modes=vector_selection_modes,
        title=args.title,
    )

    pos, node_to_comm = clustered_layout(g=g, inter_community_weight_scale=0.1)
    graph_fig = make_graph_figure(
        g=g,
        pos=pos,
        node_to_comm=node_to_comm,
        modes=modes,
        seed_nodes=seed_nodes,
        title="Synthetic Causal Graph: GraphRAG vs BunnyRAG Lambda Sweep",
    )

    out_path = Path(args.output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_combined_html(vector_fig=vector_fig, graph_fig=graph_fig, output_html=out_path)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
