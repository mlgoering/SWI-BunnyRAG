from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx


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


def load_selection_modes(
    selected_nodes_path: str,
    graphrag_path: str,
) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    selected_payload = json.loads(Path(selected_nodes_path).read_text(encoding="utf-8"))
    graphrag_payload = json.loads(Path(graphrag_path).read_text(encoding="utf-8"))

    modes: List[Tuple[str, List[str]]] = []
    seed_nodes = [str(x) for x in graphrag_payload.get("seed_nodes", [])]
    graphrag_nodes = [
        str(item.get("node_id"))
        for item in graphrag_payload.get("nodes", [])
        if isinstance(item, dict) and item.get("node_id") is not None
    ]
    modes.append(("GraphRAG", graphrag_nodes))

    lambda_keys = list(selected_payload.keys())
    try:
        lambda_keys.sort(key=lambda k: float(k))
    except Exception:
        lambda_keys.sort()

    for key in lambda_keys:
        rows = selected_payload.get(key, [])
        node_ids = []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("node_id") is not None:
                    node_ids.append(str(row["node_id"]))
        label = f"\u03bb = {float(key):+.3f}" if _is_floatish(key) else f"\u03bb = {key}"
        modes.append((label, node_ids))
    return modes, seed_nodes


def _is_floatish(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


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


def build_plot(
    g: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    node_to_comm: Dict[str, int],
    modes: Sequence[Tuple[str, List[str]]],
    seed_nodes: Sequence[str],
    title: str,
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "This script requires plotly. Install with: pip install plotly"
        ) from exc

    comm_count = max(1, (max(node_to_comm.values()) + 1) if node_to_comm else 1)
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
    node_text: List[str] = []
    node_color: List[str] = []
    node_size: List[float] = []
    for n in g.nodes():
        nid = str(n)
        x, y = pos[nid]
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
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=0.0),
                opacity=0.25,
            ),
            text=node_text,
            hoverinfo="text",
            name="nodes",
            showlegend=False,
        )
    )

    # Seed nodes are always visible as bright red.
    seed_x: List[float] = []
    seed_y: List[float] = []
    seed_text: List[str] = []
    for node_id in seed_nodes:
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        seed_x.append(x)
        seed_y.append(y)
        seed_text.append(f"Seed node={node_id}")
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
            hovertext=seed_text,
            hoverinfo="text",
            name="seed_nodes",
            visible=True,
            showlegend=False,
        )
    )

    seed_set = set(seed_nodes)
    mode_trace_indices: List[int] = []
    for mode_label, selected in modes:
        xs: List[float] = []
        ys: List[float] = []
        texts: List[str] = []
        selected_non_seed = [n for n in selected if n not in seed_set]
        for node_id in selected_non_seed:
            if node_id not in pos:
                continue
            x, y = pos[node_id]
            xs.append(x)
            ys.append(y)
            texts.append(f"{mode_label}<br>node={node_id}")
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
                hovertext=texts,
                hoverinfo="text",
                name=mode_label,
                visible=False,
                showlegend=False,
            )
        )
        mode_trace_indices.append(len(fig.data) - 1)

    if mode_trace_indices:
        fig.data[mode_trace_indices[0]].visible = True

    steps = []
    for i, (mode_label, _) in enumerate(modes):
        visible = [True, True, True] + [False] * len(mode_trace_indices)
        visible[3 + i] = True
        steps.append(
            dict(
                method="update",
                label=mode_label,
                args=[
                    {"visible": visible},
                    {"title": f"{title}<br><sup>Mode: {mode_label}</sup>"},
                ],
            )
        )

    fig.update_layout(
        title=f"{title}<br><sup>Mode: {modes[0][0] if modes else 'N/A'}</sup>",
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
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize synthetic GraphRAG/Bunny lambda sweep with slider from GraphRAG "
            "to all computed lambda modes."
        )
    )
    parser.add_argument("--graph-path", required=True, help="Path to graph JSON.")
    parser.add_argument(
        "--selected-nodes-path",
        required=True,
        help="Path to synthetic_bunny_lambda_selected_nodes.json.",
    )
    parser.add_argument(
        "--graphrag-path",
        required=True,
        help="Path to synthetic_graphrag_topk_selected_nodes.json.",
    )
    parser.add_argument(
        "--output-html",
        default="synethetic_bunny/output/lambda_sweep_visualization.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--cluster-separation",
        type=float,
        default=0.1,
        help="Lower values make communities separate more strongly in layout.",
    )
    parser.add_argument(
        "--title",
        default="Synthetic Causal Graph: GraphRAG vs BunnyRAG Lambda Sweep",
        help="Plot title.",
    )
    args = parser.parse_args()

    g = load_graph(args.graph_path)
    modes, seed_nodes = load_selection_modes(args.selected_nodes_path, args.graphrag_path)
    if not modes:
        raise ValueError("No modes found to visualize.")

    pos, node_to_comm = clustered_layout(
        g=g,
        inter_community_weight_scale=args.cluster_separation,
    )
    fig = build_plot(
        g=g,
        pos=pos,
        node_to_comm=node_to_comm,
        modes=modes,
        seed_nodes=seed_nodes,
        title=args.title,
    )

    out = Path(args.output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
