import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def random_unit_vector(dim: int, rng: random.Random, non_negative_orthant: bool = True) -> List[float]:
    # Sample then normalize to the unit sphere surface.
    # - non_negative_orthant=True: sample only in the non-negative orthant
    # - non_negative_orthant=False: sample across the full sphere
    if non_negative_orthant:
        values = [rng.random() for _ in range(dim)]
    else:
        # Normalized Gaussian samples are uniformly distributed on the sphere.
        values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return random_unit_vector(dim, rng, non_negative_orthant=non_negative_orthant)
    return [v / norm for v in values]


def random_spherical_graph(
    n: int = 50,
    dim: int = 3,
    scale_prob: float = 0.2,
    seed: Optional[int] = None,
    bidirectional: bool = True,
    non_negative_orthant: bool = True,
    edge_weight_mode: str = "unit",
) -> Tuple[List[List[float]], List[Tuple[str, str, float]]]:
    """
    Generate a random spherical graph.

    Each node gets a random unit vector in R^dim.
    For each i<j, add an edge with probability:
      p = scale_prob * (1 + cos_theta) * 0.5

    Conductance weight assignment:
      - edge_weight_mode='unit': all sampled edges get conductance 1.0 (topology-first mode)
      - edge_weight_mode='random': each sampled edge gets Uniform(0,1) conductance

    Returns:
      vectors: node embedding vectors
      edges: list of (src, dst, weight) with string node IDs
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if dim < 2:
        raise ValueError("dim must be at least 2.")
    if scale_prob < 0.0:
        raise ValueError("scale_prob must be non-negative.")
    if edge_weight_mode not in {"unit", "random"}:
        raise ValueError("edge_weight_mode must be 'unit' or 'random'.")

    rng = random.Random(seed)
    vectors = [
        random_unit_vector(dim, rng, non_negative_orthant=non_negative_orthant)
        for _ in range(n)
    ]

    edges: List[Tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_theta = sum(vectors[i][k] * vectors[j][k] for k in range(dim))
            cos_theta = max(-1.0, min(1.0, cos_theta))
            p = scale_prob * (1.0 + cos_theta) * 0.5
            p = max(0.0, min(1.0, p))
            if rng.random() < p:
                if edge_weight_mode == "random":
                    weight = rng.random()
                else:
                    weight = 1.0
                src = str(i)
                dst = str(j)
                edges.append((src, dst, weight))
                if bidirectional:
                    edges.append((dst, src, weight))

    return vectors, edges


def to_bunny_graph_json(n: int, edges: List[Tuple[str, str, float]]) -> Dict[str, object]:
    nodes = {str(i): str(i) for i in range(n)}
    return {
        "nodes": nodes,
        "variants": {},
        "edges": [[src, dst, {"weight": float(weight)}] for src, dst, weight in edges],
    }


def component_sizes(n: int, edges: List[Tuple[str, str, float]]) -> List[int]:
    adj: Dict[str, set] = defaultdict(set)
    for src, dst, _ in edges:
        adj[src].add(dst)
        adj[dst].add(src)

    seen = set()
    sizes: List[int] = []
    for i in range(n):
        node = str(i)
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def visualize_graph(
    n: int,
    edges: List[Tuple[str, str, float]],
    output_path: str,
    title: str = "Random Spherical Graph",
    inter_community_weight_scale: float = 0.2,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "Visualization requires matplotlib and networkx. "
            "Install them (for example: pip install matplotlib networkx)."
        ) from exc

    g = nx.Graph()
    g.add_nodes_from(str(i) for i in range(n))
    for src, dst, weight in edges:
        if g.has_edge(src, dst):
            continue
        g.add_edge(src, dst, weight=float(weight))

    communities = list(nx.community.greedy_modularity_communities(g))
    node_to_comm: Dict[str, int] = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = idx

    palette = plt.get_cmap("tab20", max(1, len(communities)))
    node_colors = [palette(node_to_comm.get(node, 0)) for node in g.nodes()]
    node_sizes = [40 + 15 * g.degree(node) for node in g.nodes()]

    # Community-aware layout:
    # 1) initialize each community around a distinct center on a circle
    # 2) reduce inter-community edge pull in a layout-only graph
    # This makes same-community nodes appear spatially grouped.
    k_comm = max(1, len(communities))
    radius = 3.0
    centers: Dict[int, Tuple[float, float]] = {}
    for idx in range(k_comm):
        angle = 2.0 * math.pi * idx / k_comm
        centers[idx] = (radius * math.cos(angle), radius * math.sin(angle))

    rng = random.Random(42)
    init_pos: Dict[str, Tuple[float, float]] = {}
    for node in g.nodes():
        comm_idx = node_to_comm.get(node, 0)
        cx, cy = centers[comm_idx]
        init_pos[node] = (
            cx + rng.uniform(-0.35, 0.35),
            cy + rng.uniform(-0.35, 0.35),
        )

    layout_g = nx.Graph()
    layout_g.add_nodes_from(g.nodes())
    for u, v, data in g.edges(data=True):
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
        k=1.5 / math.sqrt(max(1, n)),
        iterations=350,
    )

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_edges(g, pos, alpha=0.18, width=0.8)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, alpha=0.92)

    plt.title(f"{title}\nNodes={n}, Edges={g.number_of_edges()}, Communities={len(communities)}")
    plt.axis("off")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a random spherical graph in Bunny/GraphRAG JSON format."
    )
    parser.add_argument("--n", type=int, default=50, help="Number of nodes.")
    parser.add_argument("--dim", type=int, default=3, help="Vector dimension.")
    parser.add_argument(
        "--scale-prob",
        type=float,
        default=0.2,
        help="Edge probability scaling factor.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=True,
        help="Add each sampled edge in both directions (default: enabled).",
    )
    parser.add_argument(
        "--no-bidirectional",
        dest="bidirectional",
        action="store_false",
        help="Add only one directed edge per sampled pair.",
    )
    parser.add_argument(
        "--output-path",
        default="Graph_Algorithm/random_spherical_bunny_graph.json",
        help="Output graph JSON path.",
    )
    parser.add_argument(
        "--vectors-output-path",
        default="Graph_Algorithm/random_spherical_vectors.json",
        help="Output path for raw vectors JSON.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional PNG path for community-colored graph visualization.",
    )
    parser.add_argument(
        "--cluster-separation",
        type=float,
        default=0.2,
        help=(
            "Scale factor for inter-community edge pull in visualization only. "
            "Smaller values separate clusters more strongly."
        ),
    )
    parser.add_argument(
        "--vector-space",
        choices=("orthant", "sphere"),
        default="orthant",
        help=(
            "Vector sampling mode: 'orthant' samples in the non-negative orthant "
            "(existing behavior), 'sphere' samples over the full unit sphere."
        ),
    )
    parser.add_argument(
        "--edge-weight-mode",
        choices=("unit", "random"),
        default="unit",
        help=(
            "Edge conductance assignment for sampled topology: "
            "'unit' keeps topology-only graphs, 'random' restores legacy random conductances."
        ),
    )
    args = parser.parse_args()

    vectors, edges = random_spherical_graph(
        n=args.n,
        dim=args.dim,
        scale_prob=args.scale_prob,
        seed=args.seed,
        bidirectional=args.bidirectional,
        non_negative_orthant=(args.vector_space == "orthant"),
        edge_weight_mode=args.edge_weight_mode,
    )

    sizes = component_sizes(args.n, edges)
    if len(sizes) != 1:
        largest = sizes[0] if sizes else 0
        raise RuntimeError(
            "Generated graph is disconnected. "
            f"components={len(sizes)}, largest_component={largest}/{args.n}. "
            "Increase --scale-prob or --n, then retry."
        )

    graph_payload = to_bunny_graph_json(args.n, edges)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")

    vectors_payload = {
        "n": args.n,
        "dim": args.dim,
        "scale_prob": args.scale_prob,
        "seed": args.seed,
        "vector_space": args.vector_space,
        "edge_weight_mode": args.edge_weight_mode,
        "vectors": vectors,
    }
    vectors_path = Path(args.vectors_output_path)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_path.write_text(json.dumps(vectors_payload, indent=2), encoding="utf-8")

    edge_pairs = len(edges) // 2 if args.bidirectional else len(edges)
    print(f"Wrote graph: {output_path}")
    print(f"Wrote vectors: {vectors_path}")
    print(f"Nodes: {args.n}")
    print(f"Edges stored: {len(edges)} (sampled pairs: {edge_pairs})")
    print(f"Weak components: {len(sizes)}")
    print(f"Largest component size: {sizes[0] if sizes else 0}")

    if args.plot_path:
        visualize_graph(
            n=args.n,
            edges=edges,
            output_path=args.plot_path,
            title="Random Spherical Graph (Community Colored)",
            inter_community_weight_scale=args.cluster_separation,
        )
        print(f"Wrote plot: {args.plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
