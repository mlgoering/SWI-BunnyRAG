from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from common import (
    build_query_vector,
    build_undirected_graph,
    cosine,
    effective_resistance,
    load_graph_payload,
    load_node_embeddings,
    parse_csv_floats,
    parse_csv_list,
    rank_nodes_by_query_similarity,
    weighted_graph_distance_rank,
)
from synthetic_bunnyrag import bunny_rank


def to_float_key(value: float) -> str:
    return f"{value:.3f}"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lambda sweep for synthetic BunnyRAG vs synthetic GraphRAG baseline."
    )
    parser.add_argument("--graph-path", required=True, help="Path to Bunny-compatible graph JSON.")
    parser.add_argument("--vectors-path", required=True, help="Path to vectors JSON.")
    parser.add_argument(
        "--query-vertices",
        default=None,
        help="Comma-separated query vertex IDs (example: '0,4,12').",
    )
    parser.add_argument(
        "--query-random-points",
        type=int,
        default=None,
        help=(
            "Enable random-query mode. Value is accepted for backward compatibility, "
            "but one random unit vector is now used."
        ),
    )
    parser.add_argument(
        "--query-seed",
        type=int,
        default=123,
        help="Random seed used when --query-random-points is set.",
    )
    parser.add_argument("--seed-k", type=int, default=5, help="Number of seed nodes.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k nodes per lambda.")
    parser.add_argument("--lambdas", default="-0.2,-0.1,0.0,0.1,0.2", help="CSV lambda values.")
    parser.add_argument(
        "--graphrag-max-distance",
        type=float,
        default=3.0,
        help="Baseline GraphRAG max weighted distance (edge cost=1/weight).",
    )
    parser.add_argument(
        "--output-dir",
        default="synthetic_bunny/output/lambda_sweep",
        help="Output directory.",
    )
    args = parser.parse_args()

    node_ids, edges, _ = load_graph_payload(args.graph_path)
    embeddings = load_node_embeddings(args.vectors_path, node_ids)
    query_vertices = parse_csv_list(args.query_vertices) if args.query_vertices else []
    lambdas = parse_csv_floats(args.lambdas)

    if bool(query_vertices) == bool(args.query_random_points):
        raise ValueError(
            "Choose exactly one query mode: --query-vertices OR --query-random-points."
        )
    if query_vertices:
        query_vec = build_query_vector(query_vertices, embeddings)
        query_mode = "vertex_subset"
    else:
        first_key = next(iter(embeddings))
        dim = int(embeddings[first_key].shape[0])
        rng = np.random.default_rng(args.query_seed)
        query_vec = rng.random(size=dim)
        query_vec /= np.linalg.norm(query_vec)
        query_mode = "random_sphere_points"
    ranked_by_query = rank_nodes_by_query_similarity(embeddings, query_vec)
    seed_nodes = [node for node, _ in ranked_by_query[: args.seed_k]]
    seed_set = set(seed_nodes)

    g = build_undirected_graph(node_ids, edges)
    baseline_ranked = weighted_graph_distance_rank(
        g=g,
        seed_nodes=seed_nodes,
        embeddings=embeddings,
        query_vector=query_vec,
        max_distance=args.graphrag_max_distance,
    )[: args.top_k]
    graphrag_top = [node for node, _, _ in baseline_ranked]
    graphrag_set = set(graphrag_top)

    label_to_id, resistance = effective_resistance(node_ids, edges)

    selected_nodes_by_lambda: Dict[str, List[Dict[str, object]]] = {}
    overlap_rows_for_report: List[Tuple[float, int, float]] = []

    for lam in lambdas:
        ranked = bunny_rank(
            node_ids=node_ids,
            embeddings=embeddings,
            label_to_id=label_to_id,
            resistance=resistance,
            seed_nodes=seed_nodes,
            labda=lam,
            top_k=args.top_k,
        )
        rows: List[Dict[str, object]] = []
        node_ids_ordered: List[str] = []

        for rank, (node_id, utility, avg_cond, avg_seed_cos) in enumerate(ranked, start=1):
            q_sim = cosine(query_vec, embeddings[node_id])
            node_ids_ordered.append(node_id)
            rows.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "utility_score": float(utility),
                    "query_similarity": float(q_sim),
                    "avg_normalized_conductance": float(avg_cond),
                    "avg_seed_cosine": float(avg_seed_cos),
                }
            )

        selected_nodes_by_lambda[to_float_key(lam)] = rows

        bunny_set = set(node_ids_ordered)
        overlap = bunny_set & graphrag_set
        overlap_rows_for_report.append(
            (lam, len(overlap), float(jaccard(bunny_set, graphrag_set)))
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_path = out_dir / "synthetic_bunny_lambda_selected_nodes.json"
    selected_path.write_text(json.dumps(selected_nodes_by_lambda, indent=2), encoding="utf-8")

    graphrag_path = out_dir / "synthetic_graphrag_topk_selected_nodes.json"
    graphrag_payload = {
        "query_mode": query_mode,
        "query_vertices": query_vertices,
        "query_random_points": args.query_random_points,
        "query_seed": args.query_seed if args.query_random_points else None,
        "seed_nodes": seed_nodes,
        "top_k": args.top_k,
        "max_distance": args.graphrag_max_distance,
        "nodes": [
            {
                "rank": i + 1,
                "node_id": node_id,
                "graph_distance": float(graph_distance),
                "query_similarity": float(sim),
            }
            for i, (node_id, graph_distance, sim) in enumerate(baseline_ranked)
        ],
    }
    graphrag_path.write_text(json.dumps(graphrag_payload, indent=2), encoding="utf-8")

    report_path = out_dir / "synthetic_bunny_lambda_sweep_report.txt"
    lines = [
        "Synthetic BunnyRAG Lambda Sweep Report",
        f"Graph: {args.graph_path}",
        f"Vectors: {args.vectors_path}",
        f"Query mode: {query_mode}",
        f"Query vertices: {query_vertices}",
        f"Query random points: {args.query_random_points}",
        f"Query seed: {args.query_seed if args.query_random_points else 'n/a'}",
        f"Seed nodes: {seed_nodes}",
        f"Lambdas: {sorted(lambdas)}",
        "",
        "GraphRAG baseline top-k:",
    ]
    for i, (node_id, graph_distance, sim) in enumerate(baseline_ranked, start=1):
        lines.append(
            f"- rank={i}: node={node_id}, graph_distance={graph_distance:.6f}, "
            f"query_sim={sim:.6f}"
        )
    lines.append("")
    lines.append("Bunny vs GraphRAG overlap by lambda:")
    for lam, overlap_count, overlap_jaccard in overlap_rows_for_report:
        lines.append(
            f"- lambda={lam:+.3f}: overlap={overlap_count}, "
            f"jaccard={overlap_jaccard:.4f}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {selected_path}")
    print(f"Wrote: {graphrag_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
