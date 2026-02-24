from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    build_query_vector,
    build_undirected_graph,
    load_graph_payload,
    load_node_embeddings,
    parse_csv_list,
    rank_nodes_by_query_similarity,
    weighted_graph_distance_rank,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic GraphRAG using vector-space query + graph-hop expansion."
    )
    parser.add_argument("--graph-path", required=True, help="Path to Bunny-compatible graph JSON.")
    parser.add_argument("--vectors-path", required=True, help="Path to vectors JSON.")
    parser.add_argument(
        "--query-vertices",
        default=None,
        help="Comma-separated vertex IDs used as the query set (example: '0,4,12').",
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
    parser.add_argument("--seed-k", type=int, default=5, help="Number of starting seed nodes.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of expanded nodes to return.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=3.0,
        help="Maximum weighted graph distance from seeds (Dijkstra with edge cost=1/weight).",
    )
    parser.add_argument(
        "--output-path",
        default="synthetic_bunny/output/synthetic_graphrag_results.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    if args.seed_k <= 0:
        raise ValueError("--seed-k must be a positive integer.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")

    node_ids, edges, _ = load_graph_payload(args.graph_path)
    embeddings = load_node_embeddings(args.vectors_path, node_ids)

    query_vertices = parse_csv_list(args.query_vertices) if args.query_vertices else []
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

    all_ranked = rank_nodes_by_query_similarity(embeddings, query_vec)
    seed_nodes = [node for node, _ in all_ranked[: args.seed_k]]

    g = build_undirected_graph(node_ids, edges)
    expanded = weighted_graph_distance_rank(
        g=g,
        seed_nodes=seed_nodes,
        embeddings=embeddings,
        query_vector=query_vec,
        max_distance=args.max_distance,
    )[: args.top_k]

    payload = {
        "graph_path": args.graph_path,
        "vectors_path": args.vectors_path,
        "query_mode": query_mode,
        "query_vertices": query_vertices,
        "query_random_points": args.query_random_points,
        "query_seed": args.query_seed if args.query_random_points else None,
        "seed_k": args.seed_k,
        "top_k": args.top_k,
        "max_distance": args.max_distance,
        "seed_nodes": seed_nodes,
        "results": [
            {
                "rank": i + 1,
                "node_id": node_id,
                "graph_distance": float(graph_distance),
                "query_similarity": float(sim),
            }
            for i, (node_id, graph_distance, sim) in enumerate(expanded)
        ],
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
