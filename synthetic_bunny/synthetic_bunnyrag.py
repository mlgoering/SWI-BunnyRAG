from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from common import (
    build_query_vector,
    cosine,
    effective_resistance,
    load_graph_payload,
    load_node_embeddings,
    parse_csv_list,
    rank_nodes_by_query_similarity,
)


def bunny_rank(
    node_ids: List[str],
    embeddings: Dict[str, object],
    label_to_id: Dict[str, int],
    resistance: object,
    seed_nodes: List[str],
    labda: float,
    top_k: int,
) -> List[Tuple[str, float, float, float]]:
    seed_set = set(seed_nodes)
    resistance_zero_eps = 1e-12

    candidate_pair_data: Dict[str, List[Tuple[float, float]]] = {}
    all_raw_conductances: List[float] = []

    for node_id in node_ids:
        if node_id in seed_set:
            continue
        i = label_to_id[node_id]
        pair_data: List[Tuple[float, float]] = []
        for seed in seed_nodes:
            j = label_to_id[seed]
            r_vs = float(resistance[i, j])
            if not math.isfinite(r_vs):
                raw_conductance = 0.0
            elif r_vs <= resistance_zero_eps:
                raise ValueError(
                    f"Zero effective resistance encountered between '{node_id}' and '{seed}'."
                )
            else:
                raw_conductance = 1.0 / r_vs
                all_raw_conductances.append(raw_conductance)

            sim = cosine(embeddings[node_id], embeddings[seed])
            sim = max(0.0, min(1.0, float(sim)))
            pair_data.append((raw_conductance, sim))
        candidate_pair_data[node_id] = pair_data

    if not candidate_pair_data:
        return []

    min_c = min(all_raw_conductances) if all_raw_conductances else 0.0
    max_c = max(all_raw_conductances) if all_raw_conductances else 0.0
    denom = max_c - min_c

    ranked: List[Tuple[str, float, float, float]] = []
    for node_id, pair_data in candidate_pair_data.items():
        norm_conductances: List[float] = []
        sims: List[float] = []
        for raw_c, sim in pair_data:
            if denom > 0.0:
                c_norm = (raw_c - min_c) / denom
            elif max_c > 0.0:
                c_norm = 1.0
            else:
                c_norm = 0.0
            norm_conductances.append(float(c_norm))
            sims.append(float(sim))

        avg_cond = float(sum(norm_conductances) / len(norm_conductances))
        avg_seed_cos = float(sum(sims) / len(sims))
        score = avg_cond - labda * avg_seed_cos
        ranked.append((node_id, score, avg_cond, avg_seed_cos))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic BunnyRAG using effective resistance + vector penalties."
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
    parser.add_argument("--top-k", type=int, default=10, help="Number of ranked Bunny nodes.")
    parser.add_argument("--labda", type=float, default=0.02, help="Penalty coefficient.")
    parser.add_argument(
        "--output-path",
        default="synthetic_bunny/output/synthetic_bunnyrag_results.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

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

    label_to_id, resistance = effective_resistance(node_ids, edges)
    ranked = bunny_rank(
        node_ids=node_ids,
        embeddings=embeddings,
        label_to_id=label_to_id,
        resistance=resistance,
        seed_nodes=seed_nodes,
        labda=args.labda,
        top_k=args.top_k,
    )

    payload = {
        "graph_path": args.graph_path,
        "vectors_path": args.vectors_path,
        "query_mode": query_mode,
        "query_vertices": query_vertices,
        "query_random_points": args.query_random_points,
        "query_seed": args.query_seed if args.query_random_points else None,
        "seed_k": args.seed_k,
        "top_k": args.top_k,
        "labda": args.labda,
        "seed_nodes": seed_nodes,
        "results": [
            {
                "rank": i + 1,
                "node_id": node_id,
                "utility_score": float(score),
                "avg_normalized_conductance": float(avg_cond),
                "avg_seed_cosine": float(avg_seed_cos),
            }
            for i, (node_id, score, avg_cond, avg_seed_cos) in enumerate(ranked)
        ],
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
