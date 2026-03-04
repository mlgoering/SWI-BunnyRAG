from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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
    build_query_vector,
    build_undirected_graph,
    cosine,
    effective_resistance,
    load_graph_payload,
    load_node_embeddings,
    parse_csv_floats,
    parse_csv_list,
    rank_nodes_by_query_similarity,
    reweight_edges_by_mode,
    weighted_graph_distance_rank,
)
from synthetic_bunnyrag import bunny_rank


VARIANT_ORDER_ALL = ["random", "cosine_beta", "query_aware"]
VARIANT_SEED_OFFSETS = {"unit": 0, "random": 17, "cosine_beta": 37, "query_aware": 59}


def to_float_key(value: float) -> str:
    return f"{value:.3f}"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def resolve_variants(raw_mode: str) -> List[str]:
    if raw_mode == "all":
        return list(VARIANT_ORDER_ALL)
    return [raw_mode]


def reweight_for_variant(
    *,
    variant: str,
    edges: List[Tuple[str, str, float]],
    embeddings: Dict[str, np.ndarray],
    query_vec: np.ndarray,
    base_seed: int,
    cosine_beta_alpha: float,
    cosine_beta_kappa: float,
    cosine_beta_offset: float,
    cosine_beta_scale: float,
    cosine_beta_clip_min: float,
    cosine_beta_clip_max: float,
    query_theta0: float,
    query_theta1: float,
    query_theta2: float,
    query_theta3: float,
    target_mean_weight: float,
    min_weight: float,
    max_weight: float,
) -> List[Tuple[str, str, float]]:
    variant_seed = base_seed + VARIANT_SEED_OFFSETS.get(variant, 0)
    return reweight_edges_by_mode(
        edges,
        embeddings,
        mode=variant,
        random_seed=variant_seed,
        query_vector=query_vec if variant == "query_aware" else None,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lambda sweep for synthetic BunnyRAG vs synthetic GraphRAG baseline "
            "across selectable edge-weight variants."
        )
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
    parser.add_argument(
        "--query-vector-space",
        choices=("orthant", "sphere"),
        default="orthant",
        help=(
            "Random query-vector sampling mode when --query-random-points is used. "
            "'orthant' samples non-negative coordinates; 'sphere' samples full-sphere vectors."
        ),
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
        "--edge-weight-mode",
        choices=("unit", "random", "cosine_beta", "query_aware", "all"),
        default="random",
        help=(
            "Edge conductance mode for retrieval: "
            "single mode or 'all' to run random + cosine_beta + query_aware."
        ),
    )
    parser.add_argument(
        "--edge-weight-seed",
        type=int,
        default=None,
        help="Base RNG seed for stochastic edge reweighting modes.",
    )
    parser.add_argument(
        "--cosine-beta-alpha",
        type=float,
        default=DEFAULT_COSINE_BETA_ALPHA,
        help="Exponent on cosine affinity for cosine_beta weighting.",
    )
    parser.add_argument(
        "--cosine-beta-kappa",
        type=float,
        default=DEFAULT_COSINE_BETA_KAPPA,
        help="Concentration parameter for cosine_beta Beta sampling.",
    )
    parser.add_argument(
        "--cosine-beta-offset",
        type=float,
        default=DEFAULT_COSINE_BETA_OFFSET,
        help="Additive floor for cosine_beta mean conductance.",
    )
    parser.add_argument(
        "--cosine-beta-scale",
        type=float,
        default=DEFAULT_COSINE_BETA_SCALE,
        help="Multiplicative scale for cosine_beta mean conductance.",
    )
    parser.add_argument(
        "--cosine-beta-clip-min",
        type=float,
        default=DEFAULT_COSINE_BETA_CLIP_MIN,
        help="Lower clip for cosine_beta mean before Beta sampling.",
    )
    parser.add_argument(
        "--cosine-beta-clip-max",
        type=float,
        default=DEFAULT_COSINE_BETA_CLIP_MAX,
        help="Upper clip for cosine_beta mean before Beta sampling.",
    )
    parser.add_argument(
        "--query-theta0",
        type=float,
        default=DEFAULT_QUERY_THETA0,
        help="Bias term for query-aware edge reweighting.",
    )
    parser.add_argument(
        "--query-theta1",
        type=float,
        default=DEFAULT_QUERY_THETA1,
        help="cos(x_i,x_j) coefficient for query-aware edge reweighting.",
    )
    parser.add_argument(
        "--query-theta2",
        type=float,
        default=DEFAULT_QUERY_THETA2,
        help="query endpoint relevance coefficient for query-aware edge reweighting.",
    )
    parser.add_argument(
        "--query-theta3",
        type=float,
        default=DEFAULT_QUERY_THETA3,
        help="interaction coefficient for query-aware edge reweighting.",
    )
    parser.add_argument(
        "--target-mean-weight",
        type=float,
        default=DEFAULT_TARGET_MEAN_WEIGHT,
        help="Target mean conductance used for mode normalization.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=DEFAULT_MIN_WEIGHT,
        help="Minimum clipped conductance for generated weighted modes.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=DEFAULT_MAX_WEIGHT,
        help="Maximum clipped conductance for generated weighted modes.",
    )
    parser.add_argument(
        "--output-dir",
        default="synthetic_bunny/output/lambda_sweep",
        help="Output directory.",
    )
    args = parser.parse_args()

    if args.seed_k <= 0:
        raise ValueError("--seed-k must be a positive integer.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")

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
        if args.query_vector_space == "sphere":
            query_vec = rng.normal(loc=0.0, scale=1.0, size=dim)
        else:
            query_vec = rng.random(size=dim)
        norm = np.linalg.norm(query_vec)
        if norm <= 0.0:
            raise ValueError("Random query vector norm is zero.")
        query_vec /= norm
        query_mode = "random_sphere_points"

    ranked_by_query = rank_nodes_by_query_similarity(embeddings, query_vec)
    seed_nodes = [node for node, _ in ranked_by_query[: args.seed_k]]

    base_seed = args.edge_weight_seed if args.edge_weight_seed is not None else args.query_seed
    variants = resolve_variants(args.edge_weight_mode)
    variants_payload: Dict[str, Dict[str, object]] = {}

    for variant in variants:
        edges_variant = reweight_for_variant(
            variant=variant,
            edges=edges,
            embeddings=embeddings,
            query_vec=query_vec,
            base_seed=base_seed,
            cosine_beta_alpha=args.cosine_beta_alpha,
            cosine_beta_kappa=args.cosine_beta_kappa,
            cosine_beta_offset=args.cosine_beta_offset,
            cosine_beta_scale=args.cosine_beta_scale,
            cosine_beta_clip_min=args.cosine_beta_clip_min,
            cosine_beta_clip_max=args.cosine_beta_clip_max,
            query_theta0=args.query_theta0,
            query_theta1=args.query_theta1,
            query_theta2=args.query_theta2,
            query_theta3=args.query_theta3,
            target_mean_weight=args.target_mean_weight,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
        )

        g = build_undirected_graph(node_ids, edges_variant)
        baseline_ranked = weighted_graph_distance_rank(
            g=g,
            seed_nodes=seed_nodes,
            embeddings=embeddings,
            query_vector=query_vec,
            max_distance=args.graphrag_max_distance,
        )[: args.top_k]
        graphrag_top = [node for node, _, _ in baseline_ranked]
        graphrag_set = set(graphrag_top)

        label_to_id, resistance = effective_resistance(node_ids, edges_variant)
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

        graphrag_payload = {
            "query_mode": query_mode,
            "query_vertices": query_vertices,
            "query_random_points": args.query_random_points,
            "query_seed": args.query_seed if args.query_random_points else None,
            "query_vector_space": (
                args.query_vector_space if args.query_random_points else None
            ),
            "seed_nodes": seed_nodes,
            "top_k": args.top_k,
            "max_distance": args.graphrag_max_distance,
            "edge_weight_mode": variant,
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

        variants_payload[variant] = {
            "edge_weight_mode": variant,
            "graphrag": graphrag_payload,
            "bunny_by_lambda": selected_nodes_by_lambda,
            "overlap_by_lambda": [
                {
                    "lambda": float(lam),
                    "overlap": int(overlap_count),
                    "jaccard": float(overlap_jaccard),
                }
                for lam, overlap_count, overlap_jaccard in overlap_rows_for_report
            ],
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants_path = out_dir / "synthetic_lambda_sweep_variants.json"
    variants_root_payload = {
        "graph_path": args.graph_path,
        "vectors_path": args.vectors_path,
        "query_mode": query_mode,
        "query_vertices": query_vertices,
        "query_random_points": args.query_random_points,
        "query_seed": args.query_seed if args.query_random_points else None,
        "query_vector_space": (
            args.query_vector_space if args.query_random_points else None
        ),
        "seed_nodes": seed_nodes,
        "seed_k": args.seed_k,
        "top_k": args.top_k,
        "lambdas": [float(x) for x in sorted(lambdas)],
        "graphrag_max_distance": args.graphrag_max_distance,
        "edge_weight_mode": args.edge_weight_mode,
        "edge_weight_seed": base_seed,
        "cosine_beta_alpha": args.cosine_beta_alpha,
        "cosine_beta_kappa": args.cosine_beta_kappa,
        "cosine_beta_offset": args.cosine_beta_offset,
        "cosine_beta_scale": args.cosine_beta_scale,
        "cosine_beta_clip_min": args.cosine_beta_clip_min,
        "cosine_beta_clip_max": args.cosine_beta_clip_max,
        "query_theta0": args.query_theta0,
        "query_theta1": args.query_theta1,
        "query_theta2": args.query_theta2,
        "query_theta3": args.query_theta3,
        "target_mean_weight": args.target_mean_weight,
        "min_weight": args.min_weight,
        "max_weight": args.max_weight,
        "variant_order": variants,
        "variants": variants_payload,
    }
    variants_path.write_text(
        json.dumps(variants_root_payload, indent=2), encoding="utf-8"
    )

    # Backward-compatible single-variant files:
    # when --edge-weight-mode all, keep legacy files pointed at the random variant.
    legacy_variant = variants[0]
    if args.edge_weight_mode == "all" and "random" in variants_payload:
        legacy_variant = "random"
    legacy_selected = variants_payload[legacy_variant]["bunny_by_lambda"]
    legacy_graphrag = variants_payload[legacy_variant]["graphrag"]

    selected_path = out_dir / "synthetic_bunny_lambda_selected_nodes.json"
    selected_path.write_text(json.dumps(legacy_selected, indent=2), encoding="utf-8")

    graphrag_path = out_dir / "synthetic_graphrag_topk_selected_nodes.json"
    graphrag_path.write_text(json.dumps(legacy_graphrag, indent=2), encoding="utf-8")

    report_path = out_dir / "synthetic_bunny_lambda_sweep_report.txt"
    lines = [
        "Synthetic BunnyRAG Lambda Sweep Report",
        f"Graph: {args.graph_path}",
        f"Vectors: {args.vectors_path}",
        f"Query mode: {query_mode}",
        f"Query vertices: {query_vertices}",
        f"Query random points: {args.query_random_points}",
        f"Query seed: {args.query_seed if args.query_random_points else 'n/a'}",
        f"Query vector space: {args.query_vector_space if args.query_random_points else 'n/a'}",
        f"Seed nodes: {seed_nodes}",
        f"Lambdas: {sorted(lambdas)}",
        f"Edge-weight mode: {args.edge_weight_mode}",
        f"Variant order: {variants}",
        "",
    ]
    for variant in variants:
        variant_data = variants_payload[variant]
        lines.append(f"[Variant: {variant}]")
        graphrag_variant = variant_data["graphrag"]
        lines.append("GraphRAG baseline top-k:")
        for row in graphrag_variant["nodes"]:
            lines.append(
                f"- rank={row['rank']}: node={row['node_id']}, "
                f"graph_distance={row['graph_distance']:.6f}, "
                f"query_sim={row['query_similarity']:.6f}"
            )
        lines.append("Bunny vs GraphRAG overlap by lambda:")
        for overlap_row in variant_data["overlap_by_lambda"]:
            lines.append(
                f"- lambda={overlap_row['lambda']:+.3f}: overlap={overlap_row['overlap']}, "
                f"jaccard={overlap_row['jaccard']:.4f}"
            )
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {selected_path}")
    print(f"Wrote: {graphrag_path}")
    print(f"Wrote: {variants_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
