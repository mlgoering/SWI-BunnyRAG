from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
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
    build_undirected_graph,
    cosine,
    effective_resistance,
    parse_csv_floats,
    rank_nodes_by_query_similarity,
    reweight_edges_by_mode,
    weighted_graph_distance_rank,
)
from synthetic_bunnyrag import bunny_rank


EPS = 1e-12
MIN_BASELINE_FOR_ENTROPY_LIFT = 0.05


def _load_generator_module():
    repo_root = Path(__file__).resolve().parent.parent
    generator_path = repo_root / "Graph_Algorithm" / "random_spherical_graph_generator.py"
    spec = importlib.util.spec_from_file_location(
        "random_spherical_graph_generator", generator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load generator module from: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_comm_graph_from_edges(
    node_ids: Sequence[str],
    edges: Sequence[Tuple[str, str, float]],
) -> nx.Graph:
    # Match visualization behavior: collapse duplicate undirected edges by max weight.
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for src, dst, w in edges:
        src = str(src)
        dst = str(dst)
        if src == dst:
            continue
        weight = float(w)
        if g.has_edge(src, dst):
            g[src][dst]["weight"] = max(float(g[src][dst]["weight"]), weight)
        else:
            g.add_edge(src, dst, weight=weight)
    return g


def _community_lookup(g: nx.Graph) -> Tuple[Dict[str, int], Dict[int, int]]:
    communities = list(nx.community.greedy_modularity_communities(g, weight="weight"))
    node_to_comm: Dict[str, int] = {}
    comm_sizes: Dict[int, int] = {}
    for idx, comm in enumerate(communities):
        comm_sizes[idx] = len(comm)
        for node in comm:
            node_to_comm[str(node)] = idx
    return node_to_comm, comm_sizes


def _random_query_vector(dim: int, seed: int, *, vector_space: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if vector_space == "sphere":
        vec = rng.normal(loc=0.0, scale=1.0, size=dim)
    else:
        vec = rng.random(size=dim)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("Random query vector norm is zero.")
    return vec / norm


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return float(h)


def _kl_selected_vs_graph(
    selected_counts: Dict[int, int],
    graph_counts: Dict[int, int],
    eps: float = EPS,
) -> float:
    total_sel = sum(selected_counts.values())
    total_graph = sum(graph_counts.values())
    if total_sel <= 0 or total_graph <= 0:
        return 0.0

    comm_ids = sorted(set(graph_counts.keys()) | set(selected_counts.keys()))
    p_vals: List[float] = []
    q_vals: List[float] = []
    for comm_id in comm_ids:
        p_vals.append(selected_counts.get(comm_id, 0) / total_sel)
        q_vals.append(graph_counts.get(comm_id, 0) / total_graph)

    p = np.asarray(p_vals, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _selection_metrics(
    selected_nodes: Sequence[str],
    query_vec: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    node_to_comm: Dict[str, int],
    comm_sizes: Dict[int, int],
) -> Dict[str, float]:
    selected = [str(x) for x in selected_nodes]
    k = len(selected)
    communities_total = len(comm_sizes)
    if k == 0 or communities_total == 0:
        return {
            "selected_k": float(k),
            "communities_total": float(communities_total),
            "represented_count": 0.0,
            "coverage": 0.0,
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "max_share": 0.0,
            "kl_selected_vs_graph": 0.0,
            "avg_query_similarity": 0.0,
        }

    counts: Dict[int, int] = {}
    sims: List[float] = []
    for node in selected:
        comm = node_to_comm.get(node)
        if comm is None:
            continue
        counts[comm] = counts.get(comm, 0) + 1
        sims.append(float(cosine(query_vec, embeddings[node])))

    represented_count = len(counts)
    coverage = represented_count / max(communities_total, 1)
    entropy = _entropy_from_counts(counts)
    entropy_norm = entropy / math.log(communities_total) if communities_total > 1 else 0.0
    max_share = max((c / max(k, 1) for c in counts.values()), default=0.0)
    kl = _kl_selected_vs_graph(counts, comm_sizes)
    avg_sim = float(mean(sims)) if sims else 0.0
    return {
        "selected_k": float(k),
        "communities_total": float(communities_total),
        "represented_count": float(represented_count),
        "coverage": float(coverage),
        "entropy": float(entropy),
        "entropy_normalized": float(entropy_norm),
        "max_share": float(max_share),
        "kl_selected_vs_graph": float(kl),
        "avg_query_similarity": float(avg_sim),
    }


def _safe_lift(value: float, base: float) -> float:
    return float((value - base) / max(abs(base), EPS))


def _safe_lift_with_floor(value: float, base: float, floor: float) -> float | str:
    if abs(base) < floor:
        return "N/A"
    return float((value - base) / max(abs(base), EPS))


def _safe_improvement_lower_better(value: float, base: float) -> float:
    return float((base - value) / max(abs(base), EPS))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic Bunny behavior experiments across multiple datasets and queries, "
            "then write row-level metrics for Excel reporting."
        )
    )
    parser.add_argument("--num-datasets", type=int, default=20, help="Number of datasets.")
    parser.add_argument(
        "--queries-per-dataset", type=int, default=20, help="Queries evaluated per dataset."
    )
    parser.add_argument("--n", type=int, default=500, help="Nodes per synthetic graph.")
    parser.add_argument("--dim", type=int, default=6, help="Vector dimension.")
    parser.add_argument("--scale-prob", type=float, default=0.025, help="Graph scale probability.")
    parser.add_argument(
        "--vector-space",
        choices=("orthant", "sphere"),
        default="sphere",
        help=(
            "Vector sampling mode for synthetic graph generation: "
            "'orthant' uses non-negative vectors; 'sphere' samples full-sphere vectors."
        ),
    )
    parser.add_argument(
        "--dataset-seed-start", type=int, default=42, help="Base seed used for dataset generation."
    )
    parser.add_argument(
        "--query-seed-start", type=int, default=17, help="Base seed used for random queries."
    )
    parser.add_argument(
        "--max-query-attempts-per-dataset",
        type=int,
        default=5000,
        help=(
            "Maximum query seeds sampled per dataset to find accepted queries."
        ),
    )
    parser.add_argument(
        "--seed-community-policy",
        choices=("same", "mixed"),
        default="same",
        help=(
            "Query acceptance policy for seed-node communities. "
            "'same' keeps only queries whose top seed_k nodes are all in one community; "
            "'mixed' allows cross-community seed sets."
        ),
    )
    parser.add_argument("--seed-k", type=int, default=3, help="Number of seed nodes.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k selections.")
    parser.add_argument(
        "--lambdas",
        default="-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated lambda values.",
    )
    parser.add_argument(
        "--graphrag-max-distance",
        type=float,
        default=6.0,
        help="GraphRAG max weighted distance.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=30,
        help="Random baseline samples per (dataset, query).",
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=50,
        help="Max seed retries per dataset to get a connected graph.",
    )
    parser.add_argument(
        "--require-graphrag-selection",
        action="store_true",
        help=(
            "If set, queries where GraphRAG selects zero nodes are retried and not "
            "counted toward accepted query totals."
        ),
    )
    parser.add_argument(
        "--sim-delta-min-vs-graphrag",
        "--sim-retention-min-vs-graphrag",
        dest="sim_delta_min_vs_graphrag",
        type=float,
        default=0.0,
        help=(
            "Pass threshold for query-similarity delta vs GraphRAG "
            "(bunny_avg_query_similarity - graphrag_avg_query_similarity)."
        ),
    )
    parser.add_argument(
        "--sim-delta-min-vs-random",
        "--sim-retention-min-vs-random",
        dest="sim_delta_min_vs_random",
        type=float,
        default=0.0,
        help=(
            "Pass threshold for query-similarity delta vs Random "
            "(bunny_avg_query_similarity - random_avg_query_similarity)."
        ),
    )
    parser.add_argument(
        "--edge-weight-mode",
        choices=("unit", "random", "cosine_beta", "query_aware"),
        default="random",
        help=(
            "Edge conductance mode used during selection. "
            "'query_aware' reweights per query; other modes are reweighted once per dataset."
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
        default="synthetic_bunny/output/behavior_runner",
        help="Output directory.",
    )
    args = parser.parse_args()

    if args.num_datasets <= 0:
        raise ValueError("--num-datasets must be positive.")
    if args.queries_per_dataset <= 0:
        raise ValueError("--queries-per-dataset must be positive.")
    if args.seed_k <= 0:
        raise ValueError("--seed-k must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.random_trials <= 0:
        raise ValueError("--random-trials must be positive.")

    lambdas = parse_csv_floats(args.lambdas)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = _load_generator_module()
    rows: List[Dict[str, object]] = []
    dataset_meta: List[Dict[str, object]] = []

    query_counter = 0
    graphrag_empty_retries_total = 0
    require_same_seed_community = args.seed_community_policy == "same"
    for dataset_idx in range(args.num_datasets):
        dataset_id = f"dataset_{dataset_idx + 1:03d}"
        found = False
        chosen_seed = None
        vectors = None
        edges = None
        for attempt in range(args.max_generation_attempts):
            candidate_seed = args.dataset_seed_start + dataset_idx + attempt
            cand_vectors, cand_edges = generator.random_spherical_graph(
                n=args.n,
                dim=args.dim,
                scale_prob=args.scale_prob,
                seed=candidate_seed,
                bidirectional=True,
                non_negative_orthant=(args.vector_space == "orthant"),
                edge_weight_mode="unit",
            )
            sizes = generator.component_sizes(args.n, cand_edges)
            if len(sizes) == 1:
                found = True
                chosen_seed = candidate_seed
                vectors = cand_vectors
                edges = cand_edges
                break
        if not found or vectors is None or edges is None or chosen_seed is None:
            raise RuntimeError(
                f"Failed to generate a connected graph for {dataset_id} after "
                f"{args.max_generation_attempts} attempts."
            )

        node_ids = [str(i) for i in range(args.n)]
        embeddings = {
            str(i): np.asarray(vectors[i], dtype=float)
            for i in range(len(vectors))
        }
        topology_comm_graph = _build_comm_graph_from_edges(node_ids, edges)

        reweight_seed_base = (
            args.edge_weight_seed if args.edge_weight_seed is not None else chosen_seed
        ) + dataset_idx * 1_009

        static_dist_graph = None
        static_comm_graph = None
        static_node_to_comm = None
        static_comm_sizes = None
        static_label_to_id = None
        static_resistance = None
        if args.edge_weight_mode != "query_aware":
            static_edges = reweight_edges_by_mode(
                edges,
                embeddings,
                mode=args.edge_weight_mode,
                random_seed=reweight_seed_base,
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
            static_dist_graph = build_undirected_graph(node_ids, static_edges)
            static_comm_graph = _build_comm_graph_from_edges(node_ids, static_edges)
            static_node_to_comm, static_comm_sizes = _community_lookup(static_comm_graph)
            static_label_to_id, static_resistance = effective_resistance(node_ids, static_edges)

        dataset_meta.append(
            {
                "dataset_id": dataset_id,
                "dataset_seed": chosen_seed,
                "n": args.n,
                "dim": args.dim,
                "scale_prob": args.scale_prob,
                "edge_weight_mode": args.edge_weight_mode,
                "communities_total": (
                    len(static_comm_sizes) if static_comm_sizes is not None else "varies_by_query"
                ),
                "community_sizes_sorted": (
                    sorted(static_comm_sizes.values()) if static_comm_sizes is not None else "varies_by_query"
                ),
                "edges_stored": len(edges),
                "edges_undirected": int(
                    static_comm_graph.number_of_edges()
                    if static_comm_graph is not None
                    else topology_comm_graph.number_of_edges()
                ),
            }
        )

        accepted_queries = 0
        attempted_queries = 0
        graphrag_empty_retries = 0
        while accepted_queries < args.queries_per_dataset:
            if attempted_queries >= args.max_query_attempts_per_dataset:
                raise RuntimeError(
                    f"Failed to collect {args.queries_per_dataset} accepted queries for "
                    f"{dataset_id} within {args.max_query_attempts_per_dataset} attempts."
                )
            attempted_queries += 1
            query_seed = args.query_seed_start + query_counter
            query_counter += 1
            query_vec = _random_query_vector(
                args.dim,
                query_seed,
                vector_space=args.vector_space,
            )

            ranked = rank_nodes_by_query_similarity(embeddings, query_vec)
            seed_nodes = [node for node, _ in ranked[: args.seed_k]]
            if len(seed_nodes) != args.seed_k:
                continue

            if args.edge_weight_mode == "query_aware":
                query_edges = reweight_edges_by_mode(
                    edges,
                    embeddings,
                    mode="query_aware",
                    random_seed=reweight_seed_base + query_seed,
                    query_vector=query_vec,
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
                dist_graph = build_undirected_graph(node_ids, query_edges)
                comm_graph = _build_comm_graph_from_edges(node_ids, query_edges)
                node_to_comm, comm_sizes = _community_lookup(comm_graph)
                label_to_id, resistance = effective_resistance(node_ids, query_edges)
            else:
                assert static_dist_graph is not None
                assert static_node_to_comm is not None
                assert static_comm_sizes is not None
                assert static_label_to_id is not None
                assert static_resistance is not None
                dist_graph = static_dist_graph
                node_to_comm = static_node_to_comm
                comm_sizes = static_comm_sizes
                label_to_id = static_label_to_id
                resistance = static_resistance

            seed_communities = {
                node_to_comm[node]
                for node in seed_nodes
                if node in node_to_comm
            }
            if require_same_seed_community and len(seed_communities) != 1:
                continue

            seed_set = set(seed_nodes)

            graphrag_ranked = weighted_graph_distance_rank(
                g=dist_graph,
                seed_nodes=seed_nodes,
                embeddings=embeddings,
                query_vector=query_vec,
                max_distance=args.graphrag_max_distance,
            )[: args.top_k]
            graphrag_nodes = [node for node, _, _ in graphrag_ranked]
            if args.require_graphrag_selection and not graphrag_nodes:
                graphrag_empty_retries += 1
                graphrag_empty_retries_total += 1
                continue

            accepted_queries += 1
            query_id = f"q{accepted_queries:03d}"
            gr_metrics = _selection_metrics(
                selected_nodes=graphrag_nodes,
                query_vec=query_vec,
                embeddings=embeddings,
                node_to_comm=node_to_comm,
                comm_sizes=comm_sizes,
            )

            random_candidates = [node for node in node_ids if node not in seed_set]
            random_metrics_trials: List[Dict[str, float]] = []
            rng = np.random.default_rng(query_seed + 99_999)
            random_pick_count = min(args.top_k, len(random_candidates))
            for _ in range(args.random_trials):
                picks = rng.choice(
                    random_candidates, size=random_pick_count, replace=False
                ).tolist()
                random_metrics_trials.append(
                    _selection_metrics(
                        selected_nodes=picks,
                        query_vec=query_vec,
                        embeddings=embeddings,
                        node_to_comm=node_to_comm,
                        comm_sizes=comm_sizes,
                    )
                )
            rand_metrics: Dict[str, float] = {}
            for key in random_metrics_trials[0].keys():
                rand_metrics[key] = float(mean(m[key] for m in random_metrics_trials))

            common_row = {
                "dataset_id": dataset_id,
                "dataset_seed": chosen_seed,
                "query_id": query_id,
                "query_seed": query_seed,
                "top_k": args.top_k,
                "seed_k": args.seed_k,
                "communities_total": int(gr_metrics["communities_total"]),
            }

            for method, metric_obj, lam in (
                ("graphrag", gr_metrics, "NA"),
                ("random", rand_metrics, "NA"),
            ):
                row = dict(common_row)
                row.update(
                    {
                        "method": method,
                        "lambda": lam,
                        "selected_k": int(round(metric_obj["selected_k"])),
                        "represented_count": int(round(metric_obj["represented_count"])),
                        "coverage": metric_obj["coverage"],
                        "entropy": metric_obj["entropy"],
                        "entropy_normalized": metric_obj["entropy_normalized"],
                        "max_share": metric_obj["max_share"],
                        "kl_selected_vs_graph": metric_obj["kl_selected_vs_graph"],
                        "avg_query_similarity": metric_obj["avg_query_similarity"],
                        "lift_coverage_vs_graphrag": "N/A",
                        "lift_entropy_vs_graphrag": "N/A",
                        "delta_entropy_vs_graphrag": "N/A",
                        "improve_max_share_vs_graphrag": "N/A",
                        "improve_kl_vs_graphrag": "N/A",
                        "delta_query_similarity_vs_graphrag": "N/A",
                        "lift_coverage_vs_random": "N/A",
                        "lift_entropy_vs_random": "N/A",
                        "delta_entropy_vs_random": "N/A",
                        "improve_max_share_vs_random": "N/A",
                        "improve_kl_vs_random": "N/A",
                        "delta_query_similarity_vs_random": "N/A",
                        "pass_diversity_vs_graphrag": "N/A",
                        "pass_relevance_vs_graphrag": "N/A",
                        "pass_overall_vs_graphrag": "N/A",
                        "pass_diversity_vs_random": "N/A",
                        "pass_relevance_vs_random": "N/A",
                        "pass_overall_vs_random": "N/A",
                    }
                )
                rows.append(row)

            for lam in lambdas:
                bunny_ranked = bunny_rank(
                    node_ids=node_ids,
                    embeddings=embeddings,
                    label_to_id=label_to_id,
                    resistance=resistance,
                    seed_nodes=seed_nodes,
                    labda=lam,
                    top_k=args.top_k,
                )
                bunny_nodes = [node for node, _, _, _ in bunny_ranked]
                b_metrics = _selection_metrics(
                    selected_nodes=bunny_nodes,
                    query_vec=query_vec,
                    embeddings=embeddings,
                    node_to_comm=node_to_comm,
                    comm_sizes=comm_sizes,
                )

                lift_cov_gr = _safe_lift(b_metrics["coverage"], gr_metrics["coverage"])
                delta_ent_gr = float(
                    b_metrics["entropy_normalized"] - gr_metrics["entropy_normalized"]
                )
                lift_ent_gr = _safe_lift_with_floor(
                    b_metrics["entropy_normalized"], gr_metrics["entropy_normalized"]
                    , MIN_BASELINE_FOR_ENTROPY_LIFT
                )
                imp_share_gr = _safe_improvement_lower_better(
                    b_metrics["max_share"], gr_metrics["max_share"]
                )
                imp_kl_gr = _safe_improvement_lower_better(
                    b_metrics["kl_selected_vs_graph"], gr_metrics["kl_selected_vs_graph"]
                )
                sim_delta_gr = float(
                    b_metrics["avg_query_similarity"] - gr_metrics["avg_query_similarity"]
                )

                lift_cov_rd = _safe_lift(b_metrics["coverage"], rand_metrics["coverage"])
                delta_ent_rd = float(
                    b_metrics["entropy_normalized"] - rand_metrics["entropy_normalized"]
                )
                lift_ent_rd = _safe_lift_with_floor(
                    b_metrics["entropy_normalized"], rand_metrics["entropy_normalized"]
                    , MIN_BASELINE_FOR_ENTROPY_LIFT
                )
                imp_share_rd = _safe_improvement_lower_better(
                    b_metrics["max_share"], rand_metrics["max_share"]
                )
                imp_kl_rd = _safe_improvement_lower_better(
                    b_metrics["kl_selected_vs_graph"], rand_metrics["kl_selected_vs_graph"]
                )
                sim_delta_rd = float(
                    b_metrics["avg_query_similarity"] - rand_metrics["avg_query_similarity"]
                )

                pass_div_gr = (
                    lift_cov_gr >= 0.0
                    and delta_ent_gr >= 0.0
                    and imp_share_gr >= 0.0
                    and imp_kl_gr >= 0.0
                )
                pass_rel_gr = sim_delta_gr >= args.sim_delta_min_vs_graphrag
                pass_div_rd = (
                    lift_cov_rd >= 0.0
                    and delta_ent_rd >= 0.0
                    and imp_share_rd >= 0.0
                    and imp_kl_rd >= 0.0
                )
                pass_rel_rd = sim_delta_rd >= args.sim_delta_min_vs_random

                row = dict(common_row)
                row.update(
                    {
                        "method": "bunny",
                        "lambda": float(lam),
                        "selected_k": int(round(b_metrics["selected_k"])),
                        "represented_count": int(round(b_metrics["represented_count"])),
                        "coverage": b_metrics["coverage"],
                        "entropy": b_metrics["entropy"],
                        "entropy_normalized": b_metrics["entropy_normalized"],
                        "max_share": b_metrics["max_share"],
                        "kl_selected_vs_graph": b_metrics["kl_selected_vs_graph"],
                        "avg_query_similarity": b_metrics["avg_query_similarity"],
                        "lift_coverage_vs_graphrag": lift_cov_gr,
                        "lift_entropy_vs_graphrag": lift_ent_gr,
                        "delta_entropy_vs_graphrag": delta_ent_gr,
                        "improve_max_share_vs_graphrag": imp_share_gr,
                        "improve_kl_vs_graphrag": imp_kl_gr,
                        "delta_query_similarity_vs_graphrag": sim_delta_gr,
                        "lift_coverage_vs_random": lift_cov_rd,
                        "lift_entropy_vs_random": lift_ent_rd,
                        "delta_entropy_vs_random": delta_ent_rd,
                        "improve_max_share_vs_random": imp_share_rd,
                        "improve_kl_vs_random": imp_kl_rd,
                        "delta_query_similarity_vs_random": sim_delta_rd,
                        "pass_diversity_vs_graphrag": pass_div_gr,
                        "pass_relevance_vs_graphrag": pass_rel_gr,
                        "pass_overall_vs_graphrag": bool(pass_div_gr and pass_rel_gr),
                        "pass_diversity_vs_random": pass_div_rd,
                        "pass_relevance_vs_random": pass_rel_rd,
                        "pass_overall_vs_random": bool(pass_div_rd and pass_rel_rd),
                    }
                )
                rows.append(row)

        dataset_meta[-1]["query_attempts"] = attempted_queries
        dataset_meta[-1]["accepted_queries"] = accepted_queries
        dataset_meta[-1]["graphrag_empty_retries"] = graphrag_empty_retries

    metadata = {
        "params": vars(args),
        "lambdas": [float(x) for x in lambdas],
        "graphrag_empty_retries_total": graphrag_empty_retries_total,
        "thresholds": {
            "min_lift_coverage_vs_graphrag": 0.0,
            "min_lift_entropy_vs_graphrag": 0.0,
            "min_improve_max_share_vs_graphrag": 0.0,
            "min_improve_kl_vs_graphrag": 0.0,
            "min_delta_query_similarity_vs_graphrag": args.sim_delta_min_vs_graphrag,
            "min_lift_coverage_vs_random": 0.0,
            "min_lift_entropy_vs_random": 0.0,
            "min_improve_max_share_vs_random": 0.0,
            "min_improve_kl_vs_random": 0.0,
            "min_delta_query_similarity_vs_random": args.sim_delta_min_vs_random,
        },
        "dataset_meta": dataset_meta,
    }

    rows_json = out_dir / "behavior_results_rows.json"
    rows_csv = out_dir / "behavior_results_rows.csv"
    metadata_json = out_dir / "behavior_metadata.json"

    rows_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not rows:
        raise RuntimeError("No results rows were generated.")

    fieldnames = list(rows[0].keys())
    with rows_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {rows_json}")
    print(f"Wrote: {rows_csv}")
    print(f"Wrote: {metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
