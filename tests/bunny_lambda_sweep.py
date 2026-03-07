from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from sentence_transformers import util


def to_float_key(value: float) -> str:
    return f"{value:.3f}"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def parse_lambdas(raw: str) -> List[float]:
    values = [part.strip() for part in raw.split(",")]
    parsed: List[float] = []
    for value in values:
        if not value:
            continue
        parsed.append(float(value))
    if not parsed:
        raise ValueError("No lambda values were provided.")
    return parsed


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def causalrag_top_k_nodes(
    builder: "CausalGraphBuilder",
    query: str,
    top_k: int,
    threshold: float,
) -> List[Tuple[str, float]]:
    q_emb = builder.encoder.encode(query, convert_to_tensor=True)
    scores: List[Tuple[str, float]] = []
    for node_id, emb in builder.node_embeddings.items():
        sim = float(util.pytorch_cos_sim(q_emb, emb).item())
        if sim >= threshold:
            scores.append((node_id, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    bunny_dir = repo_root / "Bunny_Rags"
    default_graph = "Bunny_Rags/causal_math_graph_llm.json"
    default_kb = "Data_generation/wiki_math_knowledge_base_api.json"
    default_output = "tests/output"

    parser = argparse.ArgumentParser(
        description="Run BunnyRAG lambda sweep for a configurable graph and query."
    )
    parser.add_argument(
        "--graph-path",
        default=default_graph,
        help="Path to Bunny-compatible graph JSON (absolute or repo-relative).",
    )
    parser.add_argument(
        "--knowledge-base-path",
        default=default_kb,
        help="Path to chunked text JSON (kept for dataset parity with v2 chains).",
    )
    parser.add_argument(
        "--query",
        default="What happens when the circumcenter is on the side of the triangle?",
        help="Query used for retrieval scoring.",
    )
    parser.add_argument(
        "--lambdas",
        default="-0.2,-0.1,0.0,0.1,0.2",
        help="Comma-separated list of lambda values, e.g. '-0.2,-0.1,0.0,0.1,0.2'.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k nodes per lambda.")
    parser.add_argument(
        "--top-k-components",
        type=int,
        default=10,
        help="Top-k nodes used for component-term output per lambda.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help="Directory for output files (absolute or repo-relative).",
    )
    parser.add_argument(
        "--causal-top-k",
        type=int,
        default=5,
        help="Top-k semantic nodes to retrieve with CausalRAG-style baseline.",
    )
    parser.add_argument(
        "--causal-threshold",
        type=float,
        default=0.5,
        help="Minimum cosine threshold for CausalRAG-style baseline retrieval.",
    )
    args = parser.parse_args()

    graph_path = resolve_path(repo_root, args.graph_path)
    knowledge_base_path = resolve_path(repo_root, args.knowledge_base_path)
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(bunny_dir))
    from builder import CausalGraphBuilder  # noqa: E402
    from bunny_retriever import BunnyPathRetriever  # noqa: E402

    query = args.query
    lambdas = parse_lambdas(args.lambdas)
    top_k = args.top_k
    top_k_components = args.top_k_components

    builder = CausalGraphBuilder()
    loaded = builder.load(str(graph_path))
    if loaded is False:
        raise RuntimeError(f"Failed to load graph at {graph_path}")

    retriever = BunnyPathRetriever(builder)
    if builder.encoder is None:
        raise RuntimeError("Embedding encoder was not initialized.")

    q_emb = builder.encoder.encode(query, convert_to_tensor=True)

    selected_nodes_by_lambda: Dict[str, List[Dict[str, object]]] = {}
    summary_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    causal_comparison_rows: List[Dict[str, object]] = []

    causal_top_nodes = causalrag_top_k_nodes(
        builder=builder,
        query=query,
        top_k=args.causal_top_k,
        threshold=args.causal_threshold,
    )
    causal_top_node_ids = [node_id for node_id, _ in causal_top_nodes]
    causal_top_node_set = set(causal_top_node_ids)

    # Compute per-node component terms used by the utility score:
    # avg_normalized_conductance(v) and avg_seed_cosine(v).
    label_to_id, resistance_matrix = retriever.build_effective_resistance(str(graph_path))
    seed_list = retriever.retrieve_nodes(query)
    seed_ids = [
        node_sp
        for node_sp, _ in seed_list
        if node_sp in label_to_id and node_sp in builder.node_embeddings
    ]
    seed_set = set(seed_ids)

    if not seed_ids:
        raise RuntimeError("No seed nodes were found for component-term analysis.")

    resistance_zero_eps = 1e-12
    candidate_pair_data: Dict[str, List[Tuple[float, float]]] = {}
    all_raw_conductances: List[float] = []

    for node_id, emb in builder.node_embeddings.items():
        if node_id not in label_to_id or node_id in seed_set:
            continue

        i = label_to_id[node_id]
        pair_data: List[Tuple[float, float]] = []
        for seed_id in seed_ids:
            j = label_to_id[seed_id]
            resistance = float(resistance_matrix[i, j])

            if not math.isfinite(resistance):
                raw_conductance = 0.0
            elif resistance <= resistance_zero_eps:
                raise ValueError(
                    f"Zero effective resistance encountered between candidate '{node_id}' "
                    f"and source '{seed_id}'."
                )
            else:
                raw_conductance = 1.0 / resistance
                all_raw_conductances.append(raw_conductance)

            sim = float(util.pytorch_cos_sim(builder.node_embeddings[seed_id], emb).item())
            sim = max(0.0, min(1.0, sim))
            pair_data.append((raw_conductance, sim))

        candidate_pair_data[node_id] = pair_data

    min_c = min(all_raw_conductances) if all_raw_conductances else 0.0
    max_c = max(all_raw_conductances) if all_raw_conductances else 0.0
    denom = max_c - min_c

    component_terms_by_node: Dict[str, Dict[str, float]] = {}
    for node_id, pair_data in candidate_pair_data.items():
        norm_conductances: List[float] = []
        seed_similarities: List[float] = []

        for raw_conductance, sim in pair_data:
            if denom > 0.0:
                c_norm = (raw_conductance - min_c) / denom
            elif max_c > 0.0:
                c_norm = 1.0
            else:
                c_norm = 0.0
            norm_conductances.append(c_norm)
            seed_similarities.append(sim)

        avg_norm_conductance = sum(norm_conductances) / len(norm_conductances)
        avg_seed_cosine = sum(seed_similarities) / len(seed_similarities)
        component_terms_by_node[node_id] = {
            "avg_normalized_conductance": avg_norm_conductance,
            "avg_seed_cosine": avg_seed_cosine,
        }

    for lam in lambdas:
        ranked_nodes: List[Tuple[str, float]] = retriever.retrieve_nodes_part2(
            query=query,
            top_k=top_k,
            labda=lam,
            json_path=str(graph_path),
        )
        ranked_nodes_top10: List[Tuple[str, float]] = retriever.retrieve_nodes_part2(
            query=query,
            top_k=top_k_components,
            labda=lam,
            json_path=str(graph_path),
        )

        rows_for_lambda: List[Dict[str, object]] = []
        query_sims: List[float] = []
        node_ids_in_order: List[str] = []

        for rank, (node_id, utility_score) in enumerate(ranked_nodes, start=1):
            node_text = builder.node_text.get(node_id, node_id)
            node_emb = builder.node_embeddings.get(node_id)
            query_sim = float("nan")
            if node_emb is not None:
                query_sim = float(util.pytorch_cos_sim(q_emb, node_emb).item())
                query_sims.append(query_sim)

            rows_for_lambda.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "node_text": node_text,
                    "utility_score": float(utility_score),
                    "query_similarity": query_sim,
                }
            )
            node_ids_in_order.append(node_id)

        selected_nodes_by_lambda[to_float_key(lam)] = rows_for_lambda
        avg_query_similarity = float(sum(query_sims) / len(query_sims)) if query_sims else float("nan")

        summary_rows.append(
            {
                "lambda": lam,
                "top_k": top_k,
                "num_selected": len(rows_for_lambda),
                "avg_query_similarity": avg_query_similarity,
                "selected_node_ids_in_order": "|".join(node_ids_in_order),
            }
        )

        selected_set = set(node_ids_in_order)
        overlap = selected_set & causal_top_node_set
        causal_comparison_rows.append(
            {
                "lambda": lam,
                "bunny_top_k": top_k,
                "causal_top_k": args.causal_top_k,
                "overlap_count": len(overlap),
                "overlap_jaccard": jaccard(selected_set, causal_top_node_set),
                "overlap_node_ids": "|".join(sorted(overlap)),
            }
        )

        for rank, (node_id, utility_score) in enumerate(ranked_nodes_top10, start=1):
            node_text = builder.node_text.get(node_id, node_id)
            components = component_terms_by_node.get(
                node_id,
                {"avg_normalized_conductance": float("nan"), "avg_seed_cosine": float("nan")},
            )
            component_rows.append(
                {
                    "lambda": lam,
                    "rank": rank,
                    "node_id": node_id,
                    "node_text": node_text,
                    "utility_score": float(utility_score),
                    "avg_normalized_conductance": float(components["avg_normalized_conductance"]),
                    "avg_seed_cosine": float(components["avg_seed_cosine"]),
                }
            )

    # Add CausalRAG baseline nodes to component-terms output for side-by-side comparison.
    for rank, (node_id, query_similarity) in enumerate(causal_top_nodes, start=1):
        node_text = builder.node_text.get(node_id, node_id)
        components = component_terms_by_node.get(
            node_id,
            {"avg_normalized_conductance": float("nan"), "avg_seed_cosine": float("nan")},
        )
        component_rows.append(
            {
                "lambda": "causalrag",
                "rank": rank,
                "node_id": node_id,
                "node_text": node_text,
                # For causal baseline rows, store query similarity in utility_score column.
                "utility_score": float(query_similarity),
                "avg_normalized_conductance": float(components["avg_normalized_conductance"]),
                "avg_seed_cosine": float(components["avg_seed_cosine"]),
            }
        )

    # Global checks requested by the experiment.
    ordered_lambdas = sorted(lambdas)
    avg_sim_by_lambda = {
        row["lambda"]: row["avg_query_similarity"] for row in summary_rows
    }

    node_sets = []
    for lam in ordered_lambdas:
        key = to_float_key(lam)
        node_sets.append({item["node_id"] for item in selected_nodes_by_lambda[key]})

    selections_differ = len({tuple(sorted(s)) for s in node_sets}) > 1

    monotonic_nonincreasing = True
    for i in range(1, len(ordered_lambdas)):
        prev_lam = ordered_lambdas[i - 1]
        curr_lam = ordered_lambdas[i]
        prev_avg = avg_sim_by_lambda[prev_lam]
        curr_avg = avg_sim_by_lambda[curr_lam]
        if not (math.isnan(prev_avg) or math.isnan(curr_avg)):
            if curr_avg > prev_avg + 1e-12:
                monotonic_nonincreasing = False
                break

    # Save detailed selected nodes.
    selected_nodes_path = output_dir / "bunny_lambda_selected_nodes.json"
    selected_nodes_path.write_text(json.dumps(selected_nodes_by_lambda, indent=2), encoding="utf-8")

    # Save CausalRAG baseline top-k.
    causal_topk_path = output_dir / "causal_topk_selected_nodes.json"
    causal_topk_payload = {
        "query": query,
        "graph_path": str(graph_path),
        "knowledge_base_path": str(knowledge_base_path),
        "causal_top_k": args.causal_top_k,
        "causal_threshold": args.causal_threshold,
        "nodes": [
            {
                "rank": rank,
                "node_id": node_id,
                "node_text": builder.node_text.get(node_id, node_id),
                "query_similarity": float(score),
            }
            for rank, (node_id, score) in enumerate(causal_top_nodes, start=1)
        ],
    }
    causal_topk_path.write_text(json.dumps(causal_topk_payload, indent=2), encoding="utf-8")

    # Save summary CSV.
    summary_csv_path = output_dir / "bunny_lambda_sweep_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda",
                "top_k",
                "num_selected",
                "avg_query_similarity",
                "selected_node_ids_in_order",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save Bunny-vs-Causal overlap summary.
    causal_compare_csv_path = output_dir / "bunny_vs_causal_topk_overlap.csv"
    with causal_compare_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda",
                "bunny_top_k",
                "causal_top_k",
                "overlap_count",
                "overlap_jaccard",
                "overlap_node_ids",
            ],
        )
        writer.writeheader()
        writer.writerows(causal_comparison_rows)

    # Save a short plain-English report with pairwise overlap info.
    report_lines = [
        "BunnyRAG Lambda Sweep Report",
        f"Query: {query}",
        f"Graph path: {graph_path}",
        f"Knowledge-base path (reference): {knowledge_base_path}",
        f"Lambdas: {ordered_lambdas}",
        f"Top-k selected per lambda: {top_k}",
        f"Causal baseline top-k: {args.causal_top_k}",
        f"Causal baseline threshold: {args.causal_threshold}",
        "",
        "Summary checks:",
        f"- Selections differ across lambdas: {selections_differ}",
        f"- Avg query similarity is non-increasing as lambda increases: {monotonic_nonincreasing}",
        "",
        "CausalRAG baseline selected nodes:",
    ]
    for rank, (node_id, score) in enumerate(causal_top_nodes, start=1):
        node_text = builder.node_text.get(node_id, node_id)
        report_lines.append(f"- rank={rank}: {node_text} (id={node_id}, sim={score:.6f})")

    report_lines.append("")
    report_lines.append("Bunny-vs-Causal top-k overlap by lambda:")
    for row in causal_comparison_rows:
        report_lines.append(
            f"- lambda={row['lambda']:+.1f}: "
            f"overlap_count={row['overlap_count']}, "
            f"jaccard={row['overlap_jaccard']:.4f}"
        )

    report_lines.append("")
    report_lines.append("Average query similarity by lambda:")
    for lam in ordered_lambdas:
        report_lines.append(f"- lambda={lam:+.1f}: {avg_sim_by_lambda[lam]:.6f}")

    report_lines.append("")
    report_lines.append("Consecutive-lambda Jaccard overlap of selected node sets:")
    for i in range(len(ordered_lambdas) - 1):
        lam_a = ordered_lambdas[i]
        lam_b = ordered_lambdas[i + 1]
        set_a = {item["node_id"] for item in selected_nodes_by_lambda[to_float_key(lam_a)]}
        set_b = {item["node_id"] for item in selected_nodes_by_lambda[to_float_key(lam_b)]}
        report_lines.append(
            f"- ({lam_a:+.1f}, {lam_b:+.1f}) -> {jaccard(set_a, set_b):.4f}"
        )

    report_path = output_dir / "bunny_lambda_sweep_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Save top-10 component terms per lambda.
    component_csv_path = output_dir / "bunny_lambda_top10_component_terms.csv"
    with component_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda",
                "rank",
                "node_id",
                "node_text",
                "utility_score",
                "avg_normalized_conductance",
                "avg_seed_cosine",
            ],
        )
        writer.writeheader()
        writer.writerows(component_rows)

    print(f"Wrote: {selected_nodes_path}")
    print(f"Wrote: {causal_topk_path}")
    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote: {causal_compare_csv_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {component_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
