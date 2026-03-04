from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

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
    DEFAULT_TARGET_MEAN_WEIGHT,
    reweight_edges_by_mode,
)


def _load_generator_module():
    repo_root = Path(__file__).resolve().parent.parent
    generator_path = repo_root / "Graph Algorithm" / "random_spherical_graph_generator.py"
    if not generator_path.exists():
        raise FileNotFoundError(f"Generator not found: {generator_path}")

    spec = importlib.util.spec_from_file_location(
        "random_spherical_graph_generator", generator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load generator module from: {generator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper for Graph Algorithm/random_spherical_graph_generator.py "
            "with synthetic_bunny-focused defaults."
        )
    )
    parser.add_argument("--n", type=int, default=75, help="Number of nodes.")
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
        default="synthetic_bunny/output/generated/random_spherical_bunny_graph.json",
        help="Output graph JSON path.",
    )
    parser.add_argument(
        "--vectors-output-path",
        default="synthetic_bunny/output/generated/random_spherical_vectors.json",
        help="Output path for vectors JSON.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional PNG output path for graph visualization.",
    )
    parser.add_argument(
        "--cluster-separation",
        type=float,
        default=0.2,
        help="Inter-community edge pull scale for visualization only.",
    )
    parser.add_argument(
        "--vector-space",
        choices=("orthant", "sphere"),
        default="orthant",
        help=(
            "Vector sampling mode: 'orthant' samples in the non-negative orthant, "
            "'sphere' samples over the full unit sphere."
        ),
    )
    parser.add_argument(
        "--edge-weight-mode",
        choices=("unit", "random", "cosine_beta"),
        default="unit",
        help=(
            "Conductance assignment after topology generation. "
            "'unit' keeps topology-only output, "
            "'random' assigns uniform random conductances, "
            "'cosine_beta' assigns cosine-correlated stochastic conductances."
        ),
    )
    parser.add_argument(
        "--edge-weight-seed",
        type=int,
        default=None,
        help=(
            "RNG seed for post-topology edge reweighting. "
            "Defaults to --seed when omitted."
        ),
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
    args = parser.parse_args()

    mod = _load_generator_module()
    vectors, edges = mod.random_spherical_graph(
        n=args.n,
        dim=args.dim,
        scale_prob=args.scale_prob,
        seed=args.seed,
        bidirectional=args.bidirectional,
        non_negative_orthant=(args.vector_space == "orthant"),
        edge_weight_mode="unit",
    )

    if args.edge_weight_mode != "unit":
        embeddings = {str(i): np.asarray(vectors[i], dtype=float) for i in range(len(vectors))}
        edges = reweight_edges_by_mode(
            edges,
            embeddings,
            mode=args.edge_weight_mode,
            random_seed=(args.edge_weight_seed if args.edge_weight_seed is not None else args.seed),
            cosine_beta_alpha=args.cosine_beta_alpha,
            cosine_beta_kappa=args.cosine_beta_kappa,
            cosine_beta_offset=args.cosine_beta_offset,
            cosine_beta_scale=args.cosine_beta_scale,
            cosine_beta_clip_min=args.cosine_beta_clip_min,
            cosine_beta_clip_max=args.cosine_beta_clip_max,
            target_mean_weight=args.target_mean_weight,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
        )

    sizes = mod.component_sizes(args.n, edges)
    if len(sizes) != 1:
        largest = sizes[0] if sizes else 0
        raise RuntimeError(
            "Generated graph is disconnected. "
            f"components={len(sizes)}, largest_component={largest}/{args.n}. "
            "Increase --scale-prob or --n, then retry."
        )

    graph_payload = mod.to_bunny_graph_json(args.n, edges)
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
        "edge_weight_seed": args.edge_weight_seed if args.edge_weight_seed is not None else args.seed,
        "cosine_beta_alpha": args.cosine_beta_alpha,
        "cosine_beta_kappa": args.cosine_beta_kappa,
        "cosine_beta_offset": args.cosine_beta_offset,
        "cosine_beta_scale": args.cosine_beta_scale,
        "cosine_beta_clip_min": args.cosine_beta_clip_min,
        "cosine_beta_clip_max": args.cosine_beta_clip_max,
        "target_mean_weight": args.target_mean_weight,
        "min_weight": args.min_weight,
        "max_weight": args.max_weight,
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
    print(f"Edge weighting mode: {args.edge_weight_mode}")
    print(f"Weak components: {len(sizes)}")
    print(f"Largest component size: {sizes[0] if sizes else 0}")

    if args.plot_path:
        mod.visualize_graph(
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
