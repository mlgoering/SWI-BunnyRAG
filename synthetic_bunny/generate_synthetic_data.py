from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


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
    args = parser.parse_args()

    mod = _load_generator_module()
    vectors, edges = mod.random_spherical_graph(
        n=args.n,
        dim=args.dim,
        scale_prob=args.scale_prob,
        seed=args.seed,
        bidirectional=args.bidirectional,
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
