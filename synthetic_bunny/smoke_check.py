from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def assert_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Output file is empty: {path}")


def remove_path_if_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic Bunny smoke check: generate data, run lambda sweep, "
            "and build visualization."
        )
    )
    parser.add_argument("--n", type=int, default=75, help="Number of points/nodes.")
    parser.add_argument("--dim", type=int, default=4, help="Vector dimension.")
    parser.add_argument(
        "--scale-prob",
        type=float,
        default=0.1,
        help="Edge probability scaling for synthetic graph generation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graph generation seed.")
    parser.add_argument(
        "--query-seed",
        type=int,
        default=17,
        help="Random query seed used by synthetic_lambda_sweep.py.",
    )
    parser.add_argument("--seed-k", type=int, default=3, help="Number of seed nodes.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k nodes to return.")
    parser.add_argument(
        "--lambdas",
        default="0,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated lambda values for Bunny sweep.",
    )
    parser.add_argument(
        "--graphrag-max-distance",
        type=float,
        default=6.0,
        help="GraphRAG max weighted distance for baseline selection.",
    )
    parser.add_argument(
        "--output-root",
        default="synthetic_bunny/output/smoke_check",
        help="Root folder where smoke-check outputs are written.",
    )
    parser.add_argument(
        "--cleanup-run-outputs",
        action="store_true",
        default=True,
        help="Delete this smoke-check run's explicit output paths after validation.",
    )
    parser.add_argument(
        "--no-cleanup-run-outputs",
        dest="cleanup_run_outputs",
        action="store_false",
        help="Keep this smoke-check run's outputs on disk.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    synthetic_dir = repo_root / "synthetic_bunny"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / args.output_root / f"run_{run_id}"
    lambda_dir = run_dir / "lambda_sweep"

    graph_path = run_dir / "random_spherical_bunny_graph.json"
    vectors_path = run_dir / "random_spherical_vectors.json"
    selected_nodes_path = lambda_dir / "synthetic_bunny_lambda_selected_nodes.json"
    graphrag_path = lambda_dir / "synthetic_graphrag_topk_selected_nodes.json"
    report_path = lambda_dir / "synthetic_bunny_lambda_sweep_report.txt"
    html_path = lambda_dir / "synthetic_lambda_sweep_visualization.html"

    run_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            str(synthetic_dir / "generate_synthetic_data.py"),
            "--n",
            str(args.n),
            "--dim",
            str(args.dim),
            "--scale-prob",
            str(args.scale_prob),
            "--seed",
            str(args.seed),
            "--output-path",
            str(graph_path),
            "--vectors-output-path",
            str(vectors_path),
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            sys.executable,
            str(synthetic_dir / "synthetic_lambda_sweep.py"),
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--query-seed",
            str(args.query_seed),
            "--seed-k",
            str(args.seed_k),
            "--top-k",
            str(args.top_k),
            "--lambdas",
            args.lambdas,
            "--graphrag-max-distance",
            str(args.graphrag_max_distance),
            "--output-dir",
            str(lambda_dir),
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            sys.executable,
            str(synthetic_dir / "visualize_lambda_sweep.py"),
            "--graph-path",
            str(graph_path),
            "--selected-nodes-path",
            str(selected_nodes_path),
            "--graphrag-path",
            str(graphrag_path),
            "--output-html",
            str(html_path),
        ],
        cwd=repo_root,
    )

    assert_file(graph_path)
    assert_file(vectors_path)
    assert_file(selected_nodes_path)
    assert_file(graphrag_path)
    assert_file(report_path)
    assert_file(html_path)

    selected_payload = json.loads(selected_nodes_path.read_text(encoding="utf-8"))
    if not isinstance(selected_payload, dict) or not selected_payload:
        raise RuntimeError("Lambda selected-nodes output is malformed or empty.")

    if args.cleanup_run_outputs:
        removed = remove_path_if_exists(run_dir)
        if removed:
            print(f"Deleted smoke-check run outputs: {run_dir}")

    print("\nSmoke check passed.")
    if args.cleanup_run_outputs:
        print("Run outputs were cleaned up after validation.")
    else:
        print(f"Run folder: {run_dir}")
        print(f"Graph: {graph_path}")
        print(f"Vectors: {vectors_path}")
        print(f"Sweep report: {report_path}")
        print(f"Visualization: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
