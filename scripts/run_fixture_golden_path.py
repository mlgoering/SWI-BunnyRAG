#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


SYNTH_TEST_FILES = [
    "tests/test_synthetic_bunnyrag.py",
    "tests/test_synthetic_graphrag.py",
    "tests/test_synthetic_lambda_sweep.py",
    "tests/test_visualize_lambda_sweep.py",
    "tests/test_generate_synthetic_data.py",
]


def run_cmd(cmd: list[str], repo_root: Path) -> None:
    completed = subprocess.run(cmd, cwd=repo_root)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_synth_smoke_tests(repo_root: Path) -> None:
    base_temp_root = repo_root / "pytest_temp_py"
    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    base_temp = base_temp_root / run_id
    base_temp.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "pytest", "-q", f"--basetemp={base_temp}"] + SYNTH_TEST_FILES
    completed = subprocess.run(cmd, cwd=repo_root)
    if completed.returncode == 0:
        shutil.rmtree(base_temp, ignore_errors=True)
        return
    raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run synthetic fixture golden path with optional smoke tests and docs publish."
    )
    parser.add_argument("--fixture-name", default="seed42_n500_dim6_sp0025")
    parser.add_argument(
        "--output-root",
        default="",
        help="Relative output root under repo root. Default: synthetic_bunny/output/golden_path_<fixture-name>",
    )
    parser.add_argument("--publish-to-docs", action="store_true")
    parser.add_argument("--skip-synth-tests", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    fixture_dir = repo_root / "synthetic_bunny" / "fixtures" / args.fixture_name
    graph_path = fixture_dir / "random_spherical_bunny_graph.json"
    vectors_path = fixture_dir / "random_spherical_vectors.json"

    if not args.output_root.strip():
        output_root = Path(f"synthetic_bunny/output/golden_path_{args.fixture_name}")
    else:
        output_root = Path(args.output_root)

    sweep_dir = repo_root / output_root / "lambda_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    variants_path = sweep_dir / "synthetic_lambda_sweep_variants.json"
    selected_path = sweep_dir / "synthetic_bunny_lambda_selected_nodes.json"
    graphrag_path = sweep_dir / "synthetic_graphrag_topk_selected_nodes.json"
    report_path = sweep_dir / "synthetic_bunny_lambda_sweep_report.txt"
    html_path = sweep_dir / "synthetic_lambda_sweep_visualization.html"

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing fixture graph: {graph_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing fixture vectors: {vectors_path}")

    if args.skip_synth_tests:
        print("[1/3] Skipping synthetic smoke tests (--skip-synth-tests).", flush=True)
    else:
        print("[1/3] Running synthetic smoke tests", flush=True)
        run_synth_smoke_tests(repo_root)

    print(f"[2/3] Running synthetic lambda sweep from fixture: {args.fixture_name}", flush=True)
    run_cmd(
        [
            sys.executable,
            "synthetic_bunny/synthetic_lambda_sweep.py",
            "--graph-path",
            str(graph_path),
            "--vectors-path",
            str(vectors_path),
            "--query-random-points",
            "1",
            "--query-seed",
            "17",
            "--query-vector-space",
            "sphere",
            "--seed-k",
            "3",
            "--top-k",
            "10",
            "--lambdas=-0.2,-0.1,0,0.1,0.2,0.3",
            "--graphrag-max-distance",
            "6.0",
            "--edge-weight-mode",
            "random",
            "--output-dir",
            str(sweep_dir),
        ],
        repo_root,
    )

    print("[3/3] Building interactive visualization HTML", flush=True)
    run_cmd(
        [
            sys.executable,
            "synthetic_bunny/visualize_lambda_sweep.py",
            "--graph-path",
            str(graph_path),
            "--variants-path",
            str(variants_path),
            "--vectors-path",
            str(vectors_path),
            "--plot-height",
            "675",
            "--output-html",
            str(html_path),
            "--title",
            "Synthetic Fixture Golden Path",
        ],
        repo_root,
    )

    if args.publish_to_docs:
        docs_dir = repo_root / "docs" / "portfolio" / "golden_path"
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs_html_path = docs_dir / "synthetic_lambda_sweep_visualization.html"
        shutil.copy2(html_path, docs_html_path)
        print(f"Copied HTML to docs: {docs_html_path}")

    print("\nGolden path outputs:")
    print(f"- {variants_path}")
    print(f"- {selected_path}")
    print(f"- {graphrag_path}")
    print(f"- {report_path}")
    print(f"- {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
