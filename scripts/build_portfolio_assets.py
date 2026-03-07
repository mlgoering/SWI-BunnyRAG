#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild docs/portfolio assets used by GitHub Pages."
    )
    parser.add_argument(
        "--run-name",
        default="behavior_runner_n500_d6_cosine_beta_same_30x30_20260304",
        help="Run directory name under presentation/testing used for scatter export.",
    )
    parser.add_argument(
        "--skip-scatter",
        action="store_true",
        help="Skip rebuilding/copying the scatter PNG.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    print("[1/2] Building interactive portfolio page bundle...", flush=True)
    run_cmd(
        [
            sys.executable,
            "presentation/build_interactive_projection_google_sites.py",
            "--output-html",
            "docs/portfolio/interactive_projection.html",
        ],
        repo_root,
    )

    if args.skip_scatter:
        print("[2/2] Skipped scatter build (--skip-scatter).", flush=True)
        print("Done.")
        return 0

    print(f"[2/2] Building portfolio scatter figure for run: {args.run_name}", flush=True)
    run_cmd(
        [
            sys.executable,
            "presentation/build_behavior_single_scatter_lambda_gradient.py",
            "--run",
            args.run_name,
            "--formats",
            "png",
        ],
        repo_root,
    )

    scatter_source = (
        repo_root
        / "presentation"
        / "testing"
        / args.run_name
        / "single_scatter_lambda_gradient"
        / "single_scatter_lambda_gradient.png"
    )
    if scatter_source.exists():
        figure_dir = repo_root / "docs" / "portfolio" / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        scatter_target = figure_dir / "relevance_vs_coverage.png"
        shutil.copy2(scatter_source, scatter_target)
        print(f"Copied figure: {scatter_target}")
    else:
        print(f"Warning: Scatter PNG not found at: {scatter_source}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
