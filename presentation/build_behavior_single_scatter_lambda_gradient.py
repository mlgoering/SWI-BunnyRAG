from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

METHOD_GRAPHRAG = "graphrag"
METHOD_RANDOM = "random"
METHOD_BUNNY = "bunny"

COLOR_GRAPHRAG = "#E69F00"  # orange
COLOR_RANDOM = "#009E73"  # bluish green
LAMBDA_COLOR_MIN = "#0072B2"  # blue
LAMBDA_COLOR_MAX = "#CC79A7"  # reddish purple
# Nonlinear lambda->color position mapping.
# Larger gaps here mean faster hue change for those lambda intervals.
NONLINEAR_LAMBDA_ANCHORS = {
    -0.2: 0.00,
    -0.1: 0.30,
    0.0: 0.42,
    0.1: 0.54,
    0.2: 0.82,
    0.3: 1.00,
}

METHOD_DISPLAY = {
    METHOD_GRAPHRAG: "CausalRAG",
    METHOD_RANDOM: "random",
    METHOD_BUNNY: "bunny",
}


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.upper() in {"NA", "N/A", "NONE", "NULL"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_title_from_run_name(run_name: str) -> str:
    pattern = (
        r"^behavior_runner_n(?P<n>\d+)_d(?P<dim>\d+)_"
        r"(?P<weighting>.+?)_(?P<seeds>same|mixed)_\d+x\d+"
    )
    m = re.match(pattern, run_name)
    if not m:
        return run_name
    n = m.group("n")
    dim = m.group("dim")
    weighting = m.group("weighting")
    seeds = m.group("seeds")
    weighting_label = {
        "cosine_beta": "similarity based weights",
        "random": "random weights",
    }.get(weighting, f"{weighting} weighting")
    seed_label = {
        "same": "single-community restricted seeds",
        "mixed": "unrestricted seeds",
    }.get(seeds, f"{seeds} seeds")
    return f"{n} nodes in {dim} dimensions, {weighting_label}, {seed_label}"


def discover_run_dirs(testing_root: Path, only_runs: Sequence[str]) -> List[Path]:
    allowed = set(only_runs)
    out: List[Path] = []
    for child in sorted(testing_root.iterdir()):
        if not child.is_dir():
            continue
        if only_runs and child.name not in allowed:
            continue
        if (child / "behavior_results_rows.csv").exists():
            out.append(child)
    return out


def aggregate_dataset_means(rows_path: Path) -> List[Dict[str, object]]:
    sums: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    with rows_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_id = str(row.get("dataset_id", "")).strip()
            dataset_seed = str(row.get("dataset_seed", "")).strip()
            method = str(row.get("method", "")).strip().lower()
            if not dataset_id or method not in {METHOD_GRAPHRAG, METHOD_RANDOM, METHOD_BUNNY}:
                continue

            sim = _to_float(row.get("avg_query_similarity"))
            ent = _to_float(row.get("entropy"))
            if sim is None or ent is None:
                continue

            lambda_key = "NA"
            if method == METHOD_BUNNY:
                lam = _to_float(row.get("lambda"))
                if lam is None:
                    continue
                lambda_key = f"{lam:.6g}"

            key = (dataset_id, dataset_seed, method, lambda_key)
            slot = sums.setdefault(key, {"sim_sum": 0.0, "ent_sum": 0.0, "count": 0.0})
            slot["sim_sum"] += sim
            slot["ent_sum"] += ent
            slot["count"] += 1.0

    out: List[Dict[str, object]] = []
    for (dataset_id, dataset_seed, method, lambda_key), slot in sorted(sums.items()):
        count = int(slot["count"])
        if count <= 0:
            continue
        out.append(
            {
                "dataset_id": dataset_id,
                "dataset_seed": dataset_seed,
                "method": method,
                "lambda": lambda_key,
                "query_count": count,
                "avg_similarity": slot["sim_sum"] / count,
                "avg_entropy": slot["ent_sum"] / count,
            }
        )
    return out


def map_lambda_to_color_position(value: float) -> float:
    keys = np.asarray(sorted(NONLINEAR_LAMBDA_ANCHORS.keys()), dtype=float)
    vals = np.asarray([NONLINEAR_LAMBDA_ANCHORS[k] for k in keys], dtype=float)
    pos = float(np.interp(value, keys, vals))
    return max(0.0, min(1.0, pos))


def filter_bunny_lambdas(
    rows: Sequence[Mapping[str, object]],
    lambda_min: float,
    lambda_max: float,
) -> List[Mapping[str, object]]:
    out: List[Mapping[str, object]] = []
    for row in rows:
        method = str(row["method"])
        if method != METHOD_BUNNY:
            out.append(row)
            continue
        lam = _to_float(str(row["lambda"]))
        if lam is None:
            continue
        if lam < lambda_min - 1e-12 or lam > lambda_max + 1e-12:
            continue
        out.append(row)
    return out


def plot_single_scatter(
    run_name: str,
    title_text: str,
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    lambda_min: float,
    lambda_max: float,
) -> None:
    graphrag_pts = [r for r in rows if str(r["method"]) == METHOD_GRAPHRAG]
    random_pts = [r for r in rows if str(r["method"]) == METHOD_RANDOM]
    bunny_pts = [r for r in rows if str(r["method"]) == METHOD_BUNNY]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    if graphrag_pts:
        ax.scatter(
            [float(r["avg_similarity"]) for r in graphrag_pts],
            [float(r["avg_entropy"]) for r in graphrag_pts],
            s=45,
            color=COLOR_GRAPHRAG,
            marker="^",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            label=METHOD_DISPLAY[METHOD_GRAPHRAG],
        )

    if random_pts:
        ax.scatter(
            [float(r["avg_similarity"]) for r in random_pts],
            [float(r["avg_entropy"]) for r in random_pts],
            s=45,
            color=COLOR_RANDOM,
            marker="s",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            label="random",
        )

    cmap = LinearSegmentedColormap.from_list("lambda_blue_purple", [LAMBDA_COLOR_MIN, LAMBDA_COLOR_MAX])
    norm = Normalize(vmin=0.0, vmax=1.0)
    if bunny_pts:
        bunny_lambdas = np.asarray([float(str(r["lambda"])) for r in bunny_pts], dtype=float)
        bunny_color_pos = np.asarray([map_lambda_to_color_position(v) for v in bunny_lambdas], dtype=float)
        bunny_x = np.asarray([float(r["avg_similarity"]) for r in bunny_pts], dtype=float)
        bunny_y = np.asarray([float(r["avg_entropy"]) for r in bunny_pts], dtype=float)
        sc = ax.scatter(
            bunny_x,
            bunny_y,
            s=30,
            c=bunny_color_pos,
            cmap=cmap,
            norm=norm,
            marker="o",
            alpha=0.72,
            edgecolors="black",
            linewidths=0.2,
            label="bunny (lambda)",
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("bunny lambda (nonlinear color scale)")
        tick_lambdas = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        tick_positions = [map_lambda_to_color_position(v) for v in tick_lambdas]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([f"{v:.1f}" for v in tick_lambdas])

    ax.set_xlabel("Average Similarity per Dataset (mean over 30 queries)")
    ax.set_ylabel("Average Entropy per Dataset (mean over 30 queries)")
    ax.set_title(title_text)
    ax.grid(alpha=0.2)

    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_GRAPHRAG, markeredgecolor="black", markeredgewidth=0.3, markersize=8, label=METHOD_DISPLAY[METHOD_GRAPHRAG]),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_RANDOM, markeredgecolor="black", markeredgewidth=0.3, markersize=8, label="random"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LAMBDA_COLOR_MAX, markeredgecolor="black", markeredgewidth=0.3, markersize=8, label="bunny (color = lambda)"),
    ]
    ax.legend(handles=legend_handles, loc="best")

    fig.savefig(output_path)
    plt.close(fig)


def process_run(
    run_dir: Path,
    out_dir_name: str,
    formats: Sequence[str],
    lambda_min: float,
    lambda_max: float,
) -> None:
    out_dir = run_dir / out_dir_name
    ensure_dir(out_dir)
    run_name = run_dir.name
    title_text = build_title_from_run_name(run_name)

    rows = aggregate_dataset_means(run_dir / "behavior_results_rows.csv")
    filtered = filter_bunny_lambdas(rows, lambda_min=lambda_min, lambda_max=lambda_max)

    for ext in formats:
        output_path = out_dir / f"single_scatter_lambda_gradient.{ext}"
        plot_single_scatter(
            run_name=run_name,
            title_text=title_text,
            rows=filtered,
            output_path=output_path,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )

    with (out_dir / "plot_config.txt").open("w", encoding="utf-8") as f:
        f.write(f"lambda_min={lambda_min}\n")
        f.write(f"lambda_max={lambda_max}\n")
        f.write(f"causalrag_color={COLOR_GRAPHRAG}\n")
        f.write(f"random_color={COLOR_RANDOM}\n")
        f.write(f"bunny_lambda_color_min={LAMBDA_COLOR_MIN}\n")
        f.write(f"bunny_lambda_color_max={LAMBDA_COLOR_MAX}\n")
        f.write("bunny_lambda_color_mapping=nonlinear\n")
        f.write("nonlinear_lambda_anchors=-0.2:0.00,-0.1:0.30,0.0:0.42,0.1:0.54,0.2:0.82,0.3:1.00\n")
        f.write("excluded_lambdas=>0.3\n")
        f.write(f"title={title_text}\n")
    print(f"[OK] {run_name}: wrote single scatter ({', '.join(formats)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one scatter plot per run with graphrag/random fixed colors and bunny lambda nonlinear color gradient."
    )
    parser.add_argument("--testing-root", default="presentation/testing", help="Root folder with behavior outputs.")
    parser.add_argument("--run", action="append", default=[], help="Optional run folder name. Repeat for multiple.")
    parser.add_argument("--out-dir-name", default="single_scatter_lambda_gradient", help="Output subfolder per run.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    parser.add_argument("--lambda-min", type=float, default=-0.2, help="Minimum bunny lambda to include.")
    parser.add_argument("--lambda-max", type=float, default=0.3, help="Maximum bunny lambda to include.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    testing_root = Path(args.testing_root)
    if not testing_root.exists():
        raise FileNotFoundError(f"Testing root not found: {testing_root}")
    formats = [x.lower() for x in parse_csv_list(args.formats)]
    if not formats:
        formats = ["png"]

    run_dirs = discover_run_dirs(testing_root, args.run)
    if not run_dirs:
        print("No run folders found.")
        return 0

    lambda_min = float(args.lambda_min)
    lambda_max = float(args.lambda_max)
    if lambda_min >= lambda_max:
        raise ValueError("--lambda-min must be less than --lambda-max.")

    for run_dir in run_dirs:
        process_run(
            run_dir=run_dir,
            out_dir_name=args.out_dir_name,
            formats=formats,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
