from __future__ import annotations

import argparse
import csv
import re
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
METRIC_ENTROPY = "entropy"
METRIC_KL = "kl_selected_vs_graph"

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
    METHOD_GRAPHRAG: "GraphRAG",
    METHOD_RANDOM: "Random",
    METHOD_BUNNY: "BunnyRAG",
}

METRIC_CONFIG = {
    METRIC_ENTROPY: {
        "title": "Relevance vs cross-community coverage",
        "ylabel": "Cross-community coverage (higher entropy is better)",
        "invert_y": False,
        "xlabel_fontsize": 14.0,
        "ylabel_fontsize": 13.0,
        "title_fontsize": 21.0,
        "subtitle_fontsize": 12.0,
        "colorbar_label_fontsize": 11.0,
        "colorbar_tick_decimals": 2,
        "suptitle_y": 0.99,
        "subtitle_y": 0.90,
        "tight_top": 0.84,
        "subtitle_compact": False,
    },
    METRIC_KL: {
        "title": "Similarity vs KL-divergence",
        "ylabel": "KL-divergence (axis inverted)",
        "invert_y": True,
        "xlabel_fontsize": 21.0,
        "ylabel_fontsize": 19.5,
        "title_fontsize": 31.5,
        "subtitle_fontsize": 18.0,
        "colorbar_label_fontsize": 16.5,
        "colorbar_tick_decimals": 1,
        "suptitle_y": 0.975,
        "subtitle_y": 0.885,
        "tight_top": 0.87,
        "subtitle_compact": True,
    },
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


def build_subtitle_from_run_name(run_name: str, compact: bool = False) -> str:
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
    weighting_label_full = {
        "cosine_beta": "similarity-based weights",
        "random": "random weights",
    }.get(weighting, f"{weighting} weighting")
    weighting_label_compact = {
        "cosine_beta": "Similarity-based weights",
        "random": "Random weights",
    }.get(weighting, f"{weighting} weighting")
    seed_label = {
        "same": "single-community seed regime",
        "mixed": "mixed-community seed regime",
    }.get(seeds, f"{seeds} seed regime")
    if compact:
        return f"{weighting_label_compact} and {seed_label}"
    return f"{n} nodes, {dim}D; {weighting_label_full}; {seed_label}"


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


def aggregate_dataset_means(rows_path: Path, y_metric: str) -> List[Dict[str, object]]:
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
            if y_metric == METRIC_ENTROPY:
                y_value = _to_float(row.get("entropy"))
            else:
                y_value = _to_float(row.get("kl_selected_vs_graph"))
            if sim is None or y_value is None:
                continue

            lambda_key = "NA"
            if method == METHOD_BUNNY:
                lam = _to_float(row.get("lambda"))
                if lam is None:
                    continue
                lambda_key = f"{lam:.6g}"

            key = (dataset_id, dataset_seed, method, lambda_key)
            slot = sums.setdefault(key, {"sim_sum": 0.0, "y_sum": 0.0, "count": 0.0})
            slot["sim_sum"] += sim
            slot["y_sum"] += y_value
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
                "avg_y": slot["y_sum"] / count,
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
    subtitle_text: str,
    rows: Sequence[Mapping[str, object]],
    title_text: str,
    y_label: str,
    invert_y: bool,
    metric_cfg: Mapping[str, object],
    output_path: Path,
) -> None:
    graphrag_pts = [r for r in rows if str(r["method"]) == METHOD_GRAPHRAG]
    random_pts = [r for r in rows if str(r["method"]) == METHOD_RANDOM]
    bunny_pts = [r for r in rows if str(r["method"]) == METHOD_BUNNY]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    if graphrag_pts:
        ax.scatter(
            [float(r["avg_similarity"]) for r in graphrag_pts],
            [float(r["avg_y"]) for r in graphrag_pts],
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
            [float(r["avg_y"]) for r in random_pts],
            s=45,
            color=COLOR_RANDOM,
            marker="s",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            label=METHOD_DISPLAY[METHOD_RANDOM],
        )

    cmap = LinearSegmentedColormap.from_list("lambda_blue_purple", [LAMBDA_COLOR_MIN, LAMBDA_COLOR_MAX])
    norm = Normalize(vmin=0.0, vmax=1.0)
    if bunny_pts:
        bunny_lambdas = np.asarray([float(str(r["lambda"])) for r in bunny_pts], dtype=float)
        bunny_color_pos = np.asarray([map_lambda_to_color_position(v) for v in bunny_lambdas], dtype=float)
        bunny_x = np.asarray([float(r["avg_similarity"]) for r in bunny_pts], dtype=float)
        bunny_y = np.asarray([float(r["avg_y"]) for r in bunny_pts], dtype=float)
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
            label=f"{METHOD_DISPLAY[METHOD_BUNNY]} (colored by \u03bb)",
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(
            "(semantic punishment parameter;\nnegative values = reward)",
            fontsize=float(metric_cfg["colorbar_label_fontsize"]),
        )
        cbar.ax.tick_params(labelsize=10)
        tick_lambdas = sorted({float(str(r["lambda"])) for r in bunny_pts})
        tick_positions = [map_lambda_to_color_position(v) for v in tick_lambdas]
        cbar.set_ticks(tick_positions)
        cbar_tick_decimals = int(metric_cfg["colorbar_tick_decimals"])
        cbar.set_ticklabels([f"{v:.{cbar_tick_decimals}f}" for v in tick_lambdas])

    ax.set_xlabel("Semantic relevance (avg similarity)", fontsize=float(metric_cfg["xlabel_fontsize"]))
    ax.set_ylabel(y_label, fontsize=float(metric_cfg["ylabel_fontsize"]))
    if invert_y:
        ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=12)

    fig.suptitle(
        title_text,
        fontsize=float(metric_cfg["title_fontsize"]),
        fontweight="bold",
        y=float(metric_cfg["suptitle_y"]),
    )
    fig.text(
        0.5,
        float(metric_cfg["subtitle_y"]),
        subtitle_text,
        ha="center",
        va="center",
        fontsize=float(metric_cfg["subtitle_fontsize"]),
        color="#333333",
    )

    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_GRAPHRAG, markeredgecolor="black", markeredgewidth=0.3, markersize=9, label=METHOD_DISPLAY[METHOD_GRAPHRAG]),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_RANDOM, markeredgecolor="black", markeredgewidth=0.3, markersize=9, label=METHOD_DISPLAY[METHOD_RANDOM]),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LAMBDA_COLOR_MAX, markeredgecolor="black", markeredgewidth=0.3, markersize=9, label=f"{METHOD_DISPLAY[METHOD_BUNNY]} (colored by \u03bb)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.01, 0.01), fontsize=12, framealpha=0.92)

    fig.tight_layout(rect=(0, 0, 1, float(metric_cfg["tight_top"])))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def process_run(
    run_dir: Path,
    out_dir_name: str,
    output_stem: str,
    formats: Sequence[str],
    lambda_min: float,
    lambda_max: float,
    y_metric: str,
) -> None:
    out_dir = run_dir / out_dir_name
    ensure_dir(out_dir)
    run_name = run_dir.name
    metric_cfg = METRIC_CONFIG[y_metric]
    subtitle_text = build_subtitle_from_run_name(run_name, compact=bool(metric_cfg["subtitle_compact"]))

    rows = aggregate_dataset_means(run_dir / "behavior_results_rows.csv", y_metric=y_metric)
    filtered = filter_bunny_lambdas(rows, lambda_min=lambda_min, lambda_max=lambda_max)

    for ext in formats:
        output_path = out_dir / f"{output_stem}.{ext}"
        plot_single_scatter(
            subtitle_text=subtitle_text,
            rows=filtered,
            title_text=str(metric_cfg["title"]),
            y_label=str(metric_cfg["ylabel"]),
            invert_y=bool(metric_cfg["invert_y"]),
            metric_cfg=metric_cfg,
            output_path=output_path,
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
        f.write(f"y_metric={y_metric}\n")
        f.write(f"title={metric_cfg['title']}\n")
        f.write(f"subtitle={subtitle_text}\n")
    print(f"[OK] {run_name}: wrote single scatter ({y_metric}; {', '.join(formats)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one scatter plot per run with graphrag/random fixed colors and bunny lambda nonlinear color gradient."
    )
    parser.add_argument("--testing-root", default="presentation/testing", help="Root folder with behavior outputs.")
    parser.add_argument("--run", action="append", default=[], help="Optional run folder name. Repeat for multiple.")
    parser.add_argument("--out-dir-name", default="single_scatter_lambda_gradient", help="Output subfolder per run.")
    parser.add_argument("--output-stem", default="single_scatter_lambda_gradient", help="Output file stem (without extension).")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    parser.add_argument(
        "--y-metric",
        default=METRIC_ENTROPY,
        choices=[METRIC_ENTROPY, METRIC_KL],
        help="Y-axis metric for the scatter plot.",
    )
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
            output_stem=args.output_stem,
            formats=formats,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            y_metric=args.y_metric,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

