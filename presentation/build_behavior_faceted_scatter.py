from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

METHOD_ORDER = ("graphrag", "random", "bunny")
METHOD_COLORS = {
    "graphrag": "#d62728",
    "random": "#2ca02c",
    "bunny": "#1f77b4",
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


def format_lambda(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return f"{value:.6g}"


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_run_dirs(testing_root: Path, only_runs: Sequence[str]) -> List[Path]:
    run_dirs: List[Path] = []
    allowed = set(only_runs)
    for child in sorted(testing_root.iterdir()):
        if not child.is_dir():
            continue
        if only_runs and child.name not in allowed:
            continue
        if (child / "behavior_results_rows.csv").exists():
            run_dirs.append(child)
    return run_dirs


def aggregate_dataset_means(rows_path: Path) -> List[Dict[str, object]]:
    sums: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    with rows_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_id = str(row.get("dataset_id", "")).strip()
            dataset_seed = str(row.get("dataset_seed", "")).strip()
            method = str(row.get("method", "")).strip().lower()
            if not dataset_id or method not in METHOD_ORDER:
                continue

            sim = _to_float(row.get("avg_query_similarity"))
            ent = _to_float(row.get("entropy"))
            if sim is None or ent is None:
                continue

            lambda_key = "NA"
            if method == "bunny":
                lam = _to_float(row.get("lambda"))
                if lam is None:
                    continue
                lambda_key = format_lambda(lam)

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


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_maps(
    dataset_means: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    baseline: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    bunny: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for row in dataset_means:
        dataset_id = str(row["dataset_id"])
        method = str(row["method"])
        lambda_key = str(row["lambda"])
        entry = {
            "avg_similarity": float(row["avg_similarity"]),
            "avg_entropy": float(row["avg_entropy"]),
            "query_count": float(row["query_count"]),
        }
        if method == "bunny":
            bunny[lambda_key][dataset_id] = entry
        else:
            baseline[method][dataset_id] = entry
    return baseline, bunny


def build_lambda_points(
    baseline_map: Mapping[str, Mapping[str, Mapping[str, float]]],
    bunny_map: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Tuple[List[str], Dict[str, Dict[str, List[Dict[str, float]]]]]:
    lambdas = sorted(bunny_map.keys(), key=lambda x: float(x))
    per_lambda: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    graphrag_rows = baseline_map.get("graphrag", {})
    random_rows = baseline_map.get("random", {})

    for lambda_key in lambdas:
        bunny_rows = bunny_map.get(lambda_key, {})
        common_ids = sorted(set(bunny_rows) & set(graphrag_rows) & set(random_rows))
        method_points: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for dataset_id in common_ids:
            for method, src in (("graphrag", graphrag_rows), ("random", random_rows), ("bunny", bunny_rows)):
                row = src[dataset_id]
                method_points[method].append(
                    {
                        "dataset_id": dataset_id,
                        "avg_similarity": float(row["avg_similarity"]),
                        "avg_entropy": float(row["avg_entropy"]),
                    }
                )
        per_lambda[lambda_key] = method_points
    return lambdas, per_lambda


def compute_axis_limits(per_lambda: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for method_map in per_lambda.values():
        for rows in method_map.values():
            xs.extend(float(r["avg_similarity"]) for r in rows)
            ys.extend(float(r["avg_entropy"]) for r in rows)

    if not xs or not ys:
        return (-1.0, 1.0), (0.0, 2.5)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(0.02, 0.05 * (x_max - x_min if x_max > x_min else 1.0))
    y_pad = max(0.02, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_faceted_scatter(
    run_name: str,
    lambdas: Sequence[str],
    per_lambda: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]],
    output_path: Path,
    ncols: int,
) -> None:
    if not lambdas:
        return
    ncols = max(1, ncols)
    nrows = math.ceil(len(lambdas) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)
    x_lim, y_lim = compute_axis_limits(per_lambda)

    for idx, lambda_key in enumerate(lambdas):
        ax = axes_flat[idx]
        method_map = per_lambda[lambda_key]
        for method in METHOD_ORDER:
            rows = method_map.get(method, [])
            if not rows:
                continue
            x_vals = [float(r["avg_similarity"]) for r in rows]
            y_vals = [float(r["avg_entropy"]) for r in rows]
            ax.scatter(
                x_vals,
                y_vals,
                s=28,
                alpha=0.80,
                color=METHOD_COLORS[method],
                edgecolors="black",
                linewidths=0.25,
            )
        ds_count = len(method_map.get("bunny", []))
        ax.set_title(f"lambda={lambda_key} | n={ds_count}", fontsize=10)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(alpha=0.20)

    for j in range(len(lambdas), len(axes_flat)):
        axes_flat[j].axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=7,
            label=m,
        )
        for m in METHOD_ORDER
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{run_name} | Faceted Scatter by Lambda", fontsize=13)
    fig.supxlabel("Average Similarity per Dataset (mean over 30 queries)")
    fig.supylabel("Average Entropy per Dataset (mean over 30 queries)")
    fig.savefig(output_path)
    plt.close(fig)


def process_run(run_dir: Path, out_dir_name: str, ncols: int, formats: Sequence[str]) -> None:
    run_name = run_dir.name
    out_dir = run_dir / out_dir_name
    ensure_dir(out_dir)

    dataset_means = aggregate_dataset_means(run_dir / "behavior_results_rows.csv")
    dataset_means_csv = out_dir / "dataset_method_lambda_means.csv"
    write_csv(
        dataset_means_csv,
        dataset_means,
        fieldnames=(
            "dataset_id",
            "dataset_seed",
            "method",
            "lambda",
            "query_count",
            "avg_similarity",
            "avg_entropy",
        ),
    )

    baseline_map, bunny_map = build_maps(dataset_means)
    lambdas, per_lambda = build_lambda_points(baseline_map, bunny_map)

    for ext in formats:
        output_path = out_dir / f"faceted_scatter_by_lambda.{ext}"
        plot_faceted_scatter(
            run_name=run_name,
            lambdas=lambdas,
            per_lambda=per_lambda,
            output_path=output_path,
            ncols=ncols,
        )

    panel_index: List[Dict[str, object]] = []
    for lam in lambdas:
        panel_index.append({"lambda": lam, "dataset_count_common": len(per_lambda.get(lam, {}).get("bunny", []))})
    (out_dir / "panel_index.json").write_text(json.dumps(panel_index, indent=2), encoding="utf-8")
    print(f"[OK] {run_name}: wrote faceted scatter ({', '.join(formats)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build faceted scatter plots by lambda for each behavior run folder.")
    parser.add_argument("--testing-root", default="presentation/testing", help="Root folder with behavior runner outputs.")
    parser.add_argument("--run", action="append", default=[], help="Optional run folder name. Repeat for multiple.")
    parser.add_argument("--out-dir-name", default="faceted_scatter", help="Output subfolder created under each run.")
    parser.add_argument("--ncols", type=int, default=4, help="Number of columns in facet grid.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats, e.g. png,pdf.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    testing_root = Path(args.testing_root)
    if not testing_root.exists():
        raise FileNotFoundError(f"Testing root not found: {testing_root}")

    formats = [fmt.lower() for fmt in parse_csv_list(args.formats)]
    if not formats:
        formats = ["png"]

    run_dirs = discover_run_dirs(testing_root, args.run)
    if not run_dirs:
        print("No run folders found.")
        return 0

    for run_dir in run_dirs:
        process_run(run_dir=run_dir, out_dir_name=args.out_dir_name, ncols=max(1, int(args.ncols)), formats=formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
