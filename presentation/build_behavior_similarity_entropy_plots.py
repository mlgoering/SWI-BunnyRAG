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
from matplotlib.patches import Patch

METHOD_ORDER = ("graphrag", "random", "bunny")
METHOD_COLORS = {
    "graphrag": "#d62728",
    "random": "#2ca02c",
    "bunny": "#1f77b4",
}
METHOD_CMAPS = {
    "graphrag": "Reds",
    "random": "Greens",
    "bunny": "Blues",
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


def slugify_lambda(lambda_key: str) -> str:
    return lambda_key.replace("-", "m").replace(".", "p").replace("+", "")


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


def parse_metadata_lambdas(run_dir: Path) -> List[str]:
    meta_path = run_dir / "behavior_metadata.json"
    if not meta_path.exists():
        return []
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_lambdas = payload.get("lambdas", [])
    values: List[str] = []
    if isinstance(raw_lambdas, list):
        for item in raw_lambdas:
            try:
                values.append(format_lambda(float(item)))
            except Exception:
                continue
    return values


def aggregate_dataset_means(
    rows_path: Path,
) -> List[Dict[str, object]]:
    sums: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    with rows_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method", "")).strip().lower()
            dataset_id = str(row.get("dataset_id", "")).strip()
            dataset_seed = str(row.get("dataset_seed", "")).strip()
            if not dataset_id or method not in METHOD_ORDER:
                continue

            entropy = _to_float(row.get("entropy"))
            avg_query_similarity = _to_float(row.get("avg_query_similarity"))
            if entropy is None or avg_query_similarity is None:
                continue

            lambda_key = "NA"
            if method == "bunny":
                lambda_value = _to_float(row.get("lambda"))
                if lambda_value is None:
                    continue
                lambda_key = format_lambda(lambda_value)

            agg_key = (dataset_id, dataset_seed, method, lambda_key)
            slot = sums.setdefault(
                agg_key,
                {"entropy_sum": 0.0, "similarity_sum": 0.0, "query_count": 0.0},
            )
            slot["entropy_sum"] += entropy
            slot["similarity_sum"] += avg_query_similarity
            slot["query_count"] += 1.0

    out: List[Dict[str, object]] = []
    for (dataset_id, dataset_seed, method, lambda_key), slot in sorted(sums.items()):
        count = int(slot["query_count"])
        if count <= 0:
            continue
        out.append(
            {
                "dataset_id": dataset_id,
                "dataset_seed": dataset_seed,
                "method": method,
                "lambda": lambda_key,
                "query_count": count,
                "avg_similarity": slot["similarity_sum"] / count,
                "avg_entropy": slot["entropy_sum"] / count,
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_method_maps(
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


def build_mode_overall_rows(dataset_means: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in dataset_means:
        grouped[(str(row["method"]), str(row["lambda"]))].append(row)

    out: List[Dict[str, object]] = []
    for (method, lambda_key), rows in sorted(grouped.items()):
        sims = np.asarray([float(r["avg_similarity"]) for r in rows], dtype=float)
        ents = np.asarray([float(r["avg_entropy"]) for r in rows], dtype=float)
        out.append(
            {
                "method": method,
                "lambda": lambda_key,
                "datasets": int(len(rows)),
                "mean_similarity": float(np.mean(sims)),
                "std_similarity": float(np.std(sims)),
                "mean_entropy": float(np.mean(ents)),
                "std_entropy": float(np.std(ents)),
            }
        )
    return out


def _plot_scatter(
    run_name: str,
    lambda_key: str,
    points_by_method: Mapping[str, Sequence[Mapping[str, float]]],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for method in METHOD_ORDER:
        rows = points_by_method.get(method, [])
        if not rows:
            continue
        xs = [float(r["avg_similarity"]) for r in rows]
        ys = [float(r["avg_entropy"]) for r in rows]
        ax.scatter(
            xs,
            ys,
            s=34,
            alpha=0.82,
            color=METHOD_COLORS[method],
            edgecolors="black",
            linewidths=0.35,
            label=method,
        )

    dataset_count = len(points_by_method.get("bunny", []))
    ax.set_xlabel("Average Similarity per Dataset (over queries)")
    ax.set_ylabel("Average Entropy per Dataset (over queries)")
    ax.set_title(f"{run_name}\nlambda={lambda_key} | datasets={dataset_count}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_hist2d_overlay(
    run_name: str,
    lambda_key: str,
    points_by_method: Mapping[str, Sequence[Mapping[str, float]]],
    bins: int,
    output_path: Path,
    dpi: int,
) -> None:
    x_all: List[float] = []
    y_all: List[float] = []
    for method in METHOD_ORDER:
        rows = points_by_method.get(method, [])
        x_all.extend(float(r["avg_similarity"]) for r in rows)
        y_all.extend(float(r["avg_entropy"]) for r in rows)
    if not x_all or not y_all:
        return

    x_min, x_max = min(x_all), max(x_all)
    y_min, y_max = min(y_all), max(y_all)
    if math.isclose(x_min, x_max):
        x_min -= 0.05
        x_max += 0.05
    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05

    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for method in METHOD_ORDER:
        rows = points_by_method.get(method, [])
        if not rows:
            continue
        xs = [float(r["avg_similarity"]) for r in rows]
        ys = [float(r["avg_entropy"]) for r in rows]
        ax.hist2d(
            xs,
            ys,
            bins=[x_edges, y_edges],
            cmap=METHOD_CMAPS[method],
            alpha=0.42,
            cmin=1,
        )
        ax.scatter(xs, ys, s=14, color=METHOD_COLORS[method], alpha=0.85)

    patches = [Patch(facecolor=METHOD_COLORS[m], alpha=0.55, label=m) for m in METHOD_ORDER]
    dataset_count = len(points_by_method.get("bunny", []))
    ax.legend(handles=patches, loc="best")
    ax.set_xlabel("Average Similarity per Dataset (over queries)")
    ax.set_ylabel("Average Entropy per Dataset (over queries)")
    ax.set_title(f"{run_name}\n2D histogram overlay | lambda={lambda_key} | datasets={dataset_count}")
    ax.grid(alpha=0.15)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_per_lambda_outputs(
    run_name: str,
    output_dir: Path,
    baseline_map: Mapping[str, Mapping[str, Mapping[str, float]]],
    bunny_map: Mapping[str, Mapping[str, Mapping[str, float]]],
    metadata_lambdas: Sequence[str],
    bins: int,
    dpi: int,
) -> List[Dict[str, object]]:
    plots_dir = output_dir / "plots"
    payloads_dir = output_dir / "plot_payloads"
    ensure_dir(plots_dir)
    ensure_dir(payloads_dir)

    lambda_keys = sorted(bunny_map.keys(), key=lambda x: float(x))
    missing_from_data = sorted(set(metadata_lambdas) - set(lambda_keys), key=lambda x: float(x))
    if missing_from_data:
        print(f"[WARN] {run_name}: metadata lambdas missing from data: {', '.join(missing_from_data)}")

    index_rows: List[Dict[str, object]] = []
    for lambda_key in lambda_keys:
        bunny_rows = bunny_map.get(lambda_key, {})
        graphrag_rows = baseline_map.get("graphrag", {})
        random_rows = baseline_map.get("random", {})
        common_ids = sorted(set(bunny_rows) & set(graphrag_rows) & set(random_rows))
        if not common_ids:
            continue

        points_by_method: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for dataset_id in common_ids:
            for method, source in (
                ("graphrag", graphrag_rows),
                ("random", random_rows),
                ("bunny", bunny_rows),
            ):
                row = source[dataset_id]
                points_by_method[method].append(
                    {
                        "dataset_id": dataset_id,
                        "avg_similarity": float(row["avg_similarity"]),
                        "avg_entropy": float(row["avg_entropy"]),
                    }
                )

        payload = {
            "run_name": run_name,
            "lambda": lambda_key,
            "dataset_count": len(common_ids),
            "methods": {k: points_by_method[k] for k in METHOD_ORDER},
        }
        lambda_slug = slugify_lambda(lambda_key)
        payload_json = payloads_dir / f"lambda_{lambda_slug}_payload.json"
        payload_csv = payloads_dir / f"lambda_{lambda_slug}_payload.csv"
        payload_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        csv_rows: List[Dict[str, object]] = []
        for method in METHOD_ORDER:
            for row in points_by_method[method]:
                csv_rows.append(
                    {
                        "run_name": run_name,
                        "lambda": lambda_key,
                        "dataset_id": row["dataset_id"],
                        "method": method,
                        "avg_similarity": row["avg_similarity"],
                        "avg_entropy": row["avg_entropy"],
                    }
                )
        write_csv(
            payload_csv,
            csv_rows,
            fieldnames=("run_name", "lambda", "dataset_id", "method", "avg_similarity", "avg_entropy"),
        )

        scatter_path = plots_dir / f"lambda_{lambda_slug}_scatter.png"
        hist2d_path = plots_dir / f"lambda_{lambda_slug}_hist2d.png"
        _plot_scatter(
            run_name=run_name,
            lambda_key=lambda_key,
            points_by_method=points_by_method,
            output_path=scatter_path,
            dpi=dpi,
        )
        _plot_hist2d_overlay(
            run_name=run_name,
            lambda_key=lambda_key,
            points_by_method=points_by_method,
            bins=bins,
            output_path=hist2d_path,
            dpi=dpi,
        )

        index_rows.append(
            {
                "lambda": lambda_key,
                "dataset_count_common": len(common_ids),
                "scatter_plot": str(scatter_path),
                "hist2d_plot": str(hist2d_path),
                "payload_json": str(payload_json),
                "payload_csv": str(payload_csv),
            }
        )
    return index_rows


def process_run_dir(run_dir: Path, out_dir_name: str, bins: int, dpi: int) -> None:
    run_name = run_dir.name
    rows_path = run_dir / "behavior_results_rows.csv"
    output_dir = run_dir / out_dir_name
    ensure_dir(output_dir)

    dataset_means = aggregate_dataset_means(rows_path=rows_path)
    dataset_means_csv = output_dir / "dataset_method_lambda_means.csv"
    dataset_means_json = output_dir / "dataset_method_lambda_means.json"
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
    dataset_means_json.write_text(json.dumps(dataset_means, indent=2), encoding="utf-8")

    mode_overall = build_mode_overall_rows(dataset_means)
    mode_overall_csv = output_dir / "mode_overall_means.csv"
    write_csv(
        mode_overall_csv,
        mode_overall,
        fieldnames=(
            "method",
            "lambda",
            "datasets",
            "mean_similarity",
            "std_similarity",
            "mean_entropy",
            "std_entropy",
        ),
    )

    baseline_map, bunny_map = build_method_maps(dataset_means)
    metadata_lambdas = parse_metadata_lambdas(run_dir)
    lambda_index = build_per_lambda_outputs(
        run_name=run_name,
        output_dir=output_dir,
        baseline_map=baseline_map,
        bunny_map=bunny_map,
        metadata_lambdas=metadata_lambdas,
        bins=bins,
        dpi=dpi,
    )
    index_payload = {
        "run_name": run_name,
        "source_csv": str(rows_path),
        "dataset_means_csv": str(dataset_means_csv),
        "dataset_means_json": str(dataset_means_json),
        "mode_overall_csv": str(mode_overall_csv),
        "per_lambda": lambda_index,
    }
    (output_dir / "index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(f"[OK] {run_name}: {len(dataset_means)} dataset-level rows, {len(lambda_index)} lambda plot bundles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate behavior runner outputs into dataset-level means and per-lambda comparison plots."
    )
    parser.add_argument(
        "--testing-root",
        default="presentation/testing",
        help="Root directory containing behavior_runner_* folders.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional run folder name to process. Repeat to include multiple runs.",
    )
    parser.add_argument(
        "--out-dir-name",
        default="similarity_entropy_analysis",
        help="Output folder name created under each run folder.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="Number of bins per axis for 2D histogram overlays.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for output PNG files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    testing_root = Path(args.testing_root)
    if not testing_root.exists():
        raise FileNotFoundError(f"Testing root not found: {testing_root}")

    run_dirs = discover_run_dirs(testing_root=testing_root, only_runs=args.run)
    if not run_dirs:
        print("No run folders found.")
        return 0

    for run_dir in run_dirs:
        process_run_dir(
            run_dir=run_dir,
            out_dir_name=args.out_dir_name,
            bins=max(2, int(args.bins)),
            dpi=max(72, int(args.dpi)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
