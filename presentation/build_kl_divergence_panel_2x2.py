from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import build_behavior_single_scatter_lambda_gradient as base


RUN_SPECS: Sequence[Tuple[str, str]] = (
    (
        "behavior_runner_n500_d6_cosine_beta_same_30x30_20260304",
        "Similarity-based | Single-community",
    ),
    (
        "behavior_runner_n500_d6_cosine_beta_mixed_30x30_retry_noempty_20260304",
        "Similarity-based | Mixed-community",
    ),
    (
        "behavior_runner_n500_d6_random_same_30x30_20260304",
        "Random | Single-community",
    ),
    (
        "behavior_runner_n500_d6_random_mixed_30x30_retry_noempty_20260304",
        "Random | Mixed-community",
    ),
)

PANEL_SUBTITLE_FONTSIZE = 19.5
AXIS_LABEL_FONTSIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a KL-divergence 2x2 panel directly from behavior_results_rows.csv with "
            "a single shared colorbar and legend."
        )
    )
    parser.add_argument(
        "--testing-root",
        default="presentation/testing",
        help="Root folder containing behavior_runner_* directories.",
    )
    parser.add_argument(
        "--out",
        default="presentation/testing/overleaf_images/kl_divergence/kl_divergence_panel_2x2.png",
        help="Output panel image path.",
    )
    parser.add_argument("--lambda-min", type=float, default=-0.2, help="Minimum bunny lambda to include.")
    parser.add_argument("--lambda-max", type=float, default=0.3, help="Maximum bunny lambda to include.")
    parser.add_argument("--dpi", type=int, default=250, help="Output DPI.")
    return parser.parse_args()


def load_run_rows(
    run_dir: Path,
    lambda_min: float,
    lambda_max: float,
) -> List[Mapping[str, object]]:
    rows = base.aggregate_dataset_means(run_dir / "behavior_results_rows.csv", y_metric=base.METRIC_KL)
    return base.filter_bunny_lambdas(rows, lambda_min=lambda_min, lambda_max=lambda_max)


def compute_axis_limits(all_rows: Sequence[Sequence[Mapping[str, object]]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    for rows in all_rows:
        for row in rows:
            x_vals.append(float(row["avg_similarity"]))
            y_vals.append(float(row["avg_y"]))
    if not x_vals or not y_vals:
        return (0.0, 1.0), (0.0, 1.0)

    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)
    x_pad = max(0.02, 0.05 * (x_max - x_min if x_max > x_min else 1.0))
    y_pad = max(0.02, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def increase_panel_spacing_by_pixels(
    axes: Sequence[plt.Axes],
    fig: plt.Figure,
    gap_pixels: float,
) -> None:
    if len(axes) != 4:
        return
    tl, tr, bl, br = axes
    fig_w_px = fig.get_figwidth() * fig.dpi
    fig_h_px = fig.get_figheight() * fig.dpi
    if fig_w_px <= 0 or fig_h_px <= 0:
        return
    dx = gap_pixels / fig_w_px
    dy = gap_pixels / fig_h_px

    # Increase horizontal spacing in both rows by dx while preserving outer edges.
    for left, right in ((tl, tr), (bl, br)):
        l = left.get_position()
        r = right.get_position()
        left.set_position([l.x0, l.y0, max(0.01, l.width - dx / 2.0), l.height])
        right.set_position([r.x0 + dx / 2.0, r.y0, max(0.01, r.width - dx / 2.0), r.height])

    # Increase vertical spacing in both columns by dy while preserving top/bottom outer edges.
    for top, bottom in ((tl, bl), (tr, br)):
        t = top.get_position()
        b = bottom.get_position()
        top.set_position([t.x0, t.y0 + dy / 2.0, t.width, max(0.01, t.height - dy / 2.0)])
        bottom.set_position([b.x0, b.y0, b.width, max(0.01, b.height - dy / 2.0)])


def plot_single_axis(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    panel_title: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
) -> None:
    graphrag_pts = [r for r in rows if str(r["method"]) == base.METHOD_GRAPHRAG]
    random_pts = [r for r in rows if str(r["method"]) == base.METHOD_RANDOM]
    bunny_pts = [r for r in rows if str(r["method"]) == base.METHOD_BUNNY]

    if graphrag_pts:
        ax.scatter(
            [float(r["avg_similarity"]) for r in graphrag_pts],
            [float(r["avg_y"]) for r in graphrag_pts],
            s=45,
            color=base.COLOR_GRAPHRAG,
            marker="^",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
        )

    if random_pts:
        ax.scatter(
            [float(r["avg_similarity"]) for r in random_pts],
            [float(r["avg_y"]) for r in random_pts],
            s=45,
            color=base.COLOR_RANDOM,
            marker="s",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
        )

    if bunny_pts:
        bunny_lambdas = np.asarray([float(str(r["lambda"])) for r in bunny_pts], dtype=float)
        bunny_color_pos = np.asarray([base.map_lambda_to_color_position(v) for v in bunny_lambdas], dtype=float)
        bunny_x = np.asarray([float(r["avg_similarity"]) for r in bunny_pts], dtype=float)
        bunny_y = np.asarray([float(r["avg_y"]) for r in bunny_pts], dtype=float)
        ax.scatter(
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
        )

    ax.set_title(panel_title, fontsize=PANEL_SUBTITLE_FONTSIZE, pad=7)
    ax.set_xlabel("Similarity (avg similarity)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Kullback-Leibler divergence", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(alpha=0.2)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.invert_yaxis()


def main() -> int:
    args = parse_args()
    testing_root = Path(args.testing_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_payloads: List[Tuple[str, List[Mapping[str, object]]]] = []
    for run_name, panel_title in RUN_SPECS:
        run_dir = testing_root / run_name
        rows_path = run_dir / "behavior_results_rows.csv"
        if not rows_path.exists():
            raise FileNotFoundError(f"Missing input CSV: {rows_path}")
        rows = load_run_rows(
            run_dir=run_dir,
            lambda_min=float(args.lambda_min),
            lambda_max=float(args.lambda_max),
        )
        run_payloads.append((panel_title, rows))

    x_lim, y_lim = compute_axis_limits([rows for _, rows in run_payloads])
    cmap = LinearSegmentedColormap.from_list(
        "lambda_blue_purple",
        [base.LAMBDA_COLOR_MIN, base.LAMBDA_COLOR_MAX],
    )
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(16, 10), dpi=args.dpi)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.36], wspace=0.22, hspace=0.28)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    right_gs = gs[:, 2].subgridspec(2, 1, height_ratios=[0.72, 0.28], hspace=0.12)
    cax_holder = fig.add_subplot(right_gs[0, 0])
    lax = fig.add_subplot(right_gs[1, 0])
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06)

    for ax, (panel_title, rows) in zip(axes, run_payloads):
        plot_single_axis(
            ax=ax,
            rows=rows,
            panel_title=panel_title,
            cmap=cmap,
            norm=norm,
            x_lim=x_lim,
            y_lim=y_lim,
        )
    subtitle_height_px = PANEL_SUBTITLE_FONTSIZE * (fig.dpi / 72.0)
    increase_panel_spacing_by_pixels(axes=axes, fig=fig, gap_pixels=subtitle_height_px)

    cax_holder.axis("off")
    cax_pos = cax_holder.get_position()
    cbar_width = cax_pos.width * 0.18
    cbar_free = cax_pos.width - cbar_width
    cbar_x0 = cax_pos.x0 + (0.5 - 0.18) * cbar_free
    cbar_ax = fig.add_axes([cbar_x0, cax_pos.y0, cbar_width, cax_pos.height])

    tick_lambdas = sorted(base.NONLINEAR_LAMBDA_ANCHORS.keys())
    color_positions = [base.map_lambda_to_color_position(v) for v in tick_lambdas]
    color_samples = [cmap(norm(p)) for p in color_positions]
    even_positions = np.linspace(0.0, 1.0, len(color_samples))
    colorbar_cmap = LinearSegmentedColormap.from_list(
        "lambda_blue_purple_even_ticks",
        list(zip(even_positions, color_samples)),
    )
    colorbar_norm = Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=colorbar_norm, cmap=colorbar_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(
        "(semantic punishment parameter;\nnegative values = reward)",
        fontsize=21,
    )
    cbar.ax.tick_params(labelsize=18)
    tick_positions = np.linspace(0.0, 1.0, len(tick_lambdas))
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f"{v:.1f}" for v in tick_lambdas])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=base.COLOR_GRAPHRAG,
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=10,
            label=base.METHOD_DISPLAY[base.METHOD_GRAPHRAG],
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=base.COLOR_RANDOM,
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=10,
            label=base.METHOD_DISPLAY[base.METHOD_RANDOM],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=base.LAMBDA_COLOR_MAX,
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=10,
            label=f"{base.METHOD_DISPLAY[base.METHOD_BUNNY]}\n(colored by $\\lambda$)",
        ),
    ]
    lax.axis("off")
    lax.legend(handles=legend_handles, loc="center", fontsize=18, framealpha=0.92)

    fig.suptitle("Similarity vs Kullback-Leibler divergence", fontsize=30, fontweight="bold", y=0.985)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] wrote panel image: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
