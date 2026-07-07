#!/usr/bin/env python3
"""Generate 5-panel main figure for IS2-S2 depth modeling (SW Greenland prototype).

Example:
  python modeling/scripts/train_depth_model.py
  python modeling/scripts/plot_modeling_figure.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODELING_ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modeling.paths import (  # noqa: E402
    CV_PREDICTIONS_CSV,
    FIGURES_DIR,
    METRICS_JSON,
    PAIRS_CSV,
)

FIG_MAIN_PNG = os.path.join(FIGURES_DIR, "fig_modeling_main_SW.png")
FIG_MAIN_PDF = os.path.join(FIGURES_DIR, "fig_modeling_main_SW.pdf")

R2_LABEL = r"$R^2$"
COLOR_RF = "#e76f51"
COLOR_RIDGE = "#457b9d"
COLOR_TRAIN = "#264653"
COLOR_TEST = "#e76f51"


def panel_label(ax, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def lake_color_map(preds: pd.DataFrame) -> dict[str, tuple]:
    lake_ids = sorted(preds["lake_id"].astype(str).unique())
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    palette = []
    for cm in cmaps:
        palette.extend(cm.colors)
    return {lid: palette[i % len(palette)] for i, lid in enumerate(lake_ids)}


def require_cv_train(metrics: dict) -> None:
    for key in ("green_empirical", "ridge", "random_forest"):
        if "cv_train" not in metrics.get(key, {}):
            raise SystemExit(
                f"Missing cv_train for {key} in metrics JSON. "
                "Re-run: python modeling/scripts/train_depth_model.py"
            )


DEPTH_XLABEL = "ICESat-2 depth (m)"


def plot_panel_a(ax, pairs: pd.DataFrame, n_pixels: int, n_lakes: int) -> None:
    ax.hist(
        pairs["depth_m"],
        bins=30,
        color=COLOR_RF,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
        label=f"N = {n_pixels}, {n_lakes} lakes",
    )
    ax.set_xlabel(DEPTH_XLABEL, fontsize=8)
    ax.set_ylabel("Pixel count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    panel_label(ax, "a")


def plot_panel_b(ax, metrics: dict) -> None:
    models = ["Green EFM\n(B3)", "Ridge\n(12 feat.)", "Random\nforest"]
    keys = ["green_empirical", "ridge", "random_forest"]

    r2_train = [metrics[k]["cv_train"]["r2"] for k in keys]
    r2_test = [metrics[k]["cv"]["r2"] for k in keys]
    rmse_test = [metrics[k]["cv"]["rmse"] for k in keys]

    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, r2_train, width=w, color=COLOR_TRAIN, alpha=0.85, label=f"{R2_LABEL} train")
    ax.bar(x + w / 2, r2_test, width=w, color=COLOR_TEST, alpha=0.55, label=f"{R2_LABEL} test")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(R2_LABEL, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7)
    ax.tick_params(labelsize=7)

    for i, (rt, rv) in enumerate(zip(r2_train, r2_test)):
        ax.text(i - w / 2, rt + 0.02, f"{rt:.2f}", ha="center", va="bottom", fontsize=6)
        ax.text(i + w / 2, rv + 0.02, f"{rv:.2f}", ha="center", va="bottom", fontsize=6)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        rmse_test,
        color=COLOR_TEST,
        marker="s",
        lw=1.8,
        ms=6,
        ls="--",
        label="RMSE test",
    )
    ax2.set_ylabel("RMSE (m)", fontsize=8)
    ax2.tick_params(labelsize=7)
    rmse_max = max(rmse_test) * 1.15
    ax2.set_ylim(0, max(rmse_max, 1.0))

    for i, rv in enumerate(rmse_test):
        ax2.text(i, rv + 0.03, f"{rv:.2f}", ha="center", va="bottom", fontsize=6, color=COLOR_TEST)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=6, loc="lower right", framealpha=0.9)
    panel_label(ax, "b")


def plot_panel_c(ax, metrics: dict) -> None:
    imp = metrics["random_forest"]["feature_importances"]
    names = list(imp.keys())
    values = [imp[n] for n in names]
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=COLOR_RF, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_refl", "").replace("S2_", "") for n in names], fontsize=7)
    ax.set_xlabel("Importance", fontsize=8)
    ax.tick_params(labelsize=7)
    panel_label(ax, "c")


def plot_panel_d(ax, preds: pd.DataFrame, metrics: dict, colors: dict[str, tuple]) -> None:
    for lid in sorted(colors):
        sub = preds[preds["lake_id"].astype(str) == lid]
        ax.scatter(
            sub["depth_m"],
            sub["pred_random_forest"],
            s=10,
            alpha=0.55,
            c=[colors[lid]],
            edgecolors="none",
            rasterized=True,
        )

    y = preds["depth_m"].to_numpy()
    p = preds["pred_random_forest"].to_numpy()
    lim = max(float(np.max(y)), float(np.max(p)), 1.0)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    rmse = metrics["random_forest"]["cv"]["rmse"]
    ax.fill_between([0, lim], [0, lim - rmse], [0, lim + rmse], color="gray", alpha=0.12, label=f"±RMSE ({rmse:.2f} m)")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(DEPTH_XLABEL, fontsize=8)
    ax.set_ylabel("RF predicted depth (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    r2 = metrics["random_forest"]["cv"]["r2"]
    r2_label = f"{R2_LABEL} = {r2:.2f}"
    r2_handle = Line2D([0], [0], color="none", label=r2_label)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [r2_handle],
        labels + [r2_label],
        fontsize=6,
        loc="lower right",
        framealpha=0.9,
    )
    panel_label(ax, "d")


def plot_panel_e_residual(ax, preds: pd.DataFrame, colors: dict[str, tuple]) -> None:
    for lid in sorted(colors):
        sub = preds[preds["lake_id"].astype(str) == lid]
        residual = sub["pred_random_forest"] - sub["depth_m"]
        ax.scatter(
            sub["depth_m"],
            residual,
            s=8,
            alpha=0.45,
            c=[colors[lid]],
            edgecolors="none",
            rasterized=True,
        )
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel(DEPTH_XLABEL, fontsize=8)
    ax.set_ylabel("Residual", fontsize=8)
    ax.tick_params(labelsize=7)
    panel_label(ax, "e")


def build_figure(
    pairs_path: str,
    metrics_path: str,
    preds_path: str,
    out_png: str,
    out_pdf: str,
) -> None:
    pairs = pd.read_csv(pairs_path)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    preds = pd.read_csv(preds_path)
    require_cv_train(metrics)

    n_pixels = int(metrics.get("n_pixels", len(preds)))
    n_lakes = int(metrics.get("n_lakes", preds["lake_id"].nunique()))
    colors = lake_color_map(preds)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.6,
            "figure.dpi": 100,
            "mathtext.default": "regular",
        }
    )

    fig = plt.figure(figsize=(7.08, 7.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.85], hspace=0.42, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    plot_panel_a(ax_a, pairs, n_pixels, n_lakes)
    plot_panel_b(ax_b, metrics)
    plot_panel_c(ax_c, metrics)
    plot_panel_d(ax_d, preds, metrics, colors)
    plot_panel_e_residual(ax_e, preds, colors)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default=PAIRS_CSV)
    parser.add_argument("--metrics-json", default=METRICS_JSON)
    parser.add_argument("--predictions-csv", default=CV_PREDICTIONS_CSV)
    parser.add_argument("--out-png", default=FIG_MAIN_PNG)
    parser.add_argument("--out-pdf", default=FIG_MAIN_PDF)
    args = parser.parse_args()

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    pairs_path = resolve(args.pairs_csv)
    metrics_path = resolve(args.metrics_json)
    preds_path = resolve(args.predictions_csv)
    out_png = resolve(args.out_png)
    out_pdf = resolve(args.out_pdf)

    for path, name in [
        (pairs_path, "pairs CSV"),
        (metrics_path, "metrics JSON"),
        (preds_path, "predictions CSV"),
    ]:
        if not os.path.isfile(path):
            raise SystemExit(f"Missing {name}: {path}")

    build_figure(pairs_path, metrics_path, preds_path, out_png, out_pdf)
    print("Main figure saved:")
    print(" ", out_png)
    print(" ", out_pdf)


if __name__ == "__main__":
    main()
