#!/usr/bin/env python3
"""Summarize modeling pipeline counts and depth-binned CV RMSE tables.

Outputs:
  modeling/out/models/depth_pipeline_summary.csv
  modeling/out/models/depth_rmse_by_depth_bin.csv
  modeling/out/models/depth_rmse_by_depth_bin.md
  modeling/out/models/depth_pipeline_summary.md

Example:
  python modeling/scripts/train_depth_model.py
  python modeling/scripts/summarize_modeling_stats.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODELING_ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modeling.paths import (  # noqa: E402
    CANDIDATES_CSV,
    CV_PREDICTIONS_CSV,
    LAKE_DATA_DIR,
    METRICS_JSON,
    MODELS_DIR,
    PAIRS_CSV,
    PHOTON_PAIRS_CSV,
)
from modeling.scripts.extract_is2_s2_pairs import (  # noqa: E402
    extract_depth_points,
    list_lake_files,
    load_candidate_table,
)
from modeling.scripts.train_depth_model import (  # noqa: E402
    add_spectral_indices,
    filter_training_rows,
)

MODEL_COLUMNS = {
    "green_empirical": ("Green EFM (B3)", "pred_green_empirical"),
    "ridge": ("Ridge (12 feat.)", "pred_ridge"),
    "random_forest": ("Random forest", "pred_random_forest"),
}

DEPTH_BIN_WIDTH = 1.0


def depth_bins_from_data(y: np.ndarray, bin_width: float = DEPTH_BIN_WIDTH) -> list[tuple[float, float]]:
    """Build 1 m bins from 0 up to the ceiling of max observed depth."""
    if len(y) == 0:
        return []
    upper = int(np.ceil(float(np.max(y)) / bin_width) * bin_width)
    upper = max(upper, bin_width)
    bins: list[tuple[float, float]] = []
    lo = 0.0
    while lo < upper:
        bins.append((lo, lo + bin_width))
        lo += bin_width
    return bins


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def beam_from_lake_id(lake_id: str) -> str | None:
    m = re.search(r"_(gt\w+r)_", str(lake_id))
    return m.group(1) if m else None


def count_extract_photons(
    data_dir: str,
    granules: set[str],
    min_depth: float,
    min_conf: float,
    max_points_per_lake: int,
) -> tuple[int, int, int, dict[str, int], dict[str, int]]:
    lake_files = list_lake_files(data_dir, granules)
    max_pts = None if max_points_per_lake == 0 else max_points_per_lake
    all_points: list[pd.DataFrame] = []
    for lake_path in lake_files:
        pts = extract_depth_points(lake_path, min_depth, min_conf, max_pts)
        if not pts.empty:
            all_points.append(pts)

    if not all_points:
        return len(lake_files), 0, 0, {}, {}

    points_df = pd.concat(all_points, ignore_index=True)
    points_df = points_df.copy()
    points_df["beam"] = points_df["lake_id"].map(beam_from_lake_id)
    beam_lake_counts = (
        points_df.dropna(subset=["beam"]).groupby("beam")["lake_id"].nunique().sort_index()
    )
    beam_photon_counts = points_df.dropna(subset=["beam"]).groupby("beam").size().sort_index()
    n_lakes_with_photons = int(points_df["lake_id"].nunique())
    return (
        len(lake_files),
        len(points_df),
        n_lakes_with_photons,
        {str(k): int(v) for k, v in beam_lake_counts.items()},
        {str(k): int(v) for k, v in beam_photon_counts.items()},
    )


def build_pipeline_summary(
    candidates_path: str,
    data_dir: str,
    photon_csv: str,
    pixel_csv: str,
    predictions_csv: str,
    min_depth: float,
    min_conf: float,
    max_points_per_lake: int,
    min_photons: int,
) -> pd.DataFrame:
    candidates = load_candidate_table(candidates_path, modeling_ready_only=True)
    if candidates.empty:
        raise SystemExit("No modeling_ready granules found in candidates CSV.")

    granules = set(candidates["granule"].astype(str))
    granule_names = ", ".join(sorted(g.replace(".h5", "") for g in granules))

    n_lake_files, n_surrf_photons, n_lakes_with_photons, beam_lake_counts, _beam_photon_counts = count_extract_photons(
        data_dir, granules, min_depth, min_conf, max_points_per_lake
    )

    n_gee_photons = 0
    if os.path.isfile(photon_csv):
        photon_df = pd.read_csv(photon_csv)
        n_gee_photons = len(photon_df)
    elif os.path.isfile(pixel_csv):
        photon_df = pd.read_csv(pixel_csv)
        n_gee_photons = int(photon_df["n_photons"].sum()) if "n_photons" in photon_df.columns else len(photon_df)

    n_pixels = len(pd.read_csv(pixel_csv)) if os.path.isfile(pixel_csv) else 0

    n_training = 0
    n_training_lakes = "n/a"
    if os.path.isfile(predictions_csv):
        pred_df = pd.read_csv(predictions_csv)
        n_training = len(pred_df)
        n_training_lakes = str(pred_df["lake_id"].nunique())
    elif os.path.isfile(pixel_csv):
        raw = add_spectral_indices(pd.read_csv(pixel_csv))
        filtered = filter_training_rows(raw, min_depth, 20.0, min_conf, min_photons)
        n_training = len(filtered)
        n_training_lakes = str(filtered["lake_id"].nunique())

    n_unmatched = max(n_surrf_photons - n_gee_photons, 0)
    match_pct = (100.0 * n_gee_photons / n_surrf_photons) if n_surrf_photons else float("nan")
    beam_text = ", ".join(f"{k}:{v}" for k, v in sorted(beam_lake_counts.items()))

    ready = candidates.iloc[0]
    rows = [
        {
            "stage": "IS2 granule",
            "count": len(granules),
            "note": granule_names,
        },
        {
            "stage": "Detected lakes (h5 files)",
            "count": n_lake_files,
            "note": f"candidate table: {int(ready.get('n_lakes', 0))} lakes detected",
        },
        {
            "stage": "Lakes with depth photons",
            "count": n_lakes_with_photons,
            "note": beam_text or "no beam-tagged lakes with photons",
        },
        {
            "stage": "SuRRF depth photon points",
            "count": n_surrf_photons,
            "note": f"min_depth={min_depth}, min_conf={min_conf}, max_per_lake={max_points_per_lake or 'none'}",
        },
        {
            "stage": "GEE-matched S2 photons",
            "count": n_gee_photons,
            "note": (
                f"{match_pct:.0f}% match rate"
                + (f"; {n_unmatched} not returned by GEE" if n_unmatched else "")
            ),
        },
        {
            "stage": "S2 pixel samples (10 m aggregate)",
            "count": n_pixels,
            "note": "photon points aggregated to 10 m UTM grid",
        },
        {
            "stage": "Training pixels (model input)",
            "count": n_training,
            "note": f"after depth/conf/NaN filters; {n_training_lakes} lakes",
        },
    ]
    return pd.DataFrame(rows)


def build_rmse_by_depth_bin(
    predictions_csv: str,
    max_depth_bin: float | None = None,
    bin_width: float = DEPTH_BIN_WIDTH,
) -> pd.DataFrame:
    preds = pd.read_csv(predictions_csv)
    y = preds["depth_m"].to_numpy(dtype=float)

    if max_depth_bin is not None:
        bins = []
        lo = 0.0
        while lo < max_depth_bin:
            bins.append((lo, lo + bin_width))
            lo += bin_width
    else:
        bins = depth_bins_from_data(y, bin_width)
    rows: list[dict] = []
    for lo, hi in bins:
        mask = (y >= lo) & (y < hi)
        row = {
            "depth_range_m": f"{int(lo)}-{int(hi)}",
            "n_pixels": int(mask.sum()),
        }
        for key, (_, pred_col) in MODEL_COLUMNS.items():
            if pred_col not in preds.columns:
                raise SystemExit(f"Missing prediction column: {pred_col}")
            row[key] = rmse(y[mask], preds.loc[mask, pred_col].to_numpy(dtype=float))
        rows.append(row)

    row = {"depth_range_m": "Overall", "n_pixels": len(preds)}
    for key, (_, pred_col) in MODEL_COLUMNS.items():
        row[key] = rmse(y, preds[pred_col].to_numpy(dtype=float))
    rows.append(row)

    out = pd.DataFrame(rows)
    value_cols = list(MODEL_COLUMNS.keys())
    out["best_model"] = out[value_cols].idxmin(axis=1)
    return out


def format_rmse_table(df: pd.DataFrame) -> str:
    lines = [
        "| Depth range (m) | n | Green EFM (B3) | Ridge (12 feat.) | Random forest | Best |",
        "|---|---:|---:|---:|---:|---|",
    ]
    name_map = {k: v[0] for k, v in MODEL_COLUMNS.items()}
    for _, row in df.iterrows():
        vals = []
        best = row["best_model"]
        for key in MODEL_COLUMNS:
            val = row[key]
            text = f"**{val:.2f}**" if key == best and pd.notna(val) else f"{val:.2f}"
            vals.append(text if pd.notna(val) else "-")
        best_name = name_map.get(best, str(best))
        lines.append(
            f"| {row['depth_range_m']} | {int(row['n_pixels'])} | {vals[0]} | {vals[1]} | {vals[2]} | {best_name} |"
        )
    return "\n".join(lines)


def format_pipeline_table(df: pd.DataFrame) -> str:
    lines = [
        "| Stage | Count | Note |",
        "|---|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(f"| {row['stage']} | {int(row['count'])} | {row['note']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", default=CANDIDATES_CSV)
    parser.add_argument("--data-dir", default=LAKE_DATA_DIR)
    parser.add_argument("--photon-csv", default=PHOTON_PAIRS_CSV)
    parser.add_argument("--pixel-csv", default=PAIRS_CSV)
    parser.add_argument("--predictions-csv", default=CV_PREDICTIONS_CSV)
    parser.add_argument("--metrics-json", default=METRICS_JSON)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--min-photons", type=int, default=1)
    parser.add_argument("--max-points-per-lake", type=int, default=200)
    parser.add_argument(
        "--max-depth-bin",
        type=float,
        default=None,
        help="Cap depth bins at this value (m). Default: auto from max observed depth.",
    )
    args = parser.parse_args()

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    predictions_csv = resolve(args.predictions_csv)
    if not os.path.isfile(predictions_csv):
        raise SystemExit(
            f"Predictions CSV not found: {predictions_csv}\n"
            "Run: python modeling/scripts/train_depth_model.py"
        )

    os.makedirs(MODELS_DIR, exist_ok=True)

    pipeline_df = build_pipeline_summary(
        resolve(args.candidates_csv),
        resolve(args.data_dir),
        resolve(args.photon_csv),
        resolve(args.pixel_csv),
        predictions_csv,
        args.min_depth,
        args.min_conf,
        args.max_points_per_lake,
        args.min_photons,
    )

    rmse_df = build_rmse_by_depth_bin(predictions_csv, args.max_depth_bin)

    pipeline_csv = os.path.join(MODELS_DIR, "depth_pipeline_summary.csv")
    rmse_csv = os.path.join(MODELS_DIR, "depth_rmse_by_depth_bin.csv")
    pipeline_md = os.path.join(MODELS_DIR, "depth_pipeline_summary.md")
    rmse_md = os.path.join(MODELS_DIR, "depth_rmse_by_depth_bin.md")

    pipeline_df.to_csv(pipeline_csv, index=False)
    rmse_out = rmse_df.rename(
        columns={
            "green_empirical": "green_efm_rmse_m",
            "ridge": "ridge_rmse_m",
            "random_forest": "random_forest_rmse_m",
        }
    )
    rmse_out.to_csv(rmse_csv, index=False)

    with open(pipeline_md, "w", encoding="utf-8") as f:
        f.write("# Depth modeling pipeline summary\n\n")
        f.write(format_pipeline_table(pipeline_df))
        f.write("\n")

    with open(rmse_md, "w", encoding="utf-8") as f:
        f.write("# CV RMSE by depth bin (m)\n\n")
        f.write("Bold = lowest RMSE in each row. Predictions = GroupKFold held-out (test fold).\n\n")
        f.write(format_rmse_table(rmse_df))
        f.write("\n")

    if os.path.isfile(resolve(args.metrics_json)):
        with open(resolve(args.metrics_json), encoding="utf-8") as f:
            metrics = json.load(f)
        print("\n=== Overall CV metrics (from depth_model_metrics.json) ===")
        for key in MODEL_COLUMNS:
            cv = metrics[key]["cv"]
            print(f"{key:18s}  RMSE={cv['rmse']:.3f} m  R2={cv['r2']:.3f}  n={cv['n']}")

    print("\n=== Pipeline summary ===")
    print(pipeline_df.to_string(index=False))

    print("\n=== RMSE by depth bin (m) ===")
    display = rmse_df.copy()
    display = display.rename(
        columns={
            "green_empirical": "Green EFM",
            "ridge": "Ridge",
            "random_forest": "RF",
        }
    )
    print(display.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nSaved:")
    print(" ", pipeline_csv)
    print(" ", rmse_csv)
    print(" ", pipeline_md)
    print(" ", rmse_md)


if __name__ == "__main__":
    main()
