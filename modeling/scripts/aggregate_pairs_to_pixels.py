#!/usr/bin/env python3
"""Aggregate photon-level IS2-S2 pairs to Sentinel-2 pixel resolution.

Snaps each ICESat-2 point to a 10 m UTM grid (aligned with the S2 MGRS tile),
then merges points in the same pixel into one row with confidence-weighted depth.

Example:
  python modeling/scripts/aggregate_pairs_to_pixels.py
  python modeling/scripts/aggregate_pairs_to_pixels.py --in-csv modeling/out/pairs/is2_s2_pairs_SW.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODELING_ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modeling.paths import PAIRS_CSV  # noqa: E402

S2_STAC_RE = re.compile(r"^(S2[AB])_([0-9A-Z]+)_(\d{8})_\d+_L2A$")
REFL_COLS = [f"S2_{b}_refl" for b in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")]
RAW_COLS = [f"S2_{b}" for b in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")]


def utm_epsg_from_s2_id(s2_id: str, lat: float) -> int:
    """EPSG code for the UTM zone encoded in an Element84 S2 STAC id."""
    m = S2_STAC_RE.match(s2_id.strip())
    if not m:
        raise ValueError(f"Unrecognized S2 STAC id: {s2_id}")
    zone = int(m.group(2)[:2])
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, 0.5 * w.sum()))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def snap_to_pixel_grid(df: pd.DataFrame, pixel_size: float) -> pd.DataFrame:
    out = df.copy()
    epsg_by_s2: dict[str, int] = {}
    transformers: dict[int, Transformer] = {}
    utm_e = np.empty(len(out))
    utm_n = np.empty(len(out))

    for i, row in enumerate(out.itertuples(index=False)):
        s2_id = row.s2_id
        if s2_id not in epsg_by_s2:
            epsg_by_s2[s2_id] = utm_epsg_from_s2_id(s2_id, float(row.lat))
        epsg = epsg_by_s2[s2_id]
        if epsg not in transformers:
            transformers[epsg] = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        e, n = transformers[epsg].transform(float(row.lon), float(row.lat))
        utm_e[i] = e
        utm_n[i] = n

    out["utm_e"] = utm_e
    out["utm_n"] = utm_n
    out["pix_e"] = np.floor(utm_e / pixel_size) * pixel_size
    out["pix_n"] = np.floor(utm_n / pixel_size) * pixel_size
    out["pixel_id"] = (
        out["s2_id"].astype(str)
        + "_"
        + out["pix_e"].astype(int).astype(str)
        + "_"
        + out["pix_n"].astype(int).astype(str)
    )
    return out


def aggregate_group(group: pd.DataFrame, depth_agg: str) -> pd.Series:
    weights = group["conf"].to_numpy(dtype=float)
    depths = group["depth_m"].to_numpy(dtype=float)

    if depth_agg == "weighted_median":
        depth = weighted_median(depths, weights)
    elif depth_agg == "weighted_mean":
        depth = float(np.average(depths, weights=weights))
    else:
        depth = float(np.median(depths))

    row: dict = {
        "pixel_id": group["pixel_id"].iloc[0],
        "pix_e": group["pix_e"].iloc[0],
        "pix_n": group["pix_n"].iloc[0],
        "n_photons": len(group),
        "depth_m": depth,
        "depth_std": float(np.std(depths)),
        "depth_min": float(np.min(depths)),
        "depth_max": float(np.max(depths)),
        "conf": float(np.mean(weights)),
        "lat": float(group["lat"].median()),
        "lon": float(group["lon"].median()),
        "lake_id": group["lake_id"].mode().iloc[0] if not group["lake_id"].mode().empty else group["lake_id"].iloc[0],
        "granule": group["granule"].iloc[0],
        "s2_id": group["s2_id"].iloc[0],
        "s2_time_utc": group["s2_time_utc"].iloc[0],
    }

    for col in ("lake_file", "lake_quality", "detection_quality"):
        if col in group.columns:
            row[col] = group[col].iloc[0]

    for col in REFL_COLS + RAW_COLS:
        if col in group.columns:
            vals = group[col].to_numpy(dtype=float)
            row[col] = float(np.median(vals))
            row[f"{col}_pix_std"] = float(np.std(vals))

    return pd.Series(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", default=PAIRS_CSV, help="Photon-level pairing CSV")
    parser.add_argument("--out-csv", default=PAIRS_CSV, help="Output CSV (default: overwrite input)")
    parser.add_argument("--pixel-size", type=float, default=10.0, help="Pixel size in metres (default: 10)")
    parser.add_argument(
        "--depth-agg",
        choices=["weighted_median", "weighted_mean", "median"],
        default="weighted_median",
    )
    parser.add_argument("--min-photons", type=int, default=1, help="Drop pixels with fewer photons")
    parser.add_argument("--max-depth-std", type=float, default=None, help="Drop pixels with depth_std above this")
    parser.add_argument(
        "--backup-photon",
        action="store_true",
        default=True,
        help="Save photon-level copy before overwrite (default: True)",
    )
    parser.add_argument("--no-backup-photon", action="store_false", dest="backup_photon")
    args = parser.parse_args()

    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)

    in_path = resolve(args.in_csv)
    out_path = resolve(args.out_csv)

    if not os.path.isfile(in_path):
        raise SystemExit(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    required = {"lon", "lat", "depth_m", "conf", "s2_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {sorted(missing)}")

    if "pixel_id" in df.columns and df["pixel_id"].notna().any():
        print("Input already has pixel_id — assuming photon-level unless --force")
        # still proceed if user re-runs on photon backup

    n_in = len(df)
    snapped = snap_to_pixel_grid(df, args.pixel_size)
    grouped = snapped.groupby("pixel_id", sort=True)
    rows = [aggregate_group(g, args.depth_agg) for _, g in grouped]
    out_df = pd.DataFrame(rows)

    if args.min_photons > 1:
        out_df = out_df[out_df["n_photons"] >= args.min_photons].copy()
    if args.max_depth_std is not None:
        out_df = out_df[out_df["depth_std"] <= args.max_depth_std].copy()

    out_df = out_df.sort_values(["granule", "lake_id", "pix_e", "pix_n"]).reset_index(drop=True)

    if args.backup_photon and os.path.abspath(in_path) == os.path.abspath(out_path):
        backup_path = os.path.splitext(out_path)[0] + "_photon.csv"
        if not os.path.isfile(backup_path):
            df.to_csv(backup_path, index=False)
            print("Photon backup :", backup_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    photons_per_pixel = snapped.groupby("pixel_id").size()
    print("\n=== Pixel aggregation complete ===")
    print("Input photons :", n_in)
    print("Output pixels :", len(out_df))
    print("Compression   :", f"{n_in} -> {len(out_df)} ({len(out_df) / n_in:.1%} retained rows)")
    print("Photons/pixel : median", float(photons_per_pixel.median()), "| max", int(photons_per_pixel.max()))
    print("Output CSV    :", out_path)


if __name__ == "__main__":
    main()
