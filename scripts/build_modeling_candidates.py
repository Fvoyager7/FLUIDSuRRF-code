#!/usr/bin/env python3
"""Merge IS2-S2 match quality with lake detection stats for modeling sample selection.

Joins granule-level Sentinel-2 match metadata with lake counts/depths from
detect_lakes.py outputs. Flags granules that have both good S2 pairing and
at least one detected supraglacial lake suitable for depth retrieval.

Example:
  python scripts/build_modeling_candidates.py
  python scripts/build_modeling_candidates.py --max-timediff-hours 15 --min-lakes 1
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRANULE_IN_NAME = re.compile(r"(ATL03_\d{8}_\d{8}_\d{3}_\d{2})\.h5")


def granule_from_lake_filename(path: str) -> str | None:
    """Extract ATL03 granule id from a lake h5/jpg filename."""
    m = GRANULE_IN_NAME.search(os.path.basename(path))
    return m.group(1) + ".h5" if m else None


def aggregate_lakes_from_summary(summary_csv: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if "granule_id" not in df.columns:
        raise ValueError(f"Expected granule_id column in {summary_csv}")
    grouped = df.groupby("granule_id", dropna=False)
    out = grouped.agg(
        n_lakes=("granule_id", "size"),
        n_lakes_quality_gt0=("lake_quality", lambda s: int((s > 0).sum())),
        max_depth_m=("max_depth", "max"),
        median_quality=("quality_summary", "median"),
        median_lake_quality=("lake_quality", "median"),
    ).reset_index()
    out = out.rename(columns={"granule_id": "granule"})
    return out


def aggregate_lakes_from_h5(data_dir: str) -> pd.DataFrame:
    rows: list[dict] = []
    if not os.path.isdir(data_dir):
        return pd.DataFrame(
            columns=[
                "granule",
                "n_lakes",
                "n_lakes_quality_gt0",
                "max_depth_m",
                "median_quality",
                "median_lake_quality",
            ]
        )

    for name in os.listdir(data_dir):
        if not (name.startswith("lake_") and name.endswith(".h5")):
            continue
        path = os.path.join(data_dir, name)
        granule = granule_from_lake_filename(name)
        if not granule:
            continue
        try:
            with h5py.File(path, "r") as f:
                lake_quality = float(f["properties"]["lake_quality"][()])
                detection_quality = float(f["properties"]["detection_quality"][()])
                if "quality_summary" in f["properties"]:
                    quality_summary = float(f["properties"]["quality_summary"][()])
                else:
                    quality_summary = detection_quality + lake_quality
                if "max_depth" in f["properties"]:
                    max_depth = float(f["properties"]["max_depth"][()])
                else:
                    depth = f["depth_data"]["depth"][()]
                    max_depth = float(np.nanmax(depth)) if len(depth) else 0.0
        except Exception as exc:
            print("Warning: could not read", path, ":", exc, file=sys.stderr)
            continue
        rows.append(
            {
                "granule": granule,
                "lake_quality": lake_quality,
                "quality_summary": quality_summary,
                "max_depth_m": max_depth,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "granule",
                "n_lakes",
                "n_lakes_quality_gt0",
                "max_depth_m",
                "median_quality",
                "median_lake_quality",
            ]
        )

    df = pd.DataFrame(rows)
    grouped = df.groupby("granule", dropna=False)
    return grouped.agg(
        n_lakes=("granule", "size"),
        n_lakes_quality_gt0=("lake_quality", lambda s: int((s > 0).sum())),
        max_depth_m=("max_depth_m", "max"),
        median_quality=("quality_summary", "median"),
        median_lake_quality=("lake_quality", "median"),
    ).reset_index()


def load_lake_stats(data_dir: str, summary_csv: str | None) -> pd.DataFrame:
    if summary_csv and os.path.isfile(summary_csv):
        return aggregate_lakes_from_summary(summary_csv)
    return aggregate_lakes_from_h5(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-csv",
        default="granule_lists/GrIS_2022_GRE_2000_SW_is2_s2_matches.csv",
        help="IS2-S2 match table from match_is2_sentinel2_sw_2022.py",
    )
    parser.add_argument(
        "--data-dir",
        default="detection_out_data",
        help="Lake h5 directory from detect_lakes.py",
    )
    parser.add_argument(
        "--summary-csv",
        default="detection_out_stat/lakes_summary_SW.csv",
        help="Optional lake summary CSV (preferred if present)",
    )
    parser.add_argument(
        "--out-csv",
        default="granule_lists/modeling_candidates_SW.csv",
        help="Output merged modeling candidate table",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=10.0,
        help="Maximum scene cloud cover %% for good S2 (default: 10)",
    )
    parser.add_argument(
        "--max-timediff-hours",
        type=float,
        default=20.0,
        help="Maximum |IS2-S2| hours for good S2 (default: 20)",
    )
    parser.add_argument(
        "--allow-cross-day-s2",
        action="store_true",
        help="Allow cross-day S2 matches to count as good_s2_match (default: require same UTC day)",
    )
    parser.add_argument(
        "--min-lakes",
        type=int,
        default=1,
        help="Minimum lake count for modeling_ready (default: 1)",
    )
    parser.add_argument(
        "--min-lakes-quality-gt0",
        type=int,
        default=1,
        help="Minimum lakes with lake_quality>0 for modeling_ready (default: 1)",
    )
    parser.add_argument(
        "--require-detection",
        action="store_true",
        help="Only output granules that have been run through detect_lakes.py",
    )
    args = parser.parse_args()

    match_path = args.match_csv if os.path.isabs(args.match_csv) else os.path.join(REPO_ROOT, args.match_csv)
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(REPO_ROOT, args.data_dir)
    summary_path = (
        args.summary_csv
        if args.summary_csv and os.path.isabs(args.summary_csv)
        else os.path.join(REPO_ROOT, args.summary_csv) if args.summary_csv else None
    )
    out_path = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(REPO_ROOT, args.out_csv)

    if not os.path.isfile(match_path):
        raise SystemExit(f"Match CSV not found: {match_path}")

    matches = pd.read_csv(match_path)
    lakes = load_lake_stats(data_dir, summary_path if summary_path and os.path.isfile(summary_path) else None)

    merged = matches.merge(lakes, on="granule", how="left")
    for col in ("n_lakes", "n_lakes_quality_gt0"):
        merged[col] = merged[col].fillna(0).astype(int)
    for col in ("max_depth_m", "median_quality", "median_lake_quality"):
        merged[col] = merged[col].astype(float)

    require_same_day = not args.allow_cross_day_s2
    has_s2 = merged["s2_id"].notna()
    cloud_ok = merged["s2_cloud_cover"].fillna(999) < args.max_cloud
    time_ok = merged["timediff_hours"].fillna(999) < args.max_timediff_hours
    same_day_ok = merged["same_day"].fillna(False) if require_same_day else True

    merged["good_s2_match"] = has_s2 & cloud_ok & time_ok & same_day_ok
    merged["has_lakes"] = merged["n_lakes"] >= args.min_lakes
    merged["has_quality_lakes"] = merged["n_lakes_quality_gt0"] >= args.min_lakes_quality_gt0
    merged["detection_run"] = merged["n_lakes"] > 0
    merged["modeling_ready"] = (
        merged["good_s2_match"] & merged["has_lakes"] & merged["has_quality_lakes"]
    )

    if args.require_detection:
        merged = merged[merged["detection_run"]].copy()

    merged = merged.sort_values(
        ["modeling_ready", "good_s2_match", "n_lakes_quality_gt0", "timediff_hours"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)

    n_total = len(merged)
    n_good_s2 = int(merged["good_s2_match"].sum())
    n_with_lakes = int((merged["n_lakes"] > 0).sum())
    n_ready = int(merged["modeling_ready"].sum())

    print("\n=== Modeling candidates table complete ===")
    print("Output CSV       :", out_path)
    print("Granules in match:", n_total)
    print("Good S2 match    :", n_good_s2)
    print("With lakes found :", n_with_lakes)
    print("Modeling ready   :", n_ready)
    if n_ready:
        ready = merged[merged["modeling_ready"]]
        print("Ready granules   :")
        for _, row in ready.iterrows():
            print(
                " ",
                row["granule"],
                f"| lakes={int(row['n_lakes'])}",
                f"| dt={row['timediff_hours']:.1f}h",
                f"| cloud={row['s2_cloud_cover']:.1f}%",
            )


if __name__ == "__main__":
    main()
