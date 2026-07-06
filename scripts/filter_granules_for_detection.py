#!/usr/bin/env python3
"""Write a granule list CSV for detect_lakes.py from good S2 matches not yet detected.

Selects granules with acceptable IS2-S2 pairing that have not produced lake h5
outputs yet, so run_sw_lake_detection.py can prioritize them.

Example:
  python scripts/filter_granules_for_detection.py
  python scripts/filter_granules_for_detection.py --out-csv granule_lists/SW_good_s2_todo.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Reuse granule parsing from build_modeling_candidates
sys.path.insert(0, os.path.dirname(__file__))
from build_modeling_candidates import aggregate_lakes_from_h5, granule_from_lake_filename, load_lake_stats  # noqa: E402


def load_granule_submit_table(path: str) -> pd.DataFrame:
    """Load HTCondor-style granule list (no header)."""
    return pd.read_csv(
        path,
        header=None,
        names=["granule", "geojson", "description", "geojson_clip"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-csv",
        default="granule_lists/GrIS_2022_GRE_2000_SW_is2_s2_matches.csv",
    )
    parser.add_argument(
        "--granule-list",
        default="granule_lists/GrIS_2022_GRE_2000_SW.csv",
        help="Source submit list with geojson paths for detect_lakes.py",
    )
    parser.add_argument(
        "--candidates-csv",
        default=None,
        help="Optional modeling_candidates_SW.csv (if already built)",
    )
    parser.add_argument(
        "--data-dir",
        default="detection_out_data",
    )
    parser.add_argument(
        "--summary-csv",
        default="detection_out_stat/lakes_summary_SW.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="granule_lists/GrIS_2022_GRE_2000_SW_good_s2_todo.csv",
    )
    parser.add_argument("--max-cloud", type=float, default=10.0)
    parser.add_argument("--max-timediff-hours", type=float, default=20.0)
    parser.add_argument(
        "--allow-cross-day-s2",
        action="store_true",
        help="Allow cross-day S2 in good-match filter",
    )
    args = parser.parse_args()

    match_path = args.match_csv if os.path.isabs(args.match_csv) else os.path.join(REPO_ROOT, args.match_csv)
    granule_path = (
        args.granule_list if os.path.isabs(args.granule_list) else os.path.join(REPO_ROOT, args.granule_list)
    )
    out_path = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(REPO_ROOT, args.out_csv)
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(REPO_ROOT, args.data_dir)
    summary_path = (
        args.summary_csv
        if os.path.isabs(args.summary_csv)
        else os.path.join(REPO_ROOT, args.summary_csv)
    )

    if not os.path.isfile(match_path):
        raise SystemExit(f"Match CSV not found: {match_path}")
    if not os.path.isfile(granule_path):
        raise SystemExit(f"Granule list not found: {granule_path}")

    if args.candidates_csv:
        cand_path = (
            args.candidates_csv
            if os.path.isabs(args.candidates_csv)
            else os.path.join(REPO_ROOT, args.candidates_csv)
        )
        if not os.path.isfile(cand_path):
            raise SystemExit(f"Candidates CSV not found: {cand_path}")
        merged = pd.read_csv(cand_path)
        todo = merged[merged["good_s2_match"] & ~merged["detection_run"]].copy()
        todo_granules = set(todo["granule"])
    else:
        matches = pd.read_csv(match_path)
        lakes = load_lake_stats(
            data_dir,
            summary_path if os.path.isfile(summary_path) else None,
        )
        merged = matches.merge(lakes, on="granule", how="left")
        merged["n_lakes"] = merged["n_lakes"].fillna(0).astype(int)

        require_same_day = not args.allow_cross_day_s2
        good = (
            merged["s2_id"].notna()
            & (merged["s2_cloud_cover"].fillna(999) < args.max_cloud)
            & (merged["timediff_hours"].fillna(999) < args.max_timediff_hours)
        )
        if require_same_day:
            good &= merged["same_day"].fillna(False)
        todo = merged[good & (merged["n_lakes"] == 0)].copy()
        todo_granules = set(todo["granule"])

    submit = load_granule_submit_table(granule_path)
    out_df = submit[submit["granule"].isin(todo_granules)].copy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, header=False, index=False)

    print("\n=== Good-S2 granules pending detection ===")
    print("Output list      :", out_path)
    print("Granules to run  :", len(out_df))
    if len(out_df):
        print("First            :", out_df.granule.iloc[0])
        print("Last             :", out_df.granule.iloc[-1])
    print("\nRun detection with:")
    print(f"  python scripts/run_sw_lake_detection.py --granule-list {out_path}")


if __name__ == "__main__":
    main()
