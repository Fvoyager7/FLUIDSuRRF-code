#!/usr/bin/env python3
"""
Rename lake h5/jpg by quality summary and export a lakes summary CSV.

Equivalent to rename_data_and_get_stats.ipynb, adapted for local step-4 outputs:
  detection_out_data/*.h5  +  detection_out_plot/*.jpg

Usage:
  python rename_lakes_and_stats.py              # preview first 5 renames
  python rename_lakes_and_stats.py --execute    # rename all + write CSV
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import h5py
import numpy as np
import pandas as pd

from lakeanalysis.utils import convert_time_to_string, get_quality_summary


def list_lake_h5(data_dir: str) -> list[str]:
    files = []
    for name in os.listdir(data_dir):
        if name.startswith("lake_") and name.endswith(".h5") and not name.startswith("_"):
            files.append(os.path.join(data_dir, name))
    files.sort()
    return files


def plot_path_for_h5(h5_path: str, plot_dir: str) -> str:
    return os.path.join(plot_dir, os.path.basename(h5_path).replace(".h5", ".jpg"))


def update_h5_and_new_name(fn: str) -> tuple[str, float, float, float, bool]:
    """Write max_depth + quality_summary into h5; return new path and qualities."""
    with h5py.File(fn, "r+") as f:
        lake_quality = float(f["properties"]["lake_quality"][()])
        detection_quality = float(f["properties"]["detection_quality"][()])
        surf_elev = float(f["properties"]["surface_elevation"][()])
        depth = f["depth_data"]["depth"][()]
        xatc = f["depth_data"]["xatc"][()]
        fitbed = f["depth_data"]["h_fit_bed"][()]
        conf = f["depth_data"]["conf"][()]
        xtent = f["properties"]["surface_extent_detection"][()]
        isdepth = (
            (xatc >= xtent[0])
            & (xatc <= xtent[-1])
            & (fitbed < (surf_elev - 0.5))
            & (depth < 50)
        )
        max_depth = float(np.percentile(depth[isdepth], 95)) if np.any(isdepth) else 0.0
        quality_summary = float(get_quality_summary(detection_quality, lake_quality))

        for key in ("max_depth", "quality_summary"):
            if key in f["properties"]:
                del f["properties"][key]
        f.create_dataset("properties/max_depth", data=max_depth)
        f.create_dataset("properties/quality_summary", data=quality_summary)

    quality_summary = float(get_quality_summary(detection_quality, lake_quality))
    filenameonly = fn[fn.find("lake_") :]
    pathtofile = fn[: fn.find("lake_")]
    parm_list = filenameonly.split("_")
    parm_list[1] = "%08i" % int(np.round((100 - quality_summary) * 100000))
    if len(parm_list) > 2 and parm_list[2].isnumeric():
        del parm_list[2]
    newpath = pathtofile + "_".join(parm_list)
    return newpath, detection_quality, lake_quality, quality_summary, True


def rename_files(
    filelist: list[str],
    plot_dir: str,
    dry_run: bool = True,
    preview_n: int = 5,
) -> list[tuple[str, str]]:
    """Rename h5 + matching jpg. Returns list of (old_path, new_path)."""
    targets = filelist[:preview_n] if dry_run else filelist
    renamed: list[tuple[str, str]] = []
    num_missing = 0

    for i, fn in enumerate(targets):
        print("processing %5i / %5i" % (i + 1, len(targets)), end="\r")
        try:
            newpath, det_q, lake_q, qual_sum, ok = update_h5_and_new_name(fn)
        except Exception:
            num_missing += 1
            traceback.print_exc()
            continue

        plot_old = plot_path_for_h5(fn, plot_dir)
        plot_new = plot_path_for_h5(newpath, plot_dir)

        if dry_run:
            print()
            print("  detection_q=%.4f lake_q=%.4f summary=%.4f" % (det_q, lake_q, qual_sum))
            print("  h5:", fn)
            print("  ->:", newpath)
            print("  jpg:", plot_old, "->", plot_new)
        else:
            if fn != newpath:
                os.rename(fn, newpath)
            if os.path.isfile(plot_old) and plot_old != plot_new:
                os.rename(plot_old, plot_new)
            renamed.append((fn, newpath))

    print()
    if num_missing:
        print("Warning: %i file(s) could not be processed." % num_missing)
    return renamed


def build_summary_csv(filelist: list[str], csv_path: str) -> pd.DataFrame:
    rows = []
    num_missing = 0
    for i, fn in enumerate(filelist):
        print("stats %5i / %5i" % (i + 1, len(filelist)), end="\r")
        try:
            with h5py.File(fn, "r") as f:
                props = f["properties"]
                row = {
                    "file_name": fn,
                    "ice_sheet": props["ice_sheet"][()].decode("utf-8"),
                    "melt_season": props["melt_season"][()].decode("utf-8"),
                    "basin": props["polygon_name"][()].decode("utf-8").replace("simplified_", ""),
                    "quality_summary": float(props["quality_summary"][()]),
                    "max_depth": float(props["max_depth"][()]),
                    "length_water_surfaces": float(props["length_water_surfaces"][()]),
                    "surface_elevation": float(props["surface_elevation"][()]),
                    "n_photons_where_water": float(props["n_photons_where_water"][()]),
                    "lon": float(props["lon"][()]),
                    "lat": float(props["lat"][()]),
                    "date_time": convert_time_to_string(float(np.mean(f["mframe_data"]["dt"][()]))),
                    "lon_min": float(props["lon_min"][()]),
                    "lon_max": float(props["lon_max"][()]),
                    "lat_min": float(props["lat_min"][()]),
                    "lat_max": float(props["lat_max"][()]),
                    "cycle_number": int(props["cycle_number"][()]),
                    "rgt": int(props["rgt"][()]),
                    "gtx": props["gtx"][()].decode("utf-8"),
                    "beam_strength": props["beam_strength"][()].decode("utf-8"),
                    "beam_number": int(props["beam_number"][()]),
                    "detection_quality": float(props["detection_quality"][()]),
                    "lake_quality": float(props["lake_quality"][()]),
                    "granule_id": props["granule_id"][()].decode("utf-8"),
                    "lake_id": props["lake_id"][()].decode("utf-8"),
                }
            rows.append(row)
        except Exception:
            num_missing += 1
            traceback.print_exc()

    print()
    df = pd.DataFrame(rows).sort_values("quality_summary", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print("Wrote %s (%i lakes, %i missing)" % (csv_path, len(df), num_missing))
    if len(df):
        print(
            "Quality summary: min=%.3f max=%.3f median=%.3f"
            % (df.quality_summary.min(), df.quality_summary.max(), df.quality_summary.median())
        )
        print("Max depth (m): min=%.2f max=%.2f median=%.2f" % (
            df.max_depth.min(), df.max_depth.max(), df.max_depth.median()))
    return df


def main():
    parser = argparse.ArgumentParser(description="Rename lakes by quality and export summary CSV")
    parser.add_argument("--data-dir", default="detection_out_data")
    parser.add_argument("--plot-dir", default="detection_out_plot")
    parser.add_argument("--stat-dir", default="detection_out_stat")
    parser.add_argument("--csv-name", default="lakes_summary_SW.csv")
    parser.add_argument("--execute", action="store_true", help="Actually rename files (default: preview only)")
    parser.add_argument("--preview", type=int, default=5, help="Number of files to show in preview mode")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    filelist = list_lake_h5(args.data_dir)
    if not filelist:
        print("No lake_*.h5 in", args.data_dir)
        sys.exit(1)

    n_jpg = sum(1 for fn in filelist if os.path.isfile(plot_path_for_h5(fn, args.plot_dir)))
    print("Found %i h5 files, %i matching jpg plots" % (len(filelist), n_jpg))

    if not args.execute:
        print("\n--- DRY RUN (first %i files) ---" % min(args.preview, len(filelist)))
        rename_files(filelist, args.plot_dir, dry_run=True, preview_n=args.preview)
        print("\nNo files changed. Re-run with --execute to apply renames and write CSV.")
        return

    print("\n--- RENAMING ALL FILES ---")
    rename_files(filelist, args.plot_dir, dry_run=False)
    filelist = list_lake_h5(args.data_dir)

    csv_path = os.path.join(args.stat_dir, args.csv_name)
    build_summary_csv(filelist, csv_path)
    print("\nDone. Files are sorted by quality in the filename prefix (smaller = better).")
    print("Next step: python make_quicklook_sw.py --limit 3")


if __name__ == "__main__":
    main()
