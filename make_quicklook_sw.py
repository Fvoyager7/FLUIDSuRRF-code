#!/usr/bin/env python3
"""
Step 5: ICESat-2 profile + Sentinel-2 (GEE) quicklook composite plots.

Reads lake .h5 files from detect_lakes.py (step 4) and writes combined imagery
plots to detection_out_quicklook/.

Prerequisites:
  1. conda activate eeicelakes-env
  2. pip install earthengine-api rasterio   (if not already installed)
  3. earthengine authenticate               (one-time GEE login)
  4. Google Earth Engine account with API access enabled

Example (all SW lakes from step 4):
  python make_quicklook_sw.py

Test on top 4 lakes by quality_summary (from rename step CSV):
  python make_quicklook_sw.py --top-quality 4 --gee-project YOUR_GCP_PROJECT_ID

Force re-download S2 and regenerate plots:
  python make_quicklook_sw.py --replace-existing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback


def setup_geospatial_env():
    """Set PROJ/GDAL paths for Windows conda (prefer rasterio's proj_data)."""
    try:
        import rasterio
        proj_dir = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
        if os.path.isdir(proj_dir):
            os.environ["PROJ_LIB"] = proj_dir
            os.environ["PROJ_DATA"] = proj_dir
    except Exception:
        pass
    try:
        import pyogrio
        gdal_data = os.path.join(os.path.dirname(pyogrio.__file__), "proj_data")
        if os.path.isdir(gdal_data):
            os.environ["GDAL_DATA"] = gdal_data
    except Exception:
        pass


def time_string(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return f"{d} days, {h} hrs, {m} mins, {s} secs"


def list_lake_h5(data_dir: str) -> list[str]:
    files = []
    for name in os.listdir(data_dir):
        if name.startswith("lake_") and name.endswith(".h5") and not name.startswith("_"):
            files.append(os.path.join(data_dir, name))
    files.sort()
    return files


def list_top_quality_lakes(summary_csv: str, n: int, data_dir: str) -> list[str]:
    """Pick top N lakes by quality_summary from rename_lakes_and_stats CSV."""
    import pandas as pd

    df = pd.read_csv(summary_csv).sort_values("quality_summary", ascending=False)
    paths = []
    for _, row in df.head(n).iterrows():
        fn = str(row["file_name"]).replace("/", os.sep)
        if not os.path.isfile(fn):
            fn = os.path.join(data_dir, os.path.basename(fn))
        if os.path.isfile(fn):
            paths.append(fn)
        else:
            print("Warning: missing file for quality rank:", row.get("lake_id", fn))
    return paths


def main():
    parser = argparse.ArgumentParser(description="GEE Sentinel-2 + ICESat-2 quicklook plots (step 5)")
    parser.add_argument("--data-dir", default="detection_out_data",
                        help="Input lake .h5 directory from detect_lakes.py")
    parser.add_argument("--out-dir", default="detection_out_quicklook",
                        help="Output combined quicklook .jpg directory")
    parser.add_argument("--imagery-dir", default="quicklook_imagery",
                        help="Cache directory for downloaded S2 GeoTIFF mosaics")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N lakes by filename order (for testing)")
    parser.add_argument("--top-quality", type=int, default=None,
                        help="Process top N lakes by quality_summary (uses --summary-csv)")
    parser.add_argument("--summary-csv", default="detection_out_stat/lakes_summary_SW.csv",
                        help="Lake summary CSV from rename_lakes_and_stats.py")
    parser.add_argument("--replace-existing", action="store_true",
                        help="Regenerate plots even if output jpg already exists")
    parser.add_argument("--lake", default=None,
                        help="Process a single lake by .h5 basename or path (e.g. lake_09813415_..._gt3r_0106.h5)")
    parser.add_argument("--gee-project", default=os.environ.get("EE_PROJECT"),
                        help="Google Cloud project ID for Earth Engine (or set EE_PROJECT env var)")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    setup_geospatial_env()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ee

    print("Initializing Google Earth Engine...")
    try:
        if args.gee_project:
            ee.Initialize(project=args.gee_project)
        else:
            ee.Initialize()
    except Exception as e:
        print("GEE initialization failed:", e)
        print("\n1) Run once:  earthengine authenticate")
        print("2) If prompted for a Cloud project, pass it:")
        print("     python make_quicklook_sw.py --gee-project YOUR_GCP_PROJECT_ID")
        print("   or set:  $env:EE_PROJECT=\"YOUR_GCP_PROJECT_ID\"")
        sys.exit(1)

    from lakeanalysis.quicklook_core import plot_IS2_imagery

    if not os.path.isdir(args.data_dir):
        print("Data directory not found:", args.data_dir)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.imagery_dir, exist_ok=True)

    filelist = list_lake_h5(args.data_dir)
    if args.lake:
        lake_arg = args.lake
        if not lake_arg.endswith(".h5"):
            lake_arg += ".h5"
        if os.path.isfile(lake_arg):
            filelist = [os.path.abspath(lake_arg)]
        else:
            fn = os.path.join(args.data_dir, os.path.basename(lake_arg))
            if os.path.isfile(fn):
                filelist = [fn]
            else:
                print("Lake file not found:", args.lake)
                sys.exit(1)
    elif args.top_quality:
        if not os.path.isfile(args.summary_csv):
            print("Summary CSV not found:", args.summary_csv)
            print("Run first: python rename_lakes_and_stats.py --execute")
            sys.exit(1)
        filelist = list_top_quality_lakes(args.summary_csv, args.top_quality, args.data_dir)
        print(f"\nTop {args.top_quality} lakes by quality_summary:")
        import pandas as pd
        top = pd.read_csv(args.summary_csv).sort_values("quality_summary", ascending=False).head(args.top_quality)
        for i, row in top.iterrows():
            print("  %.1f  depth=%.2fm  %s" % (
                row.quality_summary, row.max_depth, os.path.basename(str(row.file_name))))
    elif args.limit:
        filelist = filelist[: args.limit]

    if not filelist:
        print("No lake_*.h5 files in", args.data_dir)
        sys.exit(1)

    print(f"\nProcessing {len(filelist)} lake file(s)...")
    print(f"  input:  {args.data_dir}/")
    print(f"  output: {args.out_dir}/")
    print(f"  S2 tif: {args.imagery_dir}/\n")

    settings = {
        "re_download": True,
        "img_aspect": 1.0,
        "days_buffer": 5,
        "max_cloud_scene": 20,
        "max_cloud_mask": 40,
        "n_imgs_mosaic": 5,
        "mosaic_method": "mean",
        "scale_out": 10,
        "gamma_value": 1.0,
        "xlm": [None, None],
        "ylm": [None, None],
        "return_fig": False,
    }

    tstart = time.time()
    idone = 0
    nskip = 0

    for i, fn in enumerate(filelist):
        base = os.path.basename(fn)
        fn_plot = os.path.join(args.out_dir, base.replace(".h5", "_imagery.jpg"))
        fn_imagery = os.path.join(args.imagery_dir, base.replace(".h5", "_imagery.tif"))

        if os.path.isfile(fn_plot) and not args.replace_existing:
            nskip += 1
            continue

        elapsed = time_string(time.time() - tstart)
        print(f"[{i + 1}/{len(filelist)}] {base}  (elapsed {elapsed})")

        fig = plt.figure(figsize=[10, 4.35])
        gs = fig.add_gridspec(ncols=9, nrows=1)
        axs = [fig.add_subplot(gs[0, :4]), fig.add_subplot(gs[0, 4:])]

        try:
            plot_IS2_imagery(fn=fn, imagery_filename=fn_imagery, **settings, axes=axs)
            fig.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.4)
            fig.savefig(fn_plot, dpi=300)
            idone += 1
            print(f"  -> saved {fn_plot}")
        except Exception:
            print(f"  -> FAILED for {base}")
            traceback.print_exc()
        finally:
            plt.close(fig)

    total = time_string(time.time() - tstart)
    print(f"\nDone: {idone} created, {nskip} skipped, {len(filelist)} total ({total})")


if __name__ == "__main__":
    main()
