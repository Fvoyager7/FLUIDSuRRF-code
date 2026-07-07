#!/usr/bin/env python3
"""Extract IS2 depth points with co-located Sentinel-2 surface reflectance (GEE).

Reads lake .h5 files from detect_lakes.py and pairs each along-track depth
measurement with S2 band values from the matched scene in
modeling/lists/modeling_candidates_SW.csv.

Example:
  python modeling/scripts/extract_is2_s2_pairs.py --dry-run
  python modeling/scripts/extract_is2_s2_pairs.py --gee-project ee-wenlinshen777
  python modeling/scripts/extract_is2_s2_pairs.py --granule ATL03_20220714010847_03381603_007_01.h5
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODELING_ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modeling.paths import CANDIDATES_CSV, LAKE_DATA_DIR, PAIRS_CSV  # noqa: E402

GRANULE_IN_NAME = re.compile(r"(ATL03_\d{14}_\d{8}_\d{3}_\d{2})")
S2_STAC_RE = re.compile(r"^(S2[AB])_([0-9A-Z]+)_(\d{8})_\d+_L2A$")

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def granule_from_lake_filename(path: str) -> str | None:
    m = GRANULE_IN_NAME.search(os.path.basename(path))
    return m.group(1) + ".h5" if m else None


def parse_s2_stac_id(s2_id: str) -> dict[str, str]:
    m = S2_STAC_RE.match(s2_id.strip())
    if not m:
        raise ValueError(f"Unrecognized S2 STAC id: {s2_id}")
    return {"platform": m.group(1), "mgrs_tile": m.group(2), "date": m.group(3)}


def load_candidate_table(path: str, modeling_ready_only: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if modeling_ready_only:
        if "modeling_ready" not in df.columns:
            raise ValueError(f"Column modeling_ready missing in {path}")
        if df["modeling_ready"].dtype == bool:
            df = df[df["modeling_ready"]].copy()
        else:
            df = df[df["modeling_ready"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if df.empty:
        raise SystemExit("No granules selected from candidate table.")
    if "s2_id" not in df.columns:
        raise ValueError(f"Column s2_id missing in {path}")
    return df


def list_lake_files(data_dir: str, granules: set[str] | None) -> list[str]:
    paths: list[str] = []
    for name in sorted(os.listdir(data_dir)):
        if not (name.startswith("lake_") and name.endswith(".h5")):
            continue
        path = os.path.join(data_dir, name)
        granule = granule_from_lake_filename(name)
        if granules is not None and granule not in granules:
            continue
        paths.append(path)
    return paths


def extract_depth_points(
    lake_path: str,
    min_depth: float,
    min_conf: float,
    max_points_per_lake: int | None,
) -> pd.DataFrame:
    from lakeanalysis.utils import read_melt_lake_h5

    lk = read_melt_lake_h5(lake_path)
    dfd = lk["depth_data"].copy()
    lake_id = os.path.splitext(os.path.basename(lake_path))[0]

    mask = (dfd["depth"] > min_depth) & (dfd["conf"] >= min_conf)
    pts = dfd.loc[mask, ["lat", "lon", "depth", "conf", "xatc"]].copy()
    if pts.empty:
        return pd.DataFrame()

    if max_points_per_lake and len(pts) > max_points_per_lake:
        pts = pts.sample(n=max_points_per_lake, random_state=0).sort_index()

    pts.insert(0, "lake_id", lake_id)
    pts.insert(1, "lake_file", os.path.basename(lake_path))
    pts["granule"] = granule_from_lake_filename(lake_path)
    if "lake_quality" in lk:
        pts["lake_quality"] = float(lk["lake_quality"])
    if "detection_quality" in lk:
        pts["detection_quality"] = float(lk["detection_quality"])
    return pts.reset_index(drop=True)


def init_earth_engine(gee_project: str | None) -> None:
    import ee

    if gee_project:
        ee.Initialize(project=gee_project)
    else:
        ee.Initialize()


def get_s2_image(s2_id: str, s2_time_utc: str, lon: float, lat: float):
    import ee

    meta = parse_s2_stac_id(s2_id)
    mgrs = meta["mgrs_tile"]
    mgrs_tiles = [mgrs]
    if not mgrs.startswith("T"):
        mgrs_tiles.append(f"T{mgrs}")

    acq = pd.Timestamp(s2_time_utc)
    if acq.tzinfo is not None:
        acq = acq.tz_convert("UTC").tz_localize(None)

    point = ee.Geometry.Point([float(lon), float(lat)])
    window_start = (acq - pd.Timedelta(hours=18)).isoformat() + "Z"
    window_end = (acq + pd.Timedelta(hours=18)).isoformat() + "Z"
    base = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point.buffer(3000))
        .filterDate(window_start, window_end)
    )

    for tile in mgrs_tiles:
        col = base.filter(ee.Filter.eq("MGRS_TILE", tile))
        if col.size().getInfo() > 0:
            target_ms = int(acq.timestamp() * 1000)

            def time_diff(img, t=target_ms):
                diff = ee.Number(img.get("system:time_start")).subtract(t).abs()
                return img.set("time_diff_ms", diff)

            return col.map(time_diff).sort("time_diff_ms").first().set("stac_id", s2_id)

    target_ms = int(acq.timestamp() * 1000)

    def time_diff(img, t=target_ms):
        diff = ee.Number(img.get("system:time_start")).subtract(t).abs()
        return img.set("time_diff_ms", diff)

    col = base.map(time_diff).sort("time_diff_ms")
    if col.size().getInfo() < 1:
        raise RuntimeError(f"No GEE image for {s2_id} near ({lon:.5f}, {lat:.5f})")
    return col.first().set("stac_id", s2_id)


def sample_s2_bands(image, points: pd.DataFrame, scale: int) -> pd.DataFrame:
    import ee

    points = points.reset_index(drop=True)

    features = []
    for row in points.itertuples(index=False):
        props = {
            "point_idx": int(row.point_idx),
            "lake_id": row.lake_id,
            "depth_m": float(row.depth),
            "conf": float(row.conf),
            "xatc": float(row.xatc),
        }
        features.append(ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), props))

    fc = ee.FeatureCollection(features)
    sampled = image.select(S2_BANDS).sampleRegions(
        collection=fc,
        scale=scale,
        geometries=False,
    )
    rows = sampled.getInfo().get("features", [])
    if not rows:
        return pd.DataFrame()

    records = [dict(feat.get("properties", {})) for feat in rows]
    out = pd.DataFrame(records)
    rename = {b: f"S2_{b}" for b in S2_BANDS if b in out.columns}
    out = out.rename(columns=rename)
    for b in S2_BANDS:
        col = f"S2_{b}"
        if col in out.columns:
            out[f"{col}_refl"] = out[col] / 10000.0
    return out


def merge_point_metadata(sampled: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "point_idx",
        "lake_id",
        "lake_file",
        "granule",
        "lat",
        "lon",
        "lake_quality",
        "detection_quality",
    ]
    meta = points[[c for c in meta_cols if c in points.columns]].copy()
    if "point_idx" not in sampled.columns:
        return sampled
    overlap = [c for c in meta.columns if c in sampled.columns and c != "point_idx"]
    sampled = sampled.drop(columns=overlap, errors="ignore")
    return sampled.merge(meta, on="point_idx", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-csv",
        default=CANDIDATES_CSV,
        help="Granule table with s2_id and modeling_ready flags",
    )
    parser.add_argument(
        "--data-dir",
        default=LAKE_DATA_DIR,
        help="Lake h5 directory (default: repo detection_out_data/)",
    )
    parser.add_argument(
        "--out-csv",
        default=PAIRS_CSV,
        help="Merged IS2-S2 pairing output",
    )
    parser.add_argument("--granule", default=None, help="Process a single granule id")
    parser.add_argument(
        "--modeling-ready-only",
        action="store_true",
        default=True,
        help="Only granules with modeling_ready=True (default: True)",
    )
    parser.add_argument(
        "--all-granules-with-s2",
        action="store_true",
        help="Ignore modeling_ready; use any granule with s2_id in the table",
    )
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum depth (m)")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Minimum SuRRF confidence")
    parser.add_argument("--max-lakes", type=int, default=None, help="Limit number of lake files")
    parser.add_argument(
        "--max-points-per-lake",
        type=int,
        default=200,
        help="Subsample depth points per lake (default: 200; 0 = no limit)",
    )
    parser.add_argument("--s2-scale", type=int, default=10, help="GEE sample scale in metres")
    parser.add_argument("--gee-project", default=os.environ.get("EE_PROJECT"), help="GCP project for GEE")
    parser.add_argument("--dry-run", action="store_true", help="List lakes/points without calling GEE")
    args = parser.parse_args()

    if args.all_granules_with_s2:
        args.modeling_ready_only = False

    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)

    candidates_path = resolve(args.candidates_csv)
    data_dir = resolve(args.data_dir)
    out_path = resolve(args.out_csv)

    if not os.path.isfile(candidates_path):
        raise SystemExit(f"Candidate CSV not found: {candidates_path}")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data directory not found: {data_dir}")

    candidates = load_candidate_table(candidates_path, args.modeling_ready_only)
    if args.granule:
        g = args.granule if args.granule.endswith(".h5") else args.granule + ".h5"
        candidates = candidates[candidates["granule"] == g]
        if candidates.empty:
            raise SystemExit(f"Granule not found in candidate table: {g}")

    granule_set = set(candidates["granule"].astype(str))
    granule_info = candidates.set_index("granule")
    lake_files = list_lake_files(data_dir, granule_set)
    if args.max_lakes:
        lake_files = lake_files[: args.max_lakes]

    if not lake_files:
        raise SystemExit("No lake h5 files found for selected granules.")

    max_pts = None if args.max_points_per_lake == 0 else args.max_points_per_lake
    all_points: list[pd.DataFrame] = []
    for lake_path in lake_files:
        pts = extract_depth_points(lake_path, args.min_depth, args.min_conf, max_pts)
        if pts.empty:
            continue
        granule = pts["granule"].iloc[0]
        info = granule_info.loc[granule]
        pts["s2_id"] = info["s2_id"]
        pts["s2_time_utc"] = info.get("s2_time_utc", "")
        pts["s2_cloud_cover"] = info.get("s2_cloud_cover", np.nan)
        pts["timediff_hours"] = info.get("timediff_hours", np.nan)
        all_points.append(pts)

    if not all_points:
        raise SystemExit("No depth points passed min_depth / min_conf filters.")

    points_df = pd.concat(all_points, ignore_index=True)

    print("\n=== IS2-S2 pair extraction ===")
    print("Granules        :", ", ".join(sorted(granule_set)))
    print("Lake files      :", len(lake_files))
    print("Lakes w/ points :", points_df["lake_id"].nunique())
    print("Depth points    :", len(points_df))

    if args.dry_run:
        print("Dry run — no GEE calls. Example rows:")
        print(points_df.head(3).to_string(index=False))
        return

    init_earth_engine(args.gee_project)

    sampled_chunks: list[pd.DataFrame] = []
    for granule, grp in points_df.groupby("granule", sort=True):
        s2_id = str(grp["s2_id"].iloc[0])
        s2_time = str(grp["s2_time_utc"].iloc[0])
        lon = float(grp["lon"].median())
        lat = float(grp["lat"].median())
        print(f"\nSampling {granule} | {s2_id} | {len(grp)} points")
        image = get_s2_image(s2_id, s2_time, lon, lat)
        grp = grp.reset_index(drop=True).copy()
        grp["point_idx"] = np.arange(len(grp), dtype=int)
        sampled = sample_s2_bands(image, grp, args.s2_scale)
        if sampled.empty:
            print("Warning: no S2 samples returned for", granule, file=sys.stderr)
            continue
        sampled["granule"] = granule
        sampled["s2_id"] = s2_id
        sampled["s2_time_utc"] = s2_time
        sampled = merge_point_metadata(sampled, grp)
        drop_cols = [c for c in sampled.columns if c.endswith("_gee")]
        sampled = sampled.drop(columns=drop_cols, errors="ignore")
        sampled_chunks.append(sampled)

    if not sampled_chunks:
        raise SystemExit("GEE sampling returned no rows.")

    out_df = pd.concat(sampled_chunks, ignore_index=True)
    out_df = out_df.drop(columns=["point_idx"], errors="ignore")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    band_cols = [c for c in out_df.columns if c.startswith("S2_")]
    print("\n=== Extraction complete ===")
    print("Output CSV :", out_path)
    print("Rows       :", len(out_df))
    print("S2 bands   :", ", ".join(band_cols))


if __name__ == "__main__":
    main()
