#!/usr/bin/env python3
"""Match ICESat-2 ATL03 granules with nearest Sentinel-2 scenes over Greenland SW.

Uses the public Element84 STAC API (no Google Earth Engine required) to build a
candidate table of temporal matches. This is intended as a planning step before
running detect_lakes.py and make_quicklook_plots.ipynb.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import pandas as pd
import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STAC_URL = 'https://earth-search.aws.element84.com/v1/search'
GRANULE_RE = re.compile(r'ATL03_(\d{8})(\d{6})_')


def parse_granule_datetime(granule_id: str) -> datetime:
    match = GRANULE_RE.search(granule_id)
    if not match:
        raise ValueError(f'Cannot parse datetime from granule id: {granule_id}')
    return datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)


def load_granule_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=['granule', 'geojson', 'description', 'geojson_clip'])
    df['is2_time'] = df['granule'].map(parse_granule_datetime)
    return df


def study_bbox(geojson_path: str) -> list[float]:
    gdf = gpd.read_file(geojson_path)
    minx, miny, maxx, maxy = gdf.total_bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def query_sentinel2(bbox: list[float], start: datetime, end: datetime, cloud_cover: float, limit: int) -> list[dict]:
    payload = {
        'collections': ['sentinel-2-l2a'],
        'bbox': bbox,
        'datetime': f'{start.isoformat().replace("+00:00", "Z")}/{end.isoformat().replace("+00:00", "Z")}',
        'limit': limit,
        'query': {'eo:cloud_cover': {'lt': cloud_cover}},
    }
    response = requests.post(STAC_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json().get('features', [])


def best_match(is2_time: datetime, features: list[dict]) -> dict | None:
    """Pick the most useful Sentinel-2 scene for optical validation.

    Selection priority (best for supraglacial-lake validation):
      1. same UTC day as the ICESat-2 overpass
      2. lower cloud cover
      3. smaller absolute time difference

    Rationale: for lake validation a clear same-day scene is far more useful
    than a temporally-closer but cloudy scene. This matters for pre-dawn
    ICESat-2 overpasses, where the temporally-nearest scene can fall on the
    previous day while the same-day (afternoon) scene is cloud-free.
    """
    if not features:
        return None

    candidates = []
    for feature in features:
        props = feature.get('properties', {})
        s2_time = datetime.fromisoformat(props['datetime'].replace('Z', '+00:00'))
        abs_seconds = abs((s2_time - is2_time).total_seconds())
        cloud = props.get('eo:cloud_cover')
        candidates.append({
            's2_id': feature.get('id'),
            's2_time': s2_time,
            'timediff_hours': abs_seconds / 3600.0,
            'same_day': s2_time.date() == is2_time.date(),
            'cloud_cover': cloud,
            'platform': props.get('platform'),
            '_abs_seconds': abs_seconds,
            '_cloud_sort': cloud if cloud is not None else 999.0,
        })

    # same-day first (True sorts before False via not), then low cloud, then time
    candidates.sort(key=lambda c: (not c['same_day'], c['_cloud_sort'], c['_abs_seconds']))
    best = candidates[0]
    best.pop('_abs_seconds', None)
    best.pop('_cloud_sort', None)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--granule-list', default='granule_lists/GrIS_2022_GRE_2000_SW.csv')
    parser.add_argument('--geojson', default='geojsons/GRE_2000_SW.geojson')
    parser.add_argument('--days-buffer', type=int, default=5, help='Search +/- this many days around each granule')
    parser.add_argument('--cloud-cover-max', type=float, default=50.0)
    parser.add_argument('--limit-per-granule', type=int, default=100)
    parser.add_argument('--max-granules', type=int, default=None, help='Process only the first N granules')
    parser.add_argument('--out-csv', default='granule_lists/GrIS_2022_GRE_2000_SW_is2_s2_matches.csv')
    args = parser.parse_args()

    granule_path = args.granule_list if os.path.isabs(args.granule_list) else os.path.join(REPO_ROOT, args.granule_list)
    geojson_path = args.geojson if os.path.isabs(args.geojson) else os.path.join(REPO_ROOT, args.geojson)
    out_path = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(REPO_ROOT, args.out_csv)

    granules = load_granule_table(granule_path)
    if args.max_granules is not None:
        granules = granules.head(args.max_granules).copy()

    bbox = study_bbox(geojson_path)
    rows = []

    print('Matching', len(granules), 'ICESat-2 granules against Sentinel-2 L2A')
    print('Study bbox:', bbox)

    for idx, row in granules.iterrows():
        is2_time = row['is2_time']
        start = is2_time - timedelta(days=args.days_buffer)
        end = is2_time + timedelta(days=args.days_buffer)
        try:
            features = query_sentinel2(bbox, start, end, args.cloud_cover_max, args.limit_per_granule)
            match = best_match(is2_time, features)
        except requests.RequestException as exc:
            print('STAC query failed for', row['granule'], ':', exc)
            match = None

        rows.append({
            'granule': row['granule'],
            'is2_time_utc': is2_time.isoformat(),
            'search_start_utc': start.isoformat(),
            'search_end_utc': end.isoformat(),
            's2_candidates': len(features) if 'features' in locals() else 0,
            's2_id': None if match is None else match['s2_id'],
            's2_time_utc': None if match is None else match['s2_time'].isoformat(),
            'timediff_hours': None if match is None else round(match['timediff_hours'], 3),
            'same_day': False if match is None else match['same_day'],
            's2_cloud_cover': None if match is None else match['cloud_cover'],
            's2_platform': None if match is None else match['platform'],
        })
        if (idx + 1) % 10 == 0:
            print(' processed', idx + 1, '/', len(granules))

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    same_day = int(out_df['same_day'].fillna(False).sum())
    matched = int(out_df['s2_id'].notna().sum())
    print('\n=== IS2 / Sentinel-2 match table complete ===')
    print('Output CSV     :', out_path)
    print('Granules total :', len(out_df))
    print('With S2 match  :', matched)
    print('Same-day pairs :', same_day)
    if matched:
        print('Median |dt| (h):', round(out_df['timediff_hours'].dropna().median(), 2))


if __name__ == '__main__':
    main()
