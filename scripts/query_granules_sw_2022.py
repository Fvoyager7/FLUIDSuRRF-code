#!/usr/bin/env python3
"""Query ICESat-2 ATL03 granules over Greenland SW for the 2022 melt season."""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo root without installing icelakes as a package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from icelakes.nsidc import make_granule_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geojson', default='simplified_GRE_2000_SW.geojson',
                        help='GeoJSON filename inside geojsons/')
    parser.add_argument('--start-date', default='2022-05-01')
    parser.add_argument('--end-date', default='2022-09-30')
    parser.add_argument('--icesheet', default='GrIS')
    parser.add_argument('--meltseason', default='2022')
    parser.add_argument('--out-csv', default='granule_lists/GrIS_2022_GRE_2000_SW.csv')
    args = parser.parse_args()

    os.makedirs(os.path.join(REPO_ROOT, 'granule_lists'), exist_ok=True)
    out_path = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(REPO_ROOT, args.out_csv)

    df = make_granule_list(
        args.geojson,
        args.start_date,
        args.end_date,
        args.icesheet,
        args.meltseason,
        out_path,
        geojson_dir_local=os.path.join(REPO_ROOT, 'geojsons/'),
        geojson_dir_remote=os.path.join('geojsons/'),
        return_df=True,
    )

    # HTCondor submit format: granule, polygon, description, polygon_full
    df['geojson_clip'] = df['geojson'].str.replace('simplified_', '', regex=False)
    out_df = df[['granule', 'geojson', 'description', 'geojson_clip']].copy()
    out_df.to_csv(out_path, header=False, index=False)

    summary_path = out_path.replace('.csv', '_summary.csv')
    df.to_csv(summary_path, index=False)

    print('\n=== Greenland SW granule query complete ===')
    print('Region geojson :', args.geojson)
    print('Date range     :', args.start_date, 'to', args.end_date)
    print('Granule count  :', len(df))
    if len(df):
        print('First granule  :', df.granule.iloc[0])
        print('Last granule   :', df.granule.iloc[-1])
        print('Total size (GB):', round(df.size_mb.sum() / 1000, 2))
    print('Submit list    :', out_path)
    print('Summary table  :', summary_path)


if __name__ == '__main__':
    main()
