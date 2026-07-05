# Greenland SW 2022 workflow scripts

Scripts for **Southwest Greenland (GRE_2000_SW)** lake-depth work and
**ICESat-2 / Sentinel-2** temporal matching in the **2022 melt season**.

## 1. Query ICESat-2 granules

```bash
conda activate eeicelakes-env
python scripts/query_granules_sw_2022.py
```

Outputs:

- `granule_lists/GrIS_2022_GRE_2000_SW.csv` — HTCondor / `detect_lakes.py` input
- `granule_lists/GrIS_2022_GRE_2000_SW_summary.csv` — includes granule sizes

## 2. Match each granule to nearest Sentinel-2 (STAC, no GEE required)

```bash
python scripts/match_is2_sentinel2_sw_2022.py
```

Useful options:

```bash
python scripts/match_is2_sentinel2_sw_2022.py --days-buffer 3 --cloud-cover-max 30
python scripts/match_is2_sentinel2_sw_2022.py --max-granules 10   # quick test
```

Output: `granule_lists/GrIS_2022_GRE_2000_SW_is2_s2_matches.csv`

Columns include `is2_time_utc`, `s2_time_utc`, `timediff_hours`, `same_day`.

## 3. Run lake detection / depth retrieval locally

Requires NASA Earthdata credentials in `ed/edcreds.py`.

```bash
python scripts/run_sw_lake_detection.py --max-jobs 3
python scripts/run_sw_lake_detection.py --start-index 5 --max-jobs 1
```

Equivalent single command:

```bash
python detect_lakes.py \
  --granule ATL03_20220501042951_05971503_007_01.h5 \
  --polygon geojsons/simplified_GRE_2000_SW.geojson
```

## 4. Satellite quicklooks after lakes are detected

Use `make_quicklook_plots.ipynb` (Google Earth Engine) to download RGB
composites and overlay ICESat-2 ground tracks on the matched Sentinel-2
scenes.

## Notes

- **Same-day** in the match table means the S2 acquisition date (UTC) equals
  the ATL03 granule date. Median time offset is still several hours because
  the satellite overpass times differ.
- STAC matching is for **planning**. Final validation imagery should still
  use the project GEE workflow in `make_quicklook_plots.ipynb`.
