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

Defaults (tuned for clear same-day validation imagery):

- `geojsons/simplified_GRE_2000_SW.geojson`
- search window **±3 days**
- scene cloud cover **< 20%**
- STAC **pagination** up to 500 candidates per granule

Useful options:

```bash
# Strict: only keep granules with a same-UTC-day S2 scene
python scripts/match_is2_sentinel2_sw_2022.py --require-same-day

# Quick test on 10 granules
python scripts/match_is2_sentinel2_sw_2022.py --max-granules 10

# Looser search (more candidates, more cross-day fallbacks)
python scripts/match_is2_sentinel2_sw_2022.py --days-buffer 5 --cloud-cover-max 30
```

Output: `granule_lists/GrIS_2022_GRE_2000_SW_is2_s2_matches.csv`

Columns include `is2_time_utc`, `s2_time_utc`, `timediff_hours`, `same_day`.

Filter in pandas for high-quality pairs:

```python
import pandas as pd
df = pd.read_csv("granule_lists/GrIS_2022_GRE_2000_SW_is2_s2_matches.csv")
good = df[df["same_day"] & df["s2_cloud_cover"].lt(10) & df["s2_id"].notna()]
```

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

Use `make_quicklook_sw.py` (recommended) or `make_quicklook_plots.ipynb` (GEE) to
download RGB composites and overlay ICESat-2 ground tracks on Sentinel-2 scenes.

```bash
python make_quicklook_sw.py --gee-project YOUR_GCP_PROJECT_ID --lake lake_xxx.h5
```

See `STEP5_QUICKLOOK.md` for setup.

## 5. Modeling sample selection (IS2-S2 match + lakes)

Build a granule table that flags rows with **good S2 pairing** and **detected lakes**
for IS2–S2 depth extrapolation modeling.

```bash
# Refresh IS2-S2 match table (strict same-day)
python scripts/match_is2_sentinel2_sw_2022.py --require-same-day

# Run detection on granules with good S2 but no lakes yet
python scripts/filter_granules_for_detection.py
python scripts/run_sw_lake_detection.py --granule-list granule_lists/GrIS_2022_GRE_2000_SW_good_s2_todo.csv --max-jobs 3

# Optional: rename lakes and export per-lake stats
python rename_lakes_and_stats.py --execute

# Merge S2 quality + lake counts into modeling candidate table
python scripts/build_modeling_candidates.py
```

Output: `granule_lists/modeling_candidates_SW.csv`

Key columns:

- `good_s2_match` — same UTC day (default), cloud < 10%, |dt| < 20 h
- `n_lakes`, `n_lakes_quality_gt0` — detect_lakes.py results for that granule
- `modeling_ready` — `good_s2_match` and at least one lake with `lake_quality > 0`

Filter high-quality training granules:

```python
import pandas as pd
df = pd.read_csv("granule_lists/modeling_candidates_SW.csv")
train = df[df["modeling_ready"]]
```

## Notes

- **Same-day** in the match table means the S2 acquisition date (UTC) equals
  the ATL03 granule date. Median time offset is still several hours because
  the satellite overpass times differ (pre-dawn ICESat-2 vs afternoon S2).
- Without `--require-same-day`, the script may pick a cross-day scene when no
  same-day candidate exists. Check the `same_day` column or pass
  `--require-same-day` to leave those granules unmatched.
- STAC matching is for **planning**. Final validation imagery should still
  use the project GEE workflow (`make_quicklook_sw.py` or `make_quicklook_plots.ipynb`).
