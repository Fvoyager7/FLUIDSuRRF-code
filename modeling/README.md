# IS2–S2 depth modeling (SW Greenland 2022)

All **modeling-phase** files live under this folder. Upstream detect/match
outputs stay at the repo root but are referenced here.

## Layout

```
modeling/
├── README.md                 ← you are here
├── paths.py                  ← shared path constants
├── lists/
│   └── modeling_candidates_SW.csv    # granule filter table
├── scripts/
│   ├── extract_is2_s2_pairs.py       # IS2 depth + S2 band extraction
│   ├── aggregate_pairs_to_pixels.py  # 10 m pixel aggregation
│   ├── train_depth_model.py          # EFM / ridge / RF + CV
│   ├── plot_modeling_figure.py       # 5-panel paper main figure
│   └── summarize_modeling_stats.py     # pipeline + depth-binned RMSE tables
├── out/                      # gitignored outputs
│   ├── pairs/is2_s2_pairs_SW.csv
│   ├── models/
│   └── figures/
└── data/
    └── README.md             # pointer to lake h5 inputs
```

## Upstream inputs (repo root)

| What | Path |
|------|------|
| Lake h5 (IS2 depth labels) | `../detection_out_data/lake_*.h5` |
| IS2–S2 match table | `../granule_lists/GrIS_2022_GRE_2000_SW_is2_s2_matches.csv` |

Lake h5 files are **not copied** here — they are shared with `detect_lakes.py`
and `make_quicklook_sw.py`. The extract script reads them via `modeling/paths.py`.

## Quick start

```bash
conda activate eeicelakes-env
cd FLUIDSuRRF-code-main

# Refresh candidate table (writes to modeling/lists/)
python scripts/build_modeling_candidates.py

# Preview extraction
python modeling/scripts/extract_is2_s2_pairs.py --dry-run

# Full GEE extraction (photon-level backup kept separately)
python modeling/scripts/extract_is2_s2_pairs.py --gee-project ee-wenlinshen777

# Aggregate to 10 m S2 pixels (overwrites is2_s2_pairs_SW.csv)
python modeling/scripts/aggregate_pairs_to_pixels.py
```

Output:

- `modeling/out/pairs/is2_s2_pairs_SW_photon.csv` — photon-level (backup)
- `modeling/out/pairs/is2_s2_pairs_SW.csv` — **10 m pixel-level** (training table)

## Train depth models

```bash
python modeling/scripts/train_depth_model.py
```

Models compared (GroupKFold by `lake_id`):

1. **Green-band empirical (EFM)** — `depth = A * exp(-B * S2_B3_refl) + C` (Lutz et al. 2024 SW form; B3 only)
2. **Ridge regression** — 10 S2 reflectance bands + NDWIice + NDWI
3. **Random forest** — same features as ridge

Indices:

- `NDWIice = (B2 - B4) / (B2 + B4)`
- `NDWI = (B3 - B8) / (B3 + B8)`

Outputs: `modeling/out/models/depth_model_metrics.json` (includes `cv` and `cv_train` per model),
`depth_cv_predictions.csv` (OOF and in-sample train predictions), `depth_model_cv_scatter.png`

## Paper main figure (5 panels)

Re-run training first so metrics include `cv_train` (in-sample train-fold predictions):

```bash
python modeling/scripts/train_depth_model.py
python modeling/scripts/plot_modeling_figure.py
```

Outputs (300 dpi PNG + vector PDF):

- `modeling/out/figures/fig_modeling_main_SW.png`
- `modeling/out/figures/fig_modeling_main_SW.pdf`

Panels **a–e**:

| Panel | Content |
|-------|---------|
| **a** | `depth_m` histogram (training sample distribution) |
| **b** | Three-model CV: R² bars (train/test) + RMSE lines (train/test) |
| **c** | Random-forest 1:1 CV scatter, colored by lake with per-lake legend |
| **d** | RF feature importance |
| **e** | RF residual diagnostic (full width) |

All CV metrics are read from `depth_model_metrics.json` (`cv` = held-out fold, `cv_train` = in-sample train-fold average).

## Summary tables (pipeline + RMSE by depth)

After training:

```bash
python modeling/scripts/summarize_modeling_stats.py
```

Outputs:

| File | Content |
|------|---------|
| `modeling/out/models/depth_pipeline_summary.csv` | Sample counts at each processing stage |
| `modeling/out/models/depth_rmse_by_depth_bin.csv` | CV RMSE (m) by 1 m depth bins for 3 models |
| `modeling/out/models/depth_pipeline_summary.md` | Markdown table for methods / SI |
| `modeling/out/models/depth_rmse_by_depth_bin.md` | Depth-binned RMSE with best model per row |

Depth bins: 1 m width from 0 m up to max observed depth (currently ~9 m), plus Overall.

**Caption draft:** Pixel-level training samples were built by aggregating ICESat-2 depth points to 10 m Sentinel-2 grid cells. Models were evaluated with 5-fold GroupKFold cross-validation grouped by lake ID. Green-band depth was modeled as z = A·exp(−B·x) + C (Lutz et al., 2024); ridge regression and random forest used 10 S2 reflectance bands plus NDWIice and NDWI.

## Branch

Develop on `modeling/is2-s2-depth-sw-2022`.
