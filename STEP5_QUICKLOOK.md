# Step 5: GEE Sentinel-2 + ICESat-2 quicklook plots

After step 4 (`detect_lakes.py`) you have `detection_out_data/lake_*.h5`.
Step 5 adds **Sentinel-2 optical imagery** from Google Earth Engine next to each ICESat-2 profile.

## One-time setup

```powershell
conda activate eeicelakes-env
cd "D:\7Miscellaneous Notes\FLUIDSuRRF-code-main"

pip install earthengine-api rasterio ipython
earthengine authenticate
```

- Register at https://earthengine.google.com/ and enable Earth Engine for your Google account.
- If `ee.Initialize` asks for a **Cloud project**, note the project ID (e.g. `ee-myname`).

## Run (recommended script)

**Test 3 lakes first** (~5–15 min, depends on GEE):

```powershell
$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""
python make_quicklook_sw.py --limit 3 --gee-project YOUR_GCP_PROJECT_ID
```

**All 110 SW lakes** (may take several hours):

```powershell
python make_quicklook_sw.py --gee-project YOUR_GCP_PROJECT_ID
```

Or set project once:

```powershell
$env:EE_PROJECT="YOUR_GCP_PROJECT_ID"
python make_quicklook_sw.py
```

## Outputs

| Directory | Content |
|-----------|---------|
| `detection_out_quicklook/` | Combined `.jpg` (ICESat-2 left + S2 right) |
| `quicklook_imagery/` | Cached S2 GeoTIFF mosaics (`.tif`) |

## Alternative: Jupyter notebook

Open `make_quicklook_plots.ipynb` — cell 1 now auto-detects PROJ/GDAL on Windows.
For your SW run, set:

- `searchdir = 'detection_out_data/'`
- `outdir = 'detection_out_quicklook/'`

Or use `make_quicklook_sw.py` instead (simpler).

## Optional step 6: manual QC

Browse quicklooks with `simple-image-labeler/` and remove obvious false positives.
