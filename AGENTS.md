# AGENTS.md

## Cursor Cloud specific instructions

This is a scientific Python codebase (**FLUIDSuRRF-code**) for automatic supraglacial
lake detection and depth retrieval from ICESat-2 photon data. The original
`environment.yml` is a macOS-only (osx-64) conda export and does **not** solve on
Linux; on Cursor Cloud the Python stack is installed with `pip` into a local
virtualenv at `.venv/` from `requirements.txt` (kept cross-platform, unpinned
except where noted). System dependency `python3.12-venv` is required to create the
venv and is pre-installed in the environment snapshot.

Activate the environment before running anything: `source .venv/bin/activate`.

### Components / how to run

- **Core detection pipeline** — `icelakes/` package + `detect_lakes.py`.
  - Batch script designed for HTCondor/OSG (see `README.md`); not a long-running service.
  - Quick sanity check: `python detect_lakes.py --help`.
  - A full run downloads an ICESat-2 ATL03 granule from NSIDC and therefore
    requires **NASA Earthdata credentials** (RSA-encrypted via `icelakes/utilities.encedc`
    and pasted into `class edc` in `icelakes/nsidc.py`; see `ed/edcreds.py`). Without
    credentials the script only prints a credentials warning — this warning is expected
    and harmless for imports/`--help`.

- **simple-image-labeler** — a Streamlit app (git submodule) used for the final manual
  QC step (sorting lake quicklook plots into good/bad/unsure/interesting).
  - Run: `cd simple-image-labeler && streamlit run image_labeler.py --server.headless true --server.port 8501`
  - It reads images from `zzz_testfolder/unsorted` (the `base_dir` hard-coded near the top
    of `image_labeler.py`); create that folder and drop `.jpg`/`.png` files in it first.
  - **Version gotcha:** the app calls `add_keyboard_shortcuts({...})`, an API that only
    exists in `streamlit-shortcuts==0.1.9` (renamed to `add_shortcuts(**kwargs)` in
    `>=1.0.0`). This exact pin is in `requirements.txt` — do not bump it without also
    changing the app code. The `use_column_width` deprecation warnings it prints are harmless.

- **Notebooks / `lakeanalysis/`** — `make_granule_list*.ipynb`, `make_quicklook_plots.ipynb`,
  `basins/make_basins.ipynb`, and `lakeanalysis/S2*.py` require a **Google Earth Engine**
  account + auth (`earthengine authenticate`) for the quicklook/Sentinel-2 parts. These
  external-auth pieces are not exercised by the default setup.

### Notes

- There is no automated test suite and no linter configured in this repo.
- The code was originally written against pandas 1.5 / numpy 1.24; `requirements.txt`
  installs current major versions, so watch for occasional deprecation warnings.
