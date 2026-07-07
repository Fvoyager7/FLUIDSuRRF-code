"""Standard paths for the SW Greenland IS2-S2 modeling workflow."""

from __future__ import annotations

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELING_ROOT = os.path.join(REPO_ROOT, "modeling")

LISTS_DIR = os.path.join(MODELING_ROOT, "lists")
OUT_DIR = os.path.join(MODELING_ROOT, "out")
PAIRS_DIR = os.path.join(OUT_DIR, "pairs")
MODELS_DIR = os.path.join(OUT_DIR, "models")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")

# Shared upstream outputs (detect_lakes / match pipeline at repo root)
LAKE_DATA_DIR = os.path.join(REPO_ROOT, "detection_out_data")
MATCH_CSV = os.path.join(REPO_ROOT, "granule_lists", "GrIS_2022_GRE_2000_SW_is2_s2_matches.csv")

CANDIDATES_CSV = os.path.join(LISTS_DIR, "modeling_candidates_SW.csv")
PAIRS_CSV = os.path.join(PAIRS_DIR, "is2_s2_pairs_SW.csv")
METRICS_JSON = os.path.join(MODELS_DIR, "depth_model_metrics.json")
CV_PREDICTIONS_CSV = os.path.join(MODELS_DIR, "depth_cv_predictions.csv")
CV_SCATTER_PNG = os.path.join(FIGURES_DIR, "depth_model_cv_scatter.png")
FIG_MAIN_PNG = os.path.join(FIGURES_DIR, "fig_modeling_main_SW.png")
FIG_MAIN_PDF = os.path.join(FIGURES_DIR, "fig_modeling_main_SW.pdf")
PIPELINE_SUMMARY_CSV = os.path.join(MODELS_DIR, "depth_pipeline_summary.csv")
RMSE_BY_DEPTH_CSV = os.path.join(MODELS_DIR, "depth_rmse_by_depth_bin.csv")
PHOTON_PAIRS_CSV = os.path.join(PAIRS_DIR, "is2_s2_pairs_SW_photon.csv")
