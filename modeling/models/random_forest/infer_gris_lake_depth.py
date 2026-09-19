#!/usr/bin/env python3
"""Infer Greenland ice-sheet (GrIS) lake depth from a trained random forest.

Run from the ``modeling/is2-s2-depth-*`` repo root (this tree is
``modeling/is2-s2-depth-sw-2022``; a sibling GrIS 2024 workflow uses the same
``modeling/`` layout):

  python modeling/models/random_forest/infer_gris_lake_depth.py
  python modeling/models/random_forest/infer_gris_lake_depth.py --limit 100
  python modeling/models/random_forest/infer_gris_lake_depth.py --dry-run

Loads (fail clearly if missing):

  modeling/out/models/random_forest/random_forest.joblib
  modeling/out/models/random_forest/metrics.json

``metrics.json`` must list the 12 training features (top-level ``features`` or
``random_forest.features``). The SW 2022 trainer writes a nested metrics file at
``modeling/out/models/depth_model_metrics.json`` — pass it with ``--metrics``.

The joblib is typically pickled with scikit-learn 1.9.x. Load it from a matching
environment when possible; a version mismatch can fail or change predictions.

Default feature table is ``modeling/out/pairs/is2_s2_pairs_gris_2024_train7744.csv``
when that file exists, otherwise ``modeling/paths.py`` ``PAIRS_CSV``
(``is2_s2_pairs_SW.csv`` on this branch). Relative CLI paths are resolved from
the ``modeling/`` root, then the repo root.

Uncertainty columns
-------------------
depth_pred_m         RF mean prediction (``estimator.predict``)
depth_std_trees_m    std across trees (``estimator.estimators_``; ddof=0)
depth_p10_m          10th percentile across trees
depth_p90_m          90th percentile across trees
resid_rmse_bin_m     CV residual RMSE in the predicted-depth 1 m bin
                     (from ``diagnostics/cv_predictions.csv`` if present; else NaN)
flag_ood             1 if any feature is NaN or outside training min/max

Optional CV residuals are read from
``modeling/out/models/random_forest/diagnostics/cv_predictions.csv`` (cols
``depth_m``, ``pred_random_forest``) or the SW file
``modeling/out/models/depth_cv_predictions.csv``. If neither exists,
``resid_rmse_bin_m`` is left NaN and a note is printed.

Locked training protocol (documented only; this script does not retrain):
  lake quality Q>=1, cloud cover <30%, depth 0.1-15 m, conf>=0.3,
  visual ice-floe lakes excluded; 12 features
  (S2_B2_refl … S2_B8A_refl, S2_B11_refl, S2_B12_refl, NDWIice, NDWI);
  GroupKFold by lake_id. Report generalization from ``metrics.json`` ``cv``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODELING_ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modeling.paths import (  # noqa: E402
    CV_PREDICTIONS_CSV,
    PAIRS_CSV,
    PAIRS_CSV_GRIS_2024,
    RF_CV_PREDICTIONS_CSV,
    RF_INFERENCE_CSV,
    RF_METRICS_JSON,
    RF_MODEL_JOBLIB,
)

PRED_COL_CANDIDATES = ("pred_random_forest", "depth_pred_m", "pred")
UNCERTAINTY_COLS = (
    "depth_pred_m",
    "depth_std_trees_m",
    "depth_p10_m",
    "depth_p90_m",
    "resid_rmse_bin_m",
    "flag_ood",
)
BIN_WIDTH_M = 1.0
# Inclusive training min/max plus slack so CSV float round-trip is not OOD.
OOD_RANGE_SLACK = 1e-9


def resolve_path(path: str) -> str:
    """Resolve a CLI path: absolute as-is, else modeling root, else repo root."""
    if os.path.isabs(path):
        return path
    via_modeling = os.path.join(MODELING_ROOT, path)
    via_repo = os.path.join(REPO_ROOT, path)
    if os.path.exists(via_modeling):
        return via_modeling
    if os.path.exists(via_repo):
        return via_repo
    return via_modeling


def default_pairs_csv() -> str:
    if os.path.isfile(PAIRS_CSV_GRIS_2024):
        return PAIRS_CSV_GRIS_2024
    return PAIRS_CSV


def require_file(path: str, kind: str) -> str:
    if not os.path.isfile(path):
        raise SystemExit(
            f"{kind} not found: {path}\n"
            "These artifacts are gitignored under modeling/out/ and live on the "
            "researcher machine. Pass --model / --metrics / --pairs-csv if they "
            "are stored elsewhere."
        )
    return path


def load_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)
    if not isinstance(metrics, dict):
        raise SystemExit(f"metrics JSON must be an object: {path}")
    return metrics


def features_from_metrics(metrics: dict) -> list[str]:
    raw = metrics.get("features")
    if raw is None:
        rf = metrics.get("random_forest")
        if isinstance(rf, dict):
            raw = rf.get("features")
    if not isinstance(raw, list) or not raw:
        raise SystemExit(
            "metrics.json has no feature list. Expected top-level 'features' "
            "or 'random_forest.features'."
        )
    return [str(name) for name in raw]


def _range_from_mapping(block: object, features: list[str]) -> tuple[pd.Series, pd.Series] | None:
    if not isinstance(block, dict) or not block:
        return None
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for feat in features:
        item = block.get(feat)
        if isinstance(item, dict) and "min" in item and "max" in item:
            mins[feat] = float(item["min"])
            maxs[feat] = float(item["max"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            mins[feat] = float(item[0])
            maxs[feat] = float(item[1])
    if len(mins) == len(features):
        return pd.Series(mins), pd.Series(maxs)
    return None


def feature_bounds(
    metrics: dict, table: pd.DataFrame, features: list[str]
) -> tuple[pd.Series, pd.Series, str]:
    """Training feature min/max from metrics if present, else the input table."""
    candidates: list[object] = [
        metrics.get("feature_range"),
        metrics.get("feature_ranges"),
        metrics.get("feature_min_max"),
    ]
    proto = metrics.get("sample_protocol")
    if isinstance(proto, dict):
        candidates.extend(
            [proto.get("feature_range"), proto.get("feature_ranges"), proto.get("feature_min_max")]
        )
    for block in candidates:
        parsed = _range_from_mapping(block, features)
        if parsed is not None:
            return parsed[0], parsed[1], "metrics.json feature_range"

    feat_min = metrics.get("feature_min")
    feat_max = metrics.get("feature_max")
    if isinstance(feat_min, dict) and isinstance(feat_max, dict):
        if all(f in feat_min and f in feat_max for f in features):
            return (
                pd.Series({f: float(feat_min[f]) for f in features}),
                pd.Series({f: float(feat_max[f]) for f in features}),
                "metrics.json feature_min/feature_max",
            )

    missing = [f for f in features if f not in table.columns]
    if missing:
        raise SystemExit(
            "Cannot build OOD feature ranges; columns missing from the input "
            f"table and metrics: {', '.join(missing)}"
        )
    numeric = table[features].apply(pd.to_numeric, errors="coerce")
    return numeric.min(), numeric.max(), "input table min/max"


def find_cv_predictions(model_path: str, metrics_path: str) -> str | None:
    candidates = (
        os.path.join(os.path.dirname(model_path), "diagnostics", "cv_predictions.csv"),
        RF_CV_PREDICTIONS_CSV,
        os.path.join(os.path.dirname(metrics_path), "diagnostics", "cv_predictions.csv"),
        os.path.join(os.path.dirname(metrics_path), "depth_cv_predictions.csv"),
        CV_PREDICTIONS_CSV,
    )
    seen: set[str] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def _pred_column(df: pd.DataFrame) -> str:
    for name in PRED_COL_CANDIDATES:
        if name in df.columns:
            return name
    raise ValueError(
        "CV predictions need a prediction column "
        f"({', '.join(PRED_COL_CANDIDATES)})"
    )


def resid_rmse_bins(cv_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """1 m predicted-depth bins -> residual RMSE. Returns (edges, rmse)."""
    pred_col = _pred_column(cv_df)
    if "depth_m" not in cv_df.columns:
        raise ValueError("CV predictions need a depth_m column")
    pred = pd.to_numeric(cv_df[pred_col], errors="coerce").to_numpy(dtype=float)
    true = pd.to_numeric(cv_df["depth_m"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(pred) & np.isfinite(true)
    pred = pred[ok]
    true = true[ok]
    if pred.size == 0:
        raise ValueError("CV predictions have no finite depth_m / pred rows")

    lo = 0.0
    hi = max(float(np.ceil(pred.max() / BIN_WIDTH_M) * BIN_WIDTH_M), BIN_WIDTH_M)
    edges = np.arange(lo, hi + BIN_WIDTH_M, BIN_WIDTH_M, dtype=float)
    resid_sq = (pred - true) ** 2
    rmse = np.full(len(edges) - 1, np.nan, dtype=float)
    for i in range(len(rmse)):
        mask = (pred >= edges[i]) & (pred < edges[i + 1])
        if i == len(rmse) - 1:
            mask = (pred >= edges[i]) & (pred <= edges[i + 1])
        if np.any(mask):
            rmse[i] = float(np.sqrt(np.mean(resid_sq[mask])))
    return edges, rmse


def lookup_bin_rmse(pred: np.ndarray, edges: np.ndarray, rmse: np.ndarray) -> np.ndarray:
    out = np.full(pred.shape, np.nan, dtype=float)
    finite = np.isfinite(pred)
    if not np.any(finite):
        return out
    idx = np.digitize(pred[finite], edges, right=False) - 1
    idx = np.clip(idx, 0, len(rmse) - 1)
    # Outside the CV depth span stays NaN (do not clip to an edge bin).
    inside = (pred[finite] >= edges[0]) & (pred[finite] <= edges[-1])
    vals = rmse[idx]
    vals = np.where(inside, vals, np.nan)
    out[finite] = vals
    return out


def unwrap_forest(obj):
    """Return (predict_estimator, forest_with_estimators_, transform_fn)."""
    if isinstance(obj, dict):
        for key in ("random_forest", "model", "estimator"):
            if key in obj:
                return unwrap_forest(obj[key])
        raise SystemExit(
            "joblib dict has no random_forest/model/estimator key. "
            "Save the fitted RandomForestRegressor (or a Pipeline ending in one)."
        )

    if hasattr(obj, "estimators_"):
        return obj, obj, (lambda x: x)

    if hasattr(obj, "named_steps") and obj.steps:
        last = obj.steps[-1][1]
        if hasattr(last, "estimators_"):
            prefix = obj[:-1]

            def _transform(x, _prefix=prefix):
                return _prefix.transform(x) if len(_prefix) else x

            return obj, last, _transform

    raise SystemExit(
        "Loaded object is not a RandomForestRegressor or a Pipeline ending in one "
        f"(got {type(obj).__name__})."
    )


def tree_matrix(forest, Xt: np.ndarray) -> np.ndarray:
    """(n_rows, n_trees) predictions from forest.estimators_."""
    if not getattr(forest, "estimators_", None):
        raise SystemExit("Random forest has no estimators_ (model not fitted?)")
    cols = [np.asarray(est.predict(Xt), dtype=float) for est in forest.estimators_]
    return np.column_stack(cols)


def add_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """NDWIice (B2,B4) and NDWI (B3,B8); same formulas as train_depth_model.py."""
    out = df.copy()
    rb = out["S2_B2_refl"].astype(float)
    rr = out["S2_B4_refl"].astype(float)
    rg = out["S2_B3_refl"].astype(float)
    rnir = out["S2_B8_refl"].astype(float)
    out["NDWIice"] = (rb - rr) / (rb + rr + 1e-8)
    out["NDWI"] = (rg - rnir) / (rg + rnir + 1e-8)
    return out


def prepare_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    needed_idx = [f for f in features if f in ("NDWIice", "NDWI") and f not in out.columns]
    if needed_idx:
        band_needed = ("S2_B2_refl", "S2_B3_refl", "S2_B4_refl", "S2_B8_refl")
        if all(c in out.columns for c in band_needed):
            out = add_spectral_indices(out)
    missing = [f for f in features if f not in out.columns]
    if missing:
        raise SystemExit(f"Feature columns missing from pairs CSV: {', '.join(missing)}")
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def flag_ood_rows(X: pd.DataFrame, feat_min: pd.Series, feat_max: pd.Series) -> np.ndarray:
    slack = OOD_RANGE_SLACK
    below = X.lt(feat_min - slack)
    above = X.gt(feat_max + slack)
    nan = X.isna()
    return (below | above | nan).any(axis=1).to_numpy(dtype=int)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pairs-csv",
        default=None,
        help="Feature table CSV (default: GrIS train table if present, else PAIRS_CSV)",
    )
    parser.add_argument("--model", default=RF_MODEL_JOBLIB, help="Path to random_forest.joblib")
    parser.add_argument("--metrics", default=RF_METRICS_JSON, help="Path to metrics.json")
    parser.add_argument("--out", default=RF_INFERENCE_CSV, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N rows (smoke)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan; do not write")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    pairs_path = resolve_path(args.pairs_csv or default_pairs_csv())
    model_path = resolve_path(args.model)
    metrics_path = resolve_path(args.metrics)
    out_path = resolve_path(args.out)

    require_file(model_path, "Trained RF joblib")
    require_file(metrics_path, "Metrics JSON")
    require_file(pairs_path, "Pairs / feature CSV")

    metrics = load_metrics(metrics_path)
    features = features_from_metrics(metrics)
    cv_path = find_cv_predictions(model_path, metrics_path)

    raw = pd.read_csv(pairs_path)
    n_in = len(raw)
    df = prepare_features(raw, features)
    feat_min, feat_max, range_source = feature_bounds(metrics, df, features)
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be >= 0")
        df = df.iloc[: args.limit].copy()

    print("=== GrIS lake-depth RF inference ===")
    print("Pairs CSV     :", pairs_path)
    print("Model         :", model_path)
    print("Metrics       :", metrics_path)
    print("Features      :", ", ".join(features))
    print("Rows in file  :", n_in)
    print("Rows to score :", len(df))
    print("OOD ranges    :", range_source)
    if cv_path:
        print("CV residuals  :", cv_path)
    else:
        print(
            "CV residuals  : not found "
            f"(looked for {RF_CV_PREDICTIONS_CSV} and {CV_PREDICTIONS_CSV}); "
            "resid_rmse_bin_m will be NaN"
        )
    print("Output CSV    :", out_path)

    if args.dry_run:
        print("Dry run — no predictions written.")
        return

    import joblib  # local import so --help stays light if joblib is unused

    estimator = joblib.load(model_path)
    predict_est, forest, transform = unwrap_forest(estimator)

    X = df[features]
    valid = X.notna().all(axis=1).to_numpy()
    depth_pred = np.full(len(df), np.nan, dtype=float)
    depth_std = np.full(len(df), np.nan, dtype=float)
    depth_p10 = np.full(len(df), np.nan, dtype=float)
    depth_p90 = np.full(len(df), np.nan, dtype=float)

    if np.any(valid):
        Xv = X.loc[valid]
        depth_pred[valid] = np.asarray(predict_est.predict(Xv), dtype=float)
        Xt = np.asarray(transform(Xv), dtype=float)
        trees = tree_matrix(forest, Xt)
        depth_std[valid] = trees.std(axis=1, ddof=0)
        depth_p10[valid] = np.percentile(trees, 10, axis=1)
        depth_p90[valid] = np.percentile(trees, 90, axis=1)

    resid = np.full(len(df), np.nan, dtype=float)
    if cv_path:
        try:
            edges, rmse = resid_rmse_bins(pd.read_csv(cv_path))
            resid = lookup_bin_rmse(depth_pred, edges, rmse)
        except (ValueError, OSError) as exc:
            print(f"Note: could not build CV residual RMSE bins ({exc}); resid_rmse_bin_m is NaN")

    ood = flag_ood_rows(X, feat_min[features], feat_max[features])

    out = df.copy()
    out["depth_pred_m"] = depth_pred
    out["depth_std_trees_m"] = depth_std
    out["depth_p10_m"] = depth_p10
    out["depth_p90_m"] = depth_p90
    out["resid_rmse_bin_m"] = resid
    out["flag_ood"] = ood

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    n_ood = int(ood.sum())
    n_pred = int(np.isfinite(depth_pred).sum())
    print("Wrote         :", out_path)
    print("Predicted     :", n_pred, "/", len(out))
    print("flag_ood = 1  :", n_ood)
    print("Columns added :", ", ".join(UNCERTAINTY_COLS))


if __name__ == "__main__":
    main()
