#!/usr/bin/env python3
"""Train lake-depth models: green-band empirical, ridge, and random forest.

Features: Sentinel-2 surface reflectance + NDWIice + NDWI.

  NDWIice = (Rb - Rr) / (Rb + Rr)     Rb=B2, Rr=B4
  NDWI    = (Rg - RNIR) / (Rg + RNIR) Rg=B3, RNIR=B8

Example:
  python modeling/scripts/train_depth_model.py
  python modeling/scripts/train_depth_model.py --min-depth 0.5 --max-depth 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODELING_ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modeling.paths import FIGURES_DIR, MODELS_DIR, PAIRS_CSV  # noqa: E402

REFL_BANDS = [
    "S2_B2_refl",
    "S2_B3_refl",
    "S2_B4_refl",
    "S2_B5_refl",
    "S2_B6_refl",
    "S2_B7_refl",
    "S2_B8_refl",
    "S2_B8A_refl",
    "S2_B11_refl",
    "S2_B12_refl",
]
INDEX_FEATURES = ["NDWIice", "NDWI"]
ML_FEATURES = REFL_BANDS + INDEX_FEATURES
GREEN_FEATURE = "S2_B3_refl"  # Sentinel-2 L2A green band only for EFM

# Lutz et al. (2024) The Cryosphere SW Greenland published coefficients (initial guess)
LUTZ_SW_A, LUTZ_SW_B, LUTZ_SW_C = 18.8999, 5.9037, 0.3237


def add_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute NDWIice (B2,B4) and NDWI (B3,B8)."""
    out = df.copy()
    rb = out["S2_B2_refl"].astype(float)
    rr = out["S2_B4_refl"].astype(float)
    rg = out["S2_B3_refl"].astype(float)
    rnir = out["S2_B8_refl"].astype(float)

    out["NDWIice"] = (rb - rr) / (rb + rr + 1e-8)
    out["NDWI"] = (rg - rnir) / (rg + rnir + 1e-8)
    return out


def filter_training_rows(
    df: pd.DataFrame,
    min_depth: float,
    max_depth: float,
    min_conf: float,
    min_photons: int,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out["depth_m"].between(min_depth, max_depth)]
    out = out[out["conf"] >= min_conf]
    if "n_photons" in out.columns:
        out = out[out["n_photons"] >= min_photons]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["depth_m"] + ML_FEATURES)
    return out.reset_index(drop=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def green_exponential(rg: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """depth = A*exp(-B*Rg) + C  (Lutz et al. 2024 TC Eq. 3 form)."""
    return a * np.exp(-b * rg) + c


class GreenBandExponentialModel:
    """SW Greenland-style green-band empirical: z = A*exp(-B*Rg) + C."""

    def __init__(self, p0: tuple[float, float, float] = (LUTZ_SW_A, LUTZ_SW_B, LUTZ_SW_C)):
        self.p0 = p0
        self.A_: float | None = None
        self.B_: float | None = None
        self.C_: float | None = None

    def fit(self, rg: np.ndarray, depth: np.ndarray) -> "GreenBandExponentialModel":
        from scipy.optimize import curve_fit

        rg = rg.astype(float)
        depth = depth.astype(float)
        try:
            popt, _ = curve_fit(
                green_exponential,
                rg,
                depth,
                p0=self.p0,
                maxfev=20000,
                bounds=([0.0, 0.0, 0.0], [500.0, 50.0, 10.0]),
            )
            self.A_, self.B_, self.C_ = float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            self.A_, self.B_, self.C_ = self.p0
        return self

    def predict(self, rg: np.ndarray) -> np.ndarray:
        if self.A_ is None:
            raise RuntimeError("Model not fitted")
        return green_exponential(rg.astype(float), self.A_, self.B_, self.C_)


def cv_green_empirical(
    df: pd.DataFrame, groups: np.ndarray, n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    preds_test = np.full(len(df), np.nan)
    preds_train_sum = np.zeros(len(df))
    preds_train_count = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    rg = df[GREEN_FEATURE].to_numpy()
    y = df["depth_m"].to_numpy()

    for train_idx, test_idx in gkf.split(df, y, groups):
        model = GreenBandExponentialModel().fit(rg[train_idx], y[train_idx])
        preds_test[test_idx] = model.predict(rg[test_idx])
        train_pred = model.predict(rg[train_idx])
        preds_train_sum[train_idx] += train_pred
        preds_train_count[train_idx] += 1

    preds_train = preds_train_sum / np.maximum(preds_train_count, 1)
    return preds_test, preds_train


def cv_sklearn_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    preds_test = np.full(len(y), np.nan)
    preds_train_sum = np.zeros(len(y))
    preds_train_count = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, test_idx in gkf.split(X, y, groups):
        fitted = clone(model)
        fitted.fit(X[train_idx], y[train_idx])
        preds_test[test_idx] = fitted.predict(X[test_idx])
        train_pred = fitted.predict(X[train_idx])
        preds_train_sum[train_idx] += train_pred
        preds_train_count[train_idx] += 1

    preds_train = preds_train_sum / np.maximum(preds_train_count, 1)
    return preds_test, preds_train


def fit_final_models(df: pd.DataFrame) -> dict:
    X = df[ML_FEATURES].to_numpy(dtype=float)
    y = df["depth_m"].to_numpy(dtype=float)
    rg = df[GREEN_FEATURE].to_numpy(dtype=float)

    green = GreenBandExponentialModel().fit(rg, y)

    ridge = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    ridge.fit(X, y)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=0,
        n_jobs=1,
    )
    rf.fit(X, y)

    return {"green_empirical": green, "ridge": ridge, "random_forest": rf}


def save_scatter_plot(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    titles = {
        "green_empirical": "Green exp. EFM (B3 only)",
        "ridge": "Ridge (12 features)",
        "random_forest": "Random forest (12 features)",
    }
    lim_max = max(float(np.max(y_true)), max(float(np.max(p)) for p in preds.values()))
    lim_max = max(lim_max, 1.0)

    for ax, key in zip(axes, titles):
        p = preds[key]
        ax.scatter(y_true, p, s=8, alpha=0.35, edgecolors="none")
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=1)
        m = metric_dict(y_true, p)
        ax.set_title(f"{titles[key]}\nRMSE={m['rmse']:.2f} m, R2={m['r2']:.3f}")
        ax.set_xlabel("IS2 depth (m)")
        ax.set_ylabel("Predicted depth (m)")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", default=PAIRS_CSV)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--min-photons", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits by lake_id")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    pairs_path = args.pairs_csv if os.path.isabs(args.pairs_csv) else os.path.join(REPO_ROOT, args.pairs_csv)
    if not os.path.isfile(pairs_path):
        raise SystemExit(f"Pairs CSV not found: {pairs_path}")

    raw = pd.read_csv(pairs_path)
    raw = add_spectral_indices(raw)
    df = filter_training_rows(raw, args.min_depth, args.max_depth, args.min_conf, args.min_photons)
    if len(df) < args.n_splits:
        raise SystemExit(f"Not enough rows ({len(df)}) for {args.n_splits}-fold group CV")

    groups = df["lake_id"].astype(str).to_numpy()
    n_lakes = df["lake_id"].nunique()
    if n_lakes < args.n_splits:
        print(f"Warning: only {n_lakes} lakes; reducing CV folds to {n_lakes}")
        args.n_splits = n_lakes

    X = df[ML_FEATURES].to_numpy(dtype=float)
    y = df["depth_m"].to_numpy(dtype=float)

    ridge_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=args.ridge_alpha)),
        ]
    )
    rf_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=0,
        n_jobs=1,
    )

    preds_test: dict[str, np.ndarray] = {}
    preds_train: dict[str, np.ndarray] = {}
    for name, result in {
        "green_empirical": cv_green_empirical(df, groups, args.n_splits),
        "ridge": cv_sklearn_model(ridge_model, X, y, groups, args.n_splits),
        "random_forest": cv_sklearn_model(rf_model, X, y, groups, args.n_splits),
    }.items():
        preds_test[name], preds_train[name] = result

    metrics_test = {name: metric_dict(y, pred) for name, pred in preds_test.items()}
    metrics_train = {name: metric_dict(y, pred) for name, pred in preds_train.items()}

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Fit on all data for exportable coefficients / importances
    final = fit_final_models(df)
    green = final["green_empirical"]
    ridge_fitted = final["ridge"]
    rf_fitted = final["random_forest"]

    summary = {
        "n_pixels": len(df),
        "n_lakes": int(n_lakes),
        "green_empirical": {
            "feature": GREEN_FEATURE,
            "formula": "depth = A * exp(-B * S2_B3_refl) + C",
            "reference_lutz_2024_sw": {
                "A": LUTZ_SW_A,
                "B": LUTZ_SW_B,
                "C": LUTZ_SW_C,
                "citation": "Lutz et al. (2024) The Cryosphere 18, 5431-5454, Eq. (3)",
            },
            "fitted_all_data": {"A": green.A_, "B": green.B_, "C": green.C_},
            "cv": metrics_test["green_empirical"],
            "cv_train": metrics_train["green_empirical"],
        },
        "ridge": {
            "features": ML_FEATURES,
            "alpha": args.ridge_alpha,
            "cv": metrics_test["ridge"],
            "cv_train": metrics_train["ridge"],
            "coefficients": dict(zip(ML_FEATURES, ridge_fitted.named_steps["model"].coef_.tolist())),
        },
        "random_forest": {
            "features": ML_FEATURES,
            "cv": metrics_test["random_forest"],
            "cv_train": metrics_train["random_forest"],
            "feature_importances": dict(
                zip(ML_FEATURES, rf_fitted.feature_importances_.tolist())
            ),
        },
        "indices": {
            "NDWIice": "(S2_B2_refl - S2_B4_refl) / (S2_B2_refl + S2_B4_refl)",
            "NDWI": "(S2_B3_refl - S2_B8_refl) / (S2_B3_refl + S2_B8_refl)",
        },
    }

    metrics_path = os.path.join(MODELS_DIR, "depth_model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pred_df = df[["pixel_id", "lake_id", "granule", "depth_m", "conf", "NDWIice", "NDWI"] + REFL_BANDS].copy()
    for name in preds_test:
        pred_df[f"pred_{name}"] = preds_test[name]
        pred_df[f"pred_{name}_train"] = preds_train[name]
    pred_path = os.path.join(MODELS_DIR, "depth_cv_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    fig_path = os.path.join(FIGURES_DIR, "depth_model_cv_scatter.png")
    save_scatter_plot(y, preds_test, fig_path)

    print("\n=== Depth model comparison (GroupKFold by lake_id) ===")
    print(f"Rows: {len(df)} pixels | Lakes: {n_lakes}")
    print(f"Green EFM feature : {GREEN_FEATURE} only")
    print(f"Ridge/RF features ({len(ML_FEATURES)}): {', '.join(ML_FEATURES)}")
    print()
    for name in preds_test:
        mt = metrics_test[name]
        mr = metrics_train[name]
        print(
            f"{name:18s}  test RMSE={mt['rmse']:.3f} R2={mt['r2']:.3f}"
            f"  |  train RMSE={mr['rmse']:.3f} R2={mr['r2']:.3f}"
        )
    print()
    print(
        "Green EFM (fitted): depth = {:.4f} * exp(-{:.4f} * B3_refl) + {:.4f}".format(
            green.A_, green.B_, green.C_
        )
    )
    print(
        "Lutz 2024 SW ref : depth = {:.4f} * exp(-{:.4f} * B3_refl) + {:.4f}".format(
            LUTZ_SW_A, LUTZ_SW_B, LUTZ_SW_C
        )
    )
    print("Metrics JSON  :", metrics_path)
    print("CV predictions:", pred_path)
    print("Scatter plot  :", fig_path)


if __name__ == "__main__":
    main()
