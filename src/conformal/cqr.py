"""MAPIE conformalized quantile regression wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from mapie.regression import ConformalizedQuantileRegressor

from src.conformal.coverage import CoverageResult, evaluate_intervals
from src.conformal.quantile_lgbm import QuantileLGBM


@dataclass(frozen=True)
class ConformalForecastResult:
    """Raw and conformalized interval evaluation on a test block."""

    raw_80: CoverageResult
    raw_90: CoverageResult
    cqr_80: CoverageResult
    cqr_90: CoverageResult
    test_frame: pd.DataFrame


def _cqr_quantiles(confidence_level: float) -> tuple[float, float, float]:
    """Return (lower, upper, median) quantile levels for CQR prefit."""
    lower = round((1.0 - confidence_level) / 2.0, 4)
    upper = round((1.0 + confidence_level) / 2.0, 4)
    return lower, upper, 0.5


def _apply_cqr(
    model: QuantileLGBM,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conformalize quantile models and predict intervals on test."""
    q_lo, q_hi, q_med = _cqr_quantiles(confidence_level)
    for q in (q_lo, q_hi, q_med):
        if q not in model.quantiles:
            msg = f"model missing quantile {q} required for confidence={confidence_level}"
            raise ValueError(msg)

    estimators = [
        model.estimator_at(q_lo),
        model.estimator_at(q_hi),
        model.estimator_at(q_med),
    ]
    X_cal = cal_df[feature_cols]
    y_cal = cal_df[target_col]
    X_test = test_df[feature_cols]

    cqr = ConformalizedQuantileRegressor(
        estimator=estimators,
        confidence_level=confidence_level,
        prefit=True,
    )
    cqr.conformalize(X_cal, y_cal)
    y_pred, y_intervals = cqr.predict_interval(X_test)
    # y_intervals shape: (n_samples, 2, 1) or (n_samples, 2)
    intervals = np.asarray(y_intervals)
    if intervals.ndim == 3:
        lo = intervals[:, 0, 0]
        hi = intervals[:, 1, 0]
    else:
        lo = intervals[:, 0]
        hi = intervals[:, 1]
    return np.asarray(y_pred, dtype=float), lo, hi


def run_conformal_cqr(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "wind_mw",
    model: QuantileLGBM | None = None,
    model_params: dict | None = None,
) -> ConformalForecastResult:
    """Fit quantile LightGBM, apply CQR at 80% and 90%, evaluate on test.

    Args:
        train_df: Training block (earliest).
        cal_df: Calibration block for conformal scores.
        test_df: Held-out evaluation block (latest).
        feature_cols: Feature column names.
        target_col: Target column (default ``wind_mw``).
        model: Optional pre-configured QuantileLGBM; fitted when None.
        model_params: LightGBM hyperparameters when building a new model.

    Returns:
        Coverage summaries for raw and CQR intervals at 80% and 90%.
    """
    if model is None:
        model = QuantileLGBM(model_params=model_params or {"n_estimators": 80})
    model.fit(train_df[feature_cols], train_df[target_col])

    raw_preds = model.predict(test_df[feature_cols])
    y_test = test_df[target_col]

    q10, q50, q90 = 0.1, 0.5, 0.9
    q05, q95 = 0.05, 0.95

    raw_80 = evaluate_intervals(y_test, raw_preds[q10], raw_preds[q90], nominal=0.80)
    raw_90 = evaluate_intervals(y_test, raw_preds[q05], raw_preds[q95], nominal=0.90)

    _, cqr80_lo, cqr80_hi = _apply_cqr(
        model, cal_df, test_df, feature_cols, target_col, confidence_level=0.80
    )
    _, cqr90_lo, cqr90_hi = _apply_cqr(
        model, cal_df, test_df, feature_cols, target_col, confidence_level=0.90
    )

    cqr_80 = evaluate_intervals(y_test, cqr80_lo, cqr80_hi, nominal=0.80)
    cqr_90 = evaluate_intervals(y_test, cqr90_lo, cqr90_hi, nominal=0.90)

    out = test_df.copy()
    out["pred_p50"] = raw_preds[q50]
    out["raw_p10"] = raw_preds[q10]
    out["raw_p90"] = raw_preds[q90]
    out["cqr80_lo"] = cqr80_lo
    out["cqr80_hi"] = cqr80_hi
    out["cqr90_lo"] = cqr90_lo
    out["cqr90_hi"] = cqr90_hi

    return ConformalForecastResult(
        raw_80=raw_80,
        raw_90=raw_90,
        cqr_80=cqr_80,
        cqr_90=cqr_90,
        test_frame=out,
    )
