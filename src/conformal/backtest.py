"""Rolling-origin conformal backtests for forecast intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.conformal.coverage import evaluate_intervals, pinball_loss
from src.conformal.cqr import run_conformal_cqr
from src.conformal.quantile_lgbm import QuantileLGBM


@dataclass(frozen=True)
class RollingOriginConfig:
    """Hour-based expanding-window rolling origin settings."""

    min_train_hours: int = 90 * 24
    cal_hours: int = 21 * 24
    test_hours: int = 30 * 24
    step_hours: int = 30 * 24
    gap_hours: int = 24


@dataclass(frozen=True)
class RollingBacktestResult:
    """Coverage summaries and predictions from a rolling-origin walk."""

    fold_table: pd.DataFrame
    predictions: pd.DataFrame


def _apply_gap(work: pd.DataFrame, start_idx: int, gap_hours: int, time_col: str) -> int:
    """Return index after ``gap_hours`` from row ``start_idx - 1``."""
    if gap_hours <= 0 or start_idx <= 0:
        return start_idx
    cutoff = work.iloc[start_idx - 1][time_col] + pd.Timedelta(hours=gap_hours)
    next_idx = int(work[time_col].searchsorted(cutoff, side="left"))
    return max(next_idx, start_idx)


def _wind_regime_labels(speed: pd.Series) -> pd.Series:
    """Label low / mid / high wind terciles from NWP hub speed."""
    q1, q2 = speed.quantile([1 / 3, 2 / 3])
    labels = pd.Series("mid", index=speed.index, dtype="string")
    labels[speed <= q1] = "low"
    labels[speed > q2] = "high"
    return labels


def _annotate_predictions(frame: pd.DataFrame, *, time_col: str, fold_id: int) -> pd.DataFrame:
    """Add fold id, calendar month, and wind-regime tags."""
    out = frame.copy()
    out["fold_id"] = fold_id
    out["month"] = pd.to_datetime(out[time_col]).dt.to_period("M").astype(str)
    out["wind_regime"] = _wind_regime_labels(out["nwp_wind_speed_hub_mps"])
    return out


def rolling_origin_backtest(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    config: RollingOriginConfig | None = None,
    time_col: str = "valid_time",
    target_col: str = "wind_mw",
    model_params: dict | None = None,
) -> RollingBacktestResult:
    """Run expanding-window CQR backtest with fixed calibration and test blocks.

    Training data grows from the series start; each fold advances the
    calibration and test windows by ``step_hours`` while keeping block sizes fixed.
    """
    cfg = config or RollingOriginConfig()
    work = frame.sort_values(time_col).reset_index(drop=True)
    n = len(work)
    need = cfg.min_train_hours + cfg.cal_hours + cfg.test_hours + 2 * cfg.gap_hours
    if n < need:
        msg = f"need at least {need} rows for rolling backtest, got {n}"
        raise ValueError(msg)

    fold_rows: list[dict[str, object]] = []
    pred_parts: list[pd.DataFrame] = []
    fold_id = 0
    train_end = cfg.min_train_hours

    while True:
        cal_start = _apply_gap(work, train_end, cfg.gap_hours, time_col)
        cal_end = cal_start + cfg.cal_hours
        if cal_end > n:
            break
        test_start = _apply_gap(work, cal_end, cfg.gap_hours, time_col)
        test_end = test_start + cfg.test_hours
        if test_end > n:
            break

        train_df = work.iloc[:train_end].copy()
        cal_df = work.iloc[cal_start:cal_end].copy()
        test_df = work.iloc[test_start:test_end].copy()
        if train_df.empty or cal_df.empty or test_df.empty:
            break

        model = QuantileLGBM(model_params=model_params or {"n_estimators": 80})
        result = run_conformal_cqr(
            train_df,
            cal_df,
            test_df,
            feature_cols,
            target_col=target_col,
            model=model,
        )

        test_start_ts = test_df[time_col].min()
        test_month = pd.Timestamp(test_start_ts).to_period("M")
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_hours": len(train_df),
                "test_start": test_start_ts,
                "test_month": str(test_month),
                "raw_80_coverage": result.raw_80.coverage,
                "raw_80_width": result.raw_80.mean_width,
                "cqr_80_coverage": result.cqr_80.coverage,
                "cqr_80_width": result.cqr_80.mean_width,
                "raw_90_coverage": result.raw_90.coverage,
                "raw_90_width": result.raw_90.mean_width,
                "cqr_90_coverage": result.cqr_90.coverage,
                "cqr_90_width": result.cqr_90.mean_width,
                "n_test": result.cqr_80.n_obs,
            }
        )
        pred_parts.append(_annotate_predictions(result.test_frame, time_col=time_col, fold_id=fold_id))
        fold_id += 1
        train_end += cfg.step_hours

    if not fold_rows:
        msg = "rolling backtest produced zero folds; shorten blocks or add data"
        raise ValueError(msg)

    predictions = pd.concat(pred_parts, ignore_index=True)
    return RollingBacktestResult(
        fold_table=pd.DataFrame(fold_rows),
        predictions=predictions,
    )


def coverage_by_month(predictions: pd.DataFrame, *, target_col: str = "wind_mw") -> pd.DataFrame:
    """Empirical coverage and mean width grouped by calendar month."""
    rows: list[dict[str, object]] = []
    for month, grp in predictions.groupby("month", sort=True):
        y = grp[target_col]
        rows.append(
            {
                "month": month,
                "n_obs": len(grp),
                "raw_80_coverage": evaluate_intervals(y, grp["raw_p10"], grp["raw_p90"], nominal=0.80).coverage,
                "cqr_80_coverage": evaluate_intervals(y, grp["cqr80_lo"], grp["cqr80_hi"], nominal=0.80).coverage,
                "raw_80_width": evaluate_intervals(y, grp["raw_p10"], grp["raw_p90"], nominal=0.80).mean_width,
                "cqr_80_width": evaluate_intervals(y, grp["cqr80_lo"], grp["cqr80_hi"], nominal=0.80).mean_width,
            }
        )
    return pd.DataFrame(rows)


def coverage_by_regime(predictions: pd.DataFrame, *, target_col: str = "wind_mw") -> pd.DataFrame:
    """Empirical coverage and mean width grouped by wind-speed tercile."""
    rows: list[dict[str, object]] = []
    for regime in ("low", "mid", "high"):
        grp = predictions[predictions["wind_regime"] == regime]
        if grp.empty:
            continue
        y = grp[target_col]
        rows.append(
            {
                "wind_regime": regime,
                "n_obs": len(grp),
                "raw_80_coverage": evaluate_intervals(y, grp["raw_p10"], grp["raw_p90"], nominal=0.80).coverage,
                "cqr_80_coverage": evaluate_intervals(y, grp["cqr80_lo"], grp["cqr80_hi"], nominal=0.80).coverage,
                "raw_80_width": evaluate_intervals(y, grp["raw_p10"], grp["raw_p90"], nominal=0.80).mean_width,
                "cqr_80_width": evaluate_intervals(y, grp["cqr80_lo"], grp["cqr80_hi"], nominal=0.80).mean_width,
            }
        )
    return pd.DataFrame(rows)


def pinball_comparison(predictions: pd.DataFrame, *, target_col: str = "wind_mw") -> pd.DataFrame:
    """Pinball loss for raw quantiles vs CQR 80% endpoints treated as P10/P90."""
    y = predictions[target_col]
    return pd.DataFrame(
        [
            {
                "quantile": "P10",
                "raw_pinball": pinball_loss(y, predictions["raw_p10"], 0.10),
                "cqr_pinball": pinball_loss(y, predictions["cqr80_lo"], 0.10),
            },
            {
                "quantile": "P50",
                "raw_pinball": pinball_loss(y, predictions["pred_p50"], 0.50),
                "cqr_pinball": pinball_loss(y, predictions["pred_p50"], 0.50),
            },
            {
                "quantile": "P90",
                "raw_pinball": pinball_loss(y, predictions["raw_p90"], 0.90),
                "cqr_pinball": pinball_loss(y, predictions["cqr80_hi"], 0.90),
            },
        ]
    )
