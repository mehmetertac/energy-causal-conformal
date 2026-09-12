"""Tests for conformal prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.conformal.backtest import (
    RollingOriginConfig,
    coverage_by_month,
    coverage_by_regime,
    pinball_comparison,
    rolling_origin_backtest,
)
from src.conformal.coverage import evaluate_intervals, pinball_loss, pi_coverage
from src.conformal.cqr import _cqr_quantiles, run_conformal_cqr
from src.conformal.quantile_lgbm import QuantileLGBM
from src.conformal.simulate import WindSimulationConfig, feature_columns, simulate_wind_forecast
from src.conformal.split import chronological_conformal_split


def test_pi_coverage_all_inside() -> None:
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([4.0, 4.0, 4.0])
    assert pi_coverage(y, lo, hi) == 1.0


def test_pi_coverage_empty() -> None:
    assert np.isnan(pi_coverage(np.array([]), np.array([]), np.array([])))


def test_chronological_split_respects_order_and_gap() -> None:
    frame = simulate_wind_forecast(WindSimulationConfig(n_days=30, seed=1))
    train, cal, test = chronological_conformal_split(
        frame,
        time_col="valid_time",
        gap_hours=24,
    )

    assert train["valid_time"].max() + pd.Timedelta(hours=24) <= cal["valid_time"].min()
    assert cal["valid_time"].max() + pd.Timedelta(hours=24) <= test["valid_time"].min()
    assert len(train) + len(cal) + len(test) < len(frame)


def test_cqr_quantile_levels_match_confidence() -> None:
    lo, hi, med = _cqr_quantiles(0.80)
    assert lo == 0.1
    assert hi == 0.9
    assert med == 0.5

    lo90, hi90, med90 = _cqr_quantiles(0.90)
    assert lo90 == 0.05
    assert hi90 == 0.95
    assert med90 == 0.5


def test_raw_undercovers_cqr_closer_to_nominal() -> None:
    frame = simulate_wind_forecast(WindSimulationConfig(n_days=120, seed=7))
    cols = feature_columns()
    train, cal, test = chronological_conformal_split(frame, gap_hours=24)

    result = run_conformal_cqr(
        train,
        cal,
        test,
        cols,
        model=QuantileLGBM(model_params={"n_estimators": 40, "num_leaves": 31}),
    )

    assert result.raw_80.coverage < 0.80
    assert abs(result.cqr_80.coverage - 0.80) < 0.12
    assert result.cqr_80.mean_width >= result.raw_80.mean_width
    assert result.cqr_90.mean_width >= result.raw_90.mean_width


def test_evaluate_intervals_reports_gap() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.5, 3.5, 3.0])
    summary = evaluate_intervals(y, lo, hi, nominal=0.80)
    assert summary.n_obs == 4
    assert summary.coverage_gap == summary.coverage - 0.80


def test_pinball_loss_zero_when_predictions_match() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == 0.0


def test_pinball_loss_positive_when_predictions_miss() -> None:
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([0.0, 0.0, 0.0])
    assert pinball_loss(y, pred, 0.5) > 0.0


def test_rolling_backtest_returns_multiple_folds() -> None:
    frame = simulate_wind_forecast(WindSimulationConfig(n_days=200, seed=3))
    cols = feature_columns()
    cfg = RollingOriginConfig(
        min_train_hours=30 * 24,
        cal_hours=7 * 24,
        test_hours=14 * 24,
        step_hours=14 * 24,
        gap_hours=24,
    )
    result = rolling_origin_backtest(
        frame,
        cols,
        config=cfg,
        model_params={"n_estimators": 20, "num_leaves": 31},
    )
    assert len(result.fold_table) >= 2
    assert {"month", "wind_regime", "fold_id"}.issubset(result.predictions.columns)
    assert not coverage_by_month(result.predictions).empty
    assert set(coverage_by_regime(result.predictions)["wind_regime"]) <= {"low", "mid", "high"}


def test_rolling_backtest_cqr_wider_and_closer_to_nominal() -> None:
    frame = simulate_wind_forecast(WindSimulationConfig(n_days=200, seed=11))
    cols = feature_columns()
    cfg = RollingOriginConfig(
        min_train_hours=30 * 24,
        cal_hours=7 * 24,
        test_hours=14 * 24,
        step_hours=14 * 24,
        gap_hours=24,
    )
    result = rolling_origin_backtest(
        frame,
        cols,
        config=cfg,
        model_params={"n_estimators": 20, "num_leaves": 31},
    )
    raw_gap = abs(result.fold_table["raw_80_coverage"].mean() - 0.80)
    cqr_gap = abs(result.fold_table["cqr_80_coverage"].mean() - 0.80)
    assert cqr_gap <= raw_gap + 0.05
    assert result.fold_table["cqr_80_width"].mean() >= result.fold_table["raw_80_width"].mean()


def test_pinball_comparison_p50_unchanged_by_cqr() -> None:
    frame = simulate_wind_forecast(WindSimulationConfig(n_days=200, seed=5))
    cols = feature_columns()
    cfg = RollingOriginConfig(
        min_train_hours=30 * 24,
        cal_hours=7 * 24,
        test_hours=14 * 24,
        step_hours=14 * 24,
        gap_hours=24,
    )
    result = rolling_origin_backtest(
        frame,
        cols,
        config=cfg,
        model_params={"n_estimators": 20, "num_leaves": 31},
    )
    table = pinball_comparison(result.predictions)
    p50 = table.loc[table["quantile"] == "P50"].iloc[0]
    assert p50["raw_pinball"] == p50["cqr_pinball"]
