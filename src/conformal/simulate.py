"""Synthetic day-ahead wind series for conformal experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindSimulationConfig:
    """Configuration for a synthetic DE-style wind forecast table."""

    n_days: int = 90
    seed: int = 42
    base_capacity_mw: float = 11_000.0
    tail_scale: float = 1.8
    seasonal_heteroskedasticity: bool = False
    drift_day: int | None = None
    drift_scale_factor: float = 1.35


def _feature_columns() -> list[str]:
    return [
        "nwp_wind_speed_hub_mps",
        "nwp_wind_dir_cos",
        "cos_hour",
        "sin_hour",
        "wind_mw_lag_1",
        "wind_mw_lag_24",
    ]


def simulate_wind_forecast(config: WindSimulationConfig | None = None) -> pd.DataFrame:
    """Return an hourly day-ahead wind table with known under-coverage dynamics.

    The target ``wind_mw`` follows NWP wind speed plus diurnal seasonality.
    Heteroskedastic, heavy-tailed noise makes raw quantile LightGBM bands
    under-cover on held-out data — the Week 3 calibration story.
    """
    config = config or WindSimulationConfig()
    rng = np.random.default_rng(config.seed)
    n_hours = config.n_days * 24

    rows: list[dict[str, object]] = []
    wind_history: list[float] = []

    for step in range(n_hours):
        ts = pd.Timestamp("2019-06-01", tz="UTC") + pd.Timedelta(hours=step)
        hour = ts.hour

        nwp_speed = 6.0 + 3.5 * np.sin(2 * np.pi * step / 168) + rng.normal(0.0, 0.8)
        nwp_speed = float(np.clip(nwp_speed, 0.5, 18.0))
        nwp_dir_cos = float(np.cos(2 * np.pi * step / 96))

        diurnal = 0.15 * np.sin(2 * np.pi * (hour - 6) / 24)
        base = config.base_capacity_mw * (0.08 + 0.045 * nwp_speed**1.35) * (1.0 + diurnal)

        lag_1 = wind_history[-1] if wind_history else base
        lag_24 = wind_history[-24] if len(wind_history) >= 24 else base

        # Heteroskedastic + heavy-tailed residuals: quantile models miss tails
        sigma = 350.0 + 0.04 * base + 120.0 * np.abs(nwp_speed - 8.0)
        if config.seasonal_heteroskedasticity:
            day_of_year = ts.dayofyear
            seasonal = 1.0 + 0.55 * np.sin(2 * np.pi * (day_of_year - 15) / 365.25)
            sigma *= seasonal
        if config.drift_day is not None and step >= config.drift_day * 24:
            sigma *= config.drift_scale_factor
        noise = rng.standard_t(df=3) * sigma * config.tail_scale
        if rng.random() < 0.03:
            noise += rng.choice([-1.0, 1.0]) * rng.uniform(800.0, 1800.0)

        wind_mw = max(0.0, base + noise)
        wind_history.append(wind_mw)

        rows.append(
            {
                "valid_time": ts,
                "wind_mw": wind_mw,
                "nwp_wind_speed_hub_mps": nwp_speed,
                "nwp_wind_dir_cos": nwp_dir_cos,
                "cos_hour": float(np.cos(2 * np.pi * hour / 24)),
                "sin_hour": float(np.sin(2 * np.pi * hour / 24)),
                "wind_mw_lag_1": lag_1,
                "wind_mw_lag_24": lag_24,
            }
        )

    frame = pd.DataFrame(rows)
    return frame


def feature_columns() -> list[str]:
    """Modeling feature names for the synthetic wind table."""
    return _feature_columns()
