"""Simulate a residential load panel with a known DiD treatment effect."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the tariff-style DiD toy panel."""

    n_treated: int = 50
    n_control: int = 50
    n_days_before: int = 30
    n_days_after: int = 30
    hours_per_day: int = 24
    att_kw: float = -0.4
    weather_jump_kw: float = 0.8
    peak_hours: tuple[int, ...] = (17, 18, 19, 20)
    seed: int = 42


def _household_ids(config: SimulationConfig) -> tuple[list[str], list[str]]:
    treated = [f"T{i:03d}" for i in range(config.n_treated)]
    control = [f"C{i:03d}" for i in range(config.n_control)]
    return treated, control


def simulate_residential_load(config: SimulationConfig | None = None) -> pd.DataFrame:
    """Return an hourly household load panel with a known ATT.

    The simulator injects:
    - a shared weather confounder that jumps after the tariff date (both groups)
    - a treatment effect on treated households in post-period peak hours only
    """
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)

    treated_ids, control_ids = _household_ids(config)
    n_days = config.n_days_before + config.n_days_after
    hours = config.hours_per_day
    tariff_day = config.n_days_before

    rows: list[dict[str, object]] = []
    for day in range(n_days):
        post = day >= tariff_day
        weather = config.weather_jump_kw if post else 0.0
        for hour in range(hours):
            diurnal = 0.6 + 0.35 * np.sin(2 * np.pi * (hour - 7) / 24)
            for household_id, treated in [
                *((hid, 1) for hid in treated_ids),
                *((hid, 0) for hid in control_ids),
            ]:
                base = 1.2 + diurnal + weather
                household_effect = 0.05 * (hash(household_id) % 7)
                noise = rng.normal(0.0, 0.05)

                treatment = 0.0
                if treated and post and hour in config.peak_hours:
                    treatment = config.att_kw

                load_kw = base + household_effect + treatment + noise
                timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, hours=hour)
                rows.append(
                    {
                        "household_id": household_id,
                        "treated": treated,
                        "timestamp": timestamp,
                        "day": day,
                        "hour": hour,
                        "post": int(post),
                        "load_kw": load_kw,
                        "weather_shock_kw": weather,
                    }
                )

    frame = pd.DataFrame(rows)
    frame["treated_post"] = frame["treated"] * frame["post"]
    return frame


def summarize_panel(frame: pd.DataFrame) -> dict[str, float]:
    """Return basic counts used in tests and notebooks."""
    return {
        "n_rows": float(len(frame)),
        "n_households": float(frame["household_id"].nunique()),
        "n_treated": float(frame.loc[frame["treated"] == 1, "household_id"].nunique()),
        "n_control": float(frame.loc[frame["treated"] == 0, "household_id"].nunique()),
        "post_share": float(frame["post"].mean()),
    }
