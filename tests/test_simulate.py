"""Tests for the residential load simulator."""

from __future__ import annotations

from src.causal.simulate import SimulationConfig, simulate_residential_load, summarize_panel


def test_simulator_schema_and_timing() -> None:
    config = SimulationConfig(n_treated=10, n_control=10, n_days_before=7, n_days_after=7)
    panel = simulate_residential_load(config)
    summary = summarize_panel(panel)

    assert summary["n_households"] == 20.0
    assert summary["n_treated"] == 10.0
    assert summary["n_control"] == 10.0
    assert set(panel.columns) >= {
        "household_id",
        "treated",
        "timestamp",
        "day",
        "hour",
        "post",
        "load_kw",
        "weather_shock_kw",
        "treated_post",
    }

    pre = panel[panel["post"] == 0]
    post = panel[panel["post"] == 1]
    assert pre["weather_shock_kw"].max() == 0.0
    assert post["weather_shock_kw"].min() > 0.0
    assert panel["day"].min() == 0
    assert panel["day"].max() == config.n_days_before + config.n_days_after - 1
