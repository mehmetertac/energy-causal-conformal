"""Tests for the residential load simulator."""

from __future__ import annotations

from src.causal.simulate import (
    SimulationConfig,
    SynthSimulationConfig,
    simulate_heterogeneous_load,
    simulate_residential_load,
    summarize_panel,
)


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


def test_heterogeneous_simulator_schema_and_no_treatment() -> None:
    config = SynthSimulationConfig(n_treated=3, n_donors=8, n_days_before=14, n_days_after=7)
    panel = simulate_heterogeneous_load(config)
    summary = summarize_panel(panel)

    assert summary["n_households"] == 11.0
    assert summary["n_treated"] == 3.0
    assert summary["n_control"] == 8.0
    assert panel["household_id"].str.startswith("PS").all()

    pre_treated = panel[(panel["treated"] == 1) & (panel["post"] == 0)]["load_kw"].mean()
    post_treated = panel[(panel["treated"] == 1) & (panel["post"] == 1)]["load_kw"].mean()
    pre_control = panel[(panel["treated"] == 0) & (panel["post"] == 0)]["load_kw"].mean()
    post_control = panel[(panel["treated"] == 0) & (panel["post"] == 1)]["load_kw"].mean()

    treated_jump = post_treated - pre_treated
    control_jump = post_control - pre_control
    assert abs(treated_jump - control_jump) < 0.05
