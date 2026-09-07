"""Tests for the DiD estimator."""

from __future__ import annotations

from src.causal.did import estimate_did, naive_before_after
from src.causal.simulate import SimulationConfig, simulate_residential_load


def test_did_recovers_known_att_on_peak_hours() -> None:
    config = SimulationConfig(att_kw=-0.4, seed=7)
    panel = simulate_residential_load(config)
    result = estimate_did(panel, peak_hours=config.peak_hours)

    assert abs(result.att_kw - config.att_kw) < 0.05
    assert config.att_kw >= result.ci_low
    assert config.att_kw <= result.ci_high


def test_naive_before_after_is_confounded() -> None:
    config = SimulationConfig(att_kw=-0.4, weather_jump_kw=0.8, seed=7)
    panel = simulate_residential_load(config)

    naive = naive_before_after(panel, peak_hours=config.peak_hours)
    did = estimate_did(panel, peak_hours=config.peak_hours).att_kw

    assert abs(naive - config.att_kw) > 0.05
    assert abs(did - config.att_kw) < 0.05
