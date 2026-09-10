"""Tests for household-level propensity-weighted ATT."""

from __future__ import annotations

import pytest

from src.causal.did import estimate_did
from src.causal.propensity import estimate_ipw_att, household_peak_deltas
from src.causal.simulate import SynthSimulationConfig, simulate_heterogeneous_load
from src.causal.synth import inject_peak_reduction


def _injected_panel(reduction_pct: float = 0.15, seed: int = 7):
    config = SynthSimulationConfig(n_donors=12, seed=seed)
    treated_ids = [f"PS{i:04d}" for i in range(config.n_treated)]
    panel = simulate_heterogeneous_load(config)
    panel = inject_peak_reduction(
        panel,
        treated_ids=treated_ids,
        intervention_day=config.n_days_before,
        reduction_pct=reduction_pct,
        peak_hours=config.peak_hours,
    )
    return config, panel


def test_household_peak_deltas_schema() -> None:
    _, panel = _injected_panel()
    households = household_peak_deltas(panel, peak_hours=(17, 18, 19, 20))

    assert set(households.columns) >= {
        "household_id",
        "treated",
        "pre_peak_kw",
        "post_peak_kw",
        "delta_peak_kw",
    }
    assert households["household_id"].nunique() == len(households)
    assert households["treated"].isin([0, 1]).all()


def test_ipw_att_matches_did_sign_and_scale() -> None:
    config, panel = _injected_panel(reduction_pct=0.15)
    ipw = estimate_ipw_att(panel, peak_hours=config.peak_hours)
    did = estimate_did(panel, peak_hours=config.peak_hours)

    assert ipw.n_treated == config.n_treated
    assert ipw.n_control == config.n_donors
    assert ipw.att_kw < 0.0
    assert abs(ipw.att_kw - did.att_kw) < 0.08
    assert 0.05 <= ipw.min_propensity <= ipw.max_propensity <= 0.95


def test_dowhy_propensity_weighting_agrees_with_ipw() -> None:
    pytest.importorskip("dowhy")
    from src.causal.propensity import estimate_dowhy_att

    config, panel = _injected_panel(reduction_pct=0.15)
    ipw = estimate_ipw_att(panel, peak_hours=config.peak_hours)
    dowhy = estimate_dowhy_att(panel, peak_hours=config.peak_hours)

    assert dowhy.att_kw < 0.0
    assert abs(dowhy.att_kw - ipw.att_kw) < 0.08
