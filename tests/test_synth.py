"""Tests for the synthetic control estimator."""

from __future__ import annotations

import numpy as np

from src.causal.did import estimate_did
from src.causal.simulate import SynthSimulationConfig, simulate_heterogeneous_load
from src.causal.synth import (
    daily_peak_series,
    donor_pool_sensitivity,
    fit_synthetic_control,
    gap_uncertainty,
    inject_peak_reduction,
    in_space_placebos,
    in_time_placebos,
    naive_before_after_series,
    pre_period_sensitivity,
)


def _run_experiment(reduction_pct: float = 0.15, seed: int = 7) -> tuple[float, float, float]:
    config = SynthSimulationConfig(seed=seed)
    treated_ids, donor_ids = (
        [f"PS{i:04d}" for i in range(config.n_treated)],
        [f"PS{i + 100:04d}" for i in range(config.n_donors)],
    )
    panel = simulate_heterogeneous_load(config)
    treated_panel = inject_peak_reduction(
        panel,
        treated_ids=treated_ids,
        intervention_day=config.n_days_before,
        reduction_pct=reduction_pct,
        peak_hours=config.peak_hours,
    )
    treated, donors = daily_peak_series(
        treated_panel,
        treated_ids=treated_ids,
        donor_ids=donor_ids,
        peak_hours=config.peak_hours,
    )
    result = fit_synthetic_control(
        treated,
        donors,
        intervention_day=config.n_days_before,
    )
    naive = naive_before_after_series(treated, intervention_day=config.n_days_before)
    return result.att_pct, naive, result.pre_rmspe


def test_weights_live_on_simplex() -> None:
    config = SynthSimulationConfig(seed=7)
    treated_ids = [f"PS{i:04d}" for i in range(config.n_treated)]
    donor_ids = [f"PS{i + 100:04d}" for i in range(config.n_donors)]
    panel = simulate_heterogeneous_load(config)
    treated, donors = daily_peak_series(
        panel,
        treated_ids=treated_ids,
        donor_ids=donor_ids,
        peak_hours=config.peak_hours,
    )
    result = fit_synthetic_control(
        treated,
        donors,
        intervention_day=config.n_days_before,
    )

    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert (result.weights >= -1e-8).all()
    assert result.pre_rmspe < 0.15


def test_synthetic_control_recovers_injected_reduction() -> None:
    reduction_pct = 0.15
    att_pct, _, pre_rmspe = _run_experiment(reduction_pct=reduction_pct, seed=7)

    assert pre_rmspe < 0.15
    assert abs(att_pct + reduction_pct) < 0.04


def test_naive_before_after_is_confounded_on_series() -> None:
    reduction_pct = 0.15
    att_pct, naive, _ = _run_experiment(reduction_pct=reduction_pct, seed=7)

    true_gap_pct = -reduction_pct
    naive_pct = naive / 1.0  # gap in kW; weather dominates magnitude
    assert abs(att_pct - true_gap_pct) < abs(naive / 2.0 - true_gap_pct)
    assert abs(naive) > abs(att_pct)


def test_in_space_placebos_return_p_value() -> None:
    config = SynthSimulationConfig(n_donors=12, seed=7)
    treated_ids = [f"PS{i:04d}" for i in range(config.n_treated)]
    donor_ids = [f"PS{i + 100:04d}" for i in range(config.n_donors)]
    panel = simulate_heterogeneous_load(config)
    panel = inject_peak_reduction(
        panel,
        treated_ids=treated_ids,
        intervention_day=config.n_days_before,
        reduction_pct=0.15,
        peak_hours=config.peak_hours,
    )
    treated, donors = daily_peak_series(
        panel,
        treated_ids=treated_ids,
        donor_ids=donor_ids,
        peak_hours=config.peak_hours,
    )
    placebo = in_space_placebos(treated, donors, intervention_day=config.n_days_before)

    assert 0.0 < placebo.p_value <= 1.0
    assert len(placebo.placebo_gaps_kw) >= 5
    assert np.isfinite(placebo.treated_gap_kw)


def _peak_series(
    n_donors: int = 12,
    reduction_pct: float = 0.15,
    seed: int = 7,
):
    config = SynthSimulationConfig(n_donors=n_donors, seed=seed)
    treated_ids = [f"PS{i:04d}" for i in range(config.n_treated)]
    donor_ids = [f"PS{i + 100:04d}" for i in range(config.n_donors)]
    panel = simulate_heterogeneous_load(config)
    panel = inject_peak_reduction(
        panel,
        treated_ids=treated_ids,
        intervention_day=config.n_days_before,
        reduction_pct=reduction_pct,
        peak_hours=config.peak_hours,
    )
    treated, donors = daily_peak_series(
        panel,
        treated_ids=treated_ids,
        donor_ids=donor_ids,
        peak_hours=config.peak_hours,
    )
    return config, panel, treated, donors


def test_in_time_placebos_are_smaller_than_real_gap() -> None:
    config, _, treated, donors = _peak_series()
    placebo = in_time_placebos(
        treated,
        donors,
        intervention_day=config.n_days_before,
        max_placebos=6,
    )

    assert len(placebo.placebo_gaps_kw) >= 3
    assert placebo.p_value <= 0.25
    assert abs(placebo.treated_gap_kw) > abs(placebo.placebo_gaps_kw).median()


def test_gap_uncertainty_interval_covers_point_estimate() -> None:
    config, _, treated, donors = _peak_series()
    result = fit_synthetic_control(treated, donors, intervention_day=config.n_days_before)
    uncertainty = gap_uncertainty(result, n_boot=200, seed=7)

    assert uncertainty.n_post > 0
    assert uncertainty.ci_low < uncertainty.att_kw < uncertainty.ci_high
    assert uncertainty.ci_high < 0.0
    assert abs(uncertainty.att_kw - result.att_kw) < 1e-9


def test_did_cross_check_agrees_with_synthetic_control() -> None:
    config, panel, treated, donors = _peak_series()
    sc = fit_synthetic_control(treated, donors, intervention_day=config.n_days_before)
    did = estimate_did(panel, peak_hours=config.peak_hours)

    assert did.ci_high < 0.0
    assert abs(sc.att_kw - did.att_kw) < 0.08


def test_donor_pool_and_pre_period_sensitivity_stay_near_injected_effect() -> None:
    config, _, treated, donors = _peak_series()
    donor_sens = donor_pool_sensitivity(
        treated,
        donors,
        intervention_day=config.n_days_before,
        drop_counts=(0, 3),
        n_draws=2,
        seed=7,
    )
    pre_sens = pre_period_sensitivity(
        treated,
        donors,
        intervention_day=config.n_days_before,
        pre_lengths=(30, 45, 60),
    )

    assert not donor_sens.empty
    assert not pre_sens.empty
    assert donor_sens["att_pct"].between(-0.22, -0.08).all()
    assert pre_sens["att_pct"].between(-0.22, -0.08).all()
