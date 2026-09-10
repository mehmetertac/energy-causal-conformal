"""Causal inference helpers for energy load experiments."""

from src.causal.did import DidResult, estimate_did, naive_before_after
from src.causal.propensity import (
    PropensityResult,
    estimate_dowhy_att,
    estimate_ipw_att,
    household_peak_deltas,
)
from src.causal.simulate import (
    SimulationConfig,
    SynthSimulationConfig,
    simulate_heterogeneous_load,
    simulate_residential_load,
    summarize_panel,
)
from src.causal.synth import (
    GapUncertainty,
    PlaceboResult,
    SyntheticControlResult,
    counterfactual_path,
    daily_peak_series,
    donor_pool_sensitivity,
    fit_synthetic_control,
    gap_uncertainty,
    in_space_placebos,
    in_time_placebos,
    inject_peak_reduction,
    naive_before_after_series,
    pre_period_sensitivity,
)

__all__ = [
    "DidResult",
    "GapUncertainty",
    "PlaceboResult",
    "PropensityResult",
    "SimulationConfig",
    "SyntheticControlResult",
    "SynthSimulationConfig",
    "counterfactual_path",
    "daily_peak_series",
    "donor_pool_sensitivity",
    "estimate_did",
    "estimate_dowhy_att",
    "estimate_ipw_att",
    "fit_synthetic_control",
    "gap_uncertainty",
    "household_peak_deltas",
    "in_space_placebos",
    "in_time_placebos",
    "inject_peak_reduction",
    "naive_before_after",
    "naive_before_after_series",
    "pre_period_sensitivity",
    "simulate_heterogeneous_load",
    "simulate_residential_load",
    "summarize_panel",
]
