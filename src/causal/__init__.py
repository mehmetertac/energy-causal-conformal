"""Causal inference helpers for energy load experiments."""

from src.causal.did import DidResult, estimate_did, naive_before_after
from src.causal.simulate import (
    SimulationConfig,
    SynthSimulationConfig,
    simulate_heterogeneous_load,
    simulate_residential_load,
    summarize_panel,
)
from src.causal.synth import (
    PlaceboResult,
    SyntheticControlResult,
    counterfactual_path,
    daily_peak_series,
    fit_synthetic_control,
    in_space_placebos,
    inject_peak_reduction,
    naive_before_after_series,
)

__all__ = [
    "DidResult",
    "PlaceboResult",
    "SimulationConfig",
    "SyntheticControlResult",
    "SynthSimulationConfig",
    "counterfactual_path",
    "daily_peak_series",
    "estimate_did",
    "fit_synthetic_control",
    "in_space_placebos",
    "inject_peak_reduction",
    "naive_before_after",
    "naive_before_after_series",
    "simulate_heterogeneous_load",
    "simulate_residential_load",
    "summarize_panel",
]
