"""Tests for the brief GPyTorch solar demo (optional dependency)."""

from __future__ import annotations

import numpy as np
import pytest

import importlib.util
import sys
from pathlib import Path

try:
    pytest.importorskip("gpytorch")
except (OSError, ImportError):
    pytest.skip("GPyTorch/torch unavailable in this environment", allow_module_level=True)

_gp_path = Path(__file__).resolve().parents[1] / "src" / "conformal" / "gp_solar.py"
_spec = importlib.util.spec_from_file_location("gp_solar_test_module", _gp_path)
assert _spec and _spec.loader
_gp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _gp
try:
    _spec.loader.exec_module(_gp)
except (OSError, ImportError) as exc:
    pytest.skip(f"gp_solar module unavailable: {exc}", allow_module_level=True)

SolarSimulationConfig = _gp.SolarSimulationConfig
fit_gp_solar = _gp.fit_gp_solar
simulate_solar_output = _gp.simulate_solar_output


def test_simulate_solar_output_shape_and_non_negative() -> None:
    frame = simulate_solar_output(SolarSimulationConfig(n_days=4, seed=1))
    assert len(frame) == 4 * 24
    assert (frame["solar_mw"] >= 0).all()
    assert "hour_index" in frame.columns
    assert "valid_time" in frame.columns


def test_simulate_solar_output_has_diurnal_peak() -> None:
    frame = simulate_solar_output(SolarSimulationConfig(n_days=2, seed=3))
    daytime = frame.loc[frame["valid_time"].dt.hour.between(10, 14), "solar_mw"]
    nighttime = frame.loc[frame["valid_time"].dt.hour.between(0, 4), "solar_mw"]
    assert daytime.mean() > nighttime.max()


def test_fit_gp_solar_returns_finite_arrays() -> None:
    frame = simulate_solar_output(SolarSimulationConfig(n_days=4, seed=7))
    train = frame.iloc[:72]
    test = frame.iloc[72:84]
    forecast = fit_gp_solar(train, test, n_iter=15)

    assert len(forecast.mean) == len(test)
    assert len(forecast.std) == len(test)
    assert np.all(np.isfinite(forecast.mean))
    assert np.all(np.isfinite(forecast.std))
    assert (forecast.std > 0).all()
