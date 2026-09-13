"""Brief GPyTorch solar demo for Bayesian UQ interviews.

ExactGP only — O(n^3), capped at ~200 points. Not exported from ``src.conformal``.
"""

from __future__ import annotations

from dataclasses import dataclass

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.kernels import AdditiveKernel, PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


_MAX_TRAIN_POINTS = 200


@dataclass(frozen=True)
class SolarSimulationConfig:
    """Configuration for a short-horizon solar output series."""

    n_days: int = 4
    seed: int = 42
    peak_mw: float = 120.0
    cloud_scale: float = 8.0


@dataclass(frozen=True)
class GPSolarForecast:
    """Posterior mean and std on a test horizon."""

    mean: np.ndarray
    std: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


class _SolarGPModel(ExactGP):
    """RBF smoothness + 24 h periodicity for diurnal solar."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        periodic = ScaleKernel(PeriodicKernel())
        periodic.base_kernel.period_length = 24.0
        self.covar_module = AdditiveKernel(ScaleKernel(RBFKernel()), periodic)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


def simulate_solar_output(config: SolarSimulationConfig | None = None) -> pd.DataFrame:
    """Return hourly PV output with night zeros and a 24 h diurnal envelope."""
    config = config or SolarSimulationConfig()
    rng = np.random.default_rng(config.seed)
    n_hours = config.n_days * 24

    rows: list[dict[str, object]] = []
    for step in range(n_hours):
        ts = pd.Timestamp("2024-06-15", tz="UTC") + pd.Timedelta(hours=step)
        hour = ts.hour + ts.minute / 60.0
        diurnal = max(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
        cloud = rng.normal(0.0, config.cloud_scale) if diurnal > 0.05 else 0.0
        solar_mw = max(0.0, config.peak_mw * diurnal + cloud)
        rows.append(
            {
                "valid_time": ts,
                "hour_index": float(step),
                "solar_mw": solar_mw,
            }
        )
    return pd.DataFrame(rows)


def fit_gp_solar(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    x_col: str = "hour_index",
    y_col: str = "solar_mw",
    n_iter: int = 40,
) -> GPSolarForecast:
    """Fit ExactGP and return posterior mean/std on ``test``.

    Training uses Adam for ``n_iter`` steps. Intended for <= ~80 train hours
    and 6–12 hour horizons — interview demo only.
    """
    if len(train) > _MAX_TRAIN_POINTS:
        msg = f"train size {len(train)} exceeds cap {_MAX_TRAIN_POINTS} for ExactGP"
        raise ValueError(msg)

    train_x = torch.tensor(train[x_col].to_numpy(dtype=float), dtype=torch.float32)
    train_y = torch.tensor(train[y_col].to_numpy(dtype=float), dtype=torch.float32)
    test_x = torch.tensor(test[x_col].to_numpy(dtype=float), dtype=torch.float32)
    test_y = test[y_col].to_numpy(dtype=float)

    likelihood = GaussianLikelihood()
    model = _SolarGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(test_x))
        mean = posterior.mean.cpu().numpy()
        std = posterior.stddev.cpu().numpy()

    return GPSolarForecast(
        mean=mean,
        std=std,
        train_x=train[x_col].to_numpy(dtype=float),
        train_y=train[y_col].to_numpy(dtype=float),
        test_x=test[x_col].to_numpy(dtype=float),
        test_y=test_y,
    )
