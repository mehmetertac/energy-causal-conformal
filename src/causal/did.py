"""Difference-in-differences estimator for the tariff toy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class DidResult:
    """Point estimate and uncertainty for the DiD interaction term."""

    att_kw: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int
    formula: str


def _prepare_panel(
    frame: pd.DataFrame,
    outcome_col: str = "load_kw",
    peak_hours: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    required = {"treated", "post", outcome_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    panel = frame.copy()
    if peak_hours is not None:
        if "hour" not in panel.columns:
            raise ValueError("peak_hours filter requires an 'hour' column")
        panel = panel[panel["hour"].isin(peak_hours)].copy()

    panel["treated"] = panel["treated"].astype(int)
    panel["post"] = panel["post"].astype(int)
    panel["treated_post"] = panel["treated"] * panel["post"]
    panel = panel.dropna(subset=[outcome_col])
    return panel


def estimate_did(
    frame: pd.DataFrame,
    outcome_col: str = "load_kw",
    peak_hours: tuple[int, ...] | None = None,
    alpha: float = 0.05,
) -> DidResult:
    """Estimate ATT via two-way DiD OLS.

    Model: Y_it = alpha + beta*Treated_i + gamma*Post_t + delta*(Treated_i*Post_t) + eps
    """
    panel = _prepare_panel(frame, outcome_col=outcome_col, peak_hours=peak_hours)
    formula = f"{outcome_col} ~ treated + post + treated_post"
    model = smf.ols(formula, data=panel).fit(cov_type="HC1")
    ci_low, ci_high = model.conf_int(alpha=alpha).loc["treated_post"]

    return DidResult(
        att_kw=float(model.params["treated_post"]),
        std_error=float(model.bse["treated_post"]),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_obs=int(model.nobs),
        formula=formula,
    )


def naive_before_after(
    frame: pd.DataFrame,
    outcome_col: str = "load_kw",
    peak_hours: tuple[int, ...] | None = None,
) -> float:
    """Naive treated-only before/after contrast (confounded in the toy)."""
    panel = _prepare_panel(frame, outcome_col=outcome_col, peak_hours=peak_hours)
    treated = panel[panel["treated"] == 1]
    before = treated.loc[treated["post"] == 0, outcome_col].mean()
    after = treated.loc[treated["post"] == 1, outcome_col].mean()
    if np.isnan(before) or np.isnan(after):
        raise ValueError("Insufficient treated before/after observations")
    return float(after - before)
