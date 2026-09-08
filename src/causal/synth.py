"""Synthetic control estimator for tariff-style peak-load experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class SyntheticControlResult:
    """Synthetic control fit and post-period effect summary."""

    weights: pd.Series
    treated: pd.Series
    synthetic: pd.Series
    pre_rmspe: float
    post_rmspe: float
    att_kw: float
    att_pct: float
    donor_ids: list[str]
    intervention_day: int


@dataclass(frozen=True)
class PlaceboResult:
    """In-space placebo gaps for each donor treated as a fake treated unit."""

    placebo_gaps_kw: pd.Series
    treated_gap_kw: float
    p_value: float


def inject_peak_reduction(
    frame: pd.DataFrame,
    treated_ids: list[str] | set[str],
    intervention_day: int,
    reduction_pct: float,
    peak_hours: tuple[int, ...] = (17, 18, 19, 20),
    load_col: str = "load_kw",
) -> pd.DataFrame:
    """Inject a known peak-load reduction for treated households after intervention_day."""
    if not 0.0 <= reduction_pct <= 1.0:
        raise ValueError("reduction_pct must be between 0 and 1")

    panel = frame.copy()
    treated_set = set(treated_ids)
    mask = (
        panel["household_id"].isin(treated_set)
        & (panel["day"] >= intervention_day)
        & panel["hour"].isin(peak_hours)
    )
    panel.loc[mask, load_col] = panel.loc[mask, load_col] * (1.0 - reduction_pct)
    return panel


def daily_peak_series(
    frame: pd.DataFrame,
    treated_ids: list[str] | set[str],
    donor_ids: list[str] | set[str],
    peak_hours: tuple[int, ...] = (17, 18, 19, 20),
    load_col: str = "load_kw",
) -> tuple[pd.Series, pd.DataFrame]:
    """Aggregate hourly load to daily mean peak-hour kW for treated unit and donors."""
    peak = frame[frame["hour"].isin(peak_hours)].copy()
    if peak.empty:
        raise ValueError("No rows match peak_hours filter")

    treated_set = set(treated_ids)
    donor_list = list(donor_ids)

    treated_daily = (
        peak.loc[peak["household_id"].isin(treated_set)]
        .groupby("day", as_index=True)[load_col]
        .mean()
        .sort_index()
    )
    donor_daily = (
        peak.loc[peak["household_id"].isin(donor_list)]
        .pivot_table(index="day", columns="household_id", values=load_col, aggfunc="mean")
        .sort_index()
    )
    donor_daily = donor_daily.reindex(columns=donor_list)
    shared_days = treated_daily.index.intersection(donor_daily.index)
    if shared_days.empty:
        raise ValueError("Treated and donor series have no overlapping days")

    return treated_daily.loc[shared_days], donor_daily.loc[shared_days]


def _rmspe(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) == 0:
        raise ValueError("Cannot compute RMSPE on empty arrays")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _fit_weights(y_pre: np.ndarray, x_pre: np.ndarray) -> np.ndarray:
    """Fit simplex weights minimizing pre-period squared error."""
    n_donors = x_pre.shape[1]
    if n_donors == 0:
        raise ValueError("Donor matrix must have at least one column")

    def objective(weights: np.ndarray) -> float:
        return float(np.sum((y_pre - x_pre @ weights) ** 2))

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    bounds = [(0.0, 1.0)] * n_donors
    x0 = np.full(n_donors, 1.0 / n_donors)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Synthetic control optimization failed: {result.message}")
    return result.x


def counterfactual_path(weights: pd.Series, donors: pd.DataFrame) -> pd.Series:
    """Build the synthetic counterfactual from donor weights."""
    aligned = donors.reindex(columns=weights.index)
    missing = weights.index.difference(aligned.columns)
    if len(missing):
        raise ValueError(f"Donor columns missing for weights: {list(missing)}")
    return aligned @ weights


def fit_synthetic_control(
    treated: pd.Series,
    donors: pd.DataFrame,
    intervention_day: int,
) -> SyntheticControlResult:
    """Fit Abadie-style simplex weights on the pre-period and estimate post gap."""
    shared_index = treated.index.intersection(donors.index)
    if shared_index.empty:
        raise ValueError("Treated and donor series share no index values")

    y = treated.loc[shared_index]
    x = donors.loc[shared_index]
    pre_mask = shared_index < intervention_day
    post_mask = shared_index >= intervention_day
    if pre_mask.sum() == 0:
        raise ValueError("Pre-period is empty; cannot fit synthetic control")
    if post_mask.sum() == 0:
        raise ValueError("Post-period is empty; cannot estimate effect")

    weights_array = _fit_weights(y.loc[pre_mask].to_numpy(), x.loc[pre_mask].to_numpy())
    weight_series = pd.Series(weights_array, index=x.columns, name="weight")
    synthetic = counterfactual_path(weight_series, x)

    pre_rmspe = _rmspe(y.loc[pre_mask].to_numpy(), synthetic.loc[pre_mask].to_numpy())
    post_rmspe = _rmspe(y.loc[post_mask].to_numpy(), synthetic.loc[post_mask].to_numpy())
    gap = y.loc[post_mask] - synthetic.loc[post_mask]
    att_kw = float(gap.mean())
    synthetic_post_mean = float(synthetic.loc[post_mask].mean())
    att_pct = float(att_kw / synthetic_post_mean) if synthetic_post_mean else 0.0

    return SyntheticControlResult(
        weights=weight_series,
        treated=y,
        synthetic=synthetic,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        att_kw=att_kw,
        att_pct=att_pct,
        donor_ids=list(x.columns),
        intervention_day=intervention_day,
    )


def naive_before_after_series(treated: pd.Series, intervention_day: int) -> float:
    """Naive treated-only before/after gap on a daily series."""
    pre = treated.loc[treated.index < intervention_day]
    post = treated.loc[treated.index >= intervention_day]
    if pre.empty or post.empty:
        raise ValueError("Insufficient pre/post observations for naive before/after")
    return float(post.mean() - pre.mean())


def in_space_placebos(
    treated: pd.Series,
    donors: pd.DataFrame,
    intervention_day: int,
) -> PlaceboResult:
    """Run in-space placebos by treating each donor as the treated unit."""
    gaps: dict[str, float] = {}
    for donor_id in donors.columns:
        placebo_treated = donors[donor_id]
        placebo_donors = donors.drop(columns=[donor_id])
        if placebo_donors.shape[1] == 0:
            continue
        try:
            result = fit_synthetic_control(
                placebo_treated,
                placebo_donors,
                intervention_day=intervention_day,
            )
        except (RuntimeError, ValueError):
            continue
        gaps[donor_id] = result.att_kw

    if not gaps:
        raise ValueError("No successful in-space placebo fits")

    placebo_gaps = pd.Series(gaps, name="placebo_gap_kw")
    treated_result = fit_synthetic_control(treated, donors, intervention_day=intervention_day)
    treated_gap = treated_result.att_kw
    n_extreme = int((np.abs(placebo_gaps) >= abs(treated_gap)).sum())
    p_value = float((n_extreme + 1) / (len(placebo_gaps) + 1))
    return PlaceboResult(
        placebo_gaps_kw=placebo_gaps.sort_values(),
        treated_gap_kw=treated_gap,
        p_value=p_value,
    )
