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
    """Placebo gaps (in-space or in-time) versus the treated-unit effect."""

    placebo_gaps_kw: pd.Series
    treated_gap_kw: float
    p_value: float


@dataclass(frozen=True)
class GapUncertainty:
    """Uncertainty for the post-period mean gap, treating the synthetic path as fixed.

    This interval captures day-to-day gap variability after weights are fit. It does
    not include uncertainty from weight estimation; placebo tests cover that role.
    """

    att_kw: float
    std_error: float
    ci_low: float
    ci_high: float
    n_post: int
    n_boot: int


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


def _placebo_p_value(placebo_gaps: pd.Series, treated_gap: float) -> float:
    n_extreme = int((np.abs(placebo_gaps) >= abs(treated_gap)).sum())
    return float((n_extreme + 1) / (len(placebo_gaps) + 1))


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
    return PlaceboResult(
        placebo_gaps_kw=placebo_gaps.sort_values(),
        treated_gap_kw=treated_gap,
        p_value=_placebo_p_value(placebo_gaps, treated_gap),
    )


def in_time_placebos(
    treated: pd.Series,
    donors: pd.DataFrame,
    intervention_day: int,
    fake_days: list[int] | None = None,
    min_pre_days: int = 15,
    min_placebo_post_days: int = 5,
    max_placebos: int = 8,
) -> PlaceboResult:
    """Run in-time placebos on fake dates that fall entirely in the true pre-period.

    Each fake date is treated as an intervention using only days before the real
    tariff, so the placebo 'post' window is known to be untreated.
    """
    pre_treated = treated.loc[treated.index < intervention_day]
    pre_donors = donors.loc[donors.index < intervention_day]
    if pre_treated.empty or pre_donors.empty:
        raise ValueError("No pre-period observations for in-time placebos")

    if fake_days is None:
        candidates = [
            int(day)
            for day in pre_treated.index
            if day >= min_pre_days and (intervention_day - int(day)) >= min_placebo_post_days
        ]
        if len(candidates) > max_placebos:
            step = max(1, len(candidates) // max_placebos)
            fake_days = candidates[::step][:max_placebos]
        else:
            fake_days = candidates

    if not fake_days:
        raise ValueError("No valid fake intervention dates for in-time placebos")

    gaps: dict[int, float] = {}
    for fake_day in fake_days:
        try:
            result = fit_synthetic_control(
                pre_treated,
                pre_donors,
                intervention_day=int(fake_day),
            )
        except (RuntimeError, ValueError):
            continue
        gaps[int(fake_day)] = result.att_kw

    if not gaps:
        raise ValueError("No successful in-time placebo fits")

    placebo_gaps = pd.Series(gaps, name="placebo_gap_kw")
    treated_result = fit_synthetic_control(treated, donors, intervention_day=intervention_day)
    treated_gap = treated_result.att_kw
    return PlaceboResult(
        placebo_gaps_kw=placebo_gaps.sort_index(),
        treated_gap_kw=treated_gap,
        p_value=_placebo_p_value(placebo_gaps, treated_gap),
    )


def gap_uncertainty(
    result: SyntheticControlResult,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> GapUncertainty:
    """Bootstrap the post-period mean gap, holding synthetic weights fixed."""
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")

    post_mask = result.treated.index >= result.intervention_day
    gaps = (result.treated.loc[post_mask] - result.synthetic.loc[post_mask]).to_numpy()
    if gaps.size == 0:
        raise ValueError("Post-period is empty; cannot compute gap uncertainty")

    att_kw = float(np.mean(gaps))
    std_error = float(np.std(gaps, ddof=1) / np.sqrt(gaps.size)) if gaps.size > 1 else 0.0
    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [float(np.mean(rng.choice(gaps, size=gaps.size, replace=True))) for _ in range(n_boot)]
    )
    ci_low = float(np.quantile(boot_means, alpha / 2.0))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return GapUncertainty(
        att_kw=att_kw,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        n_post=int(gaps.size),
        n_boot=n_boot,
    )


def donor_pool_sensitivity(
    treated: pd.Series,
    donors: pd.DataFrame,
    intervention_day: int,
    drop_counts: tuple[int, ...] = (0, 5, 10, 15),
    n_draws: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Refit SC after dropping random donor subsets; returns one row per draw."""
    n_donors = donors.shape[1]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for drop_count in drop_counts:
        if drop_count < 0 or drop_count >= n_donors:
            continue
        keep = n_donors - drop_count
        n_iter = 1 if drop_count == 0 else n_draws
        for draw in range(n_iter):
            if drop_count == 0:
                subset = donors
            else:
                keep_ids = rng.choice(donors.columns, size=keep, replace=False)
                subset = donors.loc[:, keep_ids]
            try:
                result = fit_synthetic_control(treated, subset, intervention_day=intervention_day)
            except (RuntimeError, ValueError):
                continue
            rows.append(
                {
                    "drop_count": int(drop_count),
                    "n_donors": int(subset.shape[1]),
                    "draw": int(draw),
                    "att_kw": result.att_kw,
                    "att_pct": result.att_pct,
                    "pre_rmspe": result.pre_rmspe,
                }
            )

    if not rows:
        raise ValueError("No successful donor-pool sensitivity fits")
    return pd.DataFrame(rows)


def pre_period_sensitivity(
    treated: pd.Series,
    donors: pd.DataFrame,
    intervention_day: int,
    pre_lengths: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Refit SC using only the last ``pre_length`` pre-period days."""
    n_pre = int((treated.index < intervention_day).sum())
    if n_pre < 2:
        raise ValueError("Need at least 2 pre-period days for pre-period sensitivity")

    if pre_lengths is None:
        grid = (20, 30, 40, 50, 60)
        pre_lengths = tuple(length for length in grid if 5 <= length <= n_pre)
        if not pre_lengths:
            pre_lengths = (n_pre,)

    rows: list[dict[str, float | int]] = []
    for pre_length in pre_lengths:
        start_day = intervention_day - int(pre_length)
        y = treated.loc[treated.index >= start_day]
        x = donors.loc[donors.index >= start_day]
        try:
            result = fit_synthetic_control(y, x, intervention_day=intervention_day)
        except (RuntimeError, ValueError):
            continue
        rows.append(
            {
                "pre_length": int(pre_length),
                "att_kw": result.att_kw,
                "att_pct": result.att_pct,
                "pre_rmspe": result.pre_rmspe,
            }
        )

    if not rows:
        raise ValueError("No successful pre-period sensitivity fits")
    return pd.DataFrame(rows)
