"""Propensity-weighted ATT cross-check for household-level peak-load changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PropensityResult:
    """Household-level inverse-propensity weighted ATT on the pre/post peak change."""

    att_kw: float
    n_treated: int
    n_control: int
    method: str
    min_propensity: float
    max_propensity: float


def household_peak_deltas(
    frame: pd.DataFrame,
    peak_hours: tuple[int, ...] = (17, 18, 19, 20),
    load_col: str = "load_kw",
) -> pd.DataFrame:
    """One row per household: pre-period mean peak, post-period mean peak, and delta."""
    required = {"household_id", "treated", "post", "hour", load_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    peak = frame.loc[frame["hour"].isin(peak_hours)].copy()
    if peak.empty:
        raise ValueError("No rows match peak_hours filter")

    grouped = (
        peak.groupby(["household_id", "treated", "post"], as_index=False)[load_col]
        .mean()
        .rename(columns={load_col: "peak_kw"})
    )
    wide = grouped.pivot_table(
        index=["household_id", "treated"],
        columns="post",
        values="peak_kw",
    )
    if 0 not in wide.columns or 1 not in wide.columns:
        raise ValueError("Need both pre (post=0) and post (post=1) peak means")

    out = wide.rename(columns={0: "pre_peak_kw", 1: "post_peak_kw"}).reset_index()
    out["delta_peak_kw"] = out["post_peak_kw"] - out["pre_peak_kw"]
    out["treated"] = out["treated"].astype(int)
    return out.dropna(subset=["pre_peak_kw", "post_peak_kw", "delta_peak_kw"])


def estimate_ipw_att(
    frame: pd.DataFrame,
    peak_hours: tuple[int, ...] = (17, 18, 19, 20),
    load_col: str = "load_kw",
    clip: tuple[float, float] = (0.05, 0.95),
) -> PropensityResult:
    """IPW ATT of the household peak-load change, propensity from pre-period peak.

    Treatment in the simulator is as-if random, so this is a method sketch rather
    than a necessary adjustment. On real opt-in tariffs, pre-period usage is a
    typical confounder for enrolment.
    """
    households = household_peak_deltas(frame, peak_hours=peak_hours, load_col=load_col)
    treated = households["treated"].to_numpy()
    if treated.min() == treated.max():
        raise ValueError("IPW requires both treated and control households")

    X = households[["pre_peak_kw"]].to_numpy()
    model = LogisticRegression(max_iter=500)
    model.fit(X, treated)
    propensity = np.clip(model.predict_proba(X)[:, 1], clip[0], clip[1])

    y = households["delta_peak_kw"].to_numpy()
    weights_t = treated / propensity
    weights_c = (1 - treated) / (1.0 - propensity)
    mu1 = float(np.sum(weights_t * y) / np.sum(weights_t))
    mu0 = float(np.sum(weights_c * y) / np.sum(weights_c))
    return PropensityResult(
        att_kw=mu1 - mu0,
        n_treated=int(treated.sum()),
        n_control=int((1 - treated).sum()),
        method="ipw_logistic_pre_peak",
        min_propensity=float(propensity.min()),
        max_propensity=float(propensity.max()),
    )


def estimate_dowhy_att(
    frame: pd.DataFrame,
    peak_hours: tuple[int, ...] = (17, 18, 19, 20),
    load_col: str = "load_kw",
) -> PropensityResult:
    """DoWhy backdoor propensity-score weighting on household peak-load changes."""
    try:
        from dowhy import CausalModel
    except ImportError as exc:  # pragma: no cover - optional path
        raise ImportError("DoWhy is required for estimate_dowhy_att") from exc

    households = household_peak_deltas(frame, peak_hours=peak_hours, load_col=load_col)
    model = CausalModel(
        data=households,
        treatment="treated",
        outcome="delta_peak_kw",
        common_causes=["pre_peak_kw"],
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.propensity_score_weighting",
        target_units="ate",
    )
    att = float(estimate.value)
    treated = households["treated"].to_numpy()
    return PropensityResult(
        att_kw=att,
        n_treated=int(treated.sum()),
        n_control=int((1 - treated).sum()),
        method="dowhy_ps_weighting",
        min_propensity=float("nan"),
        max_propensity=float("nan"),
    )
