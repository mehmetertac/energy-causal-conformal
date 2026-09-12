"""Empirical coverage metrics for prediction intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoverageResult:
    """Interval calibration summary on a test slice."""

    coverage: float
    nominal: float
    coverage_gap: float
    mean_width: float
    n_obs: int


def pi_coverage(
    y_true: np.ndarray | pd.Series,
    p_lo: np.ndarray | pd.Series,
    p_hi: np.ndarray | pd.Series,
) -> float:
    """Fraction of observations inside [p_lo, p_hi]."""
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(p_lo, dtype=float)
    hi = np.asarray(p_hi, dtype=float)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean((yt >= lo) & (yt <= hi)))


def mean_interval_width(
    p_lo: np.ndarray | pd.Series,
    p_hi: np.ndarray | pd.Series,
) -> float:
    """Average upper-minus-lower interval width."""
    lo = np.asarray(p_lo, dtype=float)
    hi = np.asarray(p_hi, dtype=float)
    if len(lo) == 0:
        return float("nan")
    return float(np.mean(hi - lo))


def evaluate_intervals(
    y_true: np.ndarray | pd.Series,
    p_lo: np.ndarray | pd.Series,
    p_hi: np.ndarray | pd.Series,
    *,
    nominal: float,
) -> CoverageResult:
    """Compute empirical coverage, gap vs nominal, and mean width."""
    yt = np.asarray(y_true, dtype=float)
    observed = pi_coverage(yt, p_lo, p_hi)
    return CoverageResult(
        coverage=observed,
        nominal=float(nominal),
        coverage_gap=observed - float(nominal),
        mean_width=mean_interval_width(p_lo, p_hi),
        n_obs=int(len(yt)),
    )


def pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    quantile: float,
) -> float:
    """Mean pinball (quantile) loss at ``quantile`` in (0, 1)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")
    err = yt - yp
    q = float(quantile)
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))
