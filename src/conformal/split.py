"""Chronological train / calibration / test splits for time series."""

from __future__ import annotations

import pandas as pd


def chronological_conformal_split(
    frame: pd.DataFrame,
    *,
    time_col: str = "valid_time",
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
    gap_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered frame into train, calibration, and test blocks.

    Rows are sorted by ``time_col``. Optional ``gap_hours`` drops rows between
    blocks so calibration and test do not abut training data (day-ahead mimic).

    Args:
        frame: Input table with a monotonic timestamp column.
        time_col: Timestamp column name.
        train_frac: Fraction of rows for training (earliest).
        cal_frac: Fraction for conformal calibration (middle).
        test_frac: Fraction for held-out evaluation (latest).
        gap_hours: Hours to skip between consecutive blocks.

    Returns:
        ``(train_df, cal_df, test_df)`` — disjoint, chronological subsets.
    """
    if abs(train_frac + cal_frac + test_frac - 1.0) > 1e-9:
        msg = "train_frac + cal_frac + test_frac must sum to 1.0"
        raise ValueError(msg)
    if time_col not in frame.columns:
        msg = f"time column {time_col!r} not in frame"
        raise KeyError(msg)
    if gap_hours < 0:
        msg = f"gap_hours must be >= 0, got {gap_hours}"
        raise ValueError(msg)

    work = frame.sort_values(time_col).reset_index(drop=True)
    n = len(work)
    if n < 30:
        msg = f"need at least 30 rows for a conformal split, got {n}"
        raise ValueError(msg)

    n_train = max(1, int(n * train_frac))
    n_cal = max(1, int(n * cal_frac))
    n_test = max(1, n - n_train - n_cal)

    # Adjust if rounding left too few test rows
    if n_train + n_cal + n_test > n:
        n_test = max(1, n - n_train - n_cal)

    train_end_idx = n_train
    cal_start_idx = train_end_idx
    cal_end_idx = cal_start_idx + n_cal
    test_start_idx = cal_end_idx

    train_df = work.iloc[:train_end_idx].copy()
    cal_df = work.iloc[cal_start_idx:cal_end_idx].copy()
    test_df = work.iloc[test_start_idx:].copy()

    if gap_hours > 0:
        gap = pd.Timedelta(hours=gap_hours)
        if not train_df.empty and not cal_df.empty:
            cutoff = train_df[time_col].max() + gap
            cal_df = cal_df[cal_df[time_col] >= cutoff].copy()
        if not cal_df.empty and not test_df.empty:
            cutoff = cal_df[time_col].max() + gap
            test_df = test_df[test_df[time_col] >= cutoff].copy()
        elif not train_df.empty and not test_df.empty and cal_df.empty:
            cutoff = train_df[time_col].max() + gap
            test_df = test_df[test_df[time_col] >= cutoff].copy()

    for name, part in ("train", train_df), ("calibration", cal_df), ("test", test_df):
        if part.empty:
            msg = f"{name} split is empty after applying gap_hours={gap_hours}"
            raise ValueError(msg)

    return train_df, cal_df, test_df
