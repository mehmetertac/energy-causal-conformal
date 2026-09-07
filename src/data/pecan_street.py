"""Pecan Street Dataport loader stub."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PECAN_STREET_PORTAL = "https://www.pecanstreet.org/dataport/"


class PecanStreetNotAvailableError(RuntimeError):
    """Raised when Pecan Street data has not been downloaded locally."""


def load_pecan_street(path: str | Path | None = None) -> pd.DataFrame:
    """Load Pecan Street residential load once credentials and files are available.

    Expected schema after download:
    - ``dataid``: household identifier
    - ``localminute``: timestamp
    - ``use``: kWh over the interval (typically 1 minute or 15 minutes)
    """
    if path is None:
        raise PecanStreetNotAvailableError(
            "Pecan Street Dataport access requires registration. "
            f"Register at {PECAN_STREET_PORTAL}, download CSVs to data/raw/pecan/, "
            "then call load_pecan_street(path=...). "
            "Until then, use src.data.london_smartmeter.load_lcl_sample()."
        )

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Pecan Street file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"dataid", "localminute", "use"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Pecan Street file missing columns: {sorted(missing)}")

    frame = frame.rename(
        columns={
            "dataid": "household_id",
            "localminute": "timestamp",
            "use": "kwh",
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["kwh"] = pd.to_numeric(frame["kwh"], errors="coerce")
    frame = frame.dropna(subset=["household_id", "timestamp", "kwh"])
    return frame.sort_values(["household_id", "timestamp"]).reset_index(drop=True)
