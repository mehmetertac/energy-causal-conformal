"""Low Carbon London smart-meter loader (Pecan Street substitute)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "lcl"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "lcl_sample.csv"

# London Datastore resource links (sample + tariff metadata).
LCL_SAMPLE_URL = (
    "https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/"
    "low-carbon-london-data-168-files/ccby4version/"
    "block_0.csv"
)
LCL_TARIFF_URL = (
    "https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/"
    "tariffs/ccby4version/tariffs.csv"
)

COLUMN_ALIASES = {
    "LCLid": "household_id",
    "Date": "timestamp",
    "kWh/hh": "kwh_halfhour",
    "kwh/hh": "kwh_halfhour",
}


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns=COLUMN_ALIASES)
    if "timestamp" in renamed.columns:
        renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], errors="coerce")
    if "kwh_halfhour" in renamed.columns:
        renamed["kwh_halfhour"] = pd.to_numeric(renamed["kwh_halfhour"], errors="coerce")
    return renamed


def load_lcl_csv(path: str | Path) -> pd.DataFrame:
    """Load one LCL CSV file with normalized column names."""
    frame = pd.read_csv(path)
    frame = _normalize_columns(frame)
    required = {"household_id", "timestamp", "kwh_halfhour"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LCL file missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["household_id", "timestamp", "kwh_halfhour"])
    frame["load_kw"] = frame["kwh_halfhour"] * 2.0
    return frame.sort_values(["household_id", "timestamp"]).reset_index(drop=True)


def load_lcl_sample(path: str | Path | None = None) -> pd.DataFrame:
    """Load the committed tiny LCL-shaped fixture used in tests."""
    return load_lcl_csv(path or FIXTURE_PATH)


MANUAL_DOWNLOAD_PAGE = (
    "https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/"
)


def download_file(url: str, destination: Path, timeout: int = 120) -> Path:
    """Download a remote LCL asset to ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _raise_manual_download_error(resource: str, exc: Exception) -> None:
    raise RuntimeError(
        f"Automated LCL download for '{resource}' failed. London Datastore often "
        f"requires manual download of the zip archive from {MANUAL_DOWNLOAD_PAGE} "
        "(resource: low-carbon-london-data-168-files or Tariffs). "
        "Extract CSVs to data/raw/lcl/. CI uses tests/fixtures/lcl_sample.csv."
    ) from exc


def download_lcl_sample(raw_dir: str | Path | None = None) -> Path:
    """Download one split CSV block from London Datastore.

    If the direct URL fails, download the 168-file zip manually and extract
    ``block_0.csv`` into ``data/raw/lcl/``.
    """
    raw_path = Path(raw_dir or DEFAULT_RAW_DIR)
    destination = raw_path / "block_0.csv"
    if destination.exists():
        return destination
    try:
        return download_file(LCL_SAMPLE_URL, destination)
    except requests.RequestException as exc:
        _raise_manual_download_error("block_0.csv", exc)


def download_lcl_tariffs(raw_dir: str | Path | None = None) -> Path:
    """Download the LCL tariff schedule CSV."""
    raw_path = Path(raw_dir or DEFAULT_RAW_DIR)
    destination = raw_path / "tariffs.csv"
    if destination.exists():
        return destination
    try:
        return download_file(LCL_TARIFF_URL, destination)
    except requests.RequestException as exc:
        _raise_manual_download_error("tariffs.csv", exc)


def list_households(frame: pd.DataFrame) -> Iterable[str]:
    """Return sorted household identifiers."""
    return sorted(frame["household_id"].astype(str).unique())
