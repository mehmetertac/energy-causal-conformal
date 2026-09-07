"""Tests for the Low Carbon London loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.london_smartmeter import FIXTURE_PATH, load_lcl_csv, load_lcl_sample


def test_load_lcl_sample_fixture() -> None:
    frame = load_lcl_sample()
    assert len(frame) == 6
    assert frame["household_id"].nunique() == 3
    assert frame["load_kw"].notna().all()


def test_load_lcl_csv_requires_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_lcl_csv(bad_csv)


def test_fixture_path_exists() -> None:
    assert FIXTURE_PATH.exists()
