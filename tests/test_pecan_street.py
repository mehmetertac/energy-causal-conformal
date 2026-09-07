"""Tests for the Pecan Street stub."""

from __future__ import annotations

import pytest

from src.data.pecan_street import PecanStreetNotAvailableError, load_pecan_street


def test_pecan_street_stub_without_path() -> None:
    with pytest.raises(PecanStreetNotAvailableError, match="Dataport"):
        load_pecan_street()
