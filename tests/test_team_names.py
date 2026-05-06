"""Tests for src/team_names.py mapping."""

from __future__ import annotations

import pytest

from src.team_names import (
    ODDS_API_TO_LIBRARY,
    from_library,
    to_library,
)


def test_to_library_known_team() -> None:
    assert to_library("Arsenal") == "Arsenal"
    assert to_library("Wolverhampton Wanderers") == "Wolves"


def test_to_library_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unmapped Odds API team"):
        to_library("Real Madrid")  # not in EPL


def test_from_library_round_trip() -> None:
    # Every Odds API name should round-trip via from_library
    for odds_name, lib_name in ODDS_API_TO_LIBRARY.items():
        assert from_library(lib_name) == odds_name


def test_from_library_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unmapped library team"):
        from_library("Barcelona")


def test_map_has_twenty_teams() -> None:
    # EPL has 20 teams. If this assertion fails after relegation/promotion,
    # update ODDS_API_TO_LIBRARY accordingly.
    assert len(ODDS_API_TO_LIBRARY) == 20
