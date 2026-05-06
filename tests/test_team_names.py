"""Tests for src/team_names.py 3-way registry."""

from __future__ import annotations

import pytest

from src.team_names import (
    REGISTRY,
    TeamNames,
    from_canonical,
    to_canonical,
)


def test_team_names_dataclass_holds_three_sources() -> None:
    t = TeamNames(
        canonical="Arsenal",
        odds_api="Arsenal",
        football_data="Arsenal",
        understat="Arsenal",
    )
    assert t.canonical == "Arsenal"
    assert t.odds_api == "Arsenal"
    assert t.football_data == "Arsenal"
    assert t.understat == "Arsenal"


def test_to_canonical_from_odds_api() -> None:
    # The Odds API uses the long form
    assert to_canonical("Wolverhampton Wanderers", source="odds_api") == "Wolves"


def test_to_canonical_from_football_data() -> None:
    # football-data uses short form
    assert to_canonical("Man United", source="football_data") == "Manchester United"


def test_to_canonical_from_understat() -> None:
    assert to_canonical("Manchester United", source="understat") == "Manchester United"


def test_to_canonical_unknown_raises_keyerror_with_actionable_message() -> None:
    with pytest.raises(KeyError, match="Unmapped"):
        to_canonical("FC Made Up", source="odds_api")


def test_to_canonical_invalid_source_raises() -> None:
    with pytest.raises(ValueError, match="source"):
        to_canonical("Arsenal", source="bogus_source")


def test_from_canonical_round_trip_to_each_source() -> None:
    # Pick a team known to differ across sources
    assert from_canonical("Wolves", target="odds_api") == "Wolverhampton Wanderers"
    assert from_canonical("Wolves", target="football_data") == "Wolves"
    assert from_canonical("Wolves", target="understat") == "Wolverhampton Wanderers"


def test_registry_contains_all_seed_epl_teams() -> None:
    expected_epl = {
        "Arsenal",
        "Aston Villa",
        "Bournemouth",
        "Brentford",
        "Brighton",
        "Burnley",
        "Chelsea",
        "Crystal Palace",
        "Everton",
        "Fulham",
        "Leeds",
        "Liverpool",
        "Man City",
        "Manchester United",
        "Newcastle",
        "Nottingham Forest",
        "Sunderland",
        "Tottenham",
        "West Ham",
        "Wolves",
    }
    canonical_set = set(REGISTRY.keys())
    missing = expected_epl - canonical_set
    assert not missing, f"Seed EPL registry missing: {missing}"


def test_registry_seed_includes_top_clubs_from_other_leagues() -> None:
    # Spot-check a few prominent non-EPL clubs we know must be in the seed
    expected_seed = {
        "Real Madrid",
        "Barcelona",
        "Atletico Madrid",  # La Liga
        "Inter",
        "Juventus",
        "Milan",
        "Napoli",  # Serie A
        "Bayern Munich",
        "Borussia Dortmund",
        "Bayer Leverkusen",  # Bundesliga
        "Paris Saint-Germain",
        "Marseille",
        "Lyon",  # Ligue 1
    }
    missing = expected_seed - set(REGISTRY.keys())
    assert not missing, f"Seed non-EPL registry missing: {missing}"
