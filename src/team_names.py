"""Map between Odds API team names (long form) and sports-betting library team names (short form).

The library uses football-data.co.uk-style abbreviations ("Wolves", "Brighton")
while The Odds API uses full club names ("Wolverhampton Wanderers", "Brighton and Hove Albion").

This map is hardcoded for the 20 teams in the current EPL season. After August's
promotion/relegation cycle, three teams change — update the map and the
test_map_has_twenty_teams test will pass once you've added/removed the right entries.

If a team appears in The Odds API that isn't in this map, `to_library()` raises a clear
KeyError naming the unmapped team. The dashboard catches this and falls back to demo
mode for that fixture only — so missing entries are self-diagnosing without crashing.
"""

from __future__ import annotations

ODDS_API_TO_LIBRARY: dict[str, str] = {
    # Verified at impl time against library data — adjust right-hand values
    # if Task 9's smoke run reveals different naming.
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds United": "Leeds",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham Forest",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

_LIBRARY_TO_ODDS_API: dict[str, str] = {v: k for k, v in ODDS_API_TO_LIBRARY.items()}


def to_library(odds_api_name: str) -> str:
    if odds_api_name not in ODDS_API_TO_LIBRARY:
        raise KeyError(
            f"Unmapped Odds API team: {odds_api_name!r}. Add to src/team_names.ODDS_API_TO_LIBRARY."
        )
    return ODDS_API_TO_LIBRARY[odds_api_name]


def from_library(library_name: str) -> str:
    if library_name not in _LIBRARY_TO_ODDS_API:
        raise KeyError(
            f"Unmapped library team: {library_name!r}. "
            "Add to src/team_names.ODDS_API_TO_LIBRARY (forward map)."
        )
    return _LIBRARY_TO_ODDS_API[library_name]
