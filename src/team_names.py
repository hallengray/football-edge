"""Three-way team-name registry for Big-5 European leagues.

Each team has a canonical internal name plus aliases for the three external
data sources we touch:
- The Odds API (`odds_api`) — long club names like "Wolverhampton Wanderers"
- football-data.co.uk (`football_data`) — short names like "Wolves"
- Understat (`understat`) — typically the long form, e.g. "Wolverhampton Wanderers"

`to_canonical(name, source)` and `from_canonical(name, target)` handle conversions.
Unmapped names raise a clear KeyError that names the team and its source so the
operator knows exactly what entry to add.

The registry is seeded with current EPL teams and prominent clubs from the four
other Big-5 leagues. Additional teams are added during the smoke test (Task 16)
by running training and adding any KeyError that surfaces. Self-healing pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_SOURCES = ("odds_api", "football_data", "understat")


@dataclass(frozen=True)
class TeamNames:
    """Aliases for a single team across the data sources we use."""

    canonical: str
    odds_api: str
    football_data: str
    understat: str


def _entry(canonical: str, odds_api: str, football_data: str, understat: str) -> TeamNames:
    return TeamNames(
        canonical=canonical,
        odds_api=odds_api,
        football_data=football_data,
        understat=understat,
    )


REGISTRY: dict[str, TeamNames] = {
    # ─── EPL (current season) ─────────────────────────────────────────
    "Arsenal": _entry("Arsenal", "Arsenal", "Arsenal", "Arsenal"),
    "Aston Villa": _entry("Aston Villa", "Aston Villa", "Aston Villa", "Aston Villa"),
    "Bournemouth": _entry("Bournemouth", "AFC Bournemouth", "Bournemouth", "Bournemouth"),
    "Brentford": _entry("Brentford", "Brentford", "Brentford", "Brentford"),
    "Brighton": _entry("Brighton", "Brighton and Hove Albion", "Brighton", "Brighton"),
    "Burnley": _entry("Burnley", "Burnley", "Burnley", "Burnley"),
    "Chelsea": _entry("Chelsea", "Chelsea", "Chelsea", "Chelsea"),
    "Crystal Palace": _entry(
        "Crystal Palace", "Crystal Palace", "Crystal Palace", "Crystal Palace"
    ),
    "Everton": _entry("Everton", "Everton", "Everton", "Everton"),
    "Fulham": _entry("Fulham", "Fulham", "Fulham", "Fulham"),
    "Leeds": _entry("Leeds", "Leeds United", "Leeds", "Leeds"),
    "Liverpool": _entry("Liverpool", "Liverpool", "Liverpool", "Liverpool"),
    "Man City": _entry("Man City", "Manchester City", "Man City", "Manchester City"),
    "Manchester United": _entry(
        "Manchester United", "Manchester United", "Man United", "Manchester United"
    ),
    "Newcastle": _entry("Newcastle", "Newcastle United", "Newcastle", "Newcastle United"),
    "Nottingham Forest": _entry(
        "Nottingham Forest", "Nottingham Forest", "Nottingham Forest", "Nottingham Forest"
    ),
    "Sunderland": _entry("Sunderland", "Sunderland", "Sunderland", "Sunderland"),
    "Tottenham": _entry("Tottenham", "Tottenham Hotspur", "Tottenham", "Tottenham"),
    "West Ham": _entry("West Ham", "West Ham United", "West Ham", "West Ham"),
    "Wolves": _entry("Wolves", "Wolverhampton Wanderers", "Wolves", "Wolverhampton Wanderers"),
    # ─── La Liga (top clubs — seed; remainder added during smoke test) ──
    "Real Madrid": _entry("Real Madrid", "Real Madrid", "Real Madrid", "Real Madrid"),
    "Barcelona": _entry("Barcelona", "Barcelona", "Barcelona", "Barcelona"),
    "Atletico Madrid": _entry(
        "Atletico Madrid", "Atletico Madrid", "Ath Madrid", "Atletico Madrid"
    ),
    "Sevilla": _entry("Sevilla", "Sevilla", "Sevilla", "Sevilla"),
    "Real Sociedad": _entry("Real Sociedad", "Real Sociedad", "Sociedad", "Real Sociedad"),
    "Villarreal": _entry("Villarreal", "Villarreal", "Villarreal", "Villarreal"),
    "Athletic Bilbao": _entry("Athletic Bilbao", "Athletic Bilbao", "Ath Bilbao", "Athletic Club"),
    "Real Betis": _entry("Real Betis", "Real Betis", "Betis", "Real Betis"),
    "Valencia": _entry("Valencia", "Valencia", "Valencia", "Valencia"),
    # ─── Serie A (top clubs — seed) ─────────────────────────────────────
    "Inter": _entry("Inter", "Inter Milan", "Inter", "Internazionale"),
    "Milan": _entry("Milan", "AC Milan", "Milan", "Milan"),
    "Juventus": _entry("Juventus", "Juventus", "Juventus", "Juventus"),
    "Napoli": _entry("Napoli", "Napoli", "Napoli", "Napoli"),
    "Roma": _entry("Roma", "AS Roma", "Roma", "Roma"),
    "Lazio": _entry("Lazio", "Lazio", "Lazio", "Lazio"),
    "Atalanta": _entry("Atalanta", "Atalanta", "Atalanta", "Atalanta"),
    "Fiorentina": _entry("Fiorentina", "Fiorentina", "Fiorentina", "Fiorentina"),
    # ─── Bundesliga (top clubs — seed) ─────────────────────────────────
    "Bayern Munich": _entry("Bayern Munich", "Bayern Munich", "Bayern Munich", "Bayern Munich"),
    "Borussia Dortmund": _entry(
        "Borussia Dortmund", "Borussia Dortmund", "Dortmund", "Borussia Dortmund"
    ),
    "Bayer Leverkusen": _entry(
        "Bayer Leverkusen", "Bayer Leverkusen", "Leverkusen", "Bayer Leverkusen"
    ),
    "RB Leipzig": _entry("RB Leipzig", "RB Leipzig", "RB Leipzig", "RasenBallsport Leipzig"),
    "Eintracht Frankfurt": _entry(
        "Eintracht Frankfurt", "Eintracht Frankfurt", "Ein Frankfurt", "Eintracht Frankfurt"
    ),
    "Wolfsburg": _entry("Wolfsburg", "VfL Wolfsburg", "Wolfsburg", "Wolfsburg"),
    # ─── Ligue 1 (top clubs — seed) ────────────────────────────────────
    "Paris Saint-Germain": _entry(
        "Paris Saint-Germain", "Paris Saint-Germain", "Paris SG", "Paris Saint Germain"
    ),
    "Marseille": _entry("Marseille", "Marseille", "Marseille", "Marseille"),
    "Monaco": _entry("Monaco", "Monaco", "Monaco", "Monaco"),
    "Lyon": _entry("Lyon", "Lyon", "Lyon", "Lyon"),
    "Lille": _entry("Lille", "Lille", "Lille", "Lille"),
    "Nice": _entry("Nice", "Nice", "Nice", "Nice"),
    "Rennes": _entry("Rennes", "Rennes", "Rennes", "Rennes"),
}

# Reverse-index for fast `to_canonical` lookups
_BY_ODDS_API: dict[str, str] = {t.odds_api: c for c, t in REGISTRY.items()}
_BY_FOOTBALL_DATA: dict[str, str] = {t.football_data: c for c, t in REGISTRY.items()}
_BY_UNDERSTAT: dict[str, str] = {t.understat: c for c, t in REGISTRY.items()}

_REVERSE_BY_SOURCE: dict[str, dict[str, str]] = {
    "odds_api": _BY_ODDS_API,
    "football_data": _BY_FOOTBALL_DATA,
    "understat": _BY_UNDERSTAT,
}


def to_canonical(name: str, source: str) -> str:
    """Convert a source-specific team name to its canonical form.

    Raises KeyError with an actionable message naming the unmapped team and the
    source it was looked up from. Add the entry to REGISTRY to fix.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source {source!r}. Expected one of {VALID_SOURCES}.")
    reverse = _REVERSE_BY_SOURCE[source]
    if name not in reverse:
        raise KeyError(
            f"Unmapped team {name!r} from source {source!r}. "
            f"Add an entry to src/team_names.REGISTRY."
        )
    return reverse[name]


def from_canonical(canonical: str, target: str) -> str:
    """Convert a canonical name to its form for a specific source."""
    if target not in VALID_SOURCES:
        raise ValueError(f"Invalid target {target!r}. Expected one of {VALID_SOURCES}.")
    if canonical not in REGISTRY:
        raise KeyError(
            f"Unmapped canonical name {canonical!r}. Add an entry to src/team_names.REGISTRY."
        )
    return getattr(REGISTRY[canonical], target)
