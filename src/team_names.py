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
        "Nottingham Forest", "Nottingham Forest", "Nott'm Forest", "Nottingham Forest"
    ),
    "Sunderland": _entry("Sunderland", "Sunderland", "Sunderland", "Sunderland"),
    "Tottenham": _entry("Tottenham", "Tottenham Hotspur", "Tottenham", "Tottenham"),
    "West Ham": _entry("West Ham", "West Ham United", "West Ham", "West Ham"),
    "Wolves": _entry("Wolves", "Wolverhampton Wanderers", "Wolves", "Wolverhampton Wanderers"),
    # ─── EPL (relegated / historical, 2018-2025 seasons) ─────────────────
    "Cardiff": _entry("Cardiff", "Cardiff City", "Cardiff", "Cardiff"),
    "Huddersfield": _entry("Huddersfield", "Huddersfield Town", "Huddersfield", "Huddersfield"),
    "Ipswich": _entry("Ipswich", "Ipswich Town", "Ipswich", "Ipswich"),
    "Leicester": _entry("Leicester", "Leicester City", "Leicester", "Leicester"),
    "Luton": _entry("Luton", "Luton Town", "Luton", "Luton"),
    "Norwich": _entry("Norwich", "Norwich City", "Norwich", "Norwich"),
    "Sheffield United": _entry(
        "Sheffield United", "Sheffield United", "Sheffield United", "Sheffield United"
    ),
    "Southampton": _entry("Southampton", "Southampton", "Southampton", "Southampton"),
    "Stoke": _entry("Stoke", "Stoke City", "Stoke", "Stoke"),
    "Swansea": _entry("Swansea", "Swansea City", "Swansea", "Swansea"),
    "Watford": _entry("Watford", "Watford", "Watford", "Watford"),
    "West Brom": _entry("West Brom", "West Bromwich Albion", "West Brom", "West Bromwich Albion"),
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
    "Alaves": _entry("Alaves", "Alaves", "Alaves", "Alaves"),
    "Almeria": _entry("Almeria", "Almeria", "Almeria", "Almeria"),
    "Cadiz": _entry("Cadiz", "Cadiz", "Cadiz", "Cadiz"),
    "Celta Vigo": _entry("Celta Vigo", "Celta Vigo", "Celta", "Celta Vigo"),
    "Eibar": _entry("Eibar", "Eibar", "Eibar", "Eibar"),
    "Elche": _entry("Elche", "Elche", "Elche", "Elche"),
    "Espanyol": _entry("Espanyol", "Espanyol", "Espanol", "Espanyol"),
    "Getafe": _entry("Getafe", "Getafe", "Getafe", "Getafe"),
    "Girona": _entry("Girona", "Girona", "Girona", "Girona"),
    "Granada": _entry("Granada", "Granada", "Granada", "Granada"),
    "Huesca": _entry("Huesca", "Huesca", "Huesca", "SD Huesca"),
    "Deportivo La Coruna": _entry(
        "Deportivo La Coruna", "Deportivo La Coruna", "La Coruna", "Deportivo La Coruna"
    ),
    "Las Palmas": _entry("Las Palmas", "Las Palmas", "Las Palmas", "Las Palmas"),
    "Leganes": _entry("Leganes", "Leganes", "Leganes", "Leganes"),
    "Levante": _entry("Levante", "Levante", "Levante", "Levante"),
    "Malaga": _entry("Malaga", "Malaga", "Malaga", "Malaga"),
    "Mallorca": _entry("Mallorca", "Mallorca", "Mallorca", "Mallorca"),
    "Osasuna": _entry("Osasuna", "Osasuna", "Osasuna", "Osasuna"),
    "Real Oviedo": _entry("Real Oviedo", "Real Oviedo", "Oviedo", "Real Oviedo"),
    "Valladolid": _entry("Valladolid", "Real Valladolid", "Valladolid", "Real Valladolid"),
    "Rayo Vallecano": _entry("Rayo Vallecano", "Rayo Vallecano", "Vallecano", "Rayo Vallecano"),
    # ─── Serie A (top clubs — seed) ─────────────────────────────────────
    "Inter": _entry("Inter", "Inter Milan", "Inter", "Inter"),
    "Milan": _entry("Milan", "AC Milan", "Milan", "AC Milan"),
    "Juventus": _entry("Juventus", "Juventus", "Juventus", "Juventus"),
    "Napoli": _entry("Napoli", "Napoli", "Napoli", "Napoli"),
    "Roma": _entry("Roma", "AS Roma", "Roma", "Roma"),
    "Lazio": _entry("Lazio", "Lazio", "Lazio", "Lazio"),
    "Atalanta": _entry("Atalanta", "Atalanta", "Atalanta", "Atalanta"),
    "Fiorentina": _entry("Fiorentina", "Fiorentina", "Fiorentina", "Fiorentina"),
    "Benevento": _entry("Benevento", "Benevento", "Benevento", "Benevento"),
    "Bologna": _entry("Bologna", "Bologna", "Bologna", "Bologna"),
    "Brescia": _entry("Brescia", "Brescia", "Brescia", "Brescia"),
    "Cagliari": _entry("Cagliari", "Cagliari", "Cagliari", "Cagliari"),
    "Chievo": _entry("Chievo", "Chievo Verona", "Chievo", "Chievo"),
    "Como": _entry("Como", "Como", "Como", "Como"),
    "Cremonese": _entry("Cremonese", "Cremonese", "Cremonese", "Cremonese"),
    "Crotone": _entry("Crotone", "Crotone", "Crotone", "Crotone"),
    "Empoli": _entry("Empoli", "Empoli", "Empoli", "Empoli"),
    "Frosinone": _entry("Frosinone", "Frosinone", "Frosinone", "Frosinone"),
    "Genoa": _entry("Genoa", "Genoa", "Genoa", "Genoa"),
    "Lecce": _entry("Lecce", "Lecce", "Lecce", "Lecce"),
    "Monza": _entry("Monza", "Monza", "Monza", "Monza"),
    "Parma": _entry("Parma", "Parma", "Parma", "Parma Calcio 1913"),
    "Pisa": _entry("Pisa", "Pisa", "Pisa", "Pisa"),
    "Salernitana": _entry("Salernitana", "Salernitana", "Salernitana", "Salernitana"),
    "Sampdoria": _entry("Sampdoria", "Sampdoria", "Sampdoria", "Sampdoria"),
    "Sassuolo": _entry("Sassuolo", "Sassuolo", "Sassuolo", "Sassuolo"),
    "Spal": _entry("Spal", "SPAL", "Spal", "SPAL 2013"),
    "Spezia": _entry("Spezia", "Spezia", "Spezia", "Spezia"),
    "Torino": _entry("Torino", "Torino", "Torino", "Torino"),
    "Udinese": _entry("Udinese", "Udinese", "Udinese", "Udinese"),
    "Venezia": _entry("Venezia", "Venezia", "Venezia", "Venezia"),
    "Verona": _entry("Verona", "Hellas Verona", "Verona", "Verona"),
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
    "Augsburg": _entry("Augsburg", "FC Augsburg", "Augsburg", "Augsburg"),
    "Arminia Bielefeld": _entry(
        "Arminia Bielefeld", "Arminia Bielefeld", "Bielefeld", "Arminia Bielefeld"
    ),
    "Bochum": _entry("Bochum", "VfL Bochum", "Bochum", "Bochum"),
    "Darmstadt": _entry("Darmstadt", "SV Darmstadt 98", "Darmstadt", "Darmstadt"),
    "Fortuna Dusseldorf": _entry(
        "Fortuna Dusseldorf",
        "Fortuna Dusseldorf",
        "Fortuna Dusseldorf",
        "Fortuna Duesseldorf",
    ),
    "FC Koln": _entry("FC Koln", "FC Cologne", "FC Koln", "FC Cologne"),
    "Freiburg": _entry("Freiburg", "SC Freiburg", "Freiburg", "Freiburg"),
    "Greuther Furth": _entry(
        "Greuther Furth", "Greuther Furth", "Greuther Furth", "Greuther Fuerth"
    ),
    "Hamburger SV": _entry("Hamburger SV", "Hamburger SV", "Hamburg", "Hamburger SV"),
    "Hannover 96": _entry("Hannover 96", "Hannover 96", "Hannover", "Hannover 96"),
    "Heidenheim": _entry("Heidenheim", "1. FC Heidenheim", "Heidenheim", "FC Heidenheim"),
    "Hertha Berlin": _entry("Hertha Berlin", "Hertha Berlin", "Hertha", "Hertha Berlin"),
    "Hoffenheim": _entry("Hoffenheim", "TSG Hoffenheim", "Hoffenheim", "Hoffenheim"),
    "Holstein Kiel": _entry("Holstein Kiel", "Holstein Kiel", "Holstein Kiel", "Holstein Kiel"),
    "Mainz 05": _entry("Mainz 05", "FSV Mainz", "Mainz", "Mainz 05"),
    "Borussia Monchengladbach": _entry(
        "Borussia Monchengladbach",
        "Borussia Monchengladbach",
        "M'gladbach",
        "Borussia M.Gladbach",
    ),
    "Nurnberg": _entry("Nurnberg", "Nurnberg", "Nurnberg", "Nuernberg"),
    "Paderborn": _entry("Paderborn", "Paderborn", "Paderborn", "Paderborn"),
    "Schalke 04": _entry("Schalke 04", "FC Schalke 04", "Schalke 04", "Schalke 04"),
    "St Pauli": _entry("St Pauli", "FC St. Pauli", "St Pauli", "St. Pauli"),
    "VfB Stuttgart": _entry("VfB Stuttgart", "VfB Stuttgart", "Stuttgart", "VfB Stuttgart"),
    "Union Berlin": _entry("Union Berlin", "Union Berlin", "Union Berlin", "Union Berlin"),
    "Werder Bremen": _entry("Werder Bremen", "Werder Bremen", "Werder Bremen", "Werder Bremen"),
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
    "Ajaccio": _entry("Ajaccio", "AC Ajaccio", "Ajaccio", "Ajaccio"),
    "Amiens": _entry("Amiens", "Amiens SC", "Amiens", "Amiens"),
    "Angers": _entry("Angers", "Angers", "Angers", "Angers"),
    "Auxerre": _entry("Auxerre", "Auxerre", "Auxerre", "Auxerre"),
    "Bordeaux": _entry("Bordeaux", "Bordeaux", "Bordeaux", "Bordeaux"),
    "Brest": _entry("Brest", "Stade Brestois", "Brest", "Brest"),
    "Caen": _entry("Caen", "Caen", "Caen", "Caen"),
    "Clermont Foot": _entry("Clermont Foot", "Clermont Foot", "Clermont", "Clermont Foot"),
    "Dijon": _entry("Dijon", "Dijon", "Dijon", "Dijon"),
    "Guingamp": _entry("Guingamp", "Guingamp", "Guingamp", "Guingamp"),
    "Le Havre": _entry("Le Havre", "Le Havre", "Le Havre", "Le Havre"),
    "Lens": _entry("Lens", "Lens", "Lens", "Lens"),
    "Lorient": _entry("Lorient", "Lorient", "Lorient", "Lorient"),
    "Metz": _entry("Metz", "Metz", "Metz", "Metz"),
    "Montpellier": _entry("Montpellier", "Montpellier", "Montpellier", "Montpellier"),
    "Nantes": _entry("Nantes", "Nantes", "Nantes", "Nantes"),
    "Nimes": _entry("Nimes", "Nimes", "Nimes", "Nimes"),
    "Paris FC": _entry("Paris FC", "Paris FC", "Paris FC", "Paris FC"),
    "Reims": _entry("Reims", "Stade Reims", "Reims", "Reims"),
    "Saint-Etienne": _entry("Saint-Etienne", "Saint-Etienne", "St Etienne", "Saint-Etienne"),
    "Strasbourg": _entry("Strasbourg", "Strasbourg", "Strasbourg", "Strasbourg"),
    "Toulouse": _entry("Toulouse", "Toulouse", "Toulouse", "Toulouse"),
    "Troyes": _entry("Troyes", "Troyes", "Troyes", "Troyes"),
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
