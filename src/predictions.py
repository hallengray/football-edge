"""Multi-league prediction container and routing.

`Models` holds a dict of per-league fitted bettors plus a shared fixtures DataFrame
and the parsed backtest summary. `is_ready` is True only when all five leagues'
bettors are loaded and the fixtures+backtest are present.

`predict_fixture(models, league, home_team, away_team)` routes to the right bettor
based on the `league` key (passed in by the dashboard from the Odds API's sport_key).
Demo fallback for any error: missing bettor, unmapped team, no fixture row, library
exception during predict_proba.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BIG_5_LEAGUES: list[str] = ["epl", "laliga", "seriea", "bundesliga", "ligue1"]

# Map internal league key -> football-data.co.uk league name (used in fixtures_df["league"])
LEAGUE_TO_FOOTBALL_DATA: dict[str, str] = {
    "epl": "England",
    "laliga": "Spain",
    "seriea": "Italy",
    "bundesliga": "Germany",
    "ligue1": "France",
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BETTOR_PATHS: dict[str, Path] = {
    league: MODELS_DIR / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES
}
FIXTURES_PATH: Path = MODELS_DIR / "fixtures_data.parquet"
BACKTEST_PATH: Path = MODELS_DIR / "backtest.json"


@dataclass
class MatchPrediction:
    """Model output for a single fixture."""

    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    p_over_2_5: float | None = None
    is_demo: bool = False


@dataclass
class Models:
    """Container for per-league bettors plus shared inference artifacts."""

    bettors: dict[str, Any] = field(default_factory=dict)
    fixtures_df: pd.DataFrame | None = None
    backtest: dict | None = None

    @property
    def is_ready(self) -> bool:
        """True only when all five leagues' bettors plus fixtures+backtest are loaded."""
        return (
            len(self.bettors) == len(BIG_5_LEAGUES)
            and all(self.bettors.get(league) is not None for league in BIG_5_LEAGUES)
            and self.fixtures_df is not None
            and self.backtest is not None
        )


def load_models() -> Models:
    """Read 5 *_bettor.pkl + fixtures_data.parquet + backtest.json. Any missing -> demo."""
    bettors: dict[str, Any] = {}
    for league, path in BETTOR_PATHS.items():
        loaded = _safe_unpickle(path)
        if loaded is not None:
            bettors[league] = loaded

    fixtures_df = _safe_load_parquet(FIXTURES_PATH)
    backtest_data = _safe_load_json(BACKTEST_PATH)

    return Models(bettors=bettors, fixtures_df=fixtures_df, backtest=backtest_data)


def _safe_unpickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:  # pickle errors, version skew, anything
        logger.warning(f"Could not load pickle {path}: {e}")
        return None


def _safe_load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"Could not load parquet {path}: {e}")
        return None


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # JSON syntax errors, encoding issues, anything
        logger.warning(f"Could not load JSON {path}: {e}")
        return None


def predict_fixture(
    models: Models | None,
    league: str,
    home_team: str,
    away_team: str,
) -> MatchPrediction:
    """Get probabilities for an upcoming fixture. Multi-league routing.

    home_team and away_team are Odds API names (long form). League is one of
    BIG_5_LEAGUES. Falls back to demo mode on any error.

    Y.columns order from training is locked to:
        [home_win, draw, away_win, over_2.5, under_2.5]
    indexed as 0/1/2/3 below for the four probabilities we surface.
    """
    if not isinstance(models, Models) or not models.is_ready:
        return _demo_prediction(home_team, away_team)
    if league not in BIG_5_LEAGUES:
        return _demo_prediction(home_team, away_team)

    try:
        from src.team_names import to_canonical

        home_canonical = to_canonical(home_team, source="odds_api")
        away_canonical = to_canonical(away_team, source="odds_api")
        league_fd_name = LEAGUE_TO_FOOTBALL_DATA[league]

        match = models.fixtures_df[
            (models.fixtures_df["league"] == league_fd_name)
            & (models.fixtures_df["home_team"] == home_canonical)
            & (models.fixtures_df["away_team"] == away_canonical)
        ]
        if match.empty:
            logger.warning(
                f"No library fixture for {home_team}({home_canonical}) vs "
                f"{away_team}({away_canonical}) in {league}"
            )
            return _demo_prediction(home_team, away_team)

        bettor = models.bettors[league]
        probs_per_market = bettor.predict_proba(match.iloc[[0]])

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            p_home=float(probs_per_market[0][0][1]),
            p_draw=float(probs_per_market[1][0][1]),
            p_away=float(probs_per_market[2][0][1]),
            p_over_2_5=float(probs_per_market[3][0][1]),
            is_demo=False,
        )
    except Exception as e:
        logger.warning(
            f"Real-model inference failed for {league}: {home_team} vs {away_team}: {e}",
            exc_info=True,
        )
        return _demo_prediction(home_team, away_team)


def _demo_prediction(home_team: str, away_team: str) -> MatchPrediction:
    """Generate plausible-shaped probabilities for testing the dashboard.

    Deterministic hash of team names plus a small home-advantage bias. NOT predictive.
    """
    seed_str = f"{home_team}|{away_team}"
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

    home_bias = (h % 100) / 100.0
    p_home = 0.35 + home_bias * 0.25
    p_draw = 0.20 + ((h >> 8) % 100) / 1000.0
    p_away = max(1.0 - p_home - p_draw, 0.10)

    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    p_over_2_5 = 0.45 + ((h >> 16) % 100) / 250.0

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        p_over_2_5=p_over_2_5,
        is_demo=True,
    )
