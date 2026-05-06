"""Model loading and prediction wrapper.

Wraps a `sports-betting` library multi-output ClassifierBettor and the primed
SoccerDataLoader needed for inference. Falls back to demo mode (synthetic
probabilities) when artifacts are missing or inference fails.

Three artifacts are loaded together:
    models/epl_bettor.pkl   — the fitted ClassifierBettor
    models/epl_loader.pkl   — the SoccerDataLoader, primed by extract_train_data
    models/backtest.json    — per-market backtest summary (for the dashboard)

If any artifact is missing or unloadable, the entire Models container reports
is_ready=False and the dashboard falls back to demo predictions.
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

MODELS_DIR = Path(__file__).parent.parent / "models"
BETTOR_PATH = MODELS_DIR / "epl_bettor.pkl"
LOADER_PATH = MODELS_DIR / "epl_loader.pkl"
BACKTEST_PATH = MODELS_DIR / "backtest.json"


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
    """Container for the three artifacts plus session-cached fixtures DataFrame."""

    bettor: Any | None
    loader: Any | None
    backtest: dict | None
    fixtures_df: pd.DataFrame | None = field(default=None)

    @property
    def is_ready(self) -> bool:
        """True when all three inference artifacts are loaded."""
        return self.bettor is not None and self.loader is not None and self.backtest is not None


def load_models() -> Models:
    """Load all three artifacts. Any missing or corrupt → demo mode (is_ready=False)."""
    bettor = _safe_unpickle(BETTOR_PATH)
    loader = _safe_unpickle(LOADER_PATH)
    backtest_data = _safe_load_json(BACKTEST_PATH)
    return Models(bettor=bettor, loader=loader, backtest=backtest_data)


def _safe_unpickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:  # pickle errors, version skew, anything
        logger.warning(f"Could not load pickle {path}: {e}")
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
    home_team: str,
    away_team: str,
) -> MatchPrediction:
    """Get probabilities for an upcoming fixture.

    home_team and away_team are Odds API names. Falls back to demo mode if:
    - models is None or not ready
    - team name isn't in the static Odds API → library map
    - the library has no fixture row matching the team pair
    - the bettor's predict_proba throws

    Y.columns order from training is locked to:
        [home_win, draw, away_win, over_2.5, under_2.5]
    indexed as 0/1/2/3/4 below. If `backtest.json` records a different order
    (Task 8 writes this), this function must change accordingly.
    """
    if not isinstance(models, Models) or not models.is_ready:
        return _demo_prediction(home_team, away_team)

    try:
        from src.team_names import to_library

        home_lib = to_library(home_team)
        away_lib = to_library(away_team)

        if models.fixtures_df is None:
            X_fix, _, _ = models.loader.extract_fixtures_data()
            models.fixtures_df = X_fix

        match = models.fixtures_df[
            (models.fixtures_df["home_team"] == home_lib)
            & (models.fixtures_df["away_team"] == away_lib)
        ]
        if match.empty:
            logger.warning(
                f"No library fixture for {home_team}({home_lib}) vs {away_team}({away_lib})"
            )
            return _demo_prediction(home_team, away_team)

        probs = models.bettor.predict_proba(match.iloc[[0]])[0]

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            p_home=float(probs[0]),
            p_draw=float(probs[1]),
            p_away=float(probs[2]),
            p_over_2_5=float(probs[3]),
            is_demo=False,
        )
    except Exception as e:
        logger.warning(f"Real-model inference failed for {home_team} vs {away_team}: {e}")
        return _demo_prediction(home_team, away_team)


def _demo_prediction(home_team: str, away_team: str) -> MatchPrediction:
    """Generate plausible-shaped probabilities for testing the dashboard.

    Uses a deterministic hash of the team names to get repeatable values
    plus a small home-advantage bias. NOT predictive — purely for demo.
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
