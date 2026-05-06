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

    Until Task 6, this function only returns demo predictions. The real-model
    inference branch is wired in the next commit; the structure below is intentionally
    left in three explicit branches so Task 6's diff is minimal.
    """
    if not isinstance(models, Models):
        # Legacy code path (removed in Task 10 alongside app.py changes)
        return _demo_prediction(home_team, away_team)
    if not models.is_ready:
        return _demo_prediction(home_team, away_team)
    # Real-model branch wired in Task 6
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


# ─── Compatibility shims (removed in Task 10 alongside app.py changes) ───


def model_exists() -> bool:
    """Deprecated: use Models.is_ready via load_models()."""
    return BETTOR_PATH.exists() and LOADER_PATH.exists() and BACKTEST_PATH.exists()


def load_model() -> Any | None:
    """Deprecated: use load_models()."""
    return _safe_unpickle(BETTOR_PATH)
