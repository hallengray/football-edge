"""Model loading and prediction wrapper.

Wraps the `sports-betting` library (https://github.com/georgedouzas/sports-betting)
to produce match outcome probabilities.

Includes a **demo mode** that runs when no trained model exists, so the
dashboard works end-to-end before you invest time in training. Demo
probabilities are derived from a simple home-advantage heuristic plus
small randomness — they're NOT predictive, just plausible-shaped data
to test the pipeline.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_PATH = Path(__file__).parent.parent / "models" / "epl_model.pkl"


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


def model_exists() -> bool:
    """Whether a trained model is available on disk."""
    return MODEL_PATH.exists()


def load_model() -> Any | None:
    """Load the trained model from disk, or return None if not available."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_fixture(
    model: Any | None,
    home_team: str,
    away_team: str,
) -> MatchPrediction:
    """Get probabilities for an upcoming fixture.

    If `model` is None, falls back to demo mode so the dashboard stays usable.

    Args:
        model: Trained sports-betting model, or None for demo mode.
        home_team: Home team name (must match training-data convention).
        away_team: Away team name.

    Returns:
        A MatchPrediction with probabilities summing to ~1.
    """
    if model is None:
        return _demo_prediction(home_team, away_team)

    # TODO: replace with the real sports-betting model call once trained.
    # The library's bettor objects expose `.predict()` / `.predict_proba()`,
    # but the input shape depends on how you configured the DataLoader.
    # See https://georgedouzas.github.io/sports-betting/ for the API.
    #
    # Typical pattern:
    #   features = build_features(home_team, away_team, recent_data)
    #   probs = model.predict_proba(features)
    #   return MatchPrediction(
    #       home_team=home_team,
    #       away_team=away_team,
    #       p_home=probs[0],
    #       p_draw=probs[1],
    #       p_away=probs[2],
    #   )

    raise NotImplementedError(
        "Wire this up to your trained sports-betting model. "
        "See module docstring for guidance."
    )


def _demo_prediction(home_team: str, away_team: str) -> MatchPrediction:
    """Generate plausible-shaped probabilities for testing the dashboard.

    Uses a deterministic hash of the team names to get repeatable values
    plus a small home-advantage bias. NOT predictive — purely for demo.
    """
    seed_str = f"{home_team}|{away_team}"
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

    # Pseudo-random splits with home advantage
    home_bias = (h % 100) / 100.0  # 0.0 to 0.99
    p_home = 0.35 + home_bias * 0.25  # 0.35 to 0.60
    p_draw = 0.20 + ((h >> 8) % 100) / 1000.0  # ~0.20 to 0.30
    p_away = max(1.0 - p_home - p_draw, 0.10)

    # Renormalize
    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    p_over_2_5 = 0.45 + ((h >> 16) % 100) / 250.0  # ~0.45 to 0.85

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        p_over_2_5=p_over_2_5,
        is_demo=True,
    )
