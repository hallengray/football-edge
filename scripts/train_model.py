"""Train the EPL model using the sports-betting library and save it to disk.

This is a stub — the real implementation depends on which DataLoader and
Bettor configuration you choose from the sports-betting library. See:
    https://georgedouzas.github.io/sports-betting/

The library is config-driven. You'll typically:
    1. Define a SoccerDataLoader for the EPL across multiple seasons
    2. Define a PARAM_GRID (which features, which odds source)
    3. Pick a Bettor (e.g. ClassifierBettor wrapping XGBoost)
    4. Run backtest() to evaluate before saving
    5. Pickle the fitted bettor to models/epl_model.pkl

Run: `uv run python scripts/train_model.py`
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "epl_model.pkl"


def main() -> None:
    print("Training EPL model...")
    MODELS_DIR.mkdir(exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # TODO: replace with the real sports-betting pipeline.
    #
    # Sketch (check the library docs for the current API):
    #
    #   from sportsbet.datasets import SoccerDataLoader
    #   from sportsbet.evaluation import ClassifierBettor, backtest
    #   from sklearn.ensemble import GradientBoostingClassifier
    #
    #   loader = SoccerDataLoader(
    #       param_grid={
    #           "league": ["England"],
    #           "year": [2020, 2021, 2022, 2023, 2024, 2025],
    #           "division": [1],
    #       }
    #   )
    #   X_train, Y_train, odds_train = loader.extract_train_data(
    #       odds_type="market_average",
    #       drop_na_thres=0.8,
    #   )
    #
    #   bettor = ClassifierBettor(
    #       classifier=GradientBoostingClassifier(),
    #   )
    #   bettor.fit(X_train, Y_train)
    #
    #   # Backtest before trusting it
    #   results = backtest(bettor, X_train, Y_train, odds_train, cv=5)
    #   print(results)
    #
    #   with open(MODEL_PATH, "wb") as f:
    #       pickle.dump(bettor, f)
    #
    #   print(f"Model saved to {MODEL_PATH}")
    # ─────────────────────────────────────────────────────────────────────

    print(
        "\n⚠️  This script is a stub.\n"
        "Open scripts/train_model.py and wire up the sports-betting library.\n"
        "Until then, the dashboard runs in demo mode using fake probabilities."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
