"""Train the Big-5 multi-output ClassifierBettor and save artifacts atomically.

Single command produces seven artifacts in models/:
    epl_bettor.pkl, laliga_bettor.pkl, seriea_bettor.pkl,
    bundesliga_bettor.pkl, ligue1_bettor.pkl  -- one fitted classifier per league
    fixtures_data.parquet                     -- pre-computed feature DataFrame for upcoming fixtures
    backtest.json                             -- per-league per-market backtest summary

Usage:
    uv run python scripts/train_model.py

Run time: typically 30-50 minutes on a laptop. Existing artifacts are only
overwritten if the entire run succeeds (atomic write).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pickle
import sys
from collections.abc import Callable
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
# Make project root importable so `from src.foo import ...` works when this
# script is run directly. Streamlit/pytest do this automatically; CLI entry-points
# need to do it explicitly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

BIG_5_LEAGUES: list[str] = ["epl", "laliga", "seriea", "bundesliga", "ligue1"]

# Per-league bettor pickle paths
BETTOR_PATHS: dict[str, Path] = {
    league: MODELS_DIR / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES
}
FIXTURES_PATH: Path = MODELS_DIR / "fixtures_data.parquet"
BACKTEST_PATH: Path = MODELS_DIR / "backtest.json"

# Mapping from internal league key -> football-data.co.uk capitalised league name
LEAGUE_TO_FOOTBALL_DATA: dict[str, str] = {
    "epl": "England",
    "laliga": "Spain",
    "seriea": "Italy",
    "bundesliga": "Germany",
    "ligue1": "France",
}
# Mapping from internal league key -> Understat slug
LEAGUE_TO_UNDERSTAT: dict[str, str] = {
    "epl": "EPL",
    "laliga": "La_liga",
    "seriea": "Serie_A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue_1",
}


# ─── Backtest CV ────────────────────────────────────────────────────


def _strip_market_suffix(col: str) -> str:
    """`output__home_win__full_time_goals` -> `home_win`."""
    parts = col.split("__")
    if len(parts) >= 3:
        return parts[1]
    return col


def run_backtest_cv(
    bettor_factory: Callable[[], Any],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    O: pd.DataFrame,  # noqa: E741 — capital O matches library convention
    *,
    n_splits: int = 5,
    value_threshold: float = 0.05,
) -> dict:
    """Run TimeSeriesSplit cross-validation; return per-market yield summary.

    For each fold, fit a fresh bettor on train, predict on test, select bets where
    `model_prob * decimal_odds > 1 + value_threshold`, accumulate returns.
    Yield per market = mean of returns across all folds' selected bets.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    market_keys = [_strip_market_suffix(col) for col in Y.columns]
    bets_per_market: dict[str, list[float]] = {key: [] for key in market_keys}

    for train_idx, test_idx in tscv.split(X):
        if len(test_idx) == 0:
            continue
        bettor = bettor_factory()
        bettor.fit(X.iloc[train_idx], Y.iloc[train_idx])

        # MultiOutputClassifier returns a list (one array per market)
        probs_per_market = bettor.predict_proba(X.iloc[test_idx])

        for market_idx, market_col in enumerate(Y.columns):
            market_key = market_keys[market_idx]
            test_outcomes = Y.iloc[test_idx][market_col].to_numpy()
            test_odds = O.iloc[test_idx][market_col].to_numpy()
            test_probs = probs_per_market[market_idx][:, 1]

            value_mask = test_probs * test_odds > 1 + value_threshold
            returns = np.where(value_mask, test_odds * test_outcomes - 1, 0.0)
            bets_per_market[market_key].extend([float(r) for r in returns[returns != 0]])

    summary_markets: dict[str, dict[str, float]] = {}
    for market_key, returns in bets_per_market.items():
        if not returns:
            summary_markets[market_key] = {"n_bets": 0, "yield_pct": 0.0}
        else:
            summary_markets[market_key] = {
                "n_bets": len(returns),
                "yield_pct": round(100 * float(np.mean(returns)), 2),
            }

    return {
        "n_matches": len(X),
        "markets": summary_markets,
    }


# ─── Atomic write ──────────────────────────────────────────────────


def write_artifacts_atomically(
    bettors: dict[str, Any],
    fixtures_df: pd.DataFrame,
    summary: dict,
) -> None:
    """Write all 7 artifacts to .tmp paths, then os.replace them to final paths.

    Either all 7 land at their final paths, or none do. Cleans up .tmp files
    on any failure. The five bettor pickles, the fixtures parquet, and the
    backtest JSON are promoted in one final pass.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bettor_tmps = {
        league: BETTOR_PATHS[league].with_suffix(BETTOR_PATHS[league].suffix + ".tmp")
        for league in BIG_5_LEAGUES
    }
    fixtures_tmp = FIXTURES_PATH.with_suffix(FIXTURES_PATH.suffix + ".tmp")
    backtest_tmp = BACKTEST_PATH.with_suffix(BACKTEST_PATH.suffix + ".tmp")
    all_tmps = list(bettor_tmps.values()) + [fixtures_tmp, backtest_tmp]

    try:
        # 1. Write all temps
        for league, bettor in bettors.items():
            with bettor_tmps[league].open("wb") as f:
                pickle.dump(bettor, f)
        fixtures_df.to_parquet(fixtures_tmp, index=False)
        with backtest_tmp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 2. All temp writes succeeded -- promote atomically
        for league in BIG_5_LEAGUES:
            os.replace(bettor_tmps[league], BETTOR_PATHS[league])
        os.replace(fixtures_tmp, FIXTURES_PATH)
        os.replace(backtest_tmp, BACKTEST_PATH)
    except BaseException:
        for tmp in all_tmps:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        raise


def main() -> None:
    raise NotImplementedError("Implemented in Task 11")


# Re-export imports used by main() (kept here so import order is stable)
_ = (dt, pkg_version, logger)


if __name__ == "__main__":
    main()
