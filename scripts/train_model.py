"""Train the EPL multi-output ClassifierBettor and save it to disk.

Single command produces three artifacts in models/:
    epl_bettor.pkl    — fitted ClassifierBettor (handles 5 markets)
    epl_loader.pkl    — primed SoccerDataLoader (needed at inference time
                        for extract_fixtures_data())
    backtest.json     — per-market backtest summary

Usage:
    uv run python scripts/train_model.py

Run time: typically 10-20 minutes on a laptop. Existing artifacts are only
overwritten if the entire run succeeds (atomic write).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).parent.parent
# Make project root importable so `from src.foo import ...` works when this
# script is run directly (`python scripts/train_model.py`). Streamlit/pytest
# do this automatically; CLI entry-points need to do it explicitly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
BETTOR_PATH = MODELS_DIR / "epl_bettor.pkl"
LOADER_PATH = MODELS_DIR / "epl_loader.pkl"
BACKTEST_PATH = MODELS_DIR / "backtest.json"


def write_artifacts_atomically(
    bettor: Any,
    loader: Any,
    summary: dict,
) -> None:
    """Write all three artifacts to .tmp paths, then os.replace to final names.

    All three writes are staged to .tmp paths first; if any write fails, cleanup
    runs and none are promoted. The three os.replace calls are sequential, so a
    hard crash (power cut, OS kill) BETWEEN them could in theory leave a partial
    final state — vanishingly unlikely on a single-user local tool, but worth
    knowing. Cleans up orphaned .tmp files on any failure.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bettor_tmp = BETTOR_PATH.with_suffix(BETTOR_PATH.suffix + ".tmp")
    loader_tmp = LOADER_PATH.with_suffix(LOADER_PATH.suffix + ".tmp")
    backtest_tmp = BACKTEST_PATH.with_suffix(BACKTEST_PATH.suffix + ".tmp")
    tmps = [bettor_tmp, loader_tmp, backtest_tmp]

    try:
        with bettor_tmp.open("wb") as f:
            pickle.dump(bettor, f)
        with loader_tmp.open("wb") as f:
            pickle.dump(loader, f)
        with backtest_tmp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # All temp writes succeeded — promote atomically
        os.replace(bettor_tmp, BETTOR_PATH)
        os.replace(loader_tmp, LOADER_PATH)
        os.replace(backtest_tmp, BACKTEST_PATH)
    except BaseException:
        # Clean up any temp files left behind (don't mask the original exception)
        for tmp in tmps:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        raise


def run_backtest(
    bettor,
    X,
    Y,
    O,  # noqa: E741 — capital O matches library convention
    *,
    training_years: list[int],
) -> dict:
    """Run library backtest and reduce per-market columns into a JSON-ready summary.

    Library's `backtest()` returns a DataFrame indexed by training-window start date
    with per-market columns like:
        Yield percentage per bet (home_win__full_time_goals)
        Number of bets (home_win__full_time_goals)
        Precision per bet (home_win__full_time_goals)

    We aggregate those across rows to produce one summary block per market.
    The caller must pass `training_years` (the list of seasons used by the SoccerDataLoader)
    explicitly so that backtest.json reports the actual seasons, not a hardcoded list.
    """
    import sportsbet
    from sportsbet.evaluation import backtest as library_backtest

    bt_df: pd.DataFrame = library_backtest(bettor, X, Y, O)

    markets = ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]
    market_summary: dict[str, dict[str, float]] = {}

    for m in markets:
        col_market = f"{m}__full_time_goals"
        n_bets_col = f"Number of bets ({col_market})"
        win_rate_col = f"Precision per bet ({col_market})"
        yield_col = f"Yield percentage per bet ({col_market})"

        if n_bets_col not in bt_df.columns:
            print(
                f"[warn] backtest column not found for market {m!r} "
                f"(expected '{n_bets_col}'); recording zeros. "
                f"Library naming may have changed.",
                file=sys.stderr,
            )
            market_summary[m] = {"n_bets": 0, "win_rate": 0.0, "yield_pct": 0.0}
            continue

        total_bets = int(bt_df[n_bets_col].sum())
        # Average win rate weighted by number of bets per row
        if total_bets > 0:
            weighted_win = (bt_df[win_rate_col] * bt_df[n_bets_col]).sum() / total_bets
            weighted_yield = (bt_df[yield_col] * bt_df[n_bets_col]).sum() / total_bets
        else:
            weighted_win = 0.0
            weighted_yield = 0.0

        market_summary[m] = {
            "n_bets": total_bets,
            "win_rate": float(weighted_win),
            "yield_pct": float(weighted_yield),
        }

    return {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_seasons": list(training_years),
        "library_version": sportsbet.__version__,
        "n_training_matches": int(len(X)),
        "y_columns_order": list(Y.columns),  # locked for inference index mapping
        "markets": market_summary,
    }


def main() -> None:
    """Train the multi-output bettor end-to-end and persist artifacts."""
    print("⚽ Football Edge — training EPL multi-output bettor")
    print("=" * 60)

    # Library imports here (not at top) so import errors are reported with context
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    from sportsbet.datasets import SoccerDataLoader
    from sportsbet.evaluation import ClassifierBettor

    from src.sportsbet_patch import apply_patch

    # Patch the library's broken GitHub-scraping discovery before any loader call.
    apply_patch()

    # Single source of truth for the season range — also passed to run_backtest below
    training_years = list(range(2018, 2026))

    # 1. Load historical EPL data
    print("\n[1/4] Loading historical data (2018-2025) — this hits the network...")
    loader = SoccerDataLoader(
        param_grid={
            "league": ["England"],
            "year": training_years,
            "division": [1],
        }
    )
    X, Y, O = loader.extract_train_data(  # noqa: E741
        odds_type="market_average",
        drop_na_thres=1.0,
    )
    print(f"  Loaded {len(X)} matches.")
    print(f"  Y.columns order: {list(Y.columns)}")

    # 2. Build the calibrated multi-output pipeline
    print("\n[2/4] Building pipeline...")
    pipeline = make_pipeline(
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                ["league", "home_team", "away_team"],
            ),
            remainder="passthrough",
        ),
        SimpleImputer(),
        MultiOutputClassifier(
            CalibratedClassifierCV(
                GradientBoostingClassifier(random_state=0),
                method="isotonic",
                cv=3,
            )
        ),
    )
    bettor = ClassifierBettor(classifier=pipeline)

    # 3. Fit + backtest
    print("\n[3/4] Fitting bettor and running backtest (this is the slow bit, ~10-15 min)...")
    bettor.fit(X, Y, O)
    summary = run_backtest(bettor, X, Y, O, training_years=training_years)

    # 4. Atomic persist
    print("\n[4/4] Persisting artifacts...")
    write_artifacts_atomically(bettor, loader, summary)

    print("\n✅ Done. Artifacts written to:")
    print(f"     {BETTOR_PATH}")
    print(f"     {LOADER_PATH}")
    print(f"     {BACKTEST_PATH}")
    print("\nPer-market backtest summary:")
    for market, stats in summary["markets"].items():
        print(
            f"  {market:<12} {stats['n_bets']:>4} bets  "
            f"win-rate {stats['win_rate'] * 100:>5.1f}%  "
            f"yield {stats['yield_pct']:+.2f}%"
        )
    print()


if __name__ == "__main__":
    main()
