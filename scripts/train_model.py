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


def build_calibrated_classifier():
    """Same pipeline shape as v0 - keeping classifier constant isolates feature uplift."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    return make_pipeline(
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                ["league", "home_team", "away_team"],
            ),
            remainder="passthrough",
        ),
        SimpleImputer(strategy="mean"),
        MultiOutputClassifier(
            CalibratedClassifierCV(
                GradientBoostingClassifier(random_state=0),
                method="isotonic",
                cv=3,
            )
        ),
    )


def _find_first_present_columns(df: pd.DataFrame, candidates: list[list[str]]) -> list[str] | None:
    """Return the first list from `candidates` whose every column is present in df."""
    for candidate in candidates:
        if all(col in df.columns for col in candidate):
            return candidate
    return None


def _split_by_league(features_df: pd.DataFrame, league: str):
    """Filter the master feature DataFrame to one league. Returns (X, Y, O) tuple.

    Y columns: 5 markets in the locked order home_win/draw/away_win/over_2.5/under_2.5.
    O columns: bookmaker decimal odds for each market (from football-data CSV).
    X columns: everything else minus FTR/etc. that wouldn't be available pre-match.
    """
    df = features_df[features_df["league"] == LEAGUE_TO_FOOTBALL_DATA[league]].copy()

    # Build Y from FTHG vs FTAG
    home_goals = df["FTHG"]
    away_goals = df["FTAG"]
    total_goals = home_goals + away_goals

    Y = pd.DataFrame(
        {
            "output__home_win__full_time_goals": (home_goals > away_goals).astype(int),
            "output__draw__full_time_goals": (home_goals == away_goals).astype(int),
            "output__away_win__full_time_goals": (home_goals < away_goals).astype(int),
            "output__over_2.5__full_time_goals": (total_goals > 2.5).astype(int),
            "output__under_2.5__full_time_goals": (total_goals <= 2.5).astype(int),
        }
    )

    # Odds columns from football-data CSV: B365H, B365D, B365A for h2h
    # We use AvgH/D/A (or BbAvH/D/A in older seasons) if present; fall back to B365.
    h2h_cols = _find_first_present_columns(
        df,
        [
            ["AvgH", "AvgD", "AvgA"],
            ["BbAvH", "BbAvD", "BbAvA"],
            ["B365H", "B365D", "B365A"],
        ],
    )
    over_under_cols = _find_first_present_columns(
        df,
        [
            ["AvgOver2.5", "AvgUnder2.5"],
            ["BbAv>2.5", "BbAv<2.5"],
            ["B365>2.5", "B365<2.5"],
        ],
    )

    O = pd.DataFrame(  # noqa: E741
        {
            "output__home_win__full_time_goals": df[h2h_cols[0]] if h2h_cols else 2.0,
            "output__draw__full_time_goals": df[h2h_cols[1]] if h2h_cols else 3.5,
            "output__away_win__full_time_goals": df[h2h_cols[2]] if h2h_cols else 4.0,
            "output__over_2.5__full_time_goals": df[over_under_cols[0]] if over_under_cols else 2.0,
            "output__under_2.5__full_time_goals": df[over_under_cols[1]]
            if over_under_cols
            else 1.85,
        }
    )

    # X = features only -- drop the outcome columns and odds columns
    drop_cols = ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols
        and not (h2h_cols and c in h2h_cols)
        and not (over_under_cols and c in over_under_cols)
    ]
    X = df[feature_cols].reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    O = O.reset_index(drop=True)  # noqa: E741
    return X, Y, O


def _build_upcoming_fixtures_for_features(matches_raw: pd.DataFrame) -> pd.DataFrame:
    """Snapshot upcoming fixtures from The Odds API in a shape compute_features can consume.

    Why The Odds API rather than football-data's fixtures.csv: spec flagged this as
    planner-resolvable. We pick The Odds API because it's known to cover all 5 leagues
    and tags each fixture with a sport_key we map to our internal league key. The
    football-data fixtures.csv may be EPL-only.

    Returns a DataFrame with HomeTeam, AwayTeam, Date, league, year, plus placeholder
    FTHG=0/FTAG=0 (don't affect rolling features for the fixture row itself because
    rolling uses shift(1) - only past matches contribute).
    """
    from src.odds import get_big5_odds
    from src.team_names import from_canonical, to_canonical

    fixtures = get_big5_odds()
    rows: list[dict] = []
    for fixture in fixtures:
        try:
            home_canonical = to_canonical(fixture["home_team"], source="odds_api")
            away_canonical = to_canonical(fixture["away_team"], source="odds_api")
        except KeyError as e:
            logger.warning(f"Skipping unmapped fixture team: {e}")
            continue
        home_fd = from_canonical(home_canonical, target="football_data")
        away_fd = from_canonical(away_canonical, target="football_data")
        league_fd = LEAGUE_TO_FOOTBALL_DATA[fixture["league"]]
        rows.append(
            {
                "Date": pd.to_datetime(fixture["commence_time"]).date().isoformat(),
                "HomeTeam": home_fd,
                "AwayTeam": away_fd,
                "FTHG": 0,
                "FTAG": 0,
                "league": league_fd,
                "year": pd.to_datetime(fixture["commence_time"]).year,
            }
        )

    if not rows:
        return pd.DataFrame()
    fixtures_df = pd.DataFrame(rows)
    if "date" in matches_raw.columns and "Date" in fixtures_df.columns:
        fixtures_df = fixtures_df.rename(columns={"Date": "date"})
    if "home_team" in matches_raw.columns and "HomeTeam" in fixtures_df.columns:
        fixtures_df = fixtures_df.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})
    return fixtures_df


def main() -> None:
    """Train all five per-league bettors end-to-end and persist artifacts."""
    from src.features import compute_features
    from src.ingest.football_data import download_training_data
    from src.ingest.understat import fetch_xg_data

    print("Football Edge - training Big-5 multi-output bettors")
    print("=" * 60)

    training_years = list(range(2018, 2026))

    # 1. Ingest
    print("\n[1/5] Downloading football-data.co.uk CSVs...")
    football_leagues = list(LEAGUE_TO_FOOTBALL_DATA.values())
    matches_raw = download_training_data(leagues=football_leagues, years=training_years)
    print(f"  Loaded {len(matches_raw)} historical matches.")

    # Normalise football-data's column names to lowercase that compute_features expects.
    if "Date" in matches_raw.columns and "date" not in matches_raw.columns:
        matches_raw = matches_raw.rename(columns={"Date": "date"})
    if "HomeTeam" in matches_raw.columns and "home_team" not in matches_raw.columns:
        matches_raw = matches_raw.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})

    print("  Fetching upcoming fixtures from The Odds API...")
    upcoming_fixtures = _build_upcoming_fixtures_for_features(matches_raw)
    print(f"  Snapshotted {len(upcoming_fixtures)} upcoming fixtures across the Big-5.")

    print("\n[2/5] Scraping Understat xG (this is the slow ingest step)...")
    understat_leagues = list(LEAGUE_TO_UNDERSTAT.values())
    xg_df = fetch_xg_data(leagues=understat_leagues, years=training_years)
    print(f"  Loaded xG for {len(xg_df)} historical matches.")

    # 2. Feature engineering -- combine training + upcoming so compute_features sees
    # full history when computing rolling features for upcoming-fixture rows.
    # compute_features uses shift(1), so the upcoming-fixture rows' OWN placeholder
    # outcomes don't pollute their own features.
    print("\n[3/5] Computing engineered features...")
    if not upcoming_fixtures.empty:
        combined = pd.concat([matches_raw, upcoming_fixtures], ignore_index=True)
    else:
        combined = matches_raw
    combined_features = compute_features(combined, xg_df)

    n_training = len(matches_raw)
    features_df = combined_features.iloc[:n_training].reset_index(drop=True)
    fixtures_features_df = combined_features.iloc[n_training:].reset_index(drop=True)
    print(f"  Training features shape: {features_df.shape}")
    print(f"  Fixtures features shape: {fixtures_features_df.shape}")

    # 3. Per-league training loop
    print("\n[4/5] Per-league training (~5-10 minutes per league)...")
    bettors: dict[str, Any] = {}
    league_summaries: dict[str, dict] = {}

    for league in BIG_5_LEAGUES:
        print(f"  - Training {league}...")
        X, Y, O = _split_by_league(features_df, league)  # noqa: E741
        if len(X) < 100:
            raise RuntimeError(
                f"League {league} has only {len(X)} matches - refusing to train. "
                f"Check team-name mappings and ingest output."
            )

        league_summaries[league] = {
            "n_matches": len(X),
            **run_backtest_cv(
                build_calibrated_classifier,
                X,
                Y,
                O,
                n_splits=5,
                value_threshold=0.05,
            ),
        }
        bettor = build_calibrated_classifier()
        bettor.fit(X, Y)
        bettors[league] = bettor
        print(f"    Done. {len(X)} matches.")

    # 4. Build summary
    summary = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_seasons": training_years,
        "sklearn_version": pkg_version("scikit-learn"),
        "n_training_matches": int(len(features_df)),
        "leagues": league_summaries,
    }

    # 5. Atomic persist
    print("\n[5/5] Persisting artifacts...")
    write_artifacts_atomically(bettors, fixtures_features_df, summary)

    print("\nDone. Artifacts written to:")
    for league in BIG_5_LEAGUES:
        print(f"     {BETTOR_PATHS[league]}")
    print(f"     {FIXTURES_PATH}")
    print(f"     {BACKTEST_PATH}")
    print("\nPer-league per-market backtest summary:")
    for league, league_summary in summary["leagues"].items():
        print(f"\n  {league} ({league_summary['n_matches']} matches):")
        for market, stats in league_summary["markets"].items():
            print(f"    {market:<12} {stats['n_bets']:>4} bets  yield {stats['yield_pct']:+.2f}%")
    print()


if __name__ == "__main__":
    main()
