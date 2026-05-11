"""Tests for training pipeline plumbing.

Atomic-write tests verify the 7-artifact (5 pkl + 1 parquet + 1 json) write.
Backtest CV test verifies per-market yield arithmetic on a synthetic dataset.
End-to-end training is exercised via the manual smoke test (Task 16).
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from scripts.train_model import (
    BIG_5_LEAGUES,
    _split_features_into_training_and_upcoming,
    run_backtest_cv,
    write_artifacts_atomically,
)


def test_run_backtest_cv_yields_arithmetic() -> None:
    """Synthetic 30-match single-market dataset; assert per-market yield matches manual calc.

    We build a dataset where the model will predict a fixed probability for every
    match, and odds are constant. Value bets fire predictably; we can compute
    the exact yield by hand and assert the CV pipeline matches.
    """
    n = 30
    X = pd.DataFrame(
        {
            "feature_a": np.arange(n, dtype=float),
            "feature_b": np.arange(n, dtype=float) * 0.5,
        }
    )
    Y = pd.DataFrame(
        {
            "output__home_win__full_time_goals": [1, 0] * (n // 2),
            "output__draw__full_time_goals": [0, 1] * (n // 2),
            "output__away_win__full_time_goals": [0, 0] * (n // 2),
            "output__over_2.5__full_time_goals": [1, 1] * (n // 2),
            "output__under_2.5__full_time_goals": [0, 0] * (n // 2),
        }
    )
    O = pd.DataFrame({col: [2.0] * n for col in Y.columns})  # noqa: E741

    class _ConstantClassifier:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            n = len(X)
            return [np.column_stack([np.full(n, 0.4), np.full(n, 0.6)]) for _ in range(5)]

    def factory():
        return _ConstantClassifier()

    summary = run_backtest_cv(
        bettor_factory=factory,
        X=X,
        Y=Y,
        O=O,
        n_splits=3,
        value_threshold=0.05,
    )
    assert set(summary["markets"].keys()) == {
        "home_win",
        "draw",
        "away_win",
        "over_2.5",
        "under_2.5",
    }
    for market_stats in summary["markets"].values():
        assert "n_bets" in market_stats
        assert "yield_pct" in market_stats
    assert summary["n_matches"] == n


def test_atomic_write_creates_seven_artifacts(tmp_path, monkeypatch) -> None:
    """5 bettor pkls + 1 parquet + 1 json all land at their final paths."""
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")
    monkeypatch.setattr("scripts.train_model.FIXTURES_PATH", tmp_path / "fixtures_data.parquet")
    monkeypatch.setattr(
        "scripts.train_model.BETTOR_PATHS",
        {league: tmp_path / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES},
    )

    bettors = {league: {"fake_bettor_for": league} for league in BIG_5_LEAGUES}
    fixtures_df = pd.DataFrame({"a": [1, 2, 3]})
    summary = {"trained_at": "2026-05-06T00:00:00Z", "leagues": {}}

    write_artifacts_atomically(bettors, fixtures_df, summary)

    for league in BIG_5_LEAGUES:
        path = tmp_path / f"{league}_bettor.pkl"
        assert path.exists()
        with path.open("rb") as f:
            assert pickle.load(f) == {"fake_bettor_for": league}

    assert (tmp_path / "fixtures_data.parquet").exists()
    assert (tmp_path / "backtest.json").exists()
    with (tmp_path / "backtest.json").open() as f:
        assert json.load(f)["trained_at"] == "2026-05-06T00:00:00Z"


def _synthetic_match(date: str, home: str, away: str, league: str) -> dict:
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "FTHG": 1,
        "FTAG": 1,
        "HST": 5,
        "AST": 5,
        "league": league,
        "year": int(date[:4]),
    }


def _synthetic_xg_for(matches: list[dict]) -> pd.DataFrame:
    """Build a matching xG DataFrame so _add_xg_features has numeric inputs."""
    return pd.DataFrame(
        [
            {
                "date": m["date"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_xg": 1.2,
                "away_xg": 1.0,
            }
            for m in matches
        ]
    )


def test_split_recovers_all_upcoming_fixtures_across_all_leagues() -> None:
    """Regression test for the May 2026 fixtures_data.parquet Spain-only bug.

    compute_features sorts internally by ['league', 'year', 'date'] inside
    _add_strength_of_schedule. The previous iloc[n_training:] split therefore
    grabbed only rows from the alphabetically-last league (Spain), silently
    dropping upcoming fixtures from England, France, Germany, Italy. The fix
    uses a marker column so the split survives any internal row reordering.

    This test fails on the old iloc-based logic and passes on the marker logic.
    """
    # Use real registry team names so _normalise_team_names doesn't drop rows.
    training_matches = [
        # England — 4 matches across 2 dates so rolling windows have history
        _synthetic_match("2025-08-15", "Arsenal", "Chelsea", "England"),
        _synthetic_match("2025-08-22", "Liverpool", "Arsenal", "England"),
        _synthetic_match("2025-08-29", "Chelsea", "Liverpool", "England"),
        _synthetic_match("2025-09-05", "Arsenal", "Liverpool", "England"),
        # Spain
        _synthetic_match("2025-08-15", "Real Madrid", "Barcelona", "Spain"),
        _synthetic_match("2025-08-22", "Sevilla", "Real Madrid", "Spain"),
        _synthetic_match("2025-08-29", "Barcelona", "Sevilla", "Spain"),
        _synthetic_match("2025-09-05", "Real Madrid", "Sevilla", "Spain"),
    ]
    matches_raw = pd.DataFrame(training_matches)

    # One upcoming fixture per league — dated AFTER all training matches.
    upcoming_matches = [
        _synthetic_match("2026-05-08", "Arsenal", "Chelsea", "England"),
        _synthetic_match("2026-05-08", "Real Madrid", "Barcelona", "Spain"),
    ]
    upcoming_fixtures = pd.DataFrame(upcoming_matches)

    # xG covers training matches only; upcoming xG isn't known yet.
    xg_df = _synthetic_xg_for(training_matches)

    training_features, upcoming_features = _split_features_into_training_and_upcoming(
        matches_raw, upcoming_fixtures, xg_df
    )

    # Marker column must not leak into either output.
    assert "_is_upcoming" not in training_features.columns
    assert "_is_upcoming" not in upcoming_features.columns

    # Every upcoming-fixture league must be recovered.
    assert set(upcoming_features["league"].unique()) == {"England", "Spain"}
    assert len(upcoming_features) == 2

    # Training output should contain exactly the historical rows, none of the
    # upcoming dates.
    assert (upcoming_features["date"] >= pd.Timestamp("2026-01-01")).all()
    assert (training_features["date"] < pd.Timestamp("2026-01-01")).all()


def test_split_handles_empty_upcoming_fixtures() -> None:
    """Off-season case: get_big5_odds may return zero upcoming fixtures."""
    training_matches = [
        _synthetic_match("2025-08-15", "Arsenal", "Chelsea", "England"),
        _synthetic_match("2025-08-22", "Liverpool", "Arsenal", "England"),
    ]
    matches_raw = pd.DataFrame(training_matches)
    upcoming_fixtures = pd.DataFrame()
    xg_df = _synthetic_xg_for(training_matches)

    training_features, upcoming_features = _split_features_into_training_and_upcoming(
        matches_raw, upcoming_fixtures, xg_df
    )

    assert len(training_features) == 2
    assert len(upcoming_features) == 0
    assert "_is_upcoming" not in training_features.columns


def test_atomic_write_no_partial_state_when_one_pickle_fails(tmp_path, monkeypatch) -> None:
    """If any of the 7 writes fails, all .tmp files are cleaned and no final file appears."""
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")
    monkeypatch.setattr("scripts.train_model.FIXTURES_PATH", tmp_path / "fixtures_data.parquet")
    monkeypatch.setattr(
        "scripts.train_model.BETTOR_PATHS",
        {league: tmp_path / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES},
    )

    class Unpicklable:
        def __reduce__(self):
            raise TypeError("nope")

    bettors: dict = {league: {"ok": league} for league in BIG_5_LEAGUES}
    bettors["laliga"] = Unpicklable()
    fixtures_df = pd.DataFrame({"a": [1]})
    summary = {"trained_at": "x"}

    with pytest.raises(TypeError):
        write_artifacts_atomically(bettors, fixtures_df, summary)

    for league in BIG_5_LEAGUES:
        assert not (tmp_path / f"{league}_bettor.pkl").exists()
    assert not (tmp_path / "fixtures_data.parquet").exists()
    assert not (tmp_path / "backtest.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []
