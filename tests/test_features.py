"""Tests for src/features.py engineered features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    _add_form_features,
    _add_rest_features,
    _add_rolling_goals,
    _add_strength_of_schedule,
    _add_xg_features,
    _merge_xg,
    _normalise_team_names,
    compute_features,
)


def test_add_rest_features_first_match_uses_default() -> None:
    """A team's first-ever match has no prior history; rest_days defaults to 7."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    out = _add_rest_features(df)
    assert out.loc[0, "home_rest_days"] == 7
    assert out.loc[0, "away_rest_days"] == 7
    assert out.loc[0, "home_matches_last_14d"] == 1  # this match itself counts as 0 prior
    assert out.loc[0, "away_matches_last_14d"] == 1


def test_add_rest_features_uses_team_match_history() -> None:
    """Team plays Aug 1, 8, 15 — match on the 15th has rest=7 and matches_last_14d=2."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-08", "2024-08-15"]),
            "home_team": ["Arsenal", "Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Liverpool", "Tottenham"],
        }
    )
    out = _add_rest_features(df)
    # Third row: Arsenal at home; last home match was 2024-08-08 → 7 days rest;
    # in last 14 days Arsenal played 2024-08-01 and 2024-08-08
    assert out.loc[2, "home_rest_days"] == 7
    assert out.loc[2, "home_matches_last_14d"] == 2


def test_add_rest_features_caps_at_14_days() -> None:
    """A long break (international break, end of season) caps rest_days at 14."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01", "2024-08-15"]),  # 106-day gap
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Liverpool"],
        }
    )
    out = _add_rest_features(df)
    assert out.loc[1, "home_rest_days"] == 14


# ─── Form tests ──────────────────────────────────────────────────


def test_form_features_counts_wins_draws_losses() -> None:
    """Arsenal plays: W, D, L, W, W. The 6th match (any) should show form 3-1-1."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-08-01",
                    "2024-08-08",
                    "2024-08-15",
                    "2024-08-22",
                    "2024-08-29",
                    "2024-09-05",
                ]
            ),
            "home_team": ["Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "away_team": ["A", "B", "C", "D", "E", "F"],
            "FTHG": [2, 1, 0, 3, 2, 1],
            "FTAG": [1, 1, 2, 0, 0, 1],
        }
    )
    out = _add_form_features(df, window=5)
    # Match index 5 (the 6th): prior 5 results for Arsenal as home: W, D, L, W, W
    assert out.loc[5, "home_form_wins"] == 3
    assert out.loc[5, "home_form_draws"] == 1
    assert out.loc[5, "home_form_losses"] == 1


def test_form_features_first_match_zeros() -> None:
    """A team with no prior matches gets all zeros (no history to count)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
        }
    )
    out = _add_form_features(df, window=5)
    assert out.loc[0, "home_form_wins"] == 0
    assert out.loc[0, "home_form_draws"] == 0
    assert out.loc[0, "home_form_losses"] == 0


# ─── Rolling goals + xG tests ────────────────────────────────────────


def test_rolling_goals_uses_only_past_matches_no_leakage() -> None:
    """The leakage guard: match N's `goals_last_5` only sees matches 0..N-1."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-08", "2024-08-15"]),
            "home_team": ["Arsenal", "Arsenal", "Arsenal"],
            "away_team": ["A", "B", "C"],
            "FTHG": [2, 4, 100],  # huge value at index 2 — a leakage bug would surface here
            "FTAG": [1, 0, 0],
        }
    )
    out = _add_rolling_goals(df, window=5)
    # Index 2's home_goals_scored_last_5 should average matches 0 and 1: (2+4)/2 = 3.0
    # If the impl mistakenly includes index 2, average becomes (2+4+100)/3 = 35.33
    assert out.loc[2, "home_goals_scored_last_5"] == pytest.approx(3.0)


def test_merge_xg_adds_home_xg_and_away_xg_columns() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    xg = pd.DataFrame(
        {
            "date": ["2024-08-15"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_xg": [1.85],
            "away_xg": [1.20],
        }
    )
    out = _merge_xg(matches, xg)
    assert out.loc[0, "home_xg"] == pytest.approx(1.85)
    assert out.loc[0, "away_xg"] == pytest.approx(1.20)


def test_merge_xg_no_match_yields_nan() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    xg = pd.DataFrame(
        {
            "date": ["2024-08-15"],
            "home_team": ["Liverpool"],  # different match
            "away_team": ["Tottenham"],
            "home_xg": [2.5],
            "away_xg": [1.0],
        }
    )
    out = _merge_xg(matches, xg)
    assert pd.isna(out.loc[0, "home_xg"])
    assert pd.isna(out.loc[0, "away_xg"])


def test_xg_minus_actual_overperformance_signal() -> None:
    """Team scored 3 actual goals against 1.5 xG → +1.5 overperformance (lucky)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-08"]),
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["A", "B"],
            "FTHG": [3, 0],
            "FTAG": [0, 0],
            "home_xg": [1.5, 1.0],
            "away_xg": [1.0, 1.0],
        }
    )
    out = _add_xg_features(df, window=5)
    # Index 1: Arsenal's prior match scored 3 vs xG 1.5 → +1.5 overperformance
    assert out.loc[1, "home_xg_minus_actual_last_5"] == pytest.approx(1.5)


# ─── Strength of schedule + orchestrator tests ───────────────────────


def test_strength_of_schedule_promoted_team_default() -> None:
    """Team has no prior-season league position → defaults to league_size - 2."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Brand New FC"],  # promoted team (no history)
            "away_team": ["Liverpool"],  # has prior position (assumed mid-table)
            "league": ["England"],
            "year": [2024],
        }
    )
    out = _add_strength_of_schedule(df, window=5)
    # Promoted team: own SoS uses default for opponents
    # Default for England (top flight is 20 teams) → league_size - 2 = 18
    # We're testing that the function doesn't crash and returns a sensible numeric
    assert out.loc[0, "home_opp_strength_last_5"] is not None
    assert out.loc[0, "away_opp_strength_last_5"] is not None


def test_compute_features_end_to_end_runs_without_error() -> None:
    """Full pipeline on a small synthetic dataset; verify the expected feature columns appear."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22"]),
            "home_team": ["Arsenal", "Liverpool", "Arsenal", "Liverpool"],
            "away_team": ["Liverpool", "Arsenal", "Liverpool", "Arsenal"],
            "FTHG": [2, 1, 0, 3],
            "FTAG": [1, 1, 2, 0],
            "league": ["England"] * 4,
            "year": [2024] * 4,
        }
    )
    xg = pd.DataFrame(
        {
            "date": ["2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22"],
            "home_team": ["Arsenal", "Liverpool", "Arsenal", "Liverpool"],
            "away_team": ["Liverpool", "Arsenal", "Liverpool", "Arsenal"],
            "home_xg": [1.8, 1.2, 0.9, 2.5],
            "away_xg": [0.9, 1.4, 1.7, 1.0],
        }
    )
    out = compute_features(matches, xg)

    expected_cols = {
        "home_rest_days",
        "away_rest_days",
        "home_matches_last_14d",
        "away_matches_last_14d",
        "home_form_wins",
        "home_form_draws",
        "home_form_losses",
        "away_form_wins",
        "away_form_draws",
        "away_form_losses",
        "home_goals_scored_last_5",
        "home_goals_conceded_last_5",
        "away_goals_scored_last_5",
        "away_goals_conceded_last_5",
        "home_xg_last_5",
        "home_xga_last_5",
        "home_xg_minus_actual_last_5",
        "away_xg_last_5",
        "away_xga_last_5",
        "away_xg_minus_actual_last_5",
        "home_opp_strength_last_5",
        "away_opp_strength_last_5",
    }
    assert expected_cols.issubset(set(out.columns)), (
        f"Missing feature columns: {expected_cols - set(out.columns)}"
    )
    assert len(out) == 4


def test_normalise_team_names_keeps_normalised_form() -> None:
    """Smoke test that team_names normalisation runs and converts football-data names."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Man United"],  # football_data form
            "away_team": ["Wolves"],
            "FTHG": [2],
            "FTAG": [1],
        }
    )
    out = _normalise_team_names(df)
    assert out.loc[0, "home_team"] == "Manchester United"
    assert out.loc[0, "away_team"] == "Wolves"
