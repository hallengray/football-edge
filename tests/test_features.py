"""Tests for src/features.py engineered features."""

from __future__ import annotations

import pandas as pd

from src.features import _add_rest_features


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
