"""Tests for the AI explainer. All HTTP calls are mocked via requests-mock."""

from __future__ import annotations

import pandas as pd
import pytest  # noqa: F401  # used by future tests in this module
import requests_mock

from src.ai_explainer import (
    OPENROUTER_URL,  # noqa: F401  # exported symbol; asserted by future tests
    ExplainerResult,
    Pick,  # noqa: F401  # exported symbol; asserted by future tests
    explain_top_picks,
)


# Sample inputs reused across tests
SAMPLE_VALUE_DF = pd.DataFrame(
    {
        "League": ["EPL", "Bundesliga", "Ligue 1"],
        "Match": [
            "Liverpool vs Chelsea",
            "Bayern Munich vs Borussia Dortmund",
            "Lyon vs Paris Saint-Germain",
        ],
        "Bet": ["Home: Liverpool", "Draw", "Over 2.5"],
        "Best odds": ["1.85", "3.40", "1.95"],
        "_model_prob": [0.582, 0.310, 0.520],
        "_value_pct": [0.064, 0.090, 0.052],
    }
)


SAMPLE_BACKTEST = {
    "leagues": {
        "epl": {
            "markets": {
                "home_win": {"yield_pct": -10.96, "n_bets": 842},
                "draw": {"yield_pct": -12.79, "n_bets": 507},
                "away_win": {"yield_pct": -5.08, "n_bets": 1289},
                "over_2.5": {"yield_pct": -11.92, "n_bets": 430},
                "under_2.5": {"yield_pct": -2.90, "n_bets": 820},
            }
        },
        "bundesliga": {
            "markets": {
                "home_win": {"yield_pct": -1.18, "n_bets": 615},
                "draw": {"yield_pct": 7.20, "n_bets": 679},
                "away_win": {"yield_pct": -12.92, "n_bets": 869},
                "over_2.5": {"yield_pct": -3.74, "n_bets": 725},
                "under_2.5": {"yield_pct": -22.73, "n_bets": 316},
            }
        },
        "ligue1": {
            "markets": {
                "home_win": {"yield_pct": -10.78, "n_bets": 590},
                "draw": {"yield_pct": -9.56, "n_bets": 491},
                "away_win": {"yield_pct": 1.62, "n_bets": 740},
                "over_2.5": {"yield_pct": 0.27, "n_bets": 376},
                "under_2.5": {"yield_pct": -11.72, "n_bets": 598},
            }
        },
    }
}


def test_returns_error_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with requests_mock.Mocker() as m:
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)
        assert m.call_count == 0  # no HTTP call when key missing

    assert isinstance(result, ExplainerResult)
    assert result.picks == []
    assert result.error is not None
    assert "OPENROUTER_API_KEY" in result.error


def test_returns_no_error_when_value_bets_empty(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    empty_df = pd.DataFrame(columns=SAMPLE_VALUE_DF.columns)

    with requests_mock.Mocker() as m:
        result = explain_top_picks(empty_df, SAMPLE_BACKTEST)
        assert m.call_count == 0  # no HTTP call when df is empty

    assert result.picks == []
    assert result.error is None  # empty input is not an error
