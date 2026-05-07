"""Tests for the AI explainer. All HTTP calls are mocked via requests-mock."""

from __future__ import annotations

import pandas as pd
import pytest
import requests_mock

from src.ai_explainer import (
    OPENROUTER_URL,
    ExplainerResult,
    Pick,
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


def test_returns_picks_when_api_responds_with_valid_json(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    # OpenRouter wraps the model's content as a JSON-encoded string in choices[0].message.content
    inner_content = (
        '{"picks": ['
        '{"pick_id": 0, "key_reason": "Edge of 6.4% on EPL home_win.", '
        '"risk": "EPL home_win backtest yield is -10.96% so weak prior.", '
        '"model_edge_pct": 6.4},'
        '{"pick_id": 1, "key_reason": "Bundesliga draws have +7.20% backtest yield.", '
        '"risk": "Sample size 679 is moderate.", '
        '"model_edge_pct": 9.0}'
        "]}"
    )
    api_response = {"choices": [{"message": {"role": "assistant", "content": inner_content}}]}

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, json=api_response, status_code=200)
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.error is None
    assert len(result.picks) == 2
    first = result.picks[0]
    assert isinstance(first, Pick)
    assert first.pick_id == 0
    assert first.key_reason.startswith("Edge of 6.4%")
    assert first.risk.startswith("EPL home_win")
    assert first.model_edge_pct == pytest.approx(6.4)
    second = result.picks[1]
    assert second.pick_id == 1
    assert second.model_edge_pct == pytest.approx(9.0)


def test_prompt_includes_backtest_yields_and_value_bets(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    inner_content = '{"picks": []}'
    api_response = {"choices": [{"message": {"role": "assistant", "content": inner_content}}]}

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, json=api_response, status_code=200)
        explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)
        request_body = m.last_request.json()

    user_message = next(msg["content"] for msg in request_body["messages"] if msg["role"] == "user")

    # Backtest summary must be in the prompt
    assert "Bundesliga" in user_message
    assert "draw: +7.20%" in user_message  # the standout signal from the backtest
    assert "/ 679" in user_message  # n_bets for that draw market

    # Value-bets table must be in the prompt
    assert "Liverpool vs Chelsea" in user_message
    assert "Bayern Munich vs Borussia Dortmund" in user_message
    assert "| 0 |" in user_message  # row indices rendered

    # Pre-formatted numbers (no `%` or `+` in the data cells)
    assert "| 6.4 |" in user_message  # edge for row 0
    assert "| 9.0 |" in user_message  # edge for row 1


def test_handles_429_rate_limit_gracefully(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, status_code=429, text="Too Many Requests")
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.picks == []
    assert result.error is not None
    assert "OpenRouter returned 429" in result.error


def test_handles_empty_choices_array_gracefully(monkeypatch) -> None:
    """OpenRouter sometimes returns {"choices": []} on safety refusals; must not crash."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, json={"choices": []}, status_code=200)
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.picks == []
    assert result.error is not None
    assert "AI returned unexpected response" in result.error
