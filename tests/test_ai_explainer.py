"""Tests for the AI explainer. All HTTP calls are mocked via requests-mock."""

from __future__ import annotations

import pandas as pd
import pytest
import requests_mock

from src.ai_explainer import (
    OPENROUTER_URL,
    ExplainerResult,
    Pick,
    compute_expected_yield,
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
    assert result.error_kind == "config"


def test_returns_no_error_when_value_bets_empty(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    empty_df = pd.DataFrame(columns=SAMPLE_VALUE_DF.columns)

    with requests_mock.Mocker() as m:
        result = explain_top_picks(empty_df, SAMPLE_BACKTEST)
        assert m.call_count == 0  # no HTTP call when df is empty

    assert result.picks == []
    assert result.error is None  # empty input is not an error
    assert result.error_kind is None


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
    assert "| 6.4 |" in user_message  # edge for the Liverpool row (now at row 2)
    assert "| 9.0 |" in user_message  # edge for the Bayern row (now at row 0)

    # Per-row historical-yield columns added so the AI doesn't have to
    # cross-reference the backtest summary section manually.
    assert "market_yield_pct" in user_message
    assert "expected_yield_pct" in user_message
    # Bayern draw row (now idx 0): edge 9.0 + Bundesliga draw yield +7.20 = +16.20 expected.
    assert "| +7.20 |" in user_message  # market_yield for the Bundesliga draw row
    assert "| +16.20 |" in user_message  # expected_yield for that row


def test_compute_expected_yield_combines_edge_with_market_yield() -> None:
    """expected_yield_pct = edge_pct + market_yield_pct (additive blend)."""
    df = compute_expected_yield(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    # Bayern draw: edge 9.0 + Bundesliga draw yield +7.20 = +16.20 expected
    bayern = df[df["Match"] == "Bayern Munich vs Borussia Dortmund"].iloc[0]
    assert bayern["_market_yield_pct"] == pytest.approx(7.20)
    assert bayern["_expected_yield_pct"] == pytest.approx(16.20)

    # Liverpool home_win: edge 6.4 + EPL home_win yield -10.96 = -4.56 expected
    liverpool = df[df["Match"] == "Liverpool vs Chelsea"].iloc[0]
    assert liverpool["_market_yield_pct"] == pytest.approx(-10.96)
    assert liverpool["_expected_yield_pct"] == pytest.approx(-4.56)

    # Lyon over_2.5: edge 5.2 + Ligue 1 over_2.5 yield +0.27 = +5.47 expected
    lyon = df[df["Match"] == "Lyon vs Paris Saint-Germain"].iloc[0]
    assert lyon["_market_yield_pct"] == pytest.approx(0.27)
    assert lyon["_expected_yield_pct"] == pytest.approx(5.47)


def test_compute_expected_yield_defaults_to_zero_for_unknown_market() -> None:
    """A league or market not in the backtest summary must default to 0% market yield
    so the bet falls back to ranking purely by raw edge instead of being penalised."""
    df = pd.DataFrame(
        {
            "League": ["Some Made-Up League"],
            "Match": ["Foo vs Bar"],
            "Bet": ["Draw"],
            "Best odds": ["3.40"],
            "_model_prob": [0.40],
            "_value_pct": [0.10],
        }
    )
    out = compute_expected_yield(df, SAMPLE_BACKTEST)
    assert out["_market_yield_pct"].iloc[0] == 0.0
    assert out["_expected_yield_pct"].iloc[0] == pytest.approx(10.0)  # just the edge


def test_handles_429_rate_limit_gracefully(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, status_code=429, text="Too Many Requests")
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.picks == []
    assert result.error is not None
    assert "OpenRouter returned 429" in result.error
    assert result.error_kind == "transport"


def test_handles_empty_choices_array_gracefully(monkeypatch) -> None:
    """OpenRouter sometimes returns {"choices": []} on safety refusals; must not crash."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, json={"choices": []}, status_code=200)
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.picks == []
    assert result.error is not None
    assert "AI returned unexpected response" in result.error
    assert result.error_kind == "parse"


def test_drops_picks_with_unknown_pick_id(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    # SAMPLE_VALUE_DF has indices [0, 1, 2]. AI fabricates pick_id 999.
    inner_content = (
        '{"picks": ['
        '{"pick_id": 0, "key_reason": "real", "risk": "real", "model_edge_pct": 6.4},'
        '{"pick_id": 999, "key_reason": "hallucinated", "risk": "fake", "model_edge_pct": 99.9},'
        '{"pick_id": 2, "key_reason": "real", "risk": "real", "model_edge_pct": 5.2}'
        "]}"
    )
    api_response = {"choices": [{"message": {"role": "assistant", "content": inner_content}}]}

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, json=api_response, status_code=200)
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.error is None
    assert [p.pick_id for p in result.picks] == [0, 2]  # 999 dropped, 0 and 2 kept in order
