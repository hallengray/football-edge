"""LLM-powered explainer for the dashboard's value bets.

Posts to OpenRouter's OpenAI-compatible chat-completions endpoint with a
constrained system prompt that forces JSON output. Returns an ExplainerResult
dataclass; never raises to the Streamlit caller.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash:free"
REQUEST_TIMEOUT = 30
TEMPERATURE = 0.2

# Maps the model's snake_case league keys to display names used in the prompt.
LEAGUE_DISPLAY = {
    "epl": "EPL",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue 1",
}

MARKET_KEYS = ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]

SYSTEM_PROMPT = """\
You are a betting model explainer. You receive a table of value bets identified by
a calibrated machine-learning model and a summary of the model's historical
backtest performance per league per market.

Your job: pick the top 10 bets and explain each, using ONLY the data provided.
If fewer than 10 value bets are in the input, return all of them ranked.
Do not invent stats, recent form, injuries, news, head-to-head history, or
anything not in the inputs. If you cannot justify a pick from the provided data,
do not pick it.

For each pick, output:
- pick_id: row index from the table
- key_reason: one sentence on why it stands out, citing only fields in the inputs
- risk: one sentence on what could go wrong, citing only fields in the inputs
- model_edge_pct: the edge column value

Output only this JSON, nothing else:
{"picks": [{"pick_id": int, "key_reason": str, "risk": str, "model_edge_pct": float}, ...]}
"""


@dataclass
class Pick:
    """One AI-ranked value bet with the AI's reasoning."""

    pick_id: int
    key_reason: str
    risk: str
    model_edge_pct: float


@dataclass
class ExplainerResult:
    """Result of an AI explainer call. error is None on success or when value_df is empty."""

    picks: list[Pick] = field(default_factory=list)
    error: str | None = None


def _format_backtest(backtest: dict) -> str:
    """Render the backtest summary as multi-line text for the user prompt."""
    lines = ["Backtest yields by league and market (yield% / n_bets):"]
    for league_key, summary in backtest.get("leagues", {}).items():
        markets = summary.get("markets", {})
        parts = []
        for mk in MARKET_KEYS:
            stats = markets.get(mk, {})
            parts.append(f"{mk}: {stats.get('yield_pct', 0.0):+.2f}% / {stats.get('n_bets', 0)}")
        lines.append(f"{LEAGUE_DISPLAY.get(league_key, league_key)} — " + ", ".join(parts))
    return "\n".join(lines)


def _format_value_bets(df: pd.DataFrame) -> str:
    """Render the value-bets DataFrame as a markdown-style table for the user prompt.

    Numerical columns are rendered as plain floats (no `%`, no `+` prefix) so the
    AI can echo them back as JSON numbers without parsing.
    """
    lines = [
        "Value bets identified for this week:",
        "| idx | League | Match | Bet | model_pct | fair_pct | edge_pct |",
    ]
    for idx, row in df.iterrows():
        model_pct = row["_model_prob"] * 100
        edge_pct = row["_value_pct"] * 100
        fair_pct = model_pct - edge_pct
        lines.append(
            f"| {idx} | {row['League']} | {row['Match']} | {row['Bet']} | "
            f"{model_pct:.1f} | {fair_pct:.1f} | {edge_pct:.1f} |"
        )
    return "\n".join(lines)


def _build_user_prompt(value_bets_df: pd.DataFrame, backtest: dict) -> str:
    return (
        _format_backtest(backtest)
        + "\n\n"
        + _format_value_bets(value_bets_df)
        + "\n\nPick the top 10."
    )


def _parse_picks(api_response: dict) -> list[Pick]:
    """Pull the assistant's content out of the OpenRouter response and parse it as JSON."""
    content = api_response["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    picks = []
    for raw in parsed["picks"]:
        picks.append(
            Pick(
                pick_id=int(raw["pick_id"]),
                key_reason=str(raw["key_reason"]),
                risk=str(raw["risk"]),
                model_edge_pct=float(raw["model_edge_pct"]),
            )
        )
    return picks


def explain_top_picks(
    value_bets_df: pd.DataFrame,
    backtest_summary: dict,
) -> ExplainerResult:
    """Rank and explain the top 10 value bets via OpenRouter.

    Returns an ExplainerResult; never raises. error is set on any failure mode
    that the caller should surface to the user.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ExplainerResult(
            picks=[],
            error="Set OPENROUTER_API_KEY in .env to enable AI picks.",
        )

    if value_bets_df.empty:
        return ExplainerResult(picks=[], error=None)

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    user_prompt = _build_user_prompt(value_bets_df, backtest_summary)

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": TEMPERATURE,
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        return ExplainerResult(
            picks=[],
            error=f"Couldn't reach OpenRouter: {e}. Try again in a minute.",
        )

    if response.status_code != 200:
        snippet = response.text[:200]
        return ExplainerResult(
            picks=[],
            error=f"OpenRouter returned {response.status_code}: {snippet}. Try again in a minute.",
        )

    try:
        api_response = response.json()
        picks = _parse_picks(api_response)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        return ExplainerResult(
            picks=[],
            error=f"AI returned unexpected response ({type(e).__name__}). Click again to retry.",
        )

    return ExplainerResult(picks=picks, error=None)
