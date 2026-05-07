"""LLM-powered explainer for the dashboard's value bets.

Posts to OpenRouter's OpenAI-compatible chat-completions endpoint with a
constrained system prompt that forces JSON output. Returns an ExplainerResult
dataclass; never raises to the Streamlit caller.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import requests

# Used by the dashboard renderer to dispatch to the right Streamlit visual
# (st.info / st.error / st.warning) without sniffing the error message text.
ErrorKind = Literal["config", "transport", "parse"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
REQUEST_TIMEOUT = 30
# Temperature 0 maximises determinism, which matters because:
# (a) ranking the same value bets twice should produce the same picks,
# (b) free-tier models drift from the JSON schema at higher temperatures.
TEMPERATURE = 0.0
# Cap the value-bets table sent to the model at top-N by edge. Free-tier models
# emit malformed JSON when their response gets too long (observed with
# gpt-oss-120b: mismatched closing quotes once output exceeded ~2KB). Since
# value_bets_df arrives sorted by _value_pct desc, head(MAX_PROMPT_BETS) keeps
# the 50 highest-edge candidates -- the only ones that could plausibly make
# the top 10. Caller's render flow uses iloc[pick_id] against the un-capped
# DataFrame, and head() preserves the leading indices, so the mapping stays
# valid.
MAX_PROMPT_BETS = 50

# Maps the model's snake_case league keys to display names used in the prompt.
LEAGUE_DISPLAY = {
    "epl": "EPL",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue 1",
}
LEAGUE_KEY_BY_DISPLAY = {v: k for k, v in LEAGUE_DISPLAY.items()}

MARKET_KEYS = ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]

# Maps the dashboard's bet labels (the value-bets row's "outcome" / "Bet" prefix)
# to backtest market keys, so each row can be paired with its historical yield.
MARKET_KEY_BY_OUTCOME = {
    "Home": "home_win",
    "Draw": "draw",
    "Away": "away_win",
    "Over 2.5": "over_2.5",
    "Under 2.5": "under_2.5",
}

SYSTEM_PROMPT = """\
You are a betting model explainer. You receive a table of value bets identified
by a calibrated machine-learning model. Your reader is a smart non-specialist;
your job is to translate the model's call into plain English, not to repeat
betting jargon.

The table has these columns:
- League: which of the five leagues
- Match: who's playing
- Bet: the side or outcome being backed
- model_pct: the probability the model assigns this outcome (e.g. 58.2 = 58.2%)
- fair_pct: the probability implied by the bookmaker's price after removing
  their margin
- edge_pct: model_pct - fair_pct. Positive means the model thinks this outcome
  is more likely than the bookmaker is pricing it
- market_yield_pct: how the model has done historically on this league/market.
  Positive means betting this market with this model returned a profit per
  pound staked across the model's backtest; negative means it lost money
- expected_yield_pct: edge_pct + market_yield_pct -- the ranking criterion.
  Combines the current edge with how the model has actually performed in this
  market in the past. A positive number means the bet is plausibly profitable;
  a negative number means even when the edge looks attractive, the model's
  history in this market suggests it won't pay out

The rows are pre-sorted by expected_yield_pct (descending). The first 10 are
the right picks unless something is clearly off.

Pick the top 10 (or all of them if fewer than 10 are in the input). Do not
invent stats, recent form, injuries, news, head-to-head history, or anything
not in the inputs.

For each pick, output:
- pick_id: row index from the table
- key_reason: ONE sentence on why this bet stands out, in PLAIN ENGLISH for a
  smart friend who isn't a betting professional. Translate the numbers into
  meaning instead of quoting them raw. Cite specific values when they're load-
  bearing, but explain what each one means.
- risk: ONE sentence on what could go wrong, also in plain English.
- model_edge_pct: the edge_pct column value

GOOD style (plain English, explained):
  key_reason: "When this model has flagged a Bundesliga draw as a value bet
  before, those bets returned about 7p of profit per pound staked across 679
  matches -- the strongest pattern in the model's whole history -- and this
  draw price gives the model a 9-point gap over the bookmaker, so the combined
  picture is the most attractive call this week."
  risk: "Draws are inherently low-frequency events; even with a clear edge,
  most weeks you go home with nothing."

BAD style (jargon-only, do not write like this):
  key_reason: "Bundesliga draw +9% edge with +7.20% backtest yield, +16.2%
  expected, n=679."
  risk: "Variance high; sample n=679."

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
    """Result of an AI explainer call. error is None on success or when value_df is empty.

    error_kind, when set, classifies the failure for the renderer to dispatch on:
    - "config": OPENROUTER_API_KEY missing -> Streamlit st.info
    - "transport": network failure or non-200 status -> st.error
    - "parse": valid HTTP response but unexpected body shape -> st.warning

    value_df, when set, is the (capped, sorted, reset-indexed) DataFrame that
    the picks reference. Each pick.pick_id is a positional index into this df.
    Carrying the df with the result makes multi-result rendering (e.g. one
    section for top draws + another for top mixed) work without the caller
    juggling separate dfs per section.
    """

    picks: list[Pick] = field(default_factory=list)
    error: str | None = None
    error_kind: ErrorKind | None = None
    value_df: pd.DataFrame | None = None


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


def _bet_to_market_key(bet_label: str) -> str | None:
    """Resolve a value-bets row's Bet label to its backtest market key.

    The dashboard renders Bet as "Home: <team>", "Away: <team>", "Draw",
    "Over 2.5", or "Under 2.5". We strip the colon-suffix for h2h bets and
    look up the canonical market key.
    """
    head = bet_label.split(":", 1)[0].strip()
    return MARKET_KEY_BY_OUTCOME.get(head)


def compute_expected_yield(value_df: pd.DataFrame, backtest: dict) -> pd.DataFrame:
    """Augment value_df with `_market_yield_pct` and `_expected_yield_pct` columns.

    market_yield_pct: the backtest yield_pct for the row's (league, market) pair.
    Defaults to 0.0 when the league or market isn't in the backtest summary
    (e.g., a league with no historical data) -- that way unknown markets fall
    back to ranking purely by raw edge instead of being penalised.

    expected_yield_pct: edge_pct + market_yield_pct. Combines the current bet's
    edge with how the model has historically performed in that specific market.
    A bet with a high raw edge but a strongly negative market yield will sink
    in the ranking; a bet with a moderate edge in a profitable market will
    rise. This is the column the AI ranks on.
    """
    df = value_df.copy()
    leagues_data = backtest.get("leagues", {})

    market_yields: list[float] = []
    for _, row in df.iterrows():
        league_key = LEAGUE_KEY_BY_DISPLAY.get(row["League"])
        market_key = _bet_to_market_key(row["Bet"])
        if league_key is None or market_key is None:
            market_yields.append(0.0)
            continue
        stats = leagues_data.get(league_key, {}).get("markets", {}).get(market_key, {})
        market_yields.append(float(stats.get("yield_pct", 0.0)))

    df["_market_yield_pct"] = market_yields
    df["_expected_yield_pct"] = df["_value_pct"] * 100 + df["_market_yield_pct"]
    return df


def _format_value_bets(df: pd.DataFrame) -> str:
    """Render the value-bets DataFrame as a markdown-style table for the user prompt.

    Expects the df to already have `_market_yield_pct` and `_expected_yield_pct`
    columns (produced by `compute_expected_yield`). Numerical columns are
    rendered as plain floats (no `%`, no `+` prefix) so the AI can echo them
    back as JSON numbers without parsing.
    """
    lines = [
        "Value bets identified for this week (sorted by expected_yield_pct desc):",
        "| idx | League | Match | Bet | model_pct | fair_pct | edge_pct | market_yield_pct | expected_yield_pct |",
    ]
    for idx, row in df.iterrows():
        model_pct = row["_model_prob"] * 100
        edge_pct = row["_value_pct"] * 100
        fair_pct = model_pct - edge_pct
        market_yield = row["_market_yield_pct"]
        expected_yield = row["_expected_yield_pct"]
        lines.append(
            f"| {idx} | {row['League']} | {row['Match']} | {row['Bet']} | "
            f"{model_pct:.1f} | {fair_pct:.1f} | {edge_pct:.1f} | "
            f"{market_yield:+.2f} | {expected_yield:+.2f} |"
        )
    return "\n".join(lines)


def _build_user_prompt(value_bets_df: pd.DataFrame, backtest: dict) -> str:
    return (
        _format_backtest(backtest)
        + "\n\n"
        + _format_value_bets(value_bets_df)
        + "\n\nPick the top 10. Explain each pick in plain English."
    )


def _parse_picks(api_response: dict, valid_ids: set[int]) -> list[Pick]:
    """Pull the assistant's content out of the OpenRouter response and parse it as JSON.

    Picks whose pick_id is not in valid_ids are dropped silently — defends against
    the AI hallucinating row indices that don't exist in the value-bets table.
    """
    content = api_response["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    picks = []
    for raw in parsed["picks"]:
        pick_id = int(raw["pick_id"])
        if pick_id not in valid_ids:
            continue
        picks.append(
            Pick(
                pick_id=pick_id,
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
            error_kind="config",
        )

    if value_bets_df.empty:
        return ExplainerResult(picks=[], error=None)

    # Pre-rank by expected_yield_pct (edge + historical market yield) so the AI
    # sees the rows in the order it should pick from. This stops a lazy model
    # from just echoing back the input order when the input was sorted by raw
    # edge -- raw edge alone ignores how the model actually performs per market.
    # Idempotent: callers (like app.py) can pre-compute and pre-sort, in which
    # case this is a no-op. Tests pass un-augmented fixtures and rely on the
    # fallback path here.
    if "_expected_yield_pct" not in value_bets_df.columns:
        value_bets_df = compute_expected_yield(value_bets_df, backtest_summary)
        value_bets_df = value_bets_df.sort_values(
            "_expected_yield_pct", ascending=False
        ).reset_index(drop=True)

    if len(value_bets_df) > MAX_PROMPT_BETS:
        value_bets_df = value_bets_df.head(MAX_PROMPT_BETS)

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
            error_kind="transport",
        )

    if response.status_code != 200:
        snippet = response.text[:200]
        return ExplainerResult(
            picks=[],
            error=f"OpenRouter returned {response.status_code}: {snippet}. Try again in a minute.",
            error_kind="transport",
        )

    try:
        api_response = response.json()
        picks = _parse_picks(api_response, valid_ids=set(value_bets_df.index))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, IndexError) as e:
        return ExplainerResult(
            picks=[],
            error=f"AI returned unexpected response ({type(e).__name__}). Click again to retry.",
            error_kind="parse",
        )

    return ExplainerResult(picks=picks, error=None, value_df=value_bets_df)


def explain_top_picks_in_market(
    value_bets_df: pd.DataFrame,
    backtest_summary: dict,
    market_key: str,
    top_n: int = 5,
) -> ExplainerResult:
    """Rank and explain the top N value bets restricted to one market.

    market_key must be one of the keys in MARKET_KEY_BY_OUTCOME's values:
    "home_win", "draw", "away_win", "over_2.5", "under_2.5".

    Pre-filters value_bets_df to rows whose Bet maps to the requested market,
    sorts by expected_yield_pct desc, takes top_n, then delegates to
    explain_top_picks. The returned ExplainerResult.value_df is the filtered
    df, so renderers can iloc[pick.pick_id] against it directly.
    """
    df = compute_expected_yield(value_bets_df, backtest_summary)
    df = df[df["Bet"].apply(lambda b: _bet_to_market_key(b) == market_key)]
    df = df.sort_values("_expected_yield_pct", ascending=False).head(top_n)
    df = df.reset_index(drop=True)
    return explain_top_picks(df, backtest_summary)
