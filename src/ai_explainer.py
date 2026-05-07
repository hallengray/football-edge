"""LLM-powered explainer for the dashboard's value bets.

Posts to OpenRouter's OpenAI-compatible chat-completions endpoint with a
constrained system prompt that forces JSON output. Returns an ExplainerResult
dataclass; never raises to the Streamlit caller.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash:free"


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

    # Other paths to be implemented in subsequent tasks.
    return ExplainerResult(picks=[], error=None)
