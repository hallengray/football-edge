"""Supabase client for logging predictions, bets and outcomes.

Three tables (defined in supabase/schema.sql):
    predictions — every value bet flagged by the model
    bets        — bets you actually placed (linked to a prediction)
    outcomes    — settled results (linked to a bet)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client


def get_client() -> Client:
    """Create a Supabase client from env vars (or Streamlit secrets)."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        try:
            import streamlit as st

            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_KEY")
        except Exception:
            pass

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured.")

    return create_client(url, key)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_prediction(
    client: Client,
    fixture_id: str,
    home_team: str,
    away_team: str,
    kickoff: datetime,
    market: str,
    outcome: str,
    model_probability: float,
    bookmaker: str,
    decimal_odds: float,
    value_pct: float,
    kelly_stake_fraction: float,
) -> dict[str, Any]:
    """Log a value bet flagged by the model. Idempotent on (fixture_id, market, outcome)."""
    data = {
        "fixture_id": fixture_id,
        "home_team": home_team,
        "away_team": away_team,
        "kickoff": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
        "market": market,
        "outcome": outcome,
        "model_probability": model_probability,
        "bookmaker": bookmaker,
        "decimal_odds": decimal_odds,
        "value_pct": value_pct,
        "kelly_stake_fraction": kelly_stake_fraction,
        "logged_at": _now(),
    }
    result = (
        client.table("predictions").upsert(data, on_conflict="fixture_id,market,outcome").execute()
    )
    return result.data[0] if result.data else {}


def record_bet(
    client: Client,
    prediction_id: int,
    stake_amount: float,
    notes: str = "",
) -> dict[str, Any]:
    """Mark that you actually placed a bet on a logged prediction."""
    data = {
        "prediction_id": prediction_id,
        "stake_amount": stake_amount,
        "notes": notes,
        "placed_at": _now(),
    }
    result = client.table("bets").insert(data).execute()
    return result.data[0] if result.data else {}


def record_outcome(
    client: Client,
    bet_id: int,
    won: bool,
    payout: float,
) -> dict[str, Any]:
    """Record the result of a bet after the match settles."""
    data = {
        "bet_id": bet_id,
        "won": won,
        "payout": payout,
        "settled_at": _now(),
    }
    result = client.table("outcomes").insert(data).execute()
    return result.data[0] if result.data else {}


def get_predictions(client: Client, limit: int = 200) -> list[dict[str, Any]]:
    """Fetch recent predictions, newest first."""
    return (
        client.table("predictions")
        .select("*")
        .order("logged_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )


def get_bets_with_context(client: Client) -> list[dict[str, Any]]:
    """Fetch all bets joined with their prediction + outcome (if settled)."""
    return (
        client.table("bets")
        .select("*, predictions(*), outcomes(*)")
        .order("placed_at", desc=True)
        .execute()
        .data
    )
