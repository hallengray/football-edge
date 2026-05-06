"""The Odds API client.

Fetches upcoming Premier League fixtures and bookmaker odds.
Free tier at https://the-odds-api.com gives 500 requests/month — plenty for
weekly checks.
"""

from __future__ import annotations

import os
from typing import Any

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def get_epl_odds(
    api_key: str | None = None,
    markets: str = "h2h,totals",
    regions: str = "uk",
) -> list[dict[str, Any]]:
    """Fetch upcoming EPL fixtures with bookmaker odds.

    Args:
        api_key: The Odds API key. Defaults to the ODDS_API_KEY env var.
        markets: Comma-separated markets — "h2h" is 1X2, "totals" is over/under.
        regions: Bookmaker regions — "uk" for UK books, "eu" for European, "us" for US.

    Returns:
        List of fixture dicts. Each has `home_team`, `away_team`, `commence_time`,
        and a `bookmakers` array with their offered odds.
    """
    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY is not set. Add it to your .env file.")

    url = f"{ODDS_API_BASE}/sports/soccer_epl/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def best_odds_for_outcome(
    fixture: dict[str, Any],
    market: str,
    outcome_name: str,
) -> tuple[float, str] | None:
    """Find the best (highest) odds across all bookmakers for a given outcome.

    Args:
        fixture: A fixture dict from get_epl_odds().
        market: Market key — "h2h" or "totals".
        outcome_name: For h2h: home team name, away team name, or "Draw".
                      For totals: "Over" or "Under" (with point in `point` field).

    Returns:
        (best_decimal_odds, bookmaker_title), or None if no odds available.
    """
    best_price: float | None = None
    best_book: str | None = None

    for bookmaker in fixture.get("bookmakers", []):
        for m in bookmaker.get("markets", []):
            if m.get("key") != market:
                continue
            for o in m.get("outcomes", []):
                if o.get("name") != outcome_name:
                    continue
                price = o.get("price")
                if price is None:
                    continue
                if best_price is None or price > best_price:
                    best_price = price
                    best_book = bookmaker.get("title", "Unknown")

    if best_price is None or best_book is None:
        return None
    return best_price, best_book
