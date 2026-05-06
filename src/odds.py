"""The Odds API client - Big-5 European leagues.

Fetches upcoming fixtures with bookmaker odds for all five leagues. Each league
is one API request; per-league failures are graceful (skip that league, return
others). The dashboard's @st.cache_data wraps the call to manage the 500/month
free-tier quota.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Map our internal league key -> The Odds API sport_key
LEAGUE_ENDPOINTS: dict[str, str] = {
    "epl": "soccer_epl",
    "laliga": "soccer_spain_la_liga",
    "seriea": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue1": "soccer_france_ligue_one",
}


def _fetch_one_league(
    sport_key: str,
    api_key: str,
    *,
    markets: str = "h2h,totals",
    regions: str = "uk",
) -> list[dict[str, Any]]:
    """Single Odds API call for one league. Raises on HTTP error."""
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
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


def get_big5_odds(
    api_key: str | None = None,
    markets: str = "h2h,totals",
    regions: str = "uk",
) -> list[dict[str, Any]]:
    """Fetch fixtures + odds for all 5 leagues. Tags each fixture with `league` key.

    Per-league failures are graceful - that league's fixtures are simply absent
    from the returned list. A warning is logged. This differs from training
    policy (where any failure fails the whole run); runtime fetches are best-effort.
    """
    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY is not set. Add it to your .env file.")

    fixtures: list[dict[str, Any]] = []
    for league, sport_key in LEAGUE_ENDPOINTS.items():
        try:
            league_fixtures = _fetch_one_league(
                sport_key, api_key, markets=markets, regions=regions
            )
            for fixture in league_fixtures:
                fixture["league"] = league
            fixtures.extend(league_fixtures)
        except requests.RequestException as e:
            logger.warning(
                f"Failed to fetch {league} ({sport_key}): {e}; continuing with other leagues",
                exc_info=True,
            )
    return fixtures


def best_odds_for_outcome(
    fixture: dict[str, Any],
    market: str,
    outcome_name: str,
    point: float | None = None,
) -> tuple[float, str] | None:
    """Find the best (highest) odds across all bookmakers for a given outcome.

    Args:
        fixture: A fixture dict from get_big5_odds().
        market: Market key - "h2h" or "totals".
        outcome_name: For h2h: home team name, away team name, or "Draw".
                      For totals: "Over" or "Under" (point comes from the `point` arg).
        point: Optional totals line filter (e.g. 2.5). When set, only outcomes whose
            `point` field matches are considered. Required for totals to avoid mixing
            different lines (Over 2.5 vs Over 3.5) across bookmakers.

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
                if point is not None and o.get("point") != point:
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


# Backwards-compat shim: app.py uses get_epl_odds() today; keep it as an alias
# that filters get_big5_odds() to EPL. Removed in Task 14 alongside dashboard work.
def get_epl_odds(
    api_key: str | None = None,
    markets: str = "h2h,totals",
    regions: str = "uk",
) -> list[dict[str, Any]]:
    """Deprecated: use get_big5_odds() and filter to league=='epl'."""
    return [f for f in get_big5_odds(api_key, markets, regions) if f.get("league") == "epl"]
