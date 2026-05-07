"""Fetch per-match xG from Understat for Big-5 European leagues.

Understat exposes per-league season data via an XHR endpoint at
`/getLeagueData/{league}/{year}` that returns a JSON object with three
top-level keys: `teams`, `players`, and `dates`. The `dates` array has one
entry per match with `datetime`, `h.title`, `a.title`, `xG.h`, `xG.a` -- the
exact shape our pipeline needs. Caches each (league, year) result as parquet
under data/cache/understat/.

Failure is graceful - 404, network error, or malformed response produces an
empty DataFrame for that season; missing matches downstream become NaN xG
features which the SimpleImputer fills with the mean.

Etiquette: SCRAPE_SLEEP-second pause between requests, identifying User-Agent.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "understat"

LEAGUE_DATA_URL_TEMPLATE = "https://understat.com/getLeagueData/{league}/{year}"

USER_AGENT = "football-edge-trainer/1.0 (personal-project)"
REQUEST_TIMEOUT = 30
SCRAPE_SLEEP = 1.5  # seconds between requests


def _cache_path(league: str, year: int) -> Path:
    return CACHE_DIR / f"{league}_{year}.parquet"


def fetch_league_xg(league: str, year: int) -> pd.DataFrame:
    """Fetch all matches' xG/xGA for a single league-season.

    Returns a DataFrame with columns:
        date, home_team, away_team, home_xg, away_xg, league, year

    On any failure (404, parse error, rate limit) returns an empty DataFrame.
    """
    cache_file = _cache_path(league, year)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    url = LEAGUE_DATA_URL_TEMPLATE.format(league=league, year=year)
    if SCRAPE_SLEEP > 0:
        time.sleep(SCRAPE_SLEEP)

    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(f"Understat {url} returned {resp.status_code}; skipping season")
            return pd.DataFrame()
    except requests.RequestException as e:
        logger.warning(f"Understat {url} request failed: {e}; skipping season")
        return pd.DataFrame()

    try:
        payload = resp.json()
    except ValueError as e:
        logger.warning(f"Understat {url} returned non-JSON: {e}; skipping season")
        return pd.DataFrame()

    matches = payload.get("dates") if isinstance(payload, dict) else None
    if not matches:
        logger.warning(f"No matches parsed from {url}; skipping season")
        return pd.DataFrame()

    rows: list[dict] = []
    for match in matches:
        try:
            rows.append(
                {
                    "date": match["datetime"][:10],
                    "home_team": match["h"]["title"],
                    "away_team": match["a"]["title"],
                    "home_xg": float(match["xG"]["h"]),
                    "away_xg": float(match["xG"]["a"]),
                    "league": league,
                    "year": year,
                }
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Skipping malformed Understat match in {url}: {e}")
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
    return df


def fetch_xg_data(leagues: list[str], years: list[int]) -> pd.DataFrame:
    """Fetch and concatenate xG data for the (league, year) cross-product.

    Returns the union of all per-season DataFrames; empty seasons contribute nothing.
    """
    parts: list[pd.DataFrame] = []
    for league in leagues:
        for year in years:
            df = fetch_league_xg(league, year)
            if not df.empty:
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
