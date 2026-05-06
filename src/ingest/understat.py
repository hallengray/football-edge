"""Scrape per-match xG from Understat for Big-5 European leagues.

Understat publishes match data inline in a `<script>` tag as a JSON-encoded
string assigned to `datesData`. We GET each league-season page, extract the
JSON via BeautifulSoup, decode the embedded string, and return a DataFrame
of one row per match with home/away xG.

Caches each (league, year) result as parquet under data/cache/understat/.
Failure is graceful - 404, parse errors, or rate-limit responses produce an
empty DataFrame for that season; missing matches downstream become NaN xG
features which the SimpleImputer fills with the mean.

Etiquette: SCRAPE_SLEEP-second pause between requests, identifying User-Agent.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "understat"

LEAGUE_PAGE_URL_TEMPLATE = "https://understat.com/league/{league}/{year}"

USER_AGENT = "football-edge-trainer/1.0 (personal-project)"
REQUEST_TIMEOUT = 30
SCRAPE_SLEEP = 1.5  # seconds between requests


def _cache_path(league: str, year: int) -> Path:
    return CACHE_DIR / f"{league}_{year}.parquet"


def _extract_dates_data_json(html: str) -> list[dict] | None:
    """Find the `datesData` script-tag JSON in the Understat page and decode.

    The Understat page embeds the data as: `var datesData = JSON.parse('<escaped JSON>');`
    We look for that pattern and decode the inner string.
    """
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        text = script.string or script.get_text()
        if "datesData" not in text:
            continue
        marker = "datesData = JSON.parse('"
        start = text.find(marker)
        if start == -1:
            continue
        start += len(marker)
        end = text.find("')", start)
        if end == -1:
            continue
        escaped = text[start:end]
        # The string is hex-escaped (\xHH or \uHHHH); decode via Python's string-escape
        try:
            decoded = escaped.encode("utf-8").decode("unicode_escape")
            return json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to decode Understat datesData JSON: {e}")
            return None
    return None


def fetch_league_xg(league: str, year: int) -> pd.DataFrame:
    """Fetch all matches' xG/xGA for a single league-season.

    Returns a DataFrame with columns:
        date, home_team, away_team, home_xg, away_xg, league, year

    On any failure (404, parse error, rate limit) returns an empty DataFrame.
    """
    cache_file = _cache_path(league, year)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    url = LEAGUE_PAGE_URL_TEMPLATE.format(league=league, year=year)
    if SCRAPE_SLEEP > 0:
        time.sleep(SCRAPE_SLEEP)

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(f"Understat {url} returned {resp.status_code}; skipping season")
            return pd.DataFrame()
    except requests.RequestException as e:
        logger.warning(f"Understat {url} request failed: {e}; skipping season")
        return pd.DataFrame()

    matches = _extract_dates_data_json(resp.text)
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
