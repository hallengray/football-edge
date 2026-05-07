"""Direct download of football-data.co.uk season CSVs from the sports-betting
GitHub data mirror. Replaces the library's `_get_data` method.

Caches each (league, division, year) CSV under data/cache/football_data/.
The current season (year == current calendar year - 1, since seasons start in
August and we name them by start year) is always re-fetched because it's still
being updated weekly. The fixtures CSV is also always re-fetched.

Failure policy: a per-season download fails (404, timeout, parse error) -> log
a warning, skip that season. If more than half the requested seasons fail
across the whole call, raise RuntimeError - training on partial data silently
is a false-confidence trap.
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "football_data"

TRAINING_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/georgedouzas/sports-betting/"
    "data/data/soccer/modelling/{league}_{division}_{year}.csv"
)
FIXTURES_URL = (
    "https://raw.githubusercontent.com/georgedouzas/sports-betting/"
    "data/data/soccer/modelling/fixtures.csv"
)

REQUEST_TIMEOUT = 30
RETRY_SLEEP = 5

# The sports-betting GitHub mirror serves CSVs in a normalised snake_case
# schema. The rest of the v2 pipeline (features, training, tests) was written
# against football-data.co.uk's raw schema (FTHG, AvgH, etc.). Renaming once
# at ingest keeps downstream code unchanged. Idempotent — already-renamed
# columns are passed through.
MIRROR_TO_LEGACY_COLUMNS: dict[str, str] = {
    "target__home_team__full_time_goals": "FTHG",
    "target__away_team__full_time_goals": "FTAG",
    "odds__market_average__home_win__full_time_goals": "AvgH",
    "odds__market_average__draw__full_time_goals": "AvgD",
    "odds__market_average__away_win__full_time_goals": "AvgA",
    "odds__market_average__over_2.5__full_time_goals": "AvgOver2.5",
    "odds__market_average__under_2.5__full_time_goals": "AvgUnder2.5",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=MIRROR_TO_LEGACY_COLUMNS)


def _cache_path(league: str, division: int, year: int) -> Path:
    return CACHE_DIR / f"{league}_{division}_{year}.csv"


def _download_one_season(
    league: str,
    division: int,
    year: int,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """Returns the season DataFrame, or None if the download failed.
    Caches successful downloads to disk. Retries once on transient failures.
    """
    cache_file = _cache_path(league, division, year)
    if cache_file.exists() and not force_refresh:
        return _normalise_columns(pd.read_csv(cache_file))

    url = TRAINING_URL_TEMPLATE.format(league=league, division=division, year=year)

    for attempt in (1, 2):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                logger.warning(f"Season not found at {url} (404)")
                return None
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file, index=False)
            return _normalise_columns(df)
        except (requests.RequestException, pd.errors.ParserError) as e:
            if attempt == 1:
                logger.warning(
                    f"Attempt {attempt} for {url} failed: {e}; retrying after {RETRY_SLEEP}s"
                )
                time.sleep(RETRY_SLEEP)
            else:
                logger.warning(f"Attempt {attempt} for {url} failed: {e}; giving up")
                return None
    return None


def download_training_data(
    leagues: list[str],
    years: list[int],
) -> pd.DataFrame:
    """Download and concatenate season CSVs for the given (league, year) cross-product.

    Always uses division=1 (top flight). The Big-5 leagues we use are England,
    Spain, Italy, Germany, France - passed via `leagues` in their football-data
    naming form (capitalised English words).

    Failure: if >50% of the requested (league, year) combinations fail, raises
    RuntimeError. Otherwise returns the concatenation of the successful seasons.
    """
    requested = [(lg, yr) for lg in leagues for yr in years]
    results: list[pd.DataFrame] = []
    failures = 0

    for league, year in requested:
        df = _download_one_season(league, 1, year)
        if df is None:
            failures += 1
            continue
        df = df.copy()
        df["league"] = league
        df["year"] = year
        results.append(df)

    if failures > len(requested) // 2:
        raise RuntimeError(
            f"too many seasons failed: {failures} of {len(requested)}. "
            "Refusing to train on partial data."
        )

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def download_fixtures_data() -> pd.DataFrame:
    """Download the upcoming-fixtures CSV. Always re-fetched (no cache)."""
    resp = requests.get(FIXTURES_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))
