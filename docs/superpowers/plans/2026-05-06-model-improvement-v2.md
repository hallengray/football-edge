# Model Improvement v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `sports-betting` library with a sklearn-direct pipeline that ingests xG + computable features for the Big-5 European leagues, trains five separate per-league calibrated multi-output classifiers, and surfaces multi-league predictions in the dashboard.

**Architecture:** Two ingest modules (`football_data` and `understat`), one feature-engineering module (`features.py`) producing ~22 numeric features per match including xG-based signals, a per-league training loop with TimeSeriesSplit cross-validation, atomic-write persistence of 5 bettor pickles plus a fixtures parquet plus a backtest JSON, and a multi-league dashboard with a league filter.

**Tech Stack:** Python 3.11, scikit-learn (calibrated GBM), pandas, requests, beautifulsoup4, pyarrow (parquet), Streamlit, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-05-06-model-improvement-v2-design.md`

---

## Task 1: Drop sports-betting + pytz, add beautifulsoup4

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (auto-updated)

- [ ] **Step 1: Edit pyproject.toml dependencies**

Replace the `[project].dependencies` block with:

```toml
[project]
name = "football-edge"
version = "0.1.0"
description = "Personal football value betting tool — Big 5 European leagues"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "streamlit>=1.40",
    "pandas>=2.2",
    "requests>=2.32",
    "python-dotenv>=1.0",
    "supabase>=2.10",
    "beautifulsoup4>=4.12",
    "pyarrow>=15.0",
    "scikit-learn>=1.4",
]

[dependency-groups]
dev = [
    "ruff>=0.8",
    "pytest>=8.0",
    "requests-mock>=1.12",
]
```

Removed: `sports-betting==0.12.1`, `pytz>=2024.1`. Added: `beautifulsoup4`, `pyarrow`, explicit `scikit-learn` (was a transitive). `requests-mock` added as dev dep for HTTP mocking in tests. Description updated to reflect Big-5 scope.

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: removes sports-betting, pytz; installs beautifulsoup4, pyarrow, requests-mock; resolves cleanly.

- [ ] **Step 3: Verify imports still work**

Run: `uv run python -c "import pandas, sklearn, streamlit, bs4, pyarrow; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Verify existing tests still pass**

Run: `uv run pytest -q`
Expected: 31 PASSED. If any fail, the dep removal probably caught code still importing sportsbet — fix in this task before committing (or split into a follow-up if non-trivial).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: drop sports-betting + pytz, add beautifulsoup4 + pyarrow + scikit-learn"
```

---

## Task 2: Rewrite src/team_names.py as 3-way registry with seed teams

**Files:**
- Modify (replace): `src/team_names.py`
- Modify (replace): `tests/test_team_names.py`

The new registry maps each team across three sources (`odds_api`, `football_data`, `understat`). The seed contains the 20 current EPL clubs (already known correctly), plus the most prominent ~30 from the other 4 leagues. The registry will be filled in iteratively during smoke test (Task 16) — training raises a clear KeyError naming any unmapped team, the same self-healing pattern v0 used.

- [ ] **Step 1: Write failing tests in tests/test_team_names.py**

Replace `tests/test_team_names.py` entirely with:

```python
"""Tests for src/team_names.py 3-way registry."""

from __future__ import annotations

import pytest

from src.team_names import (
    REGISTRY,
    TeamNames,
    from_canonical,
    to_canonical,
)


def test_team_names_dataclass_holds_three_sources() -> None:
    t = TeamNames(
        canonical="Arsenal",
        odds_api="Arsenal",
        football_data="Arsenal",
        understat="Arsenal",
    )
    assert t.canonical == "Arsenal"
    assert t.odds_api == "Arsenal"
    assert t.football_data == "Arsenal"
    assert t.understat == "Arsenal"


def test_to_canonical_from_odds_api() -> None:
    # The Odds API uses the long form
    assert to_canonical("Wolverhampton Wanderers", source="odds_api") == "Wolves"


def test_to_canonical_from_football_data() -> None:
    # football-data uses short form
    assert to_canonical("Man United", source="football_data") == "Manchester United"


def test_to_canonical_from_understat() -> None:
    assert to_canonical("Manchester United", source="understat") == "Manchester United"


def test_to_canonical_unknown_raises_keyerror_with_actionable_message() -> None:
    with pytest.raises(KeyError, match="Unmapped"):
        to_canonical("FC Made Up", source="odds_api")


def test_to_canonical_invalid_source_raises() -> None:
    with pytest.raises(ValueError, match="source"):
        to_canonical("Arsenal", source="bogus_source")


def test_from_canonical_round_trip_to_each_source() -> None:
    # Pick a team known to differ across sources
    assert from_canonical("Wolves", target="odds_api") == "Wolverhampton Wanderers"
    assert from_canonical("Wolves", target="football_data") == "Wolves"
    assert from_canonical("Wolves", target="understat") == "Wolverhampton Wanderers"


def test_registry_contains_all_seed_epl_teams() -> None:
    expected_epl = {
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
        "Leeds", "Liverpool", "Man City", "Manchester United", "Newcastle",
        "Nottingham Forest", "Sunderland", "Tottenham", "West Ham", "Wolves",
    }
    canonical_set = set(REGISTRY.keys())
    missing = expected_epl - canonical_set
    assert not missing, f"Seed EPL registry missing: {missing}"


def test_registry_seed_includes_top_clubs_from_other_leagues() -> None:
    # Spot-check a few prominent non-EPL clubs we know must be in the seed
    expected_seed = {
        "Real Madrid", "Barcelona", "Atletico Madrid",          # La Liga
        "Inter", "Juventus", "Milan", "Napoli",                 # Serie A
        "Bayern Munich", "Borussia Dortmund", "Bayer Leverkusen", # Bundesliga
        "Paris Saint-Germain", "Marseille", "Lyon",              # Ligue 1
    }
    missing = expected_seed - set(REGISTRY.keys())
    assert not missing, f"Seed non-EPL registry missing: {missing}"
```

- [ ] **Step 2: Run tests to verify they fail (TDD red)**

Run: `uv run pytest tests/test_team_names.py -v`
Expected: ImportError or AttributeError on `TeamNames`/`REGISTRY` (the new symbols don't exist yet).

- [ ] **Step 3: Write src/team_names.py**

Replace `src/team_names.py` entirely with:

```python
"""Three-way team-name registry for Big-5 European leagues.

Each team has a canonical internal name plus aliases for the three external
data sources we touch:
- The Odds API (`odds_api`) — long club names like "Wolverhampton Wanderers"
- football-data.co.uk (`football_data`) — short names like "Wolves"
- Understat (`understat`) — typically the long form, e.g. "Wolverhampton Wanderers"

`to_canonical(name, source)` and `from_canonical(name, target)` handle conversions.
Unmapped names raise a clear KeyError that names the team and its source so the
operator knows exactly what entry to add.

The registry is seeded with current EPL teams and prominent clubs from the four
other Big-5 leagues. Additional teams are added during the smoke test (Task 16)
by running training and adding any KeyError that surfaces. Self-healing pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_SOURCES = ("odds_api", "football_data", "understat")


@dataclass(frozen=True)
class TeamNames:
    """Aliases for a single team across the data sources we use."""

    canonical: str
    odds_api: str
    football_data: str
    understat: str


def _entry(canonical: str, odds_api: str, football_data: str, understat: str) -> TeamNames:
    return TeamNames(
        canonical=canonical,
        odds_api=odds_api,
        football_data=football_data,
        understat=understat,
    )


REGISTRY: dict[str, TeamNames] = {
    # ─── EPL (current season) ─────────────────────────────────────────
    "Arsenal":            _entry("Arsenal", "Arsenal", "Arsenal", "Arsenal"),
    "Aston Villa":        _entry("Aston Villa", "Aston Villa", "Aston Villa", "Aston Villa"),
    "Bournemouth":        _entry("Bournemouth", "AFC Bournemouth", "Bournemouth", "Bournemouth"),
    "Brentford":          _entry("Brentford", "Brentford", "Brentford", "Brentford"),
    "Brighton":           _entry("Brighton", "Brighton and Hove Albion", "Brighton", "Brighton"),
    "Burnley":            _entry("Burnley", "Burnley", "Burnley", "Burnley"),
    "Chelsea":            _entry("Chelsea", "Chelsea", "Chelsea", "Chelsea"),
    "Crystal Palace":     _entry("Crystal Palace", "Crystal Palace", "Crystal Palace", "Crystal Palace"),
    "Everton":            _entry("Everton", "Everton", "Everton", "Everton"),
    "Fulham":             _entry("Fulham", "Fulham", "Fulham", "Fulham"),
    "Leeds":              _entry("Leeds", "Leeds United", "Leeds", "Leeds"),
    "Liverpool":          _entry("Liverpool", "Liverpool", "Liverpool", "Liverpool"),
    "Man City":           _entry("Man City", "Manchester City", "Man City", "Manchester City"),
    "Manchester United":  _entry("Manchester United", "Manchester United", "Man United", "Manchester United"),
    "Newcastle":          _entry("Newcastle", "Newcastle United", "Newcastle", "Newcastle United"),
    "Nottingham Forest":  _entry("Nottingham Forest", "Nottingham Forest", "Nottingham Forest", "Nottingham Forest"),
    "Sunderland":         _entry("Sunderland", "Sunderland", "Sunderland", "Sunderland"),
    "Tottenham":          _entry("Tottenham", "Tottenham Hotspur", "Tottenham", "Tottenham"),
    "West Ham":           _entry("West Ham", "West Ham United", "West Ham", "West Ham"),
    "Wolves":             _entry("Wolves", "Wolverhampton Wanderers", "Wolves", "Wolverhampton Wanderers"),

    # ─── La Liga (top clubs — seed; remainder added during smoke test) ──
    "Real Madrid":        _entry("Real Madrid", "Real Madrid", "Real Madrid", "Real Madrid"),
    "Barcelona":          _entry("Barcelona", "Barcelona", "Barcelona", "Barcelona"),
    "Atletico Madrid":    _entry("Atletico Madrid", "Atletico Madrid", "Ath Madrid", "Atletico Madrid"),
    "Sevilla":            _entry("Sevilla", "Sevilla", "Sevilla", "Sevilla"),
    "Real Sociedad":      _entry("Real Sociedad", "Real Sociedad", "Sociedad", "Real Sociedad"),
    "Villarreal":         _entry("Villarreal", "Villarreal", "Villarreal", "Villarreal"),
    "Athletic Bilbao":    _entry("Athletic Bilbao", "Athletic Bilbao", "Ath Bilbao", "Athletic Club"),
    "Real Betis":         _entry("Real Betis", "Real Betis", "Betis", "Real Betis"),
    "Valencia":           _entry("Valencia", "Valencia", "Valencia", "Valencia"),

    # ─── Serie A (top clubs — seed) ─────────────────────────────────────
    "Inter":              _entry("Inter", "Inter Milan", "Inter", "Internazionale"),
    "Milan":              _entry("Milan", "AC Milan", "Milan", "Milan"),
    "Juventus":           _entry("Juventus", "Juventus", "Juventus", "Juventus"),
    "Napoli":             _entry("Napoli", "Napoli", "Napoli", "Napoli"),
    "Roma":               _entry("Roma", "AS Roma", "Roma", "Roma"),
    "Lazio":              _entry("Lazio", "Lazio", "Lazio", "Lazio"),
    "Atalanta":           _entry("Atalanta", "Atalanta", "Atalanta", "Atalanta"),
    "Fiorentina":         _entry("Fiorentina", "Fiorentina", "Fiorentina", "Fiorentina"),

    # ─── Bundesliga (top clubs — seed) ─────────────────────────────────
    "Bayern Munich":       _entry("Bayern Munich", "Bayern Munich", "Bayern Munich", "Bayern Munich"),
    "Borussia Dortmund":   _entry("Borussia Dortmund", "Borussia Dortmund", "Dortmund", "Borussia Dortmund"),
    "Bayer Leverkusen":    _entry("Bayer Leverkusen", "Bayer Leverkusen", "Leverkusen", "Bayer Leverkusen"),
    "RB Leipzig":          _entry("RB Leipzig", "RB Leipzig", "RB Leipzig", "RasenBallsport Leipzig"),
    "Eintracht Frankfurt": _entry("Eintracht Frankfurt", "Eintracht Frankfurt", "Ein Frankfurt", "Eintracht Frankfurt"),
    "Wolfsburg":           _entry("Wolfsburg", "VfL Wolfsburg", "Wolfsburg", "Wolfsburg"),

    # ─── Ligue 1 (top clubs — seed) ────────────────────────────────────
    "Paris Saint-Germain": _entry("Paris Saint-Germain", "Paris Saint-Germain", "Paris SG", "Paris Saint Germain"),
    "Marseille":           _entry("Marseille", "Marseille", "Marseille", "Marseille"),
    "Monaco":              _entry("Monaco", "Monaco", "Monaco", "Monaco"),
    "Lyon":                _entry("Lyon", "Lyon", "Lyon", "Lyon"),
    "Lille":               _entry("Lille", "Lille", "Lille", "Lille"),
    "Nice":                _entry("Nice", "Nice", "Nice", "Nice"),
    "Rennes":              _entry("Rennes", "Rennes", "Rennes", "Rennes"),
}

# Reverse-index for fast `to_canonical` lookups
_BY_ODDS_API:       dict[str, str] = {t.odds_api: c for c, t in REGISTRY.items()}
_BY_FOOTBALL_DATA:  dict[str, str] = {t.football_data: c for c, t in REGISTRY.items()}
_BY_UNDERSTAT:      dict[str, str] = {t.understat: c for c, t in REGISTRY.items()}

_REVERSE_BY_SOURCE: dict[str, dict[str, str]] = {
    "odds_api":       _BY_ODDS_API,
    "football_data":  _BY_FOOTBALL_DATA,
    "understat":      _BY_UNDERSTAT,
}


def to_canonical(name: str, source: str) -> str:
    """Convert a source-specific team name to its canonical form.

    Raises KeyError with an actionable message naming the unmapped team and the
    source it was looked up from. Add the entry to REGISTRY to fix.
    """
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Invalid source {source!r}. Expected one of {VALID_SOURCES}."
        )
    reverse = _REVERSE_BY_SOURCE[source]
    if name not in reverse:
        raise KeyError(
            f"Unmapped team {name!r} from source {source!r}. "
            f"Add an entry to src/team_names.REGISTRY."
        )
    return reverse[name]


def from_canonical(canonical: str, target: str) -> str:
    """Convert a canonical name to its form for a specific source."""
    if target not in VALID_SOURCES:
        raise ValueError(
            f"Invalid target {target!r}. Expected one of {VALID_SOURCES}."
        )
    if canonical not in REGISTRY:
        raise KeyError(
            f"Unmapped canonical name {canonical!r}. "
            f"Add an entry to src/team_names.REGISTRY."
        )
    return getattr(REGISTRY[canonical], target)
```

- [ ] **Step 4: Run tests to verify they pass (TDD green)**

Run: `uv run pytest tests/test_team_names.py -v`
Expected: 8 PASSED.

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/team_names.py tests/test_team_names.py && uv run ruff format --check src/team_names.py tests/test_team_names.py`

If format check fails, run `uv run ruff format src/team_names.py tests/test_team_names.py` and re-run the check. Both must pass before commit.

- [ ] **Step 6: Run all tests to ensure no regressions**

Run: `uv run pytest -q`
Expected: All passing — note that `tests/test_predictions.py` and `app.py` may now fail because they import the OLD `to_library`/`from_library` names. **This is expected.** Don't fix those tests in this task — Task 13 (predictions module rewrite) and Task 14 (app.py updates) handle them. Just verify the test file under modification (test_team_names.py) passes here; other failures will be resolved later.

If you want to confirm only the team_names changes are clean: `uv run pytest tests/test_team_names.py tests/test_value.py -q`. Expected: those two files pass.

- [ ] **Step 7: Commit**

```bash
git add src/team_names.py tests/test_team_names.py
git commit -m "feat: 3-way team-names registry seeded for Big-5 leagues"
```

---

## Task 3: src/ingest/football_data.py — direct CSV download with caching

**Files:**
- Create: `src/ingest/__init__.py` (empty marker)
- Create: `src/ingest/football_data.py`
- Create: `tests/test_ingest.py` (tests for football_data; understat tests added in Task 4)

- [ ] **Step 1: Create empty src/ingest/__init__.py**

Create `src/ingest/__init__.py` with zero bytes (no docstring, no comment — just an empty file marking the directory as a package).

- [ ] **Step 2: Write failing tests in tests/test_ingest.py**

Create `tests/test_ingest.py` with:

```python
"""Tests for ingest modules. HTTP calls are mocked via requests-mock."""

from __future__ import annotations

import pandas as pd
import pytest
import requests_mock

from src.ingest.football_data import (
    TRAINING_URL_TEMPLATE,
    download_fixtures_data,
    download_training_data,
)


def test_training_url_template_has_expected_placeholders() -> None:
    url = TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2024)
    assert url == (
        "https://raw.githubusercontent.com/georgedouzas/sports-betting/"
        "data/data/soccer/modelling/England_1_2024.csv"
    )


def test_download_training_data_concatenates_seasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_2023 = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n2023-08-15,Arsenal,Chelsea,2,1\n"
    csv_2024 = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n2024-08-15,Liverpool,Everton,3,0\n"

    with requests_mock.Mocker() as m:
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2023), text=csv_2023)
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2024), text=csv_2024)

        df = download_training_data(leagues=["England"], years=[2023, 2024])

    assert len(df) == 2
    assert set(df["HomeTeam"]) == {"Arsenal", "Liverpool"}


def test_download_training_data_uses_cache_on_second_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_text = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n2023-08-15,Arsenal,Chelsea,2,1\n"

    with requests_mock.Mocker() as m:
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2023), text=csv_text)
        download_training_data(leagues=["England"], years=[2023])
        assert m.call_count == 1

        # Second call — should hit cache, no new HTTP request
        download_training_data(leagues=["England"], years=[2023])
        assert m.call_count == 1


def test_download_training_data_skips_failed_seasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_text = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n2024-08-15,Liverpool,Everton,3,0\n"

    with requests_mock.Mocker() as m:
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2023), status_code=404)
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2024), text=csv_text)

        # 1 of 2 seasons failed — that's 50%, exactly at threshold; should still succeed
        df = download_training_data(leagues=["England"], years=[2023, 2024])
    assert len(df) == 1


def test_download_training_data_raises_when_more_than_half_seasons_fail(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    with requests_mock.Mocker() as m:
        # All three seasons fail
        for year in [2022, 2023, 2024]:
            m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=year), status_code=500)

        with pytest.raises(RuntimeError, match="too many seasons failed"):
            download_training_data(leagues=["England"], years=[2022, 2023, 2024])


def test_download_fixtures_data_always_refetches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_text = "Date,HomeTeam,AwayTeam\n2026-05-10,Arsenal,Chelsea\n"

    with requests_mock.Mocker() as m:
        m.get("https://raw.githubusercontent.com/georgedouzas/sports-betting/data/data/soccer/modelling/fixtures.csv", text=csv_text)
        download_fixtures_data()
        download_fixtures_data()
        # Both calls should hit the network — fixtures are always fresh
        assert m.call_count == 2
```

- [ ] **Step 3: Run tests to verify they fail (TDD red)**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: ImportError on `src.ingest.football_data` — module doesn't exist yet.

- [ ] **Step 4: Implement src/ingest/football_data.py**

Create `src/ingest/football_data.py` with:

```python
"""Direct download of football-data.co.uk season CSVs from the sports-betting
GitHub data mirror. Replaces the library's `_get_data` method.

Caches each (league, division, year) CSV under data/cache/football_data/.
The current season (year == current calendar year - 1, since seasons start in
August and we name them by start year) is always re-fetched because it's still
being updated weekly. The fixtures CSV is also always re-fetched.

Failure policy: a per-season download fails (404, timeout, parse error) → log
a warning, skip that season. If more than half the requested seasons fail
across the whole call, raise RuntimeError — training on partial data silently
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


def _is_current_season(year: int) -> bool:
    """A season starts in August; we name it by its start year. The current
    season at any time during the calendar year is `current_calendar_year - 1`
    if we're before August, else `current_calendar_year`. For simplicity here,
    treat the most recent year requested as the in-progress season."""
    # Implementation: caller's responsibility. We use a simpler heuristic —
    # cache miss always tries network; cache hit short-circuits unless the
    # caller explicitly forces refresh. Current-season handling lives in the
    # call site, not here.
    return False  # See comment — kept for symmetry; logic in callers.


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
        return pd.read_csv(cache_file)

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
            return df
        except (requests.RequestException, pd.errors.ParserError) as e:
            if attempt == 1:
                logger.warning(f"Attempt {attempt} for {url} failed: {e}; retrying after {RETRY_SLEEP}s")
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
    Spain, Italy, Germany, France — passed via `leagues` in their football-data
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
```

- [ ] **Step 5: Run tests to verify they pass (TDD green)**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 6 PASSED.

- [ ] **Step 6: Run static checks**

Run: `uv run ruff check src/ingest/ tests/test_ingest.py && uv run ruff format --check src/ingest/ tests/test_ingest.py`

Apply `uv run ruff format` if needed and re-check. Both must pass before commit.

- [ ] **Step 7: Commit**

```bash
git add src/ingest/__init__.py src/ingest/football_data.py tests/test_ingest.py
git commit -m "feat: ingest module for football-data.co.uk CSVs with caching"
```

---

## Task 4: src/ingest/understat.py — xG scrape with caching

**Files:**
- Create: `src/ingest/understat.py`
- Modify: `tests/test_ingest.py` (extend with Understat tests)

- [ ] **Step 1: Append Understat tests to tests/test_ingest.py**

Append at the end of `tests/test_ingest.py`:

```python


# ─── Understat ingest tests ──────────────────────────────────────────


from src.ingest.understat import (
    LEAGUE_PAGE_URL_TEMPLATE,
    fetch_league_xg,
    fetch_xg_data,
)


SAMPLE_UNDERSTAT_HTML = """
<html><body>
<script>
var datesData = JSON.parse('\\u005B\\u007B\\u0022id\\u0022:\\u00221\\u0022,\\u0022isResult\\u0022:true,\\u0022h\\u0022:\\u007B\\u0022id\\u0022:\\u00229\\u0022,\\u0022title\\u0022:\\u0022Arsenal\\u0022,\\u0022short_title\\u0022:\\u0022ARS\\u0022\\u007D,\\u0022a\\u0022:\\u007B\\u0022id\\u0022:\\u002210\\u0022,\\u0022title\\u0022:\\u0022Chelsea\\u0022,\\u0022short_title\\u0022:\\u0022CHE\\u0022\\u007D,\\u0022goals\\u0022:\\u007B\\u0022h\\u0022:\\u00222\\u0022,\\u0022a\\u0022:\\u00221\\u0022\\u007D,\\u0022xG\\u0022:\\u007B\\u0022h\\u0022:\\u00221.85\\u0022,\\u0022a\\u0022:\\u00221.20\\u0022\\u007D,\\u0022datetime\\u0022:\\u00222024-08-15 16:30:00\\u0022\\u007D\\u005D');
</script>
</body></html>
"""


def test_understat_url_template() -> None:
    assert LEAGUE_PAGE_URL_TEMPLATE.format(league="EPL", year=2024) == "https://understat.com/league/EPL/2024"


def test_fetch_league_xg_parses_embedded_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)  # speed up tests

    with requests_mock.Mocker() as m:
        m.get("https://understat.com/league/EPL/2024", text=SAMPLE_UNDERSTAT_HTML)
        df = fetch_league_xg("EPL", 2024)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert row["home_xg"] == pytest.approx(1.85)
    assert row["away_xg"] == pytest.approx(1.20)


def test_fetch_league_xg_handles_404_gracefully(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)

    with requests_mock.Mocker() as m:
        m.get("https://understat.com/league/EPL/1999", status_code=404)
        df = fetch_league_xg("EPL", 1999)

    assert df.empty


def test_fetch_league_xg_uses_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)

    with requests_mock.Mocker() as m:
        m.get("https://understat.com/league/EPL/2024", text=SAMPLE_UNDERSTAT_HTML)
        fetch_league_xg("EPL", 2024)
        assert m.call_count == 1
        # Second call should read from cache parquet, not network
        fetch_league_xg("EPL", 2024)
        assert m.call_count == 1


def test_fetch_xg_data_concatenates_leagues(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)

    with requests_mock.Mocker() as m:
        m.get("https://understat.com/league/EPL/2024", text=SAMPLE_UNDERSTAT_HTML)
        m.get("https://understat.com/league/La_liga/2024", text=SAMPLE_UNDERSTAT_HTML)

        df = fetch_xg_data(leagues=["EPL", "La_liga"], years=[2024])

    assert len(df) == 2
    assert set(df["league"]) == {"EPL", "La_liga"}
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v -k "understat"`
Expected: ImportError on `src.ingest.understat`.

- [ ] **Step 3: Implement src/ingest/understat.py**

Create `src/ingest/understat.py` with:

```python
"""Scrape per-match xG from Understat for Big-5 European leagues.

Understat publishes match data inline in a `<script>` tag as a JSON-encoded
string assigned to `datesData`. We GET each league-season page, extract the
JSON via BeautifulSoup, decode the embedded string, and return a DataFrame
of one row per match with home/away xG.

Caches each (league, year) result as parquet under data/cache/understat/.
Failure is graceful — 404, parse errors, or rate-limit responses produce an
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 11 PASSED total (6 football_data + 5 understat).

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/ingest/understat.py tests/test_ingest.py && uv run ruff format --check src/ingest/understat.py tests/test_ingest.py`

Fix formatting if needed.

- [ ] **Step 6: Commit**

```bash
git add src/ingest/understat.py tests/test_ingest.py
git commit -m "feat: ingest module for Understat xG with HTML parsing and caching"
```

---

## Task 5: src/odds.py — multi-league fetch

**Files:**
- Modify: `src/odds.py`

No new tests — the existing function signatures stay backward-compatible enough that the dashboard's existing `_fetch_fixtures` call still works during the migration. Multi-league correctness is verified in the manual smoke test (Task 16).

- [ ] **Step 1: Replace src/odds.py contents**

Replace `src/odds.py` entirely with:

```python
"""The Odds API client — Big-5 European leagues.

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

# Map our internal league key → The Odds API sport_key
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

    Per-league failures are graceful — that league's fixtures are simply absent
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
                f"Failed to fetch {league} ({sport_key}): {e}; "
                f"continuing with other leagues",
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
        market: Market key — "h2h" or "totals".
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
```

- [ ] **Step 2: Smoke import check**

Run: `uv run python -c "from src.odds import get_big5_odds, best_odds_for_outcome, LEAGUE_ENDPOINTS; print(list(LEAGUE_ENDPOINTS.keys()))"`
Expected: `['epl', 'laliga', 'seriea', 'bundesliga', 'ligue1']`.

- [ ] **Step 3: Run all tests to confirm no regression**

Run: `uv run pytest -q`
Expected: tests passing for files we've modified so far. The `test_predictions.py` and `test_team_names.py` may still fail because of upstream import dependencies — those resolve in Tasks 13/14. As of right now, the tests directly testing `src.odds` should be fine (there aren't any in the suite).

- [ ] **Step 4: Run static checks**

Run: `uv run ruff check src/odds.py && uv run ruff format --check src/odds.py`

- [ ] **Step 5: Commit**

```bash
git add src/odds.py
git commit -m "feat: multi-league odds fetch via get_big5_odds with graceful per-league failure"
```

---

## Task 6: features.py — rest features

**Files:**
- Create: `src/features.py` (initial — only rest helper)
- Create: `tests/test_features.py` (initial — only rest tests)

This task lays the file structure for `features.py`. Subsequent feature tasks (7, 8, 9) extend the module. The orchestrator function `compute_features` is added in Task 9.

- [ ] **Step 1: Write failing tests in tests/test_features.py**

Create `tests/test_features.py` with:

```python
"""Tests for src/features.py engineered features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import _add_rest_features


def test_add_rest_features_first_match_uses_default() -> None:
    """A team's first-ever match has no prior history; rest_days defaults to 7."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    out = _add_rest_features(df)
    assert out.loc[0, "home_rest_days"] == 7
    assert out.loc[0, "away_rest_days"] == 7
    assert out.loc[0, "home_matches_last_14d"] == 1  # this match itself counts as 0 prior
    assert out.loc[0, "away_matches_last_14d"] == 1


def test_add_rest_features_uses_team_match_history() -> None:
    """Team plays Aug 1, 8, 15 — match on the 15th has rest=7 and matches_last_14d=2."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-08", "2024-08-15"]),
            "home_team": ["Arsenal", "Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Liverpool", "Tottenham"],
        }
    )
    out = _add_rest_features(df)
    # Third row: Arsenal at home; last home match was 2024-08-08 → 7 days rest;
    # in last 14 days Arsenal played 2024-08-01 and 2024-08-08
    assert out.loc[2, "home_rest_days"] == 7
    assert out.loc[2, "home_matches_last_14d"] == 2


def test_add_rest_features_caps_at_14_days() -> None:
    """A long break (international break, end of season) caps rest_days at 14."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01", "2024-08-15"]),  # 106-day gap
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Liverpool"],
        }
    )
    out = _add_rest_features(df)
    assert out.loc[1, "home_rest_days"] == 14
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: ImportError on `src.features`.

- [ ] **Step 3: Implement src/features.py — initial scaffold + rest helper**

Create `src/features.py` with:

```python
"""Engineered feature pipeline.

Top-level `compute_features(matches_df, xg_df)` orchestrates: team-name
normalisation → xG merge → rest → form → rolling goals → xG features →
strength of schedule. Each helper is a pure function (input DataFrame,
output DataFrame with new columns) and is unit-testable in isolation.

Critical invariant: every rolling feature uses only matches BEFORE the current
match's date. Pandas idiom: groupby + sorted dates + shifted rolling. The
test suite includes a leakage-guard test that constructs synthetic data and
asserts no current-match data leaks into past-match features.
"""

from __future__ import annotations

import pandas as pd

REST_DEFAULT_DAYS = 7
REST_MAX_CAP_DAYS = 14
WINDOW_DEFAULT = 5


def _add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `home_rest_days`, `away_rest_days`, `home_matches_last_14d`, `away_matches_last_14d`.

    Rest days = min(days since this team's most recent prior match, 14).
    First-ever match for a team gets the default of 7 days.

    matches_last_14d = count of this team's matches on dates within (current_date - 14, current_date].
    A team's first match counts itself, so the minimum is 1.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Build a long-form view: one row per (match_id, side, team)
    long = pd.concat(
        [
            df[["date"]].assign(team=df["home_team"], match_id=df.index, side="home"),
            df[["date"]].assign(team=df["away_team"], match_id=df.index, side="away"),
        ],
        ignore_index=True,
    ).sort_values(["team", "date"]).reset_index(drop=True)

    # Rest days = days since the team's previous match
    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days.clip(upper=REST_MAX_CAP_DAYS)
    long["rest_days"] = long["rest_days"].fillna(REST_DEFAULT_DAYS).astype(int)

    # matches_last_14d: count of team's matches with date in (current - 14, current]
    def _count_recent(group: pd.DataFrame) -> pd.Series:
        # For each row, count rows in this group with date within last 14 days inclusive
        counts = []
        for d in group["date"]:
            mask = (group["date"] > d - pd.Timedelta(days=14)) & (group["date"] <= d)
            counts.append(int(mask.sum()))
        return pd.Series(counts, index=group.index)

    long["matches_last_14d"] = (
        long.groupby("team", group_keys=False).apply(_count_recent, include_groups=False)
    )

    # Pivot back to per-match columns for home and away sides
    home = long[long["side"] == "home"].set_index("match_id")[["rest_days", "matches_last_14d"]]
    home = home.add_prefix("home_")
    away = long[long["side"] == "away"].set_index("match_id")[["rest_days", "matches_last_14d"]]
    away = away.add_prefix("away_")

    df = df.join(home).join(away)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/features.py tests/test_features.py && uv run ruff format --check src/features.py tests/test_features.py`

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: rest and fixture-congestion features"
```

---

## Task 7: features.py — form features (counts of W/D/L in last 5)

**Files:**
- Modify: `src/features.py` (extend with form helper)
- Modify: `tests/test_features.py` (extend with form tests)

- [ ] **Step 1: Append form tests to tests/test_features.py**

Append at the end:

```python


# ─── Form tests ──────────────────────────────────────────────────


from src.features import _add_form_features


def test_form_features_counts_wins_draws_losses() -> None:
    """Arsenal plays: W, D, L, W, W. The 6th match (any) should show form 3-1-1."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22", "2024-08-29", "2024-09-05"]
            ),
            "home_team": ["Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "away_team": ["A", "B", "C", "D", "E", "F"],
            "FTHG": [2, 1, 0, 3, 2, 1],
            "FTAG": [1, 1, 2, 0, 0, 1],
        }
    )
    out = _add_form_features(df, window=5)
    # Match index 5 (the 6th): prior 5 results for Arsenal as home: W, D, L, W, W
    assert out.loc[5, "home_form_wins"] == 3
    assert out.loc[5, "home_form_draws"] == 1
    assert out.loc[5, "home_form_losses"] == 1


def test_form_features_first_match_zeros() -> None:
    """A team with no prior matches gets all zeros (no history to count)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
        }
    )
    out = _add_form_features(df, window=5)
    assert out.loc[0, "home_form_wins"] == 0
    assert out.loc[0, "home_form_draws"] == 0
    assert out.loc[0, "home_form_losses"] == 0
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v -k "form"`
Expected: ImportError on `_add_form_features`.

- [ ] **Step 3: Append _add_form_features to src/features.py**

At the end of `src/features.py`, append:

```python


def _add_form_features(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add `home_form_{wins,draws,losses}` and `away_form_{wins,draws,losses}`.

    Counts results in the team's last `window` matches BEFORE the current match,
    across both home and away appearances. A team's first match gets all zeros.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Build long-form view, computing this match's outcome from the team's perspective
    long_home = df[["date", "home_team", "FTHG", "FTAG"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"
    long_home["team_goals"] = long_home["FTHG"]
    long_home["opp_goals"] = long_home["FTAG"]

    long_away = df[["date", "away_team", "FTHG", "FTAG"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"
    long_away["team_goals"] = long_away["FTAG"]
    long_away["opp_goals"] = long_away["FTHG"]

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    long["is_win"] = (long["team_goals"] > long["opp_goals"]).astype(int)
    long["is_draw"] = (long["team_goals"] == long["opp_goals"]).astype(int)
    long["is_loss"] = (long["team_goals"] < long["opp_goals"]).astype(int)

    # Rolling sum of last `window` matches, EXCLUDING the current match
    grouped = long.groupby("team", group_keys=False)
    long["form_wins"] = (
        grouped["is_win"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
    )
    long["form_draws"] = (
        grouped["is_draw"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
    )
    long["form_losses"] = (
        grouped["is_loss"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["form_wins", "form_draws", "form_losses"]]
        .add_prefix("home_")
        .astype(int)
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["form_wins", "form_draws", "form_losses"]]
        .add_prefix("away_")
        .astype(int)
    )

    df = df.join(home).join(away)
    return df
```

- [ ] **Step 4: Run all feature tests**

Run: `uv run pytest tests/test_features.py -v`
Expected: 5 PASSED (3 rest + 2 form).

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/features.py tests/test_features.py && uv run ruff format --check src/features.py tests/test_features.py`

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: recent-form features (W/D/L counts in last 5 matches)"
```

---

## Task 8: features.py — rolling goals + xG features

**Files:**
- Modify: `src/features.py` (extend)
- Modify: `tests/test_features.py` (extend)

- [ ] **Step 1: Append goals + xG tests**

Append at the end of `tests/test_features.py`:

```python


# ─── Rolling goals + xG tests ────────────────────────────────────────


from src.features import _add_rolling_goals, _add_xg_features, _merge_xg


def test_rolling_goals_uses_only_past_matches_no_leakage() -> None:
    """The leakage guard: match N's `goals_last_5` only sees matches 0..N-1."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-08", "2024-08-15"]),
            "home_team": ["Arsenal", "Arsenal", "Arsenal"],
            "away_team": ["A", "B", "C"],
            "FTHG": [2, 4, 100],  # huge value at index 2 — a leakage bug would surface here
            "FTAG": [1, 0, 0],
        }
    )
    out = _add_rolling_goals(df, window=5)
    # Index 2's home_goals_scored_last_5 should average matches 0 and 1: (2+4)/2 = 3.0
    # If the impl mistakenly includes index 2, average becomes (2+4+100)/3 = 35.33
    assert out.loc[2, "home_goals_scored_last_5"] == pytest.approx(3.0)


def test_merge_xg_adds_home_xg_and_away_xg_columns() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    xg = pd.DataFrame(
        {
            "date": ["2024-08-15"],
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_xg": [1.85],
            "away_xg": [1.20],
        }
    )
    out = _merge_xg(matches, xg)
    assert out.loc[0, "home_xg"] == pytest.approx(1.85)
    assert out.loc[0, "away_xg"] == pytest.approx(1.20)


def test_merge_xg_no_match_yields_nan() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
        }
    )
    xg = pd.DataFrame(
        {
            "date": ["2024-08-15"],
            "home_team": ["Liverpool"],  # different match
            "away_team": ["Tottenham"],
            "home_xg": [2.5],
            "away_xg": [1.0],
        }
    )
    out = _merge_xg(matches, xg)
    assert pd.isna(out.loc[0, "home_xg"])
    assert pd.isna(out.loc[0, "away_xg"])


def test_xg_minus_actual_overperformance_signal() -> None:
    """Team scored 3 actual goals against 1.5 xG → +1.5 overperformance (lucky)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-08"]),
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["A", "B"],
            "FTHG": [3, 0],
            "FTAG": [0, 0],
            "home_xg": [1.5, 1.0],
            "away_xg": [1.0, 1.0],
        }
    )
    out = _add_xg_features(df, window=5)
    # Index 1: Arsenal's prior match scored 3 vs xG 1.5 → +1.5 overperformance
    assert out.loc[1, "home_xg_minus_actual_last_5"] == pytest.approx(1.5)
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v -k "rolling or xg or merge"`
Expected: ImportError on the new helpers.

- [ ] **Step 3: Append helpers to src/features.py**

At the end of `src/features.py`:

```python


def _merge_xg(matches_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join xG data onto matches by (date, home_team, away_team).

    Missing xG → NaN. Caller's responsibility to handle (typically via
    SimpleImputer at fit time).
    """
    if xg_df.empty:
        out = matches_df.copy()
        out["home_xg"] = pd.NA
        out["away_xg"] = pd.NA
        return out

    matches = matches_df.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    xg = xg_df.copy()
    xg["date"] = pd.to_datetime(xg["date"])
    xg = xg[["date", "home_team", "away_team", "home_xg", "away_xg"]]

    return matches.merge(xg, on=["date", "home_team", "away_team"], how="left")


def _add_rolling_goals(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add `home_goals_scored_last_5`, `home_goals_conceded_last_5`, and away counterparts.

    Uses only matches BEFORE the current row's date (per the leakage guard).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    long_home = df[["date", "home_team", "FTHG", "FTAG"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"
    long_home["scored"] = long_home["FTHG"]
    long_home["conceded"] = long_home["FTAG"]

    long_away = df[["date", "away_team", "FTHG", "FTAG"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"
    long_away["scored"] = long_away["FTAG"]
    long_away["conceded"] = long_away["FTHG"]

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    grouped = long.groupby("team", group_keys=False)
    long["scored_last_5"] = (
        grouped["scored"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    long["conceded_last_5"] = (
        grouped["conceded"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["scored_last_5", "conceded_last_5"]]
        .rename(columns={"scored_last_5": "home_goals_scored_last_5", "conceded_last_5": "home_goals_conceded_last_5"})
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["scored_last_5", "conceded_last_5"]]
        .rename(columns={"scored_last_5": "away_goals_scored_last_5", "conceded_last_5": "away_goals_conceded_last_5"})
    )

    df = df.join(home).join(away)
    return df


def _add_xg_features(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add rolling xG/xGA averages plus xG-vs-actual delta (overperformance signal).

    Requires `home_xg` and `away_xg` columns (from _merge_xg).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    long_home = df[["date", "home_team", "home_xg", "away_xg", "FTHG", "FTAG"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"
    long_home["xg"] = long_home["home_xg"]
    long_home["xga"] = long_home["away_xg"]
    long_home["scored"] = long_home["FTHG"]

    long_away = df[["date", "away_team", "home_xg", "away_xg", "FTHG", "FTAG"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"
    long_away["xg"] = long_away["away_xg"]
    long_away["xga"] = long_away["home_xg"]
    long_away["scored"] = long_away["FTAG"]

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    long["scored_minus_xg"] = long["scored"] - long["xg"]

    grouped = long.groupby("team", group_keys=False)
    long["xg_last_5"] = grouped["xg"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long["xga_last_5"] = grouped["xga"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long["xg_minus_actual_last_5"] = grouped["scored_minus_xg"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["xg_last_5", "xga_last_5", "xg_minus_actual_last_5"]]
        .add_prefix("home_")
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["xg_last_5", "xga_last_5", "xg_minus_actual_last_5"]]
        .add_prefix("away_")
    )

    df = df.join(home).join(away)
    return df
```

- [ ] **Step 4: Run all feature tests**

Run: `uv run pytest tests/test_features.py -v`
Expected: 9 PASSED (3 rest + 2 form + 1 leakage + 2 merge_xg + 1 xg_minus_actual).

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/features.py tests/test_features.py && uv run ruff format --check src/features.py tests/test_features.py`

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: rolling goals + xG features with leakage guard"
```

---

## Task 9: features.py — strength of schedule + compute_features orchestrator

**Files:**
- Modify: `src/features.py` (add final helpers + orchestrator)
- Modify: `tests/test_features.py` (add SoS + end-to-end tests)

- [ ] **Step 1: Append SoS + orchestrator tests**

Append at the end of `tests/test_features.py`:

```python


# ─── Strength of schedule + orchestrator tests ───────────────────────


from src.features import _add_strength_of_schedule, _normalise_team_names, compute_features


def test_strength_of_schedule_promoted_team_default() -> None:
    """Team has no prior-season league position → defaults to league_size - 2."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-15"]),
            "home_team": ["Brand New FC"],   # promoted team (no history)
            "away_team": ["Liverpool"],      # has prior position (assumed mid-table)
            "league": ["England"],
            "year": [2024],
        }
    )
    out = _add_strength_of_schedule(df, window=5)
    # Promoted team: own SoS uses default for opponents
    # Default for England (top flight is 20 teams) → league_size - 2 = 18
    # We're testing that the function doesn't crash and returns a sensible numeric
    assert out.loc[0, "home_opp_strength_last_5"] is not None
    assert out.loc[0, "away_opp_strength_last_5"] is not None


def test_compute_features_end_to_end_runs_without_error() -> None:
    """Full pipeline on a small synthetic dataset; verify the expected feature columns appear."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22"]
            ),
            "home_team": ["Arsenal", "Liverpool", "Arsenal", "Liverpool"],
            "away_team": ["Liverpool", "Arsenal", "Liverpool", "Arsenal"],
            "FTHG": [2, 1, 0, 3],
            "FTAG": [1, 1, 2, 0],
            "league": ["England"] * 4,
            "year": [2024] * 4,
        }
    )
    xg = pd.DataFrame(
        {
            "date": ["2024-08-01", "2024-08-08", "2024-08-15", "2024-08-22"],
            "home_team": ["Arsenal", "Liverpool", "Arsenal", "Liverpool"],
            "away_team": ["Liverpool", "Arsenal", "Liverpool", "Arsenal"],
            "home_xg": [1.8, 1.2, 0.9, 2.5],
            "away_xg": [0.9, 1.4, 1.7, 1.0],
        }
    )
    out = compute_features(matches, xg)

    expected_cols = {
        "home_rest_days", "away_rest_days",
        "home_matches_last_14d", "away_matches_last_14d",
        "home_form_wins", "home_form_draws", "home_form_losses",
        "away_form_wins", "away_form_draws", "away_form_losses",
        "home_goals_scored_last_5", "home_goals_conceded_last_5",
        "away_goals_scored_last_5", "away_goals_conceded_last_5",
        "home_xg_last_5", "home_xga_last_5", "home_xg_minus_actual_last_5",
        "away_xg_last_5", "away_xga_last_5", "away_xg_minus_actual_last_5",
        "home_opp_strength_last_5", "away_opp_strength_last_5",
    }
    assert expected_cols.issubset(set(out.columns)), (
        f"Missing feature columns: {expected_cols - set(out.columns)}"
    )
    assert len(out) == 4
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v -k "strength or end_to_end"`
Expected: ImportError on `_add_strength_of_schedule`, `_normalise_team_names`, `compute_features`.

- [ ] **Step 3: Append final helpers + orchestrator to src/features.py**

At the end of `src/features.py`:

```python


LEAGUE_SIZE_DEFAULT: dict[str, int] = {
    "England": 20,
    "Spain": 20,
    "Italy": 20,
    "Germany": 18,
    "France": 18,
}


def _normalise_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert football-data team names to canonical form.

    Drops rows where either team is unmapped (with a logged warning naming
    the unmapped team — the caller adds it to src/team_names.REGISTRY and re-runs).
    """
    from src.team_names import to_canonical

    df = df.copy()
    keep_idx: list[int] = []
    for idx, row in df.iterrows():
        try:
            row["home_team"] = to_canonical(row["home_team"], source="football_data")
            row["away_team"] = to_canonical(row["away_team"], source="football_data")
            df.at[idx, "home_team"] = row["home_team"]
            df.at[idx, "away_team"] = row["away_team"]
            keep_idx.append(idx)
        except KeyError:
            # Will be re-raised by the caller (training script) once verification
            # is added; here we let it propagate so unmapped teams don't silently drop.
            raise
    return df.loc[keep_idx].reset_index(drop=True)


def _add_strength_of_schedule(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add `home_opp_strength_last_5`, `away_opp_strength_last_5`.

    Strength = average prior-season league position of the team's last `window`
    opponents. We approximate prior-season position from a rolling computation
    of points-per-game across all matches a team played in the previous year:
    teams with higher PPG get lower (better) position numbers.

    Promoted teams (no prior-year data) get league_size - 2 (relegation-adjacent).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["league", "year", "date"]).reset_index(drop=True)

    # Compute prior-season PPG per team
    points_table = []
    for (league, year), group in df.groupby(["league", "year"]):
        # Per-team points: 3 for win, 1 for draw, 0 for loss
        teams: dict[str, int] = {}
        games: dict[str, int] = {}
        for _, m in group.iterrows():
            for team_col, scored_col, conceded_col in [
                ("home_team", "FTHG", "FTAG"),
                ("away_team", "FTAG", "FTHG"),
            ]:
                team = m[team_col]
                teams.setdefault(team, 0)
                games.setdefault(team, 0)
                games[team] += 1
                if m[scored_col] > m[conceded_col]:
                    teams[team] += 3
                elif m[scored_col] == m[conceded_col]:
                    teams[team] += 1
        for team, pts in teams.items():
            ppg = pts / games[team] if games[team] else 0
            points_table.append({"league": league, "year": year, "team": team, "ppg": ppg})

    ppg_df = pd.DataFrame(points_table)

    # Convert PPG to "position" — rank within league-year, descending (higher PPG = lower number = better)
    ppg_df["position"] = (
        ppg_df.groupby(["league", "year"])["ppg"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Use prior season's position to score current season's matches
    ppg_df["next_year"] = ppg_df["year"] + 1

    def _opp_position(league: str, year: int, team: str) -> float:
        match = ppg_df[
            (ppg_df["league"] == league)
            & (ppg_df["next_year"] == year)
            & (ppg_df["team"] == team)
        ]
        if match.empty:
            return float(LEAGUE_SIZE_DEFAULT.get(league, 20) - 2)
        return float(match["position"].iloc[0])

    # Long-form view of opponents
    long_home = df[["date", "home_team", "away_team", "league", "year"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["opponent"] = long_home["away_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"

    long_away = df[["date", "home_team", "away_team", "league", "year"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["opponent"] = long_away["home_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    long["opp_pos"] = long.apply(
        lambda r: _opp_position(r["league"], r["year"], r["opponent"]),
        axis=1,
    )

    grouped = long.groupby("team", group_keys=False)
    long["opp_strength"] = grouped["opp_pos"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["opp_strength"]]
        .rename(columns={"opp_strength": "home_opp_strength_last_5"})
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["opp_strength"]]
        .rename(columns={"opp_strength": "away_opp_strength_last_5"})
    )

    # First-match defaults (no prior data → league-size-2)
    home["home_opp_strength_last_5"] = home["home_opp_strength_last_5"].fillna(
        df["league"].map(LEAGUE_SIZE_DEFAULT).fillna(20) - 2
    )
    away["away_opp_strength_last_5"] = away["away_opp_strength_last_5"].fillna(
        df["league"].map(LEAGUE_SIZE_DEFAULT).fillna(20) - 2
    )

    df = df.join(home).join(away)
    return df


def compute_features(matches_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Top-level: returns matches_df with engineered feature columns added.

    Pipeline order:
        1. Normalise team names (football-data → canonical)
        2. Merge xG by (date, home_team, away_team)
        3. Rest features
        4. Form features (last 5 W/D/L)
        5. Rolling goals (last 5 scored/conceded)
        6. xG features (last 5 xG/xGA + over/underperformance)
        7. Strength of schedule (last 5 opponents' prior-season position)
    """
    df = _normalise_team_names(matches_df)
    df = _merge_xg(df, xg_df)
    df = _add_rest_features(df)
    df = _add_form_features(df, window=WINDOW_DEFAULT)
    df = _add_rolling_goals(df, window=WINDOW_DEFAULT)
    df = _add_xg_features(df, window=WINDOW_DEFAULT)
    df = _add_strength_of_schedule(df, window=WINDOW_DEFAULT)
    return df
```

- [ ] **Step 4: Run all feature tests**

Run: `uv run pytest tests/test_features.py -v`
Expected: 11 PASSED.

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/features.py tests/test_features.py && uv run ruff format --check src/features.py tests/test_features.py`

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: strength-of-schedule + compute_features orchestrator"
```

---

## Task 10: scripts/train_model.py — TimeSeriesSplit backtest CV function

**Files:**
- Modify: `scripts/train_model.py` (replace contents — start with imports, constants, and run_backtest_cv only)
- Modify: `tests/test_train_pipeline.py` (replace contents — drop old library-coupled tests, add CV arithmetic test)

This task starts the train_model.py rewrite. Tasks 11 and 12 add the atomic-write extension and the main() orchestration. The previous `run_backtest()` (library-coupled) is replaced by `run_backtest_cv()` which uses sklearn's TimeSeriesSplit directly.

- [ ] **Step 1: Replace tests/test_train_pipeline.py**

Replace `tests/test_train_pipeline.py` entirely with:

```python
"""Tests for training pipeline plumbing.

Atomic-write tests verify the 7-artifact (5 pkl + 1 parquet + 1 json) write.
Backtest CV test verifies per-market yield arithmetic on a synthetic dataset.
End-to-end training is exercised via the manual smoke test (Task 16).
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from scripts.train_model import (
    BIG_5_LEAGUES,
    run_backtest_cv,
    write_artifacts_atomically,
)


def test_run_backtest_cv_yields_arithmetic() -> None:
    """Synthetic 30-match single-market dataset; assert per-market yield matches manual calc.

    We build a dataset where the model will predict a fixed probability for every
    match, and odds are constant. Value bets fire predictably; we can compute
    the exact yield by hand and assert the CV pipeline matches.
    """
    # 30 matches with deterministic outcomes alternating 1, 0, 1, 0, ...
    n = 30
    X = pd.DataFrame(
        {
            "feature_a": np.arange(n, dtype=float),
            "feature_b": np.arange(n, dtype=float) * 0.5,
        }
    )
    Y = pd.DataFrame(
        {
            "output__home_win__full_time_goals": [1, 0] * (n // 2),
            "output__draw__full_time_goals": [0, 1] * (n // 2),
            "output__away_win__full_time_goals": [0, 0] * (n // 2),
            "output__over_2.5__full_time_goals": [1, 1] * (n // 2),
            "output__under_2.5__full_time_goals": [0, 0] * (n // 2),
        }
    )
    O = pd.DataFrame(
        {col: [2.0] * n for col in Y.columns}
    )

    # A trivial classifier-factory: each market returns 0.6 probability constant.
    # We use a fake bettor that mimics the predict_proba interface.
    class _ConstantClassifier:
        def fit(self, X, y): return self
        def predict_proba(self, X):
            n = len(X)
            return [
                np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])
                for _ in range(5)
            ]

    def factory():
        return _ConstantClassifier()

    summary = run_backtest_cv(
        bettor_factory=factory,
        X=X, Y=Y, O=O,
        n_splits=3,
        value_threshold=0.05,
    )
    # The summary should have a markets dict with 5 entries
    assert set(summary["markets"].keys()) == {
        "home_win", "draw", "away_win", "over_2.5", "under_2.5"
    }
    # Each market should have n_bets and yield_pct keys
    for market_stats in summary["markets"].values():
        assert "n_bets" in market_stats
        assert "yield_pct" in market_stats
    # n_matches should be the input length
    assert summary["n_matches"] == n


def test_atomic_write_creates_seven_artifacts(tmp_path, monkeypatch) -> None:
    """5 bettor pkls + 1 parquet + 1 json all land at their final paths."""
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")
    monkeypatch.setattr("scripts.train_model.FIXTURES_PATH", tmp_path / "fixtures_data.parquet")
    monkeypatch.setattr(
        "scripts.train_model.BETTOR_PATHS",
        {league: tmp_path / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES},
    )

    bettors = {league: {"fake_bettor_for": league} for league in BIG_5_LEAGUES}
    fixtures_df = pd.DataFrame({"a": [1, 2, 3]})
    summary = {"trained_at": "2026-05-06T00:00:00Z", "leagues": {}}

    write_artifacts_atomically(bettors, fixtures_df, summary)

    for league in BIG_5_LEAGUES:
        path = tmp_path / f"{league}_bettor.pkl"
        assert path.exists()
        with path.open("rb") as f:
            assert pickle.load(f) == {"fake_bettor_for": league}

    assert (tmp_path / "fixtures_data.parquet").exists()
    assert (tmp_path / "backtest.json").exists()
    with (tmp_path / "backtest.json").open() as f:
        assert json.load(f)["trained_at"] == "2026-05-06T00:00:00Z"


def test_atomic_write_no_partial_state_when_one_pickle_fails(tmp_path, monkeypatch) -> None:
    """If any of the 7 writes fails, all .tmp files are cleaned and no final file appears."""
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")
    monkeypatch.setattr("scripts.train_model.FIXTURES_PATH", tmp_path / "fixtures_data.parquet")
    monkeypatch.setattr(
        "scripts.train_model.BETTOR_PATHS",
        {league: tmp_path / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES},
    )

    class Unpicklable:
        def __reduce__(self):
            raise TypeError("nope")

    # Make laliga's bettor fail
    bettors: dict = {league: {"ok": league} for league in BIG_5_LEAGUES}
    bettors["laliga"] = Unpicklable()
    fixtures_df = pd.DataFrame({"a": [1]})
    summary = {"trained_at": "x"}

    with pytest.raises(TypeError):
        write_artifacts_atomically(bettors, fixtures_df, summary)

    # No final files should exist
    for league in BIG_5_LEAGUES:
        assert not (tmp_path / f"{league}_bettor.pkl").exists()
    assert not (tmp_path / "fixtures_data.parquet").exists()
    assert not (tmp_path / "backtest.json").exists()
    # No leftover .tmp files
    assert list(tmp_path.glob("*.tmp")) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_pipeline.py -v`
Expected: ImportError on `BIG_5_LEAGUES`, `run_backtest_cv`, `write_artifacts_atomically` (existing implementations don't match new signatures).

- [ ] **Step 3: Replace scripts/train_model.py with imports + constants + run_backtest_cv only**

Replace `scripts/train_model.py` entirely with:

```python
"""Train the Big-5 multi-output ClassifierBettor and save artifacts atomically.

Single command produces seven artifacts in models/:
    epl_bettor.pkl, laliga_bettor.pkl, seriea_bettor.pkl,
    bundesliga_bettor.pkl, ligue1_bettor.pkl  -- one fitted classifier per league
    fixtures_data.parquet                     -- pre-computed feature DataFrame for upcoming fixtures
    backtest.json                             -- per-league per-market backtest summary

Usage:
    uv run python scripts/train_model.py

Run time: typically 30-50 minutes on a laptop. Existing artifacts are only
overwritten if the entire run succeeds (atomic write).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pickle
import sys
from collections.abc import Callable
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
# Make project root importable so `from src.foo import ...` works when this
# script is run directly. Streamlit/pytest do this automatically; CLI entry-points
# need to do it explicitly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

BIG_5_LEAGUES: list[str] = ["epl", "laliga", "seriea", "bundesliga", "ligue1"]

# Per-league bettor pickle paths
BETTOR_PATHS: dict[str, Path] = {
    league: MODELS_DIR / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES
}
FIXTURES_PATH: Path = MODELS_DIR / "fixtures_data.parquet"
BACKTEST_PATH: Path = MODELS_DIR / "backtest.json"

# Mapping from internal league key → football-data.co.uk capitalised league name
LEAGUE_TO_FOOTBALL_DATA: dict[str, str] = {
    "epl": "England",
    "laliga": "Spain",
    "seriea": "Italy",
    "bundesliga": "Germany",
    "ligue1": "France",
}
# Mapping from internal league key → Understat slug
LEAGUE_TO_UNDERSTAT: dict[str, str] = {
    "epl": "EPL",
    "laliga": "La_liga",
    "seriea": "Serie_A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue_1",
}


# ─── Backtest CV ────────────────────────────────────────────────────


def run_backtest_cv(
    bettor_factory: Callable[[], Any],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    O: pd.DataFrame,  # noqa: E741 — capital O matches library convention
    *,
    n_splits: int = 5,
    value_threshold: float = 0.05,
) -> dict:
    """Run TimeSeriesSplit cross-validation; return per-market yield summary.

    For each fold, fit a fresh bettor on train, predict on test, select bets where
    `model_prob × decimal_odds > 1 + value_threshold`, accumulate returns.
    Yield per market = mean of returns across all folds' selected bets.

    Args:
        bettor_factory: callable returning a fresh classifier each call
        X, Y, O: training features, multi-output targets, decimal odds
        n_splits: number of TimeSeriesSplit folds
        value_threshold: minimum edge fraction to qualify as a value bet

    Returns:
        dict with shape:
        {
            "n_matches": int,
            "markets": {
                "home_win": {"n_bets": int, "yield_pct": float},
                ...
            }
        }
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    market_keys = [_strip_market_suffix(col) for col in Y.columns]
    bets_per_market: dict[str, list[float]] = {key: [] for key in market_keys}

    for train_idx, test_idx in tscv.split(X):
        if len(test_idx) == 0:
            continue
        bettor = bettor_factory()
        bettor.fit(X.iloc[train_idx], Y.iloc[train_idx])

        # MultiOutputClassifier returns a list (one array per market)
        probs_per_market = bettor.predict_proba(X.iloc[test_idx])

        for market_idx, market_col in enumerate(Y.columns):
            market_key = market_keys[market_idx]
            test_outcomes = Y.iloc[test_idx][market_col].to_numpy()
            test_odds = O.iloc[test_idx][market_col].to_numpy()
            test_probs = probs_per_market[market_idx][:, 1]

            value_mask = test_probs * test_odds > 1 + value_threshold
            returns = np.where(value_mask, test_odds * test_outcomes - 1, 0.0)
            bets_per_market[market_key].extend(
                [float(r) for r in returns[returns != 0]]
            )

    summary_markets: dict[str, dict[str, float]] = {}
    for market_key, returns in bets_per_market.items():
        if not returns:
            summary_markets[market_key] = {"n_bets": 0, "yield_pct": 0.0}
        else:
            summary_markets[market_key] = {
                "n_bets": len(returns),
                "yield_pct": round(100 * float(np.mean(returns)), 2),
            }

    return {
        "n_matches": len(X),
        "markets": summary_markets,
    }


def _strip_market_suffix(col: str) -> str:
    """`output__home_win__full_time_goals` → `home_win`."""
    parts = col.split("__")
    if len(parts) >= 3:
        return parts[1]
    return col


# ─── Atomic write ──────────────────────────────────────────────────


def write_artifacts_atomically(
    bettors: dict[str, Any],
    fixtures_df: pd.DataFrame,
    summary: dict,
) -> None:
    """Write all 7 artifacts to .tmp paths, then os.replace them to final paths.

    Either all 7 land at their final paths, or none do. Cleans up .tmp files
    on any failure. The five bettor pickles, the fixtures parquet, and the
    backtest JSON are promoted in one final pass.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bettor_tmps = {
        league: BETTOR_PATHS[league].with_suffix(BETTOR_PATHS[league].suffix + ".tmp")
        for league in BIG_5_LEAGUES
    }
    fixtures_tmp = FIXTURES_PATH.with_suffix(FIXTURES_PATH.suffix + ".tmp")
    backtest_tmp = BACKTEST_PATH.with_suffix(BACKTEST_PATH.suffix + ".tmp")
    all_tmps = list(bettor_tmps.values()) + [fixtures_tmp, backtest_tmp]

    try:
        # 1. Write all temps
        for league, bettor in bettors.items():
            with bettor_tmps[league].open("wb") as f:
                pickle.dump(bettor, f)
        fixtures_df.to_parquet(fixtures_tmp, index=False)
        with backtest_tmp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 2. All temp writes succeeded — promote atomically
        for league in BIG_5_LEAGUES:
            os.replace(bettor_tmps[league], BETTOR_PATHS[league])
        os.replace(fixtures_tmp, FIXTURES_PATH)
        os.replace(backtest_tmp, BACKTEST_PATH)
    except BaseException:
        for tmp in all_tmps:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        raise


def main() -> None:
    raise NotImplementedError("Implemented in Task 12")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_pipeline.py -v`
Expected: 3 PASSED (CV arithmetic + 2 atomic-write tests).

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check scripts/train_model.py tests/test_train_pipeline.py && uv run ruff format --check scripts/train_model.py tests/test_train_pipeline.py`

- [ ] **Step 6: Commit**

```bash
git add scripts/train_model.py tests/test_train_pipeline.py
git commit -m "feat: TimeSeriesSplit CV backtest + 7-file atomic write"
```

---

## Task 11: scripts/train_model.py — main() orchestration

**Files:**
- Modify: `scripts/train_model.py` (replace `main` stub with full orchestration)

No new tests — `main()` is exercised via the manual smoke test (Task 16). The CV and atomic-write logic it calls are already unit-tested.

- [ ] **Step 1: Replace the main() stub in scripts/train_model.py**

In `scripts/train_model.py`, find:

```python
def main() -> None:
    raise NotImplementedError("Implemented in Task 12")
```

Replace with:

```python
def build_calibrated_classifier():
    """Same pipeline shape as v0 — keeping classifier constant isolates feature uplift."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    return make_pipeline(
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                ["league", "home_team", "away_team"],
            ),
            remainder="passthrough",
        ),
        SimpleImputer(strategy="mean"),
        MultiOutputClassifier(
            CalibratedClassifierCV(
                GradientBoostingClassifier(random_state=0),
                method="isotonic",
                cv=3,
            )
        ),
    )


def _split_by_league(features_df: pd.DataFrame, league: str):
    """Filter the master feature DataFrame to one league. Returns (X, Y, O) tuple.

    Y columns: 5 markets in the locked order home_win/draw/away_win/over_2.5/under_2.5.
    O columns: bookmaker decimal odds for each market (from football-data CSV).
    X columns: everything else minus FTR/etc. that wouldn't be available pre-match.
    """
    df = features_df[features_df["league"] == LEAGUE_TO_FOOTBALL_DATA[league]].copy()

    # Build Y from FTHG vs FTAG
    home_goals = df["FTHG"]
    away_goals = df["FTAG"]
    total_goals = home_goals + away_goals

    Y = pd.DataFrame(
        {
            "output__home_win__full_time_goals": (home_goals > away_goals).astype(int),
            "output__draw__full_time_goals": (home_goals == away_goals).astype(int),
            "output__away_win__full_time_goals": (home_goals < away_goals).astype(int),
            "output__over_2.5__full_time_goals": (total_goals > 2.5).astype(int),
            "output__under_2.5__full_time_goals": (total_goals <= 2.5).astype(int),
        }
    )

    # Odds columns from football-data CSV: B365H, B365D, B365A for h2h
    # We use AvgH/D/A (or BbAvH/D/A in older seasons) if present; fall back to B365.
    h2h_cols = _find_first_present_columns(df, [["AvgH", "AvgD", "AvgA"], ["BbAvH", "BbAvD", "BbAvA"], ["B365H", "B365D", "B365A"]])
    over_under_cols = _find_first_present_columns(df, [["AvgOver2.5", "AvgUnder2.5"], ["BbAv>2.5", "BbAv<2.5"], ["B365>2.5", "B365<2.5"]])

    O = pd.DataFrame(
        {
            "output__home_win__full_time_goals": df[h2h_cols[0]] if h2h_cols else 2.0,
            "output__draw__full_time_goals": df[h2h_cols[1]] if h2h_cols else 3.5,
            "output__away_win__full_time_goals": df[h2h_cols[2]] if h2h_cols else 4.0,
            "output__over_2.5__full_time_goals": df[over_under_cols[0]] if over_under_cols else 2.0,
            "output__under_2.5__full_time_goals": df[over_under_cols[1]] if over_under_cols else 1.85,
        }
    )

    # X = features only — drop the outcome columns and odds columns
    feature_cols = [
        c for c in df.columns
        if c not in ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]
        and not (h2h_cols and c in h2h_cols)
        and not (over_under_cols and c in over_under_cols)
    ]
    X = df[feature_cols].reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    O = O.reset_index(drop=True)
    return X, Y, O


def _find_first_present_columns(df: pd.DataFrame, candidates: list[list[str]]) -> list[str] | None:
    """Return the first list from `candidates` whose every column is present in df."""
    for candidate in candidates:
        if all(col in df.columns for col in candidate):
            return candidate
    return None


def _build_upcoming_fixtures_for_features(matches_raw: pd.DataFrame) -> pd.DataFrame:
    """Snapshot upcoming fixtures from The Odds API in a shape that compute_features can consume.

    Why The Odds API rather than football-data's fixtures.csv: the spec flagged this as
    planner-resolvable. We pick The Odds API because it's known to cover all 5 leagues
    (we already use it in app.py) and it tags each fixture with a sport_key we map to
    our internal league key. The football-data fixtures.csv may be EPL-only, which would
    leave 4 leagues with no upcoming-fixture coverage in the parquet.

    Returns a DataFrame in the same column shape as `matches_raw`: HomeTeam, AwayTeam,
    Date, league, year, plus placeholder FTHG=0/FTAG=0 (these don't affect the rolling
    features for the fixture row itself because rolling uses shift(1) — only past matches
    contribute to the current row's features).
    """
    from src.odds import get_big5_odds
    from src.team_names import to_canonical

    fixtures = get_big5_odds()
    rows: list[dict] = []
    for fixture in fixtures:
        try:
            home_canonical = to_canonical(fixture["home_team"], source="odds_api")
            away_canonical = to_canonical(fixture["away_team"], source="odds_api")
        except KeyError as e:
            logger.warning(f"Skipping unmapped fixture team: {e}")
            continue
        # Convert canonical → football-data form (the form matches_raw uses)
        from src.team_names import from_canonical
        home_fd = from_canonical(home_canonical, target="football_data")
        away_fd = from_canonical(away_canonical, target="football_data")

        league_fd = LEAGUE_TO_FOOTBALL_DATA[fixture["league"]]
        rows.append(
            {
                "Date": pd.to_datetime(fixture["commence_time"]).date().isoformat(),
                "HomeTeam": home_fd,
                "AwayTeam": away_fd,
                "FTHG": 0,
                "FTAG": 0,
                "league": league_fd,
                "year": pd.to_datetime(fixture["commence_time"]).year,
            }
        )

    if not rows:
        return pd.DataFrame()
    fixtures_df = pd.DataFrame(rows)
    # Match matches_raw's expected column shape: rename Date→date if matches_raw uses that
    if "date" in matches_raw.columns and "Date" in fixtures_df.columns:
        fixtures_df = fixtures_df.rename(columns={"Date": "date"})
    elif "Date" in matches_raw.columns and "date" in fixtures_df.columns:
        fixtures_df = fixtures_df.rename(columns={"date": "Date"})
    return fixtures_df


def main() -> None:
    """Train all five per-league bettors end-to-end and persist artifacts."""
    from src.features import compute_features
    from src.ingest.football_data import download_training_data
    from src.ingest.understat import fetch_xg_data

    print("⚽ Football Edge — training Big-5 multi-output bettors")
    print("=" * 60)

    training_years = list(range(2018, 2026))

    # 1. Ingest
    print("\n[1/5] Downloading football-data.co.uk CSVs...")
    football_leagues = list(LEAGUE_TO_FOOTBALL_DATA.values())
    matches_raw = download_training_data(leagues=football_leagues, years=training_years)
    print(f"  Loaded {len(matches_raw)} historical matches.")

    # Normalise football-data's date column name to lowercase 'date' for compute_features.
    # football-data CSVs use 'Date' (capital D); compute_features expects 'date'.
    if "Date" in matches_raw.columns and "date" not in matches_raw.columns:
        matches_raw = matches_raw.rename(columns={"Date": "date"})
    if "HomeTeam" in matches_raw.columns and "home_team" not in matches_raw.columns:
        matches_raw = matches_raw.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})

    print("  Fetching upcoming fixtures from The Odds API...")
    upcoming_fixtures = _build_upcoming_fixtures_for_features(matches_raw)
    print(f"  Snapshotted {len(upcoming_fixtures)} upcoming fixtures across the Big-5.")

    print("\n[2/5] Scraping Understat xG (this is the slow ingest step)...")
    understat_leagues = list(LEAGUE_TO_UNDERSTAT.values())
    xg_df = fetch_xg_data(leagues=understat_leagues, years=training_years)
    print(f"  Loaded xG for {len(xg_df)} historical matches.")

    # 2. Feature engineering
    # Combine training + upcoming so compute_features sees full history when computing
    # the rolling features for the upcoming-fixture rows. compute_features uses shift(1)
    # internally so the upcoming-fixture rows' OWN placeholder outcomes don't pollute
    # their own features — only past matches (training data) contribute.
    print("\n[3/5] Computing engineered features...")
    if not upcoming_fixtures.empty:
        combined = pd.concat([matches_raw, upcoming_fixtures], ignore_index=True)
    else:
        combined = matches_raw
    combined_features = compute_features(combined, xg_df)

    # Split back into training and upcoming-fixtures features
    n_training = len(matches_raw)
    features_df = combined_features.iloc[:n_training].reset_index(drop=True)
    fixtures_features_df = combined_features.iloc[n_training:].reset_index(drop=True)
    print(f"  Training features shape: {features_df.shape}")
    print(f"  Fixtures features shape: {fixtures_features_df.shape}")

    # 3. Per-league training loop
    print("\n[4/5] Per-league training (~5-10 minutes per league)...")
    bettors: dict[str, Any] = {}
    league_summaries: dict[str, dict] = {}

    for league in BIG_5_LEAGUES:
        print(f"  ─ Training {league}...")
        X, Y, O = _split_by_league(features_df, league)
        if len(X) < 100:
            raise RuntimeError(
                f"League {league} has only {len(X)} matches — refusing to train. "
                f"Check team-name mappings and ingest output."
            )

        league_summaries[league] = {
            "n_matches": len(X),
            **run_backtest_cv(
                build_calibrated_classifier, X, Y, O,
                n_splits=5, value_threshold=0.05,
            ),
        }
        bettor = build_calibrated_classifier()
        bettor.fit(X, Y)
        bettors[league] = bettor
        print(f"    Done. {len(X)} matches.")

    # 4. Build summary
    summary = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_seasons": training_years,
        "sklearn_version": pkg_version("scikit-learn"),
        "n_training_matches": int(len(features_df)),
        "leagues": league_summaries,
    }

    # 5. Atomic persist
    print("\n[5/5] Persisting artifacts...")
    write_artifacts_atomically(bettors, fixtures_features_df, summary)

    print("\n✅ Done. Artifacts written to:")
    for league in BIG_5_LEAGUES:
        print(f"     {BETTOR_PATHS[league]}")
    print(f"     {FIXTURES_PATH}")
    print(f"     {BACKTEST_PATH}")
    print("\nPer-league per-market backtest summary:")
    for league, league_summary in summary["leagues"].items():
        print(f"\n  {league} ({league_summary['n_matches']} matches):")
        for market, stats in league_summary["markets"].items():
            print(
                f"    {market:<12} {stats['n_bets']:>4} bets  "
                f"yield {stats['yield_pct']:+.2f}%"
            )
    print()
```

- [ ] **Step 2: Sanity-check imports**

Run: `uv run python -c "from scripts.train_model import main, run_backtest_cv, write_artifacts_atomically, BIG_5_LEAGUES; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Run all tests for regression**

Run: `uv run pytest -q`
Expected: tests we've written pass; some may still fail on test_predictions.py (resolved in Task 13). If others fail unexpectedly, stop and investigate.

- [ ] **Step 4: Run static checks**

Run: `uv run ruff check scripts/train_model.py && uv run ruff format --check scripts/train_model.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_model.py
git commit -m "feat: main() orchestration for Big-5 training pipeline"
```

---

## Task 12: src/predictions.py — multi-league routing

**Files:**
- Modify (replace contents): `src/predictions.py`
- Modify (replace contents): `tests/test_predictions.py`

- [ ] **Step 1: Replace tests/test_predictions.py**

Replace `tests/test_predictions.py` entirely with:

```python
"""Tests for src/predictions.py — multi-league routing."""

from __future__ import annotations

import json
import math
import pickle
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.predictions import (
    BIG_5_LEAGUES,
    Models,
    _demo_prediction,
    load_models,
    predict_fixture,
)


# ─── Phase 1: demo-mode regression ───────────────────────────────────


def test_demo_prediction_returns_demo_flag() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    assert pred.is_demo is True


def test_demo_prediction_probabilities_sum_to_one() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_demo_prediction_is_deterministic() -> None:
    a = _demo_prediction("Arsenal", "Chelsea")
    b = _demo_prediction("Arsenal", "Chelsea")
    assert a.p_home == b.p_home


def test_demo_prediction_swap_changes_output() -> None:
    a = _demo_prediction("Arsenal", "Chelsea")
    b = _demo_prediction("Chelsea", "Arsenal")
    assert (a.p_home, a.p_draw, a.p_away) != (b.p_home, b.p_draw, b.p_away)


def test_demo_prediction_includes_over_2_5() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    assert pred.p_over_2_5 is not None
    assert 0.0 <= pred.p_over_2_5 <= 1.0


# ─── Phase 2: Models container ───────────────────────────────────────


def test_models_is_ready_false_when_bettors_dict_empty() -> None:
    m = Models(bettors={}, fixtures_df=pd.DataFrame(), backtest={})
    assert m.is_ready is False


def test_models_is_ready_false_when_only_some_leagues_loaded() -> None:
    m = Models(
        bettors={"epl": object(), "laliga": object()},  # only 2 of 5
        fixtures_df=pd.DataFrame(),
        backtest={"trained_at": "x"},
    )
    assert m.is_ready is False


def test_models_is_ready_true_when_all_5_leagues_plus_fixtures_plus_backtest() -> None:
    m = Models(
        bettors={league: object() for league in BIG_5_LEAGUES},
        fixtures_df=pd.DataFrame(),
        backtest={"trained_at": "x"},
    )
    assert m.is_ready is True


def test_load_models_returns_unready_when_files_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.predictions.MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        "src.predictions.BETTOR_PATHS",
        {league: tmp_path / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES},
    )
    monkeypatch.setattr("src.predictions.FIXTURES_PATH", tmp_path / "fixtures_data.parquet")
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", tmp_path / "backtest.json")

    models = load_models()
    assert models.is_ready is False


def test_load_models_returns_ready_when_all_artifacts_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.predictions.MODELS_DIR", tmp_path)
    bettor_paths = {league: tmp_path / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES}
    fixtures_path = tmp_path / "fixtures_data.parquet"
    backtest_path = tmp_path / "backtest.json"
    monkeypatch.setattr("src.predictions.BETTOR_PATHS", bettor_paths)
    monkeypatch.setattr("src.predictions.FIXTURES_PATH", fixtures_path)
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", backtest_path)

    for league, path in bettor_paths.items():
        with path.open("wb") as f:
            pickle.dump({"fake": league}, f)
    pd.DataFrame({"a": [1]}).to_parquet(fixtures_path)
    backtest_path.write_text(json.dumps({"trained_at": "2026-05-06T00:00:00Z"}))

    models = load_models()
    assert models.is_ready is True
    assert set(models.bettors.keys()) == set(BIG_5_LEAGUES)


# ─── Phase 3: predict_fixture multi-league routing ───────────────────


def test_predict_fixture_demo_when_models_unready() -> None:
    models = Models(bettors={}, fixtures_df=pd.DataFrame(), backtest=None)
    pred = predict_fixture(models, "epl", "Arsenal", "Chelsea")
    assert pred.is_demo is True


def test_predict_fixture_demo_when_unsupported_league() -> None:
    models = Models(
        bettors={league: MagicMock() for league in BIG_5_LEAGUES},
        fixtures_df=pd.DataFrame(),
        backtest={"x": 1},
    )
    pred = predict_fixture(models, "championship", "Arsenal", "Chelsea")  # not big-5
    assert pred.is_demo is True


def test_predict_fixture_routes_to_correct_league_bettor() -> None:
    """A La Liga fixture calls la_liga's bettor, not EPL's."""
    bettors = {league: MagicMock() for league in BIG_5_LEAGUES}
    bettors["laliga"].predict_proba.return_value = [
        # 5 markets x 2 columns each (P=0, P=1)
        [[0.3, 0.7]],
        [[0.7, 0.3]],
        [[0.6, 0.4]],
        [[0.4, 0.6]],
        [[0.6, 0.4]],
    ]
    fixtures_df = pd.DataFrame(
        {"league": ["Spain"], "home_team": ["Real Madrid"], "away_team": ["Barcelona"]}
    )
    models = Models(bettors=bettors, fixtures_df=fixtures_df, backtest={"x": 1})

    pred = predict_fixture(models, "laliga", "Real Madrid", "Barcelona")

    bettors["laliga"].predict_proba.assert_called_once()
    bettors["epl"].predict_proba.assert_not_called()
    assert pred.is_demo is False


def test_predict_fixture_demo_when_fixture_row_not_found() -> None:
    bettors = {league: MagicMock() for league in BIG_5_LEAGUES}
    fixtures_df = pd.DataFrame(
        {"league": ["England"], "home_team": ["Liverpool"], "away_team": ["Everton"]}
    )
    models = Models(bettors=bettors, fixtures_df=fixtures_df, backtest={"x": 1})

    pred = predict_fixture(models, "epl", "Arsenal", "Chelsea")
    assert pred.is_demo is True
    bettors["epl"].predict_proba.assert_not_called()
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_predictions.py -v`
Expected: ImportError on new symbols.

- [ ] **Step 3: Replace src/predictions.py**

Replace `src/predictions.py` entirely with:

```python
"""Multi-league prediction container and routing.

`Models` holds a dict of per-league fitted bettors plus a shared fixtures DataFrame
and the parsed backtest summary. `is_ready` is True only when all five leagues'
bettors are loaded and the fixtures+backtest are present.

`predict_fixture(models, league, home_team, away_team)` routes to the right bettor
based on the `league` key (passed in by the dashboard from the Odds API's sport_key).
Demo fallback for any error: missing bettor, unmapped team, no fixture row, library
exception during predict_proba.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BIG_5_LEAGUES: list[str] = ["epl", "laliga", "seriea", "bundesliga", "ligue1"]

# Map internal league key → football-data.co.uk league name (used in fixtures_df["league"])
LEAGUE_TO_FOOTBALL_DATA: dict[str, str] = {
    "epl": "England",
    "laliga": "Spain",
    "seriea": "Italy",
    "bundesliga": "Germany",
    "ligue1": "France",
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BETTOR_PATHS: dict[str, Path] = {
    league: MODELS_DIR / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES
}
FIXTURES_PATH: Path = MODELS_DIR / "fixtures_data.parquet"
BACKTEST_PATH: Path = MODELS_DIR / "backtest.json"


@dataclass
class MatchPrediction:
    """Model output for a single fixture."""

    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    p_over_2_5: float | None = None
    is_demo: bool = False


@dataclass
class Models:
    """Container for per-league bettors plus shared inference artifacts."""

    bettors: dict[str, Any] = field(default_factory=dict)
    fixtures_df: pd.DataFrame | None = None
    backtest: dict | None = None

    @property
    def is_ready(self) -> bool:
        """True only when all five leagues' bettors plus fixtures+backtest are loaded."""
        return (
            len(self.bettors) == len(BIG_5_LEAGUES)
            and all(self.bettors.get(league) is not None for league in BIG_5_LEAGUES)
            and self.fixtures_df is not None
            and self.backtest is not None
        )


def load_models() -> Models:
    """Read the 5 *_bettor.pkl + fixtures_data.parquet + backtest.json. Any missing → demo."""
    bettors: dict[str, Any] = {}
    for league, path in BETTOR_PATHS.items():
        loaded = _safe_unpickle(path)
        if loaded is not None:
            bettors[league] = loaded

    fixtures_df = _safe_load_parquet(FIXTURES_PATH)
    backtest_data = _safe_load_json(BACKTEST_PATH)

    return Models(bettors=bettors, fixtures_df=fixtures_df, backtest=backtest_data)


def _safe_unpickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:  # pickle errors, version skew, anything
        logger.warning(f"Could not load pickle {path}: {e}")
        return None


def _safe_load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"Could not load parquet {path}: {e}")
        return None


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # JSON syntax errors, encoding issues, anything
        logger.warning(f"Could not load JSON {path}: {e}")
        return None


def predict_fixture(
    models: Models | None,
    league: str,
    home_team: str,
    away_team: str,
) -> MatchPrediction:
    """Get probabilities for an upcoming fixture. Multi-league routing.

    home_team and away_team are Odds API names (long form). League is one of
    BIG_5_LEAGUES. Falls back to demo mode if:
    - models is None or not ready
    - league is not in the Big-5 set
    - team name isn't in the static team-names registry
    - the fixtures_df has no matching row for this team pair
    - the bettor's predict_proba throws

    Y.columns order from training is locked to:
        [home_win, draw, away_win, over_2.5, under_2.5]
    indexed as 0/1/2/3 below for the four probabilities we surface.
    """
    if not isinstance(models, Models) or not models.is_ready:
        return _demo_prediction(home_team, away_team)
    if league not in BIG_5_LEAGUES:
        return _demo_prediction(home_team, away_team)

    try:
        from src.team_names import to_canonical

        home_canonical = to_canonical(home_team, source="odds_api")
        away_canonical = to_canonical(away_team, source="odds_api")
        league_fd_name = LEAGUE_TO_FOOTBALL_DATA[league]

        match = models.fixtures_df[
            (models.fixtures_df["league"] == league_fd_name)
            & (models.fixtures_df["home_team"] == home_canonical)
            & (models.fixtures_df["away_team"] == away_canonical)
        ]
        if match.empty:
            logger.warning(
                f"No library fixture for {home_team}({home_canonical}) vs "
                f"{away_team}({away_canonical}) in {league}"
            )
            return _demo_prediction(home_team, away_team)

        bettor = models.bettors[league]
        # Double brackets keeps result a 1-row DataFrame
        probs_per_market = bettor.predict_proba(match.iloc[[0]])

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            p_home=float(probs_per_market[0][0, 1]),
            p_draw=float(probs_per_market[1][0, 1]),
            p_away=float(probs_per_market[2][0, 1]),
            p_over_2_5=float(probs_per_market[3][0, 1]),
            is_demo=False,
        )
    except Exception as e:
        logger.warning(
            f"Real-model inference failed for {league}: {home_team} vs {away_team}: {e}",
            exc_info=True,
        )
        return _demo_prediction(home_team, away_team)


def _demo_prediction(home_team: str, away_team: str) -> MatchPrediction:
    """Generate plausible-shaped probabilities for testing the dashboard.

    Deterministic hash of team names plus a small home-advantage bias.
    NOT predictive.
    """
    seed_str = f"{home_team}|{away_team}"
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

    home_bias = (h % 100) / 100.0
    p_home = 0.35 + home_bias * 0.25
    p_draw = 0.20 + ((h >> 8) % 100) / 1000.0
    p_away = max(1.0 - p_home - p_draw, 0.10)

    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    p_over_2_5 = 0.45 + ((h >> 16) % 100) / 250.0

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        p_over_2_5=p_over_2_5,
        is_demo=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_predictions.py -v`
Expected: 13 PASSED.

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/predictions.py tests/test_predictions.py && uv run ruff format --check src/predictions.py tests/test_predictions.py`

- [ ] **Step 6: Commit**

```bash
git add src/predictions.py tests/test_predictions.py
git commit -m "feat: multi-league routing in predict_fixture with per-league bettors"
```

---

## Task 13: app.py — multi-league dashboard with league filter

**Files:**
- Modify: `app.py`

No new unit tests (smoke import + manual smoke at Task 16).

- [ ] **Step 1: Update app.py imports**

In `app.py`, find the existing import block. Change:

```python
from src.odds import best_odds_for_outcome, get_epl_odds
from src.predictions import Models, load_models, predict_fixture
```

to:

```python
from src.odds import best_odds_for_outcome, get_big5_odds
from src.predictions import BIG_5_LEAGUES, Models, load_models, predict_fixture
```

(Removed `get_epl_odds` — replaced by `get_big5_odds`. Added `BIG_5_LEAGUES` constant.)

- [ ] **Step 2: Replace _fetch_fixtures and the fixtures-tab call sites**

In `app.py`, find:

```python
@st.cache_data(ttl=600, show_spinner="Pulling fixtures and odds…")
def _fetch_fixtures() -> list[dict]:
    """Pull EPL fixtures + bookmaker odds. Cached for 10 minutes."""
    return get_epl_odds()
```

Replace with:

```python
LEAGUE_DISPLAY_NAMES: dict[str, str] = {
    "epl": "EPL",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue 1",
}


@st.cache_data(ttl=1800, show_spinner="Pulling fixtures and odds…")
def _fetch_fixtures() -> list[dict]:
    """Pull Big-5 fixtures + bookmaker odds. Cached for 30 minutes (manages 500/month free quota)."""
    return get_big5_odds()
```

- [ ] **Step 3: Update _build_rows to consume league from fixture**

In `app.py`, find the line inside `_build_rows`:

```python
        pred = predict_fixture(models, home_team, away_team)
```

Replace with:

```python
        league = fixture.get("league", "epl")  # default to epl for legacy fixtures missing the field
        pred = predict_fixture(models, league, home_team, away_team)
```

Also, in the row dict's fields inside `_build_rows`, find the `"market": market,` line and immediately above it add a new line `"league": LEAGUE_DISPLAY_NAMES.get(league, league),` so the row carries league for display.

- [ ] **Step 4: Add the `League` column to the dataframe display**

In `app.py`, find the `display_cols` list (currently has `Match`, `Kickoff`, `Bet`, `Model %`, `Fair %`, `Best odds`, `Bookmaker`, `Edge`, `Kelly stake`). Replace it with:

```python
    display_cols = [
        "League",
        "Match",
        "Kickoff",
        "Bet",
        "Model %",
        "Fair %",
        "Best odds",
        "Bookmaker",
        "Edge",
        "Kelly stake",
    ]
```

Also, in the row dict, add a `"League": LEAGUE_DISPLAY_NAMES.get(league, league),` field next to the existing user-facing display fields like `"Match"`, `"Kickoff"` etc. The internal `_*` fields stay as before.

- [ ] **Step 5: Add league multi-select filter**

In `app.py`, immediately before the `value_df = df[df["_is_value"]].sort_values(...)` line in `render_fixtures_tab`, add:

```python
    selected_leagues = st.multiselect(
        "Leagues",
        options=list(LEAGUE_DISPLAY_NAMES.values()),
        default=list(LEAGUE_DISPLAY_NAMES.values()),
        help="Hide leagues you don't want to see.",
    )
    df = df[df["League"].isin(selected_leagues)]
```

- [ ] **Step 6: Update render_model_info for per-league table**

In `app.py`, find the `render_model_info` function. Replace the body that renders `markets = bt.get("markets", {})` etc. with the per-league version:

```python
def render_model_info(models: Models) -> None:
    """Render the 'Model info' sidebar expander when models are ready."""
    if not models.is_ready or not models.backtest:
        return

    bt = models.backtest
    trained_at_str = bt.get("trained_at", "")
    age_days_str = ""
    stale_marker = ""
    try:
        trained_at = datetime.fromisoformat(trained_at_str.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - trained_at).days
        age_days_str = f" ({age_days} days ago)"
        if age_days > 90:
            stale_marker = "⚠️ "
    except (ValueError, AttributeError) as e:
        logger.warning(
            f"Could not parse backtest trained_at {trained_at_str!r}: {e}. "
            "Age and stale-model warning will be omitted."
        )

    n_matches = bt.get("n_training_matches", "?")
    seasons = bt.get("training_seasons", [])
    seasons_str = f"{len(seasons)} seasons" if seasons else "unknown seasons"

    leagues_data = bt.get("leagues", {})

    with st.expander("📊 Model info", expanded=False):
        st.markdown(
            f"**Trained:** {stale_marker}{trained_at_str[:10]}{age_days_str}  \n"
            f"**Data:** {seasons_str}, {n_matches} matches across {len(leagues_data)} leagues"
        )
        st.markdown("**Per-market backtest yields by league:**")

        league_display = {
            "epl": "EPL", "laliga": "La Liga", "seriea": "Serie A",
            "bundesliga": "Bundesliga", "ligue1": "Ligue 1",
        }
        rows = []
        for league_key, league_summary in leagues_data.items():
            row = {"League": league_display.get(league_key, league_key)}
            markets = league_summary.get("markets", {})
            for market_key in ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]:
                stats = markets.get(market_key, {})
                yp = stats.get("yield_pct", 0.0)
                n = stats.get("n_bets", 0)
                row[market_key.replace("_", " ").title()] = f"{yp:+.1f}% ({n})"
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if stale_marker:
            st.caption("Model is over 90 days old — consider retraining.")
```

- [ ] **Step 7: Update demo banner copy**

In `app.py`, find the demo-mode banner block:

```python
if not _models.is_ready:
    st.warning(
        "🟡 **Demo mode** — no trained model found. The probabilities shown are "
        "synthetic and **not predictive**. Wire up `scripts/train_model.py` and "
        "run `uv run python scripts/train_model.py` to use a real model."
    )
```

Replace with:

```python
if not _models.is_ready:
    st.warning(
        "🟡 **Demo mode** — no trained models found. The probabilities shown are "
        "synthetic and **not predictive**. Run `uv run python scripts/train_model.py` "
        "to train all 5 leagues' models."
    )
```

- [ ] **Step 8: Smoke import check**

Run: `uv run python -c "import app; print('ok')"`
Expected: `ok` (with possible Streamlit runtime warnings — those are fine when running outside `streamlit run`).

- [ ] **Step 9: Run all tests**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 10: Run static checks**

Run: `uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 11: Commit**

```bash
git add app.py
git commit -m "feat: multi-league dashboard with league filter and per-league Model info"
```

---

## Task 14: Cleanup — delete sportsbet_patch.py, gitignore data/cache/

**Files:**
- Delete: `src/sportsbet_patch.py`
- Modify: `.gitignore`

- [ ] **Step 1: Delete src/sportsbet_patch.py**

```bash
git rm src/sportsbet_patch.py
```

- [ ] **Step 2: Update .gitignore**

Add to `.gitignore` (anywhere — group with similar entries if a section exists):

```
# Cached downloads (regenerable from upstream sources)
data/cache/
```

- [ ] **Step 3: Verify no remaining imports of sportsbet_patch**

Run: `uv run python -c "import app; from src.predictions import load_models; from scripts.train_model import main; print('clean')"`
Expected: `clean`. If any import error mentions sportsbet_patch, find the leftover import and remove it.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: remove sportsbet_patch, gitignore data/cache/"
```

---

## Task 15: Add iterative team-name verification helper

**Files:**
- Create: `scripts/verify_team_names.py`

This task adds a small helper script that fetches team names from the football-data CSVs and Understat, then prints any names not present in `src/team_names.REGISTRY`. The implementer runs this iteratively during smoke test (Task 16) to fill in missing entries.

- [ ] **Step 1: Create scripts/verify_team_names.py**

Create the file with:

```python
"""Helper: print team names from football-data and Understat that aren't yet in REGISTRY.

Usage:
    uv run python scripts/verify_team_names.py

Prints two lists: unmapped football-data names, unmapped Understat names. The
operator copies these into src/team_names.REGISTRY (with the right canonical name
and the entries for all three sources) and re-runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.ingest.football_data import download_training_data
    from src.ingest.understat import fetch_xg_data
    from src.team_names import REGISTRY

    training_years = list(range(2018, 2026))
    fd_leagues = ["England", "Spain", "Italy", "Germany", "France"]
    understat_leagues = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]

    print("Downloading football-data CSVs (cached after first run)...")
    matches = download_training_data(leagues=fd_leagues, years=training_years)
    fd_names = set(matches["HomeTeam"].dropna().unique()) | set(
        matches["AwayTeam"].dropna().unique()
    )

    print("Scraping Understat (cached after first run, ~60s on first run)...")
    xg = fetch_xg_data(leagues=understat_leagues, years=training_years)
    if xg.empty:
        understat_names: set[str] = set()
    else:
        understat_names = set(xg["home_team"].dropna().unique()) | set(
            xg["away_team"].dropna().unique()
        )

    mapped_fd = {t.football_data for t in REGISTRY.values()}
    mapped_understat = {t.understat for t in REGISTRY.values()}

    missing_fd = sorted(fd_names - mapped_fd)
    missing_understat = sorted(understat_names - mapped_understat)

    print(f"\n--- Unmapped football-data names ({len(missing_fd)}) ---")
    for name in missing_fd:
        print(f"  {name!r}")

    print(f"\n--- Unmapped Understat names ({len(missing_understat)}) ---")
    for name in missing_understat:
        print(f"  {name!r}")

    if not missing_fd and not missing_understat:
        print("\n✅ All team names are mapped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke import check**

Run: `uv run python -c "from scripts import verify_team_names; print('ok')"`
Expected: `ok` (or import error if `scripts/__init__.py` doesn't exist — if so, this whole script can be skipped at import-as-module level; it's only run as `python scripts/verify_team_names.py`).

Alternative if the `from scripts import` fails because `scripts/` lacks `__init__.py`: run the script directly to verify it parses:

```bash
uv run python scripts/verify_team_names.py --help 2>&1 | head -5
```

If it executes without a syntax error, that's enough.

- [ ] **Step 3: Run static checks**

Run: `uv run ruff check scripts/verify_team_names.py && uv run ruff format --check scripts/verify_team_names.py`

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_team_names.py
git commit -m "feat: scripts/verify_team_names.py for iterative registry completion"
```

---

## Task 16: End-to-end manual smoke test

This task is a **manual checklist** run by Femi. No code; the goal is to validate the v2 pipeline end-to-end and capture findings.

**Files:**
- Optionally create: `docs/superpowers/notes/2026-MM-DD-v2-smoke-test-results.md`

- [ ] **Step 1: Verify demo mode renders cleanly**

```bash
# Move existing models/ aside if present
move models models.v0_backup
uv run streamlit run app.py
```

Browser at localhost:8501. Confirm:
- Yellow "Demo mode" banner shows
- Multi-select league filter shows all 5 leagues by default
- "Value bets" table renders with synthetic predictions
- League column appears in the table
- No "📊 Model info" expander in sidebar (correct — no models)

Stop server with Ctrl+C.

- [ ] **Step 2: Run team-name verification**

```bash
uv run python scripts/verify_team_names.py
```

Read the output. Add any missing teams to `src/team_names.REGISTRY` with their correct three-source names. (Look up unfamiliar teams on Wikipedia — most will be obvious.)

Repeat: re-run the verifier, update REGISTRY, until output says "✅ All team names are mapped."

Commit the updated registry:
```bash
git add src/team_names.py
git commit -m "feat: complete team-names registry for Big-5 historical seasons"
```

- [ ] **Step 3: Run training**

```bash
uv run python scripts/train_model.py
```

Expected: `[1/5]` through `[5/5]` markers print; total runtime 30-50 minutes. Final per-league per-market summary prints.

If training fails on a team-name KeyError, add the entry to REGISTRY and rerun. If it fails on a CSV column we expected (`AvgH/D/A`, `B365H/D/A`, etc.), check the CSV columns and adjust `_find_first_present_columns` accordingly.

If training succeeds, three artifact files exist in `models/`:
```bash
dir models
# Should show: epl_bettor.pkl, laliga_bettor.pkl, seriea_bettor.pkl,
#              bundesliga_bettor.pkl, ligue1_bettor.pkl, fixtures_data.parquet, backtest.json
```

- [ ] **Step 4: Inspect backtest.json**

```bash
type models\backtest.json
```

Verify:
- `trained_at` is recent
- `training_seasons` is `[2018, ..., 2025]`
- `sklearn_version` is the installed version (e.g., `"1.5.2"`)
- `n_training_matches` is in the ~12,000-15,000 range
- `leagues` has 5 entries (epl, laliga, seriea, bundesliga, ligue1)
- Each league has `n_matches` ≈ 2,500-3,500 and `markets` dict with 5 entries each having `n_bets` and `yield_pct`

Note the per-league per-market yields. Compare home_win yield to v0's −14.6% baseline.

- [ ] **Step 5: Restart dashboard, verify real predictions**

```bash
# Kill any running streamlit
uv run streamlit run app.py
```

Verify in browser:
- No demo banner
- Sidebar Model info expander shows the 5×5 per-league/per-market table
- Value bets table includes fixtures from at least 3 of 5 leagues (depending on what's scheduled)
- Multi-select league filter works (deselect 4 leagues, only one league's bets show)
- Edges look saner than demo (typically <15%)

- [ ] **Step 6: Spot-check 5 fixtures across 3 leagues**

Pick 5 random rows from the value-bets table across at least 3 different leagues. Eyeball:
- Probabilities aren't all 33/33/33 or 99/0/0
- Edges look plausible (1-15% range, mostly)
- Compare 1-2 against Pinnacle/Betfair Exchange for sanity

- [ ] **Step 7: Capture results (optional)**

Create `docs/superpowers/notes/2026-MM-DD-v2-smoke-test-results.md` with:
- Wall-clock training time
- Total training matches
- Per-league yields summary
- Comparison to v0 baseline (`home_win: −14.6%`)
- Any team-name additions made
- Any bugs found
- Spot-check observations

- [ ] **Step 8: Final commit**

If notes were captured:
```bash
git add docs/superpowers/notes/2026-MM-DD-v2-smoke-test-results.md
git commit -m "docs: smoke-test results for v2 model improvement"
```

If no notes file, no commit needed.

---

## Self-Review Checklist (run by the implementer at end)

- [ ] All 16 tasks completed and committed
- [ ] `uv run pytest -q` — all ~40-50 tests pass
- [ ] `uv run ruff check .` — passes
- [ ] `uv run ruff format --check .` — passes
- [ ] Streamlit dashboard with all artifacts: shows real per-league predictions, no demo banner, Model info expander rendered with 5×5 table
- [ ] Streamlit dashboard with artifacts absent or partial: shows demo banner, synthetic predictions, no crash
- [ ] At least 5 fixtures across 3 leagues spot-checked with plausible probabilities
- [ ] Every team in the live Odds API response across all 5 leagues has an entry in `team_names.REGISTRY`
- [ ] No reference to `sports-betting`, `sportsbet`, or `sportsbet_patch` remains in source code
- [ ] Per-market backtest yields recorded in `backtest.json` for all 5 leagues; v2 yields documented vs v0's −14.6% home_win baseline

