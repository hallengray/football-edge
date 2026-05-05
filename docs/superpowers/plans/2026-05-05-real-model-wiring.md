# Real Model Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace synthetic match probabilities with real, calibrated predictions from a single multi-output `sports-betting` `ClassifierBettor`. Cover match-result (1X2) and goals-totals (Over/Under 2.5) markets. Surface backtest results in the dashboard.

**Architecture:** One `ClassifierBettor` wrapping a sklearn pipeline (`OneHotEncoder` + `SimpleImputer` + `MultiOutputClassifier(CalibratedClassifierCV(GradientBoostingClassifier()))`). Trained on EPL seasons 2018-2025 from the library's data source. Inference uses `loader.extract_fixtures_data()` + a static team-name map between Odds API and library naming conventions. Artifacts persisted atomically as `epl_bettor.pkl` + `epl_loader.pkl` + `backtest.json`.

**Tech Stack:** Python 3.11, sports-betting 0.12.1, scikit-learn, pandas, Streamlit, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-05-05-real-model-wiring-design.md`

---

## Task 1: Add pytest, pin sports-betting, scaffold tests/

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Update `pyproject.toml`**

Edit `pyproject.toml` to pin `sports-betting==0.12.1` (was `>=0.6`) and add `pytest>=8.0` to dev deps.

```toml
[project]
name = "football-edge"
version = "0.1.0"
description = "Personal football value betting tool — EPL focus"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "streamlit>=1.40",
    "pandas>=2.2",
    "requests>=2.32",
    "python-dotenv>=1.0",
    "supabase>=2.10",
    "sports-betting==0.12.1",
]

[dependency-groups]
dev = [
    "ruff>=0.8",
    "pytest>=8.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create `tests/__init__.py`**

Empty file. Marks `tests` as a package so pytest discovery works cleanly on Windows.

```python
```

(The file is genuinely empty — zero bytes. Don't add a docstring.)

- [ ] **Step 3: Sync deps**

Run: `uv sync`
Expected: pytest installed, sports-betting unchanged (already at 0.12.1).

- [ ] **Step 4: Verify pytest can collect**

Run: `uv run pytest --collect-only`
Expected: `0 tests collected` (no tests yet, but pytest itself runs without error).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py
git commit -m "chore: add pytest dev dep, pin sports-betting==0.12.1"
```

---

## Task 2: Regression tests for src/value.py

The math in `value.py` is correct today. We add tests now to lock that correctness so future refactors don't break it silently.

**Files:**
- Create: `tests/test_value.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_value.py`:

```python
"""Regression tests for the value/Kelly math in src/value.py."""
from __future__ import annotations

import math

from src.value import (
    assess_value,
    kelly_fraction,
    remove_bookmaker_margin,
)


def test_remove_bookmaker_margin_normalises_to_one() -> None:
    # Raw implied probs from typical 1X2 odds with ~5% margin
    raw = [0.55, 0.30, 0.20]  # sums to 1.05
    fair = remove_bookmaker_margin(raw)
    assert math.isclose(sum(fair), 1.0, abs_tol=1e-9)
    # Largest stays largest
    assert fair[0] > fair[1] > fair[2]


def test_remove_bookmaker_margin_handles_zero_total() -> None:
    # Edge case: degenerate input
    assert remove_bookmaker_margin([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_kelly_fraction_zero_when_no_edge() -> None:
    # Model says 30%, odds 2.0 → fair-implied 50%, no edge
    assert kelly_fraction(0.30, 2.0) == 0.0


def test_kelly_fraction_textbook_case() -> None:
    # Classic example: p=0.6, odds=2.0 → b=1, q=0.4 → f* = (1*0.6 - 0.4)/1 = 0.2
    assert math.isclose(kelly_fraction(0.6, 2.0), 0.2, abs_tol=1e-9)


def test_assess_value_flags_value_bet() -> None:
    result = assess_value(
        model_prob=0.55,
        decimal_odds=2.20,
        bookmaker="Test Book",
        fair_implied_prob=0.45,  # 10pp edge
        value_threshold=0.05,
        kelly_multiplier=0.25,
    )
    assert result.is_value_bet is True
    assert math.isclose(result.value_pct, 0.10, abs_tol=1e-9)
    # Quarter Kelly of full Kelly: full = (1.20*0.55 - 0.45)/1.20 ≈ 0.175 → 0.0438
    assert 0.04 < result.kelly_stake_fraction < 0.05


def test_assess_value_below_threshold() -> None:
    result = assess_value(
        model_prob=0.50,
        decimal_odds=2.10,
        bookmaker="Test Book",
        fair_implied_prob=0.48,  # 2pp edge — below 5% threshold
        value_threshold=0.05,
        kelly_multiplier=0.25,
    )
    assert result.is_value_bet is False
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_value.py -v`
Expected: 6 PASSED in <1s.

- [ ] **Step 3: Run static checks**

Run: `uv run ruff check tests/test_value.py && uv run ruff format --check tests/test_value.py`
Expected: both pass.

If `ruff format --check` fails: run `uv run ruff format tests/test_value.py` and re-check.

- [ ] **Step 4: Commit**

```bash
git add tests/test_value.py
git commit -m "test: regression suite for value/Kelly math"
```

---

## Task 3: Regression tests for src/predictions.py demo mode

Lock the demo-mode behaviour before we change `predict_fixture()`. These tests must pass against the existing code.

**Files:**
- Create: `tests/test_predictions.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_predictions.py`:

```python
"""Tests for src/predictions.py.

This module is incrementally updated alongside the real-model implementation.
Phase 1 (this commit) — regression tests for the existing demo mode.
Phase 2 (Task 5) — adds tests for load_models() and Models dataclass.
Phase 3 (Task 6) — adds tests for the real branch and team-name fallback.
"""
from __future__ import annotations

import math

from src.predictions import _demo_prediction


def test_demo_prediction_returns_demo_flag() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    assert pred.is_demo is True


def test_demo_prediction_probabilities_sum_to_one() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_demo_prediction_is_deterministic() -> None:
    # Same input → same output (hash-based seeding)
    a = _demo_prediction("Arsenal", "Chelsea")
    b = _demo_prediction("Arsenal", "Chelsea")
    assert a.p_home == b.p_home
    assert a.p_draw == b.p_draw
    assert a.p_away == b.p_away


def test_demo_prediction_swap_changes_output() -> None:
    # home/away swap should NOT yield identical probabilities
    a = _demo_prediction("Arsenal", "Chelsea")
    b = _demo_prediction("Chelsea", "Arsenal")
    assert (a.p_home, a.p_draw, a.p_away) != (b.p_home, b.p_draw, b.p_away)


def test_demo_prediction_includes_over_2_5() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    assert pred.p_over_2_5 is not None
    assert 0.0 <= pred.p_over_2_5 <= 1.0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_predictions.py -v`
Expected: 5 PASSED.

- [ ] **Step 3: Run static checks**

Run: `uv run ruff check tests/test_predictions.py && uv run ruff format --check tests/test_predictions.py`
Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_predictions.py
git commit -m "test: regression suite for predictions demo mode"
```

---

## Task 4: Create src/team_names.py with EPL mapping

Static mapping between Odds API team names and library team names. We start with names verified from the screenshots earlier; some entries flagged as "verify at training time" — Task 9's smoke run will reveal any mismatches.

**Files:**
- Create: `tests/test_team_names.py`
- Create: `src/team_names.py`

- [ ] **Step 1: Write tests first**

Create `tests/test_team_names.py`:

```python
"""Tests for src/team_names.py mapping."""
from __future__ import annotations

import pytest

from src.team_names import (
    ODDS_API_TO_LIBRARY,
    from_library,
    to_library,
)


def test_to_library_known_team() -> None:
    assert to_library("Arsenal") == "Arsenal"
    assert to_library("Wolverhampton Wanderers") == "Wolves"


def test_to_library_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unmapped Odds API team"):
        to_library("Real Madrid")  # not in EPL


def test_from_library_round_trip() -> None:
    # Every Odds API name should round-trip via from_library
    for odds_name, lib_name in ODDS_API_TO_LIBRARY.items():
        assert from_library(lib_name) == odds_name


def test_from_library_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unmapped library team"):
        from_library("Barcelona")


def test_map_has_twenty_teams() -> None:
    # EPL has 20 teams. If this assertion fails after relegation/promotion,
    # update ODDS_API_TO_LIBRARY accordingly.
    assert len(ODDS_API_TO_LIBRARY) == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_team_names.py -v`
Expected: ImportError or ModuleNotFoundError on `src.team_names`.

- [ ] **Step 3: Implement src/team_names.py**

Create `src/team_names.py`:

```python
"""Map between Odds API team names (long form) and sports-betting library team names (short form).

The library uses football-data.co.uk-style abbreviations ("Wolves", "Brighton")
while The Odds API uses full club names ("Wolverhampton Wanderers", "Brighton and Hove Albion").

This map is hardcoded for the 20 teams in the current EPL season. After August's
promotion/relegation cycle, three teams change — update the map and the
test_map_has_twenty_teams test will pass once you've added/removed the right entries.

If a team appears in The Odds API that isn't in this map, `to_library()` raises a clear
KeyError naming the unmapped team. The dashboard catches this and falls back to demo
mode for that fixture only — so missing entries are self-diagnosing without crashing.
"""
from __future__ import annotations

ODDS_API_TO_LIBRARY: dict[str, str] = {
    # Verified at impl time against library data — adjust right-hand values
    # if Task 9's smoke run reveals different naming.
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds United": "Leeds",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham Forest",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

_LIBRARY_TO_ODDS_API: dict[str, str] = {v: k for k, v in ODDS_API_TO_LIBRARY.items()}


def to_library(odds_api_name: str) -> str:
    if odds_api_name not in ODDS_API_TO_LIBRARY:
        raise KeyError(
            f"Unmapped Odds API team: {odds_api_name!r}. "
            "Add to src/team_names.ODDS_API_TO_LIBRARY."
        )
    return ODDS_API_TO_LIBRARY[odds_api_name]


def from_library(library_name: str) -> str:
    if library_name not in _LIBRARY_TO_ODDS_API:
        raise KeyError(
            f"Unmapped library team: {library_name!r}. "
            "Add to src/team_names.ODDS_API_TO_LIBRARY (forward map)."
        )
    return _LIBRARY_TO_ODDS_API[library_name]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_team_names.py -v`
Expected: 5 PASSED.

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/team_names.py tests/test_team_names.py && uv run ruff format --check src/team_names.py tests/test_team_names.py`
Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add src/team_names.py tests/test_team_names.py
git commit -m "feat: add src/team_names module for Odds API ↔ library naming"
```

---

## Task 5: Add Models dataclass and load_models() to src/predictions.py

This task introduces the new `Models` container and `load_models()` helper. The existing `predict_fixture()` is left untouched in this commit — the real branch is wired in Task 6.

**Files:**
- Modify: `src/predictions.py`
- Modify: `tests/test_predictions.py`

- [ ] **Step 1: Write tests first**

Append to `tests/test_predictions.py`:

```python


# ─── Phase 2: Models dataclass and load_models() ───

import json
from pathlib import Path

from src.predictions import Models, load_models


def test_models_is_ready_false_when_all_none() -> None:
    m = Models(bettor=None, loader=None, backtest=None)
    assert m.is_ready is False


def test_models_is_ready_true_when_all_present() -> None:
    m = Models(bettor=object(), loader=object(), backtest={"trained_at": "x"})
    assert m.is_ready is True


def test_models_is_ready_false_when_one_missing() -> None:
    m = Models(bettor=object(), loader=None, backtest={"trained_at": "x"})
    assert m.is_ready is False


def test_load_models_returns_unready_when_files_missing(tmp_path, monkeypatch) -> None:
    # Point the module's MODELS_DIR at an empty temp directory
    monkeypatch.setattr("src.predictions.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.predictions.BETTOR_PATH", tmp_path / "epl_bettor.pkl")
    monkeypatch.setattr("src.predictions.LOADER_PATH", tmp_path / "epl_loader.pkl")
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", tmp_path / "backtest.json")

    models = load_models()
    assert models.is_ready is False
    assert models.bettor is None
    assert models.loader is None
    assert models.backtest is None


def test_load_models_returns_ready_when_all_artifacts_present(tmp_path, monkeypatch) -> None:
    import pickle

    monkeypatch.setattr("src.predictions.MODELS_DIR", tmp_path)
    bettor_path = tmp_path / "epl_bettor.pkl"
    loader_path = tmp_path / "epl_loader.pkl"
    backtest_path = tmp_path / "backtest.json"
    monkeypatch.setattr("src.predictions.BETTOR_PATH", bettor_path)
    monkeypatch.setattr("src.predictions.LOADER_PATH", loader_path)
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", backtest_path)

    # Stand-in objects (the real types are only validated at use time)
    with bettor_path.open("wb") as f:
        pickle.dump({"fake": "bettor"}, f)
    with loader_path.open("wb") as f:
        pickle.dump({"fake": "loader"}, f)
    backtest_path.write_text(json.dumps({"trained_at": "2026-05-05T00:00:00Z"}))

    models = load_models()
    assert models.is_ready is True
    assert models.bettor == {"fake": "bettor"}
    assert models.loader == {"fake": "loader"}
    assert models.backtest == {"trained_at": "2026-05-05T00:00:00Z"}


def test_load_models_handles_corrupt_backtest_json(tmp_path, monkeypatch) -> None:
    import pickle

    monkeypatch.setattr("src.predictions.MODELS_DIR", tmp_path)
    bettor_path = tmp_path / "epl_bettor.pkl"
    loader_path = tmp_path / "epl_loader.pkl"
    backtest_path = tmp_path / "backtest.json"
    monkeypatch.setattr("src.predictions.BETTOR_PATH", bettor_path)
    monkeypatch.setattr("src.predictions.LOADER_PATH", loader_path)
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", backtest_path)

    with bettor_path.open("wb") as f:
        pickle.dump({"fake": "bettor"}, f)
    with loader_path.open("wb") as f:
        pickle.dump({"fake": "loader"}, f)
    backtest_path.write_text("{not valid json")

    models = load_models()
    assert models.is_ready is False
    assert models.backtest is None
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_predictions.py -v`
Expected: ImportError on `Models` and `load_models` (don't exist yet).

- [ ] **Step 3: Implement Models + load_models in src/predictions.py**

Replace the entire contents of `src/predictions.py` with:

```python
"""Model loading and prediction wrapper.

Wraps a `sports-betting` library multi-output ClassifierBettor and the primed
SoccerDataLoader needed for inference. Falls back to demo mode (synthetic
probabilities) when artifacts are missing or inference fails.

Three artifacts are loaded together:
    models/epl_bettor.pkl   — the fitted ClassifierBettor
    models/epl_loader.pkl   — the SoccerDataLoader, primed by extract_train_data
    models/backtest.json    — per-market backtest summary (for the dashboard)

If any artifact is missing or unloadable, the entire Models container reports
is_ready=False and the dashboard falls back to demo predictions.
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

MODELS_DIR = Path(__file__).parent.parent / "models"
BETTOR_PATH = MODELS_DIR / "epl_bettor.pkl"
LOADER_PATH = MODELS_DIR / "epl_loader.pkl"
BACKTEST_PATH = MODELS_DIR / "backtest.json"


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
    """Container for the three artifacts plus session-cached fixtures DataFrame."""

    bettor: Any | None
    loader: Any | None
    backtest: dict | None
    fixtures_df: pd.DataFrame | None = field(default=None)

    @property
    def is_ready(self) -> bool:
        return (
            self.bettor is not None
            and self.loader is not None
            and self.backtest is not None
        )


def load_models() -> Models:
    """Load all three artifacts. Any missing or corrupt → demo mode (is_ready=False)."""
    bettor = _safe_unpickle(BETTOR_PATH)
    loader = _safe_unpickle(LOADER_PATH)
    backtest_data = _safe_load_json(BACKTEST_PATH)
    return Models(bettor=bettor, loader=loader, backtest=backtest_data)


def _safe_unpickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:  # pickle errors, version skew, anything
        logger.warning(f"Could not load pickle {path}: {e}")
        return None


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load JSON {path}: {e}")
        return None


def predict_fixture(
    models: Models | Any | None,
    home_team: str,
    away_team: str,
) -> MatchPrediction:
    """Get probabilities for an upcoming fixture.

    Until Task 6, this function only supports demo mode. The real branch is wired in
    the next commit — for now any non-None `models` argument that's not a `Models`
    instance is treated as legacy and ignored.
    """
    if isinstance(models, Models) and not models.is_ready:
        return _demo_prediction(home_team, away_team)
    if not isinstance(models, Models):
        # Legacy code path (will be removed in Task 10 alongside app.py changes)
        return _demo_prediction(home_team, away_team)
    # Real-model branch wired in Task 6
    return _demo_prediction(home_team, away_team)


def _demo_prediction(home_team: str, away_team: str) -> MatchPrediction:
    """Generate plausible-shaped probabilities for testing the dashboard.

    Uses a deterministic hash of the team names to get repeatable values
    plus a small home-advantage bias. NOT predictive — purely for demo.
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


# ─── Compatibility shims (removed in Task 10 alongside app.py changes) ───

def model_exists() -> bool:
    """Deprecated: use Models.is_ready via load_models()."""
    return BETTOR_PATH.exists() and LOADER_PATH.exists() and BACKTEST_PATH.exists()


def load_model() -> Any | None:
    """Deprecated: use load_models()."""
    if not BETTOR_PATH.exists():
        return None
    with BETTOR_PATH.open("rb") as f:
        return pickle.load(f)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_predictions.py -v`
Expected: 11 PASSED (5 phase-1 demo tests + 6 new phase-2 tests).

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/predictions.py tests/test_predictions.py && uv run ruff format --check src/predictions.py tests/test_predictions.py`
Expected: both pass. If format check fails, run `uv run ruff format src/predictions.py tests/test_predictions.py` and re-check.

- [ ] **Step 6: Commit**

```bash
git add src/predictions.py tests/test_predictions.py
git commit -m "feat: add Models dataclass and load_models() to predictions"
```

---

## Task 6: Wire real branch in predict_fixture()

Replace the demo-mode-only stub with the real inference path: team-name normalisation → fixture lookup → `bettor.predict_proba()`. Keeps the per-fixture demo fallback for any error.

**Files:**
- Modify: `src/predictions.py`
- Modify: `tests/test_predictions.py`

- [ ] **Step 1: Write tests first**

Append to `tests/test_predictions.py`:

```python


# ─── Phase 3: real branch in predict_fixture() ───

from unittest.mock import MagicMock

from src.predictions import predict_fixture


def test_predict_fixture_demo_when_models_unready() -> None:
    models = Models(bettor=None, loader=None, backtest=None)
    pred = predict_fixture(models, "Arsenal", "Chelsea")
    assert pred.is_demo is True


def test_predict_fixture_demo_when_team_unmapped() -> None:
    # Models look ready but team name isn't in our static map
    bettor = MagicMock()
    loader = MagicMock()
    models = Models(bettor=bettor, loader=loader, backtest={"trained_at": "x"})
    pred = predict_fixture(models, "Real Madrid", "Chelsea")
    assert pred.is_demo is True
    bettor.predict_proba.assert_not_called()


def test_predict_fixture_demo_when_no_library_fixture_match() -> None:
    # Models ready, teams mapped, but library has no matching fixture row
    bettor = MagicMock()
    loader = MagicMock()
    loader.extract_fixtures_data.return_value = (
        pd.DataFrame({"home_team": ["Liverpool"], "away_team": ["Everton"]}),
        None,
        None,
    )
    models = Models(bettor=bettor, loader=loader, backtest={"trained_at": "x"})
    pred = predict_fixture(models, "Arsenal", "Chelsea")
    assert pred.is_demo is True
    bettor.predict_proba.assert_not_called()


def test_predict_fixture_returns_real_probs_when_match_found() -> None:
    bettor = MagicMock()
    bettor.predict_proba.return_value = [[0.55, 0.20, 0.25, 0.62, 0.38]]
    loader = MagicMock()
    loader.extract_fixtures_data.return_value = (
        pd.DataFrame({"home_team": ["Arsenal"], "away_team": ["Chelsea"]}),
        None,
        None,
    )
    models = Models(bettor=bettor, loader=loader, backtest={"trained_at": "x"})

    pred = predict_fixture(models, "Arsenal", "Chelsea")

    assert pred.is_demo is False
    assert pred.p_home == 0.55
    assert pred.p_draw == 0.20
    assert pred.p_away == 0.25
    assert pred.p_over_2_5 == 0.62
    bettor.predict_proba.assert_called_once()


def test_predict_fixture_caches_fixtures_df_across_calls() -> None:
    bettor = MagicMock()
    bettor.predict_proba.return_value = [[0.4, 0.3, 0.3, 0.5, 0.5]]
    loader = MagicMock()
    loader.extract_fixtures_data.return_value = (
        pd.DataFrame({"home_team": ["Arsenal"], "away_team": ["Chelsea"]}),
        None,
        None,
    )
    models = Models(bettor=bettor, loader=loader, backtest={"trained_at": "x"})

    predict_fixture(models, "Arsenal", "Chelsea")
    predict_fixture(models, "Arsenal", "Chelsea")

    # extract_fixtures_data should only be hit once thanks to fixtures_df cache
    loader.extract_fixtures_data.assert_called_once()


def test_predict_fixture_demo_when_predict_proba_throws() -> None:
    bettor = MagicMock()
    bettor.predict_proba.side_effect = RuntimeError("library exploded")
    loader = MagicMock()
    loader.extract_fixtures_data.return_value = (
        pd.DataFrame({"home_team": ["Arsenal"], "away_team": ["Chelsea"]}),
        None,
        None,
    )
    models = Models(bettor=bettor, loader=loader, backtest={"trained_at": "x"})

    pred = predict_fixture(models, "Arsenal", "Chelsea")
    assert pred.is_demo is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_predictions.py::test_predict_fixture_returns_real_probs_when_match_found -v`
Expected: FAIL — current `predict_fixture` always returns demo.

- [ ] **Step 3: Replace `predict_fixture` body in src/predictions.py**

In `src/predictions.py`, find the existing `predict_fixture` function (the demo-only stub from Task 5) and replace it with:

```python
def predict_fixture(
    models: Models | None,
    home_team: str,
    away_team: str,
) -> MatchPrediction:
    """Get probabilities for an upcoming fixture.

    home_team and away_team are Odds API names. Falls back to demo mode if:
    - models is None or not ready
    - team name isn't in the static Odds API → library map
    - the library has no fixture row matching the team pair
    - the bettor's predict_proba throws

    Y.columns order from training is locked to:
        [home_win, draw, away_win, over_2.5, under_2.5]
    indexed as 0/1/2/3/4 below. If `backtest.json` records a different order
    (Task 8 writes this), this function must change accordingly.
    """
    if not isinstance(models, Models) or not models.is_ready:
        return _demo_prediction(home_team, away_team)

    try:
        from src.team_names import to_library

        home_lib = to_library(home_team)
        away_lib = to_library(away_team)

        if models.fixtures_df is None:
            X_fix, _, _ = models.loader.extract_fixtures_data()
            models.fixtures_df = X_fix

        match = models.fixtures_df[
            (models.fixtures_df["home_team"] == home_lib)
            & (models.fixtures_df["away_team"] == away_lib)
        ]
        if match.empty:
            logger.warning(
                f"No library fixture for {home_team}({home_lib}) vs {away_team}({away_lib})"
            )
            return _demo_prediction(home_team, away_team)

        probs = models.bettor.predict_proba(match.iloc[[0]])[0]

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            p_home=float(probs[0]),
            p_draw=float(probs[1]),
            p_away=float(probs[2]),
            p_over_2_5=float(probs[3]),
            is_demo=False,
        )
    except Exception as e:
        logger.warning(
            f"Real-model inference failed for {home_team} vs {away_team}: {e}"
        )
        return _demo_prediction(home_team, away_team)
```

- [ ] **Step 4: Run all predictions tests**

Run: `uv run pytest tests/test_predictions.py -v`
Expected: all 17 PASSED (5 phase-1 + 6 phase-2 + 6 phase-3).

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check src/predictions.py tests/test_predictions.py && uv run ruff format --check src/predictions.py tests/test_predictions.py`
Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add src/predictions.py tests/test_predictions.py
git commit -m "feat: real-model branch in predict_fixture with team mapping"
```

---

## Task 7: write_artifacts_atomically() in train_model.py

Build the atomic-write helper first (testable) before writing the rest of the training pipeline.

**Files:**
- Create: `tests/test_train_pipeline.py`
- Modify: `scripts/train_model.py`

- [ ] **Step 1: Write tests first**

Create `tests/test_train_pipeline.py`:

```python
"""Tests for training pipeline plumbing.

Only the atomic-write helper is unit-testable — actual training requires
the library + a public dataset and is verified via Task 13's manual smoke test.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from scripts.train_model import write_artifacts_atomically


def test_atomic_write_creates_all_three_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BETTOR_PATH", tmp_path / "epl_bettor.pkl")
    monkeypatch.setattr("scripts.train_model.LOADER_PATH", tmp_path / "epl_loader.pkl")
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")

    write_artifacts_atomically(
        bettor={"fake": "bettor"},
        loader={"fake": "loader"},
        summary={"trained_at": "2026-05-05T00:00:00Z"},
    )

    assert (tmp_path / "epl_bettor.pkl").exists()
    assert (tmp_path / "epl_loader.pkl").exists()
    assert (tmp_path / "backtest.json").exists()
    # Round-trip check
    with (tmp_path / "epl_bettor.pkl").open("rb") as f:
        assert pickle.load(f) == {"fake": "bettor"}


def test_atomic_write_no_partial_state_when_pickle_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BETTOR_PATH", tmp_path / "epl_bettor.pkl")
    monkeypatch.setattr("scripts.train_model.LOADER_PATH", tmp_path / "epl_loader.pkl")
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")

    # An object that throws on pickle.dump — TypeError is what pickle raises for
    # unpicklable objects like local lambdas
    class Unpicklable:
        def __reduce__(self):
            raise TypeError("nope")

    with pytest.raises(TypeError):
        write_artifacts_atomically(
            bettor={"ok": "bettor"},
            loader=Unpicklable(),  # this will fail to pickle
            summary={"trained_at": "2026-05-05T00:00:00Z"},
        )

    # No final artifacts should exist — only orphaned .tmp files should be cleaned up
    assert not (tmp_path / "epl_bettor.pkl").exists()
    assert not (tmp_path / "epl_loader.pkl").exists()
    assert not (tmp_path / "backtest.json").exists()
    # No .tmp files left behind either
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_overwrites_existing(tmp_path, monkeypatch) -> None:
    """A successful re-train should replace previous artifacts."""
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BETTOR_PATH", tmp_path / "epl_bettor.pkl")
    monkeypatch.setattr("scripts.train_model.LOADER_PATH", tmp_path / "epl_loader.pkl")
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")

    # Pre-existing old artifacts
    (tmp_path / "epl_bettor.pkl").write_bytes(b"old bettor")
    (tmp_path / "epl_loader.pkl").write_bytes(b"old loader")
    (tmp_path / "backtest.json").write_text("{}")

    write_artifacts_atomically(
        bettor={"new": "bettor"},
        loader={"new": "loader"},
        summary={"trained_at": "new"},
    )

    with (tmp_path / "epl_bettor.pkl").open("rb") as f:
        assert pickle.load(f) == {"new": "bettor"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_pipeline.py -v`
Expected: ImportError on `write_artifacts_atomically`.

- [ ] **Step 3: Implement write_artifacts_atomically in scripts/train_model.py**

Replace the **entire contents** of `scripts/train_model.py` with the following stub. (Tasks 8 and 9 fill in `run_backtest` and `main`.)

```python
"""Train the EPL multi-output ClassifierBettor and save it to disk.

Single command produces three artifacts in models/:
    epl_bettor.pkl    — fitted ClassifierBettor (handles 5 markets)
    epl_loader.pkl    — primed SoccerDataLoader (needed at inference time
                        for extract_fixtures_data())
    backtest.json     — per-market backtest summary

Usage:
    uv run python scripts/train_model.py

Run time: typically 10-20 minutes on a laptop. Existing artifacts are only
overwritten if the entire run succeeds (atomic write).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
BETTOR_PATH = MODELS_DIR / "epl_bettor.pkl"
LOADER_PATH = MODELS_DIR / "epl_loader.pkl"
BACKTEST_PATH = MODELS_DIR / "backtest.json"


def write_artifacts_atomically(
    bettor: Any,
    loader: Any,
    summary: dict,
) -> None:
    """Write all three artifacts to .tmp paths, then os.replace to final names.

    Either all three end up at their final paths, or none do. Cleans up .tmp
    files on any failure.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bettor_tmp = BETTOR_PATH.with_suffix(BETTOR_PATH.suffix + ".tmp")
    loader_tmp = LOADER_PATH.with_suffix(LOADER_PATH.suffix + ".tmp")
    backtest_tmp = BACKTEST_PATH.with_suffix(BACKTEST_PATH.suffix + ".tmp")
    tmps = [bettor_tmp, loader_tmp, backtest_tmp]

    try:
        with bettor_tmp.open("wb") as f:
            pickle.dump(bettor, f)
        with loader_tmp.open("wb") as f:
            pickle.dump(loader, f)
        with backtest_tmp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # All temp writes succeeded — promote atomically
        os.replace(bettor_tmp, BETTOR_PATH)
        os.replace(loader_tmp, LOADER_PATH)
        os.replace(backtest_tmp, BACKTEST_PATH)
    except BaseException:
        # Clean up any temp files left behind (don't mask the original exception)
        for tmp in tmps:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        raise


def run_backtest(bettor, X, Y, O) -> dict:  # noqa: E741 — capital O matches library convention
    raise NotImplementedError("Implemented in Task 8")


def main() -> None:
    raise NotImplementedError("Implemented in Task 9")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_train_pipeline.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check scripts/train_model.py tests/test_train_pipeline.py && uv run ruff format --check scripts/train_model.py tests/test_train_pipeline.py`
Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_model.py tests/test_train_pipeline.py
git commit -m "feat: atomic-write helper for training artifacts"
```

---

## Task 8: run_backtest() helper in train_model.py

Library's `backtest()` returns a DataFrame with per-market columns. We reduce to a clean per-market dict for `backtest.json`.

**Files:**
- Modify: `scripts/train_model.py`

- [ ] **Step 1: Replace the run_backtest stub**

In `scripts/train_model.py`, replace the `run_backtest` stub (and its imports) with the implementation below. Add the imports at the top of the file (after the existing ones):

```python
import datetime as dt

import pandas as pd
```

Then replace the function:

```python
def run_backtest(bettor, X, Y, O) -> dict:  # noqa: E741 — capital O matches library convention
    """Run library backtest and reduce per-market columns into a JSON-ready summary.

    Library's `backtest()` returns a DataFrame indexed by training-window start date
    with per-market columns like:
        Yield percentage per bet (home_win__full_time_goals)
        Number of bets (home_win__full_time_goals)
        Precision per bet (home_win__full_time_goals)

    We aggregate those across rows to produce one summary block per market.
    """
    from sportsbet.evaluation import backtest as library_backtest

    bt_df = library_backtest(bettor, X, Y, O)

    markets = ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]
    market_summary: dict[str, dict[str, float]] = {}

    for m in markets:
        col_market = f"{m}__full_time_goals"
        n_bets_col = f"Number of bets ({col_market})"
        win_rate_col = f"Precision per bet ({col_market})"
        yield_col = f"Yield percentage per bet ({col_market})"

        if n_bets_col not in bt_df.columns:
            # Library may name slightly differently across versions; record a marker
            market_summary[m] = {"n_bets": 0, "win_rate": 0.0, "yield_pct": 0.0}
            continue

        total_bets = int(bt_df[n_bets_col].sum())
        # Average win rate weighted by number of bets per row
        if total_bets > 0:
            weighted_win = (bt_df[win_rate_col] * bt_df[n_bets_col]).sum() / total_bets
            weighted_yield = (bt_df[yield_col] * bt_df[n_bets_col]).sum() / total_bets
        else:
            weighted_win = 0.0
            weighted_yield = 0.0

        market_summary[m] = {
            "n_bets": total_bets,
            "win_rate": float(weighted_win),
            "yield_pct": float(weighted_yield),
        }

    return {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_seasons": list(range(2018, 2026)),
        "library_version": "0.12.1",
        "n_training_matches": int(len(X)),
        "y_columns_order": list(Y.columns),  # locked for inference index mapping
        "markets": market_summary,
    }
```

- [ ] **Step 2: Sanity-check the file imports cleanly**

Run: `uv run python -c "from scripts.train_model import run_backtest; print('ok')"`
Expected: `ok` (no import errors).

- [ ] **Step 3: Run static checks**

Run: `uv run ruff check scripts/train_model.py && uv run ruff format --check scripts/train_model.py`
Expected: both pass.

No new tests for this — exercised by Task 13's manual smoke run.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_model.py
git commit -m "feat: run_backtest helper reducing library output to per-market summary"
```

---

## Task 9: main() orchestration in train_model.py

Final piece: load data, build pipeline, fit, backtest, persist.

**Files:**
- Modify: `scripts/train_model.py`

- [ ] **Step 1: Replace the main() stub**

In `scripts/train_model.py`, replace the `main()` stub with:

```python
def main() -> None:
    """Train the multi-output bettor end-to-end and persist artifacts."""
    print("⚽ Football Edge — training EPL multi-output bettor")
    print("=" * 60)

    # Library imports here (not at top) so import errors are reported with context
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    from sportsbet.datasets import SoccerDataLoader
    from sportsbet.evaluation import ClassifierBettor

    # 1. Load historical EPL data
    print("\n[1/4] Loading historical data (2018-2025) — this hits the network...")
    loader = SoccerDataLoader(
        param_grid={
            "league": ["England"],
            "year": list(range(2018, 2026)),
            "division": [1],
        }
    )
    X, Y, O = loader.extract_train_data(
        odds_type="market_average",
        drop_na_thres=1.0,
    )
    print(f"  Loaded {len(X)} matches.")
    print(f"  Y.columns order: {list(Y.columns)}")

    # 2. Build the calibrated multi-output pipeline
    print("\n[2/4] Building pipeline...")
    pipeline = make_pipeline(
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                ["league", "home_team", "away_team"],
            ),
            remainder="passthrough",
        ),
        SimpleImputer(),
        MultiOutputClassifier(
            CalibratedClassifierCV(
                GradientBoostingClassifier(random_state=0),
                method="isotonic",
                cv=3,
            )
        ),
    )
    bettor = ClassifierBettor(classifier=pipeline)

    # 3. Fit + backtest
    print("\n[3/4] Fitting bettor and running backtest (this is the slow bit, ~10-15 min)...")
    bettor.fit(X, Y, O)
    summary = run_backtest(bettor, X, Y, O)

    # 4. Atomic persist
    print("\n[4/4] Persisting artifacts...")
    write_artifacts_atomically(bettor, loader, summary)

    print("\n✅ Done. Artifacts written to:")
    print(f"     {BETTOR_PATH}")
    print(f"     {LOADER_PATH}")
    print(f"     {BACKTEST_PATH}")
    print("\nPer-market backtest summary:")
    for market, stats in summary["markets"].items():
        print(
            f"  {market:<12} {stats['n_bets']:>4} bets  "
            f"win-rate {stats['win_rate']*100:>5.1f}%  "
            f"yield {stats['yield_pct']:+.2f}%"
        )
    print()
```

- [ ] **Step 2: Sanity-check the file imports cleanly**

Run: `uv run python -c "from scripts.train_model import main; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Run static checks**

Run: `uv run ruff check scripts/train_model.py && uv run ruff format --check scripts/train_model.py`
Expected: both pass.

- [ ] **Step 4: Run all tests (regression check)**

Run: `uv run pytest -q`
Expected: 31 PASSED total — 6 (value) + 17 (predictions: 5 phase-1 + 6 phase-2 + 6 phase-3) + 5 (team_names) + 3 (train_pipeline).

If any tests fail, stop and fix before committing.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_model.py
git commit -m "feat: main() orchestration — load, fit, backtest, persist"
```

---

## Task 10: Update app.py — Models loading

Switch `app.py` from the legacy `load_model()` / `model_exists()` API to the new `load_models()` / `Models.is_ready` API. After this commit we can also remove the deprecation shims.

**Files:**
- Modify: `app.py`
- Modify: `src/predictions.py` (remove deprecation shims)

- [ ] **Step 1: Update app.py imports and demo-mode check**

In `app.py`, replace the existing `from src.predictions import load_model, model_exists, predict_fixture` line with:

```python
from src.predictions import Models, load_models, predict_fixture
```

Then find the demo-mode banner block (currently checking `if not model_exists():`) and replace with:

```python
@st.cache_resource(show_spinner="Loading model…")
def _load_models_cached() -> Models:
    return load_models()


_models = _load_models_cached()

# ───── Demo mode banner ─────
if not _models.is_ready:
    st.warning(
        "🟡 **Demo mode** — no trained model found. The probabilities shown are "
        "synthetic and **not predictive**. Wire up `scripts/train_model.py` and "
        "run `uv run python scripts/train_model.py` to use a real model."
    )
```

- [ ] **Step 2: Update _build_rows to use the new Models container**

In `app.py`, find `_build_rows()`. Replace its first line:

```python
def _build_rows(
    fixtures: list[dict],
    threshold: float,
    kelly_mult: float,
) -> list[dict]:
    """Cross-reference predictions with bookmaker odds, return display rows."""
    model = load_model()  # None in demo mode
```

with:

```python
def _build_rows(
    fixtures: list[dict],
    threshold: float,
    kelly_mult: float,
    models: Models,
) -> list[dict]:
    """Cross-reference predictions with bookmaker odds, return display rows."""
```

Also update the `predict_fixture` call inside the loop from `predict_fixture(model, home_team, away_team)` to `predict_fixture(models, home_team, away_team)`.

Then update the call site in `render_fixtures_tab()`:

```python
rows = _build_rows(fixtures, value_threshold, kelly_multiplier, _models)
```

- [ ] **Step 3: Remove deprecation shims from predictions.py**

In `src/predictions.py`, delete the entire `# ─── Compatibility shims ───` block at the end:

```python
# ─── Compatibility shims (removed in Task 10 alongside app.py changes) ───

def model_exists() -> bool:
    ...

def load_model() -> Any | None:
    ...
```

- [ ] **Step 4: Smoke import**

Run: `uv run python -c "import app; print('ok')"`
Expected: `ok` (or a Streamlit-related warning, which is fine — we just want the imports to resolve).

- [ ] **Step 5: Run all tests**

Run: `uv run pytest -q`
Expected: all PASSED.

- [ ] **Step 6: Run static checks**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add app.py src/predictions.py
git commit -m "feat: switch app.py to Models container, remove legacy shims"
```

---

## Task 11: Add totals rows to app.py _build_rows()

Each fixture currently produces 3 rows (Home/Draw/Away). Now it also produces 2 totals rows (Over 2.5 / Under 2.5) when totals odds are available.

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update _build_rows to emit totals rows**

In `app.py`, find the `outcomes = [...]` block inside `_build_rows()`:

```python
        outcomes = [
            ("Home", home_team, pred.p_home, home_odds, fair[0]),
            ("Draw", "Draw", pred.p_draw, draw_odds, fair[1]),
            ("Away", away_team, pred.p_away, away_odds, fair[2]),
        ]

        for label, outcome_name, model_prob, odds_tuple, fair_prob in outcomes:
            decimal_odds, book = odds_tuple
            assessment = assess_value(...)
            rows.append({...})
```

Replace it with:

```python
        outcomes = [
            ("Home", home_team, pred.p_home, home_odds, fair[0], "h2h"),
            ("Draw", "Draw", pred.p_draw, draw_odds, fair[1], "h2h"),
            ("Away", away_team, pred.p_away, away_odds, fair[2], "h2h"),
        ]

        # Totals rows — only added if both Over and Under odds are available
        over_odds = best_odds_for_outcome(fixture, "totals", "Over")
        under_odds = best_odds_for_outcome(fixture, "totals", "Under")
        if over_odds and under_odds and pred.p_over_2_5 is not None:
            implied_totals = [1 / over_odds[0], 1 / under_odds[0]]
            fair_totals = remove_bookmaker_margin(implied_totals)
            outcomes.append(
                ("Over 2.5", "Over 2.5", pred.p_over_2_5, over_odds, fair_totals[0], "totals")
            )
            outcomes.append(
                ("Under 2.5", "Under 2.5", 1 - pred.p_over_2_5, under_odds, fair_totals[1], "totals")
            )

        for label, outcome_name, model_prob, odds_tuple, fair_prob, market in outcomes:
            decimal_odds, book = odds_tuple
            assessment = assess_value(
                model_prob=model_prob,
                decimal_odds=decimal_odds,
                bookmaker=book,
                fair_implied_prob=fair_prob,
                value_threshold=threshold,
                kelly_multiplier=kelly_mult,
            )
            rows.append(
                {
                    "fixture_id": fixture.get("id", f"{home_team}-{away_team}"),
                    "kickoff": fixture.get("commence_time", ""),
                    "home_team": home_team,
                    "away_team": away_team,
                    "market": market,  # was hardcoded "h2h" before
                    "outcome": label,
                    "outcome_label": outcome_name,
                    "Match": f"{home_team} vs {away_team}",
                    "Kickoff": (fixture.get("commence_time", "")[:16] or "").replace("T", " "),
                    "Bet": f"{label}: {outcome_name}" if market == "h2h" else label,
                    "Model %": f"{model_prob * 100:.1f}%",
                    "Fair %": f"{fair_prob * 100:.1f}%",
                    "Best odds": f"{decimal_odds:.2f}",
                    "Bookmaker": book,
                    "Edge": f"{assessment.value_pct * 100:+.1f}%",
                    "Kelly stake": f"{assessment.kelly_stake_fraction * 100:.2f}%",
                    "_value_pct": assessment.value_pct,
                    "_model_prob": model_prob,
                    "_decimal_odds": decimal_odds,
                    "_bookmaker": book,
                    "_kelly_fraction": assessment.kelly_stake_fraction,
                    "_is_value": assessment.is_value_bet,
                }
            )
```

Note: `best_odds_for_outcome` already supports the "totals" market via the existing signature; the outcome name "Over"/"Under" is what The Odds API uses. The function tolerates the `point` filter being omitted because The Odds API typically returns 2.5 as the default.

- [ ] **Step 2: Smoke import**

Run: `uv run python -c "import app; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Run all tests**

Run: `uv run pytest -q`
Expected: all PASSED.

- [ ] **Step 4: Run static checks**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: emit Over/Under 2.5 totals rows alongside h2h in dashboard"
```

---

## Task 12: Add render_model_info() expander to app.py sidebar

Read `models.backtest` and render per-market summary. Stale-model warning if older than 90 days.

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add render_model_info function**

In `app.py`, just below the `_load_models_cached()` definition and `_models = _load_models_cached()` line, add:

```python
def render_model_info(models: Models) -> None:
    """Render the 'Model info' sidebar expander when models are ready."""
    import datetime as dt

    if not models.is_ready or not models.backtest:
        return

    bt = models.backtest
    trained_at_str = bt.get("trained_at", "")
    age_days_str = ""
    stale_marker = ""
    try:
        trained_at = dt.datetime.fromisoformat(trained_at_str.replace("Z", "+00:00"))
        age_days = (dt.datetime.now(dt.timezone.utc) - trained_at).days
        age_days_str = f" ({age_days} days ago)"
        if age_days > 90:
            stale_marker = "⚠️ "
    except (ValueError, AttributeError):
        pass

    n_matches = bt.get("n_training_matches", "?")
    seasons = bt.get("training_seasons", [])
    seasons_str = f"{len(seasons)} seasons" if seasons else "unknown seasons"

    with st.sidebar.expander("📊 Model info", expanded=False):
        st.markdown(
            f"**Trained:** {stale_marker}{trained_at_str[:10]}{age_days_str}  \n"
            f"**Data:** {seasons_str}, {n_matches} matches"
        )
        st.markdown("**Per-market backtest results:**")
        markets = bt.get("markets", {})
        for market_key in ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]:
            stats = markets.get(market_key, {})
            n = stats.get("n_bets", 0)
            wr = stats.get("win_rate", 0.0) * 100
            yp = stats.get("yield_pct", 0.0)
            label = market_key.replace("_", " ").replace("2.5", "2.5").title()
            st.markdown(
                f"- **{label}**: {n} bets, {wr:.0f}% win rate, {yp:+.1f}% yield"
            )
        if stale_marker:
            st.caption(f"Model is over 90 days old — consider retraining.")
```

- [ ] **Step 2: Call render_model_info in the sidebar**

Find the existing sidebar block in `app.py`:

```python
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    ...
    st.caption("[BeGambleAware](https://www.begambleaware.org)")
```

After the `st.caption("[BeGambleAware]...")` line, add (still within the `with st.sidebar:` block):

```python
    st.divider()

# Model info expander (renders only when models loaded)
render_model_info(_models)
```

- [ ] **Step 3: Smoke import**

Run: `uv run python -c "import app; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -q`
Expected: all PASSED.

- [ ] **Step 5: Run static checks**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: Model info sidebar expander with per-market backtest summary"
```

---

## Task 13: End-to-end manual smoke test

This task is a **manual checklist**, not code. Capture findings (what worked, what broke, any team-name fixes needed) and commit a short notes file at the end.

**Files:**
- Optionally create: `docs/superpowers/notes/2026-05-05-smoke-test-results.md`

- [ ] **Step 1: Verify demo mode works (no models present)**

```bash
ls models/                      # should be empty or not exist
uv run streamlit run app.py    # opens browser at localhost:8501
```

Expected: yellow "Demo mode" banner shows, dashboard loads, ~20 value bets visible (h2h only, since pred.p_over_2_5 is set in demo but totals odds may produce additional rows). Stop the server with Ctrl+C.

- [ ] **Step 2: Run training**

```bash
uv run python scripts/train_model.py
```

Expected: completes within 10-20 minutes. Final stdout shows the per-market backtest summary block. Three files now exist in `models/`:

```bash
ls models/
# epl_bettor.pkl
# epl_loader.pkl
# backtest.json
```

If training fails, capture the error. Most likely culprits:
- Network: library can't reach `raw.githubusercontent.com/georgedouzas/sports-betting/...` — retry later
- Library version drift: column-naming in `run_backtest()` doesn't match — adjust the column-name strings to match what `library_backtest()` actually returns, then re-run
- sklearn pipeline error during `fit()` — usually due to unexpected NaNs; raise `drop_na_thres` to 1.0 (already set) or check the X DataFrame's dtypes

- [ ] **Step 3: Inspect backtest.json**

```bash
cat models/backtest.json
```

Verify shape matches the spec — should contain `trained_at`, `training_seasons`, `library_version`, `n_training_matches`, `y_columns_order`, `markets` keys. The `y_columns_order` MUST be `["output__home_win__full_time_goals", "output__draw__full_time_goals", "output__away_win__full_time_goals", "output__over_2.5__full_time_goals", "output__under_2.5__full_time_goals"]` (or equivalent order). If the order is different, the `predict_fixture()` index mapping is wrong — open `src/predictions.py` and re-order the `probs[0..3]` indices to match.

- [ ] **Step 4: Restart Streamlit and verify real predictions**

```bash
uv run streamlit run app.py
```

Expected:
- No demo banner
- Sidebar has "📊 Model info" expander showing per-market backtest stats
- Value bets table includes both h2h and "Over 2.5"/"Under 2.5" rows
- Edges look saner than demo mode (typically 1-7%, not 20-30%)

- [ ] **Step 5: Spot-check 5 fixtures**

For 5 random fixtures, eyeball the row:
- Probabilities in 0-1 range, not all equal, not all 99/0/0
- Match doesn't look obviously wrong (e.g., underdog isn't predicted at 90%)
- For a few rows, sanity-check edge against Betfair Exchange or Pinnacle — model should be in same ballpark, not orders of magnitude off

If a fixture shows `Demo mode` flag (you can detect this by checking the `_is_demo` field in the dataframe — temporarily add a debug column if needed), the most likely cause is a team name not in `ODDS_API_TO_LIBRARY`. Look at recent terminal logs — the warning will say which name is unmapped. Add it to `src/team_names.py`, restart the server.

- [ ] **Step 6: Test atomic-state guard**

```bash
rm models/epl_loader.pkl
```

Refresh the browser. Expected: dashboard goes back to demo mode (loader gone → `is_ready=False`). Re-run training to restore.

- [ ] **Step 7: Test pickle integrity**

```bash
uv run python -c "import pickle; pickle.load(open('models/epl_bettor.pkl', 'rb')); print('bettor ok')"
uv run python -c "import pickle; pickle.load(open('models/epl_loader.pkl', 'rb')); print('loader ok')"
```

Expected: both print "ok".

- [ ] **Step 8: Capture results in a notes file (optional but recommended)**

Create `docs/superpowers/notes/2026-05-05-smoke-test-results.md` with brief notes:

```markdown
# Real Model Wiring — Smoke Test Results

**Date:** 2026-05-05
**Operator:** Femi
**Outcome:** ✅ pass / ❌ fail

## Findings

### Training run
- Wall-clock time: __ minutes
- n_training_matches: __

### Backtest summary (from models/backtest.json)
- home_win: __ bets, __% win rate, __% yield
- draw: ...
- away_win: ...
- over_2.5: ...
- under_2.5: ...

### Issues encountered
- (e.g., team-name fixes, library-API tweaks, etc.)

### Spot-check examples
- Fixture 1: ...
- Fixture 2: ...
```

- [ ] **Step 9: Final commit**

```bash
git add docs/superpowers/notes/2026-05-05-smoke-test-results.md  # if you wrote one
git commit -m "docs: smoke-test results for real model wiring"
```

---

## Self-Review Checklist (run by the implementer at end)

- [ ] All 13 tasks completed and committed
- [ ] `uv run pytest -q` — all tests pass
- [ ] `uv run ruff check .` — passes
- [ ] `uv run ruff format --check .` — passes
- [ ] Streamlit dashboard with artifacts present: shows live predictions, no demo banner, Model info expander rendered
- [ ] Streamlit dashboard with artifacts absent or partial: shows demo banner, synthetic predictions, no crash
- [ ] At least 5 fixtures spot-checked with plausible probabilities
- [ ] `Y.columns` order matches the index mapping in `predict_fixture()`
- [ ] Every current EPL team has a `ODDS_API_TO_LIBRARY` entry
