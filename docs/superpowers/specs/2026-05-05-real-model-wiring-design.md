# Real Model Wiring — Design Spec

**Date:** 2026-05-05
**Author:** Oluwafemi Adadayo
**Status:** Approved (pending user review of this written spec)
**Replaces:** Demo-mode synthetic predictions in `src/predictions.py`

## Goal

Replace synthetic match probabilities with real, calibrated predictions from a trained `sports-betting` model. Cover both the match-result market (1X2: Home / Draw / Away) and the goals-totals market (Over / Under 2.5). Surface backtest results in the dashboard so the user can judge whether to trust the probabilities.

## Decisions made during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Backtest visibility | Train + persist + show in dashboard | User needs to compare live performance to backtest baseline during paper-trade window |
| Markets to predict | Match-result AND totals 2.5 | More value-bet surface area; user wants more shots on goal |
| Probability calibration | Calibrate from day one | Kelly stake formula uses probability literally — uncalibrated models silently corrupt every stake |
| Implementation style | Library-native, single multi-output fit | Library returns all 5 markets in one Y; one bettor with `MultiOutputClassifier(CalibratedClassifierCV(...))` is the documented pattern. Revised 2026-05-05 after reading library source. |
| Testing | pytest unit tests + manual smoke checklist | Plumbing testable in pytest; model quality only verifiable via backtest + paper trading |

## Architecture

```
                    [ scripts/train_model.py ]
                              │  (manual: uv run python scripts/train_model.py)
                              ▼
        ┌────────────────────────────────────────────────────┐
        │ SoccerDataLoader(England D1, 2018-25)              │
        │   ↓ extract_train_data(odds_type='market_average') │
        │   X_train, Y_train (5 outputs), O_train            │
        │   ↓                                                │
        │ ClassifierBettor wrapping a sklearn pipeline:      │
        │   OneHot + Impute +                                │
        │   MultiOutputClassifier(                           │
        │     CalibratedClassifierCV(                        │
        │       GradientBoostingClassifier(),                │
        │       method='isotonic', cv=3                      │
        │     )                                              │
        │   )                                                │
        │   ↓ bettor.fit(X, Y, O)                            │
        │   ↓ backtest(bettor, X, Y, O)                      │
        └────────────────────────────────────────────────────┘
                              │
                              ▼
                  ┌────────── outputs ──────────┐
                  │ models/epl_bettor.pkl       │  (the fitted ClassifierBettor)
                  │ models/epl_loader.pkl       │  (the SoccerDataLoader, primed by extract_train_data — required for inference)
                  │ models/backtest.json        │  (per-market backtest summary)
                  └─────────────────────────────┘
                              │
                              │ (loaded at app start, cached)
                              ▼
                  [ src/predictions.py ]
                              │
                              │ predict_fixture(home_odds_api_name, away_odds_api_name)
                              │   1. Normalise team names via static map → library names
                              │   2. loader.extract_fixtures_data() → X_fix DataFrame
                              │   3. Look up matching row by (home_team, away_team)
                              │   4. bettor.predict_proba(matched_row) → [5 probs]
                              ▼
              MatchPrediction(p_home, p_draw, p_away, p_over_2_5)
                              │
                              ▼
                       [ app.py ] → Streamlit dashboard
                              ├── Value-bets table (h2h + totals rows)
                              ├── "Model info" expander (backtest JSON)
                              └── Demo-mode banner only when artifacts missing
```

## Files

### Added
- `models/epl_bettor.pkl` — pickled fitted multi-output `ClassifierBettor`
- `models/epl_loader.pkl` — pickled `SoccerDataLoader` (primed by `extract_train_data`); required so `extract_fixtures_data()` works at inference time
- `models/backtest.json` — per-market backtest summary + metadata
- `src/team_names.py` — static mapping `ODDS_API_TO_LIBRARY` for the 20 EPL teams + helpers `to_library(name)`, `from_library(name)`
- `tests/__init__.py` — empty marker
- `tests/test_value.py` — pure-function tests for value/Kelly math
- `tests/test_predictions.py` — plumbing tests for `predictions.py`
- `tests/test_team_names.py` — round-trip mapping tests
- `tests/test_train_pipeline.py` — atomic-write smoke test

### Modified
- `scripts/train_model.py` — full training pipeline (~70 lines, replaces stub)
- `src/predictions.py` — new `Models` dataclass, `load_models()`, real branch in `predict_fixture()` using `extract_fixtures_data()` + team-name normalisation (~50 lines added/changed)
- `app.py` — totals rows in `_build_rows()`, "Model info" expander (~25 lines)
- `pyproject.toml` — add `pytest` to `[dependency-groups].dev`; pin `sports-betting==0.12.1` in `[project].dependencies`

### Unchanged
- `src/odds.py`, `src/value.py`, `src/database.py`, `src/auth.py` — no changes needed
- Supabase schema — already supports `market='totals'`

## Components

### `scripts/train_model.py`

Single entry point. Runs the full pipeline:

```python
def main() -> None:
    """Train one multi-output bettor, backtest it, persist artifacts atomically."""
    # 1. Load historical EPL data — single call returns Y with all 5 markets
    loader = SoccerDataLoader(param_grid={
        "league": ["England"],
        "year": list(range(2018, 2026)),
        "division": [1],
    })
    X, Y, O = loader.extract_train_data(
        odds_type="market_average", drop_na_thres=1.0
    )

    # 2. Build the calibrated multi-output pipeline
    pipeline = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore"), ["league", "home_team", "away_team"]),
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

    # 3. Fit
    bettor.fit(X, Y, O)

    # 4. Backtest
    backtest_summary = run_backtest(bettor, X, Y, O)

    # 5. Atomic persist (bettor + loader + backtest.json)
    write_artifacts_atomically(bettor, loader, backtest_summary)
```

Helper functions:
- `run_backtest(bettor, X, Y, O) -> dict` — calls library `backtest(bettor, X, Y, O)`. The library returns a DataFrame with per-market yield columns. We extract per-market `{n_bets, win_rate, yield_pct}` plus overall `n_training_matches`. Returns dict ready for JSON.
- `write_artifacts_atomically(bettor, loader, summary) -> None` — writes bettor → `.pkl.tmp`, loader → `.pkl.tmp`, summary → `.json.tmp`. Only on all three writes succeeding does it `os.replace()` each into the final name. On any failure, cleans up `.tmp` files. Raises if anything fails.

### `src/predictions.py`

`Models` container holds bettor + loader + backtest summary, plus a session-cached fixtures DataFrame:

```python
@dataclass
class Models:
    bettor: Any | None              # fitted ClassifierBettor or None
    loader: Any | None              # primed SoccerDataLoader or None
    backtest: dict | None           # loaded backtest.json contents
    fixtures_df: pd.DataFrame | None = None  # cached extract_fixtures_data() output

    @property
    def is_ready(self) -> bool:
        return (
            self.bettor is not None
            and self.loader is not None
            and self.backtest is not None
        )

def load_models() -> Models:
    """Load all three artifacts. Any missing/corrupt → demo mode (is_ready=False)."""
    ...

def predict_fixture(models: Models, home_team: str, away_team: str) -> MatchPrediction:
    """home_team and away_team are Odds API names. Returns demo if not ready or unmatched."""
    if not models.is_ready:
        return _demo_prediction(home_team, away_team)
    try:
        # Normalise Odds API names → library names
        home_lib = to_library(home_team)
        away_lib = to_library(away_team)

        # Lazy-fetch fixtures (cached on Models for the session)
        if models.fixtures_df is None:
            X_fix, _, _ = models.loader.extract_fixtures_data()
            models.fixtures_df = X_fix

        # Look up the row by team identity
        match = models.fixtures_df[
            (models.fixtures_df["home_team"] == home_lib)
            & (models.fixtures_df["away_team"] == away_lib)
        ]
        if match.empty:
            logger.warning(f"No library fixture for {home_team}({home_lib}) vs {away_team}({away_lib})")
            return _demo_prediction(home_team, away_team)

        # Predict — bettor returns shape (1, n_markets) where columns map to Y.columns
        probs = models.bettor.predict_proba(match.iloc[[0]])[0]
        # Y.columns order from training (locked by Task 1 of impl plan):
        #   [home_win, draw, away_win, over_2.5, under_2.5]
        return MatchPrediction(
            home_team=home_team, away_team=away_team,
            p_home=float(probs[0]),
            p_draw=float(probs[1]),
            p_away=float(probs[2]),
            p_over_2_5=float(probs[3]),
            is_demo=False,
        )
    except Exception as e:
        logger.warning(f"Real-model inference failed for {home_team} vs {away_team}: {e}")
        return _demo_prediction(home_team, away_team)
```

**Verified at impl time:** the exact column order in `Y.columns` from `extract_train_data()`. The implementation plan's first task prints that order from a smoke run and locks it as a constant.

Existing functions retired: `model_exists()` → `Models.is_ready`. `load_model()` → `load_models()`.

### `src/team_names.py` — Odds API ↔ library team-name normalisation

Static mapping for the 20 EPL teams. The Odds API uses long forms ("Wolverhampton Wanderers") while the library tends to use short forms ("Wolves"). The exact mapping is verified at impl time by listing teams in both sources after first training run.

```python
ODDS_API_TO_LIBRARY: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    # … remaining 16 teams, finalised during impl ...
}

def to_library(odds_api_name: str) -> str:
    if odds_api_name not in ODDS_API_TO_LIBRARY:
        raise KeyError(
            f"Unmapped Odds API team: {odds_api_name!r}. "
            f"Add to team_names.ODDS_API_TO_LIBRARY."
        )
    return ODDS_API_TO_LIBRARY[odds_api_name]

def from_library(library_name: str) -> str:
    """Reverse lookup."""
    ...
```

Once a year (when relegation/promotion changes the EPL roster), the map needs updating — flagged in the runbook below.

### `app.py`

**Change 1 — `_build_rows()`** adds totals outcomes alongside h2h. Each fixture now produces 5 rows (Home / Draw / Away / Over 2.5 / Under 2.5) instead of 3. Totals outcomes use `pred.p_over_2_5` and `1 - pred.p_over_2_5`, look up best totals odds via `best_odds_for_outcome(fixture, "totals", "Over", point=2.5)`. Reuses existing `assess_value()` unchanged.

If totals odds are missing for a fixture, skip the totals rows but keep the h2h rows. The h2h-odds-missing guard (already exists) skips the whole fixture.

**Change 2 — new `render_model_info()`** sidebar expander. Reads `models.backtest` and renders one block per market (5 markets, single multi-output model underneath):

```
📊 Model info
─────────────
Trained: 2026-05-05 (8 seasons of EPL, 2876 historical matches)

Per-market backtest results
  Home win:    287 bets, 51% win rate, +2.4% yield
  Draw:        102 bets, 28% win rate, -1.1% yield
  Away win:    198 bets, 41% win rate, +0.6% yield
  Over 2.5:    312 bets, 53% win rate, +1.8% yield
  Under 2.5:   289 bets, 47% win rate, -0.4% yield
```

Renders only if `models.is_ready`; otherwise the existing demo-mode banner shows. If `trained_at` is older than 90 days, prepend ⚠️ to the date and add a tooltip "Model is X days old; consider retraining."

## Data shapes

### `backtest.json`

```json
{
  "trained_at": "2026-05-05T22:00:00Z",
  "training_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  "library_version": "0.12.1",
  "n_training_matches": 2876,
  "markets": {
    "home_win":  {"n_bets": 287, "win_rate": 0.51, "yield_pct":  2.4},
    "draw":      {"n_bets": 102, "win_rate": 0.28, "yield_pct": -1.1},
    "away_win":  {"n_bets": 198, "win_rate": 0.41, "yield_pct":  0.6},
    "over_2.5":  {"n_bets": 312, "win_rate": 0.53, "yield_pct":  1.8},
    "under_2.5": {"n_bets": 289, "win_rate": 0.47, "yield_pct": -0.4}
  }
}
```

### `MatchPrediction` (existing dataclass, unchanged)

Already has all required fields. `p_over_2_5` was previously optional and unset; now always populated when `is_demo=False`.

## Error handling

| Error | Surfaces in | Handling |
|---|---|---|
| Bettor or loader pkl missing or unpicklable | `load_models()` | `Models.is_ready=False` → demo-mode banner, synthetic predictions, no crash |
| `backtest.json` missing/corrupt | `load_models()` | Same as above — treated as "not trained yet" |
| Library throws during training | `train_model.py` | Print clear error, exit code 1. Existing artifacts (if any) untouched — atomic write protects coherent state |
| Library throws during `predict_proba` for one fixture | `predict_fixture()` | Catch, log warning, demo-mode fallback for that fixture only. Other fixtures continue |
| Odds API team name not in `ODDS_API_TO_LIBRARY` | `predict_fixture()` via `to_library()` | Caught by outer try/except → demo fallback for that fixture, warning logged with the unmapped name so user can update the map |
| Library has no fixture row matching `(home_lib, away_lib)` | `predict_fixture()` | Demo fallback for that fixture, warning logged. Common when library data is stale relative to The Odds API |
| Fixture missing totals odds | `_build_rows()` | Skip totals rows, keep h2h rows |
| Fixture missing h2h odds | `_build_rows()` | Skip whole fixture (existing behaviour, unchanged) |
| Stale model (>90 days old) | Dashboard | ⚠️ icon next to trained_at date in Model info expander. No forced retrain |
| Pickle version skew (Python/sklearn upgrade) | `load_models()` | Same as "unpicklable" — demo mode + log message recommending re-train |

## Atomic write detail

Training writes to `.tmp` paths first, then `os.replace()` them into final names only after **all three** artifacts (`epl_bettor.pkl`, `epl_loader.pkl`, `backtest.json`) succeed. Guarantees:
- A crash mid-training never leaves a fresh bettor paired with a stale loader (which would silently produce wrong predictions because the loader's column schema wouldn't match the bettor's expected input)
- Dashboard always loads coherent artifacts

## Testing strategy

### Unit tests (`pytest`)

| File | Tests | Cost |
|---|---|---|
| `tests/test_value.py` | margin removal, Kelly with no edge, Kelly textbook case, `assess_value` flag T/F | ~15 min to write, ~0.5s to run |
| `tests/test_predictions.py` | demo prob normalisation, demo determinism, `load_models` returns unready when files missing, `predict_fixture` falls back to demo when models unready | ~20 min, ~1s |
| `tests/test_team_names.py` | every Odds API name maps to a library name; round-trip; unmapped name raises KeyError | ~10 min, <1s |
| `tests/test_train_pipeline.py` | atomic-write guarantee (mock pickle to fail mid-write, assert no final artifact files in target dir) | ~15 min, ~1s |

Pytest doesn't test "training runs end-to-end" — that's a 10-20 min hit on a public dataset, manual only.

### Manual smoke checklist (lives in this spec, executed after implementation)

1. With `models/` empty: `streamlit run app.py` → demo banner shows, predictions are synthetic
2. Run `uv run python scripts/train_model.py` → completes in 10-20 min, three artifact files appear (`epl_bettor.pkl`, `epl_loader.pkl`, `backtest.json`)
3. Refresh dashboard → demo banner gone, "Model info" expander shows backtest summary
4. Spot-check at least 5 random fixtures' probabilities — sanity check (not all 33%/33%/33%, not 99% home)
5. Compare a couple of edges to Betfair Exchange — same ballpark, not orders of magnitude off
6. Delete `models/epl_loader.pkl` only → dashboard back to demo mode (proves atomic-state guard)
7. `uv run python -c "import pickle; pickle.load(open('models/epl_bettor.pkl','rb'))"` runs without error
8. Quick log-grep test: trigger an unmapped team by temporarily removing one entry from `ODDS_API_TO_LIBRARY`; confirm the warning logs the unmapped name and the dashboard falls back to demo for that fixture only

### Static checks

```
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

All three must pass before commit. mypy explicitly out of scope (ROI debatable for a personal Python project).

## Out of scope (YAGNI)

- Multi-classifier comparison / ensemble (Approach 3 from brainstorming)
- Automated retraining schedule
- Calibration plots in the dashboard
- Model versioning beyond a `trained_at` timestamp
- Asian handicap, BTTS, correct score, or any other markets beyond 1X2 + Over/Under 2.5
- Live in-play predictions
- Streamlit UI tests (Playwright/Selenium overhead not justified)

## Risks

1. **Team-name mismatch between Odds API and library.** The biggest known unknown — Odds API and library use different naming conventions for some teams. Mitigation: hardcoded `ODDS_API_TO_LIBRARY` map for the 20 EPL teams, verified against actual data at impl time. Unmapped names raise a clear `KeyError` that's caught and logged so the user knows exactly which name to add.
2. **Y.columns order in library output.** The bettor's `predict_proba` returns probabilities in the order of `Y.columns` from training. If that order changes between library versions, our index-based mapping (`probs[0]=p_home, probs[1]=p_draw, ...`) breaks silently. Mitigation: assert the column order at training time and write it into `backtest.json`; check on load and raise if mismatched.
3. **Library version drift.** `sports-betting` is at v0.12.1 (per uv.lock). Mitigation: pin `sports-betting==0.12.1` in `pyproject.toml`; document `library_version` in `backtest.json`.
4. **football-data.co.uk availability.** The library's data source is a public CSV mirror. If it's down, training or fixture extraction fails. Mitigation: existing artifacts stay on disk; user retries later. Inference also needs `extract_fixtures_data()` which hits the same source — if it fails at app load, demo mode kicks in.
5. **Calibration may not save us if features are weak.** Calibration only fixes probability shape, not feature signal. If the library's default features are weak, the model will still be a poor predictor — calibration just makes it honestly poor. This is an inherent limit of the data; only paper-trading reveals it.
6. **EPL roster changes annually.** When teams get relegated/promoted (each August), the `ODDS_API_TO_LIBRARY` map needs updating. The clear `KeyError` makes this self-diagnosing but it's still a manual yearly chore.

## Acceptance criteria

The work is done when:

- [ ] `uv run python scripts/train_model.py` produces all three artifact files in `models/` within ~20 minutes
- [ ] All `pytest` tests pass
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `Y.columns` order asserted at training time matches the order assumed by `predict_fixture()` (constant in code)
- [ ] Streamlit dashboard with artifacts present: shows live predictions, no demo banner, Model info expander rendered with all 5 markets
- [ ] Streamlit dashboard with artifacts absent or partial: shows demo banner, synthetic predictions, no crash
- [ ] Manual smoke checklist (above) — all 8 steps pass
- [ ] Spot check: at least 5 fixtures have plausible (non-degenerate) probabilities
- [ ] Every current EPL team (per The Odds API) has an entry in `ODDS_API_TO_LIBRARY`
