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
| Implementation style | Library-native, two separate fits | Sticks to documented patterns; debuggable |
| Testing | pytest unit tests + manual smoke checklist | Plumbing testable in pytest; model quality only verifiable via backtest + paper trading |

## Architecture

```
                    [ scripts/train_model.py ]
                              │  (manual: uv run python scripts/train_model.py)
                              ▼
        ┌─────────────────────────────────────────┐
        │ SoccerDataLoader (England D1, 2018-25)  │
        │   ↓                                     │
        │ Two parallel training pipelines:        │
        │   • Match-result (3-class, calibrated)  │
        │   • Totals 2.5  (2-class, calibrated)   │
        │   ↓                                     │
        │ backtest() each → persist results       │
        └─────────────────────────────────────────┘
                              │
                              ▼
                  ┌────────── outputs ──────────┐
                  │ models/epl_match_result.pkl │
                  │ models/epl_totals.pkl       │
                  │ models/backtest.json        │
                  └─────────────────────────────┘
                              │
                              │ (loaded at app start, cached)
                              ▼
                  [ src/predictions.py ]
                              │
                              │ predict_fixture(home, away)
                              ▼
              MatchPrediction(p_home, p_draw, p_away, p_over_2_5)
                              │
                              ▼
                       [ app.py ] → Streamlit dashboard
                              ├── Value-bets table (h2h + totals rows)
                              ├── "Model info" expander (backtest JSON)
                              └── Demo-mode banner only when models missing
```

## Files

### Added
- `models/epl_match_result.pkl` — pickled fitted `ClassifierBettor` for 1X2 outcomes
- `models/epl_totals.pkl` — pickled fitted `ClassifierBettor` for Over/Under 2.5 goals
- `models/backtest.json` — single JSON with both backtest summaries and metadata
- `tests/test_value.py` — pure-function tests for value/Kelly math
- `tests/test_predictions.py` — plumbing tests for `predictions.py`
- `tests/test_train_pipeline.py` — atomic-write smoke test

### Modified
- `scripts/train_model.py` — full training pipeline (~60 lines, replaces stub)
- `src/predictions.py` — new `Models` dataclass, `load_models()`, real branch in `predict_fixture()` (~25 lines added/changed)
- `app.py` — totals rows in `_build_rows()`, "Model info" expander (~25 lines)
- `pyproject.toml` — add `pytest` to `[dependency-groups].dev`

### Unchanged
- `src/odds.py`, `src/value.py`, `src/database.py`, `src/auth.py` — no changes needed
- Supabase schema — already supports `market='totals'`

## Components

### `scripts/train_model.py`

Single entry point. Runs the full pipeline:

```python
def main() -> None:
    """Train both models, backtest both, persist artifacts atomically."""
    # 1. Load historical EPL data
    loader = SoccerDataLoader(param_grid={
        "league": ["England"],
        "year": list(range(2018, 2026)),
        "division": [1],
    })
    X, Y_match, odds_match = loader.extract_train_data(
        odds_type="market_average", drop_na_thres=0.8
    )
    # Totals extraction — exact API verified at impl time
    X_t, Y_totals, odds_totals = loader.extract_train_data(
        odds_type="market_average", drop_na_thres=0.8, market="totals_2.5"
    )

    # 2. Train both bettors
    match_bettor = train_calibrated_bettor(X, Y_match, odds_match, n_classes=3)
    totals_bettor = train_calibrated_bettor(X_t, Y_totals, odds_totals, n_classes=2)

    # 3. Backtest both
    match_summary = run_backtest(match_bettor, X, Y_match, odds_match)
    totals_summary = run_backtest(totals_bettor, X_t, Y_totals, odds_totals)

    # 4. Atomic persist
    write_artifacts_atomically(
        match_bettor, totals_bettor,
        match_summary, totals_summary,
    )
```

Helper functions:
- `train_calibrated_bettor(X, Y, odds, n_classes) -> ClassifierBettor` — wraps `GradientBoostingClassifier` in `CalibratedClassifierCV(method='isotonic', cv=3)`, then in `ClassifierBettor`, then fits.
- `run_backtest(bettor, X, Y, odds) -> dict` — runs library `backtest()` with 5-fold CV, value-threshold 0.05, kelly-multiplier 0.25. Extracts `n_simulated_bets`, `win_rate`, `roi_pct`, `log_loss`. Returns clean dict ready for JSON.
- `write_artifacts_atomically(...)` — writes all three artifacts to `.tmp` paths, then `os.replace()` them to final names. Raises if any single write fails, and cleans up `.tmp` files on failure.

### `src/predictions.py`

New `Models` container:

```python
@dataclass
class Models:
    match_result: Any | None     # fitted ClassifierBettor or None
    totals: Any | None           # fitted ClassifierBettor or None
    backtest: dict | None        # loaded backtest.json contents

    @property
    def is_ready(self) -> bool:
        return (
            self.match_result is not None
            and self.totals is not None
            and self.backtest is not None
        )

def load_models() -> Models:
    """Load all three artifacts from disk. Any missing/corrupt → None for that field."""
    ...

def predict_fixture(models: Models, home_team: str, away_team: str) -> MatchPrediction:
    if not models.is_ready:
        return _demo_prediction(home_team, away_team)
    try:
        features = _build_inference_row(home_team, away_team)
        match_probs = models.match_result.predict_proba(features)[0]
        totals_probs = models.totals.predict_proba(features)[0]
        return MatchPrediction(
            home_team=home_team, away_team=away_team,
            p_home=match_probs[0], p_draw=match_probs[1], p_away=match_probs[2],
            p_over_2_5=totals_probs[1],
            is_demo=False,
        )
    except Exception as e:
        # Per-fixture fallback — one bad fixture doesn't break the whole dashboard
        logger.warning(f"Real-model inference failed for {home_team} vs {away_team}: {e}")
        return _demo_prediction(home_team, away_team)
```

`_build_inference_row()` is the **most uncertain piece** — the library expects inference inputs in the same shape as training inputs. Implementation needs to verify against the library's actual API (`extract_predict_data()` is the likely entry point) and may need a small adapter helper.

Existing functions retired: `model_exists()` → `Models.is_ready`. `load_model()` → `load_models()`.

### `app.py`

**Change 1 — `_build_rows()`** adds totals outcomes alongside h2h. Each fixture now produces 5 rows (Home / Draw / Away / Over 2.5 / Under 2.5) instead of 3. Totals outcomes use `pred.p_over_2_5` and `1 - pred.p_over_2_5`, look up best totals odds via `best_odds_for_outcome(fixture, "totals", "Over", point=2.5)`. Reuses existing `assess_value()` unchanged.

If totals odds are missing for a fixture, skip the totals rows but keep the h2h rows. The h2h-odds-missing guard (already exists) skips the whole fixture.

**Change 2 — new `render_model_info()`** sidebar expander. Reads `models.backtest` and renders:

```
📊 Model info
─────────────
Trained: 2026-05-05 (8 seasons of EPL)

Match-result model
  • 287 simulated bets in backtest
  • 51% win rate
  • +2.4% ROI
  • Log-loss: 0.96

Totals (over/under 2.5) model
  • 312 simulated bets in backtest
  • 53% win rate
  • +1.8% ROI
  • Log-loss: 0.62
```

Renders only if `models.is_ready`; otherwise the existing demo-mode banner shows. If `trained_at` is older than 90 days, prepend ⚠️ to the date and add a tooltip "Model is X days old; consider retraining."

## Data shapes

### `backtest.json`

```json
{
  "trained_at": "2026-05-05T22:00:00Z",
  "training_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  "library_version": "0.12.1",
  "match_result": {
    "n_simulated_bets": 287,
    "win_rate": 0.51,
    "roi_pct": 2.4,
    "log_loss": 0.96
  },
  "totals": {
    "n_simulated_bets": 312,
    "win_rate": 0.53,
    "roi_pct": 1.8,
    "log_loss": 0.62
  }
}
```

### `MatchPrediction` (existing dataclass, unchanged)

Already has all required fields. `p_over_2_5` was previously optional and unset; now always populated when `is_demo=False`.

## Error handling

| Error | Surfaces in | Handling |
|---|---|---|
| Either model file missing or unpicklable | `load_models()` | `Models.is_ready=False` → demo-mode banner, synthetic predictions, no crash |
| `backtest.json` missing/corrupt | `load_models()` | Same as above — treated as "not trained yet" |
| Library throws during training | `train_model.py` | Print clear error, exit code 1. Existing `.pkl` files (if any) untouched — atomic write protects coherent state |
| Library throws during `predict_proba` for one fixture | `predict_fixture()` | Catch, log warning, demo-mode fallback for that fixture only. Other fixtures continue |
| Unknown team (newly promoted) | `_build_inference_row()` | Rely on library's default team-identity handling. If it errors, falls through to per-fixture demo fallback above |
| Fixture missing totals odds | `_build_rows()` | Skip totals rows, keep h2h rows |
| Fixture missing h2h odds | `_build_rows()` | Skip whole fixture (existing behaviour, unchanged) |
| Stale model (>90 days old) | Dashboard | ⚠️ icon next to trained_at date in Model info expander. No forced retrain |
| Pickle version skew (Python/sklearn upgrade) | `load_models()` | Same as "unpicklable" — demo mode + log message recommending re-train |

## Atomic write detail

Training writes to `.tmp` paths first, then `os.replace()` them into final names only after **all three** artifacts succeed. Guarantees:
- A crash mid-training never leaves match-result paired with stale totals (or vice versa)
- Dashboard always loads coherent artifacts

## Testing strategy

### Unit tests (`pytest`)

| File | Tests | Cost |
|---|---|---|
| `tests/test_value.py` | margin removal, Kelly with no edge, Kelly textbook case, `assess_value` flag T/F | ~15 min to write, ~0.5s to run |
| `tests/test_predictions.py` | demo prob normalisation, demo determinism, `load_models` returns unready when files missing, `predict_fixture` falls back to demo when models unready | ~20 min, ~1s |
| `tests/test_train_pipeline.py` | atomic-write guarantee (mock pickle to fail mid-write, assert no `.pkl` files in target dir) | ~15 min, ~1s |

Pytest doesn't test "training runs end-to-end" — that's a 10-20 min hit on a public dataset, manual only.

### Manual smoke checklist (lives in this spec, executed after implementation)

1. With `models/` empty: `streamlit run app.py` → demo banner shows, predictions are synthetic
2. Run `uv run python scripts/train_model.py` → completes in 10-20 min, three artifact files appear
3. Refresh dashboard → demo banner gone, "Model info" expander shows backtest summary
4. Spot-check at least 5 random fixtures' probabilities — sanity check (not all 33%/33%/33%, not 99% home)
5. Compare a couple of edges to Betfair Exchange — same ballpark, not orders of magnitude off
6. Delete `models/epl_totals.pkl` only → dashboard back to demo mode (proves atomic-state guard)
7. `uv run python -c "import pickle; pickle.load(open('models/epl_match_result.pkl','rb'))"` runs without error

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

1. **`_build_inference_row()` shape mismatch.** The biggest unknown — library expects inference inputs in a specific format. Mitigation: verify against `sports-betting` library API at impl time; may need a small adapter helper.
2. **Library version drift.** `sports-betting` is at v0.12.1 (per uv.lock); the README sketch was from an earlier version. Mitigation: pin version in `pyproject.toml` once implementation works; document `library_version` in `backtest.json`.
3. **football-data.co.uk availability.** The library's data source is a public CSV mirror. If it's down, training fails. Mitigation: existing artifacts stay on disk; user retries later.
4. **Calibration may not save us if features are weak.** Calibration only fixes probability shape, not feature signal. If the library's default features are weak, the model will still be a poor predictor — calibration just makes it honestly poor. This is an inherent limit of the data; only paper-trading reveals it.

## Acceptance criteria

The work is done when:

- [ ] `uv run python scripts/train_model.py` produces all three artifact files in `models/` within ~20 minutes
- [ ] All `pytest` tests pass
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] Streamlit dashboard with artifacts present: shows live predictions, no demo banner, Model info expander rendered
- [ ] Streamlit dashboard with artifacts absent or partial: shows demo banner, synthetic predictions, no crash
- [ ] Manual smoke checklist (above) — all 7 steps pass
- [ ] Spot check: at least 5 fixtures have plausible (non-degenerate) probabilities
