# Model Improvement v2 — Design Spec

**Date:** 2026-05-06
**Author:** Oluwafemi Adadayo
**Status:** Pending user review
**Replaces:** Nothing (extends `2026-05-05-real-model-wiring-design.md` which shipped the v0 plumbing)

## Goal

The v0 pipeline shipped end-to-end and works mechanically: data ingestion → calibrated multi-output classifier → atomic-write artifacts → dashboard surfaces predictions. The model itself is not profitable — backtest yields are −14.6% on home wins, −6.8% on draws, +0.8% on away wins, −3.7% on Over 2.5, −1.3% on Under 2.5. The cause is feature weakness, not pipeline correctness: the `sports-betting` library's defaults use 2010-era features (recent-goals averages, basic match stats) that bookmakers have long since priced in.

v2 replaces the entire modelling layer with an sklearn-direct pipeline that ingests richer features (xG from Understat plus computable rest/form/strength-of-schedule features), expands scope from EPL-only to all five Big-5 European leagues, and trains five separate per-league models. The dashboard is rebuilt to handle multi-league fixtures, league filtering, and per-league backtest reporting. The classifier shape is held constant from v0 to isolate the feature-improvement effect.

## Decisions made during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Library | Replace `sports-betting` with sklearn-direct pipeline | Library has burned us four times in patches (HTML scraper, date format, `__version__`, missing `Precision per bet` column). 150 lines of replacement code is a one-time cost; further patching is recurring. |
| Data sources | Free public only — Understat scrape + existing football-data.co.uk CSVs | Personal project; paid feeds premature when basic-feature ceiling not yet measured. |
| League scope | Big 5 — EPL, La Liga, Serie A, Bundesliga, Ligue 1 | More training data + larger value-bet surface area. Codebase rewiring (team-name registry, multi-endpoint odds, dashboard) is real but contained. |
| Model architecture | Five separate per-league models | Cleaner per-league diagnostics, easier to drop a league if its model is bad, parallel training keeps wall-clock similar to pooled. |
| Feature scope | xG + computable features (rest, form, congestion, strength-of-schedule, xG-vs-actual delta) | xG is the single biggest known improvement in football modelling. Computable features are essentially free — same downloads, just more processing. Wider scope (referee tendencies, weather, attendance) deferred as second-order. |
| Training window | 2018-2025 (8 seasons) — same as v0 | Apples-to-apples yield comparison with v0; modern football era post-VAR. |
| Module structure | Hybrid — `src/ingest/` directory, single `src/features.py`, training inline in `scripts/train_model.py` | Genuine separation between data ingestion and feature engineering; further decomposition (separate `training.py`, `backtest.py`) deferred until complexity justifies it. |
| Per-league failure (training) | Fail-the-whole-run | Atomic-write invariant requires all artifacts coherent. |
| Per-league failure (runtime) | Graceful — skip that league, render others | Best-effort runtime fetch matches user expectation. |
| Classifier | Same as v0 — `MultiOutputClassifier(CalibratedClassifierCV(GradientBoostingClassifier(random_state=0), method='isotonic', cv=3))` | Holding classifier constant isolates the feature-improvement effect. |

## Architecture

### File layout

```
src/
  ingest/
    __init__.py
    football_data.py    # NEW — replaces library's CSV download
    understat.py        # NEW — scrapes xG/xGA per match
  features.py           # NEW — engineered features from raw match + xG data
  predictions.py        # MODIFIED — multi-league routing
  team_names.py         # MODIFIED — 3-way registry (odds_api / football_data / understat)
  odds.py               # MODIFIED — 5-league fetch, per-league graceful failure
  sportsbet_patch.py    # DELETED

scripts/
  train_model.py        # REWRITTEN — orchestrates ingest → features → 5 fits → backtest

tests/
  test_value.py         # UNCHANGED
  test_predictions.py   # MODIFIED — phase-3 tests adapt to multi-league
  test_team_names.py    # MODIFIED — 3-way map covering ~100 teams
  test_train_pipeline.py # MODIFIED — atomic-write extends to 7 files (5 pkl + parquet + json)
  test_features.py      # NEW — unit tests for each feature function
  test_ingest.py        # NEW — unit tests for ingest modules (HTTP mocked)

models/
  epl_bettor.pkl              # NEW per-league files
  laliga_bettor.pkl
  seriea_bettor.pkl
  bundesliga_bettor.pkl
  ligue1_bettor.pkl
  fixtures_data.parquet       # cached fixtures DataFrame (replaces loader pickle)
  backtest.json               # nested per-league summary

data/cache/             # gitignored — regenerable
  football_data/        # CSVs cached by season
  understat/            # parquets cached by league-season

app.py                  # MODIFIED — multi-league handling, league filter
pyproject.toml          # MODIFIED — drop sports-betting + pytz, add beautifulsoup4
.gitignore              # MODIFIED — add data/cache/
```

### Training data flow

```
download CSVs (football_data.py) ────┐
                                     ├─→ raw match DataFrame
scrape xG (understat.py)            ─┘            │
                                                  ▼
                       features.py: compute_features(matches_df, xg_df)
                                                  │
                                                  ▼
              for league in BIG_5_LEAGUES:
                  filter to league → X, Y, O
                  bettor = build_calibrated_classifier()
                  league_summary = run_backtest_cv(bettor_factory, X, Y, O)
                  bettor.fit(X, Y)   # fit on full history for inference
                                                  │
                                                  ▼
              extract upcoming-fixtures DataFrame from features.py
                                                  │
                                                  ▼
              atomic write: 5× bettor.pkl + fixtures_data.parquet + backtest.json
```

### Inference data flow (dashboard load)

```
app.py opens
   │
   ▼
load_models() → reads 5 bettor pkls + fixtures_data.parquet + backtest.json
   │
   ▼
The Odds API → fetches fixtures for all 5 leagues (per-league graceful failure)
   │
   ▼
for each fixture:
   league from sport_key → pick the right bettor (or demo fallback)
   look up team names → normalise via team_names registry
   look up fixture row in fixtures_data.parquet
   bettor.predict_proba → MatchPrediction
   │
   ▼
_build_rows → multi-select league filter applied → table → dashboard
```

## Components

### `src/ingest/football_data.py`

Replaces what the library's `_get_data` did — direct download from `raw.githubusercontent.com/georgedouzas/sports-betting/data/...` with retry and caching.

```python
def download_training_data(leagues: list[str], years: list[int]) -> pd.DataFrame:
    """Download and concatenate season CSVs for the given leagues and years.

    Cached at data/cache/football_data/{league}_{division}_{year}.csv.
    Current season always re-fetched (it's still being updated weekly).
    Failure: retry once, then skip; if >50% of seasons skipped, raise.
    """

def download_fixtures_data() -> pd.DataFrame:
    """Download the upcoming-fixtures CSV. Always re-fetched (no cache)."""
```

### `src/ingest/understat.py`

Scrapes xG per match from Understat's HTML pages. Each page has match data embedded in a `<script>` tag as JSON; BeautifulSoup extracts it.

```python
def fetch_league_xg(league: str, year: int) -> pd.DataFrame:
    """Returns DataFrame: date, home_team, away_team, home_xg, away_xg.

    Cached at data/cache/understat/{league}_{year}.parquet.
    Etiquette: 1.5s sleep between requests, identifying User-Agent.
    Failure: retry once, then skip; missing matches → NaN xG → mean imputation later.
    """

def fetch_xg_data(leagues: list[str], years: list[int]) -> pd.DataFrame:
    """Concatenated xG across multiple league-seasons."""
```

### `src/team_names.py` — 3-way registry

Replaces the v0 2-way map. Each team has aliases for all three sources we touch:

```python
@dataclass
class TeamNames:
    canonical: str         # internal canonical form
    odds_api: str          # The Odds API form
    football_data: str     # football-data.co.uk form
    understat: str         # Understat form

REGISTRY: dict[str, TeamNames] = {
    "Arsenal": TeamNames(canonical="Arsenal", odds_api="Arsenal",
                         football_data="Arsenal", understat="Arsenal"),
    # ~100 entries across the 5 leagues
}

def to_canonical(name: str, source: str) -> str:
    """source ∈ {'odds_api', 'football_data', 'understat'}.
    Raises KeyError with actionable message if unmapped."""

def from_canonical(canonical: str, target: str) -> str:
    """Reverse lookup."""
```

Building the registry is a one-time tedious task — enumerate ~100 EPL/La Liga/Serie A/Bundesliga/Ligue 1 teams, verify each has the correct form for all three sources. Verification scripts (one per source, asserting every fetched name is mapped) catch unmapped entries early.

### `src/features.py`

Pure-function pipeline. Each helper takes a DataFrame, returns the same DataFrame with new columns added. Top-level `compute_features` orchestrates the sequence.

```python
def compute_features(matches_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Top-level: returns matches_df with feature columns added."""
    df = _normalise_team_names(matches_df)
    df = _merge_xg(df, xg_df)
    df = _add_rest_features(df)
    df = _add_form_features(df, window=5)
    df = _add_rolling_goals(df, window=5)
    df = _add_xg_features(df, window=5)
    df = _add_strength_of_schedule(df, window=5)
    return df
```

#### Feature inventory (~22 numeric + 3 categoricals after one-hot)

**Rest and fitness (computed from match dates alone)**
- `home_rest_days`, `away_rest_days` — capped at 14 to handle international breaks
- `home_matches_last_14d`, `away_matches_last_14d` — fixture congestion

**Recent form (last 5 matches per team)**
- `home_form_wins`, `home_form_draws`, `home_form_losses`
- `away_form_wins`, `away_form_draws`, `away_form_losses`

**Recent goals (last 5 matches per team)**
- `home_goals_scored_last_5`, `home_goals_conceded_last_5`
- `away_goals_scored_last_5`, `away_goals_conceded_last_5`

**xG features (Understat data, last 5 matches per team)**
- `home_xg_last_5`, `home_xga_last_5`
- `away_xg_last_5`, `away_xga_last_5`
- `home_xg_minus_actual_last_5` — overperformance/underperformance signal
- `away_xg_minus_actual_last_5`

**Strength of schedule (last 5 opponents per team)**
- `home_opp_strength_last_5` — average prior-season league position of last 5 opponents
- `away_opp_strength_last_5`
- (Promoted teams get league-size-2 default, e.g. 18 for EPL)

**Identity / context features (kept from v0)**
- `league` (5 categories, one-hot)
- `home_team`, `away_team` (~100 categories, one-hot)
- `division` (always 1), `year`

#### Data-leakage guard

Every rolling window uses pandas `groupby(...).rolling(window).agg(...)` with `closed='left'` so the window strictly excludes the current match. A unit test (`test_rolling_goals_uses_only_past_matches`) verifies this on a 3-match synthetic dataset. This is the single most important correctness invariant in the feature layer.

#### Missing-data handling

| Scenario | Fallback |
|---|---|
| Missing xG (Understat skipped) | NaN → SimpleImputer (mean) at fit time. Logged. >5% imputed → prominent warning. |
| First N matches per team (no rolling history) | 0 for counts, league-mean for averages, 7 for rest_days, 1 for matches_last_14d |
| Promoted team (no prior-season position) | League-size-2 (e.g., 18 for EPL) |

### `scripts/train_model.py`

Top-to-bottom orchestrator. Pseudocode:

```python
def main():
    training_years = list(range(2018, 2026))

    # 1. Ingest
    matches_df = download_training_data(LEAGUES, training_years)
    xg_df = fetch_xg_data(LEAGUES, training_years)
    fixtures_raw = download_fixtures_data()

    # 2. Feature engineering
    features_df = compute_features(matches_df, xg_df)
    fixtures_features = compute_features(fixtures_raw, xg_df)  # for inference

    # 3. Per-league training loop
    bettors = {}
    league_summaries = {}
    for league in BIG_5_LEAGUES:
        X, Y, O = split_by_league(features_df, league)
        bettor = build_calibrated_classifier()
        league_summaries[league] = run_backtest_cv(
            build_calibrated_classifier, X, Y, O,
            n_splits=5, value_threshold=0.05,
        )
        bettor.fit(X, Y)
        bettors[league] = bettor

    # 4. Build summary
    summary = {
        "trained_at": iso_utc_now(),
        "training_seasons": training_years,
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "n_training_matches": len(features_df),
        "leagues": league_summaries,
    }

    # 5. Atomic persist
    write_artifacts_atomically(bettors, fixtures_features, summary)
```

### `run_backtest_cv()`

TimeSeriesSplit CV; for each fold, fit + predict + select value bets where `calibrated_prob × (1 / fair_implied_prob_after_margin) > 1 + threshold`. Same value-selection logic as the dashboard's `assess_value` so backtest yields and live picks agree on what counts as a value bet.

### `build_calibrated_classifier()`

Identical shape to v0:

```python
def build_calibrated_classifier() -> Pipeline:
    return make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore"),
             ["league", "home_team", "away_team"]),
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
```

### `src/predictions.py`

```python
@dataclass
class Models:
    bettors: dict[str, Any]                 # league → fitted classifier
    fixtures_df: pd.DataFrame | None        # snapshot of upcoming-fixture features
    backtest: dict | None

    @property
    def is_ready(self) -> bool:
        return (
            len(self.bettors) == len(BIG_5_LEAGUES)
            and self.fixtures_df is not None
            and self.backtest is not None
        )

def load_models() -> Models:
    """Read 5 *_bettor.pkl + fixtures_data.parquet + backtest.json. Any missing → demo."""

def predict_fixture(
    models: Models | None,
    league: str,
    home_team: str,
    away_team: str,
) -> MatchPrediction:
    """Multi-league routing. Demo fallback per fixture on any error."""
```

### `src/odds.py`

```python
LEAGUE_ENDPOINTS = {
    "epl": "soccer_epl",
    "laliga": "soccer_spain_la_liga",
    "seriea": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue1": "soccer_france_ligue_one",
}

def get_big5_odds(api_key: str | None = None) -> list[dict]:
    """Fetch all 5 leagues. Tag each fixture with league key.
    Per-league failure is graceful — skip that league, return others."""
```

Cache TTL bumped from 600s (10 min) → 1800s (30 min) to manage 500-req/month free quota.

### `app.py`

**Demo banner** copy updated for multi-league.

**Sidebar** — multi-select `Leagues` filter above the value-bets section. Default: all 5 selected. Persists in session state.

**Value bets table** — gains a "League" column. ~50 weekly fixtures × 5 outcomes = ~250 rows; filter is essential.

**Model info expander** — flat 25-row table (5 leagues × 5 markets) showing n_bets and yield_pct per cell. Stale-model `⚠️` logic unchanged.

**Tracking tab** — unchanged for v2. League column on bet history is nice-to-have; deferred.

## Data shapes

### `backtest.json`

```json
{
  "trained_at": "2026-05-12T10:30:00Z",
  "training_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  "sklearn_version": "1.5.2",
  "n_training_matches": 15234,
  "leagues": {
    "epl": {
      "n_matches": 3040,
      "markets": {
        "home_win":  {"n_bets": 287, "yield_pct": 1.2},
        "draw":      {"n_bets": 102, "yield_pct": -3.4},
        "away_win":  {"n_bets": 198, "yield_pct": 0.6},
        "over_2.5":  {"n_bets": 312, "yield_pct": 2.1},
        "under_2.5": {"n_bets": 289, "yield_pct": -0.8}
      }
    },
    "laliga":     { "...same shape..." },
    "seriea":     { "...same shape..." },
    "bundesliga": { "...same shape..." },
    "ligue1":     { "...same shape..." }
  }
}
```

### `fixtures_data.parquet`

Pre-computed feature DataFrame for all upcoming fixtures across the 5 leagues. Columns: same ~22 features as training. Indexed by `(league, date, home_team, away_team)` for fast lookup at inference. Saves us pickling a stateful loader object and replaying its data-fetch flow at runtime.

**Source of upcoming fixtures (planner-resolvable):** v0 used the library's `fixtures.csv` URL at `raw.githubusercontent.com/georgedouzas/sports-betting/data/data/soccer/modelling/fixtures.csv` — verified there as a single file in the GitHub data folder. Whether it includes fixtures for all five Big-5 leagues or only EPL is not known until the implementer inspects it. Fallback if it's EPL-only: build the fixtures snapshot at training time from a one-time call to The Odds API (`get_big5_odds()`) and snapshot the canonical-name fixture rows after merging in feature columns. Either path produces the same parquet shape; the planner picks based on what's actually in `fixtures.csv` when verified.

### `MatchPrediction` (existing dataclass, unchanged)

Same fields. `p_over_2_5` populated when `is_demo=False` (already the case in v0).

## Error handling matrix

| Layer | Failure mode | Behaviour |
|---|---|---|
| Ingestion (football_data) | URL 404, timeout | Retry once after 5s, then skip season. >50% skipped → raise. |
| Ingestion (Understat) | HTML parse error, 4xx, rate limit | Catch + log + skip. Missing matches → NaN xG → mean imputation. >50% seasons missing → raise. |
| Feature engineering | Missing source columns, date parse error | Fail loudly — these are bugs, not transient. |
| Training: per-league fit | Convergence failure, OOM, NaN in features | Fail the whole run. Existing artifacts untouched (atomic-write protects). |
| Backtest CV | Empty test fold for a small league | Skip empty folds, log. All folds empty → raise. |
| Atomic write | Disk full, permission error mid-write | Cleanup all .tmp files, re-raise original. |
| Inference: load_models | Pickle missing/corrupt, parquet missing/corrupt | `is_ready=False` → demo for everything. |
| Inference: predict_fixture | Team name unmapped, library throws, league unrecognised | Per-fixture demo fallback. Other fixtures unaffected. |
| Dashboard: Odds API fetch | One league's endpoint fails | Per-league graceful — that league shows empty, others render. |
| Dashboard: rendering | Fixture has no model match | Per-fixture demo fallback (existing). |

## Testing strategy

### Unit tests (pytest)

| File | New/Modified | Tests | Cost |
|---|---|---|---|
| `tests/test_value.py` | UNCHANGED | 6 | <1s |
| `tests/test_predictions.py` | MODIFIED | ~12 (was 17, removed cache test, added cross-league routing) | ~1s |
| `tests/test_team_names.py` | MODIFIED | ~7 (covers all 5 leagues, 3-way round-trip) | <1s |
| `tests/test_train_pipeline.py` | MODIFIED | ~5 (extends to 7-file atomic write + backtest yield arithmetic) | ~1s |
| `tests/test_features.py` | NEW | ~6-7 (one per feature function + end-to-end + leakage guard) | <1s |
| `tests/test_ingest.py` | NEW | ~5 (URL construction, parser, cache hit, mocked failures) | <1s |
| **Total** | | **~40-50** | **~5s** |

All HTTP mocked. No real network calls in pytest. Real-network correctness verified in manual smoke test.

### Manual smoke checklist

1. With `models/` empty: dashboard shows demo banner, value bets render synthetic across 5 leagues, league filter works.
2. `uv run python scripts/train_model.py` — completes in ~30-50 minutes. Progress markers `[1/5]`-`[5/5]` print. Per-market summary block at end.
3. `cat models/backtest.json` — nested-by-league shape, `n_training_matches` ≈ 15,000, yields are sensible (not all zero, not all extreme negative).
4. `dir models/` shows 5 bettor.pkl + fixtures_data.parquet + backtest.json.
5. Dashboard restarted: no demo banner, Model info expander shows 25-row per-league/per-market table.
6. Spot-check 5 fixtures across 3 leagues — probabilities plausible, edges within sane ranges.
7. Multi-select league filter — deselect 4 leagues, verify table shows only one league.
8. Optional: write `docs/superpowers/notes/2026-MM-DD-v2-smoke-test-results.md`.

### Static checks

```
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

All three must pass before any commit. No mypy (out of scope for personal Python project).

## Out of scope (YAGNI)

- Pooled cross-league model (alternative architecture; can revisit if per-league models underperform)
- Wider feature set: referee tendencies, weather, attendance, manager-change effects, line movement (deferred to v3+ if v2 plateaus)
- Pre-paid data feeds (Opta event data, real lineup news) — deferred until v2 yields justify spend
- Automated retrain (cron or GitHub Actions) — manual `uv run python scripts/train_model.py` for now
- CI pipeline (GitHub Actions) — local pre-commit checks remain authoritative
- Tracking-tab league column — deferred (existing schema doesn't differentiate)
- AI assistant for picking + explaining the dashboard's value bets — explicitly carved out as v3
- Streamlit Cloud deployment of the trained artifacts — separate decision (commit-to-Git vs Git LFS vs external storage) deferred

## Risks

1. **Big-5 markets are even more efficient than EPL alone.** All five per-league models might still show negative yields. Mitigation: per-league diagnostics make this visible market-by-market; we can drop unprofitable leagues from the live picks while keeping the rest.

2. **Understat scraping breaks.** The site changes its HTML structure; the parser fails. Mitigation: graceful skip per league-season; loud warning if >50% missing forces investigation rather than silent bad training.

3. **The 3-way team-name registry has gaps.** A fixture's team isn't mapped → demo fallback for that fixture. Mitigation: verification scripts (one per source) that fetch live team lists and assert every name is in the registry, run at training time and at first dashboard load.

4. **xG features don't actually move the needle on EPL.** Calibrated probabilities reflect xG signal but bookmakers already incorporate it in their lines. Mitigation: the per-league backtest will show this clearly. If yields don't improve materially over v0, we know the basic approach has hit a public-data ceiling and decisions about Level 3 (paid data) become informed.

5. **`fixtures_data.parquet` snapshot becomes stale.** A new EPL fixture is added after training. The file contains only fixtures known at training time. Mitigation: the dashboard's per-fixture demo fallback already handles "no fixture row in fixtures_df" — same path as v0. Acceptable for a weekly-retrain cadence; would matter more if the model became event-driven.

6. **Calibration with only 3 CV folds is noisy on smaller leagues.** Ligue 1 has fewer high-quality matches and 3-fold isotonic calibration may produce noisy probabilities. Mitigation: calibration plots in a follow-up task if backtest yields show abnormal extreme-bet distributions.

7. **Odds API quota burn.** 5 leagues × 2 fetches/hour (30-min cache) = 240/day worst-case. Free tier is 500/month total. Realistic usage stays under 100/month. Mitigation: cache TTL is conservative; can extend further if quota becomes an issue.

## Acceptance criteria

The work is done when:

- [ ] `uv run python scripts/train_model.py` completes within ~50 minutes on a laptop, producing all 5 `*_bettor.pkl` + `fixtures_data.parquet` + `backtest.json` in `models/`.
- [ ] All ~40-50 pytest tests pass in <10 seconds.
- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `Y.columns` order assumed by `predict_fixture`'s `probs[0..3]` index mapping matches what `MultiOutputClassifier` produces (verified at training time and asserted in code).
- [ ] Dashboard with all artifacts present: no demo banner, multi-select league filter renders, Model info expander shows per-league/per-market table, value-bets table includes both h2h and totals rows across multiple leagues.
- [ ] Dashboard with artifacts absent: demo banner shows, synthetic predictions render across all 5 leagues without crash.
- [ ] Manual smoke checklist (8 steps above) all pass.
- [ ] Spot-check: at least 5 fixtures across at least 3 different leagues have plausible probabilities (not 33/33/33, not 99/0/0).
- [ ] Every team in the live Odds API response across all 5 leagues has an entry in `team_names.REGISTRY`.
- [ ] No reference to `sports-betting`, `sportsbet`, or `sportsbet_patch` remains in source code (excluding git history and docs).
- [ ] Per-market backtest yields recorded in `backtest.json` for all 5 leagues; v2 yields recorded in commit message or follow-up doc to compare against v0's `home_win: -14.6%` baseline.
