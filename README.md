# Football Edge

Personal football value-betting tool. Scans Big-5 European fixtures (EPL, La Liga, Serie A, Bundesliga, Ligue 1), runs a trained per-league model against live bookmaker odds, flags where the model's probability beats the book's fair price, and uses an LLM to explain the picks in plain English.

> **Important:** Paper trade for at least 2-3 months before risking real money. Backtests beat the future ~30% of the time. Most public betting models lose to the market. This is a research aid, not a guaranteed edge.

**Live:** [football-edge-hallengray.streamlit.app](https://football-edge-hallengray.streamlit.app) (password-gated)

## What you get

A two-tab Streamlit dashboard, mobile-friendly, password-gated:

### 📊 This Week
- All upcoming fixtures across the 5 big leagues with live UK bookmaker odds (1X2 + over/under 2.5)
- Trained per-league model probabilities, with the bookmaker's margin (vig) stripped to show the **fair price** the book is actually quoting
- Value bets flagged where your model's edge ≥ threshold (default 5%)
- Suggested fractional-Kelly stakes (default quarter Kelly — full Kelly is too aggressive when probabilities are estimates)
- **🤖 AI picker** — one button hits an LLM via OpenRouter and returns the top picks, each with a one-sentence plain-English reason and risk:
  - **Top 5 Draws** — the model's strongest backtest signal (Bundesliga draws specifically)
  - **Top 10 Across All Markets** — ranked by `expected_yield_pct = current_edge + historical_market_yield`
- **One-click bulk-log** — every AI pick is logged to Supabase as a £10 paper trade with a single button click. Dedupes across the two sections so a Bayern draw appearing in both lists only logs once.

### 📈 Tracking
- All logged bets with status (pending/settled), match context, stake, outcome, payout
- Live P&L, ROI, win rate, total staked
- Settle pending bets after kickoff with a one-form click

### Sidebar
- **Value threshold** slider (1-20%) and **Kelly fraction** slider (0.05-1.0) — tune live without redeploying
- **📊 Model info** expander — when the model was trained, how many matches, per-league/per-market backtest yields, stale-model warning if >90 days old

## What's actually trained

The model is **already trained and shipped in `models/`** — you don't need to train it yourself unless you're retraining.

| Detail | Value |
|--------|-------|
| Architecture | Per-league `ClassifierBettor` from [sports-betting](https://github.com/georgedouzas/sports-betting), one per league (5 total) |
| Training matches | 14,359 across the 5 big leagues |
| Seasons | 2018-2025 (8 seasons) |
| sklearn version | 1.8.0 |
| Markets | home_win, draw, away_win, over_2.5, under_2.5 |
| Feature sources | football-data.co.uk historicals (results + odds) + Understat (xG) |
| Engineered features | Rest days, rolling form (W/D/L), rolling goals for/against, shooting (shots, shots on target), xG / xGA, strength-of-schedule |

**Backtest results** (caveat: past performance ≠ future — 22 of 25 league/market combos lost money):

| League | Market | Yield | n_bets |
|--------|--------|------:|-------:|
| Bundesliga | Draw | **+7.20%** | 679 |
| Ligue 1 | Away win | +1.62% | 740 |
| Ligue 1 | Over 2.5 | +0.27% | 376 |

The Bundesliga-draw signal is the standout — that's why the AI picker has a dedicated "Top 5 Draws" section. Paper trade the draws specifically before believing any of this.

## Stack

- **Streamlit** (1.40+) — single-page dashboard, mobile-friendly
- **sports-betting** — model engine ([georgedouzas/sports-betting](https://github.com/georgedouzas/sports-betting))
- **scikit-learn 1.8.0** — under the hood
- **The Odds API** — live bookmaker odds (free tier, 500 requests/month)
- **OpenRouter** — LLM for AI picks (default model is on the free tier)
- **Supabase** (Postgres) — bet tracking persistence
- **Python 3.13** (pinned — see [Deploy](#deploy-for-mobile-access))
- **uv** for dependency management

## Local setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Fill in:
- `ODDS_API_KEY` — free at [the-odds-api.com](https://the-odds-api.com)
- `OPENROUTER_API_KEY` — free at [openrouter.ai](https://openrouter.ai); the default model (`nvidia/nemotron-3-nano-30b-a3b:free`) is on the free tier
- `SUPABASE_URL` and `SUPABASE_KEY` — from your Supabase project (Settings → API)
- `APP_PASSWORD` — pick a strong password (gates the app)

Optional tuning:
- `VALUE_THRESHOLD=0.05` — minimum edge to flag a bet (default 5%)
- `KELLY_FRACTION=0.25` — fraction of full Kelly to suggest (default quarter Kelly)
- `OPENROUTER_MODEL` — override the default LLM if you want a different one

### 3. Set up Supabase

Open your Supabase project → SQL Editor → New query. Paste `supabase/schema.sql` and run it. That creates the `predictions`, `bets`, and `outcomes` tables.

### 4. Run it

```bash
uv run streamlit run app.py
```

Visit http://localhost:8501. The 5 trained models in `models/` load automatically — no demo banner.

### 5. Retrain (only if you need to)

```bash
uv run python scripts/train_model.py
```

Runs ~30-50 minutes on a laptop. Pulls historicals + xG from football-data.co.uk and Understat, fits one classifier per league, runs the backtest, then writes the 7 artifacts atomically to `models/` (5 league bettors + fixtures parquet + backtest summary). Atomic write means existing artifacts are only replaced if the entire run succeeds.

## Deploy for mobile access

1. Push to a **private** GitHub repo (`.env` is gitignored — verify before pushing).
2. Sign up at [share.streamlit.io](https://share.streamlit.io).
3. New app → connect repo → main file is `app.py`.
4. **In Advanced settings, pick Python 3.13.** Not 3.14 — it has a regression in `dataclasses._is_type` that crashes module imports under Streamlit's script runner. Don't pick 3.12 either — match what's pinned.
5. **Secrets** (TOML format):
   ```toml
   ODDS_API_KEY = "your_key"
   OPENROUTER_API_KEY = "your_key"
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your_anon_key"
   APP_PASSWORD = "your_password"
   ```
6. Open the deployed URL on your phone, save to home screen.

**Changing Python version later:** Streamlit Cloud doesn't read `.python-version` or any in-repo file. To change the Python version on a deployed app, you have to delete and redeploy with the new version selected in Advanced settings. The `.python-version` file in this repo is documentation-only.

## How value detection works

For each upcoming fixture and each market (Home / Draw / Away / Over 2.5 / Under 2.5):

1. Pull the **best decimal odds** across UK bookmakers via The Odds API.
2. Convert each outcome's odds to its raw implied probability (`1/odds`). The h2h probs sum to >1 — the excess is the bookmaker's margin (vig). Dividing each by the total gives the **fair implied probability** the book is actually pricing.
3. **Value % = your model's probability − fair implied probability.**
4. Bets above your threshold (default 5%) get flagged.
5. Suggested stake uses **fractional Kelly** (default quarter Kelly). Full Kelly assumes your model probabilities are truth and is too aggressive when they're estimates. Quarter Kelly is the standard for personal use.

## How the AI picker works

When you click 🤖 Get AI picks:

1. App computes `expected_yield_pct = edge_pct + market_yield_pct` for every value bet (combines the current edge with the model's historical performance on that league/market). Bets in markets the model has historically lost money on get penalised even if their current edge looks attractive.
2. Sorts by `expected_yield_pct` descending and caps the prompt input at top 50 (free-tier LLMs emit malformed JSON when the prompt gets too long).
3. Two parallel calls to OpenRouter's chat-completions endpoint with a constrained system prompt (temperature 0, JSON-only output): "top 5 draws" and "top 10 across all markets".
4. Renders each pick with the model's reason + risk in plain English. **One bulk-log button** writes all distinct picks (deduped across both sections) to Supabase as £10 paper trades.

The AI doesn't decide what to bet on — the trained model does. The LLM just translates the math into prose. Picks whose `pick_id` doesn't exist in the value-bets table are dropped silently to defend against hallucinated row indices.

## File map

```
football-edge/
├── app.py                              # Streamlit dashboard (auth, fixtures, AI picker, tracking)
├── src/
│   ├── auth.py                         # Password gate
│   ├── odds.py                         # The Odds API client (5-league fanout)
│   ├── predictions.py                  # Multi-league model loader + predict_fixture routing
│   ├── value.py                        # Margin removal, value calc, Kelly
│   ├── ai_explainer.py                 # OpenRouter LLM client + JSON-output parsing
│   ├── database.py                     # Supabase client (predictions/bets/outcomes)
│   ├── features.py                     # Feature engineering (form, rolling goals, xG, SOS)
│   ├── team_names.py                   # 3-way team-name registry (odds_api / football_data / understat)
│   └── ingest/
│       ├── football_data.py            # football-data.co.uk historicals
│       └── understat.py                # Understat xG
├── scripts/
│   ├── train_model.py                  # Train Big-5 model, write 7 artifacts atomically
│   └── verify_team_names.py            # Smoke-test the team-name registry
├── models/                             # Shipped in the repo
│   ├── {epl,laliga,seriea,bundesliga,ligue1}_bettor.pkl
│   ├── fixtures_data.parquet
│   └── backtest.json
├── tests/                              # 70 tests
├── supabase/schema.sql                 # 3 tables: predictions, bets, outcomes
├── docs/superpowers/                   # Historical specs/plans (frozen at write-time)
├── pyproject.toml
├── .python-version                     # 3.13 (Streamlit Cloud doesn't read this — dashboard-only)
└── .env.example
```

## Running the tests

```bash
uv run pytest -q
```

70 tests covering feature engineering, ingest, prediction routing, team-name registry, training pipeline, value calculation, and the AI explainer.

## Disclaimer

For personal use only. Not financial advice. UK gambling support: [BeGambleAware](https://www.begambleaware.org).
