# ⚽ Football Edge

Personal football value betting tool. Scans EPL fixtures, identifies value bets where your model probability exceeds the bookmaker's fair implied probability, and tracks P&L over time.

> **Important:** Paper-trade for at least 2-3 months before risking real money. Most public models lose to the market. This is a research aid, not a guaranteed edge.

## Stack

- **Streamlit** — single-page dashboard, mobile-friendly
- **sports-betting** — model engine ([georgedouzas/sports-betting](https://github.com/georgedouzas/sports-betting))
- **The Odds API** — live bookmaker odds (free tier, 500 reqs/month)
- **Supabase** — persistent bet tracking
- **Python 3.11+** with `uv`

## What you get

Two tabs:
- **This Week** — upcoming EPL fixtures, model probabilities vs bookmaker fair odds, value bets highlighted, suggested fractional Kelly stake
- **Tracking** — log placed bets, settle them after matches, watch P&L and ROI accumulate

## Local setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Then fill in:
- `ODDS_API_KEY` — get one free at [the-odds-api.com](https://the-odds-api.com)
- `SUPABASE_URL` and `SUPABASE_KEY` — from your Supabase project (Settings → API)
- `APP_PASSWORD` — pick a strong password

### 3. Set up the Supabase database

Open your Supabase project → SQL Editor → New query. Paste the contents of `supabase/schema.sql` and run it.

### 4. Run it (demo mode works immediately)

```bash
uv run streamlit run app.py
```

Visit http://localhost:8501. You'll see a yellow "Demo mode" banner because no model is trained yet. The dashboard works end-to-end with synthetic probabilities — useful for verifying the pipeline before you invest in training.

### 5. Train the real model

Open `scripts/train_model.py`, wire up the `sports-betting` library calls (the file has a fully commented sketch — see the [library docs](https://georgedouzas.github.io/sports-betting/)), then:

```bash
uv run python scripts/train_model.py
```

That writes `models/epl_model.pkl`. You'll also need to fill in the `predict_fixture()` body in `src/predictions.py` to call your trained bettor's `predict_proba()`. Reload the dashboard — the demo banner disappears, real predictions take over.

## Deploy for mobile access

1. Push to a **private** GitHub repo (`.env` is gitignored — verify before pushing)
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. New app → connect the repo → main file is `app.py`
4. In the app settings, open **Secrets** and paste in TOML format:
   ```toml
   ODDS_API_KEY = "your_key"
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your_anon_key"
   APP_PASSWORD = "your_password"
   ```
5. Open the deployed URL on your phone, save to home screen

## How value detection works

For each upcoming fixture and each market (Home / Draw / Away):

1. The app pulls the **best available decimal odds** across UK bookmakers
2. The raw implied probability `1 / odds` from each outcome is summed — the excess over 1.0 is the bookmaker's margin (vig). Dividing each implied prob by the total gives the **fair implied probability** the book is actually pricing
3. **Value % = your model's probability − fair implied probability**
4. Bets above your threshold (default 5%) are flagged
5. The suggested stake uses **fractional Kelly** (default quarter Kelly). Quarter Kelly is the right starting point because your model probabilities are estimates — full Kelly assumes they're truth and is too aggressive when wrong

## File map

```
football-edge/
├── app.py                  # Streamlit dashboard (auth, fixtures tab, tracking tab)
├── src/
│   ├── auth.py             # Password gate
│   ├── odds.py             # The Odds API client
│   ├── predictions.py      # Model wrapper + demo mode fallback
│   ├── value.py            # Value calculation, margin removal, Kelly
│   └── database.py         # Supabase client (predictions/bets/outcomes)
├── scripts/
│   └── train_model.py      # Stub — wire up sports-betting library here
├── supabase/
│   └── schema.sql          # Three tables: predictions, bets, outcomes
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Disclaimer

For personal use only. Not financial advice. UK gambling support: [BeGambleAware](https://www.begambleaware.org).
