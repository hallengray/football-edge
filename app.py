"""Football Edge — personal football value betting dashboard.

Run locally:    uv run streamlit run app.py
Deploy:         push to GitHub, connect to Streamlit Community Cloud
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.auth import check_password
from src.database import (
    get_bets_with_context,
    get_client,
    log_prediction,
    record_bet,
    record_outcome,
)
from src.odds import best_odds_for_outcome, get_epl_odds
from src.predictions import Models, load_models, predict_fixture
from src.value import assess_value, remove_bookmaker_margin

logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(
    page_title="Football Edge",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───── Auth gate ─────
if not check_password():
    st.stop()


# ───── Models loader (cached for the session) ─────
@st.cache_resource(show_spinner="Loading model…")
def _load_models_cached() -> Models:
    return load_models()


_models = _load_models_cached()


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

    with st.expander("📊 Model info", expanded=False):
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
            label = market_key.replace("_", " ").title()
            st.markdown(f"- **{label}**: {n} bets, {wr:.0f}% win rate, {yp:+.1f}% yield")
        if stale_marker:
            st.caption("Model is over 90 days old — consider retraining.")


# ───── Sidebar settings ─────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    value_threshold = (
        st.slider(
            "Value threshold (%)",
            min_value=1,
            max_value=20,
            value=int(float(os.getenv("VALUE_THRESHOLD", "0.05")) * 100),
            help="Flag bets where your edge exceeds this.",
        )
        / 100
    )
    kelly_multiplier = st.slider(
        "Kelly fraction",
        min_value=0.05,
        max_value=1.0,
        value=float(os.getenv("KELLY_FRACTION", "0.25")),
        step=0.05,
        help="0.25 = quarter Kelly. Lower is safer.",
    )
    st.divider()
    st.caption(
        "Paper trade for 2-3 months before risking real money. "
        "Most public models lose to the market."
    )
    st.caption("[BeGambleAware](https://www.begambleaware.org)")
    st.divider()

    # Model info expander (renders only when models loaded)
    render_model_info(_models)


# ───── Demo mode banner ─────
if not _models.is_ready:
    st.warning(
        "🟡 **Demo mode** — no trained model found. The probabilities shown are "
        "synthetic and **not predictive**. Wire up `scripts/train_model.py` and "
        "run `uv run python scripts/train_model.py` to use a real model."
    )

# ───── Tabs ─────
tab_fixtures, tab_tracking = st.tabs(["📊 This Week", "📈 Tracking"])


# ─────────────────────────────────────────────────────────────────────
#  Fixtures tab
# ─────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=600, show_spinner="Pulling fixtures and odds…")
def _fetch_fixtures() -> list[dict]:
    """Pull EPL fixtures + bookmaker odds. Cached for 10 minutes."""
    return get_epl_odds()


def _build_rows(
    fixtures: list[dict],
    threshold: float,
    kelly_mult: float,
    models: Models,
) -> list[dict]:
    """Cross-reference predictions with bookmaker odds, return display rows."""
    rows: list[dict] = []

    for fixture in fixtures:
        home_team = fixture.get("home_team")
        away_team = fixture.get("away_team")
        if not home_team or not away_team:
            continue

        pred = predict_fixture(models, home_team, away_team)

        home_odds = best_odds_for_outcome(fixture, "h2h", home_team)
        away_odds = best_odds_for_outcome(fixture, "h2h", away_team)
        draw_odds = best_odds_for_outcome(fixture, "h2h", "Draw")

        if not (home_odds and away_odds and draw_odds):
            continue

        implied = [1 / home_odds[0], 1 / draw_odds[0], 1 / away_odds[0]]
        fair = remove_bookmaker_margin(implied)

        outcomes = [
            ("Home", home_team, pred.p_home, home_odds, fair[0], "h2h"),
            ("Draw", "Draw", pred.p_draw, draw_odds, fair[1], "h2h"),
            ("Away", away_team, pred.p_away, away_odds, fair[2], "h2h"),
        ]

        # Totals rows — only added if both Over 2.5 and Under 2.5 odds are available
        over_odds = best_odds_for_outcome(fixture, "totals", "Over", point=2.5)
        under_odds = best_odds_for_outcome(fixture, "totals", "Under", point=2.5)
        if over_odds and under_odds and pred.p_over_2_5 is not None:
            # Clamp to [0, 1] — calibrated classifiers can produce tiny noise outside the range
            p_over = max(0.0, min(1.0, pred.p_over_2_5))
            implied_totals = [1 / over_odds[0], 1 / under_odds[0]]
            fair_totals = remove_bookmaker_margin(implied_totals)
            outcomes.append(("Over 2.5", "Over 2.5", p_over, over_odds, fair_totals[0], "totals"))
            outcomes.append(
                (
                    "Under 2.5",
                    "Under 2.5",
                    1 - p_over,
                    under_odds,
                    fair_totals[1],
                    "totals",
                )
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
                    "market": market,
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
    return rows


def render_fixtures_tab() -> None:
    st.markdown("## Upcoming Premier League fixtures")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    try:
        fixtures = _fetch_fixtures()
    except Exception as e:
        st.error(f"Couldn't load fixtures: {e}")
        return

    if not fixtures:
        st.info("No upcoming fixtures with odds available right now.")
        return

    rows = _build_rows(fixtures, value_threshold, kelly_multiplier, _models)
    if not rows:
        st.info("No fixtures returned predictions.")
        return

    df = pd.DataFrame(rows)
    display_cols = [
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

    value_df = df[df["_is_value"]].sort_values("_value_pct", ascending=False)
    other_df = df[~df["_is_value"]]

    if not value_df.empty:
        st.markdown(f"### 🎯 Value bets ({len(value_df)})")
        st.dataframe(
            value_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("📝 Log a value bet", expanded=False):
            _render_log_bet_form(value_df)
    else:
        st.info(
            f"No value bets above the {value_threshold * 100:.0f}% threshold this week. "
            "Try lowering the threshold in the sidebar."
        )

    with st.expander(f"All fixtures ({len(other_df)} non-value rows)", expanded=False):
        st.dataframe(
            other_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )


def _render_log_bet_form(value_df: pd.DataFrame) -> None:
    """Form to log that you've placed a real bet on one of the flagged value picks."""
    options = [
        f"{row['Match']} — {row['Bet']} @ {row['Best odds']} ({row['Edge']} edge)"
        for _, row in value_df.iterrows()
    ]
    choice = st.selectbox("Pick a value bet", options=options, index=None)
    stake = st.number_input("Stake (£)", min_value=0.0, value=10.0, step=1.0)
    notes = st.text_input("Notes (optional)")

    if st.button("💾 Log this bet", type="primary"):
        if choice is None:
            st.warning("Pick a bet first.")
            return
        if stake <= 0:
            st.warning("Stake must be greater than zero.")
            return

        row = value_df.iloc[options.index(choice)]

        try:
            client = get_client()
            kickoff_iso = row["kickoff"]
            kickoff_dt = (
                datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
                if kickoff_iso
                else datetime.now(timezone.utc)
            )

            prediction = log_prediction(
                client,
                fixture_id=row["fixture_id"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                kickoff=kickoff_dt,
                market=row["market"],
                outcome=row["outcome"],
                model_probability=float(row["_model_prob"]),
                bookmaker=row["_bookmaker"],
                decimal_odds=float(row["_decimal_odds"]),
                value_pct=float(row["_value_pct"]),
                kelly_stake_fraction=float(row["_kelly_fraction"]),
            )

            if not prediction.get("id"):
                st.error("Couldn't save the prediction. Check Supabase config.")
                return

            record_bet(
                client,
                prediction_id=prediction["id"],
                stake_amount=float(stake),
                notes=notes,
            )
            st.success("Bet logged. Settle it in the Tracking tab after the match.")
        except Exception as e:
            st.error(f"Couldn't log: {e}")


# ─────────────────────────────────────────────────────────────────────
#  Tracking tab
# ─────────────────────────────────────────────────────────────────────


def render_tracking_tab() -> None:
    st.markdown("## Bet tracking")

    try:
        client = get_client()
        bets = get_bets_with_context(client)
    except Exception as e:
        st.warning(f"Couldn't fetch tracking data: {e}")
        st.caption("Make sure Supabase is configured and the schema is applied.")
        return

    if not bets:
        st.info("No bets logged yet. Log your first one from the **This Week** tab.")
        return

    settled, pending = [], []
    for b in bets:
        outcome_list = b.get("outcomes") or []
        if outcome_list:
            settled.append(b)
        else:
            pending.append(b)

    total_staked = sum(float(b["stake_amount"]) for b in settled)
    total_returned = sum(float((b.get("outcomes") or [{}])[0].get("payout", 0)) for b in settled)
    pnl = total_returned - total_staked
    roi = (pnl / total_staked * 100) if total_staked > 0 else 0
    wins = sum(1 for b in settled if (b.get("outcomes") or [{}])[0].get("won"))
    win_rate = (wins / len(settled) * 100) if settled else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Settled", len(settled))
    c2.metric("Win rate", f"{win_rate:.0f}%")
    c3.metric("Staked", f"£{total_staked:.2f}")
    c4.metric("P&L", f"£{pnl:+.2f}")
    c5.metric("ROI", f"{roi:+.1f}%")

    if pending:
        st.markdown(f"### ⏳ Pending bets ({len(pending)})")
        for bet in pending:
            _render_settle_form(client, bet)

    if settled:
        st.markdown(f"### 📚 History ({len(settled)})")
        history_rows = []
        for b in settled:
            pred = b.get("predictions") or {}
            outcome = (b.get("outcomes") or [{}])[0]
            stake = float(b["stake_amount"])
            payout = float(outcome.get("payout", 0))
            history_rows.append(
                {
                    "Match": f"{pred.get('home_team', '')} vs {pred.get('away_team', '')}",
                    "Bet": f"{pred.get('outcome', '')} @ {pred.get('decimal_odds', '')}",
                    "Stake": f"£{stake:.2f}",
                    "Result": "✅ Won" if outcome.get("won") else "❌ Lost",
                    "Payout": f"£{payout:.2f}",
                    "P&L": f"£{payout - stake:+.2f}",
                    "Settled": (outcome.get("settled_at", "") or "")[:10],
                }
            )
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)


def _render_settle_form(client, bet: dict) -> None:
    """Inline form to mark a pending bet as won or lost."""
    pred = bet.get("predictions") or {}
    stake = float(bet["stake_amount"])
    odds = float(pred.get("decimal_odds", 0))

    label = (
        f"{pred.get('home_team', '?')} vs {pred.get('away_team', '?')} — "
        f"{pred.get('outcome', '?')} @ {odds:.2f} (£{stake:.2f} staked)"
    )

    with st.expander(label, expanded=False):
        c1, c2 = st.columns(2)
        if c1.button("✅ Won", key=f"win_{bet['id']}", use_container_width=True):
            record_outcome(client, bet_id=bet["id"], won=True, payout=stake * odds)
            st.success("Recorded as won.")
            st.rerun()
        if c2.button("❌ Lost", key=f"loss_{bet['id']}", use_container_width=True):
            record_outcome(client, bet_id=bet["id"], won=False, payout=0.0)
            st.success("Recorded as lost.")
            st.rerun()


# ───── Render ─────
with tab_fixtures:
    render_fixtures_tab()

with tab_tracking:
    render_tracking_tab()
