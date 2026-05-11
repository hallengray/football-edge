"""Regression tests for the value/Kelly math in src/value.py."""

from __future__ import annotations

import math

import pytest

from src.value import (
    MAX_REASONABLE_EDGE_DEFAULT,
    PROFITABLE_MARKETS,
    assess_value,
    is_in_profitable_market_set,
    kelly_fraction,
    mask_demo_display_fields,
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


def test_assess_value_rejects_demo_prediction() -> None:
    # When the prediction came from _demo_prediction (fallback hash), we must
    # never flag it as a value bet — regardless of how big the apparent edge looks.
    result = assess_value(
        model_prob=0.55,
        decimal_odds=2.20,
        bookmaker="Test Book",
        fair_implied_prob=0.45,  # would normally be a 10pp value bet
        value_threshold=0.05,
        kelly_multiplier=0.25,
        is_demo=True,
    )
    assert result.is_value_bet is False


def test_assess_value_rejects_unreasonably_large_edge() -> None:
    # A 30pp gap on a liquid Big-5 market is almost always a model error, not
    # genuine value. Default max_reasonable_edge is 0.15 (15pp).
    result = assess_value(
        model_prob=0.50,
        decimal_odds=12.0,
        bookmaker="Test Book",
        fair_implied_prob=0.10,  # 40pp gap — implausible edge
        value_threshold=0.05,
        kelly_multiplier=0.25,
    )
    assert result.is_value_bet is False


def test_assess_value_allows_extreme_edge_when_overridden() -> None:
    # Operator can explicitly override the edge cap when they want to inspect
    # the underlying signal — e.g. for debugging.
    result = assess_value(
        model_prob=0.50,
        decimal_odds=12.0,
        bookmaker="Test Book",
        fair_implied_prob=0.10,  # 40pp gap
        value_threshold=0.05,
        kelly_multiplier=0.25,
        allow_extreme_edge=True,
    )
    assert result.is_value_bet is True


def test_assess_value_at_edge_cap_boundary_still_flags() -> None:
    # value_pct exactly equal to the cap is allowed (the cap uses strict >).
    # Pinning this boundary so a future refactor can't silently flip it.
    result = assess_value(
        model_prob=0.40,
        decimal_odds=4.0,
        bookmaker="Test Book",
        fair_implied_prob=0.40 - MAX_REASONABLE_EDGE_DEFAULT,  # gap == 0.15 exactly
        value_threshold=0.05,
        kelly_multiplier=0.25,
    )
    assert math.isclose(result.value_pct, MAX_REASONABLE_EDGE_DEFAULT, abs_tol=1e-9)
    assert result.is_value_bet is True


def test_assess_value_raises_when_threshold_above_edge_cap() -> None:
    # A caller setting value_threshold > max_reasonable_edge creates an empty
    # valid-edge window: no bet can ever flag. Fail loud at config time.
    with pytest.raises(ValueError, match="value_threshold"):
        assess_value(
            model_prob=0.55,
            decimal_odds=2.20,
            bookmaker="Test Book",
            fair_implied_prob=0.45,
            value_threshold=0.20,
            max_reasonable_edge=0.15,
        )


def test_mask_demo_display_fields_masks_misleading_columns() -> None:
    row = {
        "Bet": "Home: Brighton",
        "Model %": "36.4%",
        "Fair %": "8.3%",
        "Edge": "+28.1%",
        "Kelly stake": "1.42%",
        "_is_demo": True,
    }
    masked = mask_demo_display_fields(row)
    assert masked["Bet"].startswith("⚠ Demo •")
    assert "Home: Brighton" in masked["Bet"]
    assert masked["Model %"] == "—"
    assert masked["Edge"] == "—"
    assert masked["Kelly stake"] == "—"
    # Fair % comes from the bookie, not the model — must NOT be masked.
    assert masked["Fair %"] == "8.3%"


def test_mask_demo_display_fields_leaves_real_rows_untouched() -> None:
    row = {
        "Bet": "Home: Real Madrid",
        "Model %": "62.1%",
        "Fair %": "55.0%",
        "Edge": "+7.1%",
        "Kelly stake": "1.20%",
        "_is_demo": False,
    }
    snapshot = dict(row)
    result = mask_demo_display_fields(row)
    assert result == snapshot


def test_mask_demo_display_fields_handles_missing_is_demo_key() -> None:
    # Defensive: a row without the _is_demo key should be treated as non-demo.
    row = {"Bet": "Home: X", "Model %": "50%", "Edge": "+5%", "Kelly stake": "1%"}
    snapshot = dict(row)
    assert mask_demo_display_fields(row) == snapshot


def test_profitable_market_set_contains_only_the_four_approved_markets() -> None:
    # Pinning the exact set as a regression guard: if anything changes
    # the recommended-market mix, the test forces an explicit update.
    assert PROFITABLE_MARKETS == frozenset(
        {
            ("bundesliga", "draw"),
            ("bundesliga", "home_win"),
            ("ligue1", "away_win"),
            ("ligue1", "over_2.5"),
        }
    )


def test_is_in_profitable_market_set_includes_bundesliga_draw() -> None:
    assert is_in_profitable_market_set("bundesliga", "Draw") is True


def test_is_in_profitable_market_set_includes_ligue1_away() -> None:
    assert is_in_profitable_market_set("ligue1", "Away") is True


def test_is_in_profitable_market_set_includes_bundesliga_home() -> None:
    assert is_in_profitable_market_set("bundesliga", "Home") is True


def test_is_in_profitable_market_set_includes_ligue1_over_2_5() -> None:
    assert is_in_profitable_market_set("ligue1", "Over 2.5") is True


def test_is_in_profitable_market_set_excludes_la_liga_draw() -> None:
    # La Liga draw yielded -7.77% in the May 2026 backtest — a known loser.
    assert is_in_profitable_market_set("laliga", "Draw") is False


def test_is_in_profitable_market_set_excludes_bundesliga_away() -> None:
    # Same league as a profitable market, but a different outcome.
    assert is_in_profitable_market_set("bundesliga", "Away") is False


def test_is_in_profitable_market_set_rejects_unknown_outcome_label() -> None:
    # Defensive: a typo or new market shouldn't accidentally pass the gate.
    assert is_in_profitable_market_set("bundesliga", "definitely-not-a-market") is False
