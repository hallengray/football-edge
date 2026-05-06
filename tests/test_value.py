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
