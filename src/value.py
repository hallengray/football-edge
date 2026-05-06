"""Value bet calculation and Kelly staking.

Core idea: a bet has positive expected value when your model's probability
is higher than the bookmaker's *fair* implied probability (with the margin
stripped out). The Kelly criterion then tells you what fraction of your
bankroll to risk to maximise long-term growth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValueAssessment:
    """Result of evaluating a single outcome for value."""

    model_probability: float
    bookmaker_odds: float
    bookmaker: str
    fair_implied_probability: float
    value_pct: float
    kelly_stake_fraction: float
    is_value_bet: bool


def remove_bookmaker_margin(implied_probs: list[float]) -> list[float]:
    """Strip the bookmaker's margin (vig) from raw implied probabilities.

    Raw implied probs from decimal odds (1/odds) sum to >1 — the excess is the
    margin. Dividing each by the total gives the "fair" probabilities the book
    is actually pricing in.
    """
    total = sum(implied_probs)
    if total <= 0:
        return implied_probs
    return [p / total for p in implied_probs]


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Calculate the optimal Kelly stake fraction.

    f* = (bp − q) / b, where:
        b = decimal_odds − 1  (net odds)
        p = your model's probability
        q = 1 − p

    Returns 0 when the bet has no edge (negative Kelly is "don't bet").
    """
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    p = model_prob
    q = 1 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def assess_value(
    model_prob: float,
    decimal_odds: float,
    bookmaker: str,
    fair_implied_prob: float,
    value_threshold: float = 0.05,
    kelly_multiplier: float = 0.25,
) -> ValueAssessment:
    """Assess whether a single outcome offers positive expected value.

    Args:
        model_prob: Your model's probability for this outcome.
        decimal_odds: Best decimal odds available across bookmakers.
        bookmaker: Name of the book offering those odds.
        fair_implied_prob: The bookie's implied probability with margin removed.
        value_threshold: Minimum edge to flag as a value bet (0.05 = 5%).
        kelly_multiplier: Fraction of full Kelly to use (0.25 = quarter Kelly).
                         Quarter Kelly is the standard for personal use because
                         your model probabilities are estimates, not truth.
    """
    value_pct = model_prob - fair_implied_prob
    full_kelly = kelly_fraction(model_prob, decimal_odds)
    stake_fraction = full_kelly * kelly_multiplier

    return ValueAssessment(
        model_probability=model_prob,
        bookmaker_odds=decimal_odds,
        bookmaker=bookmaker,
        fair_implied_probability=fair_implied_prob,
        value_pct=value_pct,
        kelly_stake_fraction=stake_fraction,
        is_value_bet=value_pct >= value_threshold,
    )
