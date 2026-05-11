"""Value bet calculation and Kelly staking.

Core idea: a bet has positive expected value when your model's probability
is higher than the bookmaker's *fair* implied probability (with the margin
stripped out). The Kelly criterion then tells you what fraction of your
bankroll to risk to maximise long-term growth.
"""

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


# Backtest yields for the production model sit in the 0.27%-7.84% range,
# implying typical per-bet edges of roughly 3-10pp. 15pp is well above that
# floor while still suppressing the 20-50pp gaps we've seen when the model
# silently falls back to demo mode. Revisit if a future backtest shows
# legitimate edges routinely exceeding this cap.
MAX_REASONABLE_EDGE_DEFAULT = 0.15


# (league_key, backtest_market_key) pairs that the production backtest showed
# > +1% positive yield over hundreds-to-thousands of historical bets. Bets
# outside this set are never flagged as value — even when the per-bet edge is
# clean — because the model's calibration in those markets is empirically poor.
# Source: models/backtest.json from 2026-05-11 retrain. Update when retraining
# materially changes which markets clear the +1% bar.
PROFITABLE_MARKETS: frozenset[tuple[str, str]] = frozenset(
    {
        ("bundesliga", "draw"),
        ("bundesliga", "home_win"),
        ("ligue1", "away_win"),
        ("ligue1", "over_2.5"),
    }
)

# Map the dashboard's outcome label to the backtest's market key.
_OUTCOME_LABEL_TO_BACKTEST_KEY: dict[str, str] = {
    "Home": "home_win",
    "Draw": "draw",
    "Away": "away_win",
    "Over 2.5": "over_2.5",
    "Under 2.5": "under_2.5",
}


def is_in_profitable_market_set(league: str, outcome_label: str) -> bool:
    """True if (league, outcome) is one of the backtest-profitable markets.

    Final gate before flagging a bet as value: even with a real-model prediction
    and a clean edge, only recommend markets the backtest showed positive yield
    over a large historical sample. Unknown labels (typos, new markets) return
    False so a future bug can't accidentally pass the gate.
    """
    market_key = _OUTCOME_LABEL_TO_BACKTEST_KEY.get(outcome_label)
    if market_key is None:
        return False
    return (league, market_key) in PROFITABLE_MARKETS

# Tolerance for the edge-cap comparison so floating-point arithmetic doesn't
# suppress edges that are exactly at the cap (e.g. 0.4 - 0.25 = 0.15000000000000002).
_EDGE_CAP_TOLERANCE = 1e-9


def assess_value(
    model_prob: float,
    decimal_odds: float,
    bookmaker: str,
    fair_implied_prob: float,
    value_threshold: float = 0.05,
    kelly_multiplier: float = 0.25,
    is_demo: bool = False,
    max_reasonable_edge: float = MAX_REASONABLE_EDGE_DEFAULT,
    allow_extreme_edge: bool = False,
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
        is_demo: True when the prediction came from the demo-mode fallback rather
                 than a real trained model. Demo picks are never flagged as value.
        max_reasonable_edge: Upper bound on plausible edge in a liquid Big-5
                             market (0.15 = 15pp). Edges meaningfully larger are
                             almost always a model error and are suppressed.
        allow_extreme_edge: Override to permit edges above ``max_reasonable_edge``
                            (e.g. for debugging or sanity-checking the signal).

    Raises:
        ValueError: If ``value_threshold`` exceeds ``max_reasonable_edge``,
            which would create an empty valid-edge window where no bet can
            ever flag. Caller misconfiguration; surface loudly.

    Note:
        ``kelly_stake_fraction`` is always populated even when ``is_value_bet``
        is False — the value lets the display table show a "what would the stake
        have been" hint without re-running the calc.
    """
    if value_threshold > max_reasonable_edge:
        raise ValueError(
            f"value_threshold ({value_threshold}) cannot exceed "
            f"max_reasonable_edge ({max_reasonable_edge}); the valid-edge window "
            f"would be empty and no bet could ever flag."
        )

    value_pct = model_prob - fair_implied_prob
    full_kelly = kelly_fraction(model_prob, decimal_odds)
    stake_fraction = full_kelly * kelly_multiplier

    edge_exceeds_cap = (value_pct - max_reasonable_edge) > _EDGE_CAP_TOLERANCE
    is_value_bet = (
        value_pct >= value_threshold
        and not is_demo
        and (allow_extreme_edge or not edge_exceeds_cap)
    )

    return ValueAssessment(
        model_probability=model_prob,
        bookmaker_odds=decimal_odds,
        bookmaker=bookmaker,
        fair_implied_probability=fair_implied_prob,
        value_pct=value_pct,
        kelly_stake_fraction=stake_fraction,
        is_value_bet=is_value_bet,
    )


def mask_demo_display_fields(row: dict) -> dict:
    """Mask model-derived display fields on a fixtures-table row.

    Demo predictions are hash-based pseudo-probabilities, not real model output.
    Showing them alongside genuine predictions invites the operator to act on
    fake numbers — so we prefix the Bet column with a warning and replace
    Model %, Edge, and Kelly stake with em-dashes. Fair % (the bookmaker's
    margin-stripped implied probability) is untouched because it does not
    come from the model.

    No-op when ``row["_is_demo"]`` is falsy or absent. Mutates and returns
    the same dict so it composes naturally in pandas pipelines.
    """
    if not row.get("_is_demo"):
        return row
    row["Bet"] = f"⚠ Demo • {row['Bet']}"
    row["Model %"] = "—"
    row["Edge"] = "—"
    row["Kelly stake"] = "—"
    return row
