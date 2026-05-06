"""Tests for src/predictions.py.

This module is incrementally updated alongside the real-model implementation.
Phase 1 (this commit) — regression tests for the existing demo mode.
Phase 2 (Task 5) — adds tests for load_models() and Models dataclass.
Phase 3 (Task 6) — adds tests for the real branch and team-name fallback.
"""

from __future__ import annotations

import math

from src.predictions import _demo_prediction


def test_demo_prediction_returns_demo_flag() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    assert pred.is_demo is True


def test_demo_prediction_probabilities_sum_to_one() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_demo_prediction_is_deterministic() -> None:
    # Same input → same output (hash-based seeding)
    a = _demo_prediction("Arsenal", "Chelsea")
    b = _demo_prediction("Arsenal", "Chelsea")
    assert a.p_home == b.p_home
    assert a.p_draw == b.p_draw
    assert a.p_away == b.p_away


def test_demo_prediction_swap_changes_output() -> None:
    # home/away swap should NOT yield identical probabilities
    a = _demo_prediction("Arsenal", "Chelsea")
    b = _demo_prediction("Chelsea", "Arsenal")
    assert (a.p_home, a.p_draw, a.p_away) != (b.p_home, b.p_draw, b.p_away)


def test_demo_prediction_includes_over_2_5() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    assert pred.p_over_2_5 is not None
    assert 0.0 <= pred.p_over_2_5 <= 1.0
