"""Tests for src/predictions.py.

This module is incrementally updated alongside the real-model implementation.
Phase 1 (this commit) — regression tests for the existing demo mode.
Phase 2 (Task 5) — adds tests for load_models() and Models dataclass.
Phase 3 (Task 6) — adds tests for the real branch and team-name fallback.
"""

from __future__ import annotations

import json
import math
import pickle

from src.predictions import Models, _demo_prediction, load_models


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


# ─── Phase 2: Models dataclass and load_models() ───


def test_models_is_ready_false_when_all_none() -> None:
    m = Models(bettor=None, loader=None, backtest=None)
    assert m.is_ready is False


def test_models_is_ready_true_when_all_present() -> None:
    m = Models(bettor=object(), loader=object(), backtest={"trained_at": "x"})
    assert m.is_ready is True


def test_models_is_ready_false_when_one_missing() -> None:
    m = Models(bettor=object(), loader=None, backtest={"trained_at": "x"})
    assert m.is_ready is False


def test_load_models_returns_unready_when_files_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.predictions.BETTOR_PATH", tmp_path / "epl_bettor.pkl")
    monkeypatch.setattr("src.predictions.LOADER_PATH", tmp_path / "epl_loader.pkl")
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", tmp_path / "backtest.json")

    models = load_models()
    assert models.is_ready is False
    assert models.bettor is None
    assert models.loader is None
    assert models.backtest is None


def test_load_models_returns_ready_when_all_artifacts_present(tmp_path, monkeypatch) -> None:
    bettor_path = tmp_path / "epl_bettor.pkl"
    loader_path = tmp_path / "epl_loader.pkl"
    backtest_path = tmp_path / "backtest.json"
    monkeypatch.setattr("src.predictions.BETTOR_PATH", bettor_path)
    monkeypatch.setattr("src.predictions.LOADER_PATH", loader_path)
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", backtest_path)

    # Stand-in objects (the real types are only validated at use time)
    with bettor_path.open("wb") as f:
        pickle.dump({"fake": "bettor"}, f)
    with loader_path.open("wb") as f:
        pickle.dump({"fake": "loader"}, f)
    backtest_path.write_text(json.dumps({"trained_at": "2026-05-05T00:00:00Z"}))

    models = load_models()
    assert models.is_ready is True
    assert models.bettor == {"fake": "bettor"}
    assert models.loader == {"fake": "loader"}
    assert models.backtest == {"trained_at": "2026-05-05T00:00:00Z"}


def test_load_models_handles_corrupt_backtest_json(tmp_path, monkeypatch) -> None:
    bettor_path = tmp_path / "epl_bettor.pkl"
    loader_path = tmp_path / "epl_loader.pkl"
    backtest_path = tmp_path / "backtest.json"
    monkeypatch.setattr("src.predictions.BETTOR_PATH", bettor_path)
    monkeypatch.setattr("src.predictions.LOADER_PATH", loader_path)
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", backtest_path)

    with bettor_path.open("wb") as f:
        pickle.dump({"fake": "bettor"}, f)
    with loader_path.open("wb") as f:
        pickle.dump({"fake": "loader"}, f)
    backtest_path.write_text("{not valid json")

    models = load_models()
    assert models.is_ready is False
    assert models.backtest is None
