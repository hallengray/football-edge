"""Tests for src/predictions.py — multi-league routing."""

from __future__ import annotations

import json
import math
import pickle
from unittest.mock import MagicMock

import pandas as pd

from src.predictions import (
    BIG_5_LEAGUES,
    Models,
    _demo_prediction,
    load_models,
    predict_fixture,
)


# ─── Phase 1: demo-mode regression ───────────────────────────────────


def test_demo_prediction_returns_demo_flag() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    assert pred.is_demo is True


def test_demo_prediction_probabilities_sum_to_one() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_demo_prediction_is_deterministic() -> None:
    a = _demo_prediction("Arsenal", "Chelsea")
    b = _demo_prediction("Arsenal", "Chelsea")
    assert a.p_home == b.p_home


def test_demo_prediction_swap_changes_output() -> None:
    a = _demo_prediction("Arsenal", "Chelsea")
    b = _demo_prediction("Chelsea", "Arsenal")
    assert (a.p_home, a.p_draw, a.p_away) != (b.p_home, b.p_draw, b.p_away)


def test_demo_prediction_includes_over_2_5() -> None:
    pred = _demo_prediction("Arsenal", "Chelsea")
    assert pred.p_over_2_5 is not None
    assert 0.0 <= pred.p_over_2_5 <= 1.0


# ─── Phase 2: Models container ───────────────────────────────────────


def test_models_is_ready_false_when_bettors_dict_empty() -> None:
    m = Models(bettors={}, fixtures_df=pd.DataFrame(), backtest={})
    assert m.is_ready is False


def test_models_is_ready_false_when_only_some_leagues_loaded() -> None:
    m = Models(
        bettors={"epl": object(), "laliga": object()},  # only 2 of 5
        fixtures_df=pd.DataFrame(),
        backtest={"trained_at": "x"},
    )
    assert m.is_ready is False


def test_models_is_ready_true_when_all_5_leagues_plus_fixtures_plus_backtest() -> None:
    m = Models(
        bettors={league: object() for league in BIG_5_LEAGUES},
        fixtures_df=pd.DataFrame(),
        backtest={"trained_at": "x"},
    )
    assert m.is_ready is True


def test_load_models_returns_unready_when_files_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.predictions.MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        "src.predictions.BETTOR_PATHS",
        {league: tmp_path / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES},
    )
    monkeypatch.setattr("src.predictions.FIXTURES_PATH", tmp_path / "fixtures_data.parquet")
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", tmp_path / "backtest.json")

    models = load_models()
    assert models.is_ready is False


def test_load_models_returns_ready_when_all_artifacts_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.predictions.MODELS_DIR", tmp_path)
    bettor_paths = {league: tmp_path / f"{league}_bettor.pkl" for league in BIG_5_LEAGUES}
    fixtures_path = tmp_path / "fixtures_data.parquet"
    backtest_path = tmp_path / "backtest.json"
    monkeypatch.setattr("src.predictions.BETTOR_PATHS", bettor_paths)
    monkeypatch.setattr("src.predictions.FIXTURES_PATH", fixtures_path)
    monkeypatch.setattr("src.predictions.BACKTEST_PATH", backtest_path)

    for league, path in bettor_paths.items():
        with path.open("wb") as f:
            pickle.dump({"fake": league}, f)
    pd.DataFrame({"a": [1]}).to_parquet(fixtures_path)
    backtest_path.write_text(json.dumps({"trained_at": "2026-05-06T00:00:00Z"}))

    models = load_models()
    assert models.is_ready is True
    assert set(models.bettors.keys()) == set(BIG_5_LEAGUES)


# ─── Phase 3: predict_fixture multi-league routing ───────────────────


def test_predict_fixture_demo_when_models_unready() -> None:
    models = Models(bettors={}, fixtures_df=pd.DataFrame(), backtest=None)
    pred = predict_fixture(models, "epl", "Arsenal", "Chelsea")
    assert pred.is_demo is True


def test_predict_fixture_demo_when_unsupported_league() -> None:
    models = Models(
        bettors={league: MagicMock() for league in BIG_5_LEAGUES},
        fixtures_df=pd.DataFrame(),
        backtest={"x": 1},
    )
    pred = predict_fixture(models, "championship", "Arsenal", "Chelsea")  # not big-5
    assert pred.is_demo is True


def test_predict_fixture_routes_to_correct_league_bettor() -> None:
    """A La Liga fixture calls la_liga's bettor, not EPL's."""
    bettors = {league: MagicMock() for league in BIG_5_LEAGUES}
    bettors["laliga"].predict_proba.return_value = [
        # 5 markets x 2 columns each (P=0, P=1)
        [[0.3, 0.7]],
        [[0.7, 0.3]],
        [[0.6, 0.4]],
        [[0.4, 0.6]],
        [[0.6, 0.4]],
    ]
    fixtures_df = pd.DataFrame(
        {"league": ["Spain"], "home_team": ["Real Madrid"], "away_team": ["Barcelona"]}
    )
    models = Models(bettors=bettors, fixtures_df=fixtures_df, backtest={"x": 1})

    pred = predict_fixture(models, "laliga", "Real Madrid", "Barcelona")

    bettors["laliga"].predict_proba.assert_called_once()
    bettors["epl"].predict_proba.assert_not_called()
    assert pred.is_demo is False


def test_predict_fixture_demo_when_fixture_row_not_found() -> None:
    bettors = {league: MagicMock() for league in BIG_5_LEAGUES}
    fixtures_df = pd.DataFrame(
        {"league": ["England"], "home_team": ["Liverpool"], "away_team": ["Everton"]}
    )
    models = Models(bettors=bettors, fixtures_df=fixtures_df, backtest={"x": 1})

    pred = predict_fixture(models, "epl", "Arsenal", "Chelsea")
    assert pred.is_demo is True
    bettors["epl"].predict_proba.assert_not_called()
