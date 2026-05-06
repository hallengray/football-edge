"""Tests for training pipeline plumbing.

Only the atomic-write helper is unit-testable — actual training requires
the library + a public dataset and is verified via Task 13's manual smoke test.
"""

from __future__ import annotations

import pickle

import pytest

from scripts.train_model import write_artifacts_atomically


def test_atomic_write_creates_all_three_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BETTOR_PATH", tmp_path / "epl_bettor.pkl")
    monkeypatch.setattr("scripts.train_model.LOADER_PATH", tmp_path / "epl_loader.pkl")
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")

    write_artifacts_atomically(
        bettor={"fake": "bettor"},
        loader={"fake": "loader"},
        summary={"trained_at": "2026-05-05T00:00:00Z"},
    )

    assert (tmp_path / "epl_bettor.pkl").exists()
    assert (tmp_path / "epl_loader.pkl").exists()
    assert (tmp_path / "backtest.json").exists()
    # Round-trip check
    with (tmp_path / "epl_bettor.pkl").open("rb") as f:
        assert pickle.load(f) == {"fake": "bettor"}


def test_atomic_write_no_partial_state_when_pickle_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BETTOR_PATH", tmp_path / "epl_bettor.pkl")
    monkeypatch.setattr("scripts.train_model.LOADER_PATH", tmp_path / "epl_loader.pkl")
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")

    # An object that throws on pickle.dump — TypeError is what pickle raises for
    # unpicklable objects like local lambdas
    class Unpicklable:
        def __reduce__(self):
            raise TypeError("nope")

    with pytest.raises(TypeError):
        write_artifacts_atomically(
            bettor={"ok": "bettor"},
            loader=Unpicklable(),  # this will fail to pickle
            summary={"trained_at": "2026-05-05T00:00:00Z"},
        )

    # No final artifacts should exist — only orphaned .tmp files should be cleaned up
    assert not (tmp_path / "epl_bettor.pkl").exists()
    assert not (tmp_path / "epl_loader.pkl").exists()
    assert not (tmp_path / "backtest.json").exists()
    # No .tmp files left behind either
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_overwrites_existing(tmp_path, monkeypatch) -> None:
    """A successful re-train should replace previous artifacts."""
    monkeypatch.setattr("scripts.train_model.MODELS_DIR", tmp_path)
    monkeypatch.setattr("scripts.train_model.BETTOR_PATH", tmp_path / "epl_bettor.pkl")
    monkeypatch.setattr("scripts.train_model.LOADER_PATH", tmp_path / "epl_loader.pkl")
    monkeypatch.setattr("scripts.train_model.BACKTEST_PATH", tmp_path / "backtest.json")

    # Pre-existing old artifacts
    (tmp_path / "epl_bettor.pkl").write_bytes(b"old bettor")
    (tmp_path / "epl_loader.pkl").write_bytes(b"old loader")
    (tmp_path / "backtest.json").write_text("{}")

    write_artifacts_atomically(
        bettor={"new": "bettor"},
        loader={"new": "loader"},
        summary={"trained_at": "new"},
    )

    with (tmp_path / "epl_bettor.pkl").open("rb") as f:
        assert pickle.load(f) == {"new": "bettor"}
