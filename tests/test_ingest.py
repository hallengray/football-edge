"""Tests for ingest modules. HTTP calls are mocked via requests-mock."""

from __future__ import annotations

import pytest
import requests_mock

from src.ingest.football_data import (
    TRAINING_URL_TEMPLATE,
    download_fixtures_data,
    download_training_data,
)


def test_training_url_template_has_expected_placeholders() -> None:
    url = TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2024)
    assert url == (
        "https://raw.githubusercontent.com/georgedouzas/sports-betting/"
        "data/data/soccer/modelling/England_1_2024.csv"
    )


def test_download_training_data_concatenates_seasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_2023 = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n2023-08-15,Arsenal,Chelsea,2,1\n"
    csv_2024 = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n2024-08-15,Liverpool,Everton,3,0\n"

    with requests_mock.Mocker() as m:
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2023), text=csv_2023)
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2024), text=csv_2024)

        df = download_training_data(leagues=["England"], years=[2023, 2024])

    assert len(df) == 2
    assert set(df["HomeTeam"]) == {"Arsenal", "Liverpool"}


def test_download_training_data_uses_cache_on_second_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_text = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n2023-08-15,Arsenal,Chelsea,2,1\n"

    with requests_mock.Mocker() as m:
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2023), text=csv_text)
        download_training_data(leagues=["England"], years=[2023])
        assert m.call_count == 1

        # Second call — should hit cache, no new HTTP request
        download_training_data(leagues=["England"], years=[2023])
        assert m.call_count == 1


def test_download_training_data_skips_failed_seasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_text = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n2024-08-15,Liverpool,Everton,3,0\n"

    with requests_mock.Mocker() as m:
        m.get(
            TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2023), status_code=404
        )
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2024), text=csv_text)

        # 1 of 2 seasons failed — that's 50%, exactly at threshold; should still succeed
        df = download_training_data(leagues=["England"], years=[2023, 2024])
    assert len(df) == 1


def test_download_training_data_raises_when_more_than_half_seasons_fail(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    with requests_mock.Mocker() as m:
        # All three seasons fail
        for year in [2022, 2023, 2024]:
            m.get(
                TRAINING_URL_TEMPLATE.format(league="England", division=1, year=year),
                status_code=500,
            )

        with pytest.raises(RuntimeError, match="too many seasons failed"):
            download_training_data(leagues=["England"], years=[2022, 2023, 2024])


def test_download_fixtures_data_always_refetches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_text = "Date,HomeTeam,AwayTeam\n2026-05-10,Arsenal,Chelsea\n"

    with requests_mock.Mocker() as m:
        m.get(
            "https://raw.githubusercontent.com/georgedouzas/sports-betting/"
            "data/data/soccer/modelling/fixtures.csv",
            text=csv_text,
        )
        download_fixtures_data()
        download_fixtures_data()
        # Both calls should hit the network — fixtures are always fresh
        assert m.call_count == 2
