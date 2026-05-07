"""Tests for ingest modules. HTTP calls are mocked via requests-mock."""

from __future__ import annotations

import pytest
import requests_mock

from src.ingest.football_data import (
    TRAINING_URL_TEMPLATE,
    download_fixtures_data,
    download_training_data,
)
from src.ingest.understat import (
    LEAGUE_DATA_URL_TEMPLATE,
    fetch_league_xg,
    fetch_xg_data,
)


def test_training_url_template_has_expected_placeholders() -> None:
    url = TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2024)
    assert url == (
        "https://raw.githubusercontent.com/georgedouzas/sports-betting/"
        "data/data/soccer/modelling/England_1_2024.csv"
    )


# The sports-betting GitHub mirror serves snake_case columns; ingest normalises
# them to the legacy football-data schema. These fixtures match the real upstream.
MIRROR_HEADER = (
    "date,home_team,away_team,"
    "target__home_team__full_time_goals,target__away_team__full_time_goals,"
    "target__home_team__shots_on_target,target__away_team__shots_on_target,"
    "odds__market_average__home_win__full_time_goals,"
    "odds__market_average__draw__full_time_goals,"
    "odds__market_average__away_win__full_time_goals,"
    "odds__market_average__over_2.5__full_time_goals,"
    "odds__market_average__under_2.5__full_time_goals"
)


def test_download_training_data_concatenates_seasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_2023 = f"{MIRROR_HEADER}\n2023-08-15,Arsenal,Chelsea,2,1,5,3,2.10,3.40,3.60,1.95,1.95\n"
    csv_2024 = f"{MIRROR_HEADER}\n2024-08-15,Liverpool,Everton,3,0,8,2,1.50,4.20,7.00,1.80,2.05\n"

    with requests_mock.Mocker() as m:
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2023), text=csv_2023)
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2024), text=csv_2024)

        df = download_training_data(leagues=["England"], years=[2023, 2024])

    assert len(df) == 2
    assert set(df["home_team"]) == {"Arsenal", "Liverpool"}
    # Mirror columns must be normalised to legacy schema before downstream code sees them.
    assert "FTHG" in df.columns
    assert "FTAG" in df.columns
    assert "HST" in df.columns
    assert "AST" in df.columns
    assert "AvgH" in df.columns
    assert "AvgOver2.5" in df.columns
    assert "AvgUnder2.5" in df.columns
    assert "target__home_team__full_time_goals" not in df.columns
    assert "target__home_team__shots_on_target" not in df.columns


def test_download_training_data_uses_cache_on_second_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_text = f"{MIRROR_HEADER}\n2023-08-15,Arsenal,Chelsea,2,1,5,3,2.10,3.40,3.60,1.95,1.95\n"

    with requests_mock.Mocker() as m:
        m.get(TRAINING_URL_TEMPLATE.format(league="England", division=1, year=2023), text=csv_text)
        download_training_data(leagues=["England"], years=[2023])
        assert m.call_count == 1

        # Second call — should hit cache, no new HTTP request, and still return normalised columns.
        df = download_training_data(leagues=["England"], years=[2023])
        assert m.call_count == 1
        assert "FTHG" in df.columns


def test_download_training_data_skips_failed_seasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.football_data.CACHE_DIR", tmp_path)

    csv_text = f"{MIRROR_HEADER}\n2024-08-15,Liverpool,Everton,3,0,8,2,1.50,4.20,7.00,1.80,2.05\n"

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


# ─── Understat ingest tests ──────────────────────────────────────────


# The Understat XHR endpoint returns a JSON object whose `dates` array has one
# entry per match with the same shape used elsewhere in the pipeline.
SAMPLE_UNDERSTAT_JSON: dict = {
    "teams": {},
    "players": [],
    "dates": [
        {
            "id": "1",
            "isResult": True,
            "h": {"id": "9", "title": "Arsenal", "short_title": "ARS"},
            "a": {"id": "10", "title": "Chelsea", "short_title": "CHE"},
            "goals": {"h": "2", "a": "1"},
            "xG": {"h": "1.85", "a": "1.20"},
            "datetime": "2024-08-15 16:30:00",
        }
    ],
}


def test_understat_url_template() -> None:
    assert (
        LEAGUE_DATA_URL_TEMPLATE.format(league="EPL", year=2024)
        == "https://understat.com/getLeagueData/EPL/2024"
    )


def test_fetch_league_xg_parses_endpoint_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)  # speed up tests

    with requests_mock.Mocker() as m:
        m.get(
            LEAGUE_DATA_URL_TEMPLATE.format(league="EPL", year=2024),
            json=SAMPLE_UNDERSTAT_JSON,
        )
        df = fetch_league_xg("EPL", 2024)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert row["home_xg"] == pytest.approx(1.85)
    assert row["away_xg"] == pytest.approx(1.20)


def test_fetch_league_xg_handles_404_gracefully(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)

    with requests_mock.Mocker() as m:
        m.get(LEAGUE_DATA_URL_TEMPLATE.format(league="EPL", year=1999), status_code=404)
        df = fetch_league_xg("EPL", 1999)

    assert df.empty


def test_fetch_league_xg_handles_malformed_json_gracefully(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)

    with requests_mock.Mocker() as m:
        m.get(LEAGUE_DATA_URL_TEMPLATE.format(league="EPL", year=2024), text="not json at all")
        df = fetch_league_xg("EPL", 2024)

    assert df.empty


def test_fetch_league_xg_uses_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)

    with requests_mock.Mocker() as m:
        m.get(
            LEAGUE_DATA_URL_TEMPLATE.format(league="EPL", year=2024),
            json=SAMPLE_UNDERSTAT_JSON,
        )
        fetch_league_xg("EPL", 2024)
        assert m.call_count == 1
        # Second call should read from cache parquet, not network
        fetch_league_xg("EPL", 2024)
        assert m.call_count == 1


def test_fetch_xg_data_concatenates_leagues(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.ingest.understat.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.understat.SCRAPE_SLEEP", 0)

    with requests_mock.Mocker() as m:
        m.get(
            LEAGUE_DATA_URL_TEMPLATE.format(league="EPL", year=2024),
            json=SAMPLE_UNDERSTAT_JSON,
        )
        m.get(
            LEAGUE_DATA_URL_TEMPLATE.format(league="La_liga", year=2024),
            json=SAMPLE_UNDERSTAT_JSON,
        )

        df = fetch_xg_data(leagues=["EPL", "La_liga"], years=[2024])

    assert len(df) == 2
    assert set(df["league"]) == {"EPL", "La_liga"}
