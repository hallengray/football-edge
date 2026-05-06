"""Engineered feature pipeline.

Top-level `compute_features(matches_df, xg_df)` orchestrates: team-name
normalisation -> xG merge -> rest -> form -> rolling goals -> xG features ->
strength of schedule. Each helper is a pure function (input DataFrame,
output DataFrame with new columns) and is unit-testable in isolation.

Critical invariant: every rolling feature uses only matches BEFORE the current
match's date. Pandas idiom: groupby + sorted dates + shifted rolling. The
test suite includes a leakage-guard test that constructs synthetic data and
asserts no current-match data leaks into past-match features.
"""

from __future__ import annotations

import pandas as pd

REST_DEFAULT_DAYS = 7
REST_MAX_CAP_DAYS = 14
WINDOW_DEFAULT = 5


def _add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `home_rest_days`, `away_rest_days`, `home_matches_last_14d`, `away_matches_last_14d`.

    Rest days = min(days since this team's most recent prior match, 14).
    First-ever match for a team gets the default of 7 days.

    matches_last_14d = count of this team's matches on dates within (current_date - 14, current_date].
    A team's first match counts itself, so the minimum is 1.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Build a long-form view: one row per (match_id, side, team)
    long = (
        pd.concat(
            [
                df[["date"]].assign(team=df["home_team"], match_id=df.index, side="home"),
                df[["date"]].assign(team=df["away_team"], match_id=df.index, side="away"),
            ],
            ignore_index=True,
        )
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )

    # Rest days = days since the team's previous match
    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days.clip(upper=REST_MAX_CAP_DAYS)
    long["rest_days"] = long["rest_days"].fillna(REST_DEFAULT_DAYS).astype(int)

    # matches_last_14d: count of team's matches with date in (current - 14, current]
    def _count_recent(group: pd.DataFrame) -> pd.Series:
        # For each row, count rows in this group with date within last 14 days inclusive
        counts = []
        for d in group["date"]:
            mask = (group["date"] > d - pd.Timedelta(days=14)) & (group["date"] <= d)
            counts.append(int(mask.sum()))
        return pd.Series(counts, index=group.index)

    long["matches_last_14d"] = long.groupby("team", group_keys=False).apply(
        _count_recent, include_groups=False
    )

    # Pivot back to per-match columns for home and away sides
    home = long[long["side"] == "home"].set_index("match_id")[["rest_days", "matches_last_14d"]]
    home = home.add_prefix("home_")
    away = long[long["side"] == "away"].set_index("match_id")[["rest_days", "matches_last_14d"]]
    away = away.add_prefix("away_")

    df = df.join(home).join(away)
    return df


def _add_form_features(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add `home_form_{wins,draws,losses}` and `away_form_{wins,draws,losses}`.

    Counts results in the team's last `window` matches BEFORE the current match,
    across both home and away appearances. A team's first match gets all zeros.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Build long-form view, computing this match's outcome from the team's perspective
    long_home = df[["date", "home_team", "FTHG", "FTAG"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"
    long_home["team_goals"] = long_home["FTHG"]
    long_home["opp_goals"] = long_home["FTAG"]

    long_away = df[["date", "away_team", "FTHG", "FTAG"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"
    long_away["team_goals"] = long_away["FTAG"]
    long_away["opp_goals"] = long_away["FTHG"]

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    long["is_win"] = (long["team_goals"] > long["opp_goals"]).astype(int)
    long["is_draw"] = (long["team_goals"] == long["opp_goals"]).astype(int)
    long["is_loss"] = (long["team_goals"] < long["opp_goals"]).astype(int)

    # Rolling sum of last `window` matches, EXCLUDING the current match
    grouped = long.groupby("team", group_keys=False)
    long["form_wins"] = (
        grouped["is_win"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
    )
    long["form_draws"] = (
        grouped["is_draw"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        .fillna(0)
    )
    long["form_losses"] = (
        grouped["is_loss"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        .fillna(0)
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["form_wins", "form_draws", "form_losses"]]
        .add_prefix("home_")
        .astype(int)
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["form_wins", "form_draws", "form_losses"]]
        .add_prefix("away_")
        .astype(int)
    )

    df = df.join(home).join(away)
    return df


def _merge_xg(matches_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join xG data onto matches by (date, home_team, away_team).

    Missing xG -> NaN. Caller's responsibility to handle (typically via
    SimpleImputer at fit time).
    """
    if xg_df.empty:
        out = matches_df.copy()
        out["home_xg"] = pd.NA
        out["away_xg"] = pd.NA
        return out

    matches = matches_df.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    xg = xg_df.copy()
    xg["date"] = pd.to_datetime(xg["date"])
    xg = xg[["date", "home_team", "away_team", "home_xg", "away_xg"]]

    return matches.merge(xg, on=["date", "home_team", "away_team"], how="left")


def _add_rolling_goals(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add `home_goals_scored_last_5`, `home_goals_conceded_last_5`, and away counterparts.

    Uses only matches BEFORE the current row's date (per the leakage guard).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    long_home = df[["date", "home_team", "FTHG", "FTAG"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"
    long_home["scored"] = long_home["FTHG"]
    long_home["conceded"] = long_home["FTAG"]

    long_away = df[["date", "away_team", "FTHG", "FTAG"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"
    long_away["scored"] = long_away["FTAG"]
    long_away["conceded"] = long_away["FTHG"]

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    grouped = long.groupby("team", group_keys=False)
    long["scored_last_5"] = grouped["scored"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long["conceded_last_5"] = grouped["conceded"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["scored_last_5", "conceded_last_5"]]
        .rename(
            columns={
                "scored_last_5": "home_goals_scored_last_5",
                "conceded_last_5": "home_goals_conceded_last_5",
            }
        )
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["scored_last_5", "conceded_last_5"]]
        .rename(
            columns={
                "scored_last_5": "away_goals_scored_last_5",
                "conceded_last_5": "away_goals_conceded_last_5",
            }
        )
    )

    df = df.join(home).join(away)
    return df


def _add_xg_features(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add rolling xG/xGA averages plus xG-vs-actual delta (overperformance signal).

    Requires `home_xg` and `away_xg` columns (from _merge_xg).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    long_home = df[["date", "home_team", "home_xg", "away_xg", "FTHG", "FTAG"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"
    long_home["xg"] = long_home["home_xg"]
    long_home["xga"] = long_home["away_xg"]
    long_home["scored"] = long_home["FTHG"]

    long_away = df[["date", "away_team", "home_xg", "away_xg", "FTHG", "FTAG"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"
    long_away["xg"] = long_away["away_xg"]
    long_away["xga"] = long_away["home_xg"]
    long_away["scored"] = long_away["FTAG"]

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    long["scored_minus_xg"] = long["scored"] - long["xg"]

    grouped = long.groupby("team", group_keys=False)
    long["xg_last_5"] = grouped["xg"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long["xga_last_5"] = grouped["xga"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long["xg_minus_actual_last_5"] = grouped["scored_minus_xg"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["xg_last_5", "xga_last_5", "xg_minus_actual_last_5"]]
        .add_prefix("home_")
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["xg_last_5", "xga_last_5", "xg_minus_actual_last_5"]]
        .add_prefix("away_")
    )

    df = df.join(home).join(away)
    return df
