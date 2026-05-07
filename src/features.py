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


def _add_shooting_features(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add `home_sot_last_5`, `home_sota_last_5`, and away counterparts.

    Rolling shots-on-target for the team and against the team over the last
    `window` matches. Always-on baseline for shot-quality signal -- relies only
    on football-data's HST/AST, not on Understat. When xG data is also present,
    `_add_xg_features` runs in parallel and adds richer features.

    Uses only matches BEFORE the current row's date (per the leakage guard).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    long_home = df[["date", "home_team", "HST", "AST"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"
    long_home["sot"] = long_home["HST"]
    long_home["sota"] = long_home["AST"]

    long_away = df[["date", "away_team", "HST", "AST"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"
    long_away["sot"] = long_away["AST"]
    long_away["sota"] = long_away["HST"]

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    grouped = long.groupby("team", group_keys=False)
    long["sot_last_5"] = grouped["sot"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long["sota_last_5"] = grouped["sota"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["sot_last_5", "sota_last_5"]]
        .rename(
            columns={
                "sot_last_5": "home_sot_last_5",
                "sota_last_5": "home_sota_last_5",
            }
        )
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["sot_last_5", "sota_last_5"]]
        .rename(
            columns={
                "sot_last_5": "away_sot_last_5",
                "sota_last_5": "away_sota_last_5",
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


LEAGUE_SIZE_DEFAULT: dict[str, int] = {
    "England": 20,
    "Spain": 20,
    "Italy": 20,
    "Germany": 18,
    "France": 18,
}


def _normalise_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert football-data team names to canonical form.

    Drops rows where either team is unmapped (with a logged warning naming
    the unmapped team -- the caller adds it to src/team_names.REGISTRY and re-runs).
    """
    from src.team_names import to_canonical

    df = df.copy()
    keep_idx: list[int] = []
    for idx, row in df.iterrows():
        try:
            row["home_team"] = to_canonical(row["home_team"], source="football_data")
            row["away_team"] = to_canonical(row["away_team"], source="football_data")
            df.at[idx, "home_team"] = row["home_team"]
            df.at[idx, "away_team"] = row["away_team"]
            keep_idx.append(idx)
        except KeyError:
            # Will be re-raised by the caller (training script) once verification
            # is added; here we let it propagate so unmapped teams don't silently drop.
            raise
    return df.loc[keep_idx].reset_index(drop=True)


def _add_strength_of_schedule(df: pd.DataFrame, window: int = WINDOW_DEFAULT) -> pd.DataFrame:
    """Add `home_opp_strength_last_5`, `away_opp_strength_last_5`.

    Strength = average prior-season league position of the team's last `window`
    opponents. We approximate prior-season position from a rolling computation
    of points-per-game across all matches a team played in the previous year:
    teams with higher PPG get lower (better) position numbers.

    Promoted teams (no prior-year data) get league_size - 2 (relegation-adjacent).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["league", "year", "date"]).reset_index(drop=True)

    # Compute prior-season PPG per team. If FTHG/FTAG aren't in the input
    # (e.g. when scoring upcoming fixtures or a degenerate test), skip this step
    # and let every team fall back to the promoted-team default below.
    points_table = []
    has_results = "FTHG" in df.columns and "FTAG" in df.columns
    if has_results:
        for (league, year), group in df.groupby(["league", "year"]):
            teams: dict[str, int] = {}
            games: dict[str, int] = {}
            for _, m in group.iterrows():
                for team_col, scored_col, conceded_col in [
                    ("home_team", "FTHG", "FTAG"),
                    ("away_team", "FTAG", "FTHG"),
                ]:
                    team = m[team_col]
                    teams.setdefault(team, 0)
                    games.setdefault(team, 0)
                    games[team] += 1
                    if m[scored_col] > m[conceded_col]:
                        teams[team] += 3
                    elif m[scored_col] == m[conceded_col]:
                        teams[team] += 1
            for team, pts in teams.items():
                ppg = pts / games[team] if games[team] else 0
                points_table.append({"league": league, "year": year, "team": team, "ppg": ppg})

    ppg_df = pd.DataFrame(points_table)

    # Convert PPG to "position" -- rank within league-year, descending
    # (higher PPG = lower number = better)
    if not ppg_df.empty:
        ppg_df["position"] = (
            ppg_df.groupby(["league", "year"])["ppg"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        # Use prior season's position to score current season's matches
        ppg_df["next_year"] = ppg_df["year"] + 1
    else:
        ppg_df["position"] = []
        ppg_df["next_year"] = []

    def _opp_position(league: str, year: int, team: str) -> float:
        if ppg_df.empty:
            return float(LEAGUE_SIZE_DEFAULT.get(league, 20) - 2)
        match = ppg_df[
            (ppg_df["league"] == league) & (ppg_df["next_year"] == year) & (ppg_df["team"] == team)
        ]
        if match.empty:
            return float(LEAGUE_SIZE_DEFAULT.get(league, 20) - 2)
        return float(match["position"].iloc[0])

    # Long-form view of opponents
    long_home = df[["date", "home_team", "away_team", "league", "year"]].copy()
    long_home["team"] = long_home["home_team"]
    long_home["opponent"] = long_home["away_team"]
    long_home["match_id"] = long_home.index
    long_home["side"] = "home"

    long_away = df[["date", "home_team", "away_team", "league", "year"]].copy()
    long_away["team"] = long_away["away_team"]
    long_away["opponent"] = long_away["home_team"]
    long_away["match_id"] = long_away.index
    long_away["side"] = "away"

    long = pd.concat([long_home, long_away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    long["opp_pos"] = long.apply(
        lambda r: _opp_position(r["league"], r["year"], r["opponent"]),
        axis=1,
    )

    grouped = long.groupby("team", group_keys=False)
    long["opp_strength"] = grouped["opp_pos"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home = (
        long[long["side"] == "home"]
        .set_index("match_id")[["opp_strength"]]
        .rename(columns={"opp_strength": "home_opp_strength_last_5"})
    )
    away = (
        long[long["side"] == "away"]
        .set_index("match_id")[["opp_strength"]]
        .rename(columns={"opp_strength": "away_opp_strength_last_5"})
    )

    # First-match defaults (no prior data -> league-size-2)
    home["home_opp_strength_last_5"] = home["home_opp_strength_last_5"].fillna(
        df["league"].map(LEAGUE_SIZE_DEFAULT).fillna(20) - 2
    )
    away["away_opp_strength_last_5"] = away["away_opp_strength_last_5"].fillna(
        df["league"].map(LEAGUE_SIZE_DEFAULT).fillna(20) - 2
    )

    df = df.join(home).join(away)
    return df


def compute_features(matches_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Top-level: returns matches_df with engineered feature columns added.

    Pipeline order:
        1. Normalise team names (football-data -> canonical)
        2. Merge xG by (date, home_team, away_team)
        3. Rest features
        4. Form features (last 5 W/D/L)
        5. Rolling goals (last 5 scored/conceded)
        6. Rolling shots-on-target (always-on baseline; uses football-data HST/AST)
        7. xG features (last 5 xG/xGA + over/underperformance; needs Understat)
        8. Strength of schedule (last 5 opponents' prior-season position)
    """
    df = _normalise_team_names(matches_df)
    df = _merge_xg(df, xg_df)
    df = _add_rest_features(df)
    df = _add_form_features(df, window=WINDOW_DEFAULT)
    df = _add_rolling_goals(df, window=WINDOW_DEFAULT)
    df = _add_shooting_features(df, window=WINDOW_DEFAULT)
    df = _add_xg_features(df, window=WINDOW_DEFAULT)
    df = _add_strength_of_schedule(df, window=WINDOW_DEFAULT)
    return df
