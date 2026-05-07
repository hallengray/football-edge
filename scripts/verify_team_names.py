"""Helper: print team names from football-data and Understat that aren't yet in REGISTRY.

Usage:
    uv run python scripts/verify_team_names.py

Prints two lists: unmapped football-data names, unmapped Understat names. The
operator copies these into src/team_names.REGISTRY (with the right canonical name
and the entries for all three sources) and re-runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.ingest.football_data import download_training_data
    from src.ingest.understat import fetch_xg_data
    from src.team_names import REGISTRY

    training_years = list(range(2018, 2026))
    fd_leagues = ["England", "Spain", "Italy", "Germany", "France"]
    understat_leagues = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]

    print("Downloading football-data CSVs (cached after first run)...")
    matches = download_training_data(leagues=fd_leagues, years=training_years)
    fd_names = set(matches["home_team"].dropna().unique()) | set(
        matches["away_team"].dropna().unique()
    )

    print("Scraping Understat (cached after first run, ~60s on first run)...")
    xg = fetch_xg_data(leagues=understat_leagues, years=training_years)
    if xg.empty:
        understat_names: set[str] = set()
    else:
        understat_names = set(xg["home_team"].dropna().unique()) | set(
            xg["away_team"].dropna().unique()
        )

    mapped_fd = {t.football_data for t in REGISTRY.values()}
    mapped_understat = {t.understat for t in REGISTRY.values()}

    missing_fd = sorted(fd_names - mapped_fd)
    missing_understat = sorted(understat_names - mapped_understat)

    print(f"\n--- Unmapped football-data names ({len(missing_fd)}) ---")
    for name in missing_fd:
        print(f"  {name!r}")

    print(f"\n--- Unmapped Understat names ({len(missing_understat)}) ---")
    for name in missing_understat:
        print(f"  {name!r}")

    if not missing_fd and not missing_understat:
        print("\nAll team names are mapped.")


if __name__ == "__main__":
    main()
