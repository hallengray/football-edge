"""Monkey-patch sports-betting==0.12.1's broken GitHub web-scraping.

The library's `SoccerDataLoader._get_full_param_grid` scrapes GitHub's web UI HTML
to discover which season CSVs are available. GitHub redesigned its file-listing
markup in 2025 and the old `payload.tree.items` JSON shape no longer exists, so
the library raises `KeyError: 'tree'` on every `extract_train_data()` and
`extract_fixtures_data()` call.

Workaround: hit GitHub's REST API directly (stable JSON, public, rate-limited to
60 req/hour unauthenticated which is fine because lru_cache means we call it once).

Call `apply_patch()` once at the start of any process that constructs a
`SoccerDataLoader` (training or inference). Idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False
_GITHUB_API_URL = (
    "https://api.github.com/repos/georgedouzas/sports-betting/"
    "contents/data/soccer/modelling?ref=data"
)


def apply_patch() -> None:
    """Replace SoccerDataLoader._get_full_param_grid with a REST-API version."""
    global _PATCHED
    if _PATCHED:
        return

    import requests
    from sklearn.model_selection import ParameterGrid
    from sportsbet.datasets._soccer._data import SoccerDataLoader

    @classmethod  # type: ignore[misc]
    def _get_full_param_grid_via_api(cls):  # noqa: ARG001
        response = requests.get(_GITHUB_API_URL, timeout=30)
        response.raise_for_status()
        items = response.json()

        param_grid = []
        for item in items:
            name = item.get("name", "")
            if "fixtures.csv" in name or "_" not in name or not name.endswith(".csv"):
                continue
            try:
                league, division, year = name.replace(".csv", "").split("_")
                param_grid.append(
                    {
                        "league": [league.title() if league.lower() != "usa" else "USA"],
                        "division": [int(division)],
                        "year": [int(year)],
                    }
                )
            except ValueError:
                logger.debug(f"Skipping unrecognised filename: {name}")
                continue
        return ParameterGrid(param_grid)

    SoccerDataLoader._get_full_param_grid = _get_full_param_grid_via_api
    _PATCHED = True
