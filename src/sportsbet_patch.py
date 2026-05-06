"""Monkey-patch sports-betting==0.12.1 around two upstream bugs.

Bug 1 — `SoccerDataLoader._get_full_param_grid` scrapes GitHub's web UI HTML
to discover available season CSVs. GitHub redesigned its file-listing markup
in 2025 and the old `payload.tree.items` JSON shape no longer exists, so every
loader call raises `KeyError: 'tree'`.

Bug 2 — `SoccerDataLoader._get_data` parses CSV dates with `format='%d/%m/%Y'`
and falls back to `pd.to_datetime(..., infer_datetime_format=True)`. The CSVs
have since switched to ISO8601 (`2017-08-11`), and `infer_datetime_format` was
removed in pandas 2.2, so the fallback also crashes.

Workaround:
- For Bug 1: hit GitHub's REST API directly (stable JSON, public, rate-limited
  to 60 req/hour unauthenticated which is fine because lru_cache means we call
  it once per session).
- For Bug 2: parse with `format='mixed', dayfirst=True` which handles both the
  old UK format (e.g. `11/08/2017`) and ISO8601 (`2017-08-11`).

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
    """Replace SoccerDataLoader's broken methods with working versions."""
    global _PATCHED
    if _PATCHED:
        return

    import pandas as pd
    import requests
    from sklearn.model_selection import ParameterGrid
    from sportsbet.datasets._soccer import _data as _data_module
    from sportsbet.datasets._soccer._data import SoccerDataLoader
    from sportsbet.datasets._soccer._utils import _read_csv, _read_csvs

    # ─── Bug 1: _get_full_param_grid via REST API ────────────────────────

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

    # ─── Bug 2: _get_data with mixed-format date parsing ─────────────────

    def _get_data_with_robust_dates(self):
        urls = [_data_module.TRAINING_URL.format(**params) for params in self.param_grid_]
        training_data = pd.concat(_read_csvs(urls))
        training_data["fixtures"] = False
        fixtures_data = _read_csv(_data_module.FIXTURES_URL)
        fixtures_data["fixtures"] = True
        data = (
            pd.concat([training_data, fixtures_data]) if not fixtures_data.empty else training_data
        ).reset_index(drop=True)
        # Library's old %d/%m/%Y format breaks on the new ISO8601 CSVs; mixed-format
        # parsing with dayfirst=True handles both layouts. Cast to ns precision
        # because pandas defaults to us with format="mixed" but the library's
        # _validate_data strictly checks for datetime64[ns].
        data["date"] = pd.to_datetime(data["date"], format="mixed", dayfirst=True).astype(
            "datetime64[ns]"
        )
        return data

    SoccerDataLoader._get_data = _get_data_with_robust_dates

    _PATCHED = True
