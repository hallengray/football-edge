"""Train the EPL multi-output ClassifierBettor and save it to disk.

Single command produces three artifacts in models/:
    epl_bettor.pkl    — fitted ClassifierBettor (handles 5 markets)
    epl_loader.pkl    — primed SoccerDataLoader (needed at inference time
                        for extract_fixtures_data())
    backtest.json     — per-market backtest summary

Usage:
    uv run python scripts/train_model.py

Run time: typically 10-20 minutes on a laptop. Existing artifacts are only
overwritten if the entire run succeeds (atomic write).
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
BETTOR_PATH = MODELS_DIR / "epl_bettor.pkl"
LOADER_PATH = MODELS_DIR / "epl_loader.pkl"
BACKTEST_PATH = MODELS_DIR / "backtest.json"


def write_artifacts_atomically(
    bettor: Any,
    loader: Any,
    summary: dict,
) -> None:
    """Write all three artifacts to .tmp paths, then os.replace to final names.

    Either all three end up at their final paths, or none do. Cleans up .tmp
    files on any failure.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bettor_tmp = BETTOR_PATH.with_suffix(BETTOR_PATH.suffix + ".tmp")
    loader_tmp = LOADER_PATH.with_suffix(LOADER_PATH.suffix + ".tmp")
    backtest_tmp = BACKTEST_PATH.with_suffix(BACKTEST_PATH.suffix + ".tmp")
    tmps = [bettor_tmp, loader_tmp, backtest_tmp]

    try:
        with bettor_tmp.open("wb") as f:
            pickle.dump(bettor, f)
        with loader_tmp.open("wb") as f:
            pickle.dump(loader, f)
        with backtest_tmp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # All temp writes succeeded — promote atomically
        os.replace(bettor_tmp, BETTOR_PATH)
        os.replace(loader_tmp, LOADER_PATH)
        os.replace(backtest_tmp, BACKTEST_PATH)
    except BaseException:
        # Clean up any temp files left behind (don't mask the original exception)
        for tmp in tmps:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        raise


def run_backtest(bettor, X, Y, O) -> dict:  # noqa: E741 — capital O matches library convention
    raise NotImplementedError("Implemented in Task 8")


def main() -> None:
    raise NotImplementedError("Implemented in Task 9")


if __name__ == "__main__":
    main()
