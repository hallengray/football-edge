# v3 AI Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-powered "AI picks" panel above the value-bets table that ranks the top 10 bets and explains each using only the data already shown on the dashboard.

**Architecture:** New module `src/ai_explainer.py` posts to OpenRouter's OpenAI-compatible endpoint, parses structured JSON, returns an `ExplainerResult` dataclass with picks and a human-readable error string. `app.py` adds a button + render helper that calls this module on click. No new deps — uses existing `requests`. All 6 tests use `requests-mock` to stub the OpenRouter call.

**Tech Stack:** Python 3.11, Streamlit, pandas, requests, requests-mock (test), ruff, pytest, uv. OpenRouter free tier (default `google/gemini-2.5-flash:free`).

**Spec:** `docs/superpowers/specs/2026-05-07-v3-ai-explainer-design.md` (committed at `7ec4d25`)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/ai_explainer.py` | NEW | Pure logic: dataclasses, prompt construction, OpenRouter POST, JSON parsing, pick validation. No Streamlit imports. ~120 lines. |
| `tests/test_ai_explainer.py` | NEW | Six unit tests, all using `requests-mock` to stub the OpenRouter endpoint. ~150 lines. |
| `app.py` | MODIFY | Insert button between value-bets header and dataframe; add `_render_ai_picks` helper. ~40 added lines. |

The module split keeps the prompt + HTTP logic pure (testable without Streamlit) and the rendering Streamlit-only (manually verified). The two never share state — the function returns a value and Streamlit renders it.

---

## Task 1: Module skeleton + API-key-missing path

**Files:**
- Create: `src/ai_explainer.py`
- Test: `tests/test_ai_explainer.py`

The simplest failing test — no HTTP call yet. Locks in the dataclass shape and function signature.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ai_explainer.py`:

```python
"""Tests for the AI explainer. All HTTP calls are mocked via requests-mock."""

from __future__ import annotations

import pandas as pd
import pytest
import requests_mock

from src.ai_explainer import (
    OPENROUTER_URL,
    ExplainerResult,
    Pick,
    explain_top_picks,
)


# Sample inputs reused across tests
SAMPLE_VALUE_DF = pd.DataFrame(
    {
        "League": ["EPL", "Bundesliga", "Ligue 1"],
        "Match": [
            "Liverpool vs Chelsea",
            "Bayern Munich vs Borussia Dortmund",
            "Lyon vs Paris Saint-Germain",
        ],
        "Bet": ["Home: Liverpool", "Draw", "Over 2.5"],
        "Best odds": ["1.85", "3.40", "1.95"],
        "_model_prob": [0.582, 0.310, 0.520],
        "_value_pct": [0.064, 0.090, 0.052],
    }
)


SAMPLE_BACKTEST = {
    "leagues": {
        "epl": {
            "markets": {
                "home_win": {"yield_pct": -10.96, "n_bets": 842},
                "draw": {"yield_pct": -12.79, "n_bets": 507},
                "away_win": {"yield_pct": -5.08, "n_bets": 1289},
                "over_2.5": {"yield_pct": -11.92, "n_bets": 430},
                "under_2.5": {"yield_pct": -2.90, "n_bets": 820},
            }
        },
        "bundesliga": {
            "markets": {
                "home_win": {"yield_pct": -1.18, "n_bets": 615},
                "draw": {"yield_pct": 7.20, "n_bets": 679},
                "away_win": {"yield_pct": -12.92, "n_bets": 869},
                "over_2.5": {"yield_pct": -3.74, "n_bets": 725},
                "under_2.5": {"yield_pct": -22.73, "n_bets": 316},
            }
        },
        "ligue1": {
            "markets": {
                "home_win": {"yield_pct": -10.78, "n_bets": 590},
                "draw": {"yield_pct": -9.56, "n_bets": 491},
                "away_win": {"yield_pct": 1.62, "n_bets": 740},
                "over_2.5": {"yield_pct": 0.27, "n_bets": 376},
                "under_2.5": {"yield_pct": -11.72, "n_bets": 598},
            }
        },
    }
}


def test_returns_error_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with requests_mock.Mocker() as m:
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)
        assert m.call_count == 0  # no HTTP call when key missing

    assert isinstance(result, ExplainerResult)
    assert result.picks == []
    assert result.error is not None
    assert "OPENROUTER_API_KEY" in result.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_explainer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ai_explainer'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai_explainer.py`:

```python
"""LLM-powered explainer for the dashboard's value bets.

Posts to OpenRouter's OpenAI-compatible chat-completions endpoint with a
constrained system prompt that forces JSON output. Returns an ExplainerResult
dataclass; never raises to the Streamlit caller.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash:free"


@dataclass
class Pick:
    """One AI-ranked value bet with the AI's reasoning."""

    pick_id: int
    key_reason: str
    risk: str
    model_edge_pct: float


@dataclass
class ExplainerResult:
    """Result of an AI explainer call. error is None on success or when value_df is empty."""

    picks: list[Pick] = field(default_factory=list)
    error: str | None = None


def explain_top_picks(
    value_bets_df: pd.DataFrame,
    backtest_summary: dict,
) -> ExplainerResult:
    """Rank and explain the top 10 value bets via OpenRouter.

    Returns an ExplainerResult; never raises. error is set on any failure mode
    that the caller should surface to the user.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ExplainerResult(
            picks=[],
            error="Set OPENROUTER_API_KEY in .env to enable AI picks.",
        )

    # Other paths to be implemented in subsequent tasks.
    return ExplainerResult(picks=[], error=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ai_explainer.py -v`
Expected: PASS — 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai_explainer.py tests/test_ai_explainer.py
git commit -m "feat(ai): scaffold ai_explainer module with API-key-missing path"
```

---

## Task 2: Empty value-bets path

No HTTP call when there are no bets to explain — saves a wasted API call.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ai_explainer.py`:

```python
def test_returns_no_error_when_value_bets_empty(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    empty_df = pd.DataFrame(columns=SAMPLE_VALUE_DF.columns)

    with requests_mock.Mocker() as m:
        result = explain_top_picks(empty_df, SAMPLE_BACKTEST)
        assert m.call_count == 0  # no HTTP call when df is empty

    assert result.picks == []
    assert result.error is None  # empty input is not an error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_explainer.py::test_returns_no_error_when_value_bets_empty -v`
Expected: FAIL — the function will currently try to proceed past the API-key check and reach the `return ExplainerResult(picks=[], error=None)` placeholder, but it would also (in later tasks) make an HTTP call. To make the test fail meaningfully now, run it after Task 3 is partially implemented OR add a placeholder HTTP call. The simplest TDD: write the test, it passes accidentally now (placeholder returns empty picks + None error), but the assertion that `m.call_count == 0` will protect against regression in later tasks.

If the test passes immediately on the placeholder, mark this step as "PASS (regression guard for future tasks)" and move on.

- [ ] **Step 3: Write minimal implementation**

Modify `src/ai_explainer.py` to add the empty-input check explicitly:

```python
def explain_top_picks(
    value_bets_df: pd.DataFrame,
    backtest_summary: dict,
) -> ExplainerResult:
    """Rank and explain the top 10 value bets via OpenRouter.

    Returns an ExplainerResult; never raises. error is set on any failure mode
    that the caller should surface to the user.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ExplainerResult(
            picks=[],
            error="Set OPENROUTER_API_KEY in .env to enable AI picks.",
        )

    if value_bets_df.empty:
        return ExplainerResult(picks=[], error=None)

    # HTTP call + parsing to be implemented in subsequent tasks.
    return ExplainerResult(picks=[], error=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ai_explainer.py -v`
Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai_explainer.py tests/test_ai_explainer.py
git commit -m "feat(ai): short-circuit when value_bets_df is empty (no wasted API call)"
```

---

## Task 3: Happy path — prompt construction, HTTP POST, JSON parsing

The biggest task. Implements the prompt builders, the OpenRouter POST, and the response parser. All later tasks build on this.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ai_explainer.py`:

```python
def test_returns_picks_when_api_responds_with_valid_json(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    # OpenRouter wraps the model's content as a JSON-encoded string in choices[0].message.content
    inner_content = (
        '{"picks": ['
        '{"pick_id": 0, "key_reason": "Edge of 6.4% on EPL home_win.", '
        '"risk": "EPL home_win backtest yield is -10.96% so weak prior.", '
        '"model_edge_pct": 6.4},'
        '{"pick_id": 1, "key_reason": "Bundesliga draws have +7.20% backtest yield.", '
        '"risk": "Sample size 679 is moderate.", '
        '"model_edge_pct": 9.0}'
        ']}'
    )
    api_response = {"choices": [{"message": {"role": "assistant", "content": inner_content}}]}

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, json=api_response, status_code=200)
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.error is None
    assert len(result.picks) == 2
    first = result.picks[0]
    assert isinstance(first, Pick)
    assert first.pick_id == 0
    assert first.key_reason.startswith("Edge of 6.4%")
    assert first.risk.startswith("EPL home_win")
    assert first.model_edge_pct == pytest.approx(6.4)
    second = result.picks[1]
    assert second.pick_id == 1
    assert second.model_edge_pct == pytest.approx(9.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_explainer.py::test_returns_picks_when_api_responds_with_valid_json -v`
Expected: FAIL — current implementation returns `picks=[]`, test asserts `len(result.picks) == 2`

- [ ] **Step 3: Write minimal implementation**

Replace `src/ai_explainer.py` with the full happy-path implementation:

```python
"""LLM-powered explainer for the dashboard's value bets.

Posts to OpenRouter's OpenAI-compatible chat-completions endpoint with a
constrained system prompt that forces JSON output. Returns an ExplainerResult
dataclass; never raises to the Streamlit caller.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash:free"
REQUEST_TIMEOUT = 30
TEMPERATURE = 0.2

# Maps the model's snake_case league keys to display names used in the prompt.
LEAGUE_DISPLAY = {
    "epl": "EPL",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue 1",
}

MARKET_KEYS = ["home_win", "draw", "away_win", "over_2.5", "under_2.5"]

SYSTEM_PROMPT = """\
You are a betting model explainer. You receive a table of value bets identified by
a calibrated machine-learning model and a summary of the model's historical
backtest performance per league per market.

Your job: pick the top 10 bets and explain each, using ONLY the data provided.
If fewer than 10 value bets are in the input, return all of them ranked.
Do not invent stats, recent form, injuries, news, head-to-head history, or
anything not in the inputs. If you cannot justify a pick from the provided data,
do not pick it.

For each pick, output:
- pick_id: row index from the table
- key_reason: one sentence on why it stands out, citing only fields in the inputs
- risk: one sentence on what could go wrong, citing only fields in the inputs
- model_edge_pct: the edge column value

Output only this JSON, nothing else:
{"picks": [{"pick_id": int, "key_reason": str, "risk": str, "model_edge_pct": float}, ...]}
"""


@dataclass
class Pick:
    """One AI-ranked value bet with the AI's reasoning."""

    pick_id: int
    key_reason: str
    risk: str
    model_edge_pct: float


@dataclass
class ExplainerResult:
    """Result of an AI explainer call. error is None on success or when value_df is empty."""

    picks: list[Pick] = field(default_factory=list)
    error: str | None = None


def _format_backtest(backtest: dict) -> str:
    """Render the backtest summary as multi-line text for the user prompt."""
    lines = ["Backtest yields by league and market (yield% / n_bets):"]
    for league_key, summary in backtest.get("leagues", {}).items():
        markets = summary.get("markets", {})
        parts = []
        for mk in MARKET_KEYS:
            stats = markets.get(mk, {})
            parts.append(
                f"{mk}: {stats.get('yield_pct', 0.0):+.2f}% / {stats.get('n_bets', 0)}"
            )
        lines.append(f"{LEAGUE_DISPLAY.get(league_key, league_key)} — " + ", ".join(parts))
    return "\n".join(lines)


def _format_value_bets(df: pd.DataFrame) -> str:
    """Render the value-bets DataFrame as a markdown-style table for the user prompt.

    Numerical columns are rendered as plain floats (no `%`, no `+` prefix) so the
    AI can echo them back as JSON numbers without parsing.
    """
    lines = [
        "Value bets identified for this week:",
        "| idx | League | Match | Bet | model_pct | fair_pct | edge_pct |",
    ]
    for idx, row in df.iterrows():
        model_pct = row["_model_prob"] * 100
        edge_pct = row["_value_pct"] * 100
        fair_pct = model_pct - edge_pct
        lines.append(
            f"| {idx} | {row['League']} | {row['Match']} | {row['Bet']} | "
            f"{model_pct:.1f} | {fair_pct:.1f} | {edge_pct:.1f} |"
        )
    return "\n".join(lines)


def _build_user_prompt(value_bets_df: pd.DataFrame, backtest: dict) -> str:
    return (
        _format_backtest(backtest)
        + "\n\n"
        + _format_value_bets(value_bets_df)
        + "\n\nPick the top 10."
    )


def _parse_picks(api_response: dict) -> list[Pick]:
    """Pull the assistant's content out of the OpenRouter response and parse it as JSON."""
    content = api_response["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    picks = []
    for raw in parsed["picks"]:
        picks.append(
            Pick(
                pick_id=int(raw["pick_id"]),
                key_reason=str(raw["key_reason"]),
                risk=str(raw["risk"]),
                model_edge_pct=float(raw["model_edge_pct"]),
            )
        )
    return picks


def explain_top_picks(
    value_bets_df: pd.DataFrame,
    backtest_summary: dict,
) -> ExplainerResult:
    """Rank and explain the top 10 value bets via OpenRouter.

    Returns an ExplainerResult; never raises. error is set on any failure mode
    that the caller should surface to the user.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return ExplainerResult(
            picks=[],
            error="Set OPENROUTER_API_KEY in .env to enable AI picks.",
        )

    if value_bets_df.empty:
        return ExplainerResult(picks=[], error=None)

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    user_prompt = _build_user_prompt(value_bets_df, backtest_summary)

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": TEMPERATURE,
        },
        timeout=REQUEST_TIMEOUT,
    )

    api_response = response.json()
    picks = _parse_picks(api_response)
    return ExplainerResult(picks=picks, error=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ai_explainer.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai_explainer.py tests/test_ai_explainer.py
git commit -m "feat(ai): happy-path OpenRouter call + JSON parsing for top-picks"
```

---

## Task 4: Prompt-content assertion

This test verifies the user prompt actually contains both inputs — guards against silently regressing the prompt format. Implementation already done in Task 3; this just adds the test.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ai_explainer.py`:

```python
def test_prompt_includes_backtest_yields_and_value_bets(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    inner_content = '{"picks": []}'
    api_response = {"choices": [{"message": {"role": "assistant", "content": inner_content}}]}

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, json=api_response, status_code=200)
        explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)
        request_body = m.last_request.json()

    user_message = next(msg["content"] for msg in request_body["messages"] if msg["role"] == "user")

    # Backtest summary must be in the prompt
    assert "Bundesliga" in user_message
    assert "draw: +7.20%" in user_message  # the standout signal from the backtest
    assert "/ 679" in user_message  # n_bets for that draw market

    # Value-bets table must be in the prompt
    assert "Liverpool vs Chelsea" in user_message
    assert "Bayern Munich vs Borussia Dortmund" in user_message
    assert "| 0 |" in user_message  # row indices rendered

    # Pre-formatted numbers (no `%` or `+` in the data cells)
    assert "| 6.4 |" in user_message  # edge for row 0
    assert "| 9.0 |" in user_message  # edge for row 1
```

- [ ] **Step 2: Run test to verify it passes (already implemented in Task 3)**

Run: `uv run pytest tests/test_ai_explainer.py::test_prompt_includes_backtest_yields_and_value_bets -v`
Expected: PASS — the prompt builder from Task 3 already produces this content. If FAIL, fix `_format_backtest` or `_format_value_bets` until the assertions hold.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ai_explainer.py
git commit -m "test(ai): assert prompt contains backtest yields + value-bets rows"
```

---

## Task 5: Network / HTTP error handling

Adds graceful handling for `RequestException` (timeouts, DNS failures) and non-200 status codes (rate limits, server errors).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ai_explainer.py`:

```python
def test_handles_429_rate_limit_gracefully(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, status_code=429, text="Too Many Requests")
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.picks == []
    assert result.error is not None
    assert "OpenRouter returned 429" in result.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_explainer.py::test_handles_429_rate_limit_gracefully -v`
Expected: FAIL — current implementation calls `response.json()` on the 429 body and either crashes or produces a `KeyError` on `["choices"]`

- [ ] **Step 3: Write minimal implementation**

Modify `explain_top_picks` in `src/ai_explainer.py`. Replace the body from the `model = os.getenv(...)` line down to the final `return` with:

```python
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    user_prompt = _build_user_prompt(value_bets_df, backtest_summary)

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": TEMPERATURE,
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        return ExplainerResult(
            picks=[],
            error=f"Couldn't reach OpenRouter: {e}. Try again in a minute.",
        )

    if response.status_code != 200:
        snippet = response.text[:200]
        return ExplainerResult(
            picks=[],
            error=f"OpenRouter returned {response.status_code}: {snippet}. Try again in a minute.",
        )

    try:
        api_response = response.json()
        picks = _parse_picks(api_response)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        return ExplainerResult(
            picks=[],
            error=f"AI returned unexpected response ({type(e).__name__}). Click again to retry.",
        )

    return ExplainerResult(picks=picks, error=None)
```

- [ ] **Step 4: Run all tests to verify**

Run: `uv run pytest tests/test_ai_explainer.py -v`
Expected: PASS — 5 passed (the 429 test plus all earlier ones)

- [ ] **Step 5: Commit**

```bash
git add src/ai_explainer.py
git commit -m "feat(ai): graceful handling for network errors and non-200 status"
```

---

## Task 6: Drop picks with unknown pick_id

Belt-and-braces against the AI hallucinating row indices that don't exist in the value-bets table.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ai_explainer.py`:

```python
def test_drops_picks_with_unknown_pick_id(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    # SAMPLE_VALUE_DF has indices [0, 1, 2]. AI fabricates pick_id 999.
    inner_content = (
        '{"picks": ['
        '{"pick_id": 0, "key_reason": "real", "risk": "real", "model_edge_pct": 6.4},'
        '{"pick_id": 999, "key_reason": "hallucinated", "risk": "fake", "model_edge_pct": 99.9},'
        '{"pick_id": 2, "key_reason": "real", "risk": "real", "model_edge_pct": 5.2}'
        ']}'
    )
    api_response = {"choices": [{"message": {"role": "assistant", "content": inner_content}}]}

    with requests_mock.Mocker() as m:
        m.post(OPENROUTER_URL, json=api_response, status_code=200)
        result = explain_top_picks(SAMPLE_VALUE_DF, SAMPLE_BACKTEST)

    assert result.error is None
    assert [p.pick_id for p in result.picks] == [0, 2]  # 999 dropped, 0 and 2 kept in order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_explainer.py::test_drops_picks_with_unknown_pick_id -v`
Expected: FAIL — current parser keeps all three picks; test asserts only [0, 2]

- [ ] **Step 3: Write minimal implementation**

Modify `_parse_picks` in `src/ai_explainer.py` to accept the value-bets index for validation. Update the caller too.

Replace `_parse_picks` with:

```python
def _parse_picks(api_response: dict, valid_ids: set[int]) -> list[Pick]:
    """Pull the assistant's content out of the OpenRouter response and parse it as JSON.

    Picks whose pick_id is not in valid_ids are dropped silently — defends against
    the AI hallucinating row indices that don't exist in the value-bets table.
    """
    content = api_response["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    picks = []
    for raw in parsed["picks"]:
        pick_id = int(raw["pick_id"])
        if pick_id not in valid_ids:
            continue
        picks.append(
            Pick(
                pick_id=pick_id,
                key_reason=str(raw["key_reason"]),
                risk=str(raw["risk"]),
                model_edge_pct=float(raw["model_edge_pct"]),
            )
        )
    return picks
```

Update the caller in `explain_top_picks`. Find the line:

```python
        picks = _parse_picks(api_response)
```

And replace with:

```python
        picks = _parse_picks(api_response, valid_ids=set(value_bets_df.index))
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/test_ai_explainer.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/ai_explainer.py tests/test_ai_explainer.py
git commit -m "feat(ai): drop picks whose pick_id is not in value_bets_df.index"
```

---

## Task 7: Streamlit integration in app.py

UI integration. No unit test (Streamlit interactions are hard to test in isolation); manual smoke test in Task 8.

- [ ] **Step 1: Add the import**

In `app.py`, find the import block ending with `from src.value import assess_value, remove_bookmaker_margin` and add immediately after:

```python
from src.ai_explainer import ExplainerResult, explain_top_picks
```

- [ ] **Step 2: Add the render helper**

In `app.py`, add this helper function immediately above the `render_fixtures_tab` definition (find `def render_fixtures_tab() -> None:` and put the new function above it):

```python
def _render_ai_picks(result: ExplainerResult, value_df_reset: pd.DataFrame) -> None:
    """Render the AI explainer's output as banner + collapsible cards."""
    if result.error:
        if "OPENROUTER_API_KEY" in result.error:
            st.info(result.error)
        elif "OpenRouter returned" in result.error or "Couldn't reach" in result.error:
            st.error(result.error)
        else:
            st.warning(result.error)
        return

    if not result.picks:
        return

    st.warning(
        "⚠️ **Paper trade only.** These picks are the model's edge calls explained by an "
        "LLM, not investment advice. Backtests beat the future ~30% of the time. "
        "[BeGambleAware](https://www.begambleaware.org)"
    )

    for rank, pick in enumerate(result.picks, start=1):
        row = value_df_reset.iloc[pick.pick_id]
        title = (
            f"#{rank}  {row['Bet']} — {row['Match']} @ {row['Best odds']}  "
            f"(Edge: {pick.model_edge_pct:+.1f}%)"
        )
        with st.expander(title, expanded=True):
            st.markdown(f"**Why:** {pick.key_reason}")
            st.markdown(f"**Risk:** {pick.risk}")
```

- [ ] **Step 3: Insert the button + invocation**

In `app.py`, find the section in `render_fixtures_tab` that currently looks like:

```python
    if not value_df.empty:
        st.markdown(f"### 🎯 Value bets ({len(value_df)})")
        st.dataframe(
            value_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )
```

Replace with:

```python
    if not value_df.empty:
        st.markdown(f"### 🎯 Value bets ({len(value_df)})")

        if st.button("🤖 Get AI picks", type="primary"):
            value_df_reset = value_df.reset_index(drop=True)
            with st.spinner("Asking the AI to rank these picks…"):
                result = explain_top_picks(value_df_reset, _models.backtest or {})
            _render_ai_picks(result, value_df_reset)

        st.dataframe(
            value_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )
```

- [ ] **Step 4: Run the test suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: PASS — 64 passed (58 existing + 6 new)

- [ ] **Step 5: Run lint + format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: All checks passed; "21 files already formatted" (or "22" if a new test file shifted the count).

If format fails, run `uv run ruff format .` and re-check.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(app): wire AI explainer into This Week tab with Get AI picks button"
```

---

## Task 8: Manual smoke test + final acceptance check

Validate all 7 acceptance criteria from the spec end-to-end.

- [ ] **Step 1: Confirm trained artifacts present**

Run: `ls models/ | wc -l`
Expected: 7 (5 bettor pkl + fixtures parquet + backtest json). If missing, the integration test isn't meaningful — go run `uv run python scripts/train_model.py` first.

- [ ] **Step 2: Run the dashboard with `OPENROUTER_API_KEY` set**

Confirm `.env` has `OPENROUTER_API_KEY=...` set:

```bash
uv run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('key set:', bool(os.getenv('OPENROUTER_API_KEY')))"
```

Expected: `key set: True`

Then start the dashboard:

```bash
uv run streamlit run app.py --server.headless true
```

In a browser, open http://localhost:8501.

- [ ] **Step 3: Verify the happy path**

On the **This Week** tab:
- A "🤖 Get AI picks" primary-coloured button appears between the value-bets header and the table
- Click the button. Within ~5 seconds:
  - A yellow paper-trade banner appears
  - 10 collapsible cards (or fewer if there are fewer than 10 value bets) appear above the table, each with rank, bet, match, odds, edge in the title and Why/Risk markdown in the body
  - All cards default to expanded
- The full value-bets table still renders below the cards, unchanged

Each card's "Why" and "Risk" text references only fields visible in the dashboard (league, match, model %, fair %, edge, backtest yields). It must NOT mention specific players, recent results not in the data, head-to-head history, injuries, or any external knowledge.

- [ ] **Step 4: Verify the API-key-missing path**

In the running terminal, kill the Streamlit server (Ctrl+C). Temporarily mask the key:

```powershell
$env:OPENROUTER_API_KEY = ""; uv run streamlit run app.py --server.headless true
```

Click the AI button. Expected: blue info banner reads `Set OPENROUTER_API_KEY in .env to enable AI picks.`

Stop the server.

- [ ] **Step 5: Verify the network-error path**

Disconnect network OR temporarily set an unreachable URL via env var (simpler):

```powershell
# In a separate terminal, append a fake OpenRouter URL override to test
# The cleanest way: edit src/ai_explainer.py temporarily to point OPENROUTER_URL at https://localhost:1/foo
```

Actually, simpler manual check: turn off Wi-Fi, restart the dashboard with the real key, click the button. Expected: red error banner reading `Couldn't reach OpenRouter: ...`. Re-enable Wi-Fi.

- [ ] **Step 6: Final acceptance criteria check**

Verify against the spec's acceptance criteria (`docs/superpowers/specs/2026-05-07-v3-ai-explainer-design.md`):

| # | Criterion | Pass? |
|---|---|---|
| 1 | `pytest -q` shows 64/64 passing | run again to confirm |
| 2 | `ruff check .` and `ruff format --check .` clean | run again |
| 3 | With key set, button renders 10 cards within ~5 sec | Step 3 above |
| 4 | With key unset, button shows blue info banner (no crash) | Step 4 above |
| 5 | With network down, button shows red error banner (no crash) | Step 5 above |
| 6 | Paper-trade banner visible above the AI cards in yellow | Step 3 above |
| 7 | Card text references only fields in the prompt | Step 3 above (manual content review) |

- [ ] **Step 7: No commit needed (manual verification only)**

If anything failed in Steps 3-5, return to the relevant earlier task and fix. Otherwise, the work is done.
