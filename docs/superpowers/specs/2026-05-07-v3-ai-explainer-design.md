# v3 AI Explainer — Design

**Status:** approved
**Date:** 2026-05-07
**Predecessor:** v2 model improvement (`docs/superpowers/specs/2026-05-06-model-improvement-v2-design.md`)

## Goal

Add an LLM-powered "AI picks" panel to the **This Week** tab that ranks the top 10 value bets the model has identified and explains each pick using only the data already on the dashboard. The AI replaces nothing — it sits above the value-bets table as a curated, explained subset of what's already shown.

## Why

The trained v2 model produces 12-50 value bets per week across the Big-5 leagues. Femi has to scan a wide table and decide which to act on. The backtest yields are visible in a sidebar expander but aren't tied to specific bets in the table. An LLM that reads both the backtest yields *and* the current value bets can highlight which picks fit the strongest historical signals — turning a manual scan into a curated short-list.

The constrained design (no external knowledge, structured fields, manual trigger) keeps the failure mode small: the worst the AI can do is pick a bad subset of a list Femi already had to look at.

## Locked constraints

These are non-negotiable, carried forward from the project's v3 brief:

- **OpenRouter API** for inference (Femi's existing key in `OPENROUTER_API_KEY`)
- **Default model:** `google/gemini-2.5-flash:free`. Override via `OPENROUTER_MODEL` env var.
- **AI sees only data already on the dashboard.** No external knowledge, no fabricated stats, no head-to-head invention.
- **Structured JSON output.** No free-form prose anywhere except inside the `key_reason` and `risk` fields.
- **Prominent paper-trade banner** above the AI panel.

## Architecture

### New module: `src/ai_explainer.py`

Single public function:

```python
def explain_top_picks(
    value_bets_df: pd.DataFrame,
    backtest_summary: dict,
) -> ExplainerResult:
    ...
```

Returns an `ExplainerResult` dataclass:

```python
@dataclass
class Pick:
    pick_id: int          # row index in value_bets_df
    key_reason: str       # one sentence
    risk: str             # one sentence
    model_edge_pct: float # echoed from the input row's edge

@dataclass
class ExplainerResult:
    picks: list[Pick]
    error: str | None     # human-readable; non-None when picks is empty by failure
```

The `error` field lets the Streamlit caller distinguish "no value bets to explain" (picks empty, error None) from "API rate-limited" (picks empty, error set). Each failure mode populates `error` with a different message; the UI maps message → visual style (info/warning/error).

### No new dependencies

OpenRouter exposes an OpenAI-compatible HTTP API. We POST directly via the `requests` library that's already in the dep set. No `openai` SDK, no `httpx`.

### Tests: `tests/test_ai_explainer.py`

Six tests, all using `requests-mock` to stub the OpenRouter endpoint. Listed in §6 below.

## Data flow

```
User clicks "🤖 Get AI picks"
         │
         ▼
Streamlit handler in app.py
  reads value_df + _models.backtest
         │
         ▼
explain_top_picks(value_df, backtest)
  ├─ Check OPENROUTER_API_KEY present     → if missing, return ExplainerResult([], "set OPENROUTER_API_KEY...")
  ├─ Check value_bets_df non-empty        → if empty, return ExplainerResult([], None)
  ├─ Build system prompt (constant)
  ├─ Build user prompt (backtest + table rows)
  ├─ POST to OpenRouter chat/completions
  │  ├─ on RequestException → return ExplainerResult([], "Couldn't reach OpenRouter: ...")
  │  └─ on status != 200    → return ExplainerResult([], "OpenRouter returned <code>: ...")
  ├─ Parse JSON response
  │  └─ on JSONDecodeError / KeyError → return ExplainerResult([], "AI returned unexpected response...")
  ├─ Validate each pick_id is in value_bets_df.index; drop invalid
  └─ return ExplainerResult(picks=[Pick, ...], error=None)
         │
         ▼
Streamlit renders ExplainerResult:
  ├─ if error: st.info / st.error / st.warning per error category
  ├─ if picks: prominent paper-trade banner + 10 collapsible cards
  └─ else: nothing (no card section, no error — this is the "value_df was empty" case)
```

## Prompt design

### System prompt (constant)

```
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
```

### User prompt (constructed each call)

```
Backtest yields by league and market (yield% / n_bets):
EPL — home_win: -10.96% / 842, draw: -12.79% / 507, away_win: -5.08% / 1289, over_2.5: -11.92% / 430, under_2.5: -2.90% / 820
La Liga — home_win: -9.17% / 860, draw: -6.11% / 895, ...
Serie A — ...
Bundesliga — ...
Ligue 1 — ...

Value bets identified for this week:
| idx | League | Match | Bet | model_pct | fair_pct | edge_pct |
| 0   | EPL    | Liverpool vs Chelsea | Home: Liverpool | 58.2 | 51.8 | 6.4 |
| 1   | Bundesliga | Bayern vs Dortmund | Draw | 31.0 | 22.0 | 9.0 |
| ...

Pick the top 10.
```

Numerical columns are rendered as plain floats (no `%`, no `+` prefix) so the AI can echo them back as JSON numbers without parsing. The `idx` column is a clean 0..N-1 sequence: the caller in `app.py` does `value_df.reset_index(drop=True)` once before passing into `explain_top_picks`. This way `pick.pick_id` returned by the AI maps directly to `value_df.iloc[pick.pick_id]` in `_render_ai_picks`, so the prompt-building module and the rendering module agree on indexing without coupling.

### Why this shape

- The AI sees the same backtest summary Femi sees in the sidebar Model info expander, formatted as text.
- The value-bets table is constrained to the columns the AI can reason about: League, Match, Bet, Model %, Fair %, Edge. Internal columns (`_value_pct`, `_decimal_odds`, `_bookmaker`, etc.) are deliberately excluded — they don't help reasoning and broaden the prompt's surface area.
- Row indices (`idx`) make the AI's output cheap to map back: the JSON's `pick_id: 7` corresponds to `value_df.iloc[7]` (or `value_df.loc[7]` after reset_index).
- Forced JSON output prevents the AI from drifting into "as a betting expert, I think..." preambles.

## UI integration in `app.py`

Inside `render_fixtures_tab()`, after `if not value_df.empty:` and before the existing `st.dataframe(value_df[display_cols], ...)` call, add:

```
st.markdown(f"### 🎯 Value bets ({len(value_df)})")

# (NEW) AI picks section
if st.button("🤖 Get AI picks", type="primary"):
    value_df_reset = value_df.reset_index(drop=True)
    with st.spinner("Asking the AI to rank these picks…"):
        result = explain_top_picks(value_df_reset, _models.backtest)
    _render_ai_picks(result, value_df_reset)

# Existing dataframe + log-bet form below
st.dataframe(...)
```

`_render_ai_picks(result, value_df)` lives in `app.py` (Streamlit-specific, not part of the explainer module). It:

1. Renders the paper-trade banner (yellow `st.warning`)
2. Iterates `result.picks` and renders each as a `st.expander` (expanded by default)
3. Each expander title: `"#{rank} — {bet} @ {odds} (Edge: {edge})"` pulled by joining `pick.pick_id` to `value_df`
4. Each expander body: two markdown blocks — `**Why:** {pick.key_reason}` and `**Risk:** {pick.risk}`
5. If `result.error` is set: render `st.info`/`st.error`/`st.warning` per error category instead of cards

The button + AI panel sit *between* the "Value bets (N)" header and the dataframe. Eye flow: header → AI banner → AI picks (if invoked) → full table → log-bet form.

## Error handling

| Failure | Caught at | `error` field set to | Streamlit visual |
|---|---|---|---|
| `OPENROUTER_API_KEY` env var missing | start of `explain_top_picks` | `"Set OPENROUTER_API_KEY in .env to enable AI picks."` | `st.info` |
| `requests.RequestException` (network, timeout, DNS) | around POST | `"Couldn't reach OpenRouter: {exception}. Try again in a minute."` | `st.error` |
| HTTP status != 200 (rate limit 429, server error 5xx) | after POST | `"OpenRouter returned {status}: {body[:200]}. Try again in a minute."` | `st.error` |
| `json.JSONDecodeError` on response body | after `resp.json()` | `"AI returned non-JSON response. Click again to retry."` | `st.warning` |
| Missing `"picks"` key, or any pick missing required fields | after parse | `"AI returned unexpected response. Click again to retry."` | `st.warning` |
| `pick_id` not in `value_df.index` | per-pick validation | not an error; drop the bad pick, return the rest | (no banner; just fewer cards) |
| `value_df` is empty (no value bets) | start of `explain_top_picks` | `None` (not an error) | (no AI section rendered at all) |

No exception ever bubbles to Streamlit's default red error banner. The dashboard never crashes from an AI failure.

## Tests

In `tests/test_ai_explainer.py`, using `requests-mock`:

1. **`test_returns_picks_when_api_responds_with_valid_json`** — mock 200 with `{"picks": [{...}, {...}]}`; assert two `Pick` dataclasses returned with correct field types and values; assert `result.error is None`.
2. **`test_returns_error_when_api_key_missing`** — `monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)`; assert `result.picks == []`, `"OPENROUTER_API_KEY" in result.error`, and assert no HTTP call was made.
3. **`test_returns_no_error_when_value_bets_empty`** — input is empty DataFrame; assert `result.picks == []`, `result.error is None`, no HTTP call made.
4. **`test_handles_429_rate_limit_gracefully`** — mock 429 status with rate-limit body; assert `result.picks == []`, `"OpenRouter returned 429" in result.error`.
5. **`test_drops_picks_with_unknown_pick_id`** — value_df has indices [0, 1, 2]; mock API returns picks with pick_ids [0, 999, 2]; assert returned picks have ids [0, 2] only.
6. **`test_prompt_includes_backtest_yields_and_value_bets`** — assert request body contains the backtest yield text (e.g. `"draw: -12.79%"`) and a sample row from the value-bets table (e.g. `"Liverpool vs Chelsea"`).

All tests use a fixed sample `value_df` and `backtest_summary` defined as test fixtures.

## File-by-file changes

| File | Change |
|---|---|
| `src/ai_explainer.py` | NEW. ~120 lines: dataclasses, `explain_top_picks`, prompt builders, `_post_to_openrouter`. |
| `tests/test_ai_explainer.py` | NEW. 6 tests as specified above. |
| `app.py` | Insert button + `_render_ai_picks` invocation between value-bets header and dataframe. Add `_render_ai_picks` helper. ~40 added lines. |
| `pyproject.toml` | No change. `requests` is already a dep. |

## Out of scope

- **No caching of AI responses.** Manual button is the cost control. Femi clicks → one call.
- **No streaming.** 10-card response renders after the full POST returns (~2-5 sec).
- **No history of past AI picks.** Streamlit session state would lose them on rerun anyway. Supabase logging of picks is a separate question.
- **No model A/B UI.** `OPENROUTER_MODEL` env var is the override mechanism.
- **No per-pick "log this bet" integration.** Existing log-bet form below the table covers this. Tying AI rank-1 → log-bet would couple v3 to Supabase tracking, out of scope for the explainer.
- **No prompt-token-budget logic.** Inputs are bounded: ≤ 50 value bets × 7 small columns + 5-row backtest summary fits well under any free model's context.

## Acceptance criteria

The work is done when:

1. `pytest -q` shows 64/64 passing (existing 58 + 6 new).
2. `ruff check .` and `ruff format --check .` clean.
3. With `OPENROUTER_API_KEY` set in `.env`, clicking "🤖 Get AI picks" on the local dashboard renders 10 cards above the value-bets table within ~5 seconds.
4. With `OPENROUTER_API_KEY` unset, clicking the button shows a blue info banner ("Set OPENROUTER_API_KEY...") instead of crashing.
5. With network disconnected (or OpenRouter unreachable), clicking the button shows a red error banner ("Couldn't reach OpenRouter...") instead of crashing.
6. The paper-trade-only banner sits above the AI cards prominently in yellow.
7. Each AI card's text references only fields in the prompt — no fabricated injury reports, recent results not in the data, or head-to-head numbers.
