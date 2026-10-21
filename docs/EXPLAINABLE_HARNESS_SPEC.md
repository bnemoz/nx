# Spec: Explainable harness — transcript & provenance surfaces

Status: draft
Version target: 0.166.0 (semver: new feature → minor bump)

## Summary

nx makes many invisible decisions per turn — compression, retries, salvage,
tool gating, speculative reads — and the user only sees the outcomes. This
spec adds **provenance surfaces**: read-only views that show what actually
went over the wire, and what the harness did to it, so an interested user can
learn how a coding-agent harness works instead of treating nx as a black box.

Seven features, in implementation order. **Phase 1 (this spec's focus) is the
per-message transcript view in `/context`.** Phases 2–7 are specced at a
design level for follow-up specs.

Guiding principle: **"show as-sent"**. Wherever the harness transforms
messages (compression, summarization, salvage), the views must be able to
render the message exactly as the model last saw it, not the in-memory
original. The as-sent machinery built in Phase 1 is reused by Phase 2.

## Phase 1 — transcript view in `/context`

### User story

The user opens `/context`, moves to a message row, presses `enter`, and sees
the full message — text, tool-call args, tool results — not the truncated
preview. A toggle (`a`) switches between **live** (current in-memory state)
and **as-sent** (what the model actually received on the last settled call).
If compression elided the message, the as-sent view shows the elision
marker, proving it happened.

### Design (verified, file:line)

- The view is `ContextView` (`nx/_ctxreport.py:251`), registered
  `@register_view("context")` at `nx/_fullscreen.py:639`. It is already the
  one view allowed a mutation (pinning, via deferred `pending_action`).
  The transcript detail adds **no new mutation** — read-only subview.
- Rendering goes through `_render.py`/`_keys.py`; keybindings are declared
  in `nx/_keys.py:79` (`"context"` section — add `enter`/`a`/`esc` there).
- As-sent source: the context probe and metrics ring buffer
  (`st._METRICS.turns`, capped at 64, per `nx/_ctxreport.py:73-100`) hold
  per-call usage but **not payload snapshots**. Phase 1 therefore adds a
  lightweight **last-prompt tap**: capture the serialized message list
  (or its digest + per-message diffs) once per settled model call in
  `_turn.py` at the point the request is built, stored on
  `st._LAST_PROMPT` (session-lifetime, like `_ctx_pin` state in
  `_repl_state`). Cap memory: store per-message (role, index, sha256,
  compressed-flag) digests rather than full text, plus the full text of
  only the *most recent* prompt.

  - Minimal viable version: **digest + compressed-flag only** — the as-sent
    view then shows, per message, `identical to live` vs `elided/summarized
    by compression`, with the compressed form rendered only for the latest
    prompt. This avoids unbounded memory and still teaches the mechanism.

- Subview navigation: from a selected row, `enter` opens the detail pane
  (same alternate screen; `esc` returns to the list). `a` toggles
  live/as-sent; when no as-sent snapshot exists (no settled call yet, or a
  resumed session), the toggle renders `— no settled call recorded —`
  and stays on live.

- Non-tty parity: bare `/context` outside a tty prints the inline report
  (`_inline_body`, `nx/_repl_settings_cmds.py:515`); the inline body gains
  a `t <n>` argument (`/context t 12`) that prints one message's transcript
  inline, byte-compatible with the view's rendering.

### Deliverables

1. `nx/_ctxreport.py`: `_message_digests()` (per-message sha256 +
   compressed flag from the last-prompt tap), `TranscriptDetail` renderer
   (live text vs as-sent diff markers), key handling in `ContextView`.
2. `nx/_keys.py`: `enter`, `a`, `esc` in the `"context"` keymap.
3. `nx/_turn.py`: one capture point at request-build; `_LAST_PROMPT` on
   `st`. Drained by `nx_state_reset` in `tests/_nxtest.py` (same fixture
   fix, centrally — see AGENTS.md known-flake ledger rule).
4. `nx/_repl_settings_cmds.py`: `/context t <n>` inline subcommand.
5. Tests: `tests/test_ctxreport.py` (or nearest existing file) covers
   digest alignment, toggle behavior without a settled call, and inline
   `t <n>` output.

### Out of scope (Phase 1)

Editing, re-sending, or truncating messages from the view; tokenizing each
message via the server (stays estimated, per existing `/ctx` policy).

## Phase 2 — before/after compression diff

`/compress` records a **last-compression ledger**: which message ranges were
dropped, which summarized, and the summary text that replaced them. Exposed
as `/compress --show` (or a `CompressDiffView` reusing Phase 1's as-sent
machinery: render the pre-compression digest list next to the post one).
Home: `nx/_compress.py` (ledger write) + `nx/_ctxreport.py` (render).
Teaches: what compression actually costs, why pins exist.

## Phase 3 — per-turn event log (`/turns`)

A read-only view over `st._METRICS.turns` plus a new lightweight
**harness-event ring** (status changes from `_turn_status.py`, retry ladder
steps from `_turn_retry.py`, salvage events from `_argsalvage.py`, breaker
trips from `_breaker.py`, prefetch/spec-read activity from
`_tool_prefetch.py`/`_spec_reads.py`). One row per settled model call:
status, tokens, cache %, termination reason, harness interventions. Teaches:
why a turn took N model calls, what the harness did about errors.
Home: new `nx/_turnlog.py`; surfaces via the fullscreen shell
(`register_view("turns")`).

## Phase 4 — tool-gate transparency (`/tools`)

Shows the tiered registry state this turn: active tools, tier-2 tools and
their unlock cost (estimated schema tokens, computed the same way
`_turn_overhead_est()` does — `nx/_ctxreport.py:105`), withheld-tool
refusals the model received this session. Home: read-side over
`nx/_tool_gates.py` (`_TOOL_GATES`, `_active_tools`, `_gate_vector`); new
view module. Teaches: why the model can't see every tool, what unlocking
costs.

## Phase 5 — annotated system-prompt inspector (`/prompt`)

Dumps the pinned `SYSTEM` bytes (guard: `nx/_prefix.py`), each satellite
prompt from `nx/_prompts.py`, and the serialized tool schemas — each with a
per-section token estimate. Read-only; reuses the shared estimator.
Teaches: the real per-turn floor, what the harness says to the model
before the user says anything.

## Phase 6 — decision breadcrumbs inline

Dim one-line prints at the moment harness decisions happen, e.g.
`[harness] salvaged malformed tool args (attempt 2/3)`,
`[harness] retrying after stream error (ladder step 2)`. Sites: the existing
`if not quiet` guarded prints in `_turn.py`/`_stream_parse.py` — add
alongside, honoring the same quiet gate. Behind one setting
(`/settings breadcrumbs on|off`, default on). Teaches: harness at work,
live, without opening any view.

## Phase 7 — session token ledger

Cumulative footer/summary: tokens sent vs cache-hit vs generated,
tokens compressed away, per-turn overhead paid. Pure aggregation over
`st._METRICS` + the Phase 2 ledger. Home: `nx/_ctxreport.py` extension
(footer lines) with a `/ctx total` subcommand. Teaches: what the harness
costs you.

## Cross-cutting rules

- **All views are read-only.** No Phase may mutate messages, pins (beyond
  existing pinning), or harness state.
- **No new hidden state without a drain.** Any new session-scope global
  (`st._LAST_PROMPT`, Phase 2's ledger, Phase 3's event ring) is drained in
  `nx_state_reset` (`tests/_nxtest.py`) in the same change — the fixture fix
  is central, never per-test (AGENTS.md, known-flake ledger rule).
- **Cheap by default.** Digests over full-text snapshots; estimates over
  round-trips; nothing runs unless a view asks for it.
- **Offline.** No feature depends on a live model endpoint (suite rule);
  views degrade to `— no data —` rows.
- **Touch points** (CONTRIBUTING.md): each new slash command updates
  `nx/_commands.py` help list + `nx/_repl.py` dispatch; each new keybinding
  updates `nx/_keys.py`; each new view registers in `nx/_fullscreen.py`.
- **Docs**: README gains a "See what the harness is doing" section once
  Phases 1–3 land; CONTRIBUTING gains the env-var/touch-point entries.
