# Spec: Live sub-agent event surface (`/tabs` mux)

Status: implemented in 0.122.0
Version target: 0.122.0 (semver: new feature → minor bump)

## Summary

Sub-agents run live today, but the user cannot see them work: async agents
render **nothing** (`quiet=True` suppresses every render site) and only
surface at completion via a one-time `[Runtime note]`. This spec adds a
**job-tagged event sink** (ring buffer per agent) and promotes `/tabs` from a
poll-only dashboard reachable at the idle prompt to a **live region under the
composer**, openable mid-turn via Ctrl-T, in the style of codex/claude code.

Out of scope (deliberately): transcript-level interleaving of agent output,
true detach/reattach (see "Deferred: daemon detach"). Grounded precedent:
Codex CLI multiplexes *sessions* via an app-server daemon, not transcript
interleaving; Gemini CLI renders tool groups inline and batches agent
completion. nx follows the same display-layer-only rule.

## Background (verified, file:line)

- Async agents: `spawn_agent()` `nx/_subagent.py:1139`; `_spawn_async()`
  `:696` registers `_ASYNC_AGENTS` (process-global dict, `_ASYNC_LOCK`
  `:347`) and runs `_run_model_turn_loop` (called at `:1024`, inside
  `_run_sync_agent_body`) on ONE daemon thread (`threading.Thread` at
  `:750`) with quiet mode set per-thread (`_ASYNC_QUIET_SLOTS`
  `threading.local` at `:253`, set `:738`, reset in the worker's `finally`
  `:743`). Concurrency cap `threading.BoundedSemaphore`
  (`_async_semaphore()` `:381`, default from `_async_max_concurrent()`
  `:368-378`, `NX_SUBAGENT_MAX_CONCURRENT` default 4 at `:374`).
- Quiet suppression sites (all must stay suppressed on the shared console):
  `nx/_stream_parse.py:134-141` (`StreamParser(quiet=...)` — parses and
  accounts everything, writes nothing), `nx/_dispatch.py:411-414` (no tool
  box), `nx/_turn_footer.py:206-207` (no footer), spinner/interrupt arming
  `nx/_turn.py:398,406,466,493`.
- **Key asset**: `StreamParser` already accumulates all content even when
  quiet (`parser.content`, appended unconditionally at
  `_stream_parse.py:456-459`). A tap for streamed prose is a fork in two
  genuine chokepoints (`_handle_content` :456-459 and the Live render
  path :481/:492) — NOT `__init__`, which contains zero writes.
- Raw write sites are scattered, NOT one render layer: ~56 raw `print(`
  sites (+2 `sys.stdout.write`) across `_turn.py` (35), `_stream_parse.py`
  (14 + 2), `_dispatch.py` (7), guarded by 20+ independent
  `if not quiet` checks.
- Existing visibility: `/tabs` dashboard `_surfaces.py:216
  (`run_tabs_dashboard`), `_cmd_tabs` `_repl.py:82`, Ctrl-T binding
  `_chatbox.py:1066-1072`; snapshot `_surfaces_snapshot()` `:163` rows for
  `main | bg | agent`; bg rows get a real log tail, **agent rows have an
  empty tail while running** and only show the answer tail after.
- Mid-turn Ctrl-T (\x14) currently does NOTHING: `_capture_steering_byte`
  (`_interrupt.py:255-283`; called from the watcher loop at `:218-219`)
  drops it via the `ch >= " "` printable check at `:280` — it is neither
  passed through as steering text nor bound. So §3 is "add handling", not
  "reorder a check".
- Completion delivery: `_stage_subagent_answer()` `_subagent.py:617` →
  `_PENDING_SUBAGENT_NOTICE` → `_flush_pending_background_notice()`
  (`nx/_runtime_notes.py:60`) appends a `role:"user"` message between turns
  (`_turn_toolcalls.py:244,274`, `_repl.py:382`). Append-only, prompt-cache
  preserving — **unchanged by this spec**.
- `st._PROGRESS_SINK` (`nx/_state.py:156`, `_progress.py:67`) is a single
  last-writer-wins slot, text-only, no job id — insufficient for a mux.
- UI stack: plain stdout prints from whichever thread, `rich.live.Live`
  panel (`_stream_parse.py:481`), 4fps `CLEAR_LINE` spinner, hand-rolled
  raw-mode editor (`nx/_chatbox.py`), deps exactly `httpx` + `rich`.
- The known flicker precedent: `_subagent.py:964-978` documents spinner
  CLEAR_LINE interleaving when a sync sub-loop renders; the design today
  stops the parent spinner rather than mux (`:979-981`; restart in the
  `finally` at `:1032-1033`).
- Agent identity plumbing EXISTS: `_run_sync_agent_body` builds
  `_SubagentContext(job_id=...)` (`_subagent.py:940-941`) and installs it
  in thread-local `_SUBAGENT_TLS` (`:946`; `_subagent_tls.py:17-21`), so
  a sink tap can resolve `getattr(_subagent_ctx(), "job_id", None)` on the
  agent's own thread — precedent `_turn.py:954`. CAVEATS: `_batch.py:251,299`
  and `_telegram.py:473-475` set quiet with NO ctx/job_id (tap needs a
  None fallback), and `_spec_reads` speculation runs on a pool thread
  where TLS does not propagate.
- `StreamParser` itself carries no job id (`_turn.py:496` passes only
  `quiet=`); no plumbing change is needed for async agents given the TLS
  read above.
- Latent hazard: `StreamParser.clear_tool_status()`
  (`_stream_parse.py:260-264`) writes CLEAR_LINE with NO quiet check —
  currently dead-safe only because `tool_status_active` cannot become
  true while quiet. Fix while touching these paths.

## Design

### 1. Event sink (`nx/_eventsink.py`, new module)

One focused module. No asyncio, no external deps.

```python
# Per-job ring buffer + subscriber registry.
class _EventSink:            # created per spawned agent
    def write(self, job_id, kind, text): ...
    def tail(self, job_id, lines=N) -> list[str]: ...
    def subscribe(self, job_id) -> Subscriber   # optional, v2

# Module-level registry + lock.
_SINKS: dict[str, _EventSink]
def sink_for(job_id) -> _EventSink
def drop_sink(job_id) -> None
```

- `kind` ∈ `{"stream", "tool", "status"}`: `stream` = model answer deltas,
  `tool` = tool-call start/finish lines, `status` = turn counters ("turn
  2/10", consult rounds, tier).
- Ring buffer of last `NX_EVENT_TAIL_LINES` (default 200) lines per sink,
  `threading.Lock`-guarded; `drop_sink` called from the worker's `finally`
  after the report is written, retaining the final tail inside the job
  record for post-mortem via `/tabs`. The retain-then-drop must happen
  under the same lock as the drop, so a concurrent 4 fps dashboard
  repaint cannot read a half-retired sink.
- **Writers**: (a) forks at the `StreamParser` chokepoints
  (`_handle_content` and the Live render path) — the job id comes from
  `getattr(_subagent_ctx(), "job_id", None)` (None when quiet is set by
  `_batch`/`_telegram`, in which case no sink is attached); (b)
  `_progress_step` (`_subagent.py:984-1021`) gains a `status` emit; (c)
  `_dispatch` quiet
  branch emits one `tool` line per call. All writers are behind a
  `sink is not None` check — zero cost when no dashboard is open.
- The sink is **display-only**. It never enters `messages`, never reaches
  the model, and is not persisted (except the retained tail in the job
  record). This keeps the `_runtime_notes.py:8-13` append-only contract
  untouched.

### 2. `/tabs` becomes live (`nx/_surfaces.py`)

- `run_tabs_dashboard` gains a `live` mode entered mid-turn: an ANSI
  region **painted below the current render line** using the existing
  repaint idiom (`\x1b[{n}A\r\x1b[J`, `_surfaces.py:261-265`), redrawn at
  4 fps from a poll of `_SINKS` tails + `_surfaces_snapshot()`. The main
  turn's own rendering continues above it — the dashboard is a bounded
  overlay (max `NX_TABS_HEIGHT` rows, default 8), not an alternate screen.
- **Write serialization**: a module-level `threading.RLock`
  (`_WRITE_LOCK` in `_surfaces.py`) taken by the dashboard repaint and —
  in a hardening commit that is REQUIRED before v1 ships (not a follow-up:
  the flicker guardrail below is false until it lands) — by the ~56 raw
  print sites in `_turn.py`/`_stream_parse.py`/`_dispatch.py`. This is a
  review-heavy change, not a mechanical swap: writes are scattered across
  ≥6 StreamParser methods plus the turn/dispatch bodies, behind 20+
  independent quiet checks.
- **Editor ownership**: the chatbox is the sole stdin reader and sole
  rewriter of its own frame (single-reader invariant,
  `_interrupt.py:151-153`) while at the prompt; the interrupt watcher is
  parked (hands stdin off) exactly then, so a live region below the
  composer MUST be repainted from inside the chatbox's own loop (e.g. a
  `wait_ready` timeout on its blocking read), never from an external
  thread. Mid-turn live mode while the turn loop (not the chatbox) owns
  the terminal may use the armed watcher's poll tick instead.
- **Agent-row tails** come from `_eventsink.tail()` instead of being
  empty while running.
- **Tear avoidance** (grounded in Gemini CLI's
  `useGeminiStream.ts` TODO): a contiguous block of agent rows is only
  flushed from "running" to "done" styling when the whole block is
  terminal, so finishing agents don't visually reorder against still
  running siblings. Row ordering is `dict` insertion order of
  `_ASYNC_AGENTS` (spawn order); the block rule is enforced over that
  ordering — no separate sort key.

### 3. Mid-turn Ctrl-T (`nx/_interrupt.py`, `nx/_chatbox.py`)

- `_capture_steering_byte` (`_interrupt.py:255-283`) gains an explicit
  `\x14` (Ctrl-T) check before the `ch >= " "` printable test at `:280`
  (today `\x14` is silently DROPPED there, not passed as steering text).
  On match it calls `nx._surfaces.toggle_dashboard()` and consumes the
  byte.
- `toggle_dashboard()` is idempotent: opens the live region if closed,
  closes/clears it if open. Esc (or a second Ctrl-T) closes. The repaint
  driver depends on terminal ownership (see Editor ownership in §2):
  chatbox-owned → timeout in the chatbox read loop; turn-loop-owned →
  the armed watcher's 100 ms poll. There is NO pre-existing general tick
  thread; if neither hook proves viable, a single dedicated repaint thread
  is the accepted fallback (revising the "No new threads" guardrail).
- The idle-prompt full dashboard (`_repl.py:82`) keeps its current
  behavior unchanged.

### 4. Schema / surface budget

No new tools, no schema changes → prompt floor unchanged (guard:
`tests/test_initial_context.py` `BUDGET_TOKENS`). `/tabs` remains a REPL
command, not a model-visible tool. New env vars:
`NX_EVENT_TAIL_LINES` (default 200), `NX_TABS_HEIGHT` (default 8). Both
require the CONTRIBUTING.md env-var touch points (docs +
`sources.example.env`) at implementation time.

## Guardrails

- **Zero console output when closed.** With the dashboard closed, sink
  writes are appends to a bounded ring; nothing prints and `quiet`
  semantics are untouched. The flicker class of bug (`_subagent.py:964-978`)
  cannot regress *while the dashboard is closed*; while it is open, parent
  prints interleave with repaints until the hardening commit lands (hence
  it is v1-required, see §2).
- **Zero cost when unused.** All sink emission is behind a
  `sink_for(job_id) is not None` check; agents spawned with no dashboard
  ever opened pay one dict lookup per render event.
- **Transcript purity.** Sinks are unreachable from the model; no
  `role:"user"` injection changes; `_flush_pending_background_notice`
  call sites untouched.
- **No new threads (preferred).** Repaint piggybacks on the chatbox read
  loop or the armed watcher poll (see §2/§3). If neither hook works, one
  dedicated repaint thread is the accepted fallback. The single
  `rich.live.Live` and spinner ownership stay with the main thread.
- **Termination.** Dashboard closes on turn end (`finally` in
  `_repl.py`'s turn block), on agent set emptying, and on terminal
  restore paths (`_chatbox` teardown); a crash in the repaint is caught
  and disables the dashboard for the session (belt-and-braces against
  termios corruption).
- **Windows/VT fallback**: if the terminal doesn't report VT support
  (reuse `_tty` detection), mid-turn live mode is refused and Ctrl-T
  falls back to the existing idle-prompt dashboard.

## Testing

- New `tests/test_eventsink.py`: ring bounds, `drop_sink`, tail ordering,
  concurrency (N writers under lock), zero-cost path when closed.
- New `tests/test_tabs_live.py`: toggle idempotence, mid-turn Ctrl-T
  capture (extend the existing steering-capture tests in
  `tests/` for `_interrupt.py`), repaint region bounds, agent-block tear
  avoidance rule, quiet-path non-interference (spawn an async agent with
  sinks and assert parent console output is byte-identical to today).
- Existing suites that must stay green untouched: `tests/test_subagent*`
  (quiet semantics), `tests/test_initial_context.py` (budget).
- Manual matrix: 4 concurrent agents (semaphore cap) with dashboard open
  during a parent turn; Ctrl-T open/close spam; resize during repaint;
  `NX_TABS_HEIGHT=1` and `NX_TABS_HEIGHT=1000` clamps.

## Deferred: daemon detach (v2, not this spec)

True detach/reattach (codex-style) is a **process architecture**, not a
display feature: a server process owning `st` + all threads, an attach
protocol streaming what `/tabs` enumerates, relaxed `owner_pid` filter
(`_bg_completions.py:52`) and cross-process session/file claims
(`_sessionlock`, `_fileclaim`), flush-on-detach via
`_flush_save()`/`begin_exit()` (`_repl.py:292-299`). Resume-from-disk
(`nx --resume`) covers the cold case today, so v2 is only worth building
behind real demand. This spec's event sink is the natural wire-format
seed for that protocol, which is the main reason to build Tier 1 first.
