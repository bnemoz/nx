# Spec: auto-continuation — stall wakes, cost bounds, batch chaining

Status: PROPOSED (targets 0.159.3 → 0.162.0). Nothing here is implemented.
Cited file:line verified against 0.159.2 (`ce5d893`) by two read-only sweeps;
the three formerly-**[uv]** items are since verified (§6) and both §6
decisions are settled. Phase A corrected: the tuple-vs-registry delta is
**three** slots, not two (MULTI_EDIT and DELEGATION are `st` module attrs,
not `_NOTICE_DEFAULTS` keys).

## 0. The mechanism as it exists

Auto-continuation is **completion-triggered only**. Every wake path bottoms out
in `wake_due()` (`nx/_autowake.py:225-296`): True iff a `[Runtime note]` is
staged, or a bg-completion sweep stages one.

- One in-process daemon poller (`_autowake.py:301-342`, `nx-wake-poller`,
  interval `NX_WAKE_POLL_SECONDS` default 2.0, floor 0.2) flips `st._WAKE_DUE`
  (`:340`). It never starts a turn — that split is load-bearing
  (`_autowake.py:302-305`).
- `arm_wake()` (`_autowake.py:139-146`) is the only entry; called on the first
  `[background]` run_bash / fused edit (`_dispatch.py:800-801`) and on async
  spawn (`_subagent.py:989`).
- `st._WAKE_DUE` has exactly two consumers: the chatbox, which returns the
  `WAKE_INPUT` sentinel (`_chatbox.py:526,951`), and `_repl.py:450-468`, which
  turns it into a synthetic user turn (banner `▸ background job finished —
  auto-continuing`, no slash dispatch, no input-history entry). Plus the
  Telegram mirror (`_telegram.py:923-959`).
- Chain cap: `NX_WAKE_STREAK_LIMIT` default 25, clamped ≥1
  (`_autowake.py:98-103`), enforced by early-return at `_autowake.py:249-250`.
  Incremented at `_repl.py:466` (before `wake_instruction()`, so `n` is
  1-based); reset to 0 on every non-wake input at `_repl.py:452-455` (before
  slash dispatch, so slash text counts as taking the wheel). Initial 0 at
  `_state.py:275`.
- Instruction selection: `_campaign_pending()` (`_autowake.py:106-115`) =
  `st._TODOS` has a non-`done` item AND `st._SPAWNED_WRITE_TIER`
  (set at `_subagent.py:1595`). Selects `CAMPAIGN_WAKE_INSTRUCTION`
  (`_autowake.py:80-95`, stamps `wake {n} of {m}`) over `WAKE_INSTRUCTION`
  (`:67-73`). **It chooses wording only — it never causes a wake.**
- Turn-end seams available to new logic: the `status == TURN_DONE` block
  `_turn.py:1145-1173` (shared by every driver), and the REPL post-turn
  `finally` `_repl.py:541-564` (already re-checks `_any_pending_notice()` at
  `:554-555`). `nx/_hooks.py` has **no** turn-end hook — only per-tool
  pre/post (`_hooks.py:68`, `:87`).

## 1. Why this spec exists

Four failures all trace to the single trigger:

1. **A campaign stalls silently.** If a wake turn ends without work in flight
   (agent FAILED and its note already delivered; model marks a todo
   `in_progress` then answers in text; model never spawns the next phase),
   `wake_due()` sweeps, finds nothing, returns False — permanently. Pending
   todos never trigger a wake.
2. **The cap counts wakes, not cost.** 25 unattended wakes is unbounded in
   tokens; a merge-and-review wake is a heavy turn.
3. **Cap exhaustion is invisible.** `_autowake.py:249-250` returns False with no
   message; the `wake {n} of {m}` stamp only exists inside a wake that already
   happened, so the model never learns the chain stopped either.
4. **`nx run` cannot chain.** `wake_due()`'s batch guard
   (`_autowake.py:255-261`) suppresses wakes, and neither consumer exists under
   `nx run` — the unattended-campaign story is interactive-only.

Plus two latent fragilities that E-phases build on (§6).

## 2. Design invariants (hold these in every phase)

- **Poller decides, REPL acts.** Never move turn-start into the poller thread.
  The 0.109.5 guards (`_autowake.py:334-340`, `_repl.py:464`, `_repl.py:506`)
  exist because the REPL is busy whenever the state could race.
- **Every new trigger self-terminates by construction.** No new unbounded
  counters, no sleeps, no second streak cap. A trigger that cannot prove it
  stops must not ship.
- **A wake must carry information.** If a wake could legitimately be answered
  "nothing to do", it is a phantom wake — the class of bug 0.36.3 / 0.37.1 /
  0.37.4 fixed. New triggers need a "deliberate stop" escape hatch in the
  instruction text.
- **`st._TODOS` is the single plan of record.** New logic reads it; nothing
  adds a parallel plan structure or a schema field to `todo_write`.
- **New residue goes in `nx_state_reset`** (`tests/_nxtest.py:303+`), once,
  centrally — per AGENTS.md, never in the failing test.

---

## Phase A — 0.159.3: notice-slot derivation (patch, no behavior change)

**Problem.** `_pending_notice_slots()` (`_autowake.py:151-155`) hard-codes
**7** slot names. `st._NOTICE_DEFAULTS` (`_state.py:517-523`) holds only **4**
`_PENDING_*` notice slots (7 dict keys, two of which — `_BG_SCAN_DONE`,
`_BG_NOTICE_SEEN` — are not notices): `_PENDING_BG_COMPLETION_NOTICE`,
`_PENDING_SUBAGENT_NOTICE`, `_PENDING_VERIFY_NOTICE`,
`_PENDING_YOLO_ASYNC_NOTICE`. The tuple adds **three** that are not in the
registry — `_PENDING_BACKGROUND_NOTICE` (no `_state` attr at all; a plain
module global staged at `_dispatch.py:796`, read at `_autowake.py:165-170`),
`_PENDING_MULTI_EDIT_NOTICE` (`_state.py:293`, module attr) and
`_PENDING_DELEGATION_NOTICE` (`_state.py:322`, module attr). The two lists
already disagree, and a notice type added to `_NOTICE_DEFAULTS` cannot wake
the REPL. This bites the moment Phase B adds a slot.

**Change.**
1. `_autowake.py`: derive the tuple from `st._NOTICE_DEFAULTS` keys ∪ every
   `_PENDING_*` attr on `st` (the same `dir()` oracle `nx_state_reset_body`
   uses, `tests/_nxtest.py:551-552`) ∪ `{"_PENDING_BACKGROUND_NOTICE"}` (which
   has no `_state` attr; it stays a plain module global, staged at
   `_dispatch.py:796` and read at `_autowake.py:165-170` — keep that special
   case). A derivation limited to `_NOTICE_DEFAULTS | {BACKGROUND}` would
   silently drop `_PENDING_MULTI_EDIT_NOTICE` and `_PENDING_DELEGATION_NOTICE`.
   Add a module-level `_NON_WAKE_NOTICE_SLOTS: frozenset[str]` for slots that
   deliberately must not wake, with a one-line reason per entry.
2. New test in `tests/test_autowake.py`: every `_PENDING_*` slot reachable from
   `st._NOTICE_DEFAULTS` ∪ `{_PENDING_BACKGROUND_NOTICE}` is either wake-worthy
   or listed in `_NON_WAKE_NOTICE_SLOTS`. This test is the point of the phase —
   it converts a silent omission into a failure.

**Resolved delta (already enumerated — do not re-derive):** derivation is
`set(st._NOTICE_DEFAULTS) | {s for s in dir(st) if s.startswith("_PENDING_")} |
{"_PENDING_BACKGROUND_NOTICE"}`; the plain global stays a special case in the
tuple *and* in `_any_pending_notice()`'s read path (`_autowake.py:165-170`).
**Decided (§6.3):** `_PENDING_DELEGATION_NOTICE` and `_PENDING_MULTI_EDIT_NOTICE`
are NOT promoted into `_NOTICE_DEFAULTS`. `_notices.py` freezes `SLOTS` from
`_NOTICE_DEFAULTS` keys at import time (`_notices.py:34`) and `NoticeFrame`
seeding iterates `SLOTS` only — promotion would route both through frame
isolation and change their behavior (re-check `_notices.py:79-143` and
`tests/test_notice_seed_delivered_clobber.py`); the `dir(st)` oracle picks them
up without promotion. After Phase A, `_PENDING_BACKGROUND_NOTICE` should be the
only hand-listed entry left.

Confirmed: `_notices.get/set_` (`_notices.py:146-164`) seeding is unaffected
by the derived list in `_autowake` — seeding has its own frozen `SLOTS`.

**Files:** `nx/_autowake.py`, `tests/test_autowake.py`.
**Risk:** low. Watch `tests/test_state_reset_contract.py` (it asserts the reset
drains every `_PENDING_*` slot — derived and hard-coded agree).

---

## Phase B — 0.160.0: say why the chain stopped + chain visibility

**Problem.** Guard trips are silent; nothing tells the user or the model that
auto-continuation has parked, and no surface shows chain state.

**Change.**
1. New slot `_PENDING_WAKE_HOLD_NOTICE` (add to `st._NOTICE_DEFAULTS`,
   `_state.py:519-523`, so Phase A's derivation picks it up automatically).
2. `_autowake.py`: one function `_stage_hold_notice(reason: str) -> bool`,
   idempotent per (reason, streak value) so the 2s poller cannot restage it
   every tick. Stage a dim one-liner, e.g.
   `▸ auto-continuation stopped: wake streak 25/25 — type to continue`.
   Call it from the two guard trips in `wake_due()` (`:249-250` streak cap,
   and the budget guard added in Phase C) — not from the poller's hot path
   beyond that.
3. Delivery: rides the existing flush
   (`_runtime_notes.py:60-117`) so it lands as a `[Runtime note]`-shaped
   `user` message at the next prompt, and the next `wake_due()` sees the slot
   staged → returns True at `:262` → **the hold notice would itself cause a
   wake**. Break that loop explicitly: the hold slot must be in
   `_NON_WAKE_NOTICE_SLOTS` (Phase A) *and* excluded from `_any_pending_notice()`
   (`_autowake.py:158-171`). `_any_pending_notice()` also feeds
   `_repl.py:554` (post-turn flag raise) and `_telegram`, so the exclusion must
   be in the predicate, not at one call site. This is the one real trap in
   Phase B.
4. `/status` (`nx/_commands.py:82` handler — find the builder in
   `nx/_repl_status*.py`/`_repl_settings_cmds.py`, not `_commands.py` itself,
   which is registry-only): add one line
   `auto-continue: armed · wake 3/25 · plan 4 pending · held: budget`.
   Read-only fields, all already in `st`. No new slash command.

**Tests.** `tests/test_autowake.py`: hold notice staged on cap trip; staged
twice → staged once; hold notice alone never satisfies `_any_pending_notice()`
(this is the anti-loop regression test); notice delivered exactly once.
`tests/test_state_reset_contract.py`: new slot drained.
`/status` snapshot test wherever the existing `/status` assertions live.

**Files:** `nx/_autowake.py`, `nx/_state.py`, the `/status` builder,
`tests/test_autowake.py`, `tests/test_state_reset_contract.py`.

---

## Phase C — 0.160.0 (or 0.161.0): plan-stall wake

The real feature. Land after B so a stalled-by-budget/stalled-by-cap chain
explains itself when the new trigger's own guards trip.

**Change.**
1. `nx/_state.py`: `st._WAKE_STALL_AT: float = 0.0`,
   `st._WAKE_STALL_REVISION: int = -1`, `st._WAKE_STALL_HOLD: bool = False`.
   All three must be added to `nx_state_reset`.
2. Plan revision helper in `nx/_todo.py` (the plan's home, and `todo_write` is
   the only writer — `_todo.py:85-102` replaces `st._TODOS` wholesale at
   `:92`): `_plan_revision() -> int`, a stable hash of `(text, status)` pairs.
   Compute in `todo_write` and stash the value on `st` rather than hashing in
   the poller every 2s.
3. Stamp the stall clock in the REPL post-turn `finally` (`_repl.py:541-555`,
   the block that already re-checks notices at `:554`): when the turn ended
   `done` and `_campaign_pending()` and no bg job / async agent is in flight,
   `st._WAKE_STALL_AT = time.monotonic()`; else `0.0`. Mirror in
   `_telegram.py:923-959`.
4. `wake_due()` — the new branch goes **after** the existing sweep and the
   existing `:296` fallthrough return, as the last resort before `False`, so
   every existing "nothing to say" conclusion still short-circuits:

   ```
   if _campaign_pending() and _stall_due():
       return True
   return False
   ```

   `_stall_due()` is True iff all hold:
   - `_WAKE_STALL_AT` non-zero and older than `NX_WAKE_STALL_GRACE_SECONDS`
     (default 20.0 via `_env_float`, floor 0 — the grace period is the user
     returning to the keyboard);
   - `_plan_revision() != st._WAKE_STALL_REVISION` — **at most one stall wake
     per plan revision**. This replaces a new cap. If a stall wake changes
     nothing (no todo edit, no new spawn), the revision is unchanged, the next
     evaluation is False, and the trigger is dead until the user types;
   - not `st._WAKE_STALL_HOLD`.
   Stamp `_WAKE_STALL_REVISION` when the wake fires (same place the streak
   increments, `_repl.py:466`) so the model's own turn can trip the hold.
5. Clear `_WAKE_STALL_AT = 0.0` and `_WAKE_STALL_HOLD = False` on the typed-input
   reset path (`_repl.py:452-455`), next to `_WAKE_STREAK = 0`.
6. `_WAKE_STALL_HOLD` is set when a stall wake ends with no revision change and
   no newly armed work — implemented where both facts are cheap: the same
   post-turn `finally`, comparing the revision stamped at wake time.
7. New `STALL_WAKE_INSTRUCTION` in `_autowake.py`, selected in
   `wake_instruction()` (`:118-128`) when the wake is a stall wake (stash a
   flag on `st` when the branch fires, or infer from
   `_WAKE_STALL_AT != 0`). Must contain, explicitly: the pending todo items by
   text, and the deliberate-stop escape hatch — *"if you stopped because you
   need the user, or because the plan is complete, say so in one line and take
   no action."* Worst case for a deliberate stop is then one wasted turn per
   plan revision, which is the accepted cost.

**Env vars:** `NX_WAKE_STALL_GRACE_SECONDS` (default 20.0, `_env_float`,
floor 0 disables the stall wake — document it that way, it is the off switch
for this trigger; `NX_WAKE_PROMPT=0` remains the global off switch). README env
table row is **mandatory** — `tests/test_env_docs_parity.py` fails on any
undocumented `NX_*`.

**Tests** (`tests/test_wake_streak.py` is the closest sibling; a new
`tests/test_wake_stall.py` is justified given the size):
- campaign pending + grace elapsed + revision changed ⇒ `wake_due()` True with
  nothing staged (the inverse of every existing test in `test_autowake.py`,
  which all assert False here);
- same but revision unchanged ⇒ False (one wake per revision);
- second consecutive no-change wake sets `_WAKE_STALL_HOLD`; typed input clears
  it;
- grace not elapsed ⇒ False;
- todos all `done` ⇒ False (no campaign);
- pending todos but no `_SPAWNED_WRITE_TIER` ⇒ False (interactive working list
  must not be nagged — this is the `_campaign_pending` contract, `_autowake.py:106-115`);
- a live bg job or running async agent ⇒ the existing liveness guards win (the
  stall branch must sit behind them);
- streak cap tripped ⇒ stall branch cannot bypass the cap (it sits after the
  `:249` early-return, but pin it);
- instruction text names the pending items and carries the escape hatch.

**Files:** `nx/_autowake.py`, `nx/_todo.py`, `nx/_state.py`, `nx/_repl.py`,
`nx/_telegram.py`, `tests/_nxtest.py`, README, new test file.
**Risk:** highest in the spec — it fires model turns on a condition that is by
design *absence* of evidence. Ship behind `NX_WAKE_STALL_GRACE_SECONDS=0` off
switch, and consider defaulting off for one release (default `0` = disabled,
document it as opt-in) if any phantom-wake report lands. The suite is offline;
none of these tests can catch a real-model nag loop, so the escape hatch and the
revision terminator are the only defenses.

---

## Phase D — 0.161.0: chain cost bound

**Problem.** 25 wakes is a turn-count bound; cost per wake varies ~100×.

**Change.** One more guard in `wake_due()` beside `_autowake.py:249`, plus its
hold notice from Phase B (`held: budget`).

- Chain spend accumulator `st._WAKE_CHAIN_SPEND: int = 0`, `st._WAKE_CHAIN_BASE:
  int = 0`. On wake start (`_repl.py:466`) snapshot
  `st._WAKE_CHAIN_BASE = st._LAST_PROMPT_TOKENS`; at the same post-turn `finally`
  add the delta. `st._LAST_PROMPT_TOKENS` is captured per step
  (`_turn.py:764`, `_capture_last_prompt_tokens`) and the loop already reasons
  over it (`_turn.py:1059`, `:1224`) — the number is on hand, no new estimator.
  Use `nx/_tokens.py` (0.159.2) only if prompt tokens are unavailable, not as
  the primary source.
- `NX_WAKE_CHAIN_TOKEN_BUDGET`, `_env_int`, **default 0 = off**, so no behavior
  change for existing users. Clamp negatives to 0.
- Reset both accumulators on the typed-input path with the streak.

**Tests:** spend accumulates across a chain; guard trips at budget with the hold
notice; budget 0 never trips; typed input resets. Extend
`tests/test_wake_streak.py` (it already owns cap/default/clamp contracts).

**Files:** `nx/_autowake.py`, `nx/_state.py`, `nx/_repl.py`,
`tests/test_wake_streak.py`, `tests/_nxtest.py`, README.

---

## Phase E — 0.162.0: `nx run --autowake` (headless chaining)

**Do the prerequisites first; they are bugs on their own.**

1. **[uv] Per-task notice ownership.** With `--parallel >1` the notice channel
   is process-global and completion notes can cross tasks (`_batch.py:44-48`).
   The `NoticeFrame` / `install_notice_ctx` / `pop_notice_ctx` machinery
   (`_notices.py:79-143`) exists for exactly this and is already used by
   parallel in-process agents; batch tasks are not framed. First *reproduce* it
   (a two-task batch with one async agent), then frame per task.
   `tests/test_notice_flush_ownership.py` is the pattern to extend.
2. **Per-task background-job registry.** The sweep attributes jobs by
   `owner_pid != os.getpid()` (`_bg_completions.py:24-88`) — every task in a
   process sees every other task's jobs, and batch never kills or reaps
   background jobs at task end (no kill path in `_batch.py`). A chained task
   would accumulate detached jobs across its own turns. Needs a task-scoped job
   set threaded through the sweep, mirroring how `wake_due()` already threads a
   single `list_bg_jobs()` result (`_autowake.py:275-289`).
3. **New flag** `--autowake` on `nx run`, and `autowake: true` in the task
   object, off by default. Off = today's behavior exactly.
4. **Batch wake loop, on the worker thread.** `wake_due()`'s batch guard
   (`_autowake.py:255-261`, `_batch_active()` at `_batch.py:301-307`) is
   thread-local via `_TASK_TLS.task` (`_batch.py:315`), so on the worker thread
   it reports True and blocks. Give `_autowake` a batch entry point
   (`batch_wake_due(jobs=None)`) that skips the interactive-only liveness
   guards and keeps the notice/sweep logic, called between turns by the batch
   driver when `autowake` is set. No streak cap here: the bound is the
   task deadline (`_turn_retry.py:45-80`, armed `_batch.py:469`, evaluated only
   at the step boundary `_turn.py:1103-1104`) plus
   `MAX_MODEL_TURNS` 500 (`_turn.py:971`) — both already exist, both are
   wall-clock/turn bounds the streak cap was standing in for.
5. The synthetic wake turn in batch appends the same `user`-role instruction
   and flushes notes the way `_repl.py:507-517` does; the answer channel stays
   the task's own transcript (`_batch.py:44-48`).
6. Document the interaction with `NX_TASK_TIMEOUT_SECONDS` (default 0 = off,
   `_batch.py:81`): **autowake with no timeout is an unbounded headless agent.**
   Either require `--timeout` when `--autowake` is set, or auto-arm a default
   and say so in the JSONL output. Prefer refusing: fail the task with a clear
   error over silently spending forever.

**Tests** (`tests/test_batch_run.py` already asserts the bg scan is gated off
for batch — that assertion is the contract this phase deliberately changes, so
expect to update it, not only add): autowake off ⇒ today's behavior byte-identical;
on ⇒ a finished async agent produces a second model turn inside one task;
deadline tripped mid-chain ⇒ `status: error`, `timed_out: true`, exit 4
(`_batch.py:72`, `_result_exit_code`), no further wakes; `--parallel 2` with
autowake ⇒ notes never cross tasks (the regression test for prerequisite 1).

**Files:** `nx/_batch.py`, `nx/_autowake.py`, `nx/_notices.py`,
`nx/_bash_jobs.py`, `nx/_cli.py`, `nx/_bg_completions.py`,
`tests/test_batch_run.py`, `tests/test_notice_flush_ownership.py`, README,
`docs/BATCH_TASK_OVERRIDES_SPEC.md`.
**Risk:** highest blast radius (headless, unattended, spends money with no
keyboard). Ship last, behind a flag, only after A-D are in a release.

---

## 3. Out of scope, and why

- **A `wake_write`/`schedule_wake` tool.** Tool touch points are ~8 files
  (CONTRIBUTING.md:77-118 + 368-412: `TOOLS` schema, `DISPATCH`, `_TOOL_GATES`
  + `_GATE_REFUSALS`, `_assess_risk`, `SYSTEM` sync,
  `test_tool_schema_invariants`, `test_initial_context` `BUDGET_TOKENS` ledger,
  `test_unknown_tool`, `test_perf_tiered_registry`'s hard-coded gated names,
  `_argrepair` aliases, `_argsalvage.FALLBACK_FOR`). The plan already lives in
  `st._TODOS`; a second source of truth the model must keep in sync is worse
  than reading the one it has.
- **`todo_write` schema changes** (a `blocked` status, a `waiting_on` field).
  `status` is not required (`__init__.py:428`; body defaults it at
  `_todo.py:88-91`) and "exactly one in_progress" is prompt-only — a new enum
  value costs floor tokens on every request for a signal `_campaign_pending()`
  does not need.
- **Raising `NX_WAKE_STREAK_LIMIT` past 25.** With Phase C's terminator and
  Phase D's budget guard, the cap stops being the only brake.
- **Moving turn-start into the poller thread.** §2 invariant 1.
- **Telegram-side feature work.** `_telegram.py:923-959` mirrors the REPL wake
  loop; Phases C/D must be *mirrored* there for consistency but get no new
  Telegram-only behavior.
- **Cross-session wake handoff** (session A arms a wake that fires in a future
  session). `sessions.data` could carry it (the small-meta blob,
  `_sessiondb.py:245-247`) but a wake whose context was auto-compressed or
  resumed into a different cwd is a phantom wake with a bigger blast radius than
  Phase C's. Needs its own design.

## 4. Pre-existing defects found while mapping (fix or delete; do not build on)

1. **`NX_OCC` is wired at both ends and disconnected in the middle.** Re-verified
   by call-site grep (not [uv]). Live: `inspect_todo_write()` from `todo_write`
   (`_todo.py:98-99`) and `record_provider_request()` from the turn loop
   (`_turn.py:1228-1229`). Missing:
   - `record_boundary()` (`_occompact.py:107`), the only writer of
     `_OCC_PENDING_BOUNDARY` (`:114`), has **no caller in `nx/`** (only the
     re-export at `__init__.py:1013`). The value its caller needs is exactly the
     return value `todo_write` discards at `_todo.py:99-101`.
   - `maybe_compact_after_boundary()` (`_occompact.py:229`, "the turn_end seam")
     is called **only from tests** (`tests/test_sol_pi_mechanisms.py:307,365,382`).
     It early-returns False unless `_OCC_PENDING_BOUNDARY` is set (`:233`), and
     nothing sets it outside `tests/test_sol_pi_mechanisms.py:357` — so it is a
     permanent no-op in production.

   README:388 documents the whole `NX_OCC` behavior as shipped. **The wiring is
   ~3 lines**: capture `inspect_todo_write`'s return in `todo_write` →
   `record_boundary(newly, remaining)`; call `maybe_compact_after_boundary()` in
   the `status == TURN_DONE` block at `_turn.py:1145-1173`, beside the existing
   `_launch_verify_if_settled()` seam (`:1164-1172`), gated off for sub-agents
   and batch the same way. Do it as its own release **before Phase C** (A′ in
   §5): C lengthens unattended chains, and a long chain with no plan-boundary
   compaction is exactly the case `NX_OCC` exists for. **Decided (§6.4):** wire
   the seam, do not delete — deleting contradicts README:388 and strands the
   tested economics machinery in `_decide`.
2. **`load_persisted_todos()` (`_todo.py:120`) is dead in `nx/`** — cross-session
   todo pickup was removed in 0.150.x (`_repl_state.py:208-211`: the file is
   "a write-through export only"). Only `tests/test_094_features.py:104,110`
   calls it. **README:440 says the opposite**: "where `todo_write` mirrors its
   list **so fresh sessions pick up the plan**" — a docs claim contradicted by
   `_repl_state.py:208-211` since 0.150.x. Correct the README row (session todos
   restore from `sessions.data`, `_repl_state.py:205-207`) or delete the dead
   loader. Relevant here because `todo.md` reads like a cross-session wake
   schedule to a newcomer — it is not one, and Phase C deliberately does not make
   it one.
3. **[uv] `_batch_active()` is thread-local but called from the poller.** When
   `wake_due()` runs on the poller thread during a batch run, `_batch_active()`
   is False (`_TASK_TLS` is per-thread), so the poller's guard at
   `_autowake.py:253-261` does not see the batch. Harmless today only because no
   batch consumer exists; it becomes a real hole the moment Phase E lands.
   Phase E must key off an explicit parameter, not the ambient predicate.
4. **[uv] `--parallel >1` note cross-talk** — Phase E prerequisite 1.
5. **[uv] No wake interaction found in `nx/_cli.py` one-shot / `--prompt-file`
   paths** (absence of a call site, not a reproduction). Grep for
   `_WAKE_DUE`/`wake_due` in `_cli.py` before Phase E to confirm one-shot runs
   neither arm nor consume wakes.

## 5. Sequencing

| Release | Phase | Content | Size |
| --- | --- | --- | --- |
| 0.159.3 | A | notice-slot derivation + invariant test | S |
| 0.159.3 | A′ | wire the `NX_OCC` boundary seam (~3 lines) or delete it + fix README:388 | S |
| 0.160.0 | B | hold notice on guard trip + `/status` chain line | S |
| 0.160.0 | C | plan-stall wake (grace + revision terminator + hold) | M |
| 0.161.0 | D | `NX_WAKE_CHAIN_TOKEN_BUDGET` | S |
| 0.162.0 | E | `nx run --autowake` (2 prerequisites first) | L |

Every phase: `uv run pytest tests/ -n 8` clean, `uv run ruff check .` clean,
`uv run ty check nx` with no new diagnostics in touched files, README env row
for any new `NX_*` (test-enforced), new `st` residue drained in
`nx_state_reset`, and per AGENTS.md only one full-suite run per host at a time.
Semver per AGENTS.md: A/A′ are patches, the rest minors.

**A′ goes before C on purpose.** C makes unattended chains longer; a longer chain
with no plan-boundary compaction is the expensive case. If A′ instead lands as
"delete `_occompact`'s dead half", C's stall wakes simply have no boundary
compaction to assume — but decide which before writing C.

## 6. Verify-first list (do these before writing code)

1. ~~Reproduce the `--parallel >1` note cross-talk~~ **done**: it is already
   documented in the shipped `_batch.py:44-48` docstring; still reproduce with
   a live two-task test before Phase E lands (prerequisite 1 stands).
2. ~~Grep `_cli.py` for `_WAKE_DUE`/`wake_due`~~ **done**: zero matches —
   one-shot and `--prompt-file` runs neither arm nor consume wakes.
3. ~~Decide `_PENDING_DELEGATION_NOTICE`~~ **decided**: keep derived via the
   `dir(st)` oracle, no promotion (see Phase A's resolved delta).
4. ~~Decide the `_occompact` seam question~~ **decided**: wire the ~3-line
   seam as release A′, before C (see §4.1).
5. ~~Confirm `_notices.py` seeding~~ **done**: `SLOTS` is frozen from
   `_NOTICE_DEFAULTS` at import (`_notices.py:34`); a derived list in
   `_autowake` does not affect `NoticeFrame` seeding.
