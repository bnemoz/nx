# Spec: thread-local approval posture for sub-agents

Status: IMPLEMENTED (0.155.0) — with one deviation, recorded at the end of this
file.
Motivated by: 0.154.7 flake ledger (§5.0 of test-architecture.md) — the three
`test_preview_visibility` rerun-tuple entries; root cause located in the 0.154.7
census, not a theory anymore.

## 1. The bug, exactly

`nx/_subagent.py` forces the read-only-agent posture by writing process
globals:

- `:1223-1228` — under `_SPAWN_LOCK`, snapshot `prev_yolo, prev_level = st.YOLO,
  st.APPROVE_LEVEL`; read-only non-quiet agents assign `st.YOLO=True,
  st.APPROVE_LEVEL="high"`; write/quiet agents take the no-op branch.
- `:1325-1326` — restore in the `finally`.

Three defects:

1. **Leak window.** The swap sits *before* the `try:` at `:1294`. Anything that
   raises in between (spinner stop, `_install_model_target`, `note_progress`)
   skips the restore permanently: the process stays `YOLO=True` for every later
   test/turn on that thread. This is the confirmed preview_visibility trio
   cause (a later Tier-A test then hits `_confirm`'s YOLO fast path
   `_approval.py:349` and the approval preview never prints).
2. **Cross-thread visibility.** The async worker (`_run_async_agent` →
   `_run_sync_agent_body`) runs the same swap on the *worker* thread — against
   the *process* global. While a read-only agent runs, the parent thread's
   `_confirm()` reads the agent's forced posture. Accidental, not designed.
3. **No per-agent posture possible.** Batch/telegram workers read whatever the
   init set; an override per task/agent is impossible without this change.

No test pins the swap/restore directly (`_SPAWN_LOCK`/`prev_yolo`: 0 test
references) — only side effects, which is how the leak hid for this long.

## 2. Goal and non-goals

Goal: an agent's approval posture belongs to the agent, on its thread;
the parent's posture is never mutated by an agent, and cannot leak.

Non-goals (explicitly out):
- Per-agent approval *grants* (an agent that can be pre-approved for `write_file`
  only) — a security surface we don't need yet; posture stays the same
  all-or-nothing vocabulary.
- UI changes: the chatbox border/footer/panel keep showing the *parent's*
  posture (`_chatbox.py:652,663,1131`, `_panel.py:238`, `_repl.py:67-75` read
  the process globals and stay as-is). A sync read-only agent no longer
  cosmetically flips the border to yolo mid-run — this is a deliberate,
  minor behavior change: the UI tells the user *their* posture, truthfully.
- ContextVars. All agent/batch/telegram workers spawn bare threads, so a
  ContextVar would still need the same install sites and buy nothing.
- Batch per-task posture (now trivially possible later; not part of this work).

## 3. Design — one resolver, install-at-spawn, globals as fallback

House precedent is `_METRICS_TLS` (`_state.py:380`, installed per-task by
`_batch.py:341` after it replaced the identical swap-and-restore race):
`threading.local` + explicit install-at-worker-start + finally-restore +
resolver with process-global fallback. Same shape here.

### 3.1 State (`nx/_state.py`)

```python
_POSTURE_TLS = threading.local()          # .yolo / .level, installed per agent

def _posture() -> tuple[bool, Optional[str]]:
    """Live approval posture for THIS thread: TLS override if installed
    (agent body running), else the process globals. Live read is required —
    Shift+Tab at an approval prompt and /yolo /approval mutate the globals
    mid-turn and must keep working for the parent thread."""
    return (getattr(_POSTURE_TLS, "yolo", YOLO),
            getattr(_POSTURE_TLS, "level", APPROVE_LEVEL))

def _push_posture(yolo: bool, level: Optional[str]):
    """Context manager: install an agent-scoped override on this thread,
    restore in finally. Wraps the ENTIRE agent body (see 3.2) — the restore
    cannot be skipped because there is no code between push and try."""
```

`DEFAULT_APPROVE_LEVEL` keeps its only role (restore target for
`toggle_yolo`/`/yolo off`); it is not part of the TLS pair.

### 3.2 Producer (`nx/_subagent.py`)

Replace `:1223-1228` + `:1325-1326`:

- Sync + async + nested all converge on `_run_sync_agent_body`. It becomes:
  decide posture first (`force = read-only non-quiet → (True, "high")` else
  inherit → no override at all), then
  `with _posture_scope(force) if force else nullcontext(): <whole body in try/finally>`.
  For inherited posture install nothing — reading the globals live is exactly
  "inherit", so `test_subagent_write.py:311-333`'s contract passes through
  untouched.
- `_SPAWN_LOCK` was documented (swarm_optimization.md) as wrapping *only* the
  posture snapshot/restore. Verify no other section uses it (`_fileclaim`
  interactions at `:1318-1321` are already separately locked) and delete it;
  if anything else has crept in, keep it just for that.

### 3.3 Consumers — read-path migration (the full census, ~9 decision reads)

Route through `_posture()` (agent override matters):
- `_approval.py` `_confirm()`: stage 1 quiet early-return (`:334-352` reads
  `st.YOLO`), stage 2 YOLO fast path (`:349`), stage 4 level threshold
  (`:397`); `_approval_display` (`:114-118`) — the prompt line shows the level
  the decision used.
- `_subagent.py` gates: `:1106`, `:1124` (quiet write-tier needs yolo),
  `:1414` (yolo-async nudge), `:1518` (async write-tier needs yolo).
- `_tool_prefetch.py:178`, `_build.py:1002` (parallel work needs yolo) —
  these run inside agent bodies and must see the agent's posture.
- `_delegation_note.py:66` — verify intent: the note text should describe the
  *agent's* posture while an agent builds it; if it describes the session, it
  stays on globals. Decide with the test at §5.4.

Stay on the process globals (parent-facing):
- all UI/display reads listed in §2;
- `_apply_approval` / `toggle_yolo` / `/yolo` / `/approval` / CLI init /
  `nx eval` (these *are* the parent-posture writers);
- `_confirm`'s Shift+Tab handler (`_approval.py:237`) — it toggles the parent
  posture; an agent body must not have its forced posture toggled away.

Guards keep working by construction: `NX_BASH_PROMPT_SECS`/`NX_GIT_INDEX_GUARD`/
`NX_CLOBBER_GUARD` force their own prompts independent of `_confirm`'s YOLO
short-circuits; they run on the agent thread, where nothing about posture
resolution changes their code path. §5.5 pins it.

## 4. Why this fixes the flake structurally

- The leak window dies: install and restore are one context manager whose body
  is the agent body; a watchdog kill raises *inside* it, so the finally runs;
  there is no pre-`try` window left.
- The async cross-thread visibility bug dies: the worker's override lives in
  the worker's TLS; the parent's reads never see it.
- Test hygiene: `nx_state_reset_body` gains `_POSTURE_TLS.__dict__.clear()` —
  the drain contract (written once, 0.154.2) covers posture at last; today
  only Tier-B module-dict rewind cleans it, which is precisely what Tier-A
  files like `test_preview_visibility.py` don't get.

## 5. Acceptance and test plan

1. **Prune the tuple**: remove the three `test_preview_visibility` entries from
   `_FLAKY_RERUN_NODEIDS` (conftest) + rewrite the saga comment + the
   test-architecture.md ledger line as CLOSED. Bar: `tests/ -n 8` five runs,
   trio green every run (0.154.7's own bar; the fix must clear it to land).
2. **New pin — the leak window**: spawn a read-only sync agent whose body
   raises between the old swap site and the old `try` (patch
   `parent_spinner.stop` or the model-target install); assert `st.YOLO` and
   `st.APPROVE_LEVEL` are exactly the parent's before/after, and that a
   subsequent `_confirm` prompts (not auto-allows). This test FAILS against
   the current code — write it first, watch it fail, then fix.
3. **New pin — async isolation**: while a read-only async agent runs
  (`_ASYNC_QUIET_SLOTS`-style controlled body), the parent thread's `_posture()`
   reports the parent's globals. Today's
   `test_subagent_async.py:269-291` (worker asserts `yolo_during is False`)
   becomes vacuous — rewrite it to assert on the worker's TLS install instead.
4. **Resolver parity**: `_posture()` equals globals on a fresh thread; equals
   the override inside a body; mid-turn mutation of the globals by the parent
   is visible to the parent's `_posture()` (live read, Shift+Tab contract).
5. **Guard matrix under agent posture**: quiet read-only agent, forced
   posture, `NX_GIT_INDEX_GUARD` staged-WIP and `NX_BASH_PROMPT_SECRETS` —
   still refuse/prompt (mirrors `test_git_index_guard.py:70`,
   `test_bash_watchdog.py:98` on the agent path).
6. **Unchanged surface**: the ~39 `monkeypatch.setattr(nx/st, "YOLO", ...)` and
   ~64 direct-assign sites keep working because globals remain the fallback
   store; do NOT migrate them. `test_approval_env.py` (import-time precedence)
   and `test_telegram.py:808-816` (subprocess `--yolo`) are env-path —
   untouched. `tests/test_preview_visibility.py` itself: unchanged.
7. Full suite + `ruff` + `ty check nx` (no new diagnostics in touched files).

## 6. Docs touch points

- `docs/swarm_optimization.md:122-123,130-131` — describes the `_SPAWN_LOCK`
  posture swap (with stale line numbers): rewrite to the TLS scope.
- `docs/test-architecture.md` §5.0 ledger: trio CLOSED with the leak-window
  mechanism; update the `:107-109` pointers.
- `README.md` needs no user-facing change (posture UX unchanged); if §5.4's
  check changes `_delegation_note` semantics, revisit `:1105` wording.
- `CONTRIBUTING.md:338-340` — this is a behavioral approval change: full-suite
  bar already required.

## 7. Sequencing (one PR, four commits)

1. Failing pins (§5.2, §5.3-rewrite, §5.4) against current code.
2. State + resolver + producer swap-out (§3.1, §3.2); `nx_state_reset_body`
   drain; guard matrix (§5.5) green.
3. Tuple prune + ledger/doc rewrites (§6); five-run bar (§5.1).
4. Version bump (semver minor-for-product-behavior call: user-visible delta is
   the cosmetic border change and the async parent-visibility fix — propose
   0.155.0; maintainer may prefer 0.154.8).

Estimated diff: ~120 lines changed in `nx/` (state/approval/subagent/prefetch/
build/delegation), ~120 in tests, docs edits. No new files in `nx/`.

## 8. Risks

- **Missed decision read**: any decision-path `st.YOLO` direct read not in §3.3
  resolves parent posture during an agent. Mitigation: the census enumerated
  all of them; add a grep-guard test (`rg "st\.YOLO|state\.YOLO" nx/` minus an
  allowlist of parent-facing sites) so new sites trip in review.
- **`_delegation_note` semantics** (§3.3 open question) — resolved by §5.4
  before merge, not after.
- **Watchdog kills the thread *between* TLS-install and body** — impossible by
  construction in §3.2 (install is the context manager's `__enter__`, the body
  starts immediately); noted because 0.147-era watchdog flakes taught us to
  name it.

## 9. Implementation notes (0.155.0)

One deviation from §3.2's letter: the posture scope wraps ONLY the sub-loop call
(`_m._run_model_turn_loop`), not the answer/report processing after it. Reading
it against §3.2's own rationale ("install and restore are one context manager
whose body is the agent body"), this is equivalent and deliberate: the installer
is the context manager's `__enter__`, the loop's `try` begins immediately inside
it — there is no code between install and `try`, which is the entire point — and
the only thing the old snapshot/restore covered beyond the loop was its own
`finally` block, which touches no posture read. No other deviation.

§3.3's `_delegation_note.py:66` open question is settled as the spec assumed:
`_write_bg_sentence()` routes through `_posture()`, so a note built INSIDE an
agent body describes the AGENT's posture (pinned by
`test_delegation_note_describes_the_agents_posture`). `_tool_prefetch.py:178`,
`_build.py:1002`, the four `_subagent.py` gates and `_confirm`'s three decision
stages plus `_approval_display` all migrated to `_posture()`; the Shift+Tab
handler (`_approval.py:237`), the UI reads, `/yolo`, `/approval`, CLI init and
`nx eval` stay on the globals exactly as §3.3 lists.

Two additions beyond the spec's letter, both in its spirit:

- §8's grep guard is implemented as an AST scan
  (`tests/test_posture_readers_guard.py`) with the parent-facing allowlist —
  `nx/__init__.py` (CLI init + proxy), `nx/_eval_cmd.py`, `nx/_approval.py`
  (writers + Shift+Tab), `nx/_repl_settings_cmds.py`, `nx/_chatbox.py`,
  `nx/_repl.py`, and `nx/_state.py` as the store — plus a stale-entry test so
  the allowlist cannot rot into a blanket exemption.
- `tests/test_subagent_async.py`'s shared-worker hygiene bug was fixed in
  passing: `test_async_string_boolean_and_gating` assigned
  `nx._run_model_turn_loop` directly and relied on the fixture's old
  `importlib.reload(nx)` to undo it. With the reload gone the fake loop leaked,
  so every later sub-agent test on that worker returned "sync" (observed as the
  whole of `tests/test_subagent_write.py` failing when it shared a worker with
  that file). It now uses `monkeypatch.setattr`.
