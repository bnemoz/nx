# Test Suite Redesign — nx 0.142.x

## 1. Where we are

- **~192 test files, ~2,180 tests.** Parallel `-n auto`: **~34s**, but **0–15 flaky failures per run** (measured: 14 consecutive runs, median 6). Serial/fallback: order-dependent, slower, and cannot run standalone anymore (`uv run pytest tests/ -q` without xdist dies at collection — several files execute module-scope code that assumes xdist's worker bootstrap).
- **The suite's only isolation mechanism is `tests/conftest.py`'s `_restore()`** (361 lines): before and after *every* test it scrubs `NX_*` env, rewinds every live `nx.*` module dict to a first-import snapshot, and re-derives mirrors. This was built for a smaller, mostly-single-threaded suite. It no longer holds because the runtime under test has grown state the rewind **cannot see**:

| Hidden state | Home | Visible to `_restore()`? |
|---|---|---|
| Sub-agent context | `_SUBAGENT_TLS` (threading.local) | ✗ |
| Quiet slots | `_ASYNC_QUIET_SLOTS` (threading.local) | ✗ |
| Notice frames | `_notices._CURRENT` (ContextVar) | ✗ |
| Recovery store | ON st's class (deliberately out of dict reach) | only via `_reset_store()` special case |
| Async job registry | `_ASYNC_AGENTS` + daemon worker threads | ✗ (records from *live* threads re-appear) |
| Log writers | `_AsyncLogWriter` daemon threads | ✗ |
| Reloaded module objects | `sys.modules` after 64 files call `importlib.reload(nx)` | partially — restores **stale** baseline values into the new dicts |

## 2. Diagnosis: the flaky failures are one family

Every flaky failure traced in this audit is a **residue leak invisible to `_restore()`**, and the failing test is the *innocent* one:

1. **`ERROR: nested spawn_agent refused`** (`test_subagent_async`, `test_subagent_error_status`) — a leaked `_SUBAGENT_TLS.ctx` from a prior test on the same worker.
2. **`_should_parallelize` returns False** (`test_subagent_write`) — same leak, read through `_subagent_active()`.
3. **run_tool answers as a sub-agent** (`test_tool_hooks`: "not available to the read-only sub-agent") — same leak, read through `_subagent_table()`.
4. **`forced == [True, ...]` too early** (`test_subagent_budget`) — `sub_loop = _subagent_active()` is True on the parent thread; the loop pre-forces final answers.
5. **`wake_due()` False at streak-1** (`test_wake_streak`) — notice writes land in the module dict but a live frame/`_reset_store` mismatch reads the default.
6. **`web_fetch` missing from `_active_tools()`** (`test_webfetch`) — a test file or a *prior* test reloaded `nx` with `NX_WEBFETCH=0` in the environment (the schema is appended to `TOOLS` at import only when `WEBFETCH_ENABLED`, `nx/__init__.py:530-531`), leaving the reloaded module's catalog without `web_fetch` while another test's env intent re-applies `NX_WEBFETCH=1`. It is a reload-under-wrong-env bug, not a stale-baseline one — `_restore()` actually restores lists/dicts *in place* (`_cur[:] = _tmpl`), so baseline and live module agree.
7. **Gate wording flips** (`test_unknown_tool`: "read-only sub-agent" vs "not available in this session") — knobs read module-attr-first, env-fallback; a reload under leaked env pins a stale attr.

**Why `--dist loadfile` already helps (1 failure / 3 runs, measured):** it keeps each file on one worker, so cross-file residue becomes deterministic per-file ordering rather than a shuffle. That's a strong hint the problem is inter-*file* residue, not intra-file races.

**Update (0.145.1): the +14s loadfile tax is gone — the tax was never the packing, it was stragglers.** With the slowpoke sleeps removed (webfetch's un-mocked DNS in the SSRF guard, the `run_tests` xdist probe running a sleeping stub, the memory-optimize 10s barrier, throttle sleeps), the suite runs 2395 tests in ~14s under `--dist loadfile` on a 24-core box — faster than the old arbitrary-packing number (34s), because the critical path was always the slowest single file, not worker count. `--dist load` now buys nothing measurable and re-exposes residue; loadfile stays.

**Why the failure rate varies 0–15:** it depends on which files an xdist worker happens to pack together.

## 3. Design principles for the redesign

1. **No global surgery as the isolation primitive.** `_restore()`'s module-dict rewind is a leaky abstraction over real state (threads, TLS, contextvars, on-class stores). Replace "rewind everything" with "**each test constructs the state it needs**" — the standard pytest shape (fixtures + monkeypatch), which is scoped and self-unwinding.
2. **State ownership:** any module global that tests mutate gets a *single canonical accessor* (module attr OR env, never both), and tests mutate it only through `monkeypatch.setattr` on that one home.
3. **No `importlib.reload(nx)` as a test tool.** Reload re-executes ~100 modules to re-read ~5 env vars. The few tests that genuinely need "re-resolve config from env" should call a small `reconfigure()` entry point, not re-run the whole package. (This also kills the forked-module-object class of bugs.)
4. **Threads are test subjects, not test infrastructure.** A test that needs a sub-agent *loop* should drive the loop body synchronously on the test thread (the loop body is already factored: `_run_sync_agent_body`); only a handful of tests should exercise the real daemon-thread spawn path, and they must drain/teardown their records in a fixture.
5. **Breadth preserved, time budget enforced:** keep all 2,182 tests; the goal is zero flake at ~30s parallel, ~4min serial.

## 4. Target architecture

### Tier A — hermetic unit core (~80% of tests, ~10s, `-n auto` default)
Tests that import `nx` once per process (session-scoped import — **never reloaded**) and exercise pure functions or `run_tool` bodies with **no** ambient sub-agent state. Rules:
- No `os.environ` writes: config arrives via function args or a `nx_config` fixture that monkeypatches the module attr (the one canonical home).
- No real threads: async machinery tested via the sync body + a `fake_loop`.
- No conftest `_restore()` needed for these — they leak nothing. `conftest.py` shrinks to the env baseline + the tempdir reaper.
- Directory: `tests/unit/` (existing files migrate gradually; a `pytest marker`, not a big-bang move, so history stays navigable).

### Tier B — "globals" tests (~15%, ~20s, `--dist loadfile`, pinned by marker)
The 64 reload-based files and any test that must mutate process globals (TOOLS catalog, `_TOOL_GATES`, `_ACTIVE_SPINNER`). Changes:
- Marker `@pytest.mark.globals` (auto-applied via `conftest` by filename list or a `tests/globals/` dir). `pytest -n auto` runs tiers A+B with `--dist loadfile` so file→worker assignment is stable; CI can also run `tests/unit` alone for a 10s pre-commit loop.
- These files keep a **per-file** env/attr baseline fixture (already the pattern) and additionally reset the four hidden-state homes in teardown: `_SUBAGENT_TLS.ctx = None`, `_ASYNC_QUIET_SLOTS.quiet = False`, `_notices` frame check, `_ASYNC_AGENTS.clear()` — extracted into one `nx_state_reset` autouse fixture so the contract is written once, not per-file.

### Tier C — concurrency/E2E (~5%, ~60s, `test_e2e/`, `-n 4` max)
Real daemon-thread spawns, `_AsyncLogWriter`, `run_tests`/git subprocess files, the wall-clock-sensitive perf tests (`test_perf_turn_tax` — stays but its sleep becomes tunable, see fix 4), REPL-driving tests (`m.main()` with stubbed streams — the 6 files that currently can break collection).

### Fixtures replacing the conftest rewind
```python
# tests/_nxtest.py — the new (small) conftest surface
@pytest.fixture(autouse=True)
def nx_state_reset():
    yield
    # the four homes _restore() could never see.
    # Resolve modules through sys.modules at teardown time: a test that
    # reloaded nx forked these objects, so a captured import would reset the
    # STALE fork, not the live one.
    import sys
    sa = sys.modules.get("nx._subagent")
    tls = sys.modules.get("nx._subagent_tls") or sa
    no = sys.modules.get("nx._notices")
    if tls is not None:
        tls._SUBAGENT_TLS.__dict__.clear()      # ctx + any residue
        tls._ASYNC_QUIET_SLOTS.__dict__.clear() # quiet flag + residue
    if sa is not None:
        # drain live workers first: _worker() re-reads its record after the
        # lock is acquired, so clearing under the lock alone loses the note.
        for t in list(sa._live_worker_threads()):
            t.join(timeout=2)
        with sa._ASYNC_LOCK:
            sa._ASYNC_AGENTS.clear()
        sa._reset_store()  # recovery store (class-owned)
    if no is not None:
        no._CURRENT.set(None)
    for _k in [k for k in os.environ if k.startswith("NX_")]:
        del os.environ[_k]
    os.environ.update(_BASELINE)
```
Two deliberate differences from a naive "clear the dicts":
- **Workers are joined, not just forgotten.** `_worker()` (`nx/_subagent.py:819-855`) re-reads its `_ASYNC_AGENTS` record after acquiring the lock; clearing under the lock alone makes the record vanish mid-flight, the staged note is lost (`_stage_subagent_answer` bails when `rec is None`), and the worktree removal is skipped. The worker-registry (`_live_worker_threads()`) is a small production addition — spawn already has the thread object at `nx/_subagent.py:857`.
- **`clear()` is used only on slots whose keys production always re-creates.** `_ASYNC_QUIET_SLOTS.quiet` is assigned, never deleted, by 7 production modules — a test that raises between set and restore would otherwise leak `quiet=True` into the next test, which is *weaker* than `_restore()`'s pop of post-baseline names. Clearing the whole TLS `__dict__` (not popping one key) fixes that; the same reasoning applies to `_SUBAGENT_TLS.ctx`.

Cost per test: ~1ms in the common case (no live workers, four dict clears + env replace) vs `_restore()`'s per-test walk of every live `nx.*` module dict (measured comment in-tree: this was ~2min across a serial run when it also touched interrupt Events).

### The `nx_state_reset` safety net stays in Tier A/B even after migration
It's the *cheap* guarantee; `_restore()` was the *expensive* one. Tests that obey the rules don't need it, but it's 1ms — keep it as insurance.

## 5. Concrete near-term fixes (can land independently, each kills a flake family)

0. **(DONE, 0.145.1) The known-flake ledger** — the recurring flakes and their fixes, so nobody re-debuggs them:
   - `_PENDING_DELEGATION_NOTICE` / `_PENDING_MULTI_EDIT_NOTICE` / `_PENDING_VERIFY_NOTICE` / `_PENDING_OCC_REMINDER` were missing from `nx_state_reset`'s drain list. A staged-but-never-flushed nudge survived teardown and made `_any_pending_notice()` (hence `wake_due()`) True in *any later file on the worker* — the classic "bg_completion_seen fails after subagent_async" poison. Now drained centrally (tests/_nxtest.py) and pinned in that file's fixture.
   - (0.154.2, phase 1) The drain contract was unified into ONE function, `nx_state_reset_body` (tests/_nxtest.py), called by both the autouse fixture and conftest's `_restore()` — the two copies had drifted (the `_PENDING_*` hand list, `_TOOL_RESULT_CACHE`, the `_autowake._POLLER` stop and the `_BG_NOTICE_SEEN`/`_BG_SCAN_DONE` resets existed only on one side). §4's "the contract is written once, not per-file" claim was violated exactly this way; the unified body closes it. New drains: `_WAKE_DUE`/`_WAKE_ARMED`/`_WAKE_STREAK` (the comment claiming they were "swept with the state block" was false — nothing touched them), `st._METRICS_TLS.__dict__` (drained by nobody; test_batch_run hand-cleared it), and the `_PENDING_*` slots are now derived from `st._NOTICE_DEFAULTS` + `dir(st)` instead of an 8-name hand list, so a slot added in nx/_state.py is swept by construction (pinned by tests/test_state_reset_contract.py).
   - (0.154.2) Pruned from `_FLAKY_RERUN_NODEIDS`, root causes fixed: the three `test_preview_visibility` entries (rebuilt_shared_console's old_say restore skipped a pre-capture `_say_console` of `None` — legitimate — leaking the temp console; now restored via a sentinel) and `test_batch_run::test_parallel_task_sessions_carry_own_totals` (`_METRICS_TLS` drain, above). `test_forced_final_recovery`'s three entries stay: real-run_bash/messages[-1] design, phase 1.5.
   - (0.154.4) The three `test_preview_visibility` entries are BACK: after the sentinel fix all three failed together once under `-n 8` at normal (17s) wall — the restore fix was necessary, not sufficient. The remaining root cause is the incomplete leaked-`Live` drain (the teardown clears `_live_stack` but not the `Live` object or `con.file`). Passes standalone and in both serial orders with the dashboard files; treat as open, fix the drain properly before pruning again.
   - (0.154.6) **Honest-done for non-timeout mid-stream cuts** (product bug, not a flake): a mid-stream cut that is not an `httpx.TimeoutException` — connection abort mid-chunked-body, or an SSE error frame then drop — exhausted `model_turn()`'s malformed-stream ladder and returned `TURN_DONE`, so headless callers (`nx run`, `nx eval`, sub-agents) scored a truncated, never-completed task as status "done" (exit 0). The same dishonesty applied to the special-token/repetition-collapse ladder and to genuine reasoning-only rescue exhaustion. All three now return the new honest `TURN_GAVE_UP` (`nx/_turn_status.py`), which the loop maps to summary status "error" (exit 4) exactly like the P3-2 `TURN_TIMED_OUT` path — one loop, one status vocabulary, no headless-only branch; the REPL just ends the turn and returns to the prompt. The two `@pytest.mark.xfail(strict=True)` pins in `tests/test_turn_loop_e2e.py` were un-pinned (they now assert the honest status directly), plus one new REPL-side vocabulary regression test in the same file.
   - `nx_state_reset` now also stops+forgets a leaked wake poller (`nx._autowake._POLLER`) and clears `st._TOOL_RESULT_CACHE` (a read cached in file A broke `len(cache) == N` asserts in file B under `--dist load`).
   - Fixed sleeps/ordering races: webfetch tests did real DNS (un-mocked `getaddrinfo` in the SSRF guard, 10s each under a dropping resolver — mock it per-test), `test_run_tests_timeout` ran the xdist probe against its own sleeping stub (31s; probe now skipped in-test AND bounded at 10s in product), memory-optimize 10s barrier (→1.5s), subagent_async 5s barrier (→30s) and its destructive drain-poll (now non-destructive, drains once), event_taps start-vs-finish tap ordering (wait for the finish line), perf_interrupt fixed `sleep(0.05)`+`<0.05` bound (event-synced start, `<0.095`), bg_completion 3s/4s poll windows (→30s fixture env).
   - `test_git_index_guard::test_gitcommit_staged_wip_*`: genuinely environmental (product-side git subprocess timeouts under 24-worker load). Covered by a one-shot in-process rerun in tests/conftest.py (`_FLAKY_RERUN_NODEIDS`), NOT by a marker — extend that tuple if a new one earns its place.
   - Failures that only appear when two full suites run concurrently on one box (23s wall runs) are machine contention, not regressions — check `ps` before bisecting.
1. **Reset TLS + async registry in conftest `_restore()`** (one block, ~10 lines). Kills families 1–4 immediately. This is the highest-value single diff in this document.
2. **`--dist loadfile` as the documented default** for `-n auto` (measured: 15 failures → ≤1, +14s). Adopt now, revisit after fix 1. *(0.145.1: revisited — the +14s was straggler sleeps, not packing; with them gone loadfile runs ~14s, see §2 update.)*
3. **Fix standalone collection** (currently `uv run pytest tests/ -q` without xdist INTERNALERRORs at import — module-scope `m.main()` calls in 6 files race `sys.argv`). Move those calls into fixtures; until then document `-n auto` as required.
4. **Kill the two real-sleep outliers**: `test_perf_turn_tax`'s 30s sleep is a worker-thread busy-keep (the fake endpoint must not return while the assertion measures `flush()`'s enqueue latency), not a wedged call — shrink it behind an `NX_TEST_WAKE_HOLD` tunable (siblings at `:113-124` already use this pattern) instead of skipping the test in non-CI; `ram_stress*.py` → move out of `tests/` into `scripts/`.
5. **`reconfigure()` entry point** in `nx/__init__.py` re-reading the ~10 env-derived module attrs (the subset tests actually need: `NX_SUBAGENT_*`, `NX_VISION`, `NX_WEBFETCH`, `NX_SUBAGENTS`, timing budgets). Then delete `importlib.reload(nx)` from the 64 files file-by-file; each deletion removes one forked-module risk.
6. **Env/attr single-home**: for `NX_SUBAGENT_WRITE`, `NX_SUBAGENT_ASYNC`, `NX_VISION`, `NX_SUBAGENTS`, make the *module attr* the only runtime truth (env read once at import), and fix tests that set only the env var.

## 6. Migration plan (no big-bang)

| Phase | Work | Guard |
|---|---|---|
| 0 | fixes 1+2+3+4 (one PR, ~1 day). Fix 1 must resolve modules via `sys.modules` at reset time (reload forks the objects) and join live async workers before clearing their registry — see the fixture sketch above for the exact semantics. | 10 clean `-n auto` runs |
| 1 | introduce `nx_state_reset` fixture + `globals` marker; move `_restore()` behind the marker | suite time unchanged |
| 2 | migrate the 10 highest-churn files to Tier A style (drop reload, monkeypatch the attr home) | per-file: 5 clean parallel runs |
| 3 | `reconfigure()` + mass un-reload (mechanical, sed-able per file) | budget test + tier tests |
| 4 | retire `_restore()` entirely; conftest ≈ 80 lines | ledger note in `tests/perf/baselines.json` |

Each phase leaves the suite green and runnable; phase 4 is only reached when zero files depend on the rewind.

## 7. What we deliberately do NOT do

- **No pytest-asyncio / async rewrite.** The codebase is thread-based; converting test concurrency to asyncio would test a different program.
- **No Textual/Pilot harness.** The TUI is rich-based with custom REPL loop, not a Textual App; Pilot would buy nothing the existing `run_tool`/capture_stdout pattern doesn't.
- **No test-randomization plugin by default.** `--dist loadfile` gives deterministic packing; randomization would re-expose residue during migration, then be worth a CI-nightly experiment after phase 4.
- **No coverage target change.** The suite's breadth is good; this redesign is about isolation and speed, not new tests.

## 8. Success criteria

- `uv run pytest tests -n auto` green on **20 consecutive runs** (~14s each, 0.145.1; was ~35s).
- `uv run pytest tests` (serial) collects and passes (~4min) — currently impossible.
- conftest.py under 100 lines; zero `importlib.reload(nx)` in tests.
- The 0.142.5-era flake classes (TLS leak, reload fork, notice routing) each have a regression test that fails deterministically against the old conftest.
