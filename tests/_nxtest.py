"""Tier A/B isolation surface for the test suite (docs/test-architecture.md §4).

`nx_state_reset` is the cheap, per-test teardown that resets the four state
homes a module-dict rewind cannot see (sub-agent TLS, the async registry and
its live workers, the notices ContextVar, the recovery store) plus the rich
Live leak (`_drain_shared_console_live`: *stops* each leaked Live, then clears
the stack and restores a capture-swapped console file). It is autouse for every
test via the re-export in
conftest.py; conftest's expensive `nx.*` module-dict rewind is additionally
gated behind the `globals` marker, so Tier A files (listed nowhere below and
not marked) skip the walk entirely.

A file is Tier B ("globals") when it reloads nx modules, writes NX_* env at
module scope, or mutates nx globals in place — each such file is named in
`_GLOBALS_FILES` below, which auto-applies the marker (an explicit
`@pytest.mark.globals` on the module works too). Phase 2 migration is then
one line per file: fix the file, drop its name from the list.
"""
import io
import os
import sys

import pytest


def rebuilt_shared_console(env: dict[str, str]):
    """A fresh shared rich console built under `env` (a context-managed patch of
    the process env for the duration), replacing the module singleton — so tests
    force terminal/color/interactive state via rich's documented env hooks
    (TTY_INTERACTIVE, TTY_COMPATIBLE, COLORTERM, FORCE_COLOR) instead of
    patching console privates (_force_terminal / _color_system / is_interactive).

    Typical use::

        with _nxtest.rebuilt_shared_console({"TTY_INTERACTIVE": "1",
                                             "COLORTERM": "truecolor"}) as con:
            con.file = buf           # capture output
            ... open the region, assert on buf ...

    Restores the previous console singleton on exit (a Live leaked onto the
    rebuilt console is the existing nx_state_reset responsibility)."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        import nx._rich as rich
        old_console = rich._SHARED_CONSOLE
        old_env = {k: os.environ.get(k) for k in env}
        try:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            con = rich.themed_console()
            rich._SHARED_CONSOLE = con
            # the say-footer binds the singleton at import time; rebind it too
            footer = sys.modules.get("nx._turn_footer")
            # sentinel, not "old is not None": a pre-capture _say_console of
            # None is legitimate and must be restored as None (a leaked temp
            # console here was one source of the test_preview_visibility
            # rerun-list entries).
            _UNSET = object()
            old_say = _UNSET
            if footer is not None:
                old_say = footer._say_console
                footer._say_console = con
            yield con
        finally:
            rich._SHARED_CONSOLE = old_console
            if footer is not None and old_say is not _UNSET:
                footer._say_console = old_say
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _ctx()


def captured_console_output(env: dict[str, str]):
    """`rebuilt_shared_console` with output captured to a StringIO — the common
    shape (force a terminal, paint a Live, assert on the bytes).

        with _nxtest.captured_console_output(
                {"TTY_INTERACTIVE": "1", "COLORTERM": "truecolor"}) as (con, buf):
            ..."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with rebuilt_shared_console(env) as con:
            buf = io.StringIO()
            old_file = con.file
            con.file = buf
            try:
                yield con, buf
            finally:
                con.file = old_file

    return _ctx()


# Files that still depend on conftest's full module-dict rewind (Tier B).
# Everything NOT listed here is Tier A: import nx once, monkeypatch only.
# Seeded at Phase 1 (docs/test-architecture.md §6) with every file except the
# 8 verified-clean ones (test_conftest_restore_race, test_delegation_write_
# tier_096, test_env, test_env_docs_parity, test_retro, test_ripgrep_rewrite,
# test_write_fence, test_yolo_async_guidance): the ~2min module walk is
# amortized over 185 files that demonstrably still need it, and Phase 2
# migrates files off this list one at a time under the "5 clean parallel
# runs" guard. Paths are relative to tests/.
_GLOBALS_FILES = [
    "perf/test_static_budgets.py",
    "test_abbr.py",
    "test_approval_c3.py",
    "test_approval_policy.py",
    "test_approval_shifttab.py",
    "test_arg_repair.py",
    "test_async_compress.py",
    "test_async_log_writer.py",
    "test_async_save.py",
    "test_atomic_write.py",
    "test_autowake.py",
    "test_breaker_ergonomics.py",
    "test_breaker_integrity.py",
    "test_breadcrumbs.py",
    "test_build.py",
    "test_chatbox_completion_cache.py",
    "test_chatbox_file_completion.py",
    "test_chatbox_model_dropdown.py",
    "test_chatbox_resize.py",
    "test_chatbox_unified_dropdown.py",
    "test_clobber_guard.py",
    "test_color_themes.py",
    "test_commands_registry.py",
    "test_compact_alias.py",
    "test_config_cmd.py",
    "test_critique.py",
    "test_docs_parity.py",
    "test_duckdb.py",
    "test_edit_file.py",
    "test_empty_turn_recovery.py",
    "test_esc_approval.py",
    "test_eval_corpus.py",
    "test_eventsink.py",
    "test_file_claims.py",
    "test_forced_final_recovery.py",
    "test_fullscreen.py",
    "test_garble_guard.py",
    "test_git_commit.py",
    "test_git_index_guard.py",
    "test_git_path_repair.py",
    "test_gitcommit_headguard.py",
    "test_grammar.py",
    "test_http_retries.py",
    "test_image_consumers.py",
    "test_image_mentions.py",
    "test_input_history.py",
    "test_interrupt_headless_p12.py",
    "test_interrupt_watcher_singleton.py",
    "test_keys_registry.py",
    "test_length_continuation.py",
    "test_lessons.py",
    "test_local_risk_hint.py",
    "test_memindex.py",
    "test_memory_drift.py",
    "test_memory_reflect.py",
    "test_memory_tools.py",
    "test_mention_large_files.py",
    "test_midstream_args_recovery.py",
    "test_multi_edit.py",
    "test_nonmapping_args.py",
    "test_notice_ctx_isolation.py",
    "test_nx_home_isolation.py",
    "test_obsstore_p10.py",
    "test_onboarding.py",
    "test_orientation_note.py",
    "test_overflow_tiers.py",
    "test_panel.py",
    "test_parallel_dispatch.py",
    "test_parallel_toolblocks.py",
    "test_path_fail_breaker.py",
    "test_pdf_mentions.py",
    "test_perf_bounds.py",
    "test_perf_capability_manifest.py",
    "test_perf_interrupt_disarm.py",
    "test_perf_prefix_cache.py",
    "test_perf_render_path.py",
    "test_perf_tiered_registry.py",
    "test_perf_turn_tax.py",
    "test_plan_mode.py",
    "test_prefetch_gates.py",
    "test_preview_visibility.py",
    "test_private_sinks.py",
    "test_probe_backoff_abort.py",
    "test_project_tools.py",
    "test_quick_wins.py",
    "test_read_file_paging.py",
    "test_read_image.py",
    "test_reasoning_budget.py",
    "test_recovery_cache_safety.py",
    "test_recovery_context.py",
    "test_recovery_observability.py",
    "test_render_components.py",
    "test_reset_clear_alias.py",
    "test_result_redaction.py",
    "test_resume_parity.py",
    "test_risk_context.py",
    "test_run_tests.py",
    "test_search_files.py",
    "test_sed_edit.py",
    "test_session_eventlog.py",
    "test_session_images.py",
    "test_session_index.py",
    "test_session_merge_compaction.py",
    "test_session_retention.py",
    "test_session_textrefs.py",
    "test_session_worktree.py",
    "test_sessiondb.py",
    "test_sessionlock.py",
    "test_sessions.py",
    "test_sessions_search.py",
    "test_sol_pi_mechanisms.py",
    "test_sources.py",
    "test_spill_read_guard.py",
    "test_spinner.py",
    "test_sse_framing.py",
    "test_staleness.py",
    "test_stall_native_escalation.py",
    "test_store_contention_p11.py",
    "test_store_gc.py",
    "test_stream_parse_qwen_separators.py",
    "test_stream_tag_scan_watermark.py",
    "test_stream_timeout_retry.py",
    "test_subagent_advisor.py",
    "test_subagent_budget.py",
    "test_subagent_budget_108.py",
    "test_subagent_error_status.py",
    "test_subagent_exit_state.py",
    "test_subagent_model.py",
    "test_subagent_tiers_096.py",
    "test_subagent_turn_budget.py",
    "test_surfaces.py",
    "test_tabs_live.py",
    "test_telegram.py",
    "test_telegram_progress.py",
    "test_test_speed_tunables.py",
    "test_text_tool_protocol.py",
    "test_thinking.py",
    "test_todo_color.py",
    "test_todo_command.py",
    "test_tool_floor_slim_p13.py",
    "test_tool_hooks.py",
    "test_tool_output_toggle.py",
    "test_tool_overflow.py",
    "test_tps_footer.py",
    "test_turn_spinner_wipe.py",
    "test_undo_compress_v2.py",
    "test_unknown_tool.py",
    "test_usage_log.py",
    "test_verify.py",
    "test_wake_streak.py",
    "test_wake_turn_end_race.py",
    "test_webfetch.py",
    "test_worktree_097.py",
    "test_worktree_sweep.py",
]

# 0.155.4: seeded ONCE per worker (first nx_state_reset_body call): the pids of
# bg jobs that already existed (owned by this worker, finished, logs still
# present) before the worker's first test. The completion sweep
# (nx/_bg_completions.py) treats baseline pids as already-announced, so
# file-based residue from a shared/previous run's home can never be injected
# as a [Runtime note] into a later test's turn. A test that monkeypatches
# st._BG_NOTICE_BASELINE itself (raising=False) still wins — we only assign
# when the attr is absent or None.
_BG_NOTICE_BASELINE_SEEDED = False


def _seed_bg_notice_baseline(st) -> None:
    global _BG_NOTICE_BASELINE_SEEDED
    if _BG_NOTICE_BASELINE_SEEDED:
        return
    _BG_NOTICE_BASELINE_SEEDED = True
    cur = getattr(st, "_BG_NOTICE_BASELINE", None)
    if cur:
        return  # a test/fixture pinned a real baseline — keep it
    try:
        from nx._bash_jobs import list_bg_jobs
        mine = os.getpid()
        pids = {
            str(j.get("pid")) for j in list_bg_jobs(mine_only=True)
            if not j.get("alive") and j.get("owner_pid") == mine
            and j.get("log_path") and os.path.exists(j["log_path"])
        }
    except Exception:
        return  # exception-safe: no baseline better than a broken harness
    if pids:
        st._BG_NOTICE_BASELINE = pids


def nx_state_reset_body() -> None:
    """The FULL per-test teardown, callable from anywhere (the autouse
    fixture below AND conftest's `_restore()`) — one implementation, so the
    drain contract cannot drift between two copies again.

    Resolve modules through sys.modules at call time: a test that reloaded nx
    forked these objects, so a captured import would reset the STALE fork, not
    the live one. Cheap (~1ms): everything must already be in sys.modules —
    no imports here.
    """
    sa = sys.modules.get("nx._subagent")
    tls = sys.modules.get("nx._subagent_tls") or sa
    no = sys.modules.get("nx._notices")
    if tls is not None:
        tls._SUBAGENT_TLS.__dict__.clear()      # ctx + any residue
    if sa is not None:
        # quiet is assigned, never deleted, by production — clear the whole
        # TLS dict rather than popping one key.
        sa._ASYNC_QUIET_SLOTS.__dict__.clear()
        # Drain live workers BEFORE clearing their registry: _worker() re-reads
        # its _ASYNC_AGENTS record under the lock, so clearing with a worker
        # mid-flight loses the staged note and skips worktree removal.
        for t in list(sa._live_worker_threads()):
            t.join(timeout=2)
        with sa._ASYNC_LOCK:
            sa._ASYNC_AGENTS.clear()
        # Post-join hard-drain: on a loaded 4-core CI runner a leaked worker
        # can outlive the 2s join; parked ids in _ASYNC_QUEUED (status never
        # advanced to finished) keep `_async_running()` True for the rest of
        # the worker's life, making every later wake_due()/stall check on this
        # xdist worker deterministically False (CI 39733134252's test_wake_stall
        # signature). Drop the parked queue; a still-running thread re-reads
        # its record only under the lock and stages nothing once cleared.
        sa._ASYNC_QUEUED[:] = []
    # 0.163.0 (AUTOWAKE_SPEC Phase E) per-task OWNERSHIP residue: three hidden
    # homes a batch task can leave a live entry in if its finally never ran —
    # a published notice frame (a foreign thread could still stage a note into
    # a finished task's slots), the batch task TLS (task dict + owner ident),
    # and the per-thread bg-job scope. All three are released by production
    # finally paths, so this only catches a test that died mid-task; drained
    # here centrally rather than in the batch tests, per AGENTS.md.
    if no is not None:
        pub = getattr(no, "_PUBLISHED", None)
        if pub is not None:
            pub.clear()
    bt = sys.modules.get("nx._batch")
    if bt is not None:
        bt._TASK_TLS.__dict__.clear()
    bgm = sys.modules.get("nx._bg_completions")
    if bgm is not None:
        bgm._BG_SCOPE_TLS.__dict__.clear()
    rec = sys.modules.get("nx._recovery")
    if rec is not None:
        reset = getattr(rec, "_reset_store", None)
        if reset is not None:
            reset()  # recovery store (class-owned, out of dict reach)
    # run_tool's read-result cache is a hidden home keyed by absolute path:
    # a prior file's cached read (often a tmp_path file that no longer
    # exists) survives into the next test and breaks exact-shape asserts
    # (len(cache) == N) or serves stale content under arbitrary file
    # packing. Single writer is run_tool, so a clear() here is race-free.
    st = sys.modules.get("nx._state")
    if st is not None:
        cache = getattr(st, "_TOOL_RESULT_CACHE", None)
        if cache is not None:
            cache.clear()
        # Same shape as _TOOL_RESULT_CACHE but per-thread: _METRICS_TLS holds
        # the per-task metrics recorder; a test that armed it leaves the
        # thread's dict polluted for every later metrics assertion.
        mtls = getattr(st, "_METRICS_TLS", None)
        if mtls is not None:
            mtls.__dict__.clear()
        # The process-global _METRICS itself: a test that armed the CLI's
        # one-shot path (test_cli_version_flag's fake _fresh_metrics) leaves
        # st._METRICS holding whatever the fake returned — a plain object()
        # once blew up the next worker-mate's `_METRICS.model = ...` with an
        # AttributeError. Unlike _METRICS_TLS this scalar isn't swept by the
        # Tier B module-dict rewind (scalars aren't touched). Restore to the
        # factory default (None) — the same value a fresh import has.
        if getattr(st, "_METRICS", None) is not None:
            st._METRICS = None
        # 0.155.0: the per-AGENT approval posture override. Installed by
        # `_state._posture_scope` around a sub-agent body and deleted in its
        # `finally`, so a clean run leaves nothing — but a spawn that died
        # between install and restore (or a test driving the scope by hand)
        # would leave THIS thread resolving the agent's posture for every later
        # decision read on it (the leak the globals-swap had, one layer down).
        ptls = getattr(st, "_POSTURE_TLS", None)
        if ptls is not None:
            ptls.__dict__.clear()
    # 0.156.0: the per-TASK model routing target (`_subagent_model
    # .call_target_scope` / `_CALL_TARGET_TLS`). Same shape as _POSTURE_TLS —
    # installed by the scope and deleted in its finally, so a task that died
    # between install and restore (or a test driving the scope by hand) would
    # route EVERY later model call on this worker thread at the batch task's
    # endpoint. Resolved through sys.modules for the same reload reason.
    sm = sys.modules.get("nx._subagent_model")
    if sm is not None:
        ctls = getattr(sm, "_CALL_TARGET_TLS", None)
        if ctls is not None:
            ctls.__dict__.clear()
    # 0.156.0 (batch task overrides): the per-task headless threshold
    # (_approval.task_level_scope) and the per-task approval policy
    # (_approval_policy.policy_scope). Same leak shape as the slots above —
    # installed by the scope around a task body, deleted in its finally, so
    # only a crash between install and restore leaves residue that would
    # auto-approve (or policy-allow) every later quiet decision on this thread.
    ap = sys.modules.get("nx._approval")
    if ap is not None:
        ttls = getattr(ap, "_TASK_LEVEL_TLS", None)
        if ttls is not None:
            ttls.__dict__.clear()
    pol = sys.modules.get("nx._approval_policy")
    if pol is not None:
        ptls = getattr(pol, "_POLICY_TLS", None)
        if ptls is not None:
            ptls.__dict__.clear()
    # 0.155.1: the eventsink homes. _SINKS/_RETAINED_TAILS are keyed by job id
    # and read through the module attrs (sink_for/retained_tail), so a sink a
    # prior file registered (or a retained tail its worker's finally left)
    # survives into the next test on the worker — a later snapshot test with a
    # fixed job id then reads the residue's row/tail (the
    # test_event_taps test_snapshot_agent_row_prefers_the_answer_tail flake:
    # deterministic per worker, both rerun attempts died). _AGENT_QUEUES
    # likewise routes a steer line to a job id from an earlier file. Same
    # single-writer posture as the clears above: under the sink module lock.
    es = sys.modules.get("nx._eventsink")
    if es is not None:
        try:
            with es._LOCK:
                getattr(es, "_SINKS", {}).clear()
                getattr(es, "_RETAINED_TAILS", {}).clear()
                es._TAIL_LINES = None
        except Exception:
            pass
    surf = sys.modules.get("nx._surfaces")
    if surf is not None:
        try:
            with surf._STEER_LOCK:
                getattr(surf, "_AGENT_QUEUES", {}).clear()
        except Exception:
            pass
    # A wake poller left running by an async-agent test keeps sweeping after
    # its file ends and flips st._WAKE_DUE mid-LATER test (the exact
    # subagent_async -> bg_completion_seen poison: wake_due() asserts read
    # True). Same treatment test_autowake/test_wake_streak apply locally:
    # stop the daemon, then forget it.
    aw = sys.modules.get("nx._autowake")
    if aw is not None:
        poller = getattr(aw, "_POLLER", None)
        if poller is not None:
            try:
                poller.stop()
            except Exception:
                pass
        aw._POLLER = None
        # 0.160.0: the hold notice's idempotency key (the last
        # (reason, streak) pair it staged for) is a plain module global, so
        # the notice-slot sweep below cannot reach it. Left set, a later test
        # that trips the same guard at the same streak would find its hold
        # notice suppressed and assert False on a staged note.
        if hasattr(aw, "_HOLD_STAGED"):
            aw._HOLD_STAGED = None
    # Wake residue: the poller is stopped above, but the flags it (or a
    # mid-flight test) set on st survive it. Reset each to its import-time
    # initial value (nx/_state.py:189-196): _WAKE_DUE/_WAKE_ARMED start False,
    # _WAKE_STREAK starts 0 — the REPL's own user-turn path (nx/_repl.py:455)
    # treats exactly this combination as "clean". A leftover _WAKE_DUE=True
    # makes wake_due() True in an unrelated later test on the worker.
    if st is not None:
        st._WAKE_DUE = False
        st._WAKE_ARMED = False
        st._WAKE_STREAK = 0
        # 0.162.0 (AUTOWAKE_SPEC Phase D): the chain cost accumulators are
        # wake residue too — a leftover _WAKE_CHAIN_SPEND would trip the
        # budget guard in an unrelated later test, and a stale _WAKE_CHAIN_BASE
        # miscounts the first delta of the next chain.
        st._WAKE_CHAIN_SPEND = 0
        st._WAKE_CHAIN_BASE = 0
        # 0.161.0 (AUTOWAKE_SPEC Phase C): the plan-stall trigger's residue.
        # A leftover armed clock (_WAKE_STALL_AT) plus _campaign_pending()
        # state would make wake_due() True in a later test once the grace
        # elapsed, and a stale _WAKE_STALL_REVISION / _HOLD / _PENDING changes
        # which instruction the next wake picks.
        st._WAKE_STALL_AT = 0.0
        st._WAKE_STALL_REVISION = -1
        st._WAKE_STALL_HOLD = False
        st._WAKE_STALL_PENDING = False
        # Background-completion scan state: st._BG_NOTICE_SEEN is read as "pids
        # already announced" but None means "re-seed from the baseline", and
        # _BG_SCAN_DONE gates the sweep — neither is rewound by anything else.
        # A bare-fixture test (no nx_fresh) runs with whatever SESSION_HOME the
        # module holds (the developer's real ~/.nx when conftest scrubbed it),
        # so the sweep can announce a REAL just-finished bg job as a
        # [Runtime note] mid-test (seen live: test_forced_final_recovery's
        # messages[-1] was a heredoc job's completion note).
        # 0.155.4: seed _BG_NOTICE_BASELINE once per worker from the bg jobs
        # that already exist (finished, with logs) at first reset — the same
        # public view the completion sweep reads (list_bg_jobs(mine_only=True)).
        # Without it the sweep announced file-based residue from the worker's
        # shared home (owner_pid == the worker pid) as a user-role [Runtime
        # note] mid-turn, breaking forced_final_recovery's role-shape asserts.
        # Computed ONCE (not per reset): later finished jobs are newsworthy;
        # residue that predates the worker's first test is not.
        try:
            _seed_bg_notice_baseline(st)
        except Exception:
            pass
        st._BG_NOTICE_SEEN = None
        st._BG_SCAN_DONE = False
        # 0.166.0 (EXPLAINABLE_HARNESS_SPEC Phase 1): the last-prompt tap. A
        # digest snapshot left from a prior test makes the /ctx as-sent view
        # report provenance for messages the NEXT test never sent (and hides
        # the "no settled call recorded" path the view must also render).
        st._LAST_PROMPT = None
        # 0.166.0 (EXPLAINABLE_HARNESS_SPEC Phase 2): the last-compression
        # ledger. A stale ledger would make the /ctx before/after view describe
        # a fold a later test never ran (and hide its "no fold yet" path).
        st._LAST_COMPRESSION = None
    if no is not None:
        no._CURRENT.set(None)
        # Drain the module-level pending-note slots too. A worker that finished
        # after the join timeout above stages its [Runtime note] into the
        # module store (nx/_notices.set_ -> setattr(st, ...)) — invisible to
        # the ContextVar reset — and Tier A files (no module-dict rewind) would
        # otherwise inherit it: wake_due() sees a phantom notice (bg-completion
        # tests fail) and the notice flush appends it (yolo/background-gating
        # plumbing tests fail). The slots are derived from the store itself
        # (st._NOTICE_DEFAULTS / the _PENDING_* module attrs), not hand-listed,
        # so a slot added in nx/_state.py is swept by construction.
        defaults = getattr(st, "_NOTICE_DEFAULTS", {})
        for slot in _pending_slots(st):
            no.set_(slot, defaults.get(slot))
    # The leaked-Live drain — COMPLETE (0.154.5). A Live leaked by a dashboard
    # test keeps the shared console in live mode: every later `nx._rich.cprint`
    # sees a non-empty `Console._live_stack` and takes the rich branch
    # (nx/_rich.py:271), composing the line into the dead Live's frame instead
    # of calling builtins.print — invisible to the suite's captures (the
    # test_preview_visibility flake). Clearing the list was NOT enough: the Live
    # object never ran stop(), so rich's own rendering state (the cursor-hiding
    # render hook pushed on the console, `Live._started`, the alt-screen
    # bookkeeping) stayed live, and the next Live on the same console composed
    # as `_nested` against a frame nobody erases.
    # STOP each leaked Live first (Live.stop() pops the stack via clear_live(),
    # pops the render hook, restores the cursor and, for a transient Live,
    # erases the frame) and only then clear whatever a half-crashed stop left.
    rich = sys.modules.get("nx._rich")
    if rich is not None:
        try:
            _drain_shared_console_live(rich)
        except Exception:
            pass


def _drain_shared_console_live(rich) -> None:
    """STOP then clear every Live left on the shared console's stack, and
    restore the console's `file` before it is handed to the next test.

    `rich` is the live `nx._rich` module (resolved by the caller through
    sys.modules, so a forked/reloaded module is drained too). Everything here
    is defensive: this runs as teardown on mid-crash state and must never raise
    into the next test.
    """
    console = rich.shared_console()
    stack = getattr(console, "_live_stack", None)
    if stack:
        # stop() mutates the stack (clear_live -> pop), so walk a snapshot.
        # Reversed: rich's stack is a stack, and stopping an outer Live before
        # an inner one leaves the inner one's render state attached.
        for live in list(stack)[::-1]:
            try:
                live.stop()
            except Exception:
                pass
        stack = getattr(console, "_live_stack", None)
        if stack:
            # A Live whose stop() raised (or a fake that only records calls)
            # can still be pinned here. `_started` False is what a later
            # start() reads to decide `_nested`, and what rich's
            # `Console.print` path consults, so flip it before the clear.
            for live in list(stack):
                try:
                    live._started = False
                except Exception:
                    pass
            stack.clear()
    # A test that captured console output by swapping `con.file` (see
    # captured_console_output above) and then died before its finally ran
    # leaves the SHARED console writing into a dead StringIO: every later
    # cprint's rich branch disappears there, and `Console.file` is a lazy
    # property over `_file` (rich: `self._file or sys.stdout`). Only a capture
    # buffer is reset — a real file object is someone's deliberate
    # redirection and is left alone.
    try:
        if isinstance(getattr(console, "_file", None), io.StringIO):
            console._file = None
    except Exception:
        pass
    # NOT touched here: `nx._turn_footer._say_console`. It is the same
    # singleton by construction (nx/_turn_footer.py:33 `_say_console =
    # _shared_console()`), and re-binding it from teardown is actively wrong:
    # test_preview_visibility's `_no_live` fixture monkeypatches
    # `shared_console()` for the duration of a test, so a rebind here would
    # pin the throwaway console on the footer — the very leak 0.154.2 fixed
    # (see test_captured_console_output_restores_none_say_console).


def _pending_slots(st) -> list[str]:
    """Every pending-note slot name the notices store knows about: the routed
    slots (st._NOTICE_DEFAULTS keys — what nx/_notices.SLOTS is built from)
    plus any st module attr matching _PENDING_ (slots without a routed
    default). dir() here is the contract test's oracle too: a new slot in
    nx/_state.py shows up automatically."""
    names = set(getattr(st, "_NOTICE_DEFAULTS", {}) or {})
    names.update(a for a in dir(st) if a.startswith("_PENDING_"))
    return sorted(names)


@pytest.fixture(autouse=True)
def nx_state_reset():
    """Reset hidden per-test state after each test (~1ms common case)."""
    yield
    nx_state_reset_body()
