"""Tier A/B isolation surface for the test suite (docs/test-architecture.md §4).

`nx_state_reset` is the cheap, per-test teardown that resets the four state
homes a module-dict rewind cannot see (sub-agent TLS, the async registry and
its live workers, the notices ContextVar, the recovery store) plus the rich
Live stack leak. It is autouse for every test via the re-export in
conftest.py; conftest's expensive `nx.*` module-dict rewind is additionally
gated behind the `globals` marker, so Tier A files (listed nowhere below and
not marked) skip the walk entirely.

A file is Tier B ("globals") when it reloads nx modules, writes NX_* env at
module scope, or mutates nx globals in place — each such file is named in
`_GLOBALS_FILES` below, which auto-applies the marker (an explicit
`@pytest.mark.globals` on the module works too). Phase 2 migration is then
one line per file: fix the file, drop its name from the list.
"""
import sys

import pytest


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

@pytest.fixture(autouse=True)
def nx_state_reset():
    """Reset hidden per-test state after each test (~1ms common case).

    Resolve modules through sys.modules at teardown time: a test that
    reloaded nx forked these objects, so a captured import would reset the
    STALE fork, not the live one.
    """
    yield
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
    # A wake poller left running by an async-agent test keeps sweeping after
    # its file ends and flips st._WAKE_DUE mid-LATER test (the exact
    # subagent_async → bg_completion_seen poison: wake_due() asserts read
    # True). Same treatment test_autowake/test_wake_streak apply locally:
    # stop the daemon, then forget it. _WAKE_DUE itself is swept with the
    # state block below.
    aw = sys.modules.get("nx._autowake")
    if aw is not None:
        poller = getattr(aw, "_POLLER", None)
        if poller is not None:
            try:
                poller.stop()
            except Exception:
                pass
        aw._POLLER = None
    # Background-completion scan state: st._BG_NOTICE_SEEN is read as "pids
    # already announced" but None means "re-seed from the baseline", and
    # _BG_SCAN_DONE gates the sweep — neither is rewound by anything else.
    # A bare-fixture test (no nx_fresh) runs with whatever SESSION_HOME the
    # module holds (the developer's real ~/.nx when conftest scrubbed it),
    # so the sweep can announce a REAL just-finished bg job as a
    # [Runtime note] mid-test (seen live: test_forced_final_recovery's
    # messages[-1] was a heredoc job's completion note).
    if st is not None:
        st._BG_NOTICE_SEEN = None
        st._BG_SCAN_DONE = False
    if no is not None:
        no._CURRENT.set(None)
        # Drain the module-level pending-note slots too. A worker that finished
        # after the join timeout above stages its [Runtime note] into the
        # module store (nx/_notices.set_ → setattr(st, ...)) — invisible to the
        # ContextVar reset — and Tier A files (no module-dict rewind) would
        # otherwise inherit it: wake_due() sees a phantom notice (bg-completion
        # tests fail) and the notice flush appends it (yolo/background-gating
        # plumbing tests fail).
        for slot in ("_PENDING_SUBAGENT_NOTICE", "_PENDING_YOLO_ASYNC_NOTICE",
                     "_PENDING_BG_COMPLETION_NOTICE", "_PENDING_BACKGROUND_NOTICE",
                     "_PENDING_MULTI_EDIT_NOTICE", "_PENDING_DELEGATION_NOTICE",
                     "_PENDING_VERIFY_NOTICE", "_PENDING_OCC_REMINDER"):
            no.set_(slot, None)
    # A Live leaked by a dashboard test stays on the shared console's stack,
    # which reroutes every later cprint through rich — invisible to the
    # suite's builtins.print captures (the test_preview_visibility flake).
    rich = sys.modules.get("nx._rich")
    if rich is not None:
        try:
            stack = getattr(rich.shared_console(), "_live_stack", None)
            if stack:
                stack.clear()
        except Exception:
            pass
