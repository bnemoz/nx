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
    "test_094_features.py",
    "test_abbr.py",
    "test_allowed.py",
    "test_api_error_contract.py",
    "test_approval_c3.py",
    "test_approval_env.py",
    "test_approval_policy.py",
    "test_approval_shifttab.py",
    "test_arg_repair.py",
    "test_async_compress.py",
    "test_async_log_writer.py",
    "test_async_save.py",
    "test_atomic_write.py",
    "test_autocompress.py",
    "test_autowake.py",
    "test_bash_heredoc.py",
    "test_bash_watchdog.py",
    "test_batch_run.py",
    "test_bg_completion_seen.py",
    "test_bg_pid_map.py",
    "test_breaker_ergonomics.py",
    "test_breaker_integrity.py",
    "test_build.py",
    "test_cache_stale.py",
    "test_cache_vcs_stale.py",
    "test_cache_visibility.py",
    "test_chatbox_completion_cache.py",
    "test_chatbox_file_completion.py",
    "test_chatbox_model_dropdown.py",
    "test_chatbox_resize.py",
    "test_chatbox_unified_dropdown.py",
    "test_cli_version_flag.py",
    "test_clobber_guard.py",
    "test_color_themes.py",
    "test_commands_registry.py",
    "test_compact_alias.py",
    "test_config_cmd.py",
    "test_context_overflow_recovery.py",
    "test_context_window.py",
    "test_cost.py",
    "test_cost_correctness.py",
    "test_critique.py",
    "test_ctx_report.py",
    "test_delegation_nudge.py",
    "test_delegation_steering.py",
    "test_docs_parity.py",
    "test_duckdb.py",
    "test_edit_anchors.py",
    "test_edit_file.py",
    "test_empty_turn_recovery.py",
    "test_esc_approval.py",
    "test_esc_ladder.py",
    "test_eval_corpus.py",
    "test_event_taps.py",
    "test_eventsink.py",
    "test_fallback_ladder_amplification.py",
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
    "test_memory_optimize.py",
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
    "test_subagent.py",
    "test_subagent_advisor.py",
    "test_subagent_async.py",
    "test_subagent_budget.py",
    "test_subagent_budget_108.py",
    "test_subagent_error_status.py",
    "test_subagent_exit_state.py",
    "test_subagent_model.py",
    "test_subagent_tiers_096.py",
    "test_subagent_turn_budget.py",
    "test_subagent_write.py",
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
    "test_tool_schema_invariants.py",
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
    if no is not None:
        no._CURRENT.set(None)
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
