"""The interactive REPL: `_repl()` is the main input/dispatch loop (banner,
`read_multiline`, /command dispatch via `_REPL_COMMANDS`, auto-save/
auto-compress after each turn), plus `_banner()` and CLI session-id
resolution helpers.

The slash-command handlers themselves live in sibling modules, split by
concern: `_repl_settings_cmds.py` (settings/context toggles), `_sessions_cmd.py`
(`/sessions`, `/resume`, `/save`, `/delete` + the `nx sessions`/`nx --tokens`
CLI entry points), and `_memory_cmd.py` (`/memory`). `_ReplState` and the
`_save_current`/metrics helpers they all share live in `_repl_state.py`.

Functions here that are monkeypatched by tests via `m.<name> = fake`
(`_banner`, `compress`, `_confirm`, `read_multiline`) are always called
through a lazy `import nx as _m; _m.<name>(...)` — including from other
functions defined in *this* file — since the patch only ever lands on the
top-level `nx` module's attribute, never on this submodule's own
globals. See _turn.py / _compress.py for the same pattern.
"""
from __future__ import annotations

import os
import sys
from typing import Callable

from ._ansi import DIM, BOLD, RESET, C, CLEAR_LINE, notice
from ._mentions import expand_at_mentions
from ._turn import _ensure_ctx_probe, _run_model_turn_loop
from ._compress import _maybe_autocompress
from ._input_history import _load_input_history, _record_input
from . import _state as st
from . import (
    _initial_messages, _list_sessions,
    _resolve_session, _new_session_id, _maybe_refresh_description,
)

from ._repl_state import _ReplState, _save_current, _fresh_metrics, _flush_save
from ._commands import bind_handlers, command, handler_map, help_rows
from . import _repl_settings_cmds as _rsc
from ._resume import resume_session
from ._sessions_cmd import (
    _cmd_sessions_handler, _cmd_resume_handler, _cmd_save, _cmd_delete_handler,
    _cmd_export, _cmd_fork,
)
from ._memory_cmd import (
    _cmd_memory_handler,
)
from ._todo import load_persisted_todos
from . import _panel
from ._wip import _cmd_wip
from ._memory_optimize import _cmd_memory_optimize  # noqa: F401 (re-export for tests)
from ._memory_reflect import _cmd_memory_reflect  # noqa: F401 (re-export for tests)
from ._repl_settings_cmds import (
    _cmd_quit, _cmd_help, _cmd_source, _cmd_theme, _cmd_reasoning,
    _cmd_tooloutput, _cmd_toolcalls, _cmd_provider, _cmd_resources, _cmd_bg, _cmd_yolo, _cmd_approval, _cmd_reset,
    _cmd_budget, _cmd_init, _cmd_recover, _cmd_undo, _cmd_compress, _cmd_autocompress, _cmd_todo,
    _cmd_ctx, _cmd_cost, _cmd_verify, _cmd_status, _cmd_retro, _cmd_review, _cmd_build,
)
from ._plan import _cmd_plan, plan_active  # noqa: F401 (re-export for tests)


# --- repl -------------------------------------------------------------------
def _banner() -> str:
    """Banner shown at startup and after source/approval changes."""
    sep = f"{DIM} · {RESET}"
    parts = [f"{BOLD}nx{RESET}", f"{C.ACCENT}{st.MODEL}{RESET}"]
    if len(st.SOURCES) > 1 and st.ACTIVE is not None:
        parts.append(f"{C.LABEL}{st.ACTIVE.name}{RESET}")
    if st.YOLO:
        parts.append(f"{C.SUCCESS}yolo{RESET}")
    elif st.APPROVE_LEVEL is None:
        parts.append(f"{C.WARNING}prompt:all{RESET}")
    elif st.APPROVE_LEVEL == "high":
        parts.append(f"{C.SUCCESS}auto:high{RESET}")
    elif st.APPROVE_LEVEL == "medium":
        parts.append(f"{C.WARNING}auto:medium{RESET}")
    else:
        parts.append(f"{DIM}auto:low{RESET}")
    if st.client is not None:
        parts.append(f"{DIM}{st.client.base_url}{RESET}")
    return sep.join(parts)


CommandHandler = Callable[[str, _ReplState], "str | None"]

@command("/tabs")
def _cmd_tabs(user: str, state: _ReplState) -> None:
    """Live dashboard of every running surface: main turn, background bash
    jobs, async sub-agents. Steering: lines typed during a turn are queued and
    injected at the next step boundary (nx/_surfaces.py).

    Phase D (0.137.0): on a tty the dashboard takes the ALTERNATE SCREEN
    (`_fullscreen`): j/k focus, Enter steers the focused agent row through the
    same queue, `?` for the key overlay, Esc back to scroll-chat. Off a tty or
    with NX_FULLSCREEN=0 the pre-Phase-D behaviour is unchanged — the old
    inline dashboard (still `run_tabs_dashboard`) is not reached at all: the
    one-shot print takes over, exactly as it did when no usable tty fd existed.
    """
    import nx as _m
    from . import _fullscreen
    from ._surfaces import tabs_view
    _fullscreen.open_view(tabs_view(), fallback=_m.tabs_print_fallback)


@command("/diff")
def _cmd_diff(user: str, state: _ReplState) -> None:
    """Full-screen review of the WORKING TREE diff vs HEAD: the changed file
    list, the rendered diff of one file (enter), and `a` for the reviewed
    commit flow (the existing `git_commit` tool: private staging index, secret
    scan, Y/n gate — the prompt runs after leaving the alternate screen).

    Read-only until you commit. The inline fallback — off a tty, with
    NX_FULLSCREEN=0, or with an explicit path — is a plain file list, or that
    one file's diff for `/diff <path>`.

    Distinct from /review, which runs the read-only critique sub-agent.
    """
    args = user[len("/diff"):].strip()
    from . import _fullscreen
    from ._render import commit_line
    from ._review import ReviewView, review_lines
    if args:
        # One path named: the review of exactly that file, inline — the
        # full-screen browser is the bare /diff.
        for line in review_lines(path=args.split()[0]):
            commit_line(line)
        return
    _fullscreen.open_view(ReviewView(state),
                          fallback=lambda: _inline_review())


def _inline_review() -> None:
    """`/diff`'s inline fallback (off tty / NX_FULLSCREEN=0): the working-tree
    file list exactly as `_review.review_lines` renders it, printed through the
    shared console path."""
    from ._render import commit_line
    from ._review import review_lines
    for line in review_lines():
        commit_line(line)


@command("/panel")
def _cmd_panel(user: str, state: _ReplState) -> None:
    """Toggle the right status rail flag (SPEC_PANEL.md, 0.127.1): ctx
    utilization, bg-job / agent counts, pending steering. The live region is
    a MID-TURN construct owned by whoever holds stdin (EVENTSURFACE_SPEC
    §2/§3), so nothing opens at the prompt — the rail appears on the next
    turn. NX_PANEL=1 defaults it on."""
    from . import _panel
    from ._surfaces import close_dashboard, dashboard_open
    if _panel.panel_wants_on():
        _panel.set_panel_enabled(False)
        if dashboard_open():  # erase a mid-turn region we may have opened
            close_dashboard()
        notice(f"{DIM}panel: off (NX_PANEL=1 to default on){RESET}")
    else:
        _panel.set_panel_enabled(True)
        notice(f"{DIM}panel: on — rail shows on the live region during "
               f"turns (NX_PANEL_WIDTH {_panel.panel_width()}){RESET}")


# Dispatch map derived from the single command registry (nx/_commands.py,
# 0.134.0): bind the handlers defined elsewhere through the `command`
# decorator, then build name+alias -> handler. Aliases (/exit, /compact,
# /resmon, ...) bind to the same handler as their primary command. The
# module-level name is kept (moved-names rule): tests patch/read it.
bind_handlers({
    "/quit": _cmd_quit,
    "/help": _cmd_help,
    "/source": _cmd_source,
    "/theme": _cmd_theme,
    "/reasoning": _cmd_reasoning,
    "/tooloutput": _cmd_tooloutput,
    "/toolcalls": _cmd_toolcalls,  # malformed-args breaker: show / reset [tool]
    "/provider": _cmd_provider,
    "/budget": _cmd_budget,
    "/resources": _cmd_resources,
    "/context": _cmd_ctx,
    "/cost": _cmd_cost,
    "/status": _cmd_status,
    "/bg": _cmd_bg,
    "/yolo": _cmd_yolo,
    "/approval": _cmd_approval,
    "/reset": _cmd_reset,
    "/init": _cmd_init,
    "/recover": _cmd_recover,
    "/undo": _cmd_undo,
    "/compress": _cmd_compress,
    "/autocompress": _cmd_autocompress,
    "/sessions": _cmd_sessions_handler,
    "/resume": _cmd_resume_handler,
    "/export": _cmd_export,
    "/fork": _cmd_fork,
    "/save": _cmd_save,
    "/memory": _cmd_memory_handler,
    "/delete": _cmd_delete_handler,
    "/todo": _cmd_todo,
    "/wip": _cmd_wip,
    "/plan": _cmd_plan,
    "/verify": _cmd_verify,
    "/retro": _cmd_retro,
    "/review": _cmd_review,
    "/build": _cmd_build,
})

_REPL_COMMANDS: dict[str, CommandHandler] = handler_map()

# /help renders _HELP_COMMANDS (defined in _repl_settings_cmds.py, which is
# outside this refactor's file scope) — rebind it to the registry-derived
# rows here so /help, the completion menu, the arg dropdowns and dispatch
# all derive from the one table (UIUX_SPEC A1). Reading the name from the
# defining module still works: the rows are the same registry data.
_rsc._HELP_COMMANDS = help_rows()


def _release_claims_on_exit() -> None:
    """Drop this process's cross-process file claims (0.119.0) on a clean exit.

    Claims are otherwise released only by another process's liveness/expiry
    check, so quitting would leave rows blocking this repo for up to
    NX_FILE_CLAIM_TTL. Both documented clean exits (`/quit` and EOF/Ctrl-D) go
    through here; the release itself never raises (see
    `_fileclaim.release_all_for_pid`), and the import is guarded so a broken
    install can't turn an exit into a traceback."""
    try:
        from ._fileclaim import release_all_for_pid
        release_all_for_pid()
    except Exception:
        pass


def _retro_on_exit_safe(session_id: str | None = None) -> None:
    """Session-end self-retro hook (nx #1): the choke point the clean-exit
    paths call after the final save/flush. `session_id` is the session's own
    id (the EOF/Ctrl-D path has it on hand); omitted, the `st.SESSION_ID`
    mirror every per-session store keys on is used.

    ON by default (`NX_RETRO_ON_EXIT=0` opts out) so the retro fires without
    the user remembering to ask. `_retro.retro_on_exit` owns the env gate,
    the once-per-session-id guard and the never-raise contract; the import is
    guarded here too so a broken install can't turn an exit into a traceback
    (same shape as `_release_claims_on_exit`)."""
    try:
        from ._retro import retro_on_exit
        retro_on_exit(session_id or st.SESSION_ID)
    except Exception:
        pass


def _repl() -> None:
    """The interactive loop: banner, ctx-probe warm-up, session state
    resolution/resume, then the read → dispatch → run-turn → autosave cycle
    until /quit or EOF. Slash commands are dispatched via `_REPL_COMMANDS`."""
    import nx as _m

    # The banner is printed at startup and after source/approval changes.
    print(_m._banner())
    print()
    # Warm the active source's max-context-window cache before the first
    # prompt: the probe (esp. on a still-loading local server, whose /v1/models
    # + /props 503 while mmapping) can take seconds-to-tens-of-seconds, but the
    # user typically spends that long typing their first message — so by the
    # time turn 1 finishes the cache is often already filled and the footer
    # shows the "/<max>" suffix immediately instead of a turn later. Background,
    # non-blocking; degrades to "no suffix" if it never resolves.
    if st.ACTIVE is not None:
        _ensure_ctx_probe(st.ACTIVE)
    messages = _initial_messages()
    # Past user submissions, newest last; Up/Down navigates. Loaded from the
    # global recall file (~/.nx/input_history.json) so Up reaches prior runs'
    # prompts too — not just this session's.
    history = _load_input_history()

    # --- session state ------------------------------------------------------
    # Every nx run has a session id. If the user passed --resume / -r
    # (with an id, prefix, index, or title) or --session <id>, we load that;
    # otherwise a fresh id is minted. The session is auto-saved after each
    # model turn (the messages array is the source of truth — we just write
    # it to ~/.nx/sessions/<id>.json atomically). A session that never
    # got any user input is never written to disk.
    session_id = _session_id_from_args()
    _resume_requested = any(a in ("--resume", "-r") for a in sys.argv[1:])
    # A resume/--session invocation that named nothing to load (bare
    # `--resume` with an empty store): start fresh instead of hunting for the
    # id we are about to mint. `--session <id>` always yields an id — it
    # legitimately names a session that does not exist yet (it creates one).
    _wanted_session = session_id
    if _wanted_session is None:
        if _resume_requested:
            print(f"{DIM}  no saved sessions to resume — starting fresh{RESET}")
        session_id = _new_session_id()
    assert session_id  # both branches above mint one; narrows for ty
    state = _ReplState(messages, session_id, False, history)
    st.SESSION_ID = session_id  # mirror: per-session stores key on it (undo, staleness)
    # Crash recovery, now that the id is known (0.128.3): the legacy
    # _load_persisted() at CLI startup ran before this mirror existed and
    # could only ever read the legacy undo.json — the per-session
    # undo/<sid>.json that flush_undo_stack() writes was never read back.
    from ._undo import _load_persisted as _undo_load
    _undo_load()

    # --- per-session worktree isolation (0.103.0, phase 3) -------------------
    # NX_SESSION_WORKTREE=1 / `nx --isolate`: create (or re-attach to) this
    # session's own worktree off HEAD and chdir into it, so PROJECT_ROOT, the
    # write fence, git_commit and run_bash all land in this session's private
    # tree. Two isolated sessions in one repo share nothing; merging the
    # session branch back is the user's explicit `git merge nx/session-<id>`.
    if (os.environ.get("NX_SESSION_WORKTREE", "0") in ("1", "true", "on")
            or "--isolate" in sys.argv[1:]):
        from ._worktree import create_session_worktree
        wt, err = create_session_worktree(session_id)
        if wt:
            os.chdir(wt)
            st.SESSION_WORKTREE = wt
            st.PROJECT_ROOT = wt
            print(f"{DIM}  ⟂ isolated: worktree {wt} "
                  f"(branch nx/session-{session_id}; merge it back with "
                  f"`git merge nx/session-{session_id}`){RESET}")
        elif err:
            print(f"{C.WARNING}  ⚠ couldn't isolate into a worktree: {err} "
                  f"— running in the shared tree{RESET}")
    if _wanted_session is not None and (_resume_requested
                                        or any(a == "--session" for a in sys.argv[1:])):
        # Both entry points share resume_session() (nx/_resume.py) — see
        # ROADMAP 1.4. Only --resume complains when the file isn't there: a
        # `--session <id>` miss is how session creation is supposed to work.
        resumed = resume_session(_wanted_session, state,
                                 announce_missing=_resume_requested)
    else:
        resumed = False
    if not resumed:
        # Fresh session (or a failed resume): build the cumulative-token
        # recorder for this id. The optional HTTP push (NX_METRICS_URL) is off
        # by default — see _Metrics.
        st._METRICS = _fresh_metrics(session_id)
        # Persistent todo file (0.94.0): a fresh session picks up the task
        # plan a previous session wrote through todo_write (see _todo.py).
        # The /todo command views it; the next todo_write overwrites both.
        _persisted = load_persisted_todos()
        if _persisted:
            st._TODOS = _persisted
    # Cross-session repo advisory lock (0.103.0): another REPL in this repo
    # gets a warning + explicit opt-in to share, instead of both sessions
    # silently interleaving edits. An isolated session (own worktree) can't
    # contend, so it claims the lock without prompting. Never fatal; degrades
    # to unlocked when .nx/ isn't writable. Skipped in headless one-shot runs.
    if os.environ.get("NX_SESSION_LOCK", "1") not in ("0", "false", "off"):
        from ._sessionlock import acquire as _lock_acquire
        _locked = st.SESSION_WORKTREE is not None
        try:
            _lock_acquire(ask_confirm=None if _locked
                          else (lambda m: _m._confirm(m)))
        except Exception:
            pass

    while True:
        try:
            # Esc-then-type: if the user was mid-line typing a redirect when
            # the interrupt's disarm handshake completed, resume the prompt
            # with those characters already there instead of blank. Same
            # mechanism for a wake prefill: text the user had typed when a
            # background job's auto-continuation turn took over (0.36.0).
            prefill = st._INTERRUPT_REDIRECT_PREFILL or st._WAKE_PREFILL
            st._INTERRUPT_REDIRECT_PREFILL = ""
            st._WAKE_PREFILL = ""
            try:
                user = _m.read_multiline(initial=prefill, history=state.history,
                                         wake=True)
            except TypeError:
                # Test stubs predating the wake kwarg: call them with the
                # original two-kwarg signature (real read_multiline always
                # accepts it, so in production this line never runs).
                user = _m.read_multiline(initial=prefill, history=state.history)
        except (EOFError, KeyboardInterrupt):
            print()
            # Drop the repo advisory lock so the next session starts clean.
            try:
                from ._sessionlock import release as _lock_release
                _lock_release()
            except Exception:
                pass
            # Release this process's file claims (0.119.0) alongside the repo
            # lock, for the same reason: a clean exit must not leave rows
            # blocking this repo until NX_FILE_CLAIM_TTL. Never raises.
            _release_claims_on_exit()
            # Flush the current session on Ctrl-D / Ctrl-C exit so the last
            # turn isn't lost (matches Hermes's exit-path flush). Cheap exit
            # form first (0.103.28): the final write skips the post-write
            # re-read and the GC sweeps.
            from ._sessions_fs import begin_exit as _begin_exit
            _begin_exit()
            if state.session_dirty:
                _save_current(state)
            # 0.103.4: with NX_ASYNC_SAVE the save above may only have queued;
            # drain it here so the documented "exit never loses the last turn"
            # holds even if a later atexit hook calls os._exit().
            _flush_save()
            # nx #1: after the final save/flush, run the silent session-end
            # retro. ON by default; a no-op without friction and never raises.
            _retro_on_exit_safe(state.session_id)
            # Clean up this session's worktree (0.113.2): an isolated session
            # that exits with a clean tree leaves nothing behind — remove the
            # tree AND its nx/session-<sid> branch. Dirty trees are kept
            # (remove_session_worktree refuses to destroy work).
            if st.SESSION_WORKTREE:
                try:
                    from ._worktree import remove_session_worktree
                    remove_session_worktree(st.SESSION_WORKTREE)
                except Exception:
                    pass
            # Show a grey resume hint so the user can pick up right where they
            # left off without having to run `nx sessions` first.
            if state.session_id:
                print(f"{DIM}  resume with: nx --resume {state.session_id}{RESET}")
            break
        # Auto-continuation turn (0.36.0): a background job finished while the
        # user idled at the prompt and the editor returned the WAKE sentinel.
        # Identity check BEFORE any .strip() — the sentinel is an object, not a
        # string. It becomes a synthetic user instruction and runs the turn:
        # no slash dispatch, no input-history entry. The completion note itself
        # is flushed into the transcript below, before model_turn(), so a
        # wake turn that never emits a tool call still sees it.
        wake = user is _m.WAKE_INPUT
        user = "" if wake else user.strip()
        if not wake:
            # A typed turn is the user taking the wheel — the unattended
            # chain restarts from 0 (0.121.0).
            st._WAKE_STREAK = 0
        if wake:
            st._WAKE_DUE = False
            # The flag is advisory: a poller that decided mid-turn (its
            # wake_due() sweep overlapped the turn) can set it long after the
            # turn's own first-tool flush delivered the note. Re-verify before
            # spending a whole model turn on it — a stale sentinel is dropped
            # silently instead of printing the banner and answering "Nothing
            # pending" (0.109.5).
            if not _m._autowake.wake_due():
                continue
            st._WAKE_STREAK = getattr(st, "_WAKE_STREAK", 0) + 1
            user = _m._autowake.wake_instruction()
            print(f"{DIM}  ▸ background job finished — auto-continuing{RESET}")
        if not user and not wake:
            continue
        cmd = user.split(None, 1)[0]
        handler = _REPL_COMMANDS.get(cmd) if cmd.startswith("/") else None
        if handler is not None:
            if handler(user, state) == "break":
                # `/quit`/`/exit` is the OTHER clean exit (the `_cmd_quit`
                # handler signals it with "break"): release this process's file
                # claims here too, or quitting that way still leaves rows
                # blocking this repo until NX_FILE_CLAIM_TTL.
                _release_claims_on_exit()
                break
            continue
        # Record for recall (skips slash commands and repeats of the very
        # last entry) and persist it so a future run's Up reaches this
        # prompt too — see _input_history.py.
        if not wake:
            _record_input(user, state.history)
        print()  # breathing room before the spinner/text starts
        state.messages.append({"role": "user", "content": user if wake
                               else expand_at_mentions(user)})  # str or parts (@image)
        try:
            st._TURN_ACTIVE = True
            # 0.127.1: the /panel rail's live region is a mid-turn construct
            # (EVENTSURFACE_SPEC §2/§3) — open it here, at turn start, when
            # enabled; the finally below always closes it again.
            try:
                _panel.ensure_region()
            except Exception:
                pass
            # Clear the wake flag at turn start, not only on the wake path:
            # a flag raised in the race window between the editor's last
            # _WAKE_DUE poll and this point describes work a user-typed turn
            # is about to drain itself (first-tool-call flush). Leaving it set
            # produced a phantom wake — a banner with no note — right after
            # that turn's own delivery. Anything genuinely undelivered is
            # re-derived by wake_due()'s fresh sweep after the turn.
            st._WAKE_DUE = False
            if wake:
                # Run the completion sweep before flushing: wake_due() ran it
                # too (0.36.3), and its note is only staged here — the flush
                # below just drains the slots. A text-only wake turn therefore
                # still delivers the finished job's exit line, and the next
                # wake_due() finds nothing (no infinite wake loop).
                _m._run_bg_completion_scan()
                # Flush the staged completion note(s) up front: the wake turn
                # may end as plain text (no tool call), and the notes otherwise
                # only flush on the first tool call of a turn.
                _m._flush_pending_background_notice(state.messages)
            _run_model_turn_loop(
                state.messages,
                on_step=lambda meta=None: _save_current(state, meta, update_index=False),
            )
        except KeyboardInterrupt:
            # ^C during generation can land between the spinner's last frame
            # and the `finally` that wipes its line (or skip the wait
            # entirely): the "thinking · esc to interrupt" bar would survive
            # in scrollback above the next prompt. Force-wipe it before the
            # chatbox paints, so the interrupt leaves a clean line.
            _ts = st._TURN_SPINNER
            st._TURN_SPINNER = None
            if _ts is not None:
                try:
                    _ts.stop()
                except Exception:
                    sys.stdout.write(CLEAR_LINE)
                    sys.stdout.flush()
            print(f"{DIM}  ^C interrupted{RESET}")
            state.session_dirty = True
            _save_current(state)
        except Exception as e:
            print(f"{DIM}  turn failed: {e}{RESET}")
        finally:
            st._TURN_ACTIVE = False
            # Race guard: a note staged by a worker thread AFTER this turn's
            # last tool-call flush (and a text-only ending delivers nothing)
            # used to depend on the poller's next 2s tick to be noticed — a
            # tick whose wake_due() guards (_TURN_ACTIVE, _async_running)
            # were evaluated against a stale snapshot mid-turn. Re-check at
            # turn end, when every guard is finally false: if a note is
            # pending now, raise the flag so the editor's next poll wakes
            # immediately instead of idling until the poller happens to
            # agree. A user-typed turn's own delivery already ran (the flush
            # happens at its start), so this can only fire on genuinely
            # undelivered work.
            if not wake and _m._autowake._any_pending_notice():
                st._WAKE_DUE = True
            # EVENTSURFACE_SPEC guardrail, termination: the /tabs live region
            # closes at turn end (its own driver — the watcher tick — parks
            # with the turn).
            try:
                from ._surfaces import close_dashboard, maybe_close_if_idle
                maybe_close_if_idle()
                close_dashboard()
            except Exception:
                pass
        # Auto-save after the turn settles (whether it ended cleanly, hit the
        # step cap, or was escaped). This is the Hermes pattern: persist every
        # turn so a crash / accidental close never loses work.
        state.session_dirty = True
        _save_current(state)
        # PF-05 (0.61.0): the undo stack is written amortized — once per turn
        # here, alongside the session autosave, instead of re-serializing every
        # checkpoint on every edit.
        from ._undo import flush_undo_stack
        flush_undo_stack()
        # Maybe auto-compress: if the context window is over the configured
        # fill threshold, silently fold the older turns so the chat can keep
        # going. Done after the save so the pre-compression state is on disk;
        # the compressed result is persisted by the _save_current() below.
        if _maybe_autocompress(state.messages, st._LAST_PROMPT_TOKENS):
            _save_current(state)
        # Maybe refresh the model-generated description (a cheap non-streaming
        # call every SESSION_DESC_REFRESH turns). PF-09 (0.56.0): the model call
        # now runs on a background thread, so this returns immediately and the
        # prompt is not blocked on it. The description lands as a second write.
        # Failures are silent — the title is always present.
        if state.session_id is not None:
            _maybe_refresh_description(state.session_id, state.messages)
        state.session_dirty = False

def _session_id_from_args() -> str | None:
    """Resolve a starting session id from CLI flags: --resume/-r and --session.

    `--resume` with no following target resumes the most recent session
    (the natural "pick up where I left off" intent). With a target it resolves
    by index/id/prefix/title. `--session <id>` forces an exact id (creates if
    absent — useful for scripts that want a stable id)."""
    args = sys.argv[1:]
    out = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--resume", "-r"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                # --resume <target>
                target = args[i + 1]
                sessions = _list_sessions(limit=50)
                out = _resolve_session(target, sessions) or target
                i += 2
            else:
                # bare --resume → most recent session, if any
                sessions = _list_sessions(limit=1)
                out = sessions[0]["id"] if sessions else None
                i += 1
            continue
        if a == "--session" and i + 1 < len(args):
            candidate = args[i + 1]
            safe = os.path.basename(candidate)
            if not safe or safe in (".", "..") or safe != candidate:
                print(f"error: invalid --session id {candidate!r}", file=sys.stderr)
                sys.exit(1)
            out = candidate
            i += 2
            continue
        i += 1
    return out
