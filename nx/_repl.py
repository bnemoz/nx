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

from ._ansi import DIM, BOLD, RESET, C
from ._mentions import expand_at_mentions
from ._turn import _ensure_ctx_probe, _run_model_turn_loop
from ._compress import _maybe_autocompress
from ._input_history import _load_input_history, _record_input
from . import _state as st
from . import (
    _initial_messages, _list_sessions,
    _resolve_session, _new_session_id, _maybe_refresh_description,
)

from ._repl_state import _ReplState, _save_current, _fresh_metrics
from ._resume import resume_session
from ._sessions_cmd import (
    _fmt_when, _print_transcript, _session_list_page_options,
    _print_session_list, _session_next_hint, _display_path, _cli_sessions, _cli_tokens,
    _cli_prune_sessions,
    _cmd_sessions, _cmd_resume, _cmd_delete,
    _cmd_sessions_handler, _cmd_resume_handler, _cmd_save, _cmd_delete_handler,
    _cmd_export,
)
from ._memory_cmd import (
    _cmd_memory, _cmd_memory_handler, _slugify, _cmd_memory_save,
    _cmd_memory_remember, _cmd_memory_delete, _cmd_memory_list,
    _render_transcript_for_memory, memory_write, memory_search, memory_delete,
    _load_memory_note, _write_memory_file, _note_kind, MEMORY_KINDS,
)
from ._memory_optimize import _cmd_memory_optimize  # noqa: F401 (re-export for tests)
from ._memory_reflect import _cmd_memory_reflect  # noqa: F401 (re-export for tests)
from ._repl_settings_cmds import (
    _HELP_COMMANDS, _cmd_quit, _cmd_help, _cmd_source, _cmd_theme, _cmd_reasoning,
    _cmd_tooloutput, _cmd_toolcalls, _cmd_provider, _cmd_resources, _cmd_bg, _cmd_yolo, _cmd_approval, _cmd_reset,
    _cmd_budget, _cmd_init, _cmd_recover, _cmd_undo, _cmd_compress, _cmd_autocompress, _cmd_todo,
    _cmd_ctx, _cmd_cost, _cmd_verify, _cmd_status,
)
from ._config_cmd import render_status
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

_REPL_COMMANDS: dict[str, CommandHandler] = {
    "/quit": _cmd_quit,
    "/exit": _cmd_quit,
    "/help": _cmd_help,
    "/source": _cmd_source,
    "/theme": _cmd_theme,
    "/reasoning": _cmd_reasoning,
    "/tooloutput": _cmd_tooloutput,
    "/tools": _cmd_tooloutput,  # legacy alias — toggles *output* visibility, not tools
    "/toolcalls": _cmd_toolcalls,  # malformed-args breaker: show / reset [tool]
    "/provider": _cmd_provider,
    "/budget": _cmd_budget,
    "/resources": _cmd_resources,
    "/resmon": _cmd_resources,
    "/ctx": _cmd_ctx,
    "/cost": _cmd_cost,
    "/status": _cmd_status,
    "/bg": _cmd_bg,
    "/yolo": _cmd_yolo,
    "/approval": _cmd_approval,
    "/reset": _cmd_reset,
    "/clear": _cmd_reset,
    "/new": _cmd_reset,
    "/init": _cmd_init,
    "/recover": _cmd_recover,
    "/undo": _cmd_undo,
    "/compress": _cmd_compress,
    "/compact": _cmd_compress,
    "/autocompress": _cmd_autocompress,
    "/sessions": _cmd_sessions_handler,
    "/resume": _cmd_resume_handler,
    "/export": _cmd_export,
    "/save": _cmd_save,
    "/memory": _cmd_memory_handler,
    "/delete": _cmd_delete_handler,
    "/todo": _cmd_todo,
    "/plan": _cmd_plan,
    "/verify": _cmd_verify,
}


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
    state = _ReplState(messages, session_id, False, history)
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
            # Flush the current session on Ctrl-D / Ctrl-C exit so the last
            # turn isn't lost (matches Hermes's exit-path flush).
            if state.session_dirty:
                _save_current(state)
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
        if wake:
            st._WAKE_DUE = False
            user = _m._autowake.WAKE_INSTRUCTION
            print(f"{DIM}  ▸ background job finished — auto-continuing{RESET}")
        if not user and not wake:
            continue
        cmd = user.split(None, 1)[0]
        handler = _REPL_COMMANDS.get(cmd) if cmd.startswith("/") else None
        if handler is not None:
            if handler(user, state) == "break":
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
            print(f"{DIM}  ^C interrupted{RESET}")
            state.session_dirty = True
            _save_current(state)
        except Exception as e:
            print(f"{DIM}  turn failed: {e}{RESET}")
        finally:
            st._TURN_ACTIVE = False
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
