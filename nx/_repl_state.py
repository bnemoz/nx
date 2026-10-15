"""`_ReplState` plus the session-persistence helpers (`_save_current`,
`_fresh_metrics`, `_reload_metrics_totals`) that every REPL command module
needs to touch. Split out on its own, below `_repl.py` and the `_cmd_*`
command modules, so none of them have to import each other to reach these.
"""
from __future__ import annotations

import os
import signal
from typing import Any

from ._metrics import _Metrics, _env_metrics_url
from ._sessions_io import _safe_title
from ._sources import _provider_order
from ._ansi import C, RESET
from . import _state as st
from . import _write_session, _load_session


class _ReplState:
    """Mutable REPL session state, threaded through command handlers instead
    of relying on closures/nonlocal — lets every slash command live as a
    plain module-level function dispatched from one table (see
    `_REPL_COMMANDS` in `_repl.py`), rather than some commands being nested
    if/elif branches and others being separately-extracted `_cmd_*` functions."""
    __slots__ = ("messages", "session_id", "session_dirty", "history")

    def __init__(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None,
        session_dirty: bool,
        history: list[str],
    ) -> None:
        self.messages = messages
        self.session_id = session_id
        self.session_dirty = session_dirty
        self.history = history


def _flush_save() -> None:
    """Block until any queued async save has been written (0.103.4).

    Cheap no-op when nothing is pending. Call this before anything that reads
    the session back or moves the session file — /fork (before the id
    switches), /resume, /delete, an export — and on any custom exit path, so
    the async save window never exposes a stale transcript. `_load_session`
    and `_list_sessions` already flush internally, which covers the read side
    of every REPL command; this is the explicit hook for the write side."""
    from . import _session_save
    _session_save.flush_pending()


def _save_current(
    state: _ReplState, meta: dict[str, Any] | None = None, update_index: bool = True,
) -> None:
    """Persist `state.messages` to the current session file.

    Title is auto-derived from the first user message (unless the user
    has set one explicitly with /save <title>). Source and cwd are
    recorded so a resume can re-select the right endpoint.

    `update_index=False` (see `_write_session`) skips refreshing the
    sidecar listing index — pass this for mid-turn autosaves, since the
    per-turn save that follows will refresh it anyway.

    0.103.4 (nx/_session_save.py): with NX_ASYNC_SAVE=1 (the default) this
    builds the meta + a private transcript snapshot and hands them to the
    session-save writer thread, so the per-turn `on_step()` in the turn loop
    no longer pays the store-form copy + SQLite read-merge-write + re-read
    inline. `NX_ASYNC_SAVE=0` restores the exact old synchronous path.
    """
    if state.session_id is None:
        return
    from . import _session_save
    if _session_save.async_save_enabled():
        _save_current_async(state, meta, update_index, _session_save)
        return
    # The save path is the critical path of the ^C handler (`except
    # KeyboardInterrupt: ... _save_current`) and of the per-turn finally, so a
    # SECOND Ctrl+C that lands mid-save (the session-body join over a large
    # session can take a while) used to raise out of the handler itself and
    # crash the REPL with a traceback from inside _write_session. Block SIGINT
    # for the duration: the ^C that triggered this save is already handled, and
    # one pressed during the save can wait for the next prompt.
    blockable = (hasattr(signal, "pthread_sigmask")
                 and hasattr(signal, "SIGINT"))
    sigmask = {signal.SIGINT} if blockable else set()
    if blockable:
        signal.pthread_sigmask(signal.SIG_BLOCK, sigmask)
    try:
        _save_current_locked(state, meta, update_index)
    except KeyboardInterrupt:
        # A ^C that arrived in the window before the mask took effect: the
        # save must still complete — swallow it here rather than dying mid-IO.
        pass
    finally:
        if blockable:
            try:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, sigmask)
            except KeyboardInterrupt:
                pass  # a pending ^C delivered on unmask — next prompt sees it


def _build_save_meta(state: "_ReplState", messages: list[dict[str, Any]],
                     meta: dict[str, Any] | None) -> dict[str, Any]:
    """The session-JSON metadata block a save carries (title/source/cwd/model/
    provider routing/metrics/todos/plan/wip). Split out of
    `_save_current_locked` so the async path can build it on the caller thread
    — everything it reads (st.ACTIVE, st.MODEL, the metrics totals) is live
    REPL state the writer thread must not go looking for later.

    `messages` is passed in (not read off `state`) because the async path
    derives it from its private snapshot: iterating the live transcript while
    another thread appends would risk a 'list changed size during iteration'."""
    m = dict(meta or {})
    if "title" not in m:
        first_user = next(
            (msg["content"] for msg in messages
             if msg.get("role") == "user" and isinstance(msg.get("content"), str)),
            "",
        )
        t = _safe_title(first_user)
        if t:
            m["title"] = t
    m.setdefault("source", st.ACTIVE.name if st.ACTIVE else None)
    m.setdefault("cwd", os.getcwd())
    m.setdefault("model", st.MODEL)
    # Record the active provider routing (if any) so a /resume re-applies
    # the same OpenRouter upstream choice. Stored as a bare `order` list
    # (not the whole extra_body) to keep the session JSON minimal and to
    # avoid persisting unrelated request-body keys a future feature might add.
    if st.ACTIVE:
        order = _provider_order(st.ACTIVE)
        if order:
            m["provider"] = order
    # Merge cumulative token totals into the session JSON so they persist
    # across restarts and are available to any metrics consumer that reads
    # the session files (e.g. a dashboard). This is always-on — the numbers
    # are already in hand from the stats footer, so storing them costs
    # nothing and a local session log is useful regardless of a push sink.
    if st._METRICS is not None:
        m.update(st._METRICS.as_meta())
    m["todos"] = st._TODOS
    m["plan"] = {"active": st._PLAN_ACTIVE, "goal": st._PLAN_GOAL,
                 "slug": st._PLAN_SLUG, "started": st._PLAN_STARTED}
    if st._WIP is not None:
        m["wip"] = st._WIP
    return m


def _save_current_locked(
    state: "_ReplState", meta: dict[str, Any] | None = None,
    update_index: bool = True,
) -> None:
    """Body of `_save_current`, run with SIGINT blocked."""
    m = _build_save_meta(state, state.messages, meta)
    if state.session_id is None:
        return
    try:
        _write_session(state.session_id, state.messages, m, update_index=update_index)
    except OSError as e:
        from ._subagent import _subagent_quiet
        if not _subagent_quiet():
            print(f"{C.ERROR}  ✗ couldn't save session: {type(e).__name__}: {e}{RESET}")
    # Fire-and-forget push to the optional metrics endpoint. Only does
    # anything if NX_METRICS_URL is set; self-disables on first
    # failure so a down/absent dashboard never slows the REPL.
    if st._METRICS is not None:
        st._METRICS.flush()


def _save_current_async(state: "_ReplState", meta: dict[str, Any] | None,
                        update_index: bool, _session_save) -> None:
    """The async half of `_save_current` (NX_ASYNC_SAVE=1, 0.103.4).

    Everything derived from live process state (the meta block, the metrics
    push) happens here on the caller thread; the writer thread only gets the
    session id, a private transcript snapshot and the finished meta dict, so
    it never touches REPL state that may still be mutating. Errors surface on
    the writer thread where a print would interleave with the live UI, so
    `_write_session_snapshot`'s own fail-soft handling is what applies — same
    contract as a failed background description refresh."""
    msgs = _session_save.snapshot_messages(state.messages)
    m = _build_save_meta(state, msgs, meta)
    # The meta block shares these by reference; the snapshot rule is that the
    # writer owns everything it is handed, so copy the mutable ones.
    if isinstance(m.get("todos"), list):
        m["todos"] = list(m["todos"])
    if isinstance(m.get("wip"), dict):
        m["wip"] = dict(m["wip"])
    _session_save.submit_save(state.session_id, msgs, m, update_index)
    if st._METRICS is not None:
        st._METRICS.flush()


def _fresh_metrics(session_id: str) -> _Metrics:
    return _Metrics(
        session_id, model=st.MODEL,
        source=st.ACTIVE.name if st.ACTIVE else None,
        push_url=_env_metrics_url())


def _reload_metrics_totals(session_id: str) -> None:
    saved = _load_session(session_id) or {}
    if isinstance(saved.get("todos"), list) and saved["todos"]:
        st._TODOS = saved["todos"]
        # Phase C (0.161.0): a resume writes st._TODOS outside todo_write (the
        # normal single writer), so refresh the stashed plan revision here too —
        # otherwise the stall trigger compares a resumed plan against the
        # never-written sentinel on its first wake evaluation.
        try:
            from ._todo import _plan_revision
            st._PLAN_REVISION = _plan_revision(saved["todos"])
        except Exception:
            pass
    # (0.150.x) The persistent todo file is no longer a resume fallback:
    # todos are session state — a session without recorded todos starts
    # with an empty list. The file (.nx/todo.md / NX_TODO_FILE) remains a
    # write-through export only.
    plan_meta = saved.get("plan")
    if isinstance(plan_meta, dict):
        st._PLAN_ACTIVE = bool(plan_meta.get("active"))
        st._PLAN_GOAL = plan_meta.get("goal")
        st._PLAN_SLUG = plan_meta.get("slug")
        st._PLAN_STARTED = plan_meta.get("started")
    # Work-state snapshot (0.94.0): restored so /resume can print where the
    # task stood when the session was last /wip'd.
    saved_wip = saved.get("wip")
    st._WIP = saved_wip if isinstance(saved_wip, dict) else None
    if st._METRICS is None:
        return
    if isinstance(saved.get("input_tokens"), int):
        st._METRICS.input_tokens = int(saved.get("input_tokens") or 0)
        st._METRICS.output_tokens = int(saved.get("output_tokens") or 0)
        st._METRICS.cache_read_tokens = int(saved.get("cache_read_tokens") or 0)
        st._METRICS.reasoning_tokens = int(saved.get("reasoning_tokens") or 0)
        st._METRICS.api_calls = int(saved.get("api_calls") or 0)
        if saved.get("started_at"):
            st._METRICS.started_at = float(saved["started_at"])
        # by_source rides along since 0.43.1 (2.5): the /cost scoreboard kept
        # its per-source attribution across a resume instead of printing
        # "(no per-source data)" forever after. Older session files have no
        # by_source — leave the fresh (empty) buckets in that case.
        saved_by_source = saved.get("by_source")
        if isinstance(saved_by_source, dict) and saved_by_source:
            st._METRICS.by_source = {
                str(name): dict(bucket) if isinstance(bucket, dict) else {}
                for name, bucket in saved_by_source.items()
            }
