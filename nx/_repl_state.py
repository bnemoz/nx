"""`_ReplState` plus the session-persistence helpers (`_save_current`,
`_fresh_metrics`, `_reload_metrics_totals`) that every REPL command module
needs to touch. Split out on its own, below `_repl.py` and the `_cmd_*`
command modules, so none of them have to import each other to reach these.
"""
from __future__ import annotations

import os
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
    """
    if state.session_id is None:
        return
    m = dict(meta or {})
    if "title" not in m:
        first_user = next(
            (msg["content"] for msg in state.messages
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


def _fresh_metrics(session_id: str) -> _Metrics:
    return _Metrics(
        session_id, model=st.MODEL,
        source=st.ACTIVE.name if st.ACTIVE else None,
        push_url=_env_metrics_url())


def _reload_metrics_totals(session_id: str) -> None:
    saved = _load_session(session_id) or {}
    if isinstance(saved.get("todos"), list):
        st._TODOS = saved["todos"]
    plan_meta = saved.get("plan")
    if isinstance(plan_meta, dict):
        st._PLAN_ACTIVE = bool(plan_meta.get("active"))
        st._PLAN_GOAL = plan_meta.get("goal")
        st._PLAN_SLUG = plan_meta.get("slug")
        st._PLAN_STARTED = plan_meta.get("started")
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
