"""Session persistence: the filesystem side (directory layout, the sidecar
listing index, atomic read/write/delete, and the model-generated one-line
description refresh).

The pure-logic helpers (id generation, title sanitizing, message pruning,
listing-summary shaping) live in _sessions_io.py instead, since they don't
touch the filesystem or st.SESSION_HOME — see that module's docstring.

Reads st.SESSION_HOME (not a plain module global) so a /reload in tests that
reassigns __init__.py's SESSION_HOME from a fresh NX_HOME is picked up here
too — see the comment where __init__.py mirrors it onto st.SESSION_HOME.

SESSION_DESC_REFRESH / DESC_SYSTEM are config resolved in __init__.py (small,
env-derived, not touched by tests), so _maybe_refresh_description() looks
them up lazily via `import nx as _m` rather than binding them at import time
— same pattern as _repl.py's monkeypatched-name note.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
import time
from typing import Any

from . import _state as st
from ._logging import _log_event
from ._metrics import _record_usage
from ._openai_client import APIConnectionError
from ._sessions_io import (
    _flatten_message_text, _safe_title, _prune_empty_assistant_messages,
    _short_id, _summarize_session_data, _session_matches_query,
)

SESSION_LIST_DEFAULT_LIMIT = 10
SESSION_LIST_MAX_LIMIT = 100


def _sessions_dir() -> str:
    """Where session files live. Honors NX_SESSIONS_DIR, then NX_HOME/sessions."""
    return os.path.expanduser(
        os.environ.get("NX_SESSIONS_DIR",
                       os.path.join(st.SESSION_HOME, "sessions"))
    )


def _memories_dir() -> str:
    """Where memory files live. Always under SESSION_HOME/memories."""
    return os.path.join(st.SESSION_HOME, "memories")


def _session_path(session_id: str) -> str:
    return os.path.join(_sessions_dir(), f"{session_id}.json")


def _index_path() -> str:
    return os.path.join(_sessions_dir(), ".index.json")


@contextmanager
def _exclusive_file_lock(path: str):
    """Hold an advisory interprocess lock associated with ``path``.

    Session saves are a read-modify-write operation, not just an atomic
    replacement. The separate lock file keeps the lock valid while the JSON
    file itself is replaced; the OS releases it if the process exits.
    """
    with open(path + ".lock", "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load_index() -> dict[str, Any]:
    """The sidecar listing index: {session_id: summary dict}. Missing or
    corrupt just returns {}, so callers degrade to a full per-file parse —
    same as before this index existed."""
    try:
        with open(_index_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, IOError, json.JSONDecodeError):
        return {}


def _write_index(idx: dict[str, Any]) -> None:
    path = _index_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass  # best-effort — a stale/missing index just falls back to full parses


def _update_index(session_id: str, data: dict[str, Any]) -> None:
    """Refresh this session's entry in the sidecar listing index so
    `nx sessions [query]` can list/search without json.load-ing every
    full transcript on disk (some can be large — tool outputs, diffs, …).
    Called from _write_session on every save, so the index stays at least
    as fresh as the file it mirrors."""
    # Different session writers also share this read-modify-write sidecar.
    # Protect it separately, since their per-session locks are distinct.
    with _exclusive_file_lock(_index_path()):
        idx = _load_index()
        idx[session_id] = _summarize_session_data(session_id, data)
        _write_index(idx)


def _merge_session_messages(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep transcript entries from both writers when ``incoming`` is stale.

    Each REPL owns an in-memory snapshot. Once another process has appended
    to that session, replacing the file with the stale snapshot would lose
    those turns. Both snapshots normally share a prefix, so retain the
    committed transcript and append the other writer's divergent suffix.
    """
    if existing == incoming:
        return incoming
    if len(existing) <= len(incoming) and existing == incoming[:len(existing)]:
        return incoming
    if len(incoming) <= len(existing) and incoming == existing[:len(incoming)]:
        return existing

    prefix_len = 0
    for saved, proposed in zip(existing, incoming):
        if saved != proposed:
            break
        prefix_len += 1
    return existing + incoming[prefix_len:]


def _write_session(session_id: str, messages: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
    """Persist `messages` to the session file. Creates the dir if needed.

    Writes atomically (temp file + rename) so a crash mid-write can't
    corrupt the existing session. `meta` (title, source, cwd, …) is
    merged into the stored metadata.
    """
    d = _sessions_dir()
    os.makedirs(d, exist_ok=True)
    _prune_empty_assistant_messages(messages)
    path = _session_path(session_id)
    # The lock spans reading the current revision through replacing it. A
    # rename alone prevents corruption, but does not prevent lost updates.
    with _exclusive_file_lock(path):
        now = time.time()
        existing = {}
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, IOError, json.JSONDecodeError):
            pass
        existing_messages = existing.get("messages")
        if not isinstance(existing_messages, list):
            existing_messages = []
        # Re-read under the lock and compare with the caller's snapshot. On
        # a stale save, preserve the committed suffix instead of silently
        # letting the last writer replace it.
        merged_messages = _merge_session_messages(existing_messages, messages)
        data = dict(existing)
        data["id"] = session_id
        data["messages"] = merged_messages
        data["created_at"] = existing.get("created_at", now)
        data["updated_at"] = now
        data["revision"] = int(existing.get("revision", 0) or 0) + 1
        if meta:
            for k, v in meta.items():
                if v is not None:
                    data[k] = v
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
        try:
            os.utime(path, (data["updated_at"], data["updated_at"]))
        except (OSError, TypeError, ValueError):
            pass
        _update_index(session_id, data)

def _load_session(session_id: str) -> dict[str, Any] | None:
    """Read a session file. Returns the dict (id, messages, meta…) or None."""
    path = _session_path(session_id)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "messages" in data:
            _prune_empty_assistant_messages(data.get("messages"))
            return data
    except (OSError, IOError, json.JSONDecodeError):
        pass
    return None


def _session_files_newest() -> list[str]:
    """Return session JSON filenames newest-first using filesystem mtimes.

    Listing many sessions should not require parsing every old transcript.
    The session writer keeps mtime aligned with updated_at, so this is a
    cheap index for recent-session browsing.
    """
    d = _sessions_dir()
    files = []
    try:
        with os.scandir(d) as entries:
            for entry in entries:
                # ".json" but excludes the sidecar index (a dotfile, .index.json)
                if not entry.name.endswith(".json") or entry.name.startswith("."):
                    continue
                try:
                    if not entry.is_file():
                        continue
                    files.append((entry.stat().st_mtime, entry.name))
                except OSError:
                    continue
    except OSError:
        return []
    files.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [name for _, name in files]


def _session_summary_from_file(fname: str, idx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the listing summary for one session file. When `idx` (the
    sidecar index — see _load_index) already has an entry for this session,
    return that directly instead of json.load-ing the full transcript;
    `idx=None` or a cache miss falls back to a full parse exactly as before
    the index existed (e.g. a session written by an older nx, or a
    missing/corrupt index file)."""
    sid = fname[:-5] if fname.endswith(".json") else fname
    if idx is not None:
        cached = idx.get(sid)
        if cached is not None:
            return cached
    d = _sessions_dir()
    path = os.path.join(d, fname)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "messages" not in data:
            return None
    except (OSError, IOError, json.JSONDecodeError):
        return None
    return _summarize_session_data(data.get("id") or sid, data)


def _list_sessions(limit: int | None = 20, offset: int = 0, query: str | None = None) -> list[dict[str, Any]]:
    """Return sessions newest-first as dicts: id, title, description, preview, n.

    `preview` is the first ~60 chars of the first user message; `description`
    is an optional model-generated one-liner refreshed every N turns (richer
    than the static first-message title). `n` is the turn count (non-system
    messages)."""
    import nx as _m
    limit = max(0, int(limit)) if limit is not None else None
    offset = max(0, int(offset or 0))
    files = _session_files_newest()
    idx = _load_index()
    out = []
    for fname in files:
        # Looked up via nx (not the module-local binding) since a couple of
        # tests monkeypatch m._session_summary_from_file to spy on call count.
        summary = _m._session_summary_from_file(fname, idx)
        if not summary or not _session_matches_query(summary, query):
            continue
        out.append(summary)
        if query:
            continue
        if limit is not None and len(out) >= offset + limit:
            break
    if query:
        out.sort(key=lambda s: s["updated_at"], reverse=True)
    end = None if limit is None else offset + limit
    return out[offset:end]


def _delete_session(session_id: str) -> bool:
    """Remove a session file. Returns True if something was deleted."""
    path = _session_path(session_id)
    try:
        os.remove(path)
    except OSError:
        return False
    idx = _load_index()
    if idx.pop(session_id, None) is not None:
        _write_index(idx)
    return True


def _maybe_refresh_description(session_id: str, messages: list[dict[str, Any]]) -> str | None:
    """Ask the model for a one-line session description, if enough turns have
    passed since the last refresh.

    Refreshes at the configured interval (every SESSION_DESC_REFRESH user
    turns). The description is stored in the session file's `description`
    field and surfaced in `nx sessions` / `/sessions` listings. A failure
    (server down, empty response) leaves the existing description untouched.

    Returns the new description string, or None if no refresh happened.
    """
    import nx as _m
    if _m.SESSION_DESC_REFRESH <= 0:
        return None
    if st.client is None or st.MODEL is None:
        return None
    # Count user turns (exclude synthetic runtime-note-only turns).
    user_turns = sum(
        1 for m in messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and not m["content"].lstrip().startswith("[")
    )
    existing = _load_session(session_id) or {}
    last_desc_turns = existing.get("desc_turns", 0)
    # Refresh when we've crossed a multiple of SESSION_DESC_REFRESH since the
    # last recorded refresh. Don't refresh before the first threshold (so a
    # 1-turn "hello" session doesn't burn a model call).
    if user_turns < _m.SESSION_DESC_REFRESH:
        return None
    if user_turns - last_desc_turns < _m.SESSION_DESC_REFRESH:
        return None
    # Build a compact transcript for the summarizer — truncate tool outputs
    # and skip the system prompt to keep the call cheap.
    def _trim(msgs: list[dict[str, Any]], per_msg: int = 500, cap: int = 20) -> str:
        out: list[str] = []
        for m in msgs:
            if m.get("role") == "system":
                continue
            c = _flatten_message_text(m, per_msg=per_msg)
            out.append(f"[{m.get('role', '?')}] {c}")
            if len(out) >= cap:
                break
        return "\n".join(out)
    prompt = _trim(messages[-30:])  # last ~30 messages is plenty of context
    payload = [
        {"role": "system", "content": _m.DESC_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        _log_event("req", {"model": st.MODEL, "messages": payload,
                           "stream": False, "_purpose": "session_desc"})
        resp = st.client.chat.completions.create(
            model=st.MODEL, messages=payload, stream=False, timeout=20,
            **(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {}))
        try:
            _log_event("resp", {"_purpose": "session_desc", "data": resp.model_dump()})
        except Exception:
            pass
    except APIConnectionError:
        return None  # server down — leave existing description as-is
    except Exception:
        return None
    _record_usage(
        getattr(resp, "usage", None),
        fallback_chars=len(resp.choices[0].message.content or ""),
        fallback_prompt_chars=sum(len(str(m.get("content") or "")) for m in payload),
    )
    desc = (resp.choices[0].message.content or "").strip().splitlines()
    desc = desc[0].strip() if desc else ""
    if not desc:
        return None
    desc = _safe_title(desc, maxlen=70) or desc[:70]
    # Persist the description + the turn count it was generated at so we know
    # when to refresh next. Merge into existing meta (don't clobber messages).
    meta = {"description": desc, "desc_turns": user_turns}
    if st._METRICS is not None:
        meta.update(st._METRICS.as_meta())
    _write_session(session_id, messages, meta)
    if st._METRICS is not None:
        st._METRICS.flush()
    return desc


def _resolve_session(target: str | None, sessions: list[dict[str, Any]] | None = None) -> str | None:
    """Resolve a user-typed target to a session id.

    Accepts: a full id, a numeric index into the recent-sessions list,
    a unique id prefix, or an exact title. Returns the id or None.
    """
    if not target:
        return None
    target = target.strip()
    sessions = sessions if sessions is not None else _list_sessions(limit=50)
    ids = [s["id"] for s in sessions]
    # numeric index → recent-sessions slot
    if target.isdigit():
        idx = int(target)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]["id"]
    # exact id
    if target in ids:
        return target
    # unique prefix
    prefixed = [i for i in ids if i.startswith(target)]
    if len(prefixed) == 1:
        return prefixed[0]
    # short id (the 6-hex suffix shown in listings) — match the tail segment
    # since the date+time prefix is shared/redundant across sessions created
    # in the same minute.
    suffixed = [i for i in ids if i.endswith("-" + target) or _short_id(i) == target]
    if len(suffixed) == 1:
        return suffixed[0]
    # exact title
    titled = [s["id"] for s in sessions if s["title"] == target]
    if len(titled) == 1:
        return titled[0]
    return None
