"""Session persistence: the filesystem side (directory layout, the
consolidated SQLite store index, atomic read/write/delete, and the
model-generated one-line description refresh).

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
import hashlib
import json
import os
import re
import threading
import time
from typing import Any

from . import _platform
from . import _state as st
from ._logging import _log_event
from ._metrics import _record_usage
from ._images import (
    IMAGE_REF_TYPE, _data_url_to_ref, _ref_to_data_url_part,
    _collect_message_image_refs, _gc_image_sidecars, _images_dir,
)
from ._env import _env_int
from ._openai_client import APIConnectionError
from ._private import open_private, private_dir
from ._sessions_io import (
    _flatten_message_text, _safe_title, _prune_empty_assistant_messages,
    _short_id, _summarize_session_data, _session_matches_query,
    _drop_leading_dangling_tool_turns,
)

SESSION_LIST_DEFAULT_LIMIT = 10
SESSION_LIST_MAX_LIMIT = 100


def _sessions_dir() -> str:
    """Where session files live. Honors NX_SESSIONS_DIR, then NX_HOME/sessions
    (read live from the environment, before the st.SESSION_HOME mirror, so a
    test module that sets NX_HOME at import time — before its module-scope
    m.main() calls — never touches the user's real store), then
    SESSION_HOME/sessions."""
    direct = os.environ.get("NX_SESSIONS_DIR")
    if direct:
        return os.path.expanduser(direct)
    home = os.environ.get("NX_HOME")
    if home:
        return os.path.join(os.path.expanduser(home), "sessions")
    return os.path.join(st.SESSION_HOME, "sessions")


def _memories_dir() -> str:
    """Where global memory files live. Always under SESSION_HOME/memories."""
    return os.path.join(st.SESSION_HOME, "memories")


def _project_memories_dir(project_root: str | os.PathLike[str] | None) -> str | None:
    """Where project-scoped memory files live: <project_root>/.nx/memories/.
    Returns None when there is no project root (bare cwd / outside a repo)
    so callers can degrade to global-only instead of inventing a path."""
    if not project_root:
        return None
    return os.path.join(str(project_root), ".nx", "memories")


def _session_path(session_id: str) -> str:
    """The legacy mirror-file path for a session id.

    0.101.0 stores transcripts in sessions.sqlite; per-session JSON files
    are no longer written. The path helper stays because it is the
    migration source scanner's naming rule and a handful of tests/tools
    still reason in terms of the old file layout."""
    safe = os.path.basename(session_id)
    if not safe or safe in (".", "..") or safe != session_id:
        raise ValueError(f"invalid session id: {session_id!r}")
    return os.path.join(_sessions_dir(), f"{safe}.json")


def _index_dir() -> str:
    """Pre-2.3b sidecar index dir: one small JSON file per session. 2.3b
    folded this index into the consolidated SQLite store
    (<sessions_dir>/.store.sqlite, nx/_storeindex.py); the dir remains only
    as a migration source — migrate_legacy() imports any shards it finds on
    the first save/list after an upgrade, then removes the dir."""
    return os.path.join(_sessions_dir(), ".index")


def _index_entry_path(session_id: str) -> str:
    safe = os.path.basename(session_id)
    return os.path.join(_index_dir(), f"{safe}.json")


def _open_private(path: str, flags: int = os.O_CREAT | os.O_WRONLY | os.O_TRUNC):
    """Open ``path`` for writing so only the owner can read it (0600),
    regardless of the process umask. Sessions/logs hold tool output and
    command results that may include secrets — world-readable files leak
    them to any other local user on a shared host.

    Thin alias (kept for its many call sites) onto the shared private-sink
    helper in nx/_private.py, which every other sink uses too (0.40)."""
    return open_private(path, flags)


@contextmanager
def _exclusive_file_lock(path: str):
    """Hold an advisory interprocess lock associated with ``path``.

    Session saves are a read-modify-write operation, not just an atomic
    replacement. The separate lock file keeps the lock valid while the JSON
    file itself is replaced; the OS releases it if the process exits.

    The locking primitive itself lives in nx/_platform
    (exclusive_file_lock): flock on POSIX, msvcrt byte-range locking on
    Windows. It takes the OPEN FILE OBJECT rather than the raw fd — on
    Windows msvcrt.locking() locks a range of the file object's handle and
    that object must stay open for the lock's lifetime (the helper normalizes
    the position to byte 0 itself; this lock file is never written, so the
    single-byte range covers it). The call is blocking on both platforms, so
    callers see exactly the old semantics: the with block runs with the lock
    held, a contended lock waits instead of raising, and lock.close()
    releases it.
    """
    lock = open_private(path + ".lock", os.O_CREAT | os.O_RDWR | os.O_APPEND,
                        text_mode="a+")
    try:
        with _platform.exclusive_file_lock(lock):
            yield
    finally:
        lock.close()


def _load_index() -> dict[str, Any]:
    """The listing index: {session_id: summary dict}, read column-only from
    the store's indexed listing columns (0.129: preview/n_msg ride on the
    sessions row, so an index build never touches an event log). An
    unavailable store degrades to an empty dict."""
    from . import _sessiondb
    try:
        out: dict[str, Any] = {}
        for s in _sessiondb.list_summaries():
            s.pop("heal", None)
            out[s["id"]] = s
        return out
    except Exception:
        return {}


def _write_index(idx: dict[str, Any]) -> None:
    """Rebuild the whole listing table from a full {id: summary} dict.

    0.101.0: the listing IS the transcript table, so a "rebuild" here can
    only drop rows whose id the caller omitted — used by tests and
    reconstruction paths. Summaries are not stored, so this cannot
    resurrect content a summary no longer describes."""
    from . import _sessiondb
    try:
        keep = set(idx)
        for sid in _sessiondb.all_ids():
            if sid not in keep:
                _sessiondb.delete_session_row(sid)
    except Exception:
        pass  # best-effort


def _update_index(session_id: str, data: dict[str, Any]) -> None:
    """Legacy seam: the per-save index refresh. 0.101.0 the transcript row
    is written by _write_session itself (one INSERT OR REPLACE also serves
    the listing), so there is nothing separate to refresh. Kept as a
    no-op so the exported name and its call sites stay stable."""
    return None


_SUMMARY_HEADER_RE = re.compile(
    r"^\[Compressed context — \d+ earlier turns summarized")

# Extra trailing turns a compaction rewrite may be missing because the
# snapshot it carries predates a couple of just-saved turns.
_COMPACTION_TAIL_SLACK = 4


def _is_compaction_rewrite(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]],
) -> bool:
    """True when ``incoming`` is the same conversation after compress() folded
    older turns into a summary turn: shorter, same system-prompt shape, and a
    compress() header at the top of the body. In that shape ``incoming`` is
    authoritative — the dropped turns were deliberately summarized away, not
    written by another writer — so merging must not resurrect them (the
    prefix/fork logic below assumes both writers only ever append, and would
    splice the pre-compression history back in front of the summary).

    Detected structurally so no caller flag is trusted: incoming must be
    strictly shorter, carry the same system-prompt shape, start its body with
    the compress() header, and every turn after that summary must be an exact
    tail of ``existing`` — exactly the shape ``messages[:] = sys + [summary]
    + tail`` in _compress.compress() produces.
    """
    if not existing or not incoming or len(incoming) >= len(existing):
        return False

    def _is_sys(m: dict[str, Any]) -> bool:
        return m.get("role") == "system"

    off_e = 1 if _is_sys(existing[0]) else 0
    off_i = 1 if _is_sys(incoming[0]) else 0
    if bool(off_e) != bool(off_i) or len(incoming) <= off_i:
        return False
    first_body = incoming[off_i].get("content")
    if not (isinstance(first_body, str) and _SUMMARY_HEADER_RE.match(first_body)):
        return False
    # every incoming turn after the summary must be a verbatim tail of existing,
    # allowing a few trailing turns that the snapshot predates (autosaves land
    # between the compress() dispatch and this save; demanding an exact suffix
    # made every post-compression save look like a fork and append the whole
    # compressed snapshot again, growing the store unboundedly).
    tail = incoming[off_i + 1:]
    if not tail or len(tail) > len(existing) - off_e:
        return False
    for slack in range(0, _COMPACTION_TAIL_SLACK + 1):
        if len(tail) + slack > len(existing) - off_e:
            break
        if existing[len(existing) - len(tail) - slack:len(existing) - slack] == tail:
            return True
    return False


def _merge_session_messages(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep transcript entries from both writers when ``incoming`` is stale.

    Each REPL owns an in-memory snapshot. Once another process has appended
    to that session, replacing the file with the stale snapshot would lose
    those turns. Both snapshots normally share a prefix, so retain the
    committed transcript and append the other writer's divergent suffix.

    The one non-append rewrite is a context fold (compress() or the
    overrun path): incoming is then the *shortened* authoritative transcript,
    and the merge must not treat it as a stale fork that resurrects the
    summarized-away history — see _is_compaction_rewrite().
    """
    if existing == incoming:
        return incoming
    if _is_compaction_rewrite(existing, incoming):
        return incoming  # compress() folded turns away on purpose; incoming wins
    if len(existing) <= len(incoming) and existing == incoming[:len(existing)]:
        return incoming
    if len(incoming) <= len(existing) and incoming == existing[:len(incoming)]:
        return existing

    # True fork (not a prefix relationship): the two suffixes diverge from a
    # shared point onward. Splicing them back-to-back can land an assistant
    # tool_calls turn from `existing` right before an unrelated turn from
    # `incoming`'s branch — the matching tool result lived only on the
    # discarded branch, so the join point is invalid. Drop a dangling
    # trailing assistant(tool_calls) turn off `existing` before joining, and
    # any dangling leading tool/tool_calls turn off the incoming suffix.
    head = existing
    if head and head[-1].get("role") == "assistant" and head[-1].get("tool_calls"):
        ids = {tc["id"] for tc in head[-1]["tool_calls"]}
        seen = {m.get("tool_call_id") for m in head if m.get("role") == "tool"}
        if ids - seen:
            head = head[:-1]

    prefix_len = 0
    for saved, proposed in zip(existing, incoming):
        if saved != proposed:
            break
        prefix_len += 1
    tail, _ = _drop_leading_dangling_tool_turns(incoming[prefix_len:])
    return head + tail


def _write_session(
    session_id: str, messages: list[dict[str, Any]], meta: dict[str, Any] | None = None,
    update_index: bool = True,
) -> None:
    """Synchronous write: drain any pending async save first, then store.

    0.103.4 (nx/_session_save.py): saves may now be queued on the session-save
    writer thread. An inline `_write_session` must never land *ahead* of a
    snapshot that is still in that queue (it would let a stale transcript be
    merged in afterwards and bump the revision counter out of order), so this
    entry point flushes first. `flush_pending()` is a no-op both when nothing
    is queued and when called from the writer thread itself, so the async path
    (which calls `_write_session_snapshot` directly) never deadlocks here.
    """
    from . import _session_save
    _session_save.flush_pending()
    _write_session_snapshot(session_id, messages, meta, update_index=update_index)


def _write_session_snapshot(
    session_id: str, messages: list[dict[str, Any]], meta: dict[str, Any] | None = None,
    update_index: bool = True,
) -> None:
    """The store write itself, with no flush gate.

    This is what the session-save writer thread runs, so `messages` must
    already be a private snapshot (nx/_session_save.snapshot_messages): this
    function prunes `messages` in place and must never reach into the REPL's
    live transcript. Persist `messages` to the session store. Creates the dir if needed.

    0.101.0: the store is sessions.sqlite (nx/_sessiondb.py) -- one row per
    session, the transcript JSON as the `data` blob. The old per-session
    JSON file + flock is gone: store_session() does the whole
    read-merge-write inside one BEGIN IMMEDIATE transaction, and the
    `revision` counter in the blob keeps its role as the concurrency
    marker tests pin.

    `meta` (title, source, cwd, etc.) is merged into the stored metadata.
    `update_index=False` skips the FTS refresh: building the FTS body
    scales with the transcript, so mid-turn autosaves skip it and the
    per-turn save that follows refreshes search instead.
    """
    d = _sessions_dir()
    private_dir(d)
    _prune_empty_assistant_messages(messages)
    # Store form (0.39.0) is what is persisted: image payloads move to the
    # sidecar dir and messages carry image_ref parts. The COPY is compared
    # and written -- the caller's live history keeps its inline parts, so
    # live (request) form and store form never mix in one comparison and a
    # prefix merge stays exact.
    store_messages = _store_form_messages(messages)
    try:
        from . import _sessiondb
        merged = _sessiondb.store_session(session_id, store_messages, meta)
    except Exception:
        merged = None
    if merged is None:
        return
    if _EXITING:
        # Cheap exit form: no merge re-read (nothing will consume the merged
        # view before the process dies) and no GC sweeps (they exist to bound
        # growth across sessions; the next save re-runs them). FTS is kept:
        # a session searched right after a restart must see its last turn.
        if update_index:
            try:
                from . import _sessiondb
                _sessiondb.refresh_fts(
                    session_id, store_messages,
                    title=(meta or {}).get("title"),
                    description=(meta or {}).get("description"))
            except Exception:
                pass
        return
    # store_session returned the authoritative merged list, which covers
    # another writer's turns the merge preserved (so the image-GC live-set
    # protects their payloads too).
    if not isinstance(merged, list):
        merged = store_messages
    if update_index:
        try:
            from . import _sessiondb
            stored_meta: dict[str, Any] = {}
            try:
                stored_meta = _sessiondb._session_meta(session_id)
            except Exception:
                stored_meta = {}
            _sessiondb.refresh_fts(
                session_id, merged,
                title=(meta or {}).get("title") or stored_meta.get("title"),
                description=(meta or {}).get("description")
                or stored_meta.get("description"))
        except Exception:
            pass
        # Stale FTS rows are swept here too -- amortized once per
        # NX_SESSION_SWEEP_INTERVAL seconds (default 300), process-wide.
        # Same P6 gate as before.
        try:
            from . import _sessiondb
            if _sweep_due("prune"):
                _sessiondb.prune_stale()
                # Opt-in retention (disabled by default): ride the existing
                # "prune" gate, no new sweep kind.
                _sessiondb.retention_sweep()
        except Exception:
            pass
    _maybe_gc_image_sidecars(merged)
    # Swarm P10: obsstore TTL/quota, amortized (own gate) -- a packed-
    # result store must not grow forever now that it is shared.
    _maybe_gc_obsstore()


# --- amortized maintenance sweeps (swarm P6) -----------------------------------
# Per-save maintenance that is O(store size) — index prune, image sidecar GC,
# quota check — amortized to at most once per interval, process-wide. A swarm
# of agents sharing NX_HOME used to pay an O(#sessions + #images) stat walk on
# EVERY indexed save. The interval is a ceiling, not a lease: the first agent
# past it does the sweep while the rest skip.

_SWEEP_LOCK = threading.Lock()
_SWEEP_LAST: dict[str, float] = {}

# Exit posture (0.103.28): once set, the store write skips the post-write
# re-read and every amortized GC sweep. The final flush at /quit or atexit
# used to pay the full maintenance bill (an O(store) read-merge-write plus a
# whole-transcript re-read plus any sweep whose interval had elapsed) at the
# one moment the user is waiting for the process to go away. The transcript
# blob itself is still written — only the derived work is skipped; the next
# session's first save re-reads and re-sweeps as usual.
_EXITING = False


def begin_exit() -> None:
    """Switch the save path to its cheap exit form. Idempotent; safe to call
    from /quit, the EOF/^C exit branch, and the atexit hook."""
    global _EXITING
    _EXITING = True


def _sweep_due(kind: str) -> bool:
    interval = _env_int("NX_SESSION_SWEEP_INTERVAL", 300)
    if interval <= 0:
        return True  # 0/negative = legacy every-save behavior
    now = time.monotonic()
    with _SWEEP_LOCK:
        last = _SWEEP_LAST.get(kind, 0.0)
        if now - last < interval:
            return False
        _SWEEP_LAST[kind] = now
        return True


def _gc_textrefs(live: set[str], min_age_s: float = 0.0) -> int:
    """0.116.0: mirror of _gc_image_sidecars for the _textrefs dir — delete
    big-text payloads no session references anymore. Best-effort."""
    d = _textrefs_dir()
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    removed = 0
    now = time.time()
    for name in names:
        if not name.endswith(".txt") or name[:-4] in live:
            continue
        side = os.path.join(d, name)
        if min_age_s > 0:
            try:
                if now - os.path.getmtime(side) < min_age_s:
                    continue
            except OSError:
                continue
        try:
            os.remove(side)
        except OSError:
            pass
        else:
            removed += 1
    return removed


def _maybe_gc_image_sidecars(merged_messages: list) -> None:
    """Sidecar GC (0.39.0), amortized once per NX_SESSION_SWEEP_INTERVAL
    seconds (default 300) — it stats the whole _images/ dir per save
    otherwise. The just-merged messages are unioned in: an unindexed
    mid-turn autosave may have added refs whose row is stale, and a
    concurrent writer's payload gets the same 60s age grace as before.
    Runs whenever image sidecars exist: gating it on NX_SESSION_IMAGES=0
    used to orphan every already-stored payload the moment the rollback
    flag flipped. 0.116.0: the _textrefs big-text sidecars ride the same
    sweep (own live-set scan, same grace)."""
    if not _sweep_due("image_gc"):
        return
    live = _collect_message_image_refs(merged_messages)
    live |= _scan_session_image_refs()
    _gc_image_sidecars(live, min_age_s=60.0)
    _gc_image_quota()
    try:
        tlive = _collect_message_text_refs(merged_messages)
        tlive |= _scan_session_text_refs()
        _gc_textrefs(tlive, min_age_s=60.0)
    except Exception:
        pass  # GC must never break a save


def _maybe_gc_obsstore() -> None:
    """Swarm P10: obsstore TTL/quota GC, amortized on the same P6 gate as
    the image sweep. Runs only when the store dir exists (a session that
    never packs pays nothing — not even a stat)."""
    import os as _os
    from . import _obsstore
    if not _os.path.isdir(_obsstore.store_dir()):
        return
    if not _sweep_due("obs_gc"):
        return
    try:
        _obsstore.gc(grace_s=_obsstore.AGE_GRACE_S)
    except Exception:
        pass  # the store is an optimization; GC must never break a save


def _gc_image_quota() -> int:
    """2.3a size quota for the `_images/` dir: when the payload dir exceeds
    NX_IMAGES_QUOTA_MB (default 200; 0 disables), the OLDEST unreferenced
    payloads are evicted first, then oldest referenced ones, until under
    quota. Content-addressed and deduped, so an evicted payload that is
    still referenced degrades its session's resume to the
    "[image missing from session image store]" note — never a 400 — but
    nothing else breaks. Returns payloads removed.

    Unreferenced-first ordering means quota eviction never destroys live
    data while orphan-free GC has anything left to clean; the referenced
    tier only fires when the user's quota is genuinely smaller than their
    live image set."""
    quota_mb = _env_int("NX_IMAGES_QUOTA_MB", 200)
    if quota_mb <= 0:
        return 0
    limit = quota_mb * 1024 * 1024
    d = _images_dir()
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    payloads: list[tuple[float, str]] = []  # (mtime, name)
    total = 0
    for name in names:
        if not name.endswith(".b64"):
            continue
        try:
            st = os.stat(os.path.join(d, name))
        except OSError:
            continue
        payloads.append((st.st_mtime, name))
        total += st.st_size
    if total <= limit:
        return 0
    payloads.sort()  # oldest first
    live = _scan_session_image_refs()
    # pass 1: unreferenced orphans (age grace does not apply — quota is a
    # storage bound, not a concurrency concern; orphan means nothing on
    # disk references the payload)
    removed = 0
    kept: list[tuple[float, str]] = []
    for mtime, name in payloads:
        if total <= limit:
            kept.append((mtime, name))
            continue
        if name[:-4] in live:
            kept.append((mtime, name))
            continue
        side = os.path.join(d, name)
        try:
            size = os.path.getsize(side)
            os.remove(side)
        except OSError:
            kept.append((mtime, name))
            continue
        total -= size
        removed += 1
    # pass 2: still over quota — evict oldest referenced payloads
    if total > limit:
        kept.sort()
        for mtime, name in kept:
            if total <= limit:
                break
            try:
                size = os.path.getsize(os.path.join(d, name))
                os.remove(os.path.join(d, name))
                total -= size
                removed += 1
            except OSError:
                continue
    return removed


def _scan_session_text_refs() -> set[str]:
    """Every text_ref sha referenced by any stored session — the live-set
    for _textrefs GC. Scans the event log for ref parts (a cheap LIKE on
    the JSON marker, amortized behind the sweep gate like the image scan)."""
    live: set[str] = set()
    try:
        from . import _sessiondb
        conn = _sessiondb._open()
        if conn is None:
            return live
        try:
            # text_ref parts always carry "sha256"; events JSON-escape the
            # quotes, so match the escaped form.
            marker = _sessiondb.TEXT_REF_MARKER
            for (blob,) in conn.execute(
                    "SELECT msg FROM session_events"
                    " WHERE msg LIKE ? LIMIT 200000", (f"%{marker}%",)):
                try:
                    m = json.loads(blob)
                except (ValueError, TypeError):
                    continue
                if isinstance(m, dict):
                    live |= _collect_message_text_refs([m])
        finally:
            conn.close()
    except Exception:
        pass  # the live-set is an optimization; degrade to empty
    return live


def _scan_session_image_refs() -> set[str]:
    """Every image sha referenced by any session file currently on disk —
    the union live-set for sidecar GC.

    Since 2.3b this reads the consolidated store's rows (each carries the
    session's `sha256s`), so the cost is one SQLite query instead of a
    small JSON read per session per save. A session with no row (legacy
    file not yet migrated, mid-write) or a corrupt row falls back to
    parsing that one transcript — tolerant, never lossy: an unreadable
    file contributes nothing to the live-set, exactly as before."""
    live: set[str] = set()
    covered: set[str] = set()  # ids whose row gave us a trustworthy sha256s
    try:
        from . import _sessiondb
        # 0.108.6: the indexed image_shas column answers the scan with one
        # cheap query; per-row transcript parsing (up to the full store
        # materialized) only for rows the column can't answer yet.
        live, covered = _sessiondb.image_live_shas()
        return live | _scan_session_image_files(covered)
    except Exception:
        covered = set()
        live = set()
    # Anything the store could not answer (corrupt blob) falls back to
    # parsing the legacy mirror file, when one still exists.
    try:
        names = [n for n in os.listdir(_sessions_dir()) if n.endswith(".json")]
    except OSError:
        names = []
    for name in names:
        sid = name[:-len(".json")]
        if sid in covered:
            continue
        try:
            with open(os.path.join(_sessions_dir(), name), encoding="utf-8") as f:
                data = json.load(f)
            msgs = data.get("messages") if isinstance(data, dict) else None
            if isinstance(msgs, list):
                live |= _collect_message_image_refs(msgs)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return live


def _scan_session_image_files(covered: set[str]) -> set[str]:
    """Image refs found in legacy mirror files whose session id the store
    could not cover (no row / unreadable row). Split out of the store scan
    at 0.108.6 so the happy path (every row indexed) skips the listdir."""
    out: set[str] = set()
    try:
        names = [n for n in os.listdir(_sessions_dir()) if n.endswith(".json")]
    except OSError:
        return out
    for name in names:
        sid = name[:-len(".json")]
        if sid in covered:
            continue
        try:
            with open(os.path.join(_sessions_dir(), name), encoding="utf-8") as f:
                data = json.load(f)
            msgs = data.get("messages") if isinstance(data, dict) else None
            if isinstance(msgs, list):
                out |= _collect_message_image_refs(msgs)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return out


TEXT_REF_TYPE = "text_ref"


def _textrefs_dir() -> str:
    """Content-addressed sidecar dir for offloaded big-text payloads
    (0.116.0) — same 0700/0600 posture as _images."""
    d = os.path.join(_sessions_dir(), "_textrefs")
    private_dir(d)
    return d


def _text_ref_min() -> int:
    return _env_int("NX_SESSION_TEXT_REF_MIN", 65536)


def _text_to_ref(text: str) -> dict[str, Any] | None:
    """One big text payload → a text_ref part (payload into the sidecar).
    Returns None if the write fails (caller keeps the text inline —
    fallback, never data loss). Writes are idempotent per sha256."""
    try:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        side = os.path.join(_textrefs_dir(), f"{sha}.txt")
        if not os.path.exists(side):
            tmp = f"{side}.{os.getpid()}.tmp"
            fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, side)
    except (OSError, ValueError):
        return None
    return {"type": TEXT_REF_TYPE, "sha256": sha, "bytes": len(text),
            "str": True,
            "text": text[:512] + ("…" if len(text) > 512 else "")}


def _ref_to_text(ref: dict[str, Any]) -> str | None:
    """One text_ref part → the full payload, or None when the sidecar file
    is missing/corrupt (caller keeps the preview — degrade, never lose)."""
    sha = ref.get("sha256")
    if not isinstance(sha, str) or not sha:
        return None
    if "/" in sha or chr(92) in sha or "." in sha:
        return None
    try:
        with open(os.path.join(_textrefs_dir(), f"{sha}.txt"),
                  encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _collect_message_text_refs(msgs: list) -> set[str]:
    """Every sha256 referenced by text_ref parts — the GC live-set."""
    out: set[str] = set()
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == TEXT_REF_TYPE \
                        and isinstance(p.get("sha256"), str):
                    out.add(p["sha256"])
        elif isinstance(m, dict) and isinstance(m.get("content"), str):
            continue  # string content is never offloaded
    return out


def _store_form_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A deep-copied store-form of `messages`: inline image_url parts become
    image_ref sidecar references. Returns a NEW structure — the caller's live
    history is never mutated (its parts are shared dicts also held by the
    request path; writing refs into them would send image_ref parts to the
    API, which 400s). NX_SESSION_IMAGES=0 returns a plain copy with 0.38.x
    inline base64. Sidecar write failures keep that part inline (fallback,
    never data loss). Multi-MB autosaves become KB: the payload is written
    once per unique image, deduped across all sessions by sha256."""
    import copy
    if not _env_int("NX_SESSION_IMAGES", 1):
        out = copy.deepcopy(messages)
        _offload_big_text(out)
        return out
    # Copy-on-write: a history with no image parts (the overwhelmingly
    # common case) is returned as a shallow list copy sharing the message
    # dicts — nothing below mutates them, so the deepcopy per message is
    # wasted work on every save and scaled with the transcript.
    if not any(
        isinstance(p, dict) and p.get("type") == "image_url"
        for m in messages if isinstance(m, dict)
        for p in (m.get("content") if isinstance(m.get("content"), list) else ())
    ):
        out = list(messages)
        _offload_big_text(out)
        return out
    out = copy.deepcopy(messages)
    _offload_big_text(out)
    for m in out:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for i, p in enumerate(c):
            if isinstance(p, dict) and p.get("type") == "image_url":
                # provenance: the read_image result carries its own
                # "[image: <path>, N bytes]" text part (immediately BEFORE
                # the payload part) — reuse it so /sessions and resume
                # diagnostics can name the original file
                sib = None
                for j in (i - 1, i + 1):
                    q = c[j] if 0 <= j < len(c) else None
                    if isinstance(q, dict) and q.get("type") == "text" \
                            and isinstance(q.get("text"), str) \
                            and q["text"].startswith("[image: "):
                        sib = q["text"]
                        break
                ref = _data_url_to_ref(p, sibling_text=sib)
                if ref is not None:
                    p.clear()
                    p.update(ref)
    return out


def _offload_big_text(messages: list[dict[str, Any]]) -> None:
    """0.116.0 store-form pass, in place on the COPY _store_form_messages
    made: any single text payload over NX_SESSION_TEXT_REF_MIN (default
    64 KB) becomes a text_ref part / ref-bearing content — the payload
    moves to the content-addressed _textrefs sidecar, the event carries a
    512-char preview. _load_session rehydrates, so the API/compaction/
    /tooloutput consumers never see a ref. A sidecar write failure keeps
    the text inline (fallback, never data loss)."""
    if not _env_int("NX_SESSION_TEXT_REFS", 1):
        return
    threshold = _text_ref_min()
    if threshold <= 0:
        return
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            if len(c) > threshold:
                ref = _text_to_ref(c)
                if ref is not None:
                    # the ref rides a parts list; rehydration on load
                    # restores the plain string for API consumers
                    m["content"] = [ref]
        elif isinstance(c, list):
            for i, p in enumerate(c):
                if (isinstance(p, dict) and p.get("type") == "text"
                        and isinstance(p.get("text"), str)
                        and len(p["text"]) > threshold):
                    ref = _text_to_ref(p["text"])
                    if ref is not None:
                        ref["str"] = False
                        c[i] = ref


def _inflate_text_refs(messages: list[dict[str, Any]]) -> int:
    """Store → live form on load (mirror of _deserialize_image_parts):
    text_ref parts become full text again. A missing sidecar degrades that
    message to its 512-char preview (never loses the rest of the session)."""
    n = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):
            for i, p in enumerate(c):
                if isinstance(p, dict) and p.get("type") == TEXT_REF_TYPE:
                    full = _ref_to_text(p)
                    if p.get("str"):
                        if full is not None:
                            m["content"] = full
                            n += 1
                            break  # content replaced; parts loop is done
                    else:
                        c[i] = ({"type": "text", "text": full}
                                if full is not None else
                                {"type": "text", "text": p.get("text", "")
                                 + " [payload missing from session store]"})
                        n += 1
        elif isinstance(c, dict) and c.get("type") == TEXT_REF_TYPE:
            full = _ref_to_text(c)
            if full is not None:
                m["content"] = full
                n += 1
    return n

def _deserialize_image_parts(messages: list[dict[str, Any]]) -> int:
    """Store → live form on resume: image_ref parts become inline image_url
    parts again (the API never sees a ref). A missing/corrupt sidecar file
    degrades that part to a text note instead of a 400 on the next request."""
    n = 0
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for p in c:
            if isinstance(p, dict) and p.get("type") == IMAGE_REF_TYPE:
                live = _ref_to_data_url_part(p)
                if live is not None:
                    p.clear()
                    p.update(live)
                else:
                    p.clear()
                    p["type"] = "text"
                    p["text"] = "[image missing from session image store]"
                n += 1
    return n


def _load_session(session_id: str) -> dict[str, Any] | None:
    """Read a session from the store. Returns the dict (id, messages,
    meta...) or None. A legacy mirror file that predates the store is
    imported first (lazy migration, idempotent).

    0.103.4: flushes a pending async save first (nx/_session_save.py) so a
    reader — /resume, a staleness check, /compress's read-back — never sees a
    transcript one turn behind what the REPL already has."""
    from . import _session_save
    _session_save.flush_pending()
    from . import _sessiondb
    data = _sessiondb.load_session(session_id)
    if data is None:
        return None
    _prune_empty_assistant_messages(data["messages"])
    # 0.116.0: text_ref sidecar payloads → full text (mirror of the image
    # rehydration); a missing sidecar degrades to the 512-char preview.
    _inflate_text_refs(data["messages"])
    return data


def _session_files_newest() -> list[str]:
    """Session ids newest-first, as legacy-style `<id>.json` names.

    0.101.0 the order comes from the store's `updated_at` column (one
    indexed query); the .json suffix is kept only so callers that slice
    ids out of these names (and tests that spy on
    _session_summary_from_file) keep working unchanged."""
    from . import _sessiondb
    rows = _sessiondb.list_ids_newest()
    return [f"{sid}.json" for sid, _ in rows]


def _session_summary_from_file(fname: str, idx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the listing summary for one session id (given in the legacy
    `<id>.json` filename form). 0.101.0 the summary is derived from the
    transcript row in the store; `idx` (a precomputed _load_index dict) is
    still honored as a cache, and a session with no row returns None."""
    sid = fname[:-5] if fname.endswith(".json") else fname
    if idx is not None:
        cached = idx.get(sid)
        if cached is not None:
            return cached
    from . import _sessiondb
    data = _sessiondb.load_session(sid)
    if not data:
        return None
    return _summarize_session_data(data.get("id") or sid, data)


def _list_sessions(limit: int | None = 20, offset: int = 0, query: str | None = None) -> list[dict[str, Any]]:
    """Return sessions newest-first as dicts: id, title, description, preview, n.

    0.129: the listing is COLUMN-ONLY — preview and n come from the
    sessions row's indexed listing columns, so an open does O(page) work,
    never an event-log read or transcript parse. Pre-0.129 rows (NULL
    columns) self-heal once: just the returned page is inflated via
    load_session and its columns backfilled. `description` is an optional
    model-generated one-liner refreshed every N turns (richer than the
    static first-message title)."""
    from . import _session_save
    # A listing must show the newest transcript, not the last landed one.
    _session_save.flush_pending()
    limit = max(0, int(limit)) if limit is not None else None
    offset = max(0, int(offset or 0))
    from . import _sessiondb
    if query:
        # Scan ALL matching sessions, then re-sort: a match may be older
        # than page 1 (tests pin this). Still no event-log reads.
        rows = _sessiondb.list_summaries()
    else:
        rows = _sessiondb.list_summaries(
            limit=None if limit is None else offset + limit)
    if query:
        rows = [s for s in rows if _session_matches_query(s, query)]
        rows.sort(key=lambda s: s["updated_at"], reverse=True)
    end = None if limit is None else offset + limit
    page = rows[offset:end]
    # Bounded self-heal: only rows we are about to return (page-size, not
    # store-size). Computes the real summary from the event log once and
    # backfills the columns, so the next listing is column-only again.
    for row in page:
        if not row.pop("heal", False):
            continue
        data = _sessiondb.load_session(row["id"])
        if not data:
            continue
        fixed = _summarize_session_data(row["id"], data)
        fixed.pop("heal", None)
        row.update(fixed)
        try:
            _sessiondb.heal_listing_row(row["id"], fixed["preview"],
                                        fixed["n"])
        except Exception:
            pass  # best-effort backfill; the NULL fallback still serves
    return page


def _prune_sessions_store() -> dict[str, int]:
    """`nx sessions --prune` (2.3a/2.3b): reconcile the store by hand.

    1. Store rows (and any legacy shards) whose session JSON is gone
       (crash, manual rm — /delete already removes both) are dropped, so
       listings, search and the GC live-set stop counting a dead session.
       2. Unreferenced image payloads are collected immediately, no age
       grace. 3. The size quota runs last. Returns a small stats dict for
       the CLI to print."""
    from . import _sessiondb
    # 0. consolidated store: stale FTS rows (transcript row gone — a crash
    # between the two deletes is the only way now)
    try:
        _sessiondb._ensure_migrated()
        rows_pruned = _sessiondb.prune_stale()
    except Exception:
        rows_pruned = 0
    # 0b. opt-in retention (disabled by default): age x size gates.
    try:
        retention = _sessiondb.retention_sweep()
    except Exception:
        retention = {"expired_small": 0, "expired_floor": 0}
    # 1. orphaned legacy shards (pre-2.3b store)
    shards_dropped = 0
    try:
        shard_names = [n for n in os.listdir(_index_dir()) if n.endswith(".json")]
    except OSError:
        shard_names = []
    for name in shard_names:
        sid = name[:-len(".json")]
        if not os.path.exists(_session_path(sid)):
            try:
                os.remove(_index_entry_path(sid))
                shards_dropped += 1
            except OSError:
                pass
    # 1c. undo GC (0.129.0): any persisted undo file whose session is dead
    # (deleted by hand, a crash, or an old /delete predating the GC) is
    # removed — its snapshots are whole file contents, often secrets. The
    # CURRENT session's file is never a candidate, and /reset intentionally
    # keeps its predecessor's file (that session is still resumable).
    undo_removed = 0
    try:
        from . import _undo
        undo_dir = os.path.join(st.SESSION_HOME, "undo")
        for name in os.listdir(undo_dir):
            if not name.endswith(".json"):
                continue
            sid = name[:-len(".json")]
            if sid in ("undo", getattr(st, "SESSION_ID", None)):
                continue  # legacy shared file / the live session
            if os.path.exists(_session_path(sid)):
                continue
            _undo.delete_persisted_for(sid)
            undo_removed += 1
    except OSError:
        pass
    # 2. immediate GC (no grace: the user asked)
    live = _scan_session_image_refs()
    images_removed = _gc_image_sidecars(live, min_age_s=0.0)
    try:
        kept = len([n for n in os.listdir(_images_dir()) if n.endswith(".b64")])
    except OSError:
        kept = 0
    # 3. quota
    quota_removed = _gc_image_quota()
    # 4. swarm P10: the shared tool-result store (no grace — explicit)
    try:
        from . import _obsstore
        obs_removed = _obsstore.gc(grace_s=0.0)["removed"]
        obs_stats = _obsstore.store_stats()
    except Exception:
        obs_removed, obs_stats = 0, {"objects": 0, "bytes": 0}
    return {"shards_dropped": shards_dropped, "images_removed": images_removed,
            "images_kept": kept, "quota_removed": quota_removed,
            "rows_pruned": rows_pruned, "obs_removed": obs_removed,
            "obs_kept": obs_stats["objects"], "obs_bytes": obs_stats["bytes"],
            "undo_removed": undo_removed,
            "expired_small": retention["expired_small"],
            "expired_floor": retention["expired_floor"]}


def _delete_session(session_id: str) -> bool:
    """Remove a session from the store. Returns True if something was
    deleted.

    0.101.0: drops the transcript row + its FTS row (a deleted id can no
    longer surface in `/sessions <query>` via a stale FTS row), then runs
    the sidecar GC immediately: a deleted session's images used to linger
    until some *other* session happened to save. The row removal MUST
    precede the GC (it is the ref list being dropped); the scan covers
    every other session, and the 60s grace protects concurrent writers.
    A legacy mirror file, if one still exists, is removed too."""
    from . import _session_save
    # Never let a queued save resurrect a session the user just deleted.
    _session_save.flush_pending()
    from . import _sessiondb
    existed = _sessiondb.load_session(session_id) is not None
    path = _session_path(session_id)
    try:
        os.remove(path)
    except OSError:
        pass
    # Undo GC (0.129.0): the persisted checkpoints hold whole pre-edit file
    # contents (secrets, up to MAX_UNDO_BYTES) keyed by this sid. The session
    # is gone either way — drop the file even when the row was already missing.
    from . import _undo
    _undo.delete_persisted_for(session_id)
    if not existed:
        return False
    try:
        _sessiondb.delete_session_row(session_id)
    except Exception:
        pass
    # GC-on-delete: with the rows gone, nothing (else) references the
    # session's unique images. Best-effort, never raises.
    try:
        live = _scan_session_image_refs()
        _gc_image_sidecars(live, min_age_s=0.0)
        _gc_image_quota()
    except Exception:
        pass
    return True


_DESC_REFRESH_LOCK = threading.Lock()


def _maybe_refresh_description(session_id: str, messages: list[dict[str, Any]]) -> str | None:
    """Schedule a one-line session description refresh, if enough turns have
    passed since the last one.

    Refreshes at the configured interval (every SESSION_DESC_REFRESH user
    turns). The description is stored in the session file's `description`
    field and surfaced in `nx sessions` / `/sessions` listings.

    PF-09 (0.56.0): this used to make the (20s-timeout) non-streaming model
    call INLINE on the REPL thread, every SESSION_DESC_REFRESH turns, right
    after the answer rendered. Nothing consumes the value inline (the REPL
    discards the return), so the call now runs on a daemon thread and the
    prompt returns immediately. Always returns None now.
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
    # 0.103.4: reads the store WITHOUT the flush-before-read gate that
    # `_load_session` applies. This runs after every turn (right after the
    # async autosave was queued) and `desc_turns` is a refresh-threshold
    # counter where a one-save-stale read is harmless — flushing here would
    # block on the writer thread and put the save straight back on the
    # prompt's critical path, defeating NX_ASYNC_SAVE.
    from . import _sessiondb
    existing = _sessiondb.load_session(session_id) or {}
    last_desc_turns = existing.get("desc_turns", 0)
    # Refresh when we've crossed a multiple of SESSION_DESC_REFRESH since the
    # last recorded refresh. Don't refresh before the first threshold (so a
    # 1-turn "hello" session doesn't burn a model call).
    if user_turns < _m.SESSION_DESC_REFRESH:
        return None
    if user_turns - last_desc_turns < _m.SESSION_DESC_REFRESH:
        return None
    # Snapshot the transcript (a shallow copy of the message pointers) so the
    # background thread never iterates a list the REPL is appending to.
    snapshot = list(messages)
    # Snapshot the metrics ON THE CALLER THREAD (0.128.3): reading st._METRICS
    # at thread completion reads whatever recorder is current THEN — after
    # /reset//fork that is the fresh session's near-zero totals, and merging
    # them into this (old) session's row destroys its recorded bill.
    metrics_meta = st._METRICS.as_meta() if st._METRICS is not None else None
    threading.Thread(
        target=_run_desc_refresh, args=(session_id, snapshot, user_turns, metrics_meta),
        daemon=True, name="nx-session-desc").start()
    return None


def _run_desc_refresh(session_id: str, messages: list[dict[str, Any]],
                      user_turns: int,
                      metrics_meta: dict[str, Any] | None = None) -> str | None:
    """The background half of the refresh: the model call + the persist.

    A failure (server down, empty response) leaves the existing description
    untouched and is silent — the title is always present. Returns the new
    description, or None if no refresh happened. PF-09 (0.56.0).
    """
    import nx as _m
    if not _DESC_REFRESH_LOCK.acquire(blocking=False):
        return None  # a refresh is already in flight — don't stack them
    try:
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
            fallback_prompt_chars=lambda: sum(len(str(m.get("content") or "")) for m in payload),
        ) if st.SESSION_ID == session_id else None  # stale thread: don't misbill
        desc = (resp.choices[0].message.content or "").strip().splitlines()
        desc = desc[0].strip() if desc else ""
        if not desc:
            return None
        desc = _safe_title(desc, maxlen=70) or desc[:70]
        # Persist the description + the turn count it was generated at so we
        # know when to refresh next. Merge into existing meta (don't clobber
        # messages); the write is lock-guarded inside _write_session, so a
        # background refresh serializes safely with the REPL's per-turn save.
        meta = {"description": desc, "desc_turns": user_turns}
        if metrics_meta is not None:
            meta.update(metrics_meta)
        # Identity guard (0.128.3): the session was switched away (or deleted)
        # while this thread ran — skip BOTH the write and the flush. Writing
        # would merge the snapshot into a row the user left (and, after a
        # /delete, resurrect the row outright via store_session).
        if st.SESSION_ID != session_id:
            return None
        _write_session(session_id, messages, meta)
        if metrics_meta is not None and st._METRICS is not None:
            st._METRICS.flush()
        return desc
    finally:
        _DESC_REFRESH_LOCK.release()


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
