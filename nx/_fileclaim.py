"""Cross-process file-claim protocol (0.106.x).

A one-file-at-a-time advisory claim so two nx processes (two REPLs, a
sub-agent's own loop, another host sharing the tree) don't clobber the same
file. The store is the same SQLite session DB (`nx/_sessiondb.py`), so a claim
is visible across processes on one host without a lock server.

Design rules (pinned by tests/test_file_claims.py):
  * never raise into callers — any store failure fails OPEN (the edit proceeds,
    the claim is best-effort);
  * no module-level sqlite3.Connection — a fresh `_open()` per call;
  * paths resolved at call time (env read at call time, no import-time freeze).

Liveness: an existing row blocks a *different* owner only while (a) its pid is
alive, (b) it hasn't aged past `NX_FILE_CLAIM_TTL` seconds (default 3600; `0`
disables expiry), AND (c) it was claimed from the SAME working directory as
this process. A dead pid, an expired row, or a row taken from another repo (a
different `os.getcwd()` — e.g. an nx session working in an unrelated folder on
this host) is reclaimed in place: claims are repo-scoped, never host-wide.
A live row held by the SAME owner is refreshed (heartbeat) and returns ok, so
several write tools in one process/loop re-claim cleanly.

Scoping: `cwd` is `os.getcwd()` at claim time (call-time, like everything
else here), normalized with realpath. Rows written before the column existed
carry NULL, which reads as "another repo": they stop blocking and the next
writer reclaims them (see `_sessiondb._ensure_file_claims_cwd`).

Identity: `owner_id` is `<NX session id>:<thread ident>`. SESSION_ID alone is
too coarse (every tool in a session shares it) and the per-agent TLS ctx is
cleared while a write-tier body runs (`nx/_subagent_tiers.py`), so the thread
ident is the stable per-loop discriminator: tools on one event-loop thread
re-claim cleanly, and `release_for` with that tag drops exactly that loop's
claims. `pid` is os.getpid().
"""
from __future__ import annotations

import os
import threading
import time

from . import _state as st
from ._env import _env_int

DEFAULT_TTL = 3600


def _claims_enabled() -> bool:
    """Read at call time (no import-time env freeze). `NX_FILE_CLAIMS=0`
    disables the whole protocol."""
    return _env_int("NX_FILE_CLAIMS", 1) != 0


def _ttl() -> int:
    """Seconds a claim stays valid without a heartbeat; 0 = never expires."""
    return _env_int("NX_FILE_CLAIM_TTL", DEFAULT_TTL)


def _cwd() -> str:
    """This process's working directory at call time, realpath-normalized so a
    symlinked path or trailing slash doesn't fork one repo into two claim
    scopes. Empty when the cwd is unreadable (deleted) — see _same_repo."""
    try:
        return os.path.realpath(os.getcwd())
    except OSError:
        return ""


def _same_repo(holder_cwd: object) -> bool:
    """True when a row's recorded cwd belongs to this process's repo.

    A foreign cwd (and a pre-migration NULL row, which records nothing) is
    NOT this repo, so it never blocks — claim_file reclaims it the same way it
    reclaims a dead or expired holder. When our own cwd can't be read we keep
    the old host-wide behavior (block) rather than silently reclaiming rows we
    can't prove are foreign."""
    mine = _cwd()
    if not mine:
        return True
    return str(holder_cwd or "") == mine


def owner_id() -> str:
    """This agent's claim identity, composed at call time. See module docstring
    for why the thread ident is the per-loop discriminator."""
    return f"{st.SESSION_ID}:{threading.get_ident()}"


def _holder_desc(owner: str, pid: object) -> str:
    return f"owner {owner}, pid {pid}"


def _pid_alive(pid: object) -> bool:
    """os.kill(pid, 0) semantics: True when the process exists (or exists but is
    owned by another user — PermissionError means alive, not gone)."""
    try:
        pid = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def claim_file(path: str, owner_id: str) -> tuple[bool, str | None]:
    """Atomically claim `path` for `owner_id`.

    Returns (True, None) when the claim is held (fresh, refreshed-heartbeat, or
    reclaimed from a dead/expired/other-repo holder), or (False, holder_desc)
    when a live holder in THIS repo with a DIFFERENT owner_id holds it. Never
    raises: a store that can't serve the call fails open (True, None).
    """
    if not path or not _claims_enabled():
        return (True, None)
    from . import _sessiondb
    conn = _sessiondb._open()
    if conn is None:
        return (True, None)
    ttl = _ttl()
    now = time.time()
    pid = os.getpid()
    cwd = _cwd()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT owner_id, pid, claimed_at, cwd FROM file_claims"
                " WHERE path = ?",
                (path,)).fetchone()
            result: tuple[bool, str | None]
            if row is not None:
                holder_owner = row[0]
                holder_pid = row[1]
                claimed_at = row[2] or 0.0
                expired = ttl > 0 and (now - claimed_at) > ttl
                if (_pid_alive(holder_pid) and not expired
                        and _same_repo(row[3])):
                    if holder_owner == owner_id:
                        # Same owner: heartbeat-refresh, stay held.
                        conn.execute(
                            "UPDATE file_claims SET claimed_at = ?"
                            " WHERE path = ?",
                            (now, path))
                        result = (True, None)
                    else:
                        result = (False, _holder_desc(holder_owner, holder_pid))
                else:
                    # Dead pid, expired, or another repo's claim: reclaim here.
                    conn.execute("DELETE FROM file_claims WHERE path = ?", (path,))
                    conn.execute(
                        "INSERT INTO file_claims(path, owner_id, pid, claimed_at,"
                        " cwd) VALUES (?,?,?,?,?)",
                        (path, owner_id, pid, now, cwd))
                    result = (True, None)
            else:
                conn.execute(
                    "INSERT INTO file_claims(path, owner_id, pid, claimed_at,"
                    " cwd) VALUES (?,?,?,?,?)",
                    (path, owner_id, pid, now, cwd))
                result = (True, None)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
    except Exception:
        return (True, None)  # fail open: the edit is never blocked by a DB hiccup
    finally:
        try:
            conn.close()
        except Exception:
            pass


def release_for(owner_id: str) -> int:
    """Drop every claim held by `owner_id`. Returns the number removed (0 on any
    failure). Never raises."""
    if not owner_id or not _claims_enabled():
        return 0
    from . import _sessiondb
    conn = _sessiondb._open()
    if conn is None:
        return 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                "DELETE FROM file_claims WHERE owner_id = ?", (owner_id,))
            n = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            conn.commit()
            return n
        except Exception:
            conn.rollback()
            raise
    except Exception:
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass
