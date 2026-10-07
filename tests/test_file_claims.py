#!/usr/bin/env python3
"""nx/_fileclaim.py: the cross-process file-claim protocol.

Pins: claim ok; cross-owner refusal; expired-row reclaim; dead-pid reclaim;
other-repo (different cwd) reclaim — claims are repo-scoped, not host-wide;
the pre-migration NULL-cwd row; the cwd column's ALTER TABLE migration;
`NX_FILE_CLAIMS=0` no-op (no rows, always ok); `NX_FILE_CLAIM_TTL=0` never
expires; same-owner heartbeat refresh; release_for; and the write-tool wiring
(two write tools in one process re-claim the same file cleanly; a foreign
holder refuses).

0.119.0 adds three pins: the `pid_start_time` PID-reuse guard (and its
NULL-column back-compat + ALTER TABLE migration), the refusal message naming
the real TTL remedy, and `release_all_for_pid` on the REPL exit path.
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="nx-fileclaim-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_FILE_CLAIM_TTL"] = "3600"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx  # noqa: E402
import nx._bash_jobs  # noqa: E402
from nx import _fileclaim as fc  # noqa: E402
from nx import _sessiondb as db  # noqa: E402


def _clear_claims():
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    conn = db._open()
    assert conn is not None
    try:
        conn.execute("DELETE FROM file_claims")
        conn.commit()
    finally:
        conn.close()


def setup_function(_fn):
    os.environ["NX_FILE_CLAIMS"] = "1"
    os.environ["NX_FILE_CLAIM_TTL"] = "3600"
    _clear_claims()


def teardown_function(_fn):
    for k in ("NX_FILE_CLAIMS", "NX_FILE_CLAIM_TTL"):
        os.environ.pop(k, None)


def _rows():
    conn = db._open()
    assert conn is not None
    try:
        return conn.execute(
            "SELECT path, owner_id, pid, claimed_at, cwd"
            " FROM file_claims").fetchall()
    finally:
        conn.close()


def _start_times():
    conn = db._open()
    assert conn is not None
    try:
        return [r[0] for r in conn.execute(
            "SELECT pid_start_time FROM file_claims")]
    finally:
        conn.close()


def _insert(path, owner, pid, claimed_at, cwd=None, start_time=None):
    """Plant a row directly. `cwd` defaults to THIS process's cwd, i.e. a claim
    taken from the same repo (the one that blocks); pass a cwd of your own to
    stage another repo's holder. `start_time` defaults to NULL, i.e. a
    pre-0.119.0 row (bare pid liveness)."""
    conn = db._open()
    assert conn is not None
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR REPLACE INTO file_claims"
            "(path, owner_id, pid, claimed_at, cwd, pid_start_time)"
            " VALUES (?,?,?,?,?,?)",
            (path, owner, pid, claimed_at,
             fc._cwd() if cwd is None else cwd, start_time))
        conn.commit()
    finally:
        conn.close()


_HAS_PROC = os.path.isdir("/proc") and fc._pid_start_time() is not None


# --- identity ----------------------------------------------------------------

def test_owner_id_is_stable_within_a_thread_and_names_the_session():
    st = nx.st
    st.SESSION_ID = "sess-abc"
    try:
        a = fc.owner_id()
        b = fc.owner_id()
        assert a == b
        assert a.startswith("sess-abc:")
    finally:
        st.SESSION_ID = None


# --- core claim protocol -----------------------------------------------------

def test_claim_ok_when_no_row(tmp_path):
    p = str(tmp_path / "a.txt")
    ok, holder = fc.claim_file(p, "owner-1")
    assert ok and holder is None
    rows = _rows()
    assert len(rows) == 1 and rows[0][0] == os.path.realpath(p) and rows[0][1] == "owner-1"
    assert rows[0][2] == os.getpid()
    assert rows[0][4] == fc._cwd()  # outside any repo -> scope falls back to cwd


def test_cross_owner_refused_while_pid_alive(tmp_path):
    p = str(tmp_path / "b.txt")
    _insert(p, "other-owner", os.getpid(), time.time())
    ok, holder = fc.claim_file(p, "me")
    assert ok is False
    assert "other-owner" in holder and str(os.getpid()) in holder
    # the foreign row survives untouched
    assert _rows()[0][1] == "other-owner"


def test_same_owner_reclaim_is_a_heartbeat(tmp_path):
    p = str(tmp_path / "c.txt")
    assert fc.claim_file(p, "me")[0]
    _insert(p, "me", os.getpid(), 100.0)  # stale-but-live row by us
    ok, holder = fc.claim_file(p, "me")
    assert ok and holder is None
    assert _rows()[0][3] > 100.0  # claimed_at refreshed


def test_expired_row_is_reclaimed(tmp_path):
    p = str(tmp_path / "d.txt")
    os.environ["NX_FILE_CLAIM_TTL"] = "10"
    # live pid (ours) but claimed long ago -> expired
    _insert(p, "other-owner", os.getpid(), time.time() - 9999)
    ok, holder = fc.claim_file(p, "me")
    assert ok and holder is None
    assert _rows()[0][1] == "me"


def test_ttl_zero_never_expires(tmp_path):
    p = str(tmp_path / "e.txt")
    os.environ["NX_FILE_CLAIM_TTL"] = "0"
    _insert(p, "other-owner", os.getpid(), time.time() - 10_000_000)
    ok, holder = fc.claim_file(p, "me")
    assert ok is False and "other-owner" in holder


def test_dead_pid_is_reclaimed(tmp_path):
    p = str(tmp_path / "f.txt")
    proc = subprocess.Popen(["true"])
    proc.wait()
    dead_pid = proc.pid
    _insert(p, "ghost", dead_pid, time.time())
    ok, holder = fc.claim_file(p, "me")
    assert ok and holder is None
    assert _rows()[0][1] == "me"


# --- PID-reuse liveness guard (0.119.0) --------------------------------------

def test_claim_records_the_holders_start_time(tmp_path):
    """A fresh claim stores this process's /proc start-time tick, so a later
    check can tell a recycled pid from the original holder. On a host without
    /proc the column is NULL — asserted so the test is meaningful either way."""
    p = str(tmp_path / "pid1.txt")
    assert fc.claim_file(p, "me")[0]
    stored = _start_times()[0]
    if _HAS_PROC:
        assert stored and stored == fc._pid_start_time()
    else:
        assert stored is None


def test_recycled_pid_with_a_different_start_time_is_reclaimed(tmp_path):
    """The bug: a row from a process that exited, whose pid the OS reused. Bare
    os.kill(pid, 0) sees the NEW process and reads a dead holder as live,
    blocking this repo until the TTL. With the recorded start time compared,
    the row is reclaimed instead."""
    if not _HAS_PROC:
        import pytest
        pytest.skip("no /proc: no start-time guard to exercise (macOS/Windows)")
    p = str(tmp_path / "pid2.txt")
    # OUR pid is alive, but the row records a DIFFERENT incarnation of it.
    _insert(p, "recycled-holder", os.getpid(), time.time(),
            start_time="999999999")
    ok, holder = fc.claim_file(p, "me")
    assert ok is True and holder is None
    row = _rows()[0]
    assert row[1] == "me"
    assert _start_times()[0] == fc._pid_start_time()  # re-stamped here


def test_same_start_time_still_blocks(tmp_path):
    """The guard must not over-reclaim: a live holder whose recorded start time
    MATCHES its current one is still the original process, so it still blocks."""
    if not _HAS_PROC:
        import pytest
        pytest.skip("no /proc: no start-time guard to exercise (macOS/Windows)")
    p = str(tmp_path / "pid3.txt")
    _insert(p, "real-holder", os.getpid(), time.time(),
            start_time=fc._pid_start_time())
    ok, holder = fc.claim_file(p, "me")
    assert ok is False and "real-holder" in holder
    assert _rows()[0][1] == "real-holder"  # untouched


def test_null_start_time_falls_back_to_bare_pid_liveness(tmp_path):
    """Back-compat: a pre-0.119.0 row has no start time to compare, so it must
    keep the old behavior — a live pid blocks — rather than being treated as a
    recycled holder and stolen."""
    p = str(tmp_path / "pid4.txt")
    _insert(p, "legacy-holder", os.getpid(), time.time(), start_time=None)
    ok, holder = fc.claim_file(p, "me")
    assert ok is False and "legacy-holder" in holder
    assert _start_times()[0] is None


def test_pid_start_time_column_migrated_into_a_legacy_store(tmp_path):
    """A store created before 0.119.0 has no pid_start_time; _ensure_schema adds
    it via ALTER TABLE (NULL-backfilled) and preserves the rows."""
    legacy = os.path.join(_tmp, "legacy-pidstart.sqlite")
    conn = sqlite3.connect(legacy)
    try:
        conn.execute(
            "CREATE TABLE file_claims(path TEXT PRIMARY KEY,"
            " owner_id TEXT NOT NULL, pid INTEGER NOT NULL, claimed_at REAL,"
            " cwd TEXT)")
        conn.execute(
            "INSERT INTO file_claims VALUES ('/y', 'old-owner', 1, 2.0, '/repo')")
        conn.commit()
        db._ensure_schema(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(file_claims)")]
        assert "pid_start_time" in cols
        row = conn.execute(
            "SELECT owner_id, cwd, pid_start_time FROM file_claims"
            " WHERE path = '/y'").fetchone()
        assert row == ("old-owner", "/repo", None)  # NULL, row preserved
        db._ensure_schema(conn)  # idempotent — a second pass must not raise
    finally:
        conn.close()


# --- path normalization + file-scoped repo scoping (0.110.0) -----------------

def test_relative_and_absolute_paths_share_one_row(tmp_path, monkeypatch):
    """The 0.110.0 false-block: a session in a PARENT folder claimed
    'SHMForge/x.py' (relative), a session inside SHMForge used the absolute
    path — two rows for one real file. realpath-normalized, one row."""
    f = tmp_path / "real.txt"
    f.write_text("x")
    monkeypatch.chdir(tmp_path)
    assert fc.claim_file("real.txt", "relative-speller")[0]
    rows = _rows()
    assert len(rows) == 1 and rows[0][0] == str(f)
    # the absolute spelling of the same file from the same cwd hits that row
    ok, holder = fc.claim_file(str(f), "me")
    assert ok is False and "relative-speller" in holder
    assert len(_rows()) == 1  # no second row for the same file


def test_parent_folder_claim_does_not_block_the_owning_repo(tmp_path):
    """A session sitting in the PARENT of two repos claims a file in one of
    them; its claim is scoped to that file's repo, not to the parent, so a
    different-owner claim in an UNRELATED repo (or the parent itself, outside
    any repo) is not blocked."""
    repos = tmp_path / "Projects"
    a = repos / "RepoA"; b = repos / "RepoB"
    (a / ".git").mkdir(parents=True); (b / ".git").mkdir(parents=True)
    fa = a / "x.py"; fa.write_text("x")
    os.chdir(repos)  # the parent — not a repo
    try:
        assert fc.claim_file("RepoA/x.py", "parent-sitter")[0]
        row = _rows()[0]
        assert row[4] == str(a)  # scoped to RepoA, not the Projects parent
        # a different owner claiming a RepoB file is NOT blocked by RepoA's row
        fb = b / "y.py"; fb.write_text("y")
        ok, holder = fc.claim_file(str(fb), "repo-b-worker")
        assert ok and holder is None
        # ...but the same RepoA file IS blocked for another owner
        ok, holder = fc.claim_file(str(fa), "repo-a-worker")
        assert ok is False and "parent-sitter" in holder
    finally:
        os.chdir(_tmp)


# --- repo scoping (cwd) ------------------------------------------------------

def test_other_repo_claim_does_not_block(tmp_path):
    """The regression: a live nx process working in ANOTHER folder must not
    refuse an edit here. A different-cwd row is treated like a dead/expired
    holder — reclaimed in place."""
    p = str(tmp_path / "q.txt")
    _insert(p, "other-repo-owner", os.getpid(), time.time(),
            cwd=os.path.join(_tmp, "some-other-repo"))
    ok, holder = fc.claim_file(p, "me")
    assert ok and holder is None
    row = _rows()[0]
    assert row[1] == "me"                      # reclaimed in place
    assert row[4] == fc._cwd()                 # re-scoped to this repo
    # and the reclaimed claim now DOES block a different owner in this repo
    ok, holder = fc.claim_file(p, "other")
    assert ok is False and "me" in holder


def test_pre_migration_null_cwd_row_does_not_block(tmp_path):
    """Rows written before the cwd column existed carry NULL, which reads as
    "not this repo": they stop blocking and the next writer reclaims them."""
    p = str(tmp_path / "r.txt")
    _insert(p, "legacy-owner", os.getpid(), time.time(), cwd=None)
    conn = db._open()
    assert conn is not None
    try:
        conn.execute("UPDATE file_claims SET cwd = NULL WHERE path = ?", (p,))
        conn.commit()
    finally:
        conn.close()
    assert _rows()[0][4] is None
    ok, holder = fc.claim_file(p, "me")
    assert ok and holder is None
    assert _rows()[0][1] == "me"


def test_same_repo_row_still_blocks(tmp_path):
    """Scope is the cwd, not the file's location: a live same-cwd foreign owner
    refuses even when the file itself lives elsewhere."""
    p = str(tmp_path / "s.txt")
    _insert(p, "same-repo-owner", os.getpid(), time.time(), cwd=fc._cwd())
    ok, holder = fc.claim_file(p, "me")
    assert ok is False and "same-repo-owner" in holder


def test_cwd_column_migrated_into_a_legacy_store(tmp_path):
    """An existing DB's file_claims has no `cwd`; _ensure_schema adds it via
    ALTER TABLE and keeps the rows."""
    legacy = os.path.join(_tmp, "legacy.sqlite")
    conn = sqlite3.connect(legacy)
    try:
        conn.execute(
            "CREATE TABLE file_claims(path TEXT PRIMARY KEY,"
            " owner_id TEXT NOT NULL, pid INTEGER NOT NULL, claimed_at REAL)")
        conn.execute(
            "INSERT INTO file_claims VALUES ('/x', 'old-owner', 1, 2.0)")
        conn.commit()
        db._ensure_schema(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(file_claims)")]
        assert "cwd" in cols
        row = conn.execute(
            "SELECT owner_id, cwd FROM file_claims WHERE path = '/x'").fetchone()
        assert row == ("old-owner", None)  # backfilled NULL, row preserved
        db._ensure_schema(conn)  # idempotent — a second pass must not raise
    finally:
        conn.close()


# --- NX_FILE_CLAIMS=0 --------------------------------------------------------

def test_claims_disabled_is_a_noop(tmp_path):
    os.environ["NX_FILE_CLAIMS"] = "0"
    p = str(tmp_path / "g.txt")
    _insert(p, "other-owner", os.getpid(), time.time())
    assert fc.claim_file(p, "me") == (True, None)
    assert fc.release_for("other-owner") == 0
    assert _rows()[0][1] == "other-owner"  # untouched


# --- release_for -------------------------------------------------------------

def test_release_for_drops_only_that_owner(tmp_path):
    a, b = str(tmp_path / "h.txt"), str(tmp_path / "i.txt")
    assert fc.claim_file(a, "me")[0]
    assert fc.claim_file(b, "you")[0]
    assert fc.release_for("me") == 1
    assert [r[1] for r in _rows()] == ["you"]


def test_release_for_missing_owner_is_zero(tmp_path):
    assert fc.release_for("nobody") == 0


# --- release_all_for_pid: the session-exit sweep (0.119.0) -------------------

def test_release_all_for_pid_drops_this_pids_rows(tmp_path):
    """One process runs several event loops, so its claims carry several
    owner_ids but one pid: the exit sweep keys on the pid and drops them all,
    leaving another process's rows alone."""
    a, b, c = (str(tmp_path / n) for n in ("r1.txt", "r2.txt", "r3.txt"))
    assert fc.claim_file(a, "sess:100")[0]
    assert fc.claim_file(b, "sess:200")[0]   # a second loop in this process
    assert fc.claim_file(c, "other-live-hunter")[0]
    conn = db._open()
    assert conn is not None
    try:  # the foreign row is another process's — give it a foreign pid
        conn.execute("UPDATE file_claims SET pid = ? WHERE path = ?",
                     (os.getpid() + 100000, c))
        conn.commit()
    finally:
        conn.close()
    assert fc.release_all_for_pid() == 2
    assert [r[1] for r in _rows()] == ["other-live-hunter"]


def test_release_all_for_pid_is_zero_and_safe_when_empty(tmp_path):
    assert fc.release_all_for_pid() == 0
    assert fc.release_all_for_pid(os.getpid()) == 0
    assert fc.release_all_for_pid("not-a-pid") == 0  # never raises


def test_release_all_for_pid_disabled_or_store_down_is_zero(tmp_path, monkeypatch):
    p = str(tmp_path / "r4.txt")
    assert fc.claim_file(p, "me")[0]
    os.environ["NX_FILE_CLAIMS"] = "0"
    assert fc.release_all_for_pid() == 0
    assert _rows()[0][1] == "me"  # still there: the disabled run was a no-op
    os.environ["NX_FILE_CLAIMS"] = "1"
    monkeypatch.setattr(db, "_open", lambda: None)
    assert fc.release_all_for_pid() == 0


# --- fail-open ---------------------------------------------------------------

def test_store_failure_fails_open(monkeypatch, tmp_path):
    p = str(tmp_path / "j.txt")
    monkeypatch.setattr(db, "_open", lambda: None)
    assert fc.claim_file(p, "me") == (True, None)
    assert fc.release_for("me") == 0


def test_missing_pid_start_time_column_fails_open(monkeypatch, tmp_path):
    """A store still on the pre-0.119.0 shape (the migration didn't take, e.g. a
    contended store) must not raise into a write tool: the SELECT fails and
    claim_file fails OPEN."""
    p = str(tmp_path / "j2.txt")
    real_open = db._open

    def _legacy_open():
        conn = real_open()
        assert conn is not None
        conn.execute("DROP TABLE IF EXISTS file_claims")
        conn.execute(  # rebuild without pid_start_time
            "CREATE TABLE file_claims(path TEXT PRIMARY KEY,"
            " owner_id TEXT NOT NULL, pid INTEGER NOT NULL, claimed_at REAL,"
            " cwd TEXT)")
        return conn

    monkeypatch.setattr(db, "_open", _legacy_open)
    assert fc.claim_file(p, "me") == (True, None)


# --- write-tool wiring -------------------------------------------------------

def _yolo(monkeypatch):
    monkeypatch.setattr(nx, "YOLO", True)
    monkeypatch.setattr(nx._bash_jobs, "_confirm", lambda *a, **k: True)
    monkeypatch.setattr(nx._edit_tools, "_confirm", lambda *a, **k: True)


def test_write_tools_in_one_process_reclaim_cleanly(tmp_path, monkeypatch):
    _yolo(monkeypatch)
    f = tmp_path / "k.txt"
    f.write_text("v1\n")
    assert "wrote" in nx._edit_tools.write_file(str(f), "v2\n")
    # second write tool in the SAME process/thread: same owner -> re-claim ok
    assert "wrote" in nx._edit_tools.write_file(str(f), "v3\n")
    assert f.read_text() == "v3\n"


def test_foreign_holder_refuses_write(tmp_path, monkeypatch):
    _yolo(monkeypatch)
    f = tmp_path / "l.txt"
    f.write_text("v1\n")
    _insert(str(f), "someone-else", os.getpid(), time.time())
    res = nx._edit_tools.write_file(str(f), "clobbered\n")
    assert res.startswith("ERROR:") and "claimed by another agent" in res
    assert "someone-else" in res
    assert f.read_text() == "v1\n"  # untouched


# --- the refusal message names the real remedies (0.119.0) -------------------

def test_refusal_message_names_the_ttl(tmp_path):
    """There is no steal/force override in the protocol, so the message must not
    imply one: it names the TTL auto-expiry (the actual remedy) and the
    dead-holder reclaim."""
    f = str(tmp_path / "ttl.txt")
    _insert(f, "someone-else", os.getpid(), time.time())
    msg = nx._edit_tools._claim_error(f)
    assert msg is not None
    assert "3600s" in msg and "NX_FILE_CLAIM_TTL" in msg
    assert "heartbeat" in msg          # why it expires
    assert "reclaimed" in msg          # the dead-holder remedy
    assert "coordinate with the holder" not in msg  # the invented remedy is gone


def test_refusal_message_names_a_non_default_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_FILE_CLAIM_TTL", "120")
    f = str(tmp_path / "ttl2.txt")
    _insert(f, "someone-else", os.getpid(), time.time())
    msg = nx._edit_tools._claim_error(f)
    assert msg is not None and "120s" in msg


def test_refusal_message_when_claims_never_expire(tmp_path, monkeypatch):
    """NX_FILE_CLAIM_TTL=0: the claim never expires by age, and the message must
    say so instead of quoting a TTL that will not fire."""
    monkeypatch.setenv("NX_FILE_CLAIM_TTL", "0")
    f = str(tmp_path / "ttl3.txt")
    _insert(f, "someone-else", os.getpid(), time.time())
    msg = nx._edit_tools._claim_error(f)
    assert msg is not None
    assert "do not expire" in msg and "expires after" not in msg


def test_write_refusal_carries_the_ttl(tmp_path, monkeypatch):
    """End to end: the string a write tool returns to the model names the TTL,
    so the model can report a real remedy rather than improvising one."""
    _yolo(monkeypatch)
    f = tmp_path / "ttl4.txt"
    f.write_text("v1\n")
    _insert(str(f), "someone-else", os.getpid(), time.time())
    res = nx._edit_tools.write_file(str(f), "v2\n")
    assert "3600s" in res and "NX_FILE_CLAIM_TTL" in res
    assert f.read_text() == "v1\n"


# --- release on session exit -------------------------------------------------

def test_repl_exit_releases_this_pids_claims(tmp_path, monkeypatch):
    """The REPL's clean-exit path (EOF/Ctrl-D) must drop this process's claims
    next to _sessionlock.release(), so a quit doesn't block this repo until the
    TTL. Drives the real _repl() loop to EOF with the model path stubbed out."""
    import builtins
    repl = sys.modules["nx._repl"]
    f = str(tmp_path / "exit.txt")
    assert fc.claim_file(f, "me")[0]
    monkeypatch.setenv("NX_SESSION_LOCK", "0")
    monkeypatch.setattr(nx._state, "ACTIVE", None)
    monkeypatch.setattr(nx, "_banner", lambda: "")
    monkeypatch.setattr(repl, "_initial_messages",
                        lambda: [{"role": "system", "content": "s"}])
    monkeypatch.setattr(repl, "_load_input_history", lambda: [])
    monkeypatch.setattr(repl, "_ensure_ctx_probe", lambda *a, **k: None)
    monkeypatch.setattr(repl, "_session_id_from_args", lambda: "sess-exit")
    monkeypatch.setattr(repl, "_flush_save", lambda: None)
    monkeypatch.setattr(repl, "_save_current", lambda *a, **k: None)
    monkeypatch.setattr(repl, "begin_exit", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(sys, "argv", ["nx"])

    def _eof(*a, **k):
        raise EOFError

    monkeypatch.setattr(nx, "read_multiline", _eof)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    nx._repl()
    assert _rows() == []  # the exit sweep removed this pid's claim


def test_quit_path_releases_this_pids_claims(tmp_path, monkeypatch):
    """`/quit`/`/exit` is the OTHER clean exit (the handler returns "break")
    and must release too, or quitting that way still leaves rows blocking this
    repo until the TTL."""
    import builtins
    repl = sys.modules["nx._repl"]
    f = str(tmp_path / "quit.txt")
    assert fc.claim_file(f, "me")[0]
    monkeypatch.setenv("NX_SESSION_LOCK", "0")
    monkeypatch.setattr(nx._state, "ACTIVE", None)
    monkeypatch.setattr(nx, "_banner", lambda: "")
    monkeypatch.setattr(repl, "_initial_messages",
                        lambda: [{"role": "system", "content": "s"}])
    monkeypatch.setattr(repl, "_load_input_history", lambda: [])
    monkeypatch.setattr(repl, "_ensure_ctx_probe", lambda *a, **k: None)
    monkeypatch.setattr(repl, "_session_id_from_args", lambda: "sess-quit")
    monkeypatch.setattr(sys, "argv", ["nx"])
    monkeypatch.setattr(nx, "read_multiline", lambda *a, **k: "/quit")
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    nx._repl()
    assert _rows() == []  # the /quit exit released this pid's claim


def test_repl_exit_claim_release_never_raises(tmp_path, monkeypatch):
    """A store that can't serve the sweep must not break the exit path."""
    import builtins
    repl = sys.modules["nx._repl"]
    monkeypatch.setenv("NX_SESSION_LOCK", "0")
    monkeypatch.setattr(nx._state, "ACTIVE", None)
    monkeypatch.setattr(nx, "_banner", lambda: "")
    monkeypatch.setattr(repl, "_initial_messages",
                        lambda: [{"role": "system", "content": "s"}])
    monkeypatch.setattr(repl, "_load_input_history", lambda: [])
    monkeypatch.setattr(repl, "_ensure_ctx_probe", lambda *a, **k: None)
    monkeypatch.setattr(repl, "_session_id_from_args", lambda: "sess-exit2")
    monkeypatch.setattr(repl, "_flush_save", lambda: None)
    monkeypatch.setattr(repl, "_save_current", lambda *a, **k: None)
    monkeypatch.setattr(repl, "begin_exit", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(sys, "argv", ["nx"])
    monkeypatch.setattr(nx, "read_multiline",
                        lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)

    def _boom(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(fc, "release_all_for_pid", _boom)
    nx._repl()  # must not propagate

def test_foreign_holder_refuses_edit_and_delete(tmp_path, monkeypatch):
    _yolo(monkeypatch)
    f = tmp_path / "m.txt"
    f.write_text("hello\n")
    _insert(str(f), "someone-else", os.getpid(), time.time())
    assert "claimed by another agent" in nx._edit_tools.edit_file(str(f), "hello", "bye")
    assert "claimed by another agent" in nx._edit_tools.delete_file(str(f))
    assert f.read_text() == "hello\n"


def test_sed_edit_refuses_only_the_claimed_file(tmp_path, monkeypatch):
    _yolo(monkeypatch)
    a, b = tmp_path / "n.txt", tmp_path / "o.txt"
    a.write_text("foo\n")
    b.write_text("foo\n")
    _insert(str(a), "someone-else", os.getpid(), time.time())
    res = nx._edit_tools.sed_edit("foo", "bar", str(tmp_path / "*.txt"))
    assert "edited 1 file(s)" in res
    assert b.read_text() == "bar\n"   # unclaimed file still edited
    assert a.read_text() == "foo\n"   # claimed file untouched


def test_sed_edit_all_claimed_returns_refusal(tmp_path, monkeypatch):
    _yolo(monkeypatch)
    a = tmp_path / "p.txt"
    a.write_text("foo\n")
    _insert(str(a), "someone-else", os.getpid(), time.time())
    res = nx._edit_tools.sed_edit("foo", "bar", str(a))
    assert "claimed by another agent" in res
    assert a.read_text() == "foo\n"
