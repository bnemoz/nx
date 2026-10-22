#!/usr/bin/env python3
"""Tests for nx/_sessiondb.py — the 0.101.0 SQLite session store.

Pins: round-trip through the DB, lazy file→DB migration (idempotent,
files kept as backup), revision counting, corrupt-DB set-aside and
self-healing re-import, listing order, delete, prune.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-sessiondb-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _sessiondb as db  # noqa: E402


def _use_test_sessions_dir():
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    return m._sessions_dir()


def setup_function(_function):
    _use_test_sessions_dir()


def _clear():
    d = _use_test_sessions_dir()
    assert os.path.basename(d).startswith("nx-sessiondb-"), d
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    db._MIGRATED.clear()
    return d


def _msg(role, content):
    return {"role": role, "content": content}


def test_store_session_roundtrip_and_revision():
    _clear()
    sid = "20250101-120000-dbaa1"
    assert db.store_session(sid, [_msg("user", "hello")], {"title": "hi"})
    data = db.load_session(sid)
    assert data["title"] == "hi"
    assert data["revision"] == 1
    assert data["messages"] == [_msg("user", "hello")]
    # second write bumps revision and keeps created_at
    db.store_session(sid, [_msg("user", "hello"), _msg("assistant", "yo")])
    data2 = db.load_session(sid)
    assert data2["revision"] == 2
    # 0.115.0: created_at lives only in the (removed) blob — first-save
    # updated_at is the closest surviving creation marker.
    assert data2["updated_at"] >= data["updated_at"]


def test_store_session_merges_stale_writer():
    _clear()
    sid = "20250101-120000-dbaa2"
    base = [_msg("user", "start")]
    db.store_session(sid, base)
    db.store_session(sid, base + [_msg("assistant", "first")])
    # a stale snapshot that never saw "first" must not erase it
    db.store_session(sid, base + [_msg("user", "second")])
    data = db.load_session(sid)
    assert data["messages"] == base + [_msg("assistant", "first"),
                                       _msg("user", "second")]
    assert data["revision"] == 3


def test_migrate_files_imports_legacy_and_is_idempotent():
    _clear()
    d = _sessions_dir = _use_test_sessions_dir()
    legacy = {"20250101-120000-dbb1": [_msg("user", "old one")],
              "20250101-120000-dbb2": [_msg("user", "old two")]}
    for sid, msgs in legacy.items():
        with open(os.path.join(d, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump({"id": sid, "messages": msgs, "updated_at": 100.0,
                       "revision": 4, "title": f"t-{sid[-4:]}"}, f)
    n = db.migrate_files()
    assert n == 2
    # files are left in place (backup, never deleted)
    assert os.path.exists(os.path.join(d, "20250101-120000-dbb1.json"))
    # idempotent: a second pass imports nothing
    assert db.migrate_files(force=True) == 0
    # and the loaded transcripts match
    assert db.load_session("20250101-120000-dbb1")["messages"] == legacy[
        "20250101-120000-dbb1"]
    assert db.load_session("20250101-120000-dbb1")["revision"] == 4


def test_lazy_migration_runs_when_store_empty():
    """The self-heal path: mirror files present + empty store → next
    load/list imports them without an explicit migrate call."""
    _clear()
    d = _use_test_sessions_dir()
    sid = "20250101-120000-dbb5"
    with open(os.path.join(d, f"{sid}.json"), "w", encoding="utf-8") as f:
        json.dump({"id": sid, "messages": [_msg("user", "resurrected")],
                   "updated_at": 5.0, "revision": 1}, f)
    assert m._load_session(sid) is not None
    assert m._load_session(sid)["messages"] == [_msg("user", "resurrected")]


def test_corrupt_store_set_aside_and_recovers():
    _clear()
    sid = "20250101-120000-dbc1"
    db.store_session(sid, [_msg("user", "keep me")], {"title": "t"})
    with open(db._store_path(), "w", encoding="utf-8") as f:
        f.write("not a database")
    assert db.load_session(sid) is None          # degrades, never raises
    assert os.path.exists(db._store_path() + ".corrupt")
    # fresh store accepts a new write
    assert db.store_session("20250101-120000-dbc2", [_msg("user", "fresh")])
    assert db.load_session("20250101-120000-dbc2")["revision"] == 1


def test_corrupt_store_archives_wal_siblings_and_notices(capsys):
    """0.168.4 (audit P1-2): set-aside moves -wal/-shm WITH the main file —
    a fresh DB opening next to a stale WAL re-corrupts — and emits the
    user-visible notice naming the archive and `--recover`."""
    _clear()
    sid = "20250101-120000-dbc3"
    db.store_session(sid, [_msg("user", "wal me")], {"title": "t"})
    # simulate an unclean shutdown: a stale WAL + SHM next to a corrupt header
    with open(db._store_path(), "r+b") as f:
        f.write(b"not a databaseXXXX")
    for suffix in ("-wal", "-shm"):
        with open(db._store_path() + suffix, "w", encoding="utf-8") as f:
            f.write("stale " + suffix)
    db.load_session(sid)                          # triggers the set-aside
    for suffix in ("", "-wal", "-shm"):
        assert os.path.exists(db._store_path() + suffix + ".corrupt"), suffix
    out = capsys.readouterr().out
    assert ".corrupt" in out and "--recover" in out
    # the fresh store (the retry re-created it) is empty but usable
    assert db.load_session(sid) is None
    assert db.store_session("20250101-120000-dbc4", [_msg("user", "fresh")])


def test_recover_roundtrips_corrupt_archive():
    """0.168.4 (audit P1-2 acceptance): `--recover` re-imports readable rows
    from an archive into the live store; the archive survives. (Recovery
    rotates the staged -wal/-shm next to the archive before the read-only
    open, so nothing written next to the live store taints the read.)

    Shape note (probed): SQLite cannot open a main whose page 0 is damaged
    even with its WAL present — the salvageable case is an archive whose
    main is intact (set aside after a transient mislabel) or whose WAL
    page-cache made it readable in place. What this pins is the re-import
    path and idempotence, not sqlite-level resurrection of arbitrary
    header damage."""
    _clear()
    sid = "20250101-120000-dbc5"
    assert db.store_session(sid, [_msg("user", "salvage me")], {"title": "kept"})
    # Build the archive by hand from an INTACT store file (the case where
    # recovery can genuinely help) and re-import it over a fresh store.
    shutil.copy(db._store_path(), db._store_path() + ".corrupt")
    assert db.store_session(sid, [_msg("user", "salvage me"),
                                  _msg("assistant", "second")], {"title": "kept2"})
    # drop the live session, keep the archive at the older snapshot
    from nx._sessiondb import prune_stale  # noqa: F401  (existence check)
    os.remove(db._store_path())
    if os.path.exists(db._store_path() + "-wal"):
        os.remove(db._store_path() + "-wal")
    rc = db.recover_corrupt_archive()
    assert rc["recovered"] == 1 and rc["failed"] == 0, rc
    assert rc["error"] is None
    data = db.load_session(sid)
    assert data["messages"] == [_msg("user", "salvage me")]
    assert data["title"] == "kept"
    assert os.path.exists(db._store_path() + ".corrupt")  # archive survives
    # idempotent-ish: recovering again re-imports the same rows harmlessly
    assert db.recover_corrupt_archive()["recovered"] == 1


def test_recover_unreadable_archive_reports_cleanly():
    """A genuinely destroyed main (truncated to a text stub, WAL junk) is
    not recoverable — recovery reports the failure, never raises, and
    leaves the archive in place."""
    _clear()
    sid = "20250101-120000-dbc7"
    assert db.store_session(sid, [_msg("user", "gone")], {"title": "t"})
    with open(db._store_path(), "w", encoding="utf-8") as f:
        f.write("not a database")
    with open(db._store_path() + "-wal", "w", encoding="utf-8") as f:
        f.write("junk")
    db.load_session(sid)                          # set-aside archives all
    assert os.path.exists(db._store_path() + ".corrupt")
    rc = db.recover_corrupt_archive()
    assert rc["recovered"] == 0 and rc["failed"] == 0
    assert rc["error"] is not None
    assert os.path.exists(db._store_path() + ".corrupt")  # kept in place


def test_set_aside_leaves_no_stale_wal_next_to_fresh_store():
    """The audit's F-2 hazard: a corrupt main's WAL/SHM re-attaching to the
    recreated DB. After the 0.168.4 set-aside, the staged siblings live in
    the archives — the fresh store's first open starts with no WAL (its
    own is created fresh on first write), never the stale bytes."""
    _clear()
    sid = "20250101-120000-dbc9"
    assert db.store_session(sid, [_msg("user", "x")], {"title": "t"})
    sp = db._store_path()
    with open(sp, "w", encoding="utf-8") as f:
        f.write("not a database")
    for suffix in ("-wal", "-shm"):
        with open(sp + suffix, "w", encoding="utf-8") as f:
            f.write("stale " + suffix)
    db.load_session(sid)
    # the fresh main was recreated by the retry; the stale bytes are gone
    # from the live names (either archived or deleted), never re-attached
    assert os.path.exists(sp + "-wal.corrupt")
    assert os.path.exists(sp + "-shm.corrupt")
    wal = open(sp + "-wal", "rb").read() if os.path.exists(sp + "-wal") else b""
    assert wal != b"stale -wal"


def test_recover_without_archive_is_clean():
    _clear()
    rc = db.recover_corrupt_archive()
    assert rc == {"archive": rc["archive"], "exists": False,
                  "recovered": 0, "failed": 0, "error": None}


def test_list_ids_newest_order():
    _clear()
    for i, ts in enumerate((300.0, 100.0, 200.0)):
        db.store_session(f"20250101-12000{i}-ord{i}", [_msg("user", f"s{i})")])
        # rewrite updated_at via a direct row edit to pin pure column order
        conn = sqlite3.connect(db._store_path())
        conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                     (ts, f"20250101-12000{i}-ord{i}"))
        conn.commit()
        conn.close()
    rows = db.list_ids_newest()
    assert [sid for sid, _ in rows] == [
        "20250101-120000-ord0", "20250101-120002-ord2", "20250101-120001-ord1"]


def test_delete_session_row():
    _clear()
    sid = "20250101-120000-dbd1"
    db.store_session(sid, [_msg("user", "bye")])
    assert db.load_session(sid) is not None
    assert db.delete_session_row(sid) is True
    assert db.load_session(sid) is None


def test_refresh_fts_and_prune_stale():
    _clear()
    sid = "20250101-120000-dbe1"
    db.store_session(sid, [_msg("user", "quokka habitat")], {"title": "q"})
    db.refresh_fts(sid, [_msg("user", "quokka habitat")], title="q")
    conn = sqlite3.connect(db._store_path())
    # 0.115.0: per-message index rows + one metadata row
    n = conn.execute("SELECT COUNT(*) FROM fts_msgs").fetchone()[0]
    m = conn.execute("SELECT COUNT(*) FROM fts_meta").fetchone()[0]
    conn.close()
    assert n == 1 and m == 1
    # transcript row gone → the FTS rows are stale and prune removes them
    conn = sqlite3.connect(db._store_path())
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
    conn.commit()
    conn.close()
    assert db.prune_stale() == 1
    conn = sqlite3.connect(db._store_path())
    n = conn.execute("SELECT COUNT(*) FROM fts_msgs").fetchone()[0]
    m = conn.execute("SELECT COUNT(*) FROM fts_meta").fetchone()[0]
    conn.close()
    assert n == 0 and m == 0


def test_store_file_private():
    _clear()
    db.store_session("20250101-120000-dbf1", [_msg("user", "hi")])
    mode = os.stat(db._store_path()).st_mode & 0o777
    assert mode == 0o600, oct(mode)


def test_fts_delete_drops_index_postings():
    """0.116.1 regression: deleting a session must shrink msgs_fts_data too —
    the external-content index rows go BEFORE the content rows, or the
    subselect keys off an empty table and every posting orphans (a full
    session sweep once left a 500 MB index behind 2 sessions of content)."""
    _clear()
    sid = "20250101-120000-ftsd1"
    db.store_session(sid, [_msg("user", "orphan index probe")], {"title": "x"})
    db.refresh_fts(sid, [_msg("user", "orphan index probe")], title="x")
    conn = sqlite3.connect(db._store_path())
    before = conn.execute(
        "SELECT SUM(pgsize) FROM dbstat WHERE name = 'msgs_fts_data'"
    ).fetchone()[0]
    assert before > 0
    assert db.delete_session_row(sid) is True
    after = conn.execute(
        "SELECT COALESCE(SUM(pgsize), 0) FROM dbstat"
        " WHERE name = 'msgs_fts_data'").fetchone()[0]
    n = conn.execute("SELECT COUNT(*) FROM msgs_fts").fetchone()[0]
    # page accounting can keep an empty root page; the postings themselves
    # must be gone — verify via a MATCH, the consumer's view
    hit = conn.execute(
        "SELECT COUNT(*) FROM msgs_fts WHERE msgs_fts MATCH 'probe'"
    ).fetchone()[0]
    conn.close()
    assert n == 0
    assert hit == 0, "no postings may survive the delete"


def test_refresh_fts_rebuild_drops_old_postings():
    _clear()
    sid = "20250101-120000-ftsd2"
    db.store_session(sid, [_msg("user", "long body " * 5000)], {"title": "x"})
    db.refresh_fts(sid, [_msg("user", "long body " * 5000)], title="x")
    # rebuild with a much smaller transcript (compaction) — postings shrink
    db.refresh_fts(sid, [_msg("user", "tiny")], title="x")
    conn = sqlite3.connect(db._store_path())
    n = conn.execute("SELECT COUNT(*) FROM msgs_fts").fetchone()[0]
    orphan = conn.execute(
        "SELECT COUNT(*) FROM msgs_fts WHERE rowid NOT IN"
        " (SELECT rowid FROM fts_msgs)").fetchone()[0]
    conn.close()
    assert n == 1 and orphan == 0


def test_refresh_fts_skips_injected_blocks_but_keeps_stored_transcript():
    """`msgs_fts` (the recall_lessons body) must not index nx's own injected
    blocks: a [Saved memories …] line used to be the FIRST line of an indexed
    row, i.e. the snippet recall_lessons surfaced as a 'past lesson'. Index-only
    exclusion — the stored message list stays byte-identical."""
    _clear()
    sid = "20250101-120000-sat01"
    msgs = [
        _msg("system", "sys"),
        _msg("user", "[Saved memories — use memory_search to read any:]\n"
                     "[project] quokkatitle a lesson title"),
        _msg("user", "[Runtime note: a command you ran just backgrounded zephyrnote]"),
        _msg("user", "a real user line about quokkatitleflange"),
        _msg("assistant", "ack"),
    ]
    db.store_session(sid, msgs, {"title": "sat"})
    db.refresh_fts(sid, msgs, title="sat")
    conn = sqlite3.connect(db._store_path())
    try:
        bodies = [r[0] for r in conn.execute(
            "SELECT body FROM fts_msgs WHERE sid = ? ORDER BY seq", (sid,))]
        # one row per message, an excluded block stored as ''
        assert len(bodies) == len(msgs), bodies
        assert bodies[1] == "" and bodies[2] == "", bodies
        joined = "\n".join(bodies)
        assert "Saved memories" not in joined, joined
        assert "quokkatitle a lesson title" not in joined, joined
        assert "zephyrnote" not in joined, joined
        # The real user line in the same session still hits.
        hit = conn.execute("SELECT COUNT(*) FROM msgs_fts WHERE msgs_fts MATCH ?",
                           ("quokkatitleflange",)).fetchone()[0]
        assert hit == 1, bodies
        miss = conn.execute("SELECT COUNT(*) FROM msgs_fts WHERE msgs_fts MATCH ?",
                            ("quokkatitle",)).fetchone()[0]
        assert miss == 0, bodies
        # Appending a turn after the excluded rows stays INCREMENTAL: the empty
        # body is what keeps `seq` gapless (max_seq == cnt - 1), so this is the
        # regression guard against a full reindex on every indexed save.
        grew = msgs + [_msg("user", "a second real quokkatitleflange turn")]
        db.refresh_fts(sid, grew, title="sat")
        seqs = [r[0] for r in conn.execute(
            "SELECT seq FROM fts_msgs WHERE sid = ? ORDER BY seq", (sid,))]
        assert len(seqs) == len(grew), seqs
    finally:
        conn.close()
    assert db.load_session(sid)["messages"] == msgs


# --- FTS body-schema version backfill (0.158.22) ------------------------------
# Stores indexed before the 0.158.19 satellite exclusion must stop recalling
# "[Saved memories …]"-shaped harness text once the amortized pass (or
# `nx sessions --reindex`) ran. Pins: PRAGMA user_version drives detection
# (0/older → rebuild; current → a bare version check); stale bodies are
# RE-DERIVED through the save path's own builder and satisfy the same
# index-only-exclusion invariants; the pass is batched and advances; the
# marker is written only once the stale-body probe completes empty; the pass
# is idempotent; `--reindex` finishes the whole job in one call.

# A pre-0.158.19 satellite body: exactly what the old _body_of wrote for a
# prefix-note message — "[user] " + the raw content, indexed whole.
STALE_SAT_BODY = ("[user] [Saved memories — use memory_search to read any:]"
                  " [project] stalelessonzephyr a stale lesson title")
FB_REAL_BODY = "[user] a real user line about quokkatitleflange"
FB_SYS_BODY = "[system] sys"  # system messages are ordinary index content


def _seed_pre_filter_store(sid, n_real=2):
    """A store as a pre-0.158.19 nx would have left it: the transcript is
    current, but the fts_msgs/msgs_fts rows were hand-built WITHOUT the
    satellite exclusion (the satellite row first, exactly like the bug)."""
    msgs = [_msg("system", "sys"),
            _msg("user", "[Saved memories — use memory_search to read any:]\n"
                         "[project] stalelessonzephyr a stale lesson title"),
            _msg("user", "a real user line about quokkatitleflange")]
    for i in range(n_real):
        msgs.append(_msg("assistant", f"ack {i} quokkatitleflange"))
    db.store_session(sid, msgs, {"title": f"t-{sid[-4:]}"})
    conn = sqlite3.connect(db._store_path())
    with conn:
        conn.execute("DELETE FROM msgs_fts WHERE rowid IN"
                     " (SELECT rowid FROM fts_msgs WHERE sid = ?)", (sid,))
        conn.execute("DELETE FROM fts_msgs WHERE sid = ?", (sid,))
        conn.execute("INSERT INTO fts_meta(sid, title, description, mtime)"
                     " VALUES (?,?,?,?)", (sid, "t", "d", 1.0))
        # the meta postings too: an external-content index DELETE of a
        # rowid with no docsize entry raises "malformed", and a real
        # pre-0.158.19 store always had the pair written together
        conn.execute("INSERT INTO meta_fts(rowid, title, description)"
                     " SELECT rowid, title, description FROM fts_meta"
                     " WHERE sid = ?", (sid,))
        rows = [FB_SYS_BODY, STALE_SAT_BODY, FB_REAL_BODY] + [
            f"[assistant] ack {i} quokkatitleflange" for i in range(n_real)]
        for seq, body in enumerate(rows):
            cur = conn.execute(
                "INSERT INTO fts_msgs(sid, seq, role, body, mtime)"
                " VALUES (?,?,?,?,?)",
                (sid, seq, body.split("]")[0][1:], body, 1.0))
            if body:
                conn.execute(
                    "INSERT INTO msgs_fts(rowid, body) VALUES (?,?)",
                    (cur.lastrowid, body))
    conn.close()
    return msgs


def test_marker_absent_triggers_rebuild_and_lessons_stops_recalling():
    _clear()
    sid = "20250101-120000-fb001"
    _seed_pre_filter_store(sid)
    conn = sqlite3.connect(db._store_path())
    assert db._fts_body_version(conn) == 0
    conn.close()
    # BEFORE: the stale satellite line is recallable as a "past lesson".
    from nx import _lessons
    hits = _lessons.search_lessons("stalelessonzephyr")
    assert [h["id"] for h in hits] == [sid], hits
    # The amortized pass runs the whole (small) job and stamps the marker.
    assert db.fts_body_backfill_batch(batch=10) == 1
    conn = sqlite3.connect(db._store_path())
    assert db._fts_body_version(conn) == db.FTS_BODY_SCHEMA_VERSION
    bodies = [r[0] for r in conn.execute(
        "SELECT body FROM fts_msgs WHERE sid = ? ORDER BY seq", (sid,))]
    conn.close()
    # AFTER: satellite text is gone from the index; the real line survives.
    assert "stalelessonzephyr" not in "\n".join(bodies), bodies
    assert _lessons.search_lessons("stalelessonzephyr") == []
    hits = _lessons.search_lessons("quokkatitleflange")
    assert [h["id"] for h in hits] == [sid], hits


def test_backfill_preserves_index_only_exclusion_invariants():
    """A backfilled store satisfies exactly what a fresh indexed save
    writes: one row per stored message, gapless seq, '' body for the
    satellite row, no msgs_fts postings for empty bodies."""
    _clear()
    sid = "20250101-120000-fb002"
    msgs = _seed_pre_filter_store(sid, n_real=2)
    assert db.fts_body_backfill_batch(batch=10) == 1
    conn = sqlite3.connect(db._store_path())
    rows = conn.execute(
        "SELECT seq, body FROM fts_msgs WHERE sid = ? ORDER BY seq",
        (sid,)).fetchall()
    assert [s for s, _ in rows] == list(range(len(msgs))), rows
    bodies = [b for _, b in rows]
    assert bodies[0] == FB_SYS_BODY, bodies  # system: ordinary content
    assert bodies[1] == "", bodies  # satellite: '' row, seq preserved
    assert FB_REAL_BODY in bodies, bodies
    max_seq, cnt = conn.execute(
        "SELECT COALESCE(MAX(seq),-1), COUNT(*) FROM fts_msgs"
        " WHERE sid = ?", (sid,)).fetchone()
    assert max_seq == cnt - 1  # the gapless invariant the append path reads
    # Postings must carry no satellite tokens: MATCH is the consumer's view
    # (a plain rowid query on external-content FTS just reads fts_msgs).
    hit = conn.execute("SELECT COUNT(*) FROM msgs_fts WHERE msgs_fts MATCH ?",
                       ("stalelessonzephyr",)).fetchone()[0]
    assert hit == 0
    conn.close()
    # The stored transcript is untouched (index-only exclusion).
    assert db.load_session(sid)["messages"] == msgs


def test_batched_progress_advances_and_marker_lands_only_when_done():
    _clear()
    for i in range(3):
        _seed_pre_filter_store(f"20250101-12000{i}-fb00i", n_real=0)
    conn = sqlite3.connect(db._store_path())
    # one session per call: progress, no marker while the probe window still
    # names another stale session; the last named session is rebuilt AND the
    # marker lands in the same call (the probe proved nothing else remains).
    assert db._fts_body_backfill_conn(conn, 1) == 1
    assert db._fts_body_version(conn) == 0
    assert db._fts_body_backfill_conn(conn, 1) == 1
    assert db._fts_body_version(conn) == 0
    assert db._fts_body_backfill_conn(conn, 1) == 1
    assert db._fts_body_version(conn) == db.FTS_BODY_SCHEMA_VERSION
    assert db._fts_body_backfill_conn(conn, 1) == 0
    conn.close()


def test_backfill_idempotent_second_run_rewrites_nothing():
    _clear()
    sid = "20250101-120000-fb003"
    _seed_pre_filter_store(sid)
    assert db.fts_body_backfill_batch(batch=10) == 1
    conn = sqlite3.connect(db._store_path())
    before = conn.execute(
        "SELECT seq, body, mtime FROM fts_msgs WHERE sid = ? ORDER BY seq",
        (sid,)).fetchall()
    conn.close()
    # further passes (every entry point) are no-ops…
    assert db.fts_body_backfill_batch(batch=10) == 0
    assert db.prune_stale() == 0
    assert db.fts_body_reindex_all() == {
        "sessions_reindexed": 0, "version": db.FTS_BODY_SCHEMA_VERSION}
    conn = sqlite3.connect(db._store_path())
    after = conn.execute(
        "SELECT seq, body, mtime FROM fts_msgs WHERE sid = ? ORDER BY seq",
        (sid,)).fetchall()
    conn.close()
    # …that touch no row: identical mtime proves nothing was rewritten.
    assert after == before


def test_current_version_pays_only_the_version_check(monkeypatch):
    _clear()
    sid = "20250101-120000-fb004"
    db.store_session(sid, [_msg("user", "plain body")], {"title": "p"})
    db.refresh_fts(sid, [_msg("user", "plain body")], title="p")
    conn = sqlite3.connect(db._store_path())
    db._fts_body_version_set(conn, db.FTS_BODY_SCHEMA_VERSION)
    conn.commit()
    conn.close()
    calls = {"n": 0}
    real = db._stale_body_sids

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(db, "_stale_body_sids", counting)
    assert db.fts_body_backfill_batch(batch=10) == 0
    assert db.prune_stale() == 0
    assert calls["n"] == 0, "a current store must not pay the probe scan"


def test_empty_store_gets_marked_and_later_saves_stay_current():
    """No transcripts + version 0 → the probe answers "clean", the marker
    lands, and the next save indexes current (no endless re-check)."""
    _clear()
    assert db.fts_body_backfill_batch(batch=10) == 0
    conn = sqlite3.connect(db._store_path())
    assert db._fts_body_version(conn) == db.FTS_BODY_SCHEMA_VERSION
    conn.close()
    sid = "20250101-120000-fb005"
    db.store_session(sid, [_msg("user", "first indexed save")], {"title": "n"})
    db.refresh_fts(sid, [_msg("user", "first indexed save")], title="n")
    assert db.fts_body_backfill_batch(batch=10) == 0


def test_reindex_cli_does_the_whole_job_in_one_call(capsys):
    _clear()
    for i in range(4):
        _seed_pre_filter_store(f"20250101-12000{i}-fb00c", n_real=0)
    from nx import _sessions_cmd
    # the listing entry routes `nx sessions --reindex` to the handler before
    # the flag can be swallowed as a search query
    assert _sessions_cmd._cli_sessions(["sessions", "--reindex"]) is True
    out = capsys.readouterr().out
    assert "reindexed 4 session(s)" in out, out
    conn = sqlite3.connect(db._store_path())
    assert db._fts_body_version(conn) == db.FTS_BODY_SCHEMA_VERSION
    assert db._stale_body_sids(conn, 100)[0] == []
    conn.close()
    assert _sessions_cmd._cli_reindex_sessions(["--reindex"]) is True
    assert "nothing to reindex" in capsys.readouterr().out


def test_bare_reindex_flag_routes_through_main(monkeypatch):
    """`nx --reindex` reaches the handler via main() instead of falling
    through to the REPL — the placement trap --prune documents (a later verb
    could swallow the flag as a query). Handler faked; the assertion is about
    routing order, not the backfill (pinned end-to-end by the two tests
    above)."""
    import sys

    from nx import _cli
    calls = {}

    def fake(args):
        calls["args"] = list(args)
        return True
    monkeypatch.setattr(_cli, "_cli_reindex_sessions", fake)
    monkeypatch.setattr(_cli, "_repl",
                        lambda *a, **k: pytest.fail("REPL launched: the "
                                                   "--reindex flag fell "
                                                   "through routing"))
    monkeypatch.setattr(_cli, "_load_env_file", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(sys, "argv", ["nx", "--reindex"])
    _cli.main()  # returns (does not raise): the handler returned True
    assert calls.get("args") == ["--reindex"]


def test_reindex_leaves_a_current_store_alone():
    _clear()
    sid = "20250101-120000-fb00d"
    db.store_session(sid, [_msg("user", "clean save body")], {"title": "c"})
    db.refresh_fts(sid, [_msg("user", "clean save body")], title="c")
    conn = sqlite3.connect(db._store_path())
    before = conn.execute(
        "SELECT mtime FROM fts_msgs WHERE sid = ?", (sid,)).fetchone()
    conn.close()
    # a store whose index is already built by the current builder, but whose
    # marker was never stamped: one clean probe → marker, no rewrites.
    assert db.fts_body_reindex_all()["sessions_reindexed"] == 0
    conn = sqlite3.connect(db._store_path())
    after = conn.execute(
        "SELECT mtime FROM fts_msgs WHERE sid = ?", (sid,)).fetchone()
    assert db._fts_body_version(conn) == db.FTS_BODY_SCHEMA_VERSION
    conn.close()
    assert after == before  # rows untouched


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
