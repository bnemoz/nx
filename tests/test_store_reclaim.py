#!/usr/bin/env python3
"""Store-residue reclamation (0.115.0 events-authoritative migration).

The 0.115.0 migration was forward-only and left two residue classes behind in
every upgraded store:

1. the dead 0.104-era FTS pair `sessions_fts` + `fts_body` — dropped only when
   `msgs_fts` was absent/stale, so a store upgraded while `msgs_fts` was
   already current kept the legacy index forever (`_ensure_fts_schema`'s
   early-return branch, fixed by step A);
2. stale full-transcript `sessions.data` blobs — pre-0.115 rows still hold the
   whole transcript even though `session_events` holds the same text (step B,
   `_reclaim_stale_blobs`, run once per process from `_ensure_migrated`).

What these tests pin:
- the legacy pair is gone after any open, in BOTH shapes (msgs_fts already
  current, and a genuine pre-0.115 store that gets rebuilt), and the live
  fts_msgs/fts_meta/fts_meta trio is untouched;
- a big blob is blanked exactly when (events exist) AND (image_shas is NOT
  NULL) AND (length > 1MB) — a NULL-image_shas row's blob is the only
  image-ref source for the sidecar-GC fallback, and a row with no events IS
  the transcript (pre-0.104 inflation);
- load_session still returns the full transcript from the event log after a
  reclaim, and the image live-set still resolves the non-NULL path from the
  indexed column;
- `reclaim_store()` reports counts, actually VACUUMs (store + WAL shrink),
  and `nx sessions --prune --reclaim` invokes it.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="nx-store-reclaim-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _sessiondb as db  # noqa: E402
from nx import _sessions_cmd as cmd  # noqa: E402

# Tier B, explicitly: this module writes NX_* env at import scope and mutates
# the `nx._sessiondb`/`nx._sessions_fs` globals under test (see _nxtest's
# classification rules), so it opts into conftest's module-dict rewind.
pytestmark = pytest.mark.globals


def _use_test_sessions_dir():
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    return m._sessions_dir()


def setup_function(_function):
    _use_test_sessions_dir()


def teardown_function(_function):
    _clear()


def _clear():
    d = _use_test_sessions_dir()
    assert os.path.basename(d).startswith("nx-store-reclaim-"), d
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    # both process-level gates are keyed by sessions dir and survive the
    # store file being deleted (they are "once per process" by design)
    db._MIGRATED.clear()
    db._RECLAIMED.clear()
    return d


def _conn():
    return sqlite3.connect(db._store_path())


def _exec(sql, params=()):
    c = _conn()
    try:
        cur = c.execute(sql, params)
        c.commit()
        return cur
    finally:
        c.close()


def _one(sql, params=()):
    c = _conn()
    try:
        row = c.execute(sql, params).fetchone()
        return row[0] if row else None
    finally:
        c.close()


def _tables():
    c = _conn()
    try:
        return {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        c.close()


def _msg(role, content):
    return {"role": role, "content": content}


def _ref(sha):
    return {"type": "image_ref", "sha256": sha, "mime": "image/png", "bytes": 10}


def _big_text(n=1_200_000):
    return "x" * n


def _bootstrap():
    """Create the store file + the current schema, exactly as any real
    facade call would (db._open runs _ensure_schema/_ensure_fts_schema)."""
    conn = db._open()
    if conn is not None:
        conn.close()


def _reclaim():
    """Run step B on its own committed connection (the production caller
    always commits; an uncommitted write here would hold the store's write
    lock and make every later _open() see "database is locked")."""
    c = _conn()
    try:
        n = db._reclaim_stale_blobs(c)
        c.commit()
        return n
    finally:
        c.close()


def _legacy_row(sid, *, shas='"aaaaaaaaaaaa"', events=2,
                content=None, n=1_200_000):
    """A pre-0.115-shaped row: a full-transcript blob (over the 1MB floor by
    default) plus an optional event log, exactly the residue the migration
    left behind. `shas=None` writes a NULL image_shas (the deliberate
    exception); `events=0` writes no events (the pre-0.104 inflation source)."""
    _bootstrap()
    payload = content if content is not None else [{"role": "user",
                                                    "content": _big_text(n)}]
    blob = json.dumps({"id": sid, "messages": payload, "updated_at": 1.0,
                       "revision": 1}, ensure_ascii=False)
    _exec("INSERT OR REPLACE INTO sessions"
          " (session_id, data, updated_at, revision, image_shas)"
          " VALUES (?,?,?,?,?)", (sid, blob, 1.0, 1, shas))
    for i in range(events):
        _exec("INSERT INTO session_events (session_id, seq, msg)"
              " VALUES (?,?,?)",
              (sid, i, json.dumps(_msg("user", f"event {i}"))))
    return blob


def _make_legacy_fts():
    """Hand-create the 0.104-era external-content pair. Skips when the
    interpreter has no FTS5 (the reclaim is covered either way)."""
    try:
        _exec("CREATE TABLE fts_body(body TEXT, title TEXT)")
        _exec("CREATE VIRTUAL TABLE sessions_fts USING fts5("
              "session_id UNINDEXED, title, description, body, mtime,"
              " content='fts_body')")
    except sqlite3.OperationalError as e:
        pytest.skip(f"no FTS5 in this interpreter: {e}")


# --- A. legacy FTS pair ------------------------------------------------------

def test_legacy_fts_pair_dropped_when_msgs_fts_is_current():
    """The exact residue: msgs_fts is current, so _ensure_fts_schema used to
    early-return and leave the dead pair behind forever."""
    _clear()
    db.store_session("20250101-120000-rcl01", [_msg("user", "hi")],
                     {"title": "t"})
    assert db._fts_is_current(_conn())
    _make_legacy_fts()
    assert {"sessions_fts", "fts_body"} <= _tables()

    conn = db._open()
    assert conn is not None
    conn.close()  # _ensure_schema / _ensure_fts_schema ran

    names = _tables()
    assert "sessions_fts" not in names
    assert "fts_body" not in names
    assert db._fts_is_current(_conn())
    # the live trio is untouched
    assert {"msgs_fts", "fts_msgs", "meta_fts", "fts_meta"} <= names


def test_pre_0115_store_drops_legacy_pair_and_rebuilds_the_trio():
    """msgs_fts absent (a genuine pre-0.115 store): the existing migration
    branch drops the legacy pair and creates the per-message trio. The
    residue is gone here too — this pins which branch owns which outcome."""
    _clear()
    conn = db._open()  # creates the current schema, incl. msgs_fts
    assert conn is not None
    conn.close()
    _make_legacy_fts()
    # simulate the pre-0.115 shape: no current index to early-return on
    _exec("DROP TABLE msgs_fts")
    _exec("DROP TABLE meta_fts")

    conn = db._open()
    assert conn is not None
    conn.close()

    names = _tables()
    assert "sessions_fts" not in names
    assert "fts_body" not in names
    assert db._fts_is_current(_conn())


def test_legacy_pair_survives_no_reader_it_is_simply_dropped_once():
    """Idempotent: a second open on a clean store is a no-op (no error, and
    the trio still works for a fresh search-index write)."""
    _clear()
    db.store_session("20250101-120000-rcl02", [_msg("user", "hi")])
    for _ in range(2):
        conn = db._open()
        assert conn is not None
        conn.close()
    assert db._fts_is_current(_conn())
    db.refresh_fts("20250101-120000-rcl02", [_msg("user", "hi")],
                   title="t", description="d")
    assert _one("SELECT COUNT(*) FROM fts_msgs") == 1


# --- B. stale transcript blobs ----------------------------------------------

def test_stale_blob_reclaimed_through_ensure_migrated():
    """The wired path: `_ensure_migrated` (first facade call of the process)
    blanks a big blob whose row has events + a non-NULL image_shas."""
    _clear()
    sid = "20250101-120000-rcl03"
    _legacy_row(sid)
    assert _one("SELECT length(data) FROM sessions WHERE session_id = ?",
                (sid,)) > 1_048_576

    db._ensure_migrated()

    assert _one("SELECT data FROM sessions WHERE session_id = ?", (sid,)) == "{}"


def test_reclaim_skips_null_image_shas_rows():
    """A NULL image_shas row's blob is the ONLY image-ref source for the
    sidecar-GC fallback (image_live_shas), so its transcript must survive."""
    _clear()
    sid = "20250101-120000-rcl04"
    _legacy_row(sid, shas=None)
    n = _reclaim()
    assert n == 0
    assert _one("SELECT length(data) FROM sessions WHERE session_id = ?",
                (sid,)) > 1_048_576


def test_reclaim_skips_rows_without_events():
    """Pre-0.104: no events, so the blob IS the transcript until
    load_session inflates it — emptying it would destroy the session."""
    _clear()
    sid = "20250101-120000-rcl05"
    _legacy_row(sid, events=0)
    assert _reclaim() == 0
    loaded = db.load_session(sid)
    assert loaded["messages"] == [{"role": "user", "content": _big_text()}]


def test_reclaim_never_touches_small_shells():
    """A 0.115+ small-meta shell is the contract, not residue — under the 1MB
    floor nothing is rewritten even with events + shas present."""
    _clear()
    sid = "20250101-120000-rcl06"
    _legacy_row(sid, n=1000)
    before = _one("SELECT data FROM sessions WHERE session_id = ?", (sid,))
    assert _reclaim() == 0
    assert _one("SELECT data FROM sessions WHERE session_id = ?", (sid,)) == before


def test_reclaim_counts_only_the_covered_rows():
    """One pass over a mixed store: exactly the eligible rows are blanked."""
    _clear()
    covered, null_shas, no_events = ("20250101-120000-rcl07",
                                     "20250101-120000-rcl08",
                                     "20250101-120000-rcl09")
    _legacy_row(covered)
    _legacy_row(null_shas, shas=None)
    _legacy_row(no_events, events=0)
    assert _reclaim() == 1
    assert _one("SELECT data FROM sessions WHERE session_id = ?", (covered,)) == "{}"
    assert _one("SELECT data FROM sessions WHERE session_id = ?",
                (null_shas,)) != "{}"
    assert _one("SELECT data FROM sessions WHERE session_id = ?",
                (no_events,)) != "{}"


def test_load_session_after_reclaim_returns_full_messages_from_events():
    _clear()
    sid = "20250101-120000-rcl10"
    msgs = [_msg("user", _big_text()), _msg("assistant", "hi there")]
    db.store_session(sid, msgs, {"title": "kept"})
    # make it look like un-reclaimed residue: same messages in the blob
    _exec("UPDATE sessions SET data = ? WHERE session_id = ?",
          (json.dumps({"id": sid, "messages": msgs,
                       "updated_at": 1.0}), sid))
    _exec("UPDATE sessions SET image_shas = ? WHERE session_id = ?",
          ('["bbbbbbbbbbbb"]', sid))
    assert _reclaim() == 1

    loaded = db.load_session(sid)
    assert loaded["messages"] == msgs
    assert loaded["title"] == "kept"
    assert loaded["id"] == sid


def test_image_live_set_reads_the_column_for_a_reclaimed_row():
    """Non-NULL path: after the blob is blanked the live-set still reports the
    session's refs from the indexed image_shas column, and the session stays
    'covered' (no sidecar is collected)."""
    _clear()
    sid = "20250101-120000-rcl11"
    sha = "cafe" * 16
    # a big transcript blob that also carries the ref part
    _legacy_row(sid, shas=json.dumps([sha]), n=1_200_000)
    _exec("UPDATE sessions SET data = ? WHERE session_id = ?",
          (json.dumps({"id": sid, "updated_at": 1.0,
                       "messages": [{"role": "user", "content": _big_text()},
                                    {"role": "user", "content": [_ref(sha)]}]}),
           sid))
    assert _reclaim() == 1

    live, covered = db.image_live_shas()
    assert sha in live
    assert sid in covered


def test_image_live_set_falls_back_to_a_null_shas_blob():
    """The reason NULL-image_shas rows are exempt: their blob is the ref
    source, so the live-set must still surface their shas."""
    _clear()
    sid = "20250101-120000-rcl12"
    sha = "beef" * 16
    _legacy_row(sid, shas=None, content=[{"role": "user", "content": _big_text()},
                                         {"role": "user", "content": [_ref(sha)]}])
    assert _reclaim() == 0

    live, covered = db.image_live_shas()
    assert sha in live
    assert sid in covered


# --- C. reclaim_store --------------------------------------------------------

def _store_bytes():
    p = db._store_path()
    return sum(os.path.getsize(q) for q in
               (p, p + "-wal", p + "-shm") if os.path.exists(q))


def test_reclaim_store_reports_counts_and_vacuums():
    _clear()
    sid = "20250101-120000-rcl13"
    _legacy_row(sid)
    _make_legacy_fts()
    before = _store_bytes()
    assert before > 1_048_576  # the residue is really on disk

    out = db.reclaim_store()

    assert out["blobs_emptied"] == 1
    assert out["fts_dropped"] is True
    names = _tables()
    assert "sessions_fts" not in names and "fts_body" not in names
    assert _one("SELECT data FROM sessions WHERE session_id = ?", (sid,)) == "{}"
    # VACUUM + wal_checkpoint(TRUNCATE): the space is handed back, not just
    # logically freed
    assert _store_bytes() < before


def test_reclaim_store_is_idempotent_and_clean_is_a_noop():
    _clear()
    db.store_session("20250101-120000-rcl14", [_msg("user", "hi")])
    first = db.reclaim_store()
    second = db.reclaim_store()
    assert first == {"blobs_emptied": 0, "fts_dropped": False}
    assert second == {"blobs_emptied": 0, "fts_dropped": False}


def test_plain_prune_does_not_vacuum(monkeypatch):
    """The lock cost is opt-in: plain `--prune` must never call reclaim."""
    calls = []
    monkeypatch.setattr("nx._sessiondb.reclaim_store",
                        lambda: calls.append(1) or {
                            "blobs_emptied": 0, "fts_dropped": False})
    assert cmd._cli_prune_sessions(["sessions", "--prune"]) is True
    assert calls == []


# --- D. the CLI flag ---------------------------------------------------------

def test_prune_reclaim_flag_parses_and_invokes_reclaim_store(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("nx._sessiondb.reclaim_store",
                        lambda: calls.append(1) or {"blobs_emptied": 3,
                                                    "fts_dropped": True})
    monkeypatch.setattr("nx._sessions_fs._prune_sessions_store", lambda: {
        "rows_pruned": 0, "shards_dropped": 0, "images_removed": 0,
        "images_kept": 0, "quota_removed": 0, "undo_removed": 0,
        "expired_small": 0, "expired_floor": 0})

    assert cmd._cli_prune_sessions(["sessions", "--prune", "--reclaim"]) is True
    assert calls == [1]
    out = capsys.readouterr().out
    assert "reclaimed 3 stale transcript blob(s)" in out
    assert "sessions_fts/fts_body" in out
    # bare-verb spelling reaches the same path
    calls.clear()
    assert cmd._cli_prune_sessions(["--prune", "--reclaim"]) is True
    assert calls == [1]


def test_prune_reclaim_flag_does_not_hijack_other_invocations():
    assert cmd._cli_prune_sessions(["sessions", "refactor"]) is False
    assert cmd._cli_prune_sessions(["--reclaim"]) is False
    # a stray argument after `sessions prune` is still a typo, not a prune
    assert cmd._cli_prune_sessions(["sessions", "prune", "junk"]) is True
