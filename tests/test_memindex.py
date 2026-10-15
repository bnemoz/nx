#!/usr/bin/env python3
"""2.6 memory retrieval v2: the FTS5 accelerator (`nx/_memindex.py`), the
one-scanner rewire, the `/memory list` scope fix, the no-clobber memory
writer and the optimize oldest-tail selection.

No model/network round-trip — `_confirm` is stubbed and everything touches
temp dirs. The FTS tests skip (never fail) when this sqlite build lacks the
trigram tokenizer, mirroring the production fallback.
"""
import os
import sys
import tempfile

os.environ["NX_HOME"] = tempfile.mkdtemp(prefix="nx-test-memidx-")
os.environ["NX_SESSIONS_DIR"] = os.environ["NX_HOME"]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _memindex  # noqa: E402
from nx import _memory_cmd  # noqa: E402
from nx import _memory_optimize as opt  # noqa: E402
from nx._private import atomic_write_private  # noqa: E402

try:
    import sqlite3 as _sq
    _c = _sq.connect(":memory:")
    _c.execute("CREATE VIRTUAL TABLE _p USING fts5(x, tokenize='trigram')")
    _c.close()
    HAVE_FTS = True
except Exception:  # pragma: no cover - only on odd builds
    HAVE_FTS = False


_saved_confirm = None
_saved_home = None
_saved_root = None
_OUR_HOME = os.environ["NX_HOME"]


def _setup_function():
    global _saved_confirm, _saved_home, _saved_root
    _saved_confirm = m._confirm
    _saved_home = m.st.SESSION_HOME
    _saved_root = m.st.PROJECT_ROOT
    m._confirm = lambda action, preview=None: True
    m.st.SESSION_HOME = _OUR_HOME
    m.st.PROJECT_ROOT = None


def _teardown_function():
    global _saved_confirm, _saved_home, _saved_root
    m._confirm = _saved_confirm
    m.st.SESSION_HOME = _saved_home
    m.st.PROJECT_ROOT = _saved_root
    _saved_confirm = None


setup_function = lambda _fn: _setup_function()  # noqa: E731 (file-local convention)
teardown_function = lambda _fn: _teardown_function()  # noqa: E731


def _note(d, name, body):
    """Write one note directly (full control over name/timestamp)."""
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    atomic_write_private(p, body)
    return p


def _proj_dir():
    root = tempfile.mkdtemp(prefix="nx-memidx-store-")
    d = os.path.join(root, ".nx", "memories")
    m.st.PROJECT_ROOT = root
    return d


# --- the accelerator -----------------------------------------------------------

def test_search_dir_finds_by_body_and_title_terms():
    d = _proj_dir()
    _note(d, "20260101-000001_alpha.md",
          "# Alpha Note\n\nThe API needs X-Header: 1 for auth.\n")
    _note(d, "20260101-000002_beta.md",
          "# Beta Note\n\nsomething about renderer internals\n")
    rows = _memindex.search_dir(d, "X-Header")
    if not HAVE_FTS:
        assert rows is None
        return
    assert rows is not None and [r[0] for r in rows] == ["20260101-000001_alpha.md"], rows
    name, mtime, title, body = rows[0]   # (name, mtime, title, body) — no snippet column
    assert title == "Alpha Note" and "X-Header" in body
    rows2 = _memindex.search_dir(d, "renderer internals")
    assert rows2 is not None and [r[0] for r in rows2] == ["20260101-000002_beta.md"]
    # no hits: the caller falls back (empty list, not None, means "index worked")
    rows3 = _memindex.search_dir(d, "zzz-totally-absent")
    assert rows3 == [], rows3


def test_reconcile_indexes_new_and_prunes_deleted():
    d = _proj_dir()
    p1 = _note(d, "20260101-000001_keep.md", "# Keep\n\nstays indexed\n")
    rows = _memindex.search_dir(d, "stays")
    assert rows and rows[0][0].endswith("keep.md")
    # new note appears without touching the index
    _note(d, "20260101-000002_new.md", "# New\n\nfreshly written fact\n")
    # deleted note vanishes
    os.remove(p1)
    rows2 = _memindex.search_dir(d, "stays")
    assert rows2 == [], rows2
    rows3 = _memindex.search_dir(d, "freshly")
    assert rows3 and rows3[0][0].endswith("new.md")
    # and a content change is picked up (mtime bumped)
    _note(d, "20260101-000002_new.md", "# New\n\nrewritten with uvloop\n")
    rows4 = _memindex.search_dir(d, "uvloop")
    assert rows4 and rows4[0][0].endswith("new.md")


def test_search_dir_none_when_disabled_or_noterms():
    d = _proj_dir()
    _note(d, "20260101-000001_a.md", "# A\n\nbody one\n")
    os.environ["NX_MEMORY_FTS"] = "0"
    try:
        assert _memindex.search_dir(d, "body") is None
        assert _memindex.enabled() is False
    finally:
        os.environ.pop("NX_MEMORY_FTS", None)
    assert _memindex.enabled() is True
    # empty query terms -> caller must use its own scan
    assert _memindex.search_dir(d, "") is None
    assert _memindex.search_dir(d, "   ") is None


def test_short_terms_use_the_plain_scan():
    d = _proj_dir()
    _note(d, "20260101-000001_a.md", "# A\n\nti-ny term\n")
    # "ti" has no trigram; _memory_scan must route it to the plain scan and
    # still find it (never through the accelerator).
    hits, _of, _tot = _memory_cmd._memory_scan("ti", "project", 0)
    assert [h.name for h in hits] == ["20260101-000001_a.md"], hits


def test_fts_off_results_match_fts_on_results():
    """The accelerator must not change the answer: same names, same order,
    same snippets, with and without it (the _rank_hit scoring is shared)."""
    d = _proj_dir()
    _note(d, "20260101-000001_quirk.md",
          "# Quirk about the renderer\n\nunrelated body words here\n")
    _note(d, "20260101-000002_plain.md",
          "# Unrelated note\n\nsomething about renderer internals\n")
    def _results():
        hits, _of, tot = _memory_cmd._memory_scan("renderer", "project", 0)
        return [(h.name, h.title, h.snippet, round(h.score, 3)) for h in hits], tot
    with_fts = _results()
    assert len(with_fts[0]) == 2
    os.environ["NX_MEMORY_FTS"] = "0"
    try:
        without = _results()
    finally:
        os.environ.pop("NX_MEMORY_FTS", None)
    assert with_fts == without, (with_fts, without)
    # The FTS path used to answer with a token-bounded ~12-char snippet() and
    # the plain scan with _memory_snippet's four body lines; parity above now
    # includes the snippet, and a real preview must survive the accelerator.
    assert "renderer" in with_fts[0][1][2].lower(), with_fts[0][1][2]
    # title hit ranks above body hit on both paths
    assert with_fts[0][0][0].endswith("quirk.md")


def test_corrupt_index_is_set_aside_and_rebuilt():
    d = _proj_dir()
    _note(d, "20260101-000001_a.md", "# A\n\nsearchable body\n")
    rows = _memindex.search_dir(d, "searchable")
    if not HAVE_FTS:
        assert rows is None
        return
    assert rows
    db = os.path.join(d, _memindex.INDEX_FILENAME)
    with open(db, "wb") as f:
        f.write(b"this is not a database at all")
    rows2 = _memindex.search_dir(d, "searchable")  # must not raise
    assert rows2 and rows2[0][0].endswith("a.md")
    assert os.path.exists(db + ".corrupt"), os.listdir(d)


def test_memory_search_end_to_end_hits_index_and_reports_dirs():
    d = _proj_dir()
    _note(d, "20260101-000001_m.md", "# Path memo\n\nmarkerxyz body\n")
    out = _memory_cmd.memory_search(query="markerxyz", scope="project")
    if HAVE_FTS:
        # served from the index: dir hint still correct, hit present
        assert "Path memo" in out and d in out, out
        assert os.path.exists(os.path.join(d, _memindex.INDEX_FILENAME))
    # and the write tool keeps the index coherent through its own search path
    res = _memory_cmd.memory_write(title="Second memo", body="secondmemo body",
                                   scope="project")
    assert res.startswith("saved project memory"), res
    out2 = _memory_cmd.memory_search(query="secondmemo", scope="project")
    assert "Second memo" in out2, out2


# --- the one scanner ------------------------------------------------------------

def test_iter_notes_is_the_single_scanner():
    d = _proj_dir()
    _note(d, "20260101-000002_b.md", "# B\n\ntwo\n")
    _note(d, "20260101-000001_a.md", "# A\n\none\n")
    os.makedirs(os.path.join(d, "archive", "20260101-000000"), exist_ok=True)
    _note(os.path.join(d, "archive", "20260101-000000"),
          "20260101-000000_old.md", "# Old\n\narchived\n")
    with open(os.path.join(d, "20260101-000003_bad.md"), "wb") as f:
        f.write(b"\xff\xfe not utf-8")
    notes = _memory_cmd._iter_notes(d)
    assert [n[0] for n in notes] == ["20260101-000002_b.md", "20260101-000001_a.md"], notes
    names = [n[0] for n in notes]
    titles = [n[3] for n in notes]
    assert titles == ["B", "A"]
    assert "20260101-000003_bad.md" not in names  # unreadable: skipped
    assert not any("archive" in n for n in names)
    # delete resolver (mtime-newest-first, unlike the name-ordered scanner)
    # + list + memory note all see exactly these names
    entry_names = sorted(h.name for h in _memory_cmd._scan_memory_entries("project"))
    assert entry_names == sorted(names), entry_names
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _memory_cmd._cmd_memory_list()
    out = buf.getvalue()
    assert "[project] B" in out and "[project] A" in out, out
    assert "archived" not in out


# --- /memory list scope fix -----------------------------------------------------

def test_memory_list_shows_project_only_store():
    d = _proj_dir()  # project scope only; global NX_HOME has no memories
    _note(d, "20260101-000001_solo.md", "# Solo note\n\nbody\n")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _memory_cmd._cmd_memory_list()
    out = buf.getvalue()
    assert "no memories saved yet" not in out, out
    assert "Solo note" in out and "[project]" in out, out


# --- writer race -----------------------------------------------------------------

def test_write_memory_file_never_clobbers_same_name():
    """Two writers that both pick the same (timestamp, slug) name must both
    survive with distinct suffixes — the pre-2.6 claim was exists-then-write
    with an os.replace publication, so the loser's bytes vanished."""
    d = _proj_dir()
    m.st.PROJECT_ROOT = os.path.dirname(os.path.dirname(d))
    a = _memory_cmd._write_memory_file("Clash Title", "first body", "project")
    b = _memory_cmd._write_memory_file("Clash Title", "second body", "project")
    assert a and b and a[0] != b[0], (a, b)
    fa, fb = open(a[0]).read(), open(b[0]).read()
    assert "first body" in fa and "second body" in fb
    # both are valid notes the scanner resolves
    titles = sorted(n[3] for n in _memory_cmd._iter_notes(d))
    assert titles == ["Clash Title", "Clash Title"]


def test_write_memory_file_suffix_walk_under_parallel_writers():
    """10 threads racing the writer: exactly 10 notes, none lost, none
    truncated, all readable."""
    d = _proj_dir()
    errs = []

    def w(i):
        try:
            r = _memory_cmd._write_memory_file(f"Race {i}", f"race body {i}", "project")
            if r is None:
                errs.append(f"writer {i} returned None")
        except Exception as e:  # pragma: no cover
            errs.append(f"{type(e).__name__}: {e}")

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(w, range(10)))
    assert not errs, errs
    notes = _memory_cmd._iter_notes(d)
    assert len(notes) == 10, [n[0] for n in notes]
    bodies = sorted(open(n[1]).read() for n in notes)
    assert len({b for b in bodies}) == 10
    for i in range(10):
        assert any(f"race body {i}" in b for b in bodies), i


def test_atomic_write_private_noreplace_contract():
    from nx._private import atomic_write_private_noreplace
    d = tempfile.mkdtemp(prefix="nx-noreplace-")
    p = os.path.join(d, "x.md")
    atomic_write_private_noreplace(p, "first")
    try:
        atomic_write_private_noreplace(p, "second")
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    assert open(p).read() == "first"
    assert os.listdir(d) == ["x.md"]  # no tmp left behind
    mode = oct(os.stat(p).st_mode & 0o777)
    assert mode == "0o600", mode


# --- optimize oldest-tail --------------------------------------------------------

def _opt_client_text(text):
    class _Msg:
        content = text
    class _Choice:
        message = _Msg()
        finish_reason = "stop"
    class _Resp:
        choices = [_Choice()]
        usage = None
        def model_dump(self):
            return {}
    class _Completions:
        def create(self, **kwargs):
            return _Resp()
    class _Chat:
        completions = _Completions()
    class _Client:
        chat = _Chat()
        base_url = "http://localhost:9/v1"
    return _Client()


def test_compact_reclaims_index_and_preserves_results():
    """Reconcile deletes rows when notes are archived but never shrinks the
    file; compact() reclaims the dead pages without changing results."""
    d = _proj_dir()
    for i in range(20):
        _note(d, "20260101-%06d_%02d.md" % (i, i),
              "# Note %d\n\n" % i + "filler line body words here\n" * 40)
    if not HAVE_FTS:
        assert _memindex.compact(d, force=True) is False  # disabled build: no-op
        return
    before = _memindex.search_dir(d, "filler")
    assert before is not None and len(before) == 20
    # force=True so the test doesn't depend on the free-page threshold
    assert _memindex.compact(d, force=True) is True
    after = _memindex.search_dir(d, "filler")
    assert [r[0] for r in after] == [r[0] for r in before]
    # the threshold path is a no-op on a freshly compacted index
    assert _memindex.compact(d) is False


def test_prune_archive_keeps_newest_and_never_touches_this_run(tmp_path):
    """Retention keeps the newest N run dirs, and a run dir touched this run
    (mtime >= region_start) is never deleted — a prune can't eat the run it is
    part of."""
    import time as _t
    from nx import _memory_optimize as opt
    d = str(tmp_path)
    arc = os.path.join(d, "archive")
    old_runs = ["20260101-000000", "20260102-000000", "20260103-000000"]
    for r in old_runs:
        os.makedirs(os.path.join(arc, r))
        os.utime(os.path.join(arc, r), (1_100_000_000, 1_100_000_000))
    fresh = "20260104-000000"
    os.makedirs(os.path.join(arc, fresh))
    region_start = _t.time() - 1  # the fresh run happened "this run"
    # runs sorted newest-name-first = [fresh, r2, r1, r0]; keep=2 drops r1, r0
    assert opt.prune_archive(d, 2, region_start) == 2
    assert not os.path.isdir(os.path.join(arc, old_runs[0]))
    assert not os.path.isdir(os.path.join(arc, old_runs[1]))
    assert os.path.isdir(os.path.join(arc, old_runs[2]))
    assert os.path.isdir(os.path.join(arc, fresh))
    # keep<=0 (the default) prunes nothing
    assert opt.prune_archive(d, 0, 0.0) == 0


def test_memory_archived_lists_and_restores(capsys):
    root = tempfile.mkdtemp(prefix="nx-arch-")
    m.st.PROJECT_ROOT = root
    d = os.path.join(root, ".nx", "memories")
    run = os.path.join(d, "archive", "20260101-000000")
    _note(run, "20251231-000000_retired.md", "# Retired Topic\n\nold body\n")
    m._confirm = lambda action, preview=None: True

    _memory_cmd._cmd_memory_archived("")
    out = capsys.readouterr().out
    assert "Retired Topic" in out and "archived memories" in out
    assert not os.path.exists(os.path.join(d, "20251231-000000_retired.md"))

    _memory_cmd._cmd_memory_archived("Retired Topic")
    out = capsys.readouterr().out
    assert "restored" in out
    # landed back in the live store with a fresh name and is gone from archive
    live = [x for x in _memory_cmd._iter_notes(d) if x[3] == "Retired Topic"]
    assert len(live) == 1
    assert live[0][4].count("_scope:") == 1      # fresh footer, not doubled
    assert not os.path.exists(os.path.join(run, "20251231-000000_retired.md"))


def test_iter_notes_titles_only_skips_body_reads():
    """titles_only keeps the same (name, mtime, title) view without reading
    every body — the session-start note and delete resolver use it."""
    d = _proj_dir()
    _note(d, "20260101-000001_a.md", "# Alpha\n" + "body line\n" * 500)
    _note(d, "20260101-000002_b.md", "# Beta\n" + "body line\n" * 500)
    full = _memory_cmd._iter_notes(d)
    head = _memory_cmd._iter_notes(d, titles_only=True)
    assert [x[:4] for x in full] == [x[:4] for x in head]
    # the body read was skipped: titles_only content is only the bounded head
    assert all(len(h[4]) < len(f[4]) for h, f in zip(head, full))


def test_optimize_max_notes_reviews_oldest_tail():
    root = tempfile.mkdtemp(prefix="nx-opt-tail-")
    m.st.PROJECT_ROOT = root
    d = os.path.join(root, ".nx", "memories")
    # 6 notes; the two NEWEST are what a session note shows and must stay
    # reviewable while the cap bites.
    stamps = ["20260101-00000%d" % i for i in range(1, 7)]
    for i, s in enumerate(stamps):
        _note(d, f"{s}_n{i}.md", f"# Note {i}\n\nbody {i}\n")
    os.utime(os.path.join(d, f"{stamps[0]}_n0.md"), (1_100_000_000, 1_100_000_000))
    os.utime(os.path.join(d, f"{stamps[1]}_n1.md"), (1_100_000_001, 1_100_000_001))
    os.utime(os.path.join(d, f"{stamps[2]}_n2.md"), (1_100_000_002, 1_100_000_002))
    os.utime(os.path.join(d, f"{stamps[3]}_n3.md"), (1_100_000_003, 1_100_000_003))
    os.utime(os.path.join(d, f"{stamps[4]}_n4.md"), (1_100_000_004, 1_100_000_004))
    os.utime(os.path.join(d, f"{stamps[5]}_n5.md"), (1_100_000_005, 1_100_000_005))
    real_plan_prompt = opt._plan_prompt
    seen_names: list[str] = []

    def _spy(chunk, focus, carry=None):
        seen_names.extend(n.name for n in chunk)
        return real_plan_prompt(chunk, focus, carry)

    opt._plan_prompt = _spy
    saved = os.environ.get("NX_MEMORY_OPTIMIZE_MAX_NOTES")
    os.environ["NX_MEMORY_OPTIMIZE_MAX_NOTES"] = "4"
    try:
        opt._cmd_memory_optimize([{"role": "user", "content": "hi"}], "")
    finally:
        opt._plan_prompt = real_plan_prompt
        if saved is None:
            os.environ.pop("NX_MEMORY_OPTIMIZE_MAX_NOTES", None)
        else:
            os.environ["NX_MEMORY_OPTIMIZE_MAX_NOTES"] = saved
    assert len(seen_names) == 4, seen_names
    # the OLDEST four (n0..n3) were reviewed; the newest tail was not
    assert sorted(seen_names) == [
        "20260101-000001_n0.md", "20260101-000002_n1.md",
        "20260101-000003_n2.md", "20260101-000004_n3.md"], seen_names


def _build_bloated_index(d):
    """Churn thousands of rows through reconcile so the FTS segment index
    piles up without a fold (the real-store bloat shape)."""
    for i in range(400):
        _note(d, "20260101-%06d_churn.md" % i,
              "# Churn %d\n\n" % i + "churn filler body words %d\n" % i * 40)
        if not HAVE_FTS:
            return None
        conn = _memindex._open(d)
        assert conn is not None
        try:
            _memindex.reconcile(conn, d)
        finally:
            conn.close()
    # delete everything again (the note bodies vanish; index keeps segments)
    for i in range(400):
        try:
            os.remove(os.path.join(d, "20260101-%06d_churn.md" % i))
        except OSError:
            pass
    conn = _memindex._open(d)
    assert conn is not None
    try:
        _memindex.reconcile(conn, d)
        data_rows = conn.execute(
            "SELECT count(*) FROM memories_fts_data").fetchone()[0]
        size = os.path.getsize(os.path.join(d, _memindex.INDEX_FILENAME))
    finally:
        conn.close()
    return data_rows, size


def test_compact_folds_segments_then_decides():
    """The GC seam must see segment bloat: fold BEFORE measuring, then gate on
    the post-fold freelist / bytes-per-doc. Pre-fix, the pre-fold freelist on
    a churn-bloated index sits far under _COMPACT_FREE_RATIO and the gate
    returned False — this test fails on the old ordering."""
    d = _proj_dir()
    # leave a couple of live notes so the store is not empty
    _note(d, "20260101-999999_live.md", "# Live\n\nlive note body marker\n")
    built = _build_bloated_index(d)
    if not HAVE_FTS or built is None:
        return
    db = os.path.join(d, _memindex.INDEX_FILENAME)
    data_rows_pre, size_pre = built
    assert size_pre >= 64 * 1024, (size_pre,)  # bloat actually happened
    # pre-fix the gate would NOT fire: fold here to compute the post-fold
    # freelist ratio the fixed gate now sees
    conn = _memindex._open(d)
    assert conn is not None
    try:
        _memindex._freq_fold(conn)
        conn.commit()
        pc = int(conn.execute("PRAGMA page_count").fetchone()[0] or 0)
        fl = int(conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
    finally:
        conn.close()
    assert fl >= _memindex._COMPACT_FREE_RATIO * pc or (
        os.path.getsize(db) > _memindex._COMPACT_BYTES_PER_DOC * 1), (
        fl, pc, "expected reclaimable space post-fold")
    assert _memindex.compact(d) is True, "unforced compact must VACUUM now"
    size_post = os.path.getsize(db)
    assert size_post < size_pre / 4, (size_pre, size_post)
    conn = _memindex._open(d)
    try:
        dr = conn.execute(
            "SELECT count(*) FROM memories_fts_data").fetchone()[0]
    finally:
        conn.close()
    # the fold inside compact() collapsed the segment index; after a VACUUM
    # the data table is at its floor (a handful of b-tree rows), not the
    # churned pile-up the file had before
    assert dr <= 32, (data_rows_pre, dr)
    # results unchanged, and the freshly compacted index is left alone
    rows = _memindex.search_dir(d, "live note body marker")
    assert rows and rows[0][0].endswith("live.md")
    assert _memindex.compact(d) is False


def test_reconcile_fold_probe_keeps_ratio_healthy():
    """After churn cycles the data-rows/doc ratio stays under the fold
    threshold (the probe folds), and search answers are unchanged by a fold."""
    d = _proj_dir()
    _note(d, "20260101-999999_steady.md", "# Steady\n\nsteady marker body\n")
    if not HAVE_FTS:
        return
    built = _build_bloated_index(d)
    if built is None:
        return
    conn = _memindex._open(d)
    assert conn is not None
    try:
        _memindex.reconcile(conn, d)
        docs = conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
        dr = conn.execute(
            "SELECT count(*) FROM memories_fts_data").fetchone()[0]
    finally:
        conn.close()
    assert docs == 1, docs
    assert dr < _memindex._FOLD_RATIO * docs, (dr, docs)
    # search results are stable across a fold
    before = _memindex.search_dir(d, "steady marker body")
    conn = _memindex._open(d)
    try:
        _memindex._maybe_fold(conn)
    finally:
        conn.close()
    after = _memindex.search_dir(d, "steady marker body")
    assert [r[0] for r in (before or [])] == [r[0] for r in (after or [])]
    assert after and after[0][0].endswith("steady.md")


def test_emptied_store_unlinks_index_nonempty_keeps_it():
    """Zero live notes -> the stale index file (and -wal/-shm) is dropped; a
    non-empty store's index survives."""
    d = _proj_dir()
    _note(d, "20260101-000001_a.md", "# A\n\nsome body\n")
    if not HAVE_FTS:
        return
    assert _memindex.search_dir(d, "some body")
    db = os.path.join(d, _memindex.INDEX_FILENAME)
    wal = db + "-wal"
    open(wal, "wb").close()  # simulate a sibling
    # empty the store
    os.remove(os.path.join(d, "20260101-000001_a.md"))
    conn = _memindex._open(d)
    assert conn is not None
    try:
        _memindex.reconcile(conn, d)
    finally:
        conn.close()
    assert not os.path.exists(db), os.listdir(d)
    assert not os.path.exists(wal), os.listdir(d)
    # a search rebuilds it (pure cache)
    _note(d, "20260101-000002_b.md", "# B\n\nrebuilt body\n")
    rows = _memindex.search_dir(d, "rebuilt body")
    assert rows and rows[0][0].endswith("b.md")
    # non-empty store: index survives reconcile
    conn = _memindex._open(d)
    try:
        _memindex.reconcile(conn, d)
    finally:
        conn.close()
    assert os.path.exists(db)


# --- supersession filtering (0.158.20) -----------------------------------------
#
# `reconcile` builds its rows from a RAW disk listing, so it is the one place
# that does not inherit `_iter_notes`' supersession filter for free. It mirrors
# the exclusion (dropping the row, exactly as a vanished note's row is dropped),
# which keeps a retired note out of `search_dir`'s candidates — those flow
# straight into `_memory_scan`'s re-filter, which never re-reads the file.

_SUPER_FOOT = "_scope: project  _saved: 2026-01-01 00:00:00"


def test_reconcile_drops_the_row_of_a_superseded_note():
    if not HAVE_FTS:
        return
    d = _proj_dir()
    live = _note(d, "20260101-000002_live.md",
                 "# Live note\n\nprosemark body here\n\n---\n" + _SUPER_FOOT + "\n")
    retired = _note(d, "20260101-000001_old.md",
                    "# Old note\n\nprosemark body here\n\n---\n" + _SUPER_FOOT +
                    "  _superseded_by: 20260101-000002_live.md  "
                    "_valid_until: 2026-01-02T00:00:00\n")
    rows = _memindex.search_dir(d, "prosemark body")
    assert rows is not None
    names = [r[0] for r in rows]
    assert os.path.basename(live) in names, names
    assert os.path.basename(retired) not in names, names
    # unsuperseding (clearing the pair) re-indexes it on the next search:
    # the footer rewrite moves the mtime, which IS the cache-invalidation token.
    with open(retired, encoding="utf-8") as f:
        text = f.read()
    _memindex_live = text.replace(
        "  _superseded_by: 20260101-000002_live.md  "
        "_valid_until: 2026-01-02T00:00:00", "")
    import time as _t
    with open(retired, "w", encoding="utf-8") as f:
        f.write(_memindex_live)
    os.utime(retired, (_t.time() + 5, _t.time() + 5))
    rows = _memindex.search_dir(d, "prosemark body")
    assert os.path.basename(retired) in [r[0] for r in rows], rows


def test_search_and_plain_scan_agree_that_a_superseded_note_is_invisible():
    """The accelerator and the fallback must answer identically — otherwise the
    invariant would depend on whether FTS happened to be warm."""
    if not HAVE_FTS:
        return
    d = _proj_dir()
    _note(d, "20260101-000002_b.md", "# Both\n\nagree mark body\n\n---\n"
          + _SUPER_FOOT + "\n")
    _note(d, "20260101-000001_a.md", "# Both\n\nagree mark body\n\n---\n"
          + _SUPER_FOOT + "  _superseded_by: 20260101-000002_b.md\n")
    fts = sorted(r[0] for r in _memindex.search_dir(d, "agree mark body"))
    plain = sorted(n[0] for n in _memory_cmd._iter_notes(d)
                   if "agree mark body" in n[4])
    assert fts == plain == ["20260101-000002_b.md"], (fts, plain)
    os.environ["NX_MEMORY_FTS"] = "0"
    try:
        hits, _ov, total = _memory_cmd._memory_scan("agree mark body", "project", 0)
        assert total == 1 and hits[0].name == "20260101-000002_b.md"
    finally:
        os.environ["NX_MEMORY_FTS"] = "1"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
