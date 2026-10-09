#!/usr/bin/env python3
"""Memory drift stamping: which commit a note was verified against.

Covers the five seams of the feature — the write-time stamp, the retroactive
backfill in the scanner, the read-time annotation (fresh / stale / unstamped),
the write-time challenge on a superseded stale note, and the ranking demotion.

Tests build their OWN throwaway git repos (the suite runs inside the nx repo,
which is itself a git repo — nothing here may depend on its history): a note is
written "as of" an older commit by committing, recording that sha, committing
more, and stamping the note with the older sha directly in its footer. Fully
offline: no model, no network, git subprocesses over temp dirs only.

The drift layer itself lives in nx/_memory_store.py (imported here as `drift`),
so adding it does not grow the eagerly-imported nx module set the perf ledger
budgets.
"""
import os
import subprocess
import sys
import tempfile

# Isolation convention from test_memory_tools.py: a private NX_HOME (so a real
# memory store can't leak in), no env file, repo root on sys.path.
_tmp = tempfile.mkdtemp(prefix="nx-test-drift-")
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"
# The FTS accelerator is exercised explicitly (test_fts_and_plain_scan_...);
# every other test pins it off so recall differences can't change an ordering.
os.environ["NX_MEMORY_FTS"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _memory_cmd  # noqa: E402
from nx import _memory_search  # noqa: E402
from nx import _memory_store as drift  # noqa: E402  (the drift layer lives here)

_GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t",
        "-c", "commit.gpgsign=false"]


def _git(repo, *args):
    r = subprocess.run(_GIT + ["-C", repo] + list(args),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"git {' '.join(args)}: {r.stderr}"
    return (r.stdout or "").strip()


def _mkrepo():
    """A fresh git repo with one commit touching src/app.py."""
    repo = tempfile.mkdtemp(prefix="nx-drift-repo-")
    _git(repo, "init", "-q")
    _commit(repo, "src/app.py", "def app():\n    return 1\n", "c1")
    return repo


def _commit(repo, relpath, text, msg):
    full = os.path.join(repo, relpath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


_saved = {}


def _setup_function():
    _saved["confirm"] = m._confirm
    _saved["home"] = m.st.SESSION_HOME
    _saved["root"] = m.st.PROJECT_ROOT
    _saved["isgit"] = m.st.PROJECT_ROOT_IS_GIT
    m._confirm = lambda action, preview=None: True
    m.st.SESSION_HOME = _tmp


def _teardown_function():
    m._confirm = _saved["confirm"]
    m.st.SESSION_HOME = _saved["home"]
    m.st.PROJECT_ROOT = _saved["root"]
    m.st.PROJECT_ROOT_IS_GIT = _saved["isgit"]


setup_function = lambda _f: _setup_function()
teardown_function = lambda _f: _teardown_function()


def _in_repo(repo, is_git=True):
    m.st.PROJECT_ROOT = repo
    m.st.PROJECT_ROOT_IS_GIT = is_git


def _write(title, body, scope="project"):
    return _memory_cmd.memory_write(title=title, body=body, scope=scope)


def _notes_dir(repo):
    return os.path.join(repo, ".nx", "memories")


def _only_note(repo):
    d = _notes_dir(repo)
    names = [n for n in os.listdir(d) if n.endswith(".md")]
    assert len(names) == 1, names
    with open(os.path.join(d, names[0]), encoding="utf-8") as f:
        return f.read()


# --- 1. stamping on write ------------------------------------------------------

def test_write_in_git_repo_stamps_head_and_mentioned_paths():
    repo = _mkrepo()
    head = _git(repo, "rev-parse", "HEAD")
    _in_repo(repo)
    res = _write("Widget wiring",
                 "The widget lives in `src/app.py`; covered by tests/test_app.py. "
                 "See https://example.com/docs/x.html and and/or for noise.")
    assert res.startswith("saved project memory: Widget wiring"), res
    stamp = drift._parse_drift(_only_note(repo))
    assert stamp is not None
    assert stamp.sha == head
    assert stamp.paths == ("src/app.py", "tests/test_app.py"), stamp.paths
    assert stamp.root == repo
    # A stamp at HEAD is fresh, and pays no demotion.
    assert drift._drift_freshness(stamp) == (0, 0)
    assert drift._drift_demote(stamp) == 0.0


def test_write_outside_a_git_repo_records_unknown_sha():
    plain = tempfile.mkdtemp(prefix="nx-drift-plain-")
    _in_repo(plain, is_git=False)
    res = _write("No repo note", "Mentions src/app.py anyway.")
    assert res.startswith("saved project memory: No repo note"), res
    content = _only_note(plain)
    assert "_drift: -|-|" in content  # the field exists, the sha is unknown
    assert drift._parse_drift(content) is None
    assert drift._annotate_drift(None) == "[unverified freshness]"


# --- 2. retroactive backfill ---------------------------------------------------

def test_scanner_backfills_unstamped_notes_once():
    repo = _mkrepo()
    head = _git(repo, "rev-parse", "HEAD")
    _in_repo(repo)
    d = _notes_dir(repo)
    os.makedirs(d, exist_ok=True)
    old = os.path.join(d, "2025-01-01-000000_legacy.md")
    with open(old, "w", encoding="utf-8") as f:
        f.write("# Legacy\n\nthe widget is defined in `src/app.py`\n\n"
                "---\n_scope: project  _saved: 2025-01-01 00:00:00\n")

    rows = _memory_cmd._iter_notes(d)
    assert len(rows) == 1
    name, _p, _m, title, content = rows[0]
    stamp = drift._parse_drift(content)
    assert stamp is not None and stamp.sha == head
    assert stamp.paths == ("src/app.py",), stamp.paths
    # persisted, and annotated fresh (it was written against a recent state)
    with open(old, encoding="utf-8") as f:
        assert "_drift: " in f.read()
    assert drift._annotate_drift(stamp) == f"[fresh: verified at {head[:7]}, 0 commits behind]"

    # A second scan leaves it byte-identical (the field exists => never
    # retouched), and the stamped body is what the FTS accelerator now serves
    # too (the mtime moved, so reconcile re-indexed the row).
    before = open(old, encoding="utf-8").read()
    again = _memory_cmd._iter_notes(d)[0][4]
    assert drift._parse_drift(again).sha == head
    assert open(old, encoding="utf-8").read() == before


def test_backfill_skips_dirs_that_are_not_the_current_project():
    repo = _mkrepo()
    _in_repo(repo, is_git=False)  # no drift target: the scanner must not write
    d = _notes_dir(repo)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "2025-01-01-000000_legacy.md")
    body = "# Legacy\n\nsrc/app.py\n\n---\n_scope: project  _saved: x\n"
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    _memory_cmd._iter_notes(d)
    assert open(p, encoding="utf-8").read() == body


def test_backfill_skips_notes_without_a_footer():
    """A footer-less note has no `_scope:` line for `_MEMORY_FOOTER_RE` to strip
    against, so stamping it would leak `_drift:` into the body the archiver
    re-saves. Such notes stay unverified instead."""
    repo = _mkrepo()
    _in_repo(repo)
    d = _notes_dir(repo)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "2025-01-01-000000_headonly.md")
    body = "# Head only\n\nsrc/app.py\n"
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    rows = _memory_cmd._iter_notes(d)
    assert rows[0][4] == body
    assert open(p, encoding="utf-8").read() == body
    hits, _o, _t = _memory_search._memory_scan("head", "project", 0)
    assert hits[0].ann == "[unverified freshness]", hits


# --- 3. read-time annotation ---------------------------------------------------

def test_annotation_fresh_stale_and_unstamped():
    repo = _mkrepo()
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "src/app.py", "def app():\n    return 2\n", "c2")
    _commit(repo, "docs/x.md", "unrelated\n", "c3")
    head = _git(repo, "rev-parse", "HEAD")  # HEAD now: 2 commits past `old`

    stale = drift._DriftStamp(old, ("src/app.py", "nx/missing.py"), "", repo)
    assert drift._drift_freshness(stale) == (2, 1)
    assert drift._annotate_drift(stale) == (
        f"[stale: verified at {old[:7]}, 2 commits behind, 1 of its files changed]")

    fresh = drift._DriftStamp(head, ("src/app.py",), "", repo)
    assert drift._annotate_drift(fresh) == f"[fresh: verified at {head[:7]}, 0 commits behind]"

    # unstamped, unknown sha, and a sha this repo doesn't know -> all unverified
    assert drift._annotate_drift(None) == "[unverified freshness]"
    assert drift._annotate_drift(drift._parse_drift("# T\n\nx\n")) == "[unverified freshness]"
    bogus = drift._DriftStamp("0" * 40, (), "", repo)
    assert drift._annotate_drift(bogus) == "[unverified freshness]"
    assert drift._drift_demote(bogus) == 0.0


def test_annotation_via_memory_search_output():
    repo = _mkrepo()
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "src/gadget.py", "X = 1\n", "c2")
    _in_repo(repo)
    d = _notes_dir(repo)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "2025-01-01-000000_zzz.md"), "w", encoding="utf-8") as f:
        f.write("# Gadget note\n\nthe gadget registry\n\n---\n"
                f"_scope: project  _saved: x  _drift: {old}|-|2025-01-01T00:00:00|{repo}|-\n")
    hits = _memory_search.memory_search("gadget registry", scope="project")
    assert "[stale: verified at " in hits, hits
    # the malformed variant (too few fields) reads as unstamped, not as a crash
    with open(os.path.join(d, "2025-01-01-000000_yyy.md"), "w", encoding="utf-8") as f:
        f.write("# Gadget broken\n\nthe gadget registry\n\n---\n"
                "_scope: project  _saved: x  _drift: deadbe\n")
    assert "[unverified freshness]" in _memory_search.memory_search("gadget broken")


# --- 4. write-time challenge ---------------------------------------------------

def test_write_over_stale_note_appends_challenge_but_still_writes():
    repo = _mkrepo()
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "src/app.py", "def app():\n    return 3\n", "c2")
    _in_repo(repo)
    d = _notes_dir(repo)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "2025-01-01-000000_widget.md"), "w", encoding="utf-8") as f:
        f.write("# Widget wiring\n\nold claim\n\n---\n"
                f"_scope: project  _saved: x  _drift: {old}|-|2025-01-01T00:00:00|{repo}|-\n")

    res = _write("Widget wiring", "new claim about `src/app.py`")
    assert res.startswith("saved project memory: Widget wiring"), res  # never blocked
    assert ("Note was 1 commits stale with 0 changed files; "
            "confirm its claims still hold or scope your update.") in res, res

    # The superseded stale note is still on disk (writes never replace files),
    # so the challenge repeats — and the write still lands.
    res2 = _write("Widget wiring", "even newer claim")
    assert res2.startswith("saved project memory: Widget wiring"), res2


def test_challenge_absent_when_note_is_fresh():
    repo = _mkrepo()
    _in_repo(repo)
    assert _write("Gizmo fact", "one").startswith("saved project memory")
    res = _write("Gizmo fact", "two")
    assert "stale" not in res, res


# --- 5. ranking demotion -------------------------------------------------------

def test_stale_note_ranks_below_an_equal_scoring_fresh_note():
    repo = _mkrepo()
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "src/wig.py", "X = 1\n", "c2")
    _commit(repo, "src/wig.py", "X = 2\n", "c3")
    _commit(repo, "src/wig.py", "X = 3\n", "c4")
    _in_repo(repo)
    d = _notes_dir(repo)
    os.makedirs(d, exist_ok=True)
    footer = f"_scope: project  _saved: x  _drift: {old}|src/wig.py|t|{repo}|-"
    # Identical bodies -> identical raw scores; only the stamp differs, so the
    # order can only come from the demotion.
    with open(os.path.join(d, "2025-01-01-000001_stale.md"), "w", encoding="utf-8") as f:
        f.write("# Wobble stale\n\nwobble registry notes\n\n---\n" + footer + "\n")
    with open(os.path.join(d, "2025-01-01-000002_fresh.md"), "w", encoding="utf-8") as f:
        f.write("# Wobble fresh\n\nwobble registry notes\n\n---\n"
                "_scope: project  _saved: x\n")  # backfilled at HEAD -> fresh

    hits, overflow, total = _memory_search._memory_scan("wobble registry", "project", 0)
    assert total == 2, hits
    assert hits[0].title == "Wobble fresh", [h.title for h in hits]
    assert hits[0].ann.startswith("[fresh"), hits[0].ann
    assert hits[1].ann.startswith("[stale"), hits[1].ann
    # the raw scores are equal; the demoted one is what sorted
    assert hits[0].score == hits[1].score
    assert hits[1].eff < hits[0].eff
    assert drift._drift_demote(drift._parse_drift(footer)) > 0.0

    out = _memory_search.memory_search("wobble registry")
    assert out.index("Wobble fresh") < out.index("Wobble stale")


def test_unstamped_note_keeps_its_raw_score():
    repo = _mkrepo()
    _in_repo(repo, is_git=False)
    d = _notes_dir(repo)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "2025-01-01-000000_x.md"), "w", encoding="utf-8") as f:
        f.write("# Yammer\n\nyammer registry\n\n---\n_scope: project  _saved: x\n")
    hits, _o, total = _memory_search._memory_scan("yammer registry", "project", 0)
    assert total == 1
    assert hits[0].eff == hits[0].score
    assert hits[0].ann == "[unverified freshness]"


def test_fts_and_plain_scan_annotate_identically():
    """The accelerator must not change the freshness verdict: the plain scan
    annotates from the file on disk, the FTS path from the indexed body — same
    names, raw/demoted scores and annotation tag either way."""
    repo = _mkrepo()
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "src/giz.py", "X = 1\n", "c2")
    _commit(repo, "src/giz.py", "X = 2\n", "c3")
    _in_repo(repo)
    d = _notes_dir(repo)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "2025-01-01-000001_giz.md"), "w", encoding="utf-8") as f:
        f.write("# Gizmo notes\n\ngizmo registry\n\n---\n"
                f"_scope: project  _saved: x  _drift: {old}|src/giz.py|t|{repo}|-\n")

    def _results():
        hits, _o, tot = _memory_search._memory_scan("gizmo registry", "project", 0)
        return [(h.name, round(h.score, 3), round(h.eff, 3), h.ann) for h in hits], tot

    os.environ["NX_MEMORY_FTS"] = "1"
    try:
        with_fts = _results()
    finally:
        os.environ["NX_MEMORY_FTS"] = "0"  # this module's baseline
    assert len(with_fts[0]) == 1, with_fts
    assert with_fts[0][0][3].startswith("[stale"), with_fts
    assert with_fts[0][0][2] < with_fts[0][0][1]  # demoted
    assert _results() == with_fts
