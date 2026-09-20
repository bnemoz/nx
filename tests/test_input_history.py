"""Global cross-session input recall (~/.nx/input_history.json):
what Up/Down at the prompt remembers across runs.

Covers the record policy (skip blanks/slash commands, consecutive-dedup),
the ring-buffer cap, the NX_HISTORY_LINES=0 kill switch, corrupt-file
fallback, per-entry truncation, file privacy, and locked concurrent appends.
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile

# --- test isolation: no local credentials, no real ~/.nx -------------------
for _var in list(os.environ):
    if _var.startswith("NX_"):
        del os.environ[_var]
os.environ["NX_ENV_FILE"] = "/dev/null"

_tmp = tempfile.mkdtemp(prefix="nx-input-history-")
atexit.register(lambda: shutil.rmtree(_tmp, ignore_errors=True))
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = os.path.join(_tmp, "sessions")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def _hist_path():
    return m._input_history._history_path()


def _reset_env(cap=None):
    """Point the recall file at a fresh temp path (and optionally cap lines),
    then reload just the path resolution — SESSION_HOME is read live from
    _state, so no module reload is needed."""
    # Temp files live in the module's own scratch dir, NOT /tmp: some tests
    # leave the history file (and every written one leaves a .lock sidecar
    # that _exclusive_file_lock intentionally never unlinks), so dropping
    # them in /tmp would leak a pair per test, per run.
    fd, path = tempfile.mkstemp(prefix="nx-hist-", suffix=".json", dir=_tmp)
    os.close(fd)
    os.unlink(path)  # want a missing file, not an empty one
    os.environ["NX_INPUT_HISTORY_FILE"] = path
    if cap is not None:
        m.HISTORY_LINES = cap
    else:
        m.HISTORY_LINES = m._input_history.INPUT_HISTORY_LINES_DEFAULT
    return path


def _read_file(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_record_appends_to_memory_and_disk():
    path = _reset_env()
    hist = []
    assert m._record_input("hello world", hist) is True
    assert hist == ["hello world"]
    assert _read_file(path) == ["hello world"]


def test_record_persists_across_load():
    _reset_env()
    hist = []
    m._record_input("first", hist)
    m._record_input("second", hist)
    # A fresh run (new in-memory list) sees both prior submissions.
    assert m._load_input_history() == ["first", "second"]


def test_slash_commands_and_blanks_are_skipped():
    path = _reset_env()
    hist = []
    assert m._record_input("/yolo", hist) is False
    assert m._record_input("   ", hist) is False
    assert m._record_input("", hist) is False
    assert hist == []
    assert not os.path.exists(path)


def test_only_consecutive_duplicates_are_deduped():
    _reset_env()
    hist = []
    m._record_input("same", hist)
    assert m._record_input("same", hist) is False  # repeat of the very last
    m._record_input("other", hist)
    assert m._record_input("same", hist) is True   # non-consecutive repeat stays
    assert hist == ["same", "other", "same"]


def test_cap_keeps_newest_entries_in_memory_and_on_disk():
    path = _reset_env(cap=3)
    hist = []
    for i in range(5):
        m._record_input(f"msg-{i}", hist)
    # In-memory list is uncapped (this session typed all five); the file is
    # the ring buffer.
    assert hist == [f"msg-{i}" for i in range(5)]
    assert _read_file(path) == ["msg-2", "msg-3", "msg-4"]
    assert m._load_input_history() == ["msg-2", "msg-3", "msg-4"]


def test_load_trims_an_oversized_existing_file():
    path = _reset_env(cap=2)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(["a", "b", "c", "d"], f)
    assert m._load_input_history() == ["c", "d"]


def test_history_lines_zero_disables_everything():
    path = _reset_env(cap=0)
    hist = []
    assert m._load_input_history() == []
    # _record_input returns True (it recorded in memory) but writes nothing
    # when disabled — the on-disk file must not appear.
    assert m._record_input("hello", hist) is True
    assert hist == ["hello"]
    assert not os.path.exists(path)


def test_corrupt_file_degrades_to_empty_then_recovers():
    path = _reset_env()
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert m._load_input_history() == []
    # Recording over a corrupt file rewrites it with just the new entry.
    hist = []
    m._record_input("after corruption", hist)
    assert _read_file(path) == ["after corruption"]


def test_wrong_shape_file_degrades_to_empty():
    path = _reset_env()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"not": "a list"}, f)
    assert m._load_input_history() == []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(["ok", 42, None, "also ok"], f)
    assert m._load_input_history() == ["ok", "also ok"]


def test_missing_file_degrades_to_empty():
    _reset_env()
    assert m._load_input_history() == []


def test_long_entries_are_truncated():
    _reset_env()
    old = m._input_history.INPUT_HISTORY_ENTRY_CHARS
    m._input_history.INPUT_HISTORY_ENTRY_CHARS = 10
    try:
        hist = []
        m._record_input("0123456789abcdef", hist)
        assert hist == ["0123456789"]
    finally:
        m._input_history.INPUT_HISTORY_ENTRY_CHARS = old


def test_history_file_is_owner_only():
    path = _reset_env()
    hist = []
    m._record_input("secret-ish prompt", hist)
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600


def test_concurrent_appends_are_not_lost():
    path = _reset_env()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import sys; sys.path.insert(0, %r); import nx as m; "
        "m._record_input(sys.argv[1], [])" % root
    )
    env = {k: v for k, v in os.environ.items()}
    env["NX_INPUT_HISTORY_FILE"] = path
    procs = [
        subprocess.Popen([sys.executable, "-c", script, f"concurrent-{i}"], env=env)
        for i in range(4)
    ]
    for p in procs:
        assert p.wait() == 0
    # No lost updates: all four appends survived the locked read-modify-write.
    assert sorted(_read_file(path)) == sorted(f"concurrent-{i}" for i in range(4))


def test_repl_state_history_is_loaded_not_fresh():
    """The REPL seeds its recall list from the global file (regression guard
    for the _repl.py wiring) — checked at the function level: a fresh _ReplState
    history passed through _record_input keeps entries from _load_input_history."""
    _reset_env()
    hist = m._load_input_history()
    m._record_input("typed at the prompt", hist)
    assert hist == ["typed at the prompt"]
    # And the next run picks it up:
    assert m._load_input_history() == ["typed at the prompt"]
