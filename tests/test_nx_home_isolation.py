"""The user's real ~/.nx must never be written by a test run.

Regression (0.60.1): standalone-style test files that set NX_HOME at import
time and then call m.main() at module scope (no test function ran yet) were
appending "hello"/"world" to the user's REAL input_history.json — and
trimming it to the ring-buffer cap on every run, evicting genuine entries.
Root cause: _input_history._history_path() / _sessions_fs._sessions_dir()
read only the st.SESSION_HOME mirror (a snapshot of the real home taken by
conftest's baseline import), never the live NX_HOME the module just set.
"""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-home-isolation-")
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = os.path.join(_tmp, "sessions")
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def _hist_path():
    return m._input_history._history_path()


def test_history_path_honors_live_nx_home():
    assert _hist_path() == os.path.join(_tmp, "input_history.json")


def test_sessions_dir_honors_live_nx_home():
    assert m._sessions_dir() == os.path.join(_tmp, "sessions")


def test_prune_verb_reaches_the_gc():
    # Regression (0.60.1): `nx sessions --prune` fell through to the listing
    # verb, which treated "--prune" as a search query.
    from nx._sessions_cmd import _cli_prune_sessions
    assert _cli_prune_sessions(["sessions", "--prune"]) is True


def test_recorded_input_lands_in_tmp_not_real_home():
    # Reproduce the leak: what test_compact_alias et al. did at module scope.
    hist = m._input_history._load_input_history()
    m._input_history._record_input("isolation-probe-never-written", hist)
    on_disk = json.load(open(_hist_path()))
    assert "isolation-probe-never-written" in on_disk
    # The real recall file must be untouched by this run.
    real = os.path.expanduser("~/.nx/input_history.json")
    if os.path.exists(real):
        real_entries = json.load(open(real))
        assert "isolation-probe-never-written" not in real_entries
