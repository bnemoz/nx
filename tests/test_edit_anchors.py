"""F8: line-anchored edit_file (old_lines), whitespace-tolerant fallback,
and delete_file with undo support.
"""
import os

os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest

import nx  # Tier A: import once, monkeypatch only — no module-dict rewind


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_HOME", str(tmp_path / "nxhome"))
    monkeypatch.setenv("NX_ENV_FILE", str(tmp_path / "env"))
    monkeypatch.setenv("NX_SESSIONS_DIR", str(tmp_path / "nxhome" / "sessions"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nx, "YOLO", True, raising=False)
    monkeypatch.setattr(nx._state, "YOLO", True)
    # project-root globals the risk-hint path reads
    monkeypatch.setattr(nx._state, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(nx._state, "PROJECT_ROOT_IS_GIT", True)
    yield nx, tmp_path
    nx._state._UNDO_STACK.clear()
    nx._state._TOOL_RESULT_CACHE.clear()


SAMPLE = "alpha\nbeta\ngamma\ndelta\nepsilon\n"


# --- old_lines anchored edits ------------------------------------------------

def test_edit_file_old_lines_replaces_range(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "f.txt"
    p.write_text(SAMPLE)
    out = nx.edit_file(str(p), new="BETA2\ngamma2", old_lines="2-3")
    assert out.startswith("edited")
    assert p.read_text() == "alpha\nBETA2\ngamma2\ndelta\nepsilon\n"


def test_edit_file_old_lines_single_line(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "f.txt"
    p.write_text(SAMPLE)
    nx.edit_file(str(p), new="DELTA!", old_lines="4")
    assert p.read_text() == "alpha\nbeta\ngamma\nDELTA!\nepsilon\n"


def test_edit_file_old_lines_delete_range_with_empty_new(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "f.txt"
    p.write_text(SAMPLE)
    nx.edit_file(str(p), new="", old_lines="2-3")
    assert p.read_text() == "alpha\n\ndelta\nepsilon\n"


def test_edit_file_old_lines_range_validation(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "f.txt"
    p.write_text(SAMPLE)
    assert "outside 1..5" in nx.edit_file(str(p), new="x", old_lines="4-9")
    assert "old_lines wants" in nx.edit_file(str(p), new="x", old_lines="abc")
    # 0 is a parseable range but outside the file
    assert "outside 1..5" in nx.edit_file(str(p), new="x", old_lines="0")
    assert "not both" in nx.edit_file(str(p), new="x", old="beta", old_lines="2")


def test_edit_file_old_lines_preserves_trailing_newline_shape(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "no-nl.txt"
    p.write_text("a\nb\nc")  # no trailing newline
    nx.edit_file(str(p), new="B", old_lines="2")
    assert p.read_text() == "a\nB\nc"


def test_edit_file_old_lines_schema_no_longer_requires_old(nx_fresh):
    nx, tmp = nx_fresh
    schema = next(t for t in nx.TOOLS if t["function"]["name"] == "edit_file")
    # mode-aware required: only `path` is schema-required; old/old_lines/new
    # requirements are enforced per-mode in the body + check_required.
    assert schema["function"]["parameters"]["required"] == ["path"]
    assert "old_lines" in schema["function"]["parameters"]["properties"]


# --- whitespace-tolerant fallback ---------------------------------------------

def test_edit_file_whitespace_tolerant_unique_match(nx_fresh):
    nx, tmp = nx_fresh
    src = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
    p = tmp / "w.py"
    p.write_text(src)
    # model pasted the block with wrong indentation
    bad_old = "def f():\nreturn 1"  # missing the 4-space indent on return
    out = nx.edit_file(str(p), old=bad_old, new="def f():\n    return 42")
    assert out.startswith("edited")
    assert "whitespace-tolerantly" in out
    assert "    return 42" in p.read_text()


def test_edit_file_whitespace_tolerant_ambiguous_never_guesses(nx_fresh):
    nx, tmp = nx_fresh
    src = "def f():\n    return 1\n\n\ndef g():\n    return 1\n"  # `    return 1` twice
    p = tmp / "w.py"
    p.write_text(src)
    out = nx.edit_file(str(p), old="return 1", new="return 9")
    assert out.startswith("ERROR")  # ambiguous even tolerance-wise → no edit
    assert p.read_text() == src


def test_edit_file_exact_match_still_preferred_and_silent(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "w.py"
    p.write_text("def f():\n    return 1\n")
    out = nx.edit_file(str(p), old="    return 1", new="    return 2")
    assert out.startswith("edited")
    assert "whitespace-tolerantly" not in out  # exact path — no note
    assert "return 2" in p.read_text()


def test_edit_file_true_zero_match_still_errors(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "w.py"
    p.write_text("alpha\n")
    out = nx.edit_file(str(p), old="completely different text", new="x")
    assert out.startswith("ERROR") and "matched 0 times" in out


# --- delete_file ---------------------------------------------------------------

def test_delete_file_removes_and_is_undoable(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "victim.txt"
    p.write_text("precious")
    out = nx.delete_file(str(p))
    assert out.startswith("deleted") and "undo" in out
    assert not p.exists()
    # /undo restores content
    from nx._undo import undo_last
    _, msg = undo_last()
    assert p.exists() and p.read_text() == "precious"


def test_delete_file_refuses_directories_and_dotfiles(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    d = tmp / "adir"
    d.mkdir()
    assert "directory" in nx.delete_file(str(d))
    assert "refusing" in nx.delete_file(str(tmp / ".env"))
    assert "refusing" in nx.delete_file(".")
    assert "does not exist" in nx.delete_file(str(tmp / "nope.txt"))


def test_delete_file_in_dispatch_and_risk(nx_fresh):
    nx, tmp = nx_fresh
    from nx._dispatch import DISPATCH
    assert "delete_file" in DISPATCH
    # in-project delete classifies medium locally without a model call
    p = tmp / "x.txt"
    p.write_text("x")
    hint = nx._risk._local_risk_hint(f"delete {p}")
    assert hint is not None and hint[0] == "medium"
    # delete + undo restores read cache invalidation path doesn't crash
    nx.delete_file(str(p))
    assert p.exists() is False


def test_delete_file_run_tool_undo_roundtrip(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "rt.txt"
    p.write_text("content-here")
    out = nx.run_tool("delete_file", {"path": str(p)})
    assert out.startswith("deleted")
    from nx._undo import undo_last
    _, msg = undo_last()
    assert "delete_file" in msg and p.exists()
