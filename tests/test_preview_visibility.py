"""0.53.1: approval previews ride TOOL_OUTPUT_VISIBLE.

The approval preview (git_commit's staged diff, edit_file's before/after) is
tool output on screen too, but `_confirm` printed it unconditionally on every
path — so with `/tooloutput` collapsed (the default) a git_commit diff still
printed in full, defeating the toggle. The fix gates the preview on
TOOL_OUTPUT_VISIBLE at the choke point; the Y/n prompt line itself must keep
showing (it names the action being approved).

Run offline — the classifier/prompt are stubbed, and the git_commit test uses
only the local `git` binary in a tmp repo.
"""
import builtins
import contextlib
import importlib
import io
import os
import subprocess
import sys

_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    ".tmp-preview-visibility")
os.environ.setdefault("NX_SESSIONS_DIR", _tmp)
os.environ.setdefault("NX_HOME", _tmp)
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import nx  # noqa: E402  (conftest laid the safe NX_* baseline already)


@contextlib.contextmanager
def capture_stdout():
    """Capture print() output the same way test_tool_output_toggle.py does."""
    out = io.StringIO()
    real = builtins.print

    def fake(*a, **kw):
        out.write(" ".join(str(x) for x in a) + (kw.get("end", "\n")))

    builtins.print = fake
    try:
        yield out
    finally:
        builtins.print = real


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_HOME", str(tmp_path / "nxhome"))
    monkeypatch.setenv("NX_ENV_FILE", str(tmp_path / "env"))
    monkeypatch.setenv("NX_ALLOWED_FILE",
                       str(tmp_path / "nxhome" / "allowed.json"))
    monkeypatch.setenv("NX_SESSIONS_DIR", str(tmp_path / "nxhome" / "sessions"))
    import nx
    importlib.reload(nx)
    nx.st.PROJECT_ROOT = str(tmp_path)
    nx.st.PROJECT_ROOT_IS_GIT = True
    yield nx
    nx.YOLO = False
    nx.APPROVE_LEVEL = None


PREVIEW = "diff --git a/a.py b/a.py\n+x = 2\n-x = 1\n"


# --- the choke point: every _confirm path respects the toggle ----------------

def test_yolo_path_hides_preview_when_collapsed(nx_fresh):
    nx.YOLO = True
    nx.st.TOOL_OUTPUT_VISIBLE = False
    with capture_stdout() as out:
        assert nx._confirm("git commit: fix: x", preview=PREVIEW) is True
    assert "+x = 2" not in out.getvalue()


def test_yolo_path_shows_preview_when_visible(nx_fresh):
    nx.YOLO = True
    nx.st.TOOL_OUTPUT_VISIBLE = True
    with capture_stdout() as out:
        assert nx._confirm("git commit: fix: x", preview=PREVIEW) is True
    assert "+x = 2" in out.getvalue()


def test_prompt_path_hides_preview_but_keeps_the_prompt(nx_fresh, monkeypatch):
    """Collapsed hides the diff but must not swallow the decision point."""
    nx.st.TOOL_OUTPUT_VISIBLE = False
    monkeypatch.setattr(nx, "_assess_risk", lambda action: ("high", "test"))
    prompts = []
    monkeypatch.setattr(nx, "_ask_approval",
                        lambda prompt, allow_hint=None: prompts.append(prompt) or "y")
    with capture_stdout() as out:
        assert nx._confirm("git commit: fix: x", preview=PREVIEW) is True
    assert "+x = 2" not in out.getvalue()
    # the prompt itself always reaches _ask_approval, preview or no preview
    assert prompts and "allow git commit" in prompts[0]


def test_prompt_path_shows_preview_when_visible(nx_fresh, monkeypatch):
    nx.st.TOOL_OUTPUT_VISIBLE = True
    monkeypatch.setattr(nx, "_assess_risk", lambda action: ("high", "test"))
    monkeypatch.setattr(nx, "_ask_approval",
                        lambda prompt, allow_hint=None: "y")
    with capture_stdout() as out:
        assert nx._confirm("git commit: fix: x", preview=PREVIEW) is True
    assert "+x = 2" in out.getvalue()


def test_auto_allow_path_hides_preview_when_collapsed(nx_fresh):
    """APPROVE_LEVEL auto-allow prints the preview after the allow line."""
    nx.APPROVE_LEVEL = "medium"
    nx.st.TOOL_OUTPUT_VISIBLE = False
    with capture_stdout() as out:
        assert nx._confirm("git commit: fix: x", preview=PREVIEW) is True
    text = out.getvalue()
    assert "auto-allow" in text       # the paper-trail line stays
    assert "+x = 2" not in text       # the diff collapses with it


def test_saved_rule_path_hides_preview_when_collapsed(nx_fresh, monkeypatch):
    from nx import _allowed
    _allowed.add_rule("run_bash", "run: pytest*", scope="project")
    nx.st.TOOL_OUTPUT_VISIBLE = False
    with capture_stdout() as out:
        assert nx._confirm("run: pytest -q", preview=PREVIEW) is True
    text = out.getvalue()
    assert "auto-allow [rule:" in text
    assert "+x = 2" not in text


def test_always_allow_path_hides_preview_when_collapsed(nx_fresh, monkeypatch):
    """Pressing `a` at the prompt re-prints the preview — also gated."""
    nx.st.TOOL_OUTPUT_VISIBLE = False
    monkeypatch.setattr(nx, "_assess_risk", lambda action: ("high", "test"))
    monkeypatch.setattr(nx, "_ask_approval",
                        lambda prompt, allow_hint=None: "a")
    with capture_stdout() as out:
        assert nx._confirm("git commit: fix: x", preview=PREVIEW) is True
    assert "+x = 2" not in out.getvalue()


# --- end to end: the reported git_commit leak --------------------------------

@pytest.fixture()
def repo(nx_fresh, tmp_path, monkeypatch):
    """A real git repo (mirrors tests/test_git_commit.py's fixture)."""
    nx, rp = nx_fresh, tmp_path
    monkeypatch.chdir(rp)
    subprocess.run(["git", "init", "-q"], cwd=rp, check=True,
                   capture_output=True)
    for k, v in (("user.email", "nx@example.com"), ("user.name", "nx"),
                 ("commit.gpgsign", "false")):
        subprocess.run(["git", "config", k, v], cwd=rp, check=True,
                       capture_output=True)
    (rp / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=rp, check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=rp, check=True,
                   capture_output=True)
    (rp / "a.py").write_text("x = 2\n", encoding="utf-8")
    return nx, rp


def test_git_commit_yolo_collapsed_leaks_no_diff(repo):
    """The reported bug: YOLO + collapsed, git_commit printed the whole staged
    diff into the tool box. The commit still happens; the diff stays hidden."""
    nx, rp = repo
    nx.YOLO = True
    nx.st.TOOL_OUTPUT_VISIBLE = False
    with capture_stdout() as out:
        result = nx._gitcommit.git_commit(files="a.py", message="fix: bump x")
    assert "[git_commit]" in result
    assert "+x = 2" not in out.getvalue()
    r = subprocess.run(["git", "log", "--oneline", "-1"], cwd=rp,
                       capture_output=True, text=True)
    assert "bump x" in r.stdout


def test_git_commit_yolo_visible_still_shows_diff(repo):
    nx, _ = repo
    nx.YOLO = True
    nx.st.TOOL_OUTPUT_VISIBLE = True
    with capture_stdout() as out:
        nx._gitcommit.git_commit(files="a.py", message="fix: bump x")
    assert "+x = 2" in out.getvalue()


def test_git_commit_prompt_collapsed_leaks_no_diff(repo, monkeypatch):
    """Non-YOLO: the user is prompted, sees only the prompt line, and the diff
    stays hidden unless /tooloutput is expanded."""
    nx, rp = repo
    nx.st.TOOL_OUTPUT_VISIBLE = False
    monkeypatch.setattr(nx, "_ask_approval",
                        lambda prompt, allow_hint=None: "y")
    with capture_stdout() as out:
        result = nx._gitcommit.git_commit(files="a.py", message="fix: bump x")
    assert "[git_commit]" in result
    assert "+x = 2" not in out.getvalue()
    r = subprocess.run(["git", "log", "--oneline", "-1"], cwd=rp,
                       capture_output=True, text=True)
    assert "bump x" in r.stdout
