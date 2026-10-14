#!/usr/bin/env python3
"""0.133.0 feature #2 — the spec-driven build pipeline (`/build`).

Offline: the sub-agent layer is stubbed the way the existing sub-agent tests
fake the model (monkeypatch `nx._subagent._run_sync_agent_body`), git is a real
tmp repo, and the main-model side call of the phase-4 inline fix is a stub
object. The contract this file pins:

1. USER-TRIGGERED ONLY — `/build` is a slash command with no tool schema, so
   the always-on floor in tests/test_initial_context.py must not move.
2. PREFLIGHT REFUSALS (dirty tree, run already in flight, non-git root) leave
   no state: no branch, no run dir.
3. PHASE 0/5 — the work happens on `nx/build-<stamp>`; the starting branch is
   untouched, the feature branch is committed and left checked out.
4. THE GATE — a declined spec aborts: back on the starting branch, feature
   branch deleted, run dir removed.
5. MERGE — a package agent's work (committed in its worktree, or dirty) lands
   on the feature branch; a conflict re-runs the package exactly ONCE.
6. CRITIQUE LADDER — one-file findings are fixed inline, structural findings go
   to exactly one sub-agent, and there is at most ONE re-critique.
"""
import json
import os
import subprocess
import sys
import threading  # noqa: F401  (the fake-thread test patches B.threading)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import nx  # noqa: E402
import nx._state as st  # noqa: E402
from nx import _build as B  # noqa: E402


def _git(cwd, *args):
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                       text=True)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def _git_ok(cwd, *args):
    rc, out, err = _git(cwd, *args)
    assert rc == 0, f"git {args} failed: {err}"
    return out


def _repo(tmp_path, name="repo"):
    repo = tmp_path / name
    repo.mkdir()
    _git_ok(repo, "init", "-q", "-b", "main")
    _git_ok(repo, "config", "user.email", "t@t")
    _git_ok(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text(".nx/\n")
    (repo / "nx").mkdir()
    (repo / "nx" / "mod.py").write_text("x = 1\n")
    _git_ok(repo, "add", "-A")
    _git_ok(repo, "commit", "-qm", "base")
    return repo


def _sandbox(tmp_path, repo, monkeypatch):
    """Point nx's project state at the tmp repo and give the test a session
    home OUTSIDE it (a worktree nested in its own repo is refused)."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(st, "PROJECT_ROOT", str(repo), raising=False)
    monkeypatch.setattr(nx, "SESSION_HOME", str(home), raising=False)
    monkeypatch.setattr(st, "SESSION_HOME", str(home), raising=False)
    monkeypatch.setattr(st, "PROJECT_ROOT_IS_GIT", True, raising=False)
    monkeypatch.setattr(st, "YOLO", False, raising=False)
    monkeypatch.setattr(nx, "NX_SUBAGENTS", True, raising=False)
    return home


SPEC = """# Spec: add a helper

## Problem

No helper exists.

## Approach

Add one.

## Work packages

### alpha-helper

- Files: `nx/mod.py`
- Done means: the helper exists and tests pass
- Verify: `python -c "import nx.mod"`

### beta-doc

- Files: `README.md`
- Done means: the README mentions the helper
- Verify: `grep -q helper README.md`

## Acceptance criteria

- `nx/mod.py` defines the helper
- the README mentions it
"""


def _stub_agents(monkeypatch, calls, spec=SPEC, on_package=None):
    """Fake the sub-agent layer: the spec agent writes SPEC.md (where the
    instruction names it) and package agents edit their package's first file in
    the tree they were given. `calls` records every (kind, kwargs)."""

    def fake(instruction, max_turns, tier="read", scope=None, worktree=None,
             job_id=None):
        low = instruction.lower()
        tree = worktree or str(st.PROJECT_ROOT)
        if "spec writer" in low or "spec file" in low:
            calls.append(("spec", {"scope": scope, "worktree": worktree}))
            import re
            m = re.search(r"(\S*SPEC\.md)", instruction)
            if m:
                path = m.group(1)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(spec)
            return "wrote the spec"
        if "work package" in low:
            calls.append(("package", {"scope": scope, "worktree": worktree,
                                      "instruction": instruction}))
            if on_package is not None:
                on_package(instruction, tree, calls)
                return "done"
            target = os.path.join(tree, "nx", "mod.py")
            with open(target, "a", encoding="utf-8") as f:
                f.write("# package edit\n")
            return "implemented"
        calls.append(("other", {"instruction": instruction, "worktree": worktree}))
        return "done"

    monkeypatch.setattr("nx._subagent._run_sync_agent_body", fake)
    return calls


def _confirm(monkeypatch, answer):
    monkeypatch.setattr(nx, "_confirm", lambda *a, **k: answer, raising=False)


def _no_side_model(monkeypatch):
    """`st.client is None` makes the phase-4 inline fix return None — enough for
    every test that only needs the ladder's shape."""
    monkeypatch.setattr(st, "client", None, raising=False)


def _collect_output(capsys):
    return capsys.readouterr().out


# --- registration trio -------------------------------------------------------

def test_build_is_registered_everywhere():
    from nx._repl import _REPL_COMMANDS
    from nx._repl_settings_cmds import _HELP_COMMANDS, _cmd_build
    from nx._chatbox import SLASH_COMMANDS

    assert _REPL_COMMANDS["/build"] is _cmd_build
    assert "/build" in {c for c, _ in SLASH_COMMANDS}
    assert any(c.startswith("/build") for c, _ in _HELP_COMMANDS)


def test_build_adds_no_tool_schema():
    """`/build` is a slash command only: the model can never fire it, so the
    always-on tool floor is untouched (test_initial_context pins the budget)."""
    assert "build" not in {t["function"]["name"] for t in nx.TOOLS}


def test_cmd_build_without_an_issue_prints_usage(capsys):
    from nx._repl_settings_cmds import _cmd_build
    _cmd_build("/build", None)
    assert "usage: /build" in _collect_output(capsys)


def test_run_build_returns_the_report(capsys, tmp_path, monkeypatch):
    """The handler prints what run_build returns (the final report)."""
    from nx._repl_settings_cmds import _cmd_build
    from nx import _build as mod
    monkeypatch.setattr(mod, "run_build", lambda issue, state=None: "REPORT")
    _cmd_build("/build do the thing", None)
    assert "REPORT" in _collect_output(capsys)


# --- preflight ---------------------------------------------------------------

def test_preflight_refuses_a_dirty_tree(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    (repo / "nx" / "mod.py").write_text("x = 2\n")  # uncommitted
    out = B.run_build("do a thing")
    assert "refused" in out and "commit or stash first" in out
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD")[1] == "main"
    assert not os.path.exists(B._build_root(str(repo)) + "/" + "no-such-run")
    # no run dir was created at all
    root = B._build_root(str(repo))
    assert not os.path.exists(root) or os.listdir(root) == []


def test_preflight_refuses_a_concurrent_run(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    path = B._active_path(str(repo))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "run_id": "build-x", "branch": "nx/build-x"},
                  f)
    out = B.run_build("do a thing")
    assert "already in progress" in out
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD")[1] == "main"
    assert os.path.exists(path)  # the live record is kept


def test_preflight_clears_a_stale_run_record(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    path = B._active_path(str(repo))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pid": 999999, "run_id": "build-dead"}, f)
    assert B._active_run(str(repo)) is None
    assert not os.path.exists(path)


def test_preflight_refuses_with_subagents_off(tmp_path, monkeypatch):
    """`/build` IS a sub-agent harness: with them disabled it must refuse up
    front rather than run a degraded pipeline"""
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    monkeypatch.setattr(nx, "NX_SUBAGENTS", False)
    out = B.run_build("do a thing")
    assert "sub-agents are off" in out
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD")[1] == "main"


def test_preflight_refuses_a_non_git_root(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(plain), raising=False)
    out = B.run_build("do a thing")
    assert "not a git repository" in out


# --- abort at the gate -------------------------------------------------------

def test_declined_gate_cleans_up_and_returns_to_the_start_branch(
        tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    calls = _stub_agents(monkeypatch, [])
    _confirm(monkeypatch, False)
    out = B.run_build("do a thing")
    assert "aborted at the spec gate" in out
    # back on the starting branch …
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD")[1] == "main"
    # … no feature branch survives …
    rc, branches, _ = _git(repo, "branch", "--list", "nx/build-*")
    assert branches == "", branches
    # … the run dir is removed …
    root = B._build_root(str(repo))
    assert not os.path.isdir(root) or not os.listdir(root)
    # … nothing was committed, and the spec agent did run.
    assert _git(repo, "status", "--porcelain")[1] == ""
    assert len(_git(repo, "log", "--oneline")[1].splitlines()) == 1
    assert [c[0] for c in calls] == ["spec"]


def test_spec_md_is_written_and_printed_before_the_gate(
        tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    seen = {}

    def fake_confirm(action, preview=None):
        seen["action"] = action
        seen["preview"] = preview
        return False

    monkeypatch.setattr(nx, "_confirm", fake_confirm, raising=False)
    _stub_agents(monkeypatch, [])
    out = B.run_build("do a thing")
    assert "### alpha-helper" in seen["preview"]
    assert "## Work packages" in seen["preview"]
    assert "aborted at the spec gate" in out
    assert "spec gate" in out


# --- the happy path: branch, packages, commit --------------------------------

def test_happy_path_branches_packages_and_commits(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "README.md").write_text("base\n")
    _git_ok(repo, "add", "-A")
    _git_ok(repo, "commit", "-qm", "readme")
    _sandbox(tmp_path, repo, monkeypatch)
    _no_side_model(monkeypatch)
    calls = _stub_agents(monkeypatch, [])
    _confirm(monkeypatch, True)
    out = B.run_build("add a helper\nsecond line of the issue")

    assert "nx/build-" in out
    # the feature branch is checked out and committed
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")[1]
    assert branch.startswith("nx/build-")
    assert _git(repo, "status", "--porcelain")[1] == ""
    # main never moved: the package commits live only on the feature branch
    assert _git(repo, "rev-parse", "main")[1] != _git(repo, "rev-parse", "HEAD")[1]
    assert len(_git(repo, "log", "--oneline", "main..HEAD")[1].splitlines()) >= 2
    # both packages ran, each with its own scope
    scopes = [c[1].get("scope") for c in calls if c[0] == "package"]
    assert len(scopes) == 2
    assert ["nx/mod.py"] in scopes and ["README.md"] in scopes
    # the packages' edits are on the branch
    blob = _git_ok(repo, "show", "HEAD:nx/mod.py")
    assert "package edit" in blob
    assert "2/2 done" in out
    assert "Nothing was pushed" in out


def test_parallel_packages_land_in_their_own_worktrees(tmp_path, monkeypatch):
    """Yolo + worktree isolation: each package agent gets its own checkout, and
    the orchestrator merges its branch back onto the feature branch."""
    repo = _repo(tmp_path)
    (repo / "README.md").write_text("base\n")
    _git_ok(repo, "add", "-A")
    _git_ok(repo, "commit", "-qm", "readme")
    _sandbox(tmp_path, repo, monkeypatch)
    monkeypatch.setattr(st, "YOLO", True, raising=False)
    _no_side_model(monkeypatch)
    _confirm(monkeypatch, True)
    calls = _stub_agents(monkeypatch, [])
    B.run_build("add a helper")

    trees = [c[1].get("worktree") for c in calls if c[0] == "package"]
    assert all(trees) and len(set(trees)) == 2
    for t in trees:
        assert os.path.isdir(t)
    assert _git(repo, "status", "--porcelain")[1] == ""
    assert "package edit" in _git_ok(repo, "show", "HEAD:nx/mod.py")


# --- merge orchestration -----------------------------------------------------

def _pkg(name="alpha", paths=None, wt=None, base=None):
    p = B._BuildPackage(name, paths or ["nx/mod.py"], "")
    p.worktree = wt
    p.base_head = base
    return p


def test_merge_package_lands_a_dirty_worktree(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    home = _sandbox(tmp_path, repo, monkeypatch)
    # a worktree the agent left dirty (uncommitted edit)
    wt = str(home / "wt")
    _git_ok(repo, "worktree", "add", "--detach", wt, "HEAD")
    with open(os.path.join(wt, "nx", "mod.py"), "a", encoding="utf-8") as f:
        f.write("# dirty work\n")
    run = B._BuildRun(str(repo), "issue", False, False, True)
    run.start_branch = "main"
    pkg = _pkg(wt=wt, base=_git_ok(wt, "rev-parse", "HEAD"))
    run.packages = [pkg]
    B._merge_package(run, pkg)
    assert pkg.status == "done"
    assert "# dirty work" in _git_ok(repo, "show", "HEAD:nx/mod.py")


def test_merge_package_merges_worktree_commits(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    home = _sandbox(tmp_path, repo, monkeypatch)
    wt = str(home / "wt")
    _git_ok(repo, "worktree", "add", "--detach", wt, "HEAD")
    with open(os.path.join(wt, "nx", "mod.py"), "a", encoding="utf-8") as f:
        f.write("# committed work\n")
    _git_ok(wt, "add", "-A")
    _git_ok(wt, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "pkg")
    run = B._BuildRun(str(repo), "issue", False, False, True)
    run.start_branch = "main"
    pkg = _pkg(wt=wt, base=_git_ok(repo, "rev-parse", "HEAD"))
    run.packages = [pkg]
    B._merge_package(run, pkg)
    assert pkg.status == "done"
    assert "# committed work" in _git_ok(repo, "show", "HEAD:nx/mod.py")


def test_merge_conflict_retries_once_then_stops(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    home = _sandbox(tmp_path, repo, monkeypatch)
    wt = str(home / "wt")
    _git_ok(repo, "worktree", "add", "--detach", wt, "HEAD")
    # both sides change the same line → a real merge conflict
    with open(os.path.join(wt, "nx", "mod.py"), "w", encoding="utf-8") as f:
        f.write("x = 'worktree'\n")
    _git_ok(wt, "add", "-A")
    _git_ok(wt, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "pkg")
    with open(os.path.join(repo, "nx", "mod.py"), "w", encoding="utf-8") as f:
        f.write("x = 'feature'\n")
    _git_ok(repo, "add", "-A")
    _git_ok(repo, "commit", "-qm", "feature")
    run = B._BuildRun(str(repo), "issue", False, False, True)
    run.start_branch = "main"
    pkg = _pkg(wt=wt, base=_git_ok(repo, "rev-parse", "HEAD~1"))
    run.packages = [pkg]
    seen = []
    monkeypatch.setattr(B, "_run_agent",
                        lambda run_, instruction, **kw: seen.append(instruction))
    B._merge_package(run, pkg)
    assert pkg.retries == 1
    assert len(seen) == 1 and "conflicted" in seen[0]
    # the retry's agent left nothing → the package is failed, the branch unsullied
    assert pkg.status == "failed"
    assert _git(repo, "status", "--porcelain")[1] == ""
    assert _git(repo, "rev-parse", "--verify", "MERGE_HEAD")[0] != 0


def test_retry_package_refuses_a_second_retry(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    run = B._BuildRun(str(repo), "issue", False, False, True)
    pkg = _pkg()
    pkg.retries = 1
    monkeypatch.setattr(B, "_run_agent",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    B._retry_package(run, pkg, "conflict")
    assert pkg.status == "failed" and pkg.retries == 2


# --- the critique ladder -----------------------------------------------------

def test_finding_classification():
    assert B._finding_is_simple("major — `nx/_turn.py:12` — the loop exits early")
    assert not B._finding_is_simple(
        "major — `nx/_turn.py:12` and `nx/_subagent.py:90` — retry design is wrong")
    assert not B._finding_is_simple(
        "critical — `nx/_turn.py:12` — this refactor breaks the design")


def test_parse_findings_and_all_clear():
    assert B._parse_findings("No findings above the floor.\nchecked: x") == []
    got = B._parse_findings("minor — `a.py:1` — thing\nnoise line\n"
                            "**major** — `b.py:2` — other")
    assert len(got) == 2


def test_ladder_simple_finding_is_inline_structural_is_an_agent(
        tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    run = B._BuildRun(str(repo), "issue", False, False, False)
    run.spec = SPEC
    simple = "major — `nx/mod.py:1` — x should be 2 — set it to 2"
    structural = ("major — `nx/mod.py:1` `nx/other.py:2` — the interface between "
                  "these modules is wrong — redesign it")

    monkeypatch.setattr(B, "_ask_main_model",
                        lambda run_, finding: {"path": "nx/mod.py",
                                               "content": "x = 2\n"})
    agent_calls = []
    monkeypatch.setattr(B, "_run_agent",
                        lambda run_, instruction, **kw:
                        agent_calls.append(kw.get("scope")))
    monkeypatch.setattr(B, "_phase_critique", lambda run_: [])
    remaining = B._phase_fix(run, [simple, structural])
    assert remaining == []
    assert (repo / "nx" / "mod.py").read_text() == "x = 2\n"
    assert agent_calls == [["nx/mod.py", "nx/other.py"]]
    # both fixes were committed on the branch
    assert _git(repo, "status", "--porcelain")[1] == ""
    assert _git(repo, "log", "--oneline", "-1")[1].endswith("critique fix (inline)")


def test_inline_fix_refuses_a_path_outside_the_repo(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    run = B._BuildRun(str(repo), "issue", False, False, False)
    monkeypatch.setattr(B, "_ask_main_model",
                        lambda run_, finding: {"path": "../escape.py",
                                               "content": "boom\n"})
    assert B._inline_fix(run, "minor — `x.py:1` — nope") is False


def test_fix_failure_stops_without_a_re_critique(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    run = B._BuildRun(str(repo), "issue", False, False, False)
    run.spec = SPEC
    rounds = []
    monkeypatch.setattr(B, "_inline_fix", lambda run_, f: False)
    monkeypatch.setattr(B, "_phase_critique",
                        lambda run_: rounds.append(1) or [])
    remaining = B._phase_fix(run, ["major — `a.py:1` — broke"])
    assert remaining == ["major — `a.py:1` — broke"]
    assert rounds == []  # a fix that did not land is reported, not re-reviewed


def test_phase_critique_caps_at_one_re_critique(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    monkeypatch.setattr(nx, "NX_SUBAGENTS", True, raising=False)
    run = B._BuildRun(str(repo), "issue", False, False, False)
    run.spec = SPEC
    seen = []

    def fake_critique(focus, paths=None, async_=False):
        seen.append(focus)
        return "major — `nx/mod.py:1` — still broken"

    monkeypatch.setattr("nx._subagent.critique", fake_critique)
    monkeypatch.setattr(B, "_inline_fix", lambda run_, f: True)
    remaining = B._phase_fix(run, B._phase_critique(run))
    assert run.critique_rounds == 2          # phase 3 + exactly one re-critique
    assert len(seen) == 2
    assert "acceptance criteria" in seen[0]  # focus = the spec's criteria
    assert remaining == ["major — `nx/mod.py:1` — still broken"]
    # a third round is refused
    assert B._phase_critique(run) == []
    assert len(seen) == 2


def test_critique_is_skipped_without_subagents(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    monkeypatch.setattr(nx, "NX_SUBAGENTS", False)
    run = B._BuildRun(str(repo), "issue", False, False, False)
    run.spec = SPEC
    assert B._phase_critique(run) == []
    assert run.critique_rounds == 0


def test_acceptance_criteria_parsing():
    got = B._acceptance_criteria(SPEC)
    assert got == ["`nx/mod.py` defines the helper", "the README mentions it"]


def test_merge_package_commits_a_shared_tree_edit(tmp_path, monkeypatch):
    """No worktree (isolation off): the agent edited the feature branch's own
    working tree, and the orchestrator commits it there."""
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    run = B._BuildRun(str(repo), "issue", False, False, False)
    run.start_branch = "main"
    pkg = _pkg()
    run.packages = [pkg]
    (repo / "nx" / "mod.py").write_text("x = 99\n")
    B._merge_package(run, pkg)
    assert pkg.status == "done"
    assert _git(repo, "status", "--porcelain")[1] == ""
    assert "x = 99" in _git_ok(repo, "show", "HEAD:nx/mod.py")


def test_run_packages_spawns_threads_only_when_parallel(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    home = _sandbox(tmp_path, repo, monkeypatch)
    monkeypatch.setattr(B, "_package_agent", lambda run_, pkg, quiet: None)
    monkeypatch.setattr(B, "_merge_packages", lambda run_: None)

    def boom(*a, **k):
        raise AssertionError("no threads in sequential mode")

    monkeypatch.setattr(B.threading, "Thread", boom)
    seq = B._BuildRun(str(repo), "issue", False, False, True)
    seq.packages = [_pkg("a"), _pkg("b")]
    B._run_packages(seq)  # parallel=False → no threads

    started = []

    class FakeThread:
        def __init__(self, target=None, args=(), **kw):
            self._t, self._a = target, args

        def start(self):
            started.append(1)
            self._t(*self._a)

        def join(self):
            pass

    monkeypatch.setattr(B.threading, "Thread", FakeThread)
    par = B._BuildRun(str(repo), "issue", False, True, True)
    par.packages = [_pkg("a"), _pkg("b")]
    B._run_packages(par)
    assert len(started) == 2


# --- spec parsing ------------------------------------------------------------

def test_spec_scope_is_repo_relative_or_disabled(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    run = B._BuildRun(str(repo), "issue", False, False, False)
    # gitignored .nx (the repo's own .gitignore) → the scope is repo-relative
    assert B._spec_scope(run) == [".nx/build/" + run.run_id + "/SPEC.md"]
    # a run dir outside the repo cannot be named by a repo-relative glob
    run.run_dir = str(tmp_path / "elsewhere")
    run.spec_path = os.path.join(run.run_dir, "SPEC.md")
    assert B._spec_scope(run) is None


def test_parse_spec_reads_packages():
    run = B._BuildRun("/tmp/x", "issue", False, False, False)
    run.spec = SPEC
    pkgs = B._parse_spec(run)
    assert [p.name for p in pkgs] == ["alpha-helper", "beta-doc"]
    assert pkgs[0].paths == ["nx/mod.py"]
    assert "import nx.mod" in pkgs[0].verify


def test_parse_spec_degrades_to_one_package():
    run = B._BuildRun("/tmp/x", "issue", False, False, False)
    run.spec = "Do the thing in `nx/mod.py` and `tests/test_mod.py`."
    pkgs = B._parse_spec(run)
    assert len(pkgs) == 1
    assert pkgs[0].paths == ["nx/mod.py", "tests/test_mod.py"]


# --- abort safety ------------------------------------------------------------

def test_keyboard_interrupt_aborts_cleanly(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _sandbox(tmp_path, repo, monkeypatch)
    _confirm(monkeypatch, True)
    _stub_agents(monkeypatch, [])
    monkeypatch.setattr(B, "_run_packages",
                        lambda run: (_ for _ in ()).throw(KeyboardInterrupt()))
    out = B.run_build("do a thing")
    assert "interrupted" in out and "ABORTED.md" in out
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD")[1] == "main"
    # the run dir survives with its abort note
    root = B._build_root(str(repo))
    runs = [d for d in os.listdir(root) if d.startswith("build-")]
    assert runs and os.path.exists(os.path.join(root, runs[0], "ABORTED.md"))


# --- _ask_main_model via the shared side call (0.156.3) ----------------------

class _SideResp:
    def __init__(self, text):
        import types as _t
        self.choices = [_t.SimpleNamespace(
            message=_t.SimpleNamespace(content=text))]

    def model_dump(self):
        return {}


class _SideClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    class _Chat:
        def __init__(self, outer):
            self._outer = outer

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kw):
                self._outer.calls.append(dict(kw))
                item = self._outer.replies.pop(0)
                if isinstance(item, Exception):
                    raise item
                return _SideResp(item)

        @property
        def completions(self):
            return self._Completions(self._outer)

    @property
    def chat(self):
        return self._Chat(self)


def _ask(monkeypatch, replies):
    monkeypatch.setattr(st, "ACTIVE", None, raising=False)
    monkeypatch.setattr(st, "MODEL", "test-model", raising=False)
    c = _SideClient(replies)
    monkeypatch.setattr(st, "client", c, raising=False)
    run = B._BuildRun("/nonexistent", "issue", False, False, False)
    run.spec = "SPEC BODY"
    return c, run


def test_ask_main_model_accepts_prose_wrapped_json(monkeypatch):
    # brace-slice tolerance is load-bearing: the fix model answers in prose
    c, run = _ask(monkeypatch, ['Sure:\n{"path": "nx/mod.py", "content": "x = 2"}\nDone!'])
    obj = B._ask_main_model(run, "a finding")
    assert obj == {"path": "nx/mod.py", "content": "x = 2"}
    call = c.calls[0]
    assert call["stream"] is False
    assert call["timeout"] == 120
    assert "a finding" in call["messages"][1]["content"]
    assert "SPEC BODY" in call["messages"][1]["content"]


def test_ask_main_model_failure_is_none_with_note(monkeypatch):
    c, run = _ask(monkeypatch, [RuntimeError("boom")])
    assert B._ask_main_model(run, "a finding") is None
    assert any("inline fix" in n for n in run.notes)


def test_ask_main_model_non_json_is_none(monkeypatch):
    _ask(monkeypatch, ["I cannot help with that."])
    run = B._BuildRun("/nonexistent", "issue", False, False, False)
    assert B._ask_main_model(run, "a finding") is None


def test_ask_main_model_no_source_is_none(monkeypatch):
    monkeypatch.setattr(st, "client", None, raising=False)
    run = B._BuildRun("/nonexistent", "issue", False, False, False)
    assert B._ask_main_model(run, "a finding") is None
