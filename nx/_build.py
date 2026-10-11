"""Spec-driven build pipeline (nx #2): `/build <issue text>`.

A user-triggered, five-phase pipeline that turns one issue paragraph into a
reviewable feature branch:

  phase 0  branch    create `nx/build-<YYMMDD-HHMMSS>` from the current HEAD
                     (the pipeline's work never touches the starting branch)
  phase 1  mini-spec ONE implement-tier sub-agent writes SPEC.md into the run
                     dir (`<repo>/.nx/build/<run-id>/`, gitignored): problem,
                     approach, 2-3 work packages (files / done-means / verify),
                     acceptance criteria. The spec is PRINTED and the user
                     answers y/n — the single gate before any code is written.
  phase 2  packages  2-3 write-tier sub-agents, one per package, each with its
                     own scope (and its own git worktree when the session is in
                     yolo mode, so parallel writers cannot trample each other);
                     the orchestrator commits each package's work into the
                     feature branch (a merge conflict re-runs that package ONCE
                     with the conflict as its spec, then stops).
  phase 3  critique  the existing scoped reviewer (`nx._subagent.critique`),
                     focus = the spec's acceptance criteria.
  phase 4  correct   a one-file mechanical finding is fixed inline by the main
                     model; a structural finding gets ONE follow-up write-tier
                     agent with a mini-spec written from the finding. At most
                     ONE re-critique; anything still open is reported, never
                     looped.
  phase 5  handoff   commit the feature branch, print branch / packages /
                     critique rounds / remaining findings / diff stat, and leave
                     the feature branch checked out. Never pushed, never merged
                     into the starting branch.

Deliberately a slash command ONLY — there is no `build` tool schema, so the
model cannot fire a heavy multi-agent harness on its own (0 schema tokens; the
always-on budget in tests/test_initial_context.py must not move). The model may
*suggest* `/build` in prose when a task looks complex; the user decides.

State is the run dir's files plus one in-memory object per invocation; there is
no new storage, no sqlite, and no new dependency. `NX_BUILD_VERBOSE=1` prints a
progress line per step (default: the spec, the gate, and the final report).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time

from . import _state as st
from ._ansi import DIM, RESET
from ._env import _env_int

_MAX_PACKAGES = 3          # more than three packages is a design smell, not a plan
_MAX_CRITIQUE_ROUNDS = 2   # phase 3, plus at most one re-critique in phase 4
_PACKAGE_RETRIES = 1       # one conflict retry per package, then stop and report
_SPEC_AGENT_TURNS = 60


# --------------------------------------------------------------------------
# small git / fs helpers (no new deps, no state beyond the run object)
# --------------------------------------------------------------------------

def _git(repo: str, *args: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run git in `repo`; return (returncode, stdout, stderr). Never raises —
    a git failure is a reported condition, not a crash inside a REPL command."""
    try:
        r = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                           text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def _repo_root() -> str:
    return os.path.realpath(st.PROJECT_ROOT or os.getcwd())


def _is_git(repo: str) -> bool:
    rc, out, _ = _git(repo, "rev-parse", "--is-inside-work-tree")
    return rc == 0 and out.strip() == "true"


def _dirty(repo: str) -> bool:
    rc, out, _ = _git(repo, "status", "--porcelain")
    return rc == 0 and bool(out.strip())


def _current_branch(repo: str) -> str | None:
    rc, out, _ = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    return out if rc == 0 and out and out != "HEAD" else None


def _head_of(repo: str) -> str | None:
    rc, out, _ = _git(repo, "rev-parse", "HEAD")
    return out if rc == 0 and out else None


def _branch_exists(repo: str, name: str) -> bool:
    rc, _, _ = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
    return rc == 0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", os.path.realpath(path)).strip("-")[-40:]


def _build_root(repo: str) -> str:
    """Where a run's scratch dir lives. Inside the repo's `.nx/build` when that
    path is gitignored (the project convention — a pipeline must never dirty the
    tree it is about to commit), else under SESSION_HOME so the repo stays clean
    either way."""
    rc, _, _ = _git(repo, "check-ignore", "-q", os.path.join(".nx", "build"))
    if rc == 0:
        return os.path.join(repo, ".nx", "build")
    home = st.SESSION_HOME or os.path.expanduser("~/.nx")
    return os.path.join(home, "build", _slug(repo))


def _active_path(repo: str) -> str:
    return os.path.join(_build_root(repo), "ACTIVE")


def _active_run(repo: str) -> dict | None:
    """The record of a run still in flight for this repo, or None. A record
    whose pid is gone is stale (crashed process) and is cleared here."""
    path = _active_path(repo)
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    pid = rec.get("pid")
    if isinstance(pid, int) and not _pid_alive(pid):
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    return rec


def _mark_active(run: _BuildRun) -> None:
    try:
        os.makedirs(os.path.dirname(_active_path(run.repo)), exist_ok=True)
        with open(_active_path(run.repo), "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "run_id": run.run_id,
                       "branch": run.branch, "started": run.started}, f)
    except OSError:
        pass


def _clear_active(run: _BuildRun) -> None:
    if getattr(run, "owns_active", False):
        try:
            os.remove(_active_path(run.repo))
        except OSError:
            pass


# --------------------------------------------------------------------------
# run state
# --------------------------------------------------------------------------

def _issue_first_line(issue: str) -> str:
    """The issue's first line, for branch/commit/report summaries."""
    lines = (issue or "").strip().splitlines()
    return (lines[0] if lines else "empty issue")[:60]


class _BuildPackage:
    """One work package: the scope, the agent's answer, and what the
    orchestrator did with it (merged / retried / failed)."""

    __slots__ = ("name", "paths", "verify", "job_id", "worktree", "base_head",
                 "answer", "status", "note", "retries")

    def __init__(self, name: str, paths: list[str], verify: str) -> None:
        self.name = name
        self.paths = paths
        self.verify = verify
        self.job_id = ""
        self.worktree: str | None = None
        self.base_head: str | None = None
        self.answer = ""
        self.status = "pending"
        self.note = ""
        self.retries = 0


class _BuildRun:
    """One pipeline invocation's state. Files in the run dir are the durable
    half (SPEC.md, the agent reports, ABORTED.md); this object is the rest."""

    __slots__ = ("repo", "issue", "run_id", "branch", "start_branch", "base_head",
                 "run_dir", "spec_path", "spec", "packages", "critique_rounds",
                 "critique_texts", "remaining", "notes", "aborted", "started",
                 "verbose", "parallel", "use_worktrees", "owns_active", "threads")

    def __init__(self, repo: str, issue: str, verbose: bool, parallel: bool,
                 use_worktrees: bool) -> None:
        self.repo = repo
        self.issue = issue
        self.run_id = "build-" + time.strftime("%y%m%d-%H%M%S")
        self.branch = f"nx/{self.run_id}"
        self.start_branch: str | None = None
        self.base_head: str | None = None
        self.run_dir = os.path.join(_build_root(repo), self.run_id)
        self.spec_path = os.path.join(self.run_dir, "SPEC.md")
        self.spec = ""
        self.packages: list[_BuildPackage] = []
        self.critique_rounds = 0
        self.critique_texts: list[str] = []
        self.remaining: list[str] = []
        self.notes: list[str] = []
        self.aborted = False
        self.started = time.time()
        self.verbose = verbose
        self.parallel = parallel
        self.use_worktrees = use_worktrees
        self.owns_active = False
        # Live package worker threads (parallel mode). Kept so an abort can wait
        # for them before moving HEAD out from under a still-writing agent.
        self.threads: list[threading.Thread] = []


def _step(run: _BuildRun, msg: str) -> None:
    if run.verbose:
        print(f"{DIM}  {msg}{RESET}", flush=True)


def _note(run: _BuildRun, msg: str) -> None:
    run.notes.append(msg)
    _step(run, msg)


# --------------------------------------------------------------------------
# phase 0 — branch
# --------------------------------------------------------------------------

def _create_branch(run: _BuildRun) -> str | None:
    """Create the feature branch from the current HEAD and check it out.
    Returns an error string, or None on success."""
    run.start_branch = _current_branch(run.repo)
    run.base_head = _head_of(run.repo)
    if run.base_head is None:
        return "could not read HEAD (is this a repo with a commit?)"
    name = run.branch
    n = 1
    while _branch_exists(run.repo, name):
        n += 1
        name = f"{run.branch}-{n}"
    rc, _, err = _git(run.repo, "checkout", "-b", name, "HEAD")
    if rc != 0:
        return f"could not create branch {name}: {err or 'git checkout failed'}"
    run.branch = name
    return None


def _return_to_start(run: _BuildRun) -> str | None:
    """Put the working tree back on the branch the pipeline started from.
    Returns an error string when it could not (detached HEAD, checkout failure)
    — the caller tells the user rather than pretending the abort was clean."""
    if not run.start_branch:
        return None
    rc, _, err = _git(run.repo, "checkout", run.start_branch)
    if rc != 0:
        return f"could not return to {run.start_branch}: {err or 'checkout failed'}"
    return None


# --------------------------------------------------------------------------
# phase 1 — the mini-spec
# --------------------------------------------------------------------------

def _spec_instruction(run: _BuildRun) -> str:
    return (
        "Turn this issue into a SPEC FILE for a multi-agent build pipeline.\n\n"
        f"ISSUE:\n{run.issue}\n\n"
        f"Write the spec to exactly this path: {run.spec_path}\n"
        "(create the directory if needed). It is the ONLY file you write.\n\n"
        "Do not implement anything: read just enough of the repository to name "
        "real files. The spec must contain, in this order:\n"
        "## Problem\n## Approach\n## Work packages\n"
        "## Acceptance criteria\n\n"
        "Under '## Work packages' write 2-3 packages as `### <kebab-name>` "
        "headings, each with exactly these three bullet lines:\n"
        "- Files: `path/a.py` `path/b.py`\n"
        "- Done means: one sentence a reviewer can check\n"
        "- Verify: `command` (a real command, runnable from the repo root)\n\n"
        "Packages must be independently implementable: no package depends on "
        "another's edits. Keep the whole file under 120 lines. End with an "
        "'## Acceptance criteria' list of 3-6 checkable bullets."
    )


def _parse_spec(run: _BuildRun) -> list[_BuildPackage]:
    """Packages from the spec's `### <name>` headings (Files/Done means/Verify
    bullets). A spec that ignored the format degrades to ONE package whose
    scope is every path the spec names — never an empty pipeline."""
    text = run.spec
    blocks = re.split(r"^###\s+", text, flags=re.MULTILINE)[1:]
    packages: list[_BuildPackage] = []
    for block in blocks:
        lines = block.splitlines()
        raw = lines[0].strip().strip("`#").strip()
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")[:40]
        body = "\n".join(lines[1:])
        files_m = re.search(r"^\s*[-*]\s*Files?\s*:\s*(.+)$", body,
                            re.MULTILINE | re.IGNORECASE)
        verify_m = re.search(r"^\s*[-*]\s*Verify\s*:\s*(.+)$", body,
                             re.MULTILINE | re.IGNORECASE)
        paths = _paths_in(files_m.group(1)) if files_m else []
        verify = _first_command(verify_m.group(1)) if verify_m else ""
        packages.append(_BuildPackage(name or f"package-{len(packages) + 1}",
                                      paths, verify))
        if len(packages) >= _MAX_PACKAGES:
            break
    if packages:
        return packages
    # No `###` headings: one package scoped to the paths the spec mentions.
    return [_BuildPackage("package-1", _paths_in(text), "")]


def _paths_in(text: str) -> list[str]:
    """Backticked tokens that look like repo paths, plus bare path-shaped words,
    deduped, order kept."""
    out: list[str] = []
    toks = (re.findall(r"`([^`\n]+)`", text)
            + re.findall(r"(?<![\w/])([\w./-]+\.(?:py|md|toml|cfg|ini|json|yaml"
                         r"|yml|sh|txt))", text))
    for tok in toks:
        tok = tok.strip().strip(".,;:")
        if not tok or " " in tok or tok.startswith(("http", "-")):
            continue
        if tok not in out:
            out.append(tok)
    return out[:20]


def _first_command(text: str) -> str:
    m = re.search(r"`([^`\n]+)`", text)
    return (m.group(1) if m else text).strip()


def _spec_scope(run: _BuildRun) -> list[str] | None:
    """The spec agent's write scope. It is the one file that may be written,
    expressed relative to the repo so the scope gate can match it — and None
    when the run dir lives outside the repo (a non-gitignored `.nx`), since a
    repo-relative glob could never match an absolute path out there."""
    rel = os.path.relpath(run.spec_path, run.repo)
    if rel.startswith(".."):
        return None
    return [rel.replace(os.sep, "/")]


def _confirm_spec(run: _BuildRun) -> bool:
    """The ONE user gate: print the spec, ask y/n. Esc (nx's `_EscToChat`) and
    `n` both mean abort. Tests monkeypatch `nx._confirm`."""
    import nx as _m
    try:
        return bool(_m._confirm(
            f"build {run.run_id}: implement these packages?", preview=run.spec))
    except _m._EscToChat:
        return False


# --------------------------------------------------------------------------
# phase 2 — parallel implementers
# --------------------------------------------------------------------------

def _run_agent(run: _BuildRun, instruction: str, *, tier: str, scope,
               worktree: str | None, max_turns: int, quiet: bool,
               system_prompt: str | None = None) -> str:
    """Run one sub-agent through the shared sub-loop (lazy module access so
    tests can patch `nx._subagent._run_sync_agent_body`, exactly the way the
    existing sub-agent tests fake the model).

    The phase's satellite prompt (nx/_prompts.py) rides the head of the
    instruction: the sub-loop picks its system prompt from the TIER, and the
    tier is a shared contract other callers rely on — a pipeline phase must not
    quietly change what `tier="implement"` means for everyone else."""
    from . import _subagent as S
    if system_prompt:
        instruction = f"{system_prompt}\n\n---\n\n{instruction}"
    job_id = S._new_job_id()
    slots = S._ASYNC_QUIET_SLOTS
    prev = getattr(slots, "quiet", False)
    slots.quiet = quiet
    try:
        return S._run_sync_agent_body(instruction, max_turns, tier=tier,
                                      scope=scope, worktree=worktree,
                                      job_id=job_id)
    finally:
        slots.quiet = prev


def _phase_prompt(name: str) -> str:
    from . import _prompts as P
    return getattr(P, name, "")


def _package_instruction(run: _BuildRun, pkg: _BuildPackage) -> str:
    files = ", ".join(f"`{p}`" for p in pkg.paths) or "(the spec names no files)"
    verify = (f"`{pkg.verify}`" if pkg.verify
              else "the project's test command for the files you touched")
    return (
        "You are implementing ONE work package of a spec-driven build.\n\n"
        f"ISSUE (context only — the package below is your contract):\n"
        f"{run.issue}\n\n"
        f"PACKAGE: {pkg.name}\n"
        f"Files you may touch: {files}\n"
        f"Verify with: {verify}\n\n"
        "SPEC EXCERPT (this package's section):\n"
        f"{_package_excerpt(run.spec, pkg.name)}\n\n"
        "Implement ONLY this package. Do not touch files outside your scope, "
        "do not restructure other packages' code, and do not 'improve' "
        "unrelated code. Verify with the package's verify command (or the "
        "narrowest test command covering your files) before you report. Then "
        "report: files changed with what each edit did, the exact verify "
        "command and its result, and any deviation. If the package cannot be "
        "implemented as written, say so and stop — a wrong implementation is "
        "worse than a clear refusal."
    )


def _package_excerpt(spec: str, name: str) -> str:
    """The `### <name>` block from the spec, or the whole spec when the heading
    cannot be located (a short spec is better context than none)."""
    m = re.search(rf"^###\s+.*{re.escape(name)}.*$", spec, re.MULTILINE)
    if not m:
        return spec[:4000]
    nxt = re.search(r"^###\s+", spec[m.end():], re.MULTILINE)
    end = m.end() + nxt.start() if nxt else len(spec)
    return spec[m.start():end].strip()[:4000]


def _create_pkg_worktree(run: _BuildRun, pkg: _BuildPackage) -> str | None:
    """A detached worktree for this package, from the feature branch's current
    tip. Returns the path, or None when isolation is off/unavailable (the
    package then runs in the shared tree — sequential mode only)."""
    if not run.use_worktrees:
        return None
    from ._worktree import create_worktree, sweep_stale_worktrees
    sweep_stale_worktrees()
    job_id = pkg.job_id or f"build-{pkg.name}-{run.run_id[-6:]}"
    if pkg.retries:
        job_id = f"{job_id}-r{pkg.retries}"
    pkg.job_id = job_id
    wt, err = create_worktree(job_id)
    if wt is None:
        _note(run, f"  {pkg.name}: worktree isolation failed ({err}) — running "
                   "in the shared tree")
        return None
    pkg.base_head = _head_of(wt)
    return wt


def _package_agent(run: _BuildRun, pkg: _BuildPackage, quiet: bool) -> None:
    """One package's agent run: its own worktree (when enabled), its own scope.
    Never raises — a failed package is a reported status, not a dead pipeline.
    In parallel mode the caller has already created the worktree (so the repo's
    worktree lock is never touched from two threads); `pkg.worktree is None`
    here therefore means isolation is off, not "create it now"."""
    try:
        if pkg.worktree is None and not run.parallel:
            pkg.worktree = _create_pkg_worktree(run, pkg)
        answer = _run_agent(run, _package_instruction(run, pkg), tier="implement",
                            scope=pkg.paths or None, worktree=pkg.worktree,
                            max_turns=_env_int("NX_BUILD_PACKAGE_TURNS", 80),
                            quiet=quiet,
                            system_prompt=_phase_prompt("BUILD_PACKAGE_SYSTEM"))
        pkg.answer = answer or ""
        if isinstance(answer, str) and answer.startswith("ERROR"):
            pkg.status = "failed"
            pkg.note = answer[:400]
        _write_package_report(run, pkg)
    except Exception as e:  # noqa: BLE001 — one package must not kill the run
        pkg.status = "failed"
        pkg.note = f"{type(e).__name__}: {e}"


def _write_package_report(run: _BuildRun, pkg: _BuildPackage) -> None:
    d = os.path.join(run.run_dir, "packages")
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{pkg.name}.md"), "w", encoding="utf-8") as f:
            f.write(f"# Package {pkg.name}\n\n- status: {pkg.status}\n"
                    f"- worktree: {pkg.worktree or '(shared tree)'}\n\n"
                    f"## Answer\n\n{pkg.answer.strip()}\n")
    except OSError:
        pass


def _run_packages(run: _BuildRun) -> None:
    """Phase 2. Parallel only where it is honest: a yolo session (nobody is
    waiting to approve a diff, so a quiet worker thread cannot auto-deny an
    edit) with worktree isolation available. Anywhere else the packages run one
    after another on the parent thread so every write hits the normal Y/n gate.

    Concurrency note: the sub-agent LAYER has no joinable job handle — an async
    spawn returns a job id and its answer is only delivered to the parent's turn
    loop as a [Runtime note]. So the orchestrator joins its own threads instead
    of the sub-agent registry: each thread runs the shared synchronous sub-loop
    (which the async worker also runs) with isolation on, so the observable
    contract — one agent per package, landing in its own checkout — is the
    same."""
    pkgs = run.packages
    if run.parallel and run.use_worktrees and len(pkgs) > 1:
        _step(run, f"phase 2: {len(pkgs)} packages in parallel (worktrees)")
        # Worktrees are created HERE, on the main thread, before any worker
        # starts: `git worktree add`/`prune` take a lock in the parent repo, so
        # two threads doing it at once is a needless race. The workers then only
        # run their agent loops.
        for pkg in pkgs:
            pkg.worktree = _create_pkg_worktree(run, pkg)
        # A package with no checkout cannot join a parallel phase: two quiet
        # writers in the shared tree is exactly the race worktrees exist to
        # prevent (and a quiet thread auto-approves under yolo). Refuse it here,
        # with the reason, instead of racing.
        live = [p for p in pkgs if p.worktree]
        for pkg in pkgs:
            if not pkg.worktree:
                pkg.status = "failed"
                pkg.note = ("worktree isolation failed — a package cannot run "
                            "in parallel without its own checkout")
        pkgs = live
        if not pkgs:
            _note(run, "phase 2: no package could get a worktree — nothing ran")
            return
        threads = [threading.Thread(target=_package_agent, args=(run, pkg, True),
                                    name=f"nx-build-{pkg.name}", daemon=True)
                   for pkg in pkgs]
        run.threads = threads
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        why = "not yolo" if not run.parallel else "no worktree isolation"
        _step(run, f"phase 2: {len(pkgs)} packages sequentially ({why})")
        for pkg in pkgs:
            _package_agent(run, pkg, quiet=False)
    _merge_packages(run)


def _merge_packages(run: _BuildRun) -> None:
    for pkg in run.packages:
        if pkg.status != "failed":
            _merge_package(run, pkg)


def _merge_package(run: _BuildRun, pkg: _BuildPackage) -> None:
    """Land one package's work on the feature branch. The agent may have
    committed in its worktree or left the tree dirty — both are landed, because
    a pipeline that reports a package as done must have its bytes in the branch
    it is handing over."""
    wt = pkg.worktree
    if wt and not os.path.isdir(wt):
        pkg.status = "failed"
        pkg.note = "the package's worktree disappeared before the merge"
        return
    tree = wt or run.repo
    rc, dirty, _ = _git(tree, "status", "--porcelain")
    if rc != 0:
        pkg.status = "failed"
        pkg.note = "could not read the package's working tree"
        return
    if dirty:
        rc, _, err = _git(tree, "add", "-A")
        if rc != 0:
            pkg.status = "failed"
            pkg.note = f"could not commit the package's work: {err}"
            return
        _git(tree, "commit", "-m",
             f"build {run.run_id}: package {pkg.name} (pipeline commit)")
    head = _head_of(tree)
    if head is None:
        pkg.status = "failed"
        pkg.note = "could not read the package's HEAD"
        return
    if wt and head == pkg.base_head:
        # An agent that edited nothing (its tree is clean and HEAD never moved)
        # produced prose, not a package — the same outcome check the sub-agent
        # worker applies. Reporting it as done would hand the user a branch that
        # does not contain the package.
        pkg.status = "failed"
        pkg.note = "the package agent left no changes in its worktree"
        _note(run, f"  {pkg.name}: no changes found — not merged")
        return
    if wt and head != pkg.base_head:
        rc, out, err = _git(run.repo, "merge", "--no-edit", head)
        if rc != 0:
            _git(run.repo, "merge", "--abort")
            _retry_package(run, pkg, (err or out))
            return
    if _dirty(run.repo):
        # The merge landed the commits; this is the work an agent left
        # uncommitted in the shared tree (no-worktree packages).
        if not _commit_feature(run, f"build {run.run_id}: package {pkg.name}"):
            pkg.status = "failed"
            pkg.note = "could not commit the package's work on the feature branch"
            return
    pkg.status = "done"
    _step(run, f"  {pkg.name}: merged ({', '.join(pkg.paths) or 'unscoped'})")


def _retry_package(run: _BuildRun, pkg: _BuildPackage, detail: str) -> None:
    """One retry with the conflict as the package's spec, then stop and report.
    Never a loop: the retry is the whole ladder."""
    pkg.retries += 1
    if pkg.retries > _PACKAGE_RETRIES:
        pkg.status = "failed"
        pkg.note = ("merge conflict and the retry did not resolve it — the "
                    "feature branch was left at the last good merge; the "
                    f"package's tree is {pkg.worktree or 'gone'}")
        _note(run, f"  {pkg.name}: conflict retry failed — stopping")
        return
    _note(run, f"  {pkg.name}: merge conflict — re-running the package once "
               "with the conflict as its spec")
    pkg.worktree = None
    pkg.answer = ""
    pkg.status = "pending"
    conflict = (
        "Your package conflicted when it was merged into the feature branch. "
        "Re-implement it so it applies cleanly against the CURRENT feature "
        "branch (another package has already landed there).\n\n"
        f"GIT OUTPUT:\n{detail[:2000]}\n\n"
        + _package_instruction(run, pkg))
    pkg.worktree = _create_pkg_worktree(run, pkg)
    answer = _run_agent(run, conflict, tier="implement", scope=pkg.paths or None,
                        worktree=pkg.worktree,
                        max_turns=_env_int("NX_BUILD_PACKAGE_TURNS", 80),
                        quiet=run.parallel and run.use_worktrees,
                        system_prompt=_phase_prompt("BUILD_PACKAGE_SYSTEM"))
    pkg.answer = answer or ""
    _write_package_report(run, pkg)
    if isinstance(answer, str) and answer.startswith("ERROR"):
        pkg.status = "failed"
        pkg.note = answer[:400]
        return
    _merge_package(run, pkg)


def _commit_feature(run: _BuildRun, message: str) -> bool:
    """`git add -A` + commit on the feature branch. True when the branch is
    clean afterwards (including "nothing to commit")."""
    if not _dirty(run.repo):
        return True
    rc, _, err = _git(run.repo, "add", "-A")
    if rc != 0:
        _note(run, f"  git add failed: {err}")
        return False
    rc, out, err = _git(run.repo, "commit", "-m", message)
    if rc != 0:
        if "nothing to commit" in (out + err).lower():
            return True
        _note(run, f"  git commit failed: {err or out}")
        return False
    return True


# --------------------------------------------------------------------------
# phase 3 — critique
# --------------------------------------------------------------------------

_SEVERITY_RE = re.compile(r"^\s*\**\s*(critical|major|minor|info)\b", re.IGNORECASE)
_FILE_RE = re.compile(r"[\w./-]+\.(?:py|md|toml|cfg|ini|json|yaml|yml|sh|txt)"
                      r"(?::\d+)?")
_STRUCTURAL_WORDS = ("cross-", "architecture", "design", "interface", "schema",
                     "refactor", "rewrite", "multiple", "several", "package",
                     "concurrency", "thread", "across")


def _critique_focus(run: _BuildRun) -> str:
    criteria = _acceptance_criteria(run.spec)
    crit = "; ".join(criteria) if criteria else "(the spec names none)"
    return (f"the work implemented on branch {run.branch} against the spec's "
            f"acceptance criteria: {crit}. Judge whether the code actually "
            "satisfies each criterion, and name anything that would break it.")


def _acceptance_criteria(spec: str) -> list[str]:
    m = re.search(r"^##\s+Acceptance criteria\s*$", spec, re.MULTILINE | re.IGNORECASE)
    if not m:
        return []
    rest = spec[m.end():]
    nxt = re.search(r"^##\s+", rest, re.MULTILINE)
    block = rest[:nxt.start()] if nxt else rest
    out = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith(("-", "*")) or re.match(r"^\d+[.)]", line):
            item = re.sub(r"^[-*\d.)\s]+", "", line).strip()
            if item:
                out.append(item[:200])
    return out


def _parse_findings(text: str) -> list[str]:
    """Findings from the reviewer's one-line-per-finding format. The explicit
    all-clear line means no findings — never a phantom one."""
    if not text or "No findings above the floor" in text:
        return []
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and _SEVERITY_RE.match(ln)]


def _phase_critique(run: _BuildRun) -> list[str]:
    import nx as _m
    if not getattr(_m, "NX_SUBAGENTS", False):
        _note(run, "critique skipped — the reviewer needs NX_SUBAGENTS=1")
        return []
    if run.critique_rounds >= _MAX_CRITIQUE_ROUNDS:
        return []
    from ._subagent import critique
    paths = " ".join(p for pkg in run.packages for p in pkg.paths)
    run.critique_rounds += 1
    _step(run, f"critique round {run.critique_rounds} "
               "(focus = the acceptance criteria)")
    text = critique(_critique_focus(run), paths=paths or None, async_=False)
    run.critique_texts.append(text)
    _write_critique(run, text)
    if isinstance(text, str) and text.startswith("ERROR"):
        _note(run, "critique unavailable: " + text[:200])
        return []
    findings = _parse_findings(text)
    _step(run, f"  critique: {len(findings)} finding(s) at or above the floor")
    return findings


def _write_critique(run: _BuildRun, text: str) -> None:
    try:
        os.makedirs(run.run_dir, exist_ok=True)
        with open(os.path.join(run.run_dir, f"critique-{run.critique_rounds}.md"),
                  "w", encoding="utf-8") as f:
            f.write(text or "(no output)\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# phase 4 — correct or redo
# --------------------------------------------------------------------------

def _finding_is_simple(finding: str) -> bool:
    """A simple finding is one-file and mechanical: exactly one file named, and
    no wording that signals a cross-package or design-level change. Everything
    else is structural and gets a sub-agent."""
    files = {f.split(":")[0] for f in _FILE_RE.findall(finding)}
    if len(files) != 1:
        return False
    low = finding.lower()
    return not any(w in low for w in _STRUCTURAL_WORDS)


def _phase_fix(run: _BuildRun, findings: list[str]) -> list[str]:
    """The correct-or-redo ladder. Returns the findings still open after at most
    ONE re-critique: simple → inline by the main model; structural → one
    follow-up sub-agent with a mini-spec from the finding."""
    remaining: list[str] = []
    for f in findings:
        if _finding_is_simple(f):
            _step(run, f"phase 4: inline fix — {f[:120]}")
            if not _inline_fix(run, f):
                remaining.append(f)
        else:
            _step(run, f"phase 4: one sub-agent for a structural finding — "
                       f"{f[:120]}")
            if not _structural_fix(run, f):
                remaining.append(f)
    if remaining:
        # A fix that did not land is not something another review round can
        # settle — report it instead of looping.
        return remaining
    return _phase_critique(run)


def _fix_instruction(run: _BuildRun, finding: str) -> str:
    return (
        "A reviewer found this problem in the work on the feature branch:\n\n"
        f"{finding}\n\n"
        "Fix ONLY this. Write the smallest change that resolves it, in the "
        "files the finding names; do not restructure anything else and do not "
        "touch unrelated code. Verify with the narrowest test or command that "
        "covers the change, then report: the file(s) changed, what the edit "
        "did, the verify command and its result. If the finding is wrong, say "
        "why and make no edit.\n\n"
        "CONTEXT — the spec the work is judged against:\n"
        f"{run.spec[:3000]}"
    )


def _structural_fix(run: _BuildRun, finding: str) -> bool:
    """ONE follow-up write-tier agent with a mini-spec written FROM the finding
    (the files it names, what done means). It runs in the shared tree on the
    feature branch: sequential, so there is no merge step and no second writer
    to race."""
    paths = [f.split(":")[0] for f in _FILE_RE.findall(finding)]
    paths = list(dict.fromkeys(paths))[:5]
    try:
        answer = _run_agent(run, _fix_instruction(run, finding), tier="implement",
                            scope=paths or None, worktree=None,
                            max_turns=_env_int("NX_BUILD_FIX_TURNS", 60),
                            quiet=False,
                            system_prompt=_phase_prompt("BUILD_PACKAGE_SYSTEM"))
    except Exception as e:  # noqa: BLE001 — a failed fix is a reported finding
        _note(run, f"  fix agent failed: {type(e).__name__}: {e}")
        return False
    if isinstance(answer, str) and answer.startswith("ERROR"):
        _note(run, f"  fix agent refused: {answer[:200]}")
        return False
    _commit_feature(run, f"build {run.run_id}: critique fix (agent)")
    return True


def _inline_fix(run: _BuildRun, finding: str) -> bool:
    """A one-file mechanical finding, fixed by the main model in one tools-free
    call. No new tool schema, no file-scoped sub-agent, and no extra gate: the
    user already approved implementation at the spec gate."""
    obj = _ask_main_model(run, finding)
    if obj is None:
        return False
    path = obj.get("path")
    body = obj.get("content")
    if not isinstance(path, str) or not isinstance(body, str) or not path.strip():
        _note(run, "  inline fix: the model did not return {path, content}")
        return False
    target = os.path.realpath(os.path.join(run.repo, path))
    if not target.startswith(run.repo + os.sep):
        _note(run, f"  inline fix refused: {path} is outside the repo")
        return False
    if not os.path.exists(target):
        _note(run, f"  inline fix refused: {path} does not exist")
        return False
    try:
        with open(target, encoding="utf-8") as f:
            before = f.read()
    except OSError:
        before = ""
    if before == body:
        _note(run, f"  inline fix: {path} already matches — nothing to change")
        return True
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(body)
    except OSError as e:
        _note(run, f"  inline fix failed to write {path}: {e}")
        return False
    _commit_feature(run, f"build {run.run_id}: critique fix (inline)")
    _note(run, f"  inline fix applied to {path}")
    return True


def _ask_main_model(run: _BuildRun, finding: str) -> dict | None:
    """One no-tools side call to the session's active model, returning the full
    new content of the single file the finding names. Mirrors the shape of
    `_risk`'s classifier call: never the turn loop, never the user."""
    if st.client is None or st.MODEL is None:
        return None
    from ._prompts import BUILD_FIX_SYSTEM
    payload = [{"role": "system", "content": BUILD_FIX_SYSTEM},
               {"role": "user", "content":
                f"FINDING:\n{finding}\n\nSPEC CONTEXT:\n{run.spec[:2000]}"}]
    try:
        resp = st.client.chat.completions.create(
            model=st.MODEL, messages=payload, stream=False, timeout=120,
            **(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {}))
    except Exception:  # noqa: BLE001 — a failed side call is a finding, not a crash
        _note(run, "  inline fix: the model call failed")
        return None
    try:
        from ._metrics import _record_api_call
        text = (resp.choices[0].message.content or "").strip()
        _record_api_call(resp, fallback_chars=len(text))
    except Exception:  # noqa: BLE001 — a failed reply is just no fix
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------
# phase 5 — report + handoff
# --------------------------------------------------------------------------

def _diff_stat(run: _BuildRun) -> str:
    base = run.start_branch or run.base_head
    if not base:
        return "(no diff stat)"
    rc, out, _ = _git(run.repo, "diff", "--stat", f"{base}...HEAD")
    return out if rc == 0 and out else "(no changes vs the starting branch)"


def _report(run: _BuildRun) -> str:
    first_line = run.issue.splitlines()[0][:80] if run.issue.strip() else ""
    done = [p.name for p in run.packages if p.status == "done"]
    failed = [f"{p.name} ({p.note[:120]})" for p in run.packages
              if p.status == "failed"]
    lines = [
        f"build {run.run_id}: {first_line}".rstrip(),
        f"  branch:   {run.branch} (checked out; started from "
        f"{run.start_branch or 'detached HEAD'})",
        f"  packages: {len(done)}/{len(run.packages)} done"
        + (f" — failed: {'; '.join(failed)}" if failed else ""),
        f"  critique: {run.critique_rounds} round(s)",
        f"  run dir:  {run.run_dir}",
    ]
    if run.remaining:
        lines.append(f"  REMAINING FINDINGS ({len(run.remaining)}) — yours to judge:")
        lines += [f"    - {f[:200]}" for f in run.remaining]
    else:
        lines.append("  remaining findings: none")
    for note in run.notes:
        if not note.startswith("  "):
            lines.append(f"  note: {note}")
    lines.append("  diff vs the starting branch:")
    lines += [f"    {ln}" for ln in _diff_stat(run).splitlines()[:40]]
    lines.append(f"  committed on {run.branch} — review it, then merge or delete "
                 "it yourself. Nothing was pushed.")
    return "\n".join(lines)


def _abort_note(run: _BuildRun, reason: str) -> None:
    try:
        os.makedirs(run.run_dir, exist_ok=True)
        with open(os.path.join(run.run_dir, "ABORTED.md"), "w",
                  encoding="utf-8") as f:
            f.write(f"# build {run.run_id} — ABORTED\n\n"
                    f"- reason: {reason}\n"
                    f"- issue: {run.issue}\n"
                    f"- started: "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.started))}\n"
                    f"- branch: {run.branch}\n"
                    f"- spec: {run.spec_path}\n\n"
                    "## Notes\n\n"
                    + "\n".join(f"- {n}" for n in run.notes)
                    + "\n\nResume by hand: the spec (if written) is the plan; "
                      f"`git checkout {run.branch}` re-enters the work if any "
                      "of it was committed.\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _preflight(repo: str) -> str | None:
    """Refuse before any state exists: a non-git root, a dirty tree, a run
    already in flight, or sub-agents switched off (`/build` IS a sub-agent
    harness — running it with them disabled would silently degrade every
    phase). Returns the refusal text, or None when the pipeline may start."""
    import nx as _m
    if not getattr(_m, "NX_SUBAGENTS", False):
        return ("sub-agents are off (NX_SUBAGENTS=0) — /build runs a spec agent, "
                "one agent per work package and a reviewer; turn sub-agents on "
                "first")
    if not _is_git(repo):
        return (f"not a git repository: {repo} — /build needs a git project "
                "root (it works on its own feature branch)")
    if _dirty(repo):
        return ("the working tree has uncommitted changes — commit or stash "
                "first: /build creates a feature branch and would otherwise "
                "commit your work-in-progress into it")
    active = _active_run(repo)
    if active is not None:
        return (f"a build is already in progress for this repo "
                f"({active.get('run_id')} on branch {active.get('branch')}) — "
                "one at a time; wait for it to finish, or delete "
                f"{_active_path(repo)} if that process is gone")
    return None


def run_build(issue: str, state=None) -> str:
    """The whole pipeline. Prints progress (and the spec gate) itself and
    returns the final report. Ctrl-C / a refused gate aborts cleanly: the
    working tree returns to the starting branch, the run dir is kept with an
    ABORTED.md note, and a feature branch with nothing on it is removed."""
    repo = _repo_root()
    refusal = _preflight(repo)
    if refusal:
        return f"/build refused: {refusal}"

    parallel = bool(st.YOLO) and _env_int("NX_BUILD_PARALLEL", 1) > 0
    # Preflight proved this IS a git repo, so the worktree machinery's cached
    # flag is truthful — without it create_worktree refuses and a repro that
    # started outside the project root would silently run in the shared tree.
    if _is_git(repo) and not getattr(st, "PROJECT_ROOT_IS_GIT", False):
        st.PROJECT_ROOT_IS_GIT = True
    use_worktrees = (bool(getattr(st, "PROJECT_ROOT_IS_GIT", False))
                     and _env_int("NX_BUILD_WORKTREE", 1) > 0)
    run = _BuildRun(repo, issue, _env_int("NX_BUILD_VERBOSE", 0) > 0,
                    parallel, use_worktrees)
    _mark_active(run)
    run.owns_active = True
    print(f"{DIM}  build {run.run_id}: branch {run.branch}, "
          f"{'parallel' if parallel and use_worktrees else 'sequential'} "
          f"packages, run dir {run.run_dir}{RESET}", flush=True)
    gate_passed = False
    try:
        err = _create_branch(run)
        if err:
            _abort_note(run, err)
            return f"/build failed: {err}"
        _step(run, f"phase 0: created {run.branch} from "
                   f"{run.start_branch or 'detached HEAD'}")

        _step(run, "phase 1: writing the spec (one sub-agent)...")
        try:
            run.spec = _run_agent(run, _spec_instruction(run), tier="implement",
                                  scope=_spec_scope(run),
                                  worktree=None, max_turns=_SPEC_AGENT_TURNS,
                                  quiet=False,
                                  system_prompt=_phase_prompt("BUILD_SPEC_SYSTEM")) or ""
        except Exception as e:  # noqa: BLE001 — a dead spec agent is an abort
            run.spec = ""
            _note(run, f"spec agent failed: {type(e).__name__}: {e}")
        if os.path.exists(run.spec_path):
            try:
                with open(run.spec_path, encoding="utf-8") as f:
                    disk = f.read()
            except OSError:
                disk = ""
            if len(disk.strip()) > len(run.spec.strip()):
                run.spec = disk
        if not run.spec.strip():
            _abort_note(run, "the spec agent produced no spec")
            return f"/build aborted: the spec agent produced no spec — see {run.run_dir}"
        run.packages = _parse_spec(run)
        if not run.spec.lstrip().startswith("#"):
            # The agent answered in prose instead of writing the file: keep the
            # prose as the spec rather than failing the run on formatting.
            _note(run, "the spec agent did not write SPEC.md — using its answer")
        if not os.path.exists(run.spec_path):
            try:
                os.makedirs(run.run_dir, exist_ok=True)
                with open(run.spec_path, "w", encoding="utf-8") as f:
                    f.write(run.spec)
            except OSError:
                pass
        if not _confirm_spec(run):
            # The one gate the design calls out: a declined spec leaves nothing
            # behind — no branch, no run dir (the branch is dropped in the
            # finally's abort cleanup, after the tree is back on the start branch).
            run.aborted = True
            _discard_run_dir(run)
            return (f"/build aborted at the spec gate — nothing implemented; "
                    f"branch {run.branch} and its run dir were removed.")
        gate_passed = True

        _run_packages(run)
        run.remaining = _phase_fix(run, _phase_critique(run))
        _commit_feature(run, f"build {run.run_id}: {_issue_first_line(issue)}")
        report = _report(run)
        print(report)
        return report
    except KeyboardInterrupt:
        run.aborted = True
        _abort_note(run, "interrupted (Ctrl-C)")
        print(f"{DIM}  build interrupted — aborting cleanly{RESET}")
        where = (f"the work already committed is on branch {run.branch} — check "
                 "it out to inspect it" if gate_passed
                 else f"branch {run.branch} was removed")
        return (f"/build interrupted. The run dir {run.run_dir} is kept (see "
                f"ABORTED.md); {where}.")
    finally:
        if run.aborted or not gate_passed:
            _cleanup_aborted(run, gate_passed)
        _clear_active(run)


def _cleanup_aborted(run: _BuildRun, gate_passed: bool) -> None:
    """Abort cleanup: back to the starting branch, then drop the feature branch
    when nothing of the user's can be on it. A post-gate abort keeps the branch
    (work exists) and only returns the tree — and commits whatever the aborted
    phase had left in the tree, so the work is on the branch the abort messages
    name rather than sitting in a tree the user is about to leave."""
    if gate_passed:
        _join_package_threads(run)
        if _current_branch(run.repo) == run.branch and _dirty(run.repo):
            _commit_feature(run, f"build {run.run_id}: interrupted (partial work)")
    back_err = _return_to_start(run)
    if back_err:
        _note(run, back_err)
        return
    if gate_passed or _current_branch(run.repo) == run.branch:
        return  # work exists, or we could not leave the branch — never delete
    rc, _, _ = _git(run.repo, "branch", "-D", run.branch)
    if rc != 0:
        _note(run, f"could not delete {run.branch} (it may hold work)")


def _join_package_threads(run: _BuildRun, timeout: float = 20.0) -> None:
    """Wait (briefly) for live package workers before an abort moves HEAD. A
    worker that outlives the wait is left running — it is a daemon thread and
    its package's work stays in its own worktree, never in the shared tree."""
    for t in getattr(run, "threads", []):
        if not t.is_alive():
            continue
        t.join(timeout=timeout)
        if t.is_alive():
            _note(run, f"  worker {t.name} is still running; its work stays in "
                       "its worktree")


def _discard_run_dir(run: _BuildRun) -> None:
    """Delete a run dir. Used ONLY for the abort the design calls out by name
    (a spec the user declined at the gate, before any implementation); every
    other abort KEEPS the run dir with its ABORTED.md note. Nothing else in the
    repo is ever deleted by the pipeline."""
    shutil.rmtree(run.run_dir, ignore_errors=True)
