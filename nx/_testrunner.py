"""The run_tests tool (0.49.0, ROADMAP P3-5): closed-loop test verification.

One command per ecosystem, detected from marker files, run targeted, parsed
into a structured summary, and handed back as a tail note the model can act on
without a second tool call. This is the scoped, structured sibling of
NX_VERIFY_CMD (which stays the hands-off post-edit loop): here the MODEL
chooses when to run tests and with what target, the command shape is fixed
per ecosystem (no free-text shell), and the result is folded to the failure
so a fat suite doesn't eat the tool-result budget.

Approval/approval-wording contract: the action string handed to _confirm() is
"run tests: <argv>" — nx/_risk.py's local fast path scores exactly that shape
("medium, contained test run") so a run_tests approval never costs a
classifier round-trip, mirroring the "run: <cmd>" fast path for run_bash.
Sub-agent posture: run_tests joins _WRITE_TIER_TOOL_NAMES (the write tier's
whole point is verifying its own edits); read-only agents keep the _subagent
refusal. Plan mode gates it (PLAN_GATE_TOOLS): a test run writes caches
(.pyc, .pytest_cache, target/) while a plan holds.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from . import _state as st
from ._env import _env_int

# Hard wall-clock ceiling for one run_tests call. Tests are usually the slowest
# thing a session does; without a ceiling a hung suite (a fixture that never
# yields) wedges the turn until the model's client times out. NX_TESTRUNNER_
# TIMEOUT=0 disables the cap (the process must then be trusted to exit).
_DEFAULT_TIMEOUT = 600

# Test-impact map (0.94.0): per-project JSONL of {file -> [node ids]} recorded
# from each run's failures, so a scoped re-run after editing one file can
# propose the tests that failed there last time instead of the whole suite.
# Lives under the sessions store (NX_HOME), keyed by project root hash.

def _testmap_path() -> str:
    import hashlib
    root = st.PROJECT_ROOT or os.getcwd()
    h = hashlib.sha256(os.path.realpath(root).encode()).hexdigest()[:12]
    home = os.environ.get("NX_HOME") or os.path.expanduser(
        os.path.join("~", ".nx"))
    return os.path.join(home, "testmaps", f"{h}.jsonl")

# Fold budget: per-failure-block caps (head/tail of each traceback) and the
# whole-result cap. Sized so ~40 failing pytest blocks still fit the default
# NX_TOOL_RESULT_CHARS with room for the summary.
_BLOCK_HEAD = 4
_BLOCK_TAIL = 8
_MAX_LINES = 200
# Below this many raw output lines the body already carries everything worth
# reading — a temp-log "full output" line would be pure noise on green runs.
_LOG_MIN_LINES = 20

# Marker files, in priority order: the first ecosystem whose marker exists
# wins. go.mod before package.json so a Go project that happens to vendor a
# stray package.json isn't misdetected; pytest.ini/pyproject/setup.cfg before
# bare tests/ so configuration wins over convention.
_PYTEST_MARKERS = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini",
                   "conftest.py")


def _detect_runner(d: str) -> str | None:
    """Detect the test runner for directory *d* from its marker files.

    Pure filesystem sniffing — no subprocess, no network, never raises.
    Returns "pytest" | "go" | "cargo" | "npm", or None when nothing matches
    (the caller refuses with the markers it looked for, so the model can
    either name the runner explicitly or report the gap).
    """
    try:
        names = set(os.listdir(d))
    except OSError:
        return None
    if "go.mod" in names:
        return "go"
    if "Cargo.toml" in names:
        return "cargo"
    if "package.json" in names:
        return "npm"
    if "pytest.ini" in names or "tox.ini" in names or "conftest.py" in names:
        return "pytest"
    if "pyproject.toml" in names or "setup.cfg" in names:
        # A python project without a bare tests/ dir might still have its
        # tests elsewhere; treat the project as pytest if any test dir or
        # file exists at the top level.
        for name in names:
            if name.startswith("test"):
                return "pytest"
        return "pytest"
    for name in names:
        if name.startswith("test"):
            return "pytest"
    return None


def _has_xdist() -> bool:
    """True when pytest-xdist is importable (enables `-n auto` parallel runs)."""
    try:
        import xdist  # noqa: F401
        return True
    except ImportError:
        return False


def _test_command(runner: str, targets: str) -> list[str]:
    """The fixed command shape for *runner* (never free-text shell).

    uv is preferred for pytest because that is this repo's own documented
    workflow; a plain interpreter falls back to ``-m pytest`` so a venv-less
    checkout still works. Targets are appended verbatim — they are
    file-or-directory paths / `node::test` ids for the runner to interpret,
    not a shell fragment.
    """
    if runner == "pytest":
        uv = shutil.which("uv") or shutil.which("uv3")
        if uv:
            base = [uv, "run", "pytest", "-q"]
        else:
            base = [sys.executable, "-m", "pytest", "-q"]
        if _has_xdist():
            base.extend(["-n", "auto"])
        else:
            print("pytest-xdist not found; running pytest sequentially.")
            print("Install pytest-xdist to enable parallel test execution.")
    elif runner == "go":
        base = ["go", "test", "./..."]
    elif runner == "cargo":
        base = ["cargo", "test"]
    elif runner == "npm":
        npm = shutil.which("npm") or shutil.which("pnpm") or shutil.which("yarn")
        base = [npm or "npm", "test", "--", "--run"]
    else:  # unreachable via _detect_runner; keeps mypy/ty honest
        return []
    if targets:
        base.extend(targets.split())
    return base


def _extract_path(line: str) -> str | None:
    """Pull a filesystem-ish path out of a failure line (for the folded
    block's leading path). Returns None when the line carries no path."""
    m = re.search(r"([A-Za-z0-9_./\\-]+\.(?:py|rs|go|ts|tsx|js|jsx))", line)
    return m.group(1) if m else None


def _summarize(runner: str, out: str, exit_code: int | None) -> str:
    """Fold a test run's full output down to the failures the model needs.

    Per ecosystem:
    - pytest: the (===== N failed/M passed =====) summary lines, then one
      folded block per `_____ test_x _____` failure section (assertion tail),
      then FAILED lines. A fully green run keeps only the summary line.
    - go: every `--- FAIL: TestX` line plus the package summary.
    - cargo: the `failures:` list and `test result:` lines.
    - npm: error/FAIL lines and the `Tests:` summary.

    The fold is capped twice (_MAX_LINES whole-result, head/tail per block)
    so even a fully red suite returns something the model can read in one
    screen instead of a 20K-character wall of traceback.
    """
    lines = out.splitlines()
    if runner == "pytest":
        # -q summaries drop the trailing '=' pad (`1 passed in 0.01s` alone is
        # common), so the marker is an optional lead/trail, not a closed box.
        summary = [ln for ln in lines
                   if re.match(r"=+ .+? =+$", ln.strip())
                   or re.fullmatch(r"\d+ (?:passed|failed|error|skipped)"
                                   r"(?:, \d+ (?:passed|failed|error|skipped))*"
                                   r" in [\d.]+s(?:\n? .*)?", ln.strip())]
        failed_idx = [i for i, ln in enumerate(lines)
                      if re.match(r"_{5,} .+ _{5,}$", ln.strip())]
        if not failed_idx:
            keep = summary or ([ln for ln in lines if ln.strip().endswith("%]")]
                               or [f"(no test output; exit {exit_code})"])
        else:
            keep = []
            for j, i in enumerate(failed_idx):
                end = failed_idx[j + 1] if j + 1 < len(failed_idx) else len(lines)
                block = lines[i:end]
                if len(block) > _BLOCK_HEAD + _BLOCK_TAIL + 2:
                    block = (block[:_BLOCK_HEAD]
                             + [f"    ... [{len(block) - _BLOCK_HEAD - _BLOCK_TAIL} "
                                f"traceback lines folded — full output below]"]
                             + block[-_BLOCK_TAIL:])
                keep.extend(block)
                if j + 1 < len(failed_idx):
                    keep.append("")
            # The FAILED lines name node ids — usually all the model needs.
            keep.extend(ln for ln in lines if ln.startswith("FAILED ")
                        or ln.startswith("ERROR "))
            # The header line above each block already carries the test id;
            # a path in the block helps locate the file without a re-run.
            if summary:
                keep.append("")
                keep.extend(summary)
            else:
                keep.append(f"(exit {exit_code})")
        result = keep
    elif runner == "go":
        fails = [ln for ln in lines if ln.startswith("--- FAIL:")
                 or ln.startswith("FAIL")]
        pkgs = [ln for ln in lines if ln.startswith("ok  ") or ln.startswith("FAIL")]
        result = fails + pkgs or [f"(no test output; exit {exit_code})"]
    elif runner == "cargo":
        in_failures = False
        fails: list[str] = []
        summary_lines = [ln for ln in lines if ln.startswith("test result:")]
        for ln in lines:
            if ln.strip() == "failures:":
                in_failures = True
                continue
            if in_failures:
                if ln.strip().startswith("test result:"):
                    in_failures = False
                    continue
                if ln.strip():
                    fails.append("    " + ln.strip())
        result = summary_lines + (["failures:"] + fails if fails else [])
        if not result:
            result = [f"(no test output; exit {exit_code})"]
    elif runner == "npm":
        errs = [ln for ln in lines if "FAIL" in ln or "Error" in ln
                or ln.strip().startswith("✕") or ln.strip().startswith("✗")]
        tail = [ln for ln in lines if ln.strip().startswith(("Tests:", "Test Suites:"))]
        result = errs[-40:] + tail or [f"(no test output; exit {exit_code})"]
    else:
        result = lines

    if len(result) > _MAX_LINES:
        head, tail = result[:120], result[-60:]
        result = head + [f"... [{len(result) - 180} summary lines folded]"] + tail
    return "\n".join(result)


def _record_failures(runner: str, out: str) -> None:
    """Append file -> failing-node-id pairs from a run's output to the project
    test map. Fire-and-forget: any error is swallowed (the map is an
    optimization, never a correctness dependency). Only pytest node ids and
    go/cargo/npm failure lines that clearly carry a path are recorded."""
    try:
        pairs: list[tuple[str, str]] = []
        if runner == "pytest":
            for ln in out.splitlines():
                m = re.match(r"(FAILED|ERROR)\s+(\S+)", ln.strip())
                if not m:
                    continue
                node = m.group(2).split(" - ")[0]
                path = node.split("::")[0]
                if path.endswith(".py"):
                    pairs.append((path, node))
        else:
            for ln in out.splitlines():
                p = _extract_path(ln)
                if p and ("FAIL" in ln or "failed" in ln or "Error" in ln):
                    pairs.append((p, ln.strip()[:120]))
        if not pairs:
            return
        path = _testmap_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json
        with open(path, "a", encoding="utf-8") as f:
            for src, node in pairs:
                f.write(json.dumps({"f": src, "t": node}) + "\n")
    except Exception:  # noqa: BLE001 — never break the test run
        pass


def _mapped_targets(files: list[str]) -> list[str]:
    """Node ids that previously failed in the given source files (most recent
    first, deduped, capped at 20). Empty when the map has nothing for them."""
    try:
        import json
        path = _testmap_path()
        if not os.path.exists(path):
            return []
        want = {os.path.relpath(f, st.PROJECT_ROOT or ".") if os.path.isabs(f) else f
                for f in files}
        want = {w.replace("\\", "/") for w in want}
        hits: list[str] = []
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                f = str(rec.get("f", "")).replace("\\", "/")
                if f in want and rec.get("t") and rec["t"] not in hits:
                    hits.append(rec["t"])
        return hits[-20:]
    except Exception:  # noqa: BLE001
        return []


def _spin_out(out: str) -> list[str]:
    return out.splitlines()


def run_tests(targets: str = "", **_: Any) -> str:
    """Detect + run the project's test suite, return a failure-focused summary.

    Detection is marker-based (see _detect_runner); the command shape is
    fixed per ecosystem — this tool never runs free-text shell, that is
    run_bash's job. Output is folded to the failures (see _summarize); when
    anything was folded the full raw output is written to a temp file whose
    path rides the result for read_file.
    """
    # Target scoping: a `path` that exists narrows both detection and the run;
    # `node_id`/filter-shaped targets pass through to the runner against the
    # project root (pytest -q <node id>, go -run handled by the runner args).
    target_dir = targets.strip() if isinstance(targets, str) else ""
    detect_dir = target_dir
    if target_dir and not os.path.isdir(target_dir):
        parent = os.path.dirname(target_dir) or "."
        detect_dir = parent if os.path.isdir(parent) else (st.PROJECT_ROOT or ".")
    elif not target_dir:
        detect_dir = st.PROJECT_ROOT or "."
    # Test-impact targeting (0.94.0): a bare run right after edits to files
    # that previously failed proposes those tests first, as a scoped hint —
    # never a silent restriction of the run itself.
    map_hint = ""
    if not target_dir:
        try:
            from ._undo import list_checkpoints
            edited = [f.path for cp in list_checkpoints()
                      for f in (cp.files or ())]
        except Exception:  # noqa: BLE001
            edited = []
        mapped = _mapped_targets(edited) if edited else []
        if mapped:
            map_hint = ("\nnote: previously failed in the files edited this "
                        f"session: {', '.join(mapped[:8])} — re-run with "
                        "targets to scope to them once green")
    runner = _detect_runner(detect_dir)
    if runner is None:
        return ("ERROR: run_tests found no test project at "
                f"{detect_dir or '.'} (looked for pytest.ini/pyproject.toml/"
                "setup.cfg/tox.ini/conftest.py, go.mod, Cargo.toml, "
                "package.json, a tests/ dir). Pass `targets` at the test "
                "root, or run the suite yourself with run_bash.")
    argv = _test_command(runner, target_dir)
    if not argv:
        return f"ERROR: run_tests: no command shape for runner {runner!r}"
    action = f"run tests: {' '.join(argv)}"
    from ._bash_jobs import _confirm  # lazy: avoid an import cycle at module load
    if not _confirm(action):
        return "DENIED by user"
    timeout = _env_int("NX_TESTRUNNER_TIMEOUT", _DEFAULT_TIMEOUT)
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout or None, cwd=st.PROJECT_ROOT or None)
        out = (r.stdout or "") + (r.stderr or "")
        exit_code: int | None = r.returncode
    except subprocess.TimeoutExpired:
        return (f"[run_tests: {runner}] TIMED OUT after {timeout}s — the suite "
                "may hang; run it with run_bash (backgrounds) to investigate")
    except (OSError, FileNotFoundError):
        return (f"ERROR: run_tests: {argv[0]!r} not found on PATH — install it "
                "or run the suite with run_bash instead")
    out = ((r.stdout or "") + (r.stderr or ""))
    body = _summarize(runner, out, exit_code)
    _record_failures(runner, out)
    lines = out.splitlines()
    folded = (len(lines) > _LOG_MIN_LINES
              and len(_spin_out(body)) < len(lines))
    log_line = ""
    if folded and out.strip():
        try:
            fd, log_path = tempfile.mkstemp(prefix="nx-testrun-", suffix=".log")
            with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
                f.write(out)
            log_line = f"\nfull output ({len(lines)} lines): read_file('{log_path}')"
        except OSError:
            log_line = ""
    verdict = ("All tests passed." if exit_code == 0
               else f"{runner} tests FAILED")
    return (f"[run_tests: {runner}] {verdict} (exit {exit_code})\n"
            f"{body}{map_hint}{log_line}")
