"""F2: spawn_agent read-only sub-agent — schema gating (NX_SUBAGENTS),
read-only dispatch enforcement inside the sub-loop, answer extraction, and
turn-cap handling.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
# Clear ambient NX_* config so a developer's ~/.env / shell (real endpoints,
# keys, NX_VERIFY_CMD, ...) can't affect results — per repo test guidelines.
_AMBIENT_NX = [k for k in os.environ if k == "NX_ENV_FILE" or k.startswith("NX_")]
for k in _AMBIENT_NX:
    os.environ.pop(k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ.pop("NX_VERIFY_CMD", None)
os.environ.pop("NX_BASH_PROMPT_SECRETS", None)
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR", "NX_SUBAGENTS"):
    os.environ.pop(k, None)

import pytest

# 0.96.0: foreground spawn results are head+report-path by default; these
# tests assert on the full inline answer, so opt into the legacy inline mode.
os.environ["NX_SUBAGENT_SYNC_INLINE"] = "1"


@pytest.fixture(autouse=True)
def _sync_inline(monkeypatch):
    # Load-bearing (not a no-op): nx/_subagent._sync_inline prefers the module
    # attr _m.NX_SUBAGENT_SYNC_INLINE over the env, and that attr does not exist
    # at import time, so the module-scope os.environ line alone would not pin
    # inline mode. raising=False because the attr is not defined at import time.
    import nx as _nx
    monkeypatch.setattr(_nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    yield


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    os.environ["NX_SUBAGENT_ASYNC"] = "0"  # these tests exercise the foreground sub-loop; the 0.18.8 default is background (covered in test_subagent_async.py)
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    # The old `importlib.reload(nx)` here rebuilt nx's import-time state from
    # this fixture's env — most importantly the model source: conftest imports
    # nx once under a scrubbed env, so st.client/st.MODEL stay None (no active
    # source) and every spawn would answer "ERROR: no active model source".
    # Instead of a reload, rebuild just that wiring: re-discover sources from
    # the env we just set and switch to the first one (binds st.client +
    # st.MODEL). resolve_model() reads NX_MODEL ("test-model"), so no network
    # round-trip happens; the dead 127.0.0.1:9 endpoint is never contacted.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    # nx reads these knobs through the MODULE ATTR first, falling back to the
    # env (see _subagent._is_async_default / _sync_inline), and the attr holds
    # the import-time default. The reload used to rebuild the attrs from this
    # fixture's env; with the reload gone, state the one behavior-changing knob
    # the env alone can no longer reach as a per-test monkeypatch
    # (auto-restored). NX_SUBAGENT_ASYNC=0 = foreground sub-loop.
    monkeypatch.setattr(nx, "NX_SUBAGENT_ASYNC", 0, raising=False)
    yield nx
    # env rewind is universal (conftest), so no manual pop here; nx globals a
    # test mutates go through monkeypatch (per-test, auto-restored).


def test_schema_gated_by_env(nx_fresh, monkeypatch):
    from nx._dispatch import _active_tools
    names = [t["function"]["name"] for t in _active_tools()]
    assert "spawn_agent" in names  # default ON since 0.77.0
    assert any(t["function"]["name"] == "spawn_agent" for t in nx_fresh.TOOLS)
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENTS", 0)
    names = [t["function"]["name"] for t in _active_tools()]
    assert "spawn_agent" not in names  # explicit opt-out still honored
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENTS", 1)


def test_subagent_loop_uses_readonly_dispatch(nx_fresh, monkeypatch):
    """Inside the sub-loop, run_tool must resolve only read-only tools, even
    when the model tries a mutating one."""
    from nx import _dispatch
    calls = []

    def fake_turn(messages, *a, **kw):
        calls.append([m["role"] for m in messages])
        if len(calls) == 1:
            # the sub-model tries a mutating tool on its first turn
            out = _dispatch.run_tool("write_file", {"path": "evil.txt", "content": "x"})
            assert "not available to the read-only sub-agent" in out
            out2 = _dispatch.run_tool("run_bash", {"command": "rm -rf /"})
            assert "read-only commands" in out2  # allowlist-gated bash tier
            # read-only tools DO work inside the sub-loop
            out3 = _dispatch.run_tool("search_files", {"pattern": "x", "path": "."})
            assert not out3.startswith("ERROR")
            messages.append({"role": "assistant", "content": "findings: module X is in nx/_x.py"})
            return nx_fresh.TURN_DONE
        raise AssertionError("loop should have ended")

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    # dispatch table swap must be active only inside spawn_agent
    from nx._subagent import spawn_agent
    assert nx_fresh._state._SUBAGENT_DISPATCH is None
    result = spawn_agent("where is X implemented?")
    assert nx_fresh._state._SUBAGENT_DISPATCH is None  # restored after
    assert nx_fresh._state._SUBAGENT_ACTIVE is False
    assert result == "findings: module X is in nx/_x.py"
    # the sub-loop's messages never leaked into any global conversation
    assert all("evil.txt" not in json_dumps(m) for m in calls[0])


def json_dumps(m):
    import json
    return json.dumps(m, default=str)


def test_subagent_empty_answer_and_cap(nx_fresh, monkeypatch):
    from nx._subagent import spawn_agent

    def fake_turn(messages, *a, **kw):
        return nx_fresh.TURN_DONE  # model produced nothing

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    out = spawn_agent("summarize nx/_todo.py")
    assert "produced no answer" in out

    def fake_turn_cap(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "partial"})
        return nx_fresh.TURN_TOOL  # never "done" → loop exhausts the cap

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn_cap)
    # default is a FLOOR (0.36.1): drop it to 1 so this test's max_turns=2 is
    # actually honored and the tiny loop reaches the cap quickly.
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_DEFAULT_TURNS", 1)
    out = spawn_agent("summarize", max_turns=2)
    assert "stopped early after 2/2 turns" in out and "partial" in out


def test_subagent_rejects_bad_args_and_nested(nx_fresh, monkeypatch):
    from nx._subagent import spawn_agent
    # empty instruction still errors, but now names the right key so a
    # mis-keyed model call (prompt/description instead of instruction)
    # self-corrects next turn instead of trial-and-erroring
    out = spawn_agent("")
    assert out.startswith("ERROR")
    assert "instruction" in out
    assert spawn_agent("   ").startswith("ERROR")
    # max_turns clamped to sane bounds
    seen = {}

    def fake_loop(messages, *a, **kw):
        seen["max_turns"] = kw.get("max_turns_override")
        messages.append({"role": "assistant", "content": "ok"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    spawn_agent("q", max_turns=9999)
    assert seen["max_turns"] == 500  # ceiling (0.103.10)
    # a request below the default floor is RAISED to it (0.36.1); with the
    # default left at 100, max_turns=0 lands on 100, not 1.
    spawn_agent("q", max_turns=0)
    assert seen["max_turns"] == 100
    # only by lowering the floor can a genuinely tiny budget survive the clamp
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_DEFAULT_TURNS", 1)
    spawn_agent("q", max_turns=0)
    assert seen["max_turns"] == 1


# --- read-only bash tier ----------------------------------------------------

def test_subagent_bash_allowlist_gate(nx_fresh):
    from nx._subagent import _subagent_bash_allowed as a
    ok = [
        "grep -rn TODO nx/", "ls -la", "du -sh .", "ps aux", "pgrep -f nx",
        "wc -l nx/_turn.py", "head -5 f", "tail -5 f", "cat f", "pwd",
        "find . -name x -type f",
        # 0.99.2: harmless text utilities joined the read tier
        "sort f.txt", "uniq f.txt", "cut -d, -f2 f.csv", "paste a b",
        "join a.tsv b.tsv", "column -t f.txt", "comm a b", "fold -w 80 f",
        "nl f", "rev f", "tac f", "expand -t 2 f", "seq 1 10", "date",
        # git read-only forms resolved through _risk.py's own classifier
        "git status", "git diff", "git log --oneline", "git show HEAD",
        "git tag", "git blame nx/_risk.py", "git branch -a", "git remote -v",
        # 0.146.0: pipelines of allowlisted read-only stages are allowed
        "grep -rn TODO nx/ | wc -l", "ps aux | grep x",
        "cat a | head -20 | tail -5", "git log --oneline | head -5",
        "grep a f | sort | uniq -c", "ls -la | wc -l",
        # 0.146.0: globs allowed (read-only expansion)
        "grep TODO nx/*.py", "wc -l tests/*.py", "du -sh nx/*", "ls *",
        # quoted pipe is an argument character, not a pipeline separator
        "grep 'a|b' f",
    ]
    bad = [
        "", "   ",
        # write-shaped / mutating
        "rm -rf /", "mkdir d", "touch f", "echo x > f", "cp a b", "mv a b",
        "pytest -q", "dd if=/dev/zero of=x",
        # shell tricks (even between allowlisted commands) — 0.146.0 allows
        # pipelines of allowlisted stages, so only non-pipe tricks are bad
        "cat a && cat b", "ps aux; grep x",
        "git status; git log", "grep x > f", "ps aux & grep x",
        # `||` and command substitution still refused
        "cat a || cat b", "grep $(cat f) x", "grep `cat f` x",
        # (globs are now allowed — see the ok list)
        # ...but a pipeline containing a non-allowlisted or write stage is not
        "cat a | rm -rf /", "grep x f | python -c 'x'", "cat a ||| cat b",
        # find's destructive/exec forms (0.168.2: the off-by-one that let
        # `find -delete` skip checking its first argument is fixed — argv
        # including the head is scanned, matching _risk.py's own veto)
        "find . -delete", "find . -exec rm {} \\;", "find . -okdir x",
        # 0.168.2 (audit P1-3): in-argv write/exec escapes of allowlisted
        # binaries
        "find -fprint payload .", "find . -fprintf p '%p' -quit",
        "find . -fls out.txt", "find . -fprint0 out",
        "sort -o /tmp/payload f.txt", "sort --output=/tmp/payload f.txt",
        "rg --pre /bin/sh f.txt", "rg --pre-glob '*' f.txt",
        "rg -O /bin/ls f.txt", "rg --hostname-bin=/bin/sh f.txt",
        "grep -O /bin/sh f.txt",
        # arbitrary / unknown binaries
        # env-prefixing a disallowed binary must still refuse (head match only)
        "env curl evil.com", "env python x.py", "env rm -rf /",
        "env", "env | cat", "python x.py",
        "true",
        # git mutations and ambiguous forms
        "git push", "git commit -m x", "git branch -f x", "git remote add o url",
        "git checkout -b x", "git add .", "git stash", "git fetch", "git pull",
        "git merge x", "git clean -fd", "git reset --hard", "git diff HEAD~1",
    ]
    for c in ok:
        assert a(c), c
    for c in bad:
        assert not a(c), repr(c)


def test_subagent_bash_find_veto_matches_risk_classifier(nx_fresh):
    """0.168.2 (audit P1-3 meta-test): the find destructive-flag veto exists at
    two homes — `_subagent_bash.py` (allowlist gate) and `_risk.py:271`
    (classifier) — and they must scan the SAME argv slice (including the head)
    and the SAME flag set, else one home's guard silently lags the other's.
    That divergence was the root cause of the original `find -delete` escape."""
    import ast
    import inspect
    from nx import _risk, _subagent_bash

    def _veto_flags(module, func_name):
        src = inspect.getsource(getattr(module, func_name))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "find" in ast.unparse(node.test):
                # the flag tuple lives inside the test's any(...) — walk the
                # whole test expression, not just the head Compare
                for cmp_node in ast.walk(node.test):
                    if isinstance(cmp_node, (ast.Tuple, ast.List)):
                        return ast.literal_eval(cmp_node)
        raise AssertionError("find veto tuple not found in "
                             f"{module.__name__}.{func_name}")

    sub_flags = _veto_flags(_subagent_bash, "_subagent_bash_allowed")
    risk_flags = _veto_flags(_risk, "_local_risk_hint_bash")
    assert sub_flags == risk_flags


def test_subagent_bash_write_tier_unblocked(nx_fresh, monkeypatch):
    """0.103.17: a write/implement-tier agent gets the parent's real run_bash
    (its own _confirm gate) instead of the read-only allowlist — building and
    running the test suite is what a write agent exists to do. Read-tier
    agents keep the allowlist."""
    from nx import _subagent_tiers as t
    from nx._subagent import spawn_agent
    seen = {}

    def fake_run_bash(command="", **kw):
        seen["command"] = command
        return "ran"

    monkeypatch.setattr("nx._bash_jobs.run_bash", fake_run_bash)
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_WRITE", 1)

    def fake_loop(messages, *a, **kw):
        table = t._subagent_ctx().table
        # write tier: the gate steps aside, arbitrary commands reach run_bash
        assert table["run_bash"]("cargo check") == "ran"
        assert seen["command"] == "cargo check"
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("q", tier="write")
    assert not out.startswith("ERROR"), out


def test_subagent_bash_tier_dispatch(nx_fresh, monkeypatch):
    """Inside the sub-loop, run_bash resolves to the allowlist-gated wrapper:
    allowlisted commands reach run_bash, everything else is refused before it
    can run. The parent's DISPATCH entry stays unwrapped."""
    from nx import _dispatch
    from nx._subagent import spawn_agent, _read_only_dispatch
    ran = []

    def fake_run_bash(command="", **kw):
        ran.append(command)
        return f"ran:{command}"

    monkeypatch.setattr(nx_fresh._bash_jobs, "run_bash", fake_run_bash)
    table = _read_only_dispatch()
    assert table["run_bash"] is not _dispatch.DISPATCH["run_bash"]  # wrapped

    def fake_turn(messages, *a, **kw):
        ok = _dispatch.run_tool("run_bash", {"command": "grep -rn spawn nx/"})
        assert ok == "ran:grep -rn spawn nx/", ok
        ok2 = _dispatch.run_tool("run_bash", {"command": "ps aux"})
        assert ok2 == "ran:ps aux"
        for bad in ("rm -rf x", "echo hi > f", "git push", "ps aux; grep x",
                    "mkdir d", "find . -delete", "cat a | python -c 'x'"):
            out = _dispatch.run_tool("run_bash", {"command": bad})
            assert out.startswith("ERROR"), (bad, out)
        # the redirection case now dies at the write fence before the
        # read-only allowlist is reached (the fence layer runs first in
        # run_tool); the rest still come from the allowlist itself
        assert "read-only" in _dispatch.run_tool("run_bash", {"command": "mkdir d"})
        assert ran == ["grep -rn spawn nx/", "ps aux"]  # nothing else reached bash
        messages.append({"role": "assistant", "content": "bash-tier ok"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("scan") == "bash-tier ok"
    assert _dispatch.DISPATCH["run_bash"] is nx_fresh.run_bash  # parent untouched


def test_subagent_bash_nested_write_still_refused(nx_fresh, monkeypatch):
    """The bash tier must not widen the rest of the restriction: mutating
    tools stay unknown inside the sub-loop even when run_bash is allowed."""
    from nx import _dispatch
    from nx._subagent import spawn_agent

    def fake_turn(messages, *a, **kw):
        out = _dispatch.run_tool("write_file", {"path": "evil.txt", "content": "x"})
        assert "not available to the read-only sub-agent" in out
        out2 = _dispatch.run_tool("wait_background", {"pid": 1})
        assert "not available to the read-only sub-agent" in out2
        messages.append({"role": "assistant", "content": "still-readonly"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("q") == "still-readonly"
    assert not os.path.exists("evil.txt")


# --- concurrent spawn_agent fan-out -----------------------------------------

def test_concurrent_spawn_agents_overlap(nx_fresh, monkeypatch, tmp_path):
    """A 2-spawn_agent turn runs both agents concurrently, each with its own
    restricted dispatch table (thread-local), and results map back in order."""
    from nx import _dispatch
    from nx._turn import _should_parallelize, _run_read_only_parallel
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENTS", 1)  # run_tool() refuses a gated spawn (0.28.1); this tests the fan-out
    lock = threading.Lock()
    state = {"cur": 0, "max": 0}
    barrier = threading.Barrier(2, timeout=10)

    def fake_loop(messages, *a, **kw):
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        # per-agent restriction holds even while the sibling agent is mid-loop
        out = _dispatch.run_tool("write_file", {"path": "evil.txt", "content": "x"})
        assert "not available to the read-only sub-agent" in out
        try:
            barrier.wait(timeout=10)  # both agents genuinely in flight at once
        except threading.BrokenBarrierError:
            pass  # scheduler starved one worker; state["max"] check below still fails the test
        with lock:
            state["cur"] -= 1
        # messages[1] is the sub-agent's instruction, now suffixed with the
        # budget line — match on the instruction itself
        inst = messages[1]["content"].split("\n\n[Budget:")[0]
        messages.append({"role": "assistant", "content": "answer:" + inst})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    calls = [("spawn_agent", {"instruction": "question A"}),
             ("spawn_agent", {"instruction": "question B"})]
    assert _should_parallelize(calls) is True
    results = _run_read_only_parallel(calls)
    assert results[0] == "answer:question A"
    assert results[1] == "answer:question B"
    assert state["max"] == 2, state  # actually overlapped
    assert nx_fresh._state._SUBAGENT_ACTIVE is False
    assert nx_fresh._state._SUBAGENT_DISPATCH is None
    assert not os.path.exists("evil.txt")


def test_native_path_fans_out_two_spawns(nx_fresh, monkeypatch):
    """_handle_native_tool_calls routes a 2-spawn turn through the parallel
    executor: one assistant tool_calls turn, two ordered tool results."""
    from nx._turn import _handle_native_tool_calls
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENTS", 1)  # run_tool() refuses a gated spawn (0.28.1); this tests the fan-out
    tcs = {0: {"id": "a1", "name": "spawn_agent", "args": '{"instruction": "find A"}'},
           1: {"id": "a2", "name": "spawn_agent", "args": '{"instruction": "find B"}'}}

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "done:" + messages[1]["content"]})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    messages = [{"role": "system", "content": "s"}]
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == nx_fresh.TURN_TOOL
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "a1"
    assert tool_msgs[0]["content"].startswith("done:find A")
    assert tool_msgs[1]["content"].startswith("done:find B")


def test_should_parallelize_shapes(nx_fresh):
    from nx._turn import _should_parallelize
    # 2+ spawns, optionally with reads riding along -> parallel
    assert _should_parallelize([("spawn_agent", {}), ("spawn_agent", {})]) is True
    assert _should_parallelize([("spawn_agent", {}), ("spawn_agent", {}),
                                ("read_file", {"path": "a"})]) is True
    # single spawn + reads can overlap; mutations never parallel
    assert _should_parallelize([("spawn_agent", {}), ("read_file", {"path": "a"})]) is True
    assert _should_parallelize([("spawn_agent", {})]) is False
    # plain read batches unchanged; mutations never parallel
    assert _should_parallelize([("read_file", {"path": "a"}), ("search_files", {"pattern": "x"})]) is True
    assert _should_parallelize([("read_file", {"path": "a"}),
                                ("write_file", {"path": "b", "content": "c"})]) is False
    assert _should_parallelize([("spawn_agent", {}), ("write_file", {"path": "b", "content": "c"})]) is False


def test_no_nested_fanout_inside_subagent(nx_fresh, monkeypatch):
    """A sub-agent that itself emits 2+ spawn_agent calls stays sequential
    (they're refused one at a time) — no recursive fan-out."""
    from nx import _dispatch
    from nx._subagent import spawn_agent
    from nx._turn import _should_parallelize

    def fake_turn(messages, *a, **kw):
        calls = [("spawn_agent", {"instruction": "x"}), ("spawn_agent", {"instruction": "y"})]
        assert _should_parallelize(calls) is False  # TLS context says: inside a sub-agent
        # run_tool resolves against the restricted table first, which has no
        # spawn_agent entry — refused as unavailable before any nested-loop check
        out = _dispatch.run_tool("spawn_agent", {"instruction": "x"})
        assert out.startswith("ERROR") and "not available to the read-only sub-agent" in out
        messages.append({"role": "assistant", "content": "flat"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("q") == "flat"


def test_nested_spawn_agent_refused_serially(nx_fresh, monkeypatch):
    """Direct call inside a serial sub-loop: refused, without crashing the
    parent turn (the old _SUBAGENT_ACTIVE global check, now via TLS+global)."""
    from nx import _dispatch
    from nx._subagent import spawn_agent

    def fake_turn(messages, *a, **kw):
        out = _dispatch.run_tool("spawn_agent", {"instruction": "x"})
        assert out.startswith("ERROR") and "not available to the read-only sub-agent" in out
        messages.append({"role": "assistant", "content": "done"})
        return nx_fresh.TURN_DONE

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    assert spawn_agent("outer") == "done"


def test_subagent_system_and_toolset(nx_fresh):
    from nx._subagent import SUBAGENT_SYSTEM, _subagent_tools, _READ_ONLY_TOOL_NAMES
    assert "read-only" in SUBAGENT_SYSTEM.lower()
    tools = _subagent_tools()
    names = {t["function"]["name"] for t in tools}
    assert names == set(_READ_ONLY_TOOL_NAMES)
    assert "run_bash" in names  # read-only allowlisted tier — see test below
    forbidden = {"write_file", "edit_file", "multi_edit", "sed_edit",
                 "wait_background", "memory_write", "spawn_agent"}
    assert not (names & forbidden)


# --- headless --json trace pairing ------------------------------------------

def test_one_shot_result_pairs_multi_tool_results(nx_fresh):
    """_one_shot_result must attach each tool result to ITS call. A fan-out
    turn (2+ spawn_agent) emits one tool message per call — pairing on
    trace[-1] kept only the last sub-agent's answer."""
    from nx._cli import _one_shot_result
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [
            {"id": "a1", "type": "function",
             "function": {"name": "spawn_agent", "arguments": '{"instruction": "A"}'}},
            {"id": "a2", "type": "function",
             "function": {"name": "spawn_agent", "arguments": '{"instruction": "B"}'}},
        ]},
        {"role": "tool", "tool_call_id": "a1", "content": "answer A"},
        {"role": "tool", "tool_call_id": "a2", "content": "answer B"},
        {"role": "assistant", "content": "merged"},
    ]
    out = _one_shot_result(messages, 2, {"status": "done", "turns": 2}, 1.0)
    assert [t["result"] for t in out["tool_calls"]] == ["answer A", "answer B"]
    assert out["final_answer"] == "merged"


def test_one_shot_result_text_protocol_observations(nx_fresh):
    """Text-protocol fan-outs fold every observation into one user message of
    'Observation (name): result' blocks — pair them back to each call."""
    from nx._cli import _one_shot_result
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content":
         '<tool_call>{"name": "spawn_agent", "arguments": {"instruction": "A"}}</tool_call>'
         '<tool_call>{"name": "spawn_agent", "arguments": {"instruction": "B"}}</tool_call>'},
        {"role": "user", "content":
         "Observation (spawn_agent): answer A\nObservation (spawn_agent): answer B"},
    ]
    out = _one_shot_result(messages, 0, {"status": "done", "turns": 1}, 1.0)
    assert [t["result"] for t in out["tool_calls"]] == ["answer A", "answer B"]


def test_one_shot_result_single_tool_still_pairs(nx_fresh):
    from nx._cli import _one_shot_result
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [
            {"id": "x1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "f"}'}},
        ]},
        {"role": "tool", "tool_call_id": "x1", "content": "file body"},
    ]
    out = _one_shot_result(messages, 0, {"status": "done", "turns": 1}, 1.0)
    assert out["tool_calls"][0]["result"] == "file body"
    assert out["final_answer"] is None


def test_foreground_spawn_pauses_parent_spinner(nx_fresh, monkeypatch):
    """0.62.1: a foreground spawn_agent ran on the parent thread while the
    parent's run_tool LifeSpinner thread kept redrawing at 4 fps — its
    CLEAR_LINE rewrites interleaved with the sub-loop's own spinner / Live
    panel on the same cursor line, so both flickered. The sub-loop must stop
    the parent spinner for its duration and restart it after (same handshake
    _confirm() uses)."""
    from nx._ansi import LifeSpinner
    from nx._subagent import spawn_agent
    sp = LifeSpinner(label="running spawn_agent")
    events = []

    class TrackedSpinner(LifeSpinner):
        def start(self):
            events.append("start")
            if events.count("start") == 1:
                super().start()  # only the original start may actually thread

        def stop(self):
            events.append("stop")
            if events.count("stop") == 1:
                super().stop()

    tracked = TrackedSpinner(label="running spawn_agent")
    monkeypatch.setattr(nx_fresh._state, "_ACTIVE_SPINNER", tracked)

    def fake_loop(messages, *a, **kw):
        # mid-sub-loop: the parent spinner must be stopped NOW
        assert events and events[-1] == "stop"
        assert nx_fresh._state._ACTIVE_SPINNER is tracked  # note_progress still lands
        messages.append({"role": "assistant", "content": "done"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("q")
    assert out == "done"
    # spinner was paused for the loop and resumed after it
    assert events == ["stop", "start"]
    assert nx_fresh._state._ACTIVE_SPINNER is tracked  # run_tool still owns it

    # quiet (background worker) path: no parent spinner ownership, no pause
    from nx._subagent import _ASYNC_QUIET_SLOTS
    events.clear()
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)

    def quiet_loop(messages, *a, **kw):
        assert not events  # a quiet agent owns no spinner and pauses nothing
        messages.append({"role": "assistant", "content": "done"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", quiet_loop)
    out = spawn_agent("q")
    assert out == "done"
    assert events == []  # never touched a spinner it doesn't own
