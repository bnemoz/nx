"""Offline eval corpus + scored runs: `nx eval list` / `nx eval run` (P3-14).

The seed corpus is a set of SCRIPTED STUB-SERVER SCENARIOS replayed through the
real `nx run` batch harness — the same turn loop, recovery ladder, result-line
shape, and exit codes a live benchmark run produces, minus the model. Each
scenario's recording is a list of turn scripts served as OpenAI-style SSE by a
threaded stub HTTP server on a loopback ephemeral port; the batch harness runs
against it exactly as it would against a real endpoint, and each task is then
scored against its scenario's expectation (status, final answer, error shape,
post-run workspace file state).

What this measures is NOT model quality — the stub's replies are fixed — but
whether nx's harness and recovery stack still produce the right outcomes:
silent empty turns get nudged and recovered, collapsed streams get cut and
recovered, a mid-stream read timeout is reported as an honest `error` (never a
`done`), and tool calls with real schemas land real effects in a seeded
workspace directory.

Layout: recordings live in `<repo>/eval_corpus/` (tasks.jsonl listing the
scenario ids + instructions, and per-scenario `<id>.json` turn scripts) so a
new scenario is a file drop, not a code change. `nx eval run` regenerates the
task file for the harness, runs the batch inside a throwaway workspace under
`<out>/workspace/` (default `.nx/eval/` under the current directory — tasks
run with --yolo, so their real tool effects must stay confined there), writes
`results.jsonl` (verbatim harness output) and `summary.json` (scored), and
exits 0 when every scenario passes, 1 when any fails, 2 on misuse.

Continuous scoring (ROADMAP P3-14's second half) is a later band: this is the
seed corpus + the scoring contract it hangs on.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CORPUS_DIR = _REPO_ROOT / "eval_corpus"

# Mid-stream read timeout used for the run (the e-timeout scenario leans on
# it: the stub starts a stream, then stops delivering until the client gives
# up — P3-2's step ladder must report an honest error, never a done).
_EVAL_READ_TIMEOUT = 2.0
# Wall-clock backstop per task (a hung scenario must fail the task, not the
# harness). Carried on every task as its own timeout_seconds field.
_EVAL_TASK_TIMEOUT = 30.0

_USAGE_TEXT = """\
usage: nx eval list
       nx eval run [--source NAME] [--out DIR] [--parallel N] [--json]
                   [--filter ID]

Offline eval corpus (P3-14): recorded stub-server scenarios replayed through
the real `nx run` batch harness, each scored against its expectation.

verbs:
  list              print the corpus: one row per scenario
  run               run the corpus and score it

run flags:
  --source NAME     model-source name to shadow while the stub source is
                    registered (default: shadow the active source)
  --out DIR         artifact directory (default <cwd>/.nx/eval); a fresh
                    workspace/ is seeded inside it and tasks run there with
                    --yolo, so their real tool effects stay confined
  --parallel N      tasks run concurrently (default 1)
  --filter ID       run one scenario by id (still scored; a filtered run that
                    passes exits 0)
  --json            print the scored summary as JSON instead of a table

Exit codes: 0 all scenarios passed, 1 any failed, 2 harness misuse.
Artifacts: <out>/tasks.jsonl, <out>/results.jsonl (verbatim harness lines),
<out>/summary.json (scored), <out>/workspace/ (seeded task workspace).
"""


def _sse(obj: dict[str, Any]) -> str:
    return "data: " + json.dumps(obj) + "\n\n"


def _chunk(delta: dict[str, Any] | None = None, finish: str | None = None,
           usage: dict[str, Any] | None = None, rid: str = "chatcmpl-eval") -> dict[str, Any]:
    d = {"role": "assistant"} if delta is None else delta
    c: dict[str, Any] = {"id": rid, "object": "chat.completion.chunk",
                         "model": "eval-stub",
                         "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


def _tool_delta(idx: int, call_id: str, name: str, args: str) -> dict[str, Any]:
    return {"index": idx, "id": call_id, "type": "function",
            "function": {"name": name, "arguments": args}}


def _tc_chunks(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """One native tool call as streamed chunks: header delta, args split in
    two continuation deltas (id/name only on the first — the standard shape),
    finish_reason=tool_calls, then a usage-only final chunk."""
    a = json.dumps(args)
    mid = len(a) // 2
    return [
        _chunk({"tool_calls": [_tool_delta(0, "call_0", name, "")]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": a[:mid]}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": a[mid:]}}],
                "finish_reason": "tool_calls"}),
        _chunk(None, None, {"prompt_tokens": 120, "completion_tokens": 24}),
    ]


def _text_chunks(text: str) -> list[dict[str, Any]]:
    return [_chunk({"content": text}), _chunk(None, "stop"),
            _chunk(None, None, {"prompt_tokens": 100, "completion_tokens": 20})]


# Leaked tokenizer-internal special token (full-width pipe form) — 4 chunks
# crosses LEAKED_TOKEN_RUN_LIMIT (3) and cuts the stream mid-answer.
_GARBLE_TOKEN = "<｜begin▁of▁sentence｜>"


def _garble_chunks() -> list[dict[str, Any]]:
    return [_chunk({"content": _GARBLE_TOKEN}) for _ in range(4)]


def _render_turn(turn: dict[str, Any]) -> list[tuple[str, float]]:
    """One scripted turn → the SSE frames the stub serves for it: a list of
    (frame, hold) pairs, where frame is a ready-to-write SSE string and hold
    is seconds to stall before writing it (only the timeout scenario holds).
    Every turn ends with the terminal [DONE] marker; a cut (garble/timeout)
    scenario's stream simply ends client-side before it arrives."""

    def sse_all(chunks: list[dict[str, Any]]) -> list[tuple[str, float]]:
        return [(_sse(c), 0.0) for c in chunks]

    if "final" in turn:
        frames = sse_all(_text_chunks(turn["final"]))
    elif "tool_call" in turn:
        tc = turn["tool_call"]
        frames = sse_all(_tc_chunks(tc["name"], tc["args"]))
    elif turn.get("empty"):
        # No content, no tool calls, finish=stop → the silent-stall path.
        frames = sse_all([_chunk(None, "stop"),
                          _chunk(None, None, {"prompt_tokens": 80, "completion_tokens": 1})])
    elif turn.get("garble"):
        # Special-token collapse mid-answer — enough leaked-token chunks to
        # cross LEAKED_TOKEN_RUN_LIMIT. The loop_cut discards the degenerated
        # output; the recovery nudge re-opens the stream, so the next script
        # turn serves the clean recovery.
        frames = sse_all(_garble_chunks())
    elif "timeout" in turn:
        # One delivered chunk, then the server goes silent past the client's
        # read timeout (it holds the rest of the response open). The client
        # raises mid-stream — the honest-timeout scenario.
        frames = [(_sse(_chunk({"content": "payload flush in progress"})),
                   float(turn["timeout"]))]
    else:
        frames = sse_all(_text_chunks(""))
    frames.append(("data: [DONE]\n\n", 0.0))
    return frames


class _EvalStubServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        # The client legitimately hangs up mid-stream (a collapse/timeout cut,
        # or the process finishing): the resulting ConnectionReset is noise.
        import errno
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        if isinstance(exc, OSError) and exc.errno in (errno.EPIPE, errno.ECONNRESET):
            return
        super().handle_error(request, client_address)


class _EvalStubHandler(BaseHTTPRequestHandler):
    """Serves one scenario script: each POST /v1/chat/completions consumes the
    next scripted turn. The scenario is resolved from the REQUEST PAYLOAD — the
    task's instruction text is unique per scenario and rides in message[0] on
    every turn, including the re-opens after a recovery nudge — and the turn
    cursor is per scenario id, so one server serves a whole batch (--parallel
    included) without tasks stealing each other's turns. A script past its end
    serves its final turn forever, so a nudge loop can never 500.
    GET /v1/models answers for startup wiring."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # silence per-request stderr
        pass

    # --- request resolution ---------------------------------------------
    def _scenario_id(self, payload: dict[str, Any]) -> str | None:
        blob = "\n".join(str(m.get("content") or "")
                         for m in payload.get("messages") or [])
        for sid in _scenario_ids():
            if SCENARIOS[sid]["instruction"] in blob:
                return sid
        return None

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object": "list",
                               "data": [{"id": "eval-stub", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        sid = self._scenario_id(payload)
        if sid is None:
            self.send_error(400, "no eval scenario instruction in the request")
            return
        script = SCENARIOS[sid]["script"]
        with self.server._lock:  # type: ignore[attr-defined]
            idx = min(self.server._turns.get(sid, 0), len(script) - 1)  # type: ignore[attr-defined]
            self.server._turns[sid] = self.server._turns.get(sid, 0) + 1  # type: ignore[attr-defined]
        turn = script[idx]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        # One turn -> the SSE frames it renders. _render_turn returns a list of
        # (text, hold) pairs: text is one SSE frame or the terminal chunked
        # marker, hold is seconds to stall BEFORE that frame (the timeout
        # scenario delivers its first chunk, then holds the rest open).
        try:
            for frame, hold in _render_turn(turn):
                if hold:
                    time.sleep(float(hold))
                # HTTP/1.1 chunked framing: <hex size>\r\n<data>\r\n
                data = frame.encode("utf-8")
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")  # terminal chunk
            self.wfile.flush()
        except Exception:
            return  # client hung up mid-stream; nothing useful to serve



def _stub_server() -> tuple[_EvalStubServer, str]:
    srv = _EvalStubServer(("127.0.0.1", 0), _EvalStubHandler)
    srv._turns = {}       # scenario id -> turns served (shared with handlers)
    srv._lock = threading.Lock()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


# --- the seed corpus ---------------------------------------------------------
# Workspace files are seeded fresh for every run; tool calls use real nx tool
# names and real argument names, so the calls EXECUTE against the workspace
# (the recorded args are what the stub replays, the results are nx's own).

_WORKSPACE_FILES: dict[str, str] = {
    "README.md": "# eval workspace\n\nSeeded by `nx eval run`; tasks operate here.\n",
    "eval_src/util.py": ('"""Small helpers for the eval workspace."""\n\n\n'
                         "def load_payload():\n"
                         '    return {"payload_raw": "seed"}\n\n\n'
                         "def render():\n"
                         "    payload_raw = load_payload()\n"
                         '    return f"raw={payload_raw!r}"\n'),
    "notes/handoff.md": ("# Handoff\n\nThe queued payload was never flushed "
                         "after the restart.\n"),
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "e-happy": {
        "title": "one tool call, then a clean final answer",
        "instruction": "Survey the eval workspace with list_dir and report what is present.",
        "script": [
            {"tool_call": {"name": "list_dir", "args": {"path": "."}}},
            {"final": "Eval run complete: the workspace is ready and holds the notes directory."},
        ],
        "expect": {"status": "done", "final_contains": "the workspace is ready"},
    },
    "e-oneedit": {
        "title": "one real write_file in the workspace",
        "instruction": "Create eval_write.txt in the workspace root marking that the eval ran.",
        "script": [
            {"tool_call": {"name": "write_file",
                           "args": {"path": "eval_write.txt",
                                    "content": "written by the nx eval corpus\n"}}},
            {"final": "Wrote eval_write.txt; the workspace now carries the marker file."},
        ],
        "expect": {"status": "done", "final_contains": "eval_write.txt"},
        "expect_file": {"path": "eval_write.txt", "contains": "nx eval corpus"},
    },
    "e-refactor": {
        "title": "read, then a real edit_file rename in a seeded file",
        "instruction": "In eval_src/util.py rename the local variable payload_raw to body_raw.",
        "script": [
            {"tool_call": {"name": "read_file", "args": {"path": "eval_src/util.py"}}},
            {"tool_call": {"name": "edit_file",
                           "args": {"path": "eval_src/util.py",
                                    "old_string": "payload_raw = load_payload()",
                                    "new_string": "body_raw = load_payload()"}}},
            {"final": "Refactor applied: payload_raw renamed to body_raw in eval_src/util.py."},
        ],
        "expect": {"status": "done", "final_contains": "body_raw"},
        "expect_file": {"path": "eval_src/util.py", "contains": "body_raw = load_payload()",
                        "not_contains": "payload_raw = load_payload()"},
    },
    "e-detect": {
        "title": "explore with real tools, then report what the files say",
        "instruction": "Find out what the handoff note in the notes directory says and report it.",
        "script": [
            {"tool_call": {"name": "list_dir", "args": {"path": "."}}},
            {"tool_call": {"name": "read_file", "args": {"path": "notes/handoff.md"}}},
            {"final": "The handoff note says the queued payload was never flushed after the restart."},
        ],
        "expect": {"status": "done", "final_contains": "queued payload was never flushed"},
    },
    "e-empty": {
        "title": "two silent empty turns, then the recovery nudge lands",
        "instruction": "The queued payload is stuck; flush it and confirm the flush.",
        "script": [
            {"empty": True},
            {"empty": True},
            {"final": "Recovered after 2 empty turns: the queued payload was flushed."},
        ],
        "expect": {"status": "done", "final_contains": "Recovered after 2 empty turns"},
    },
    "e-garble": {
        "title": "special-token collapse mid-answer, then clean recovery",
        "instruction": "Flush the pending queue and confirm it was flushed.",
        "script": [
            {"garble": True},
            {"final": "Recovered from the special-token collapse; the queued payload was flushed."},
        ],
        "expect": {"status": "done", "final_contains": "Recovered from the special-token collapse"},
    },
    "e-timeout": {
        "title": "mid-stream read timeout scores as an honest error (never done)",
        "instruction": "Flush the payload queue now and confirm it flushed.",
        "script": [
            {"timeout": 10},
        ],
        # The stream-timeout path reports status "error" naming the timeout;
        # `timed_out: true` is the wall-clock path's marker (a different, also
        # honest, failure) — this scenario pins the P3-2 contract: a mid-stream
        # timeout is an error, never a done.
        "expect": {"status": "error", "error_contains": "timeout"},
    },
}


def _scenario_ids() -> list[str]:
    ids = sorted(SCENARIOS)
    return [i for i in ids if not i.startswith("_")]


def _eval_list() -> int:
    sys.stdout.write("nx eval corpus (%d scenarios, stub-server recordings)\n\n" % len(_scenario_ids()))
    sys.stdout.write(f"{'id':<12} {'turns':<6} expectation\n")
    for sid in _scenario_ids():
        sc = SCENARIOS[sid]
        exp = sc["expect"]
        want = f"status={exp.get('status', 'done')}"
        if exp.get("final_contains"):
            want += f", final~{exp['final_contains']!r}"
        if exp.get("error_contains"):
            want += f", error~{exp['error_contains']!r}"
        if exp.get("timed_out"):
            want += ", timed_out"
        sys.stdout.write(f"{sid:<12} {len(sc['script']):<6} {want}\n"
                         f"{'':<12} {'':<6} {sc['title']}\n")
    return 0


def _score(result: dict[str, Any], expect: dict[str, Any]) -> list[dict[str, Any]]:
    """Check one result line against its scenario's expectation."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    want_status = expect.get("status", "done")
    got_status = result.get("status")
    add("status", got_status == want_status, f"status={got_status!r} (want {want_status!r})")
    fc = expect.get("final_contains")
    if fc:
        fa = result.get("final_answer") or ""
        add("final_contains", fc in fa, f"final_answer={fa!r:.120}")
    ec = expect.get("error_contains")
    if ec:
        err = (result.get("error") or "").lower()
        add("error_contains", ec in err, f"error={result.get('error')!r}")
    if expect.get("timed_out"):
        add("timed_out", result.get("timed_out") is True,
            f"timed_out={result.get('timed_out')!r}")
    return checks


def _check_workspace(ws: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    p = ws / spec["path"]
    checks: list[dict[str, Any]] = []
    if not p.is_file():
        checks.append({"name": "expect_file", "ok": False,
                       "detail": f"{spec['path']} not found in the workspace"})
        return checks
    text = p.read_text(encoding="utf-8")
    ok = spec.get("contains", "") in text
    if ok and spec.get("not_contains"):
        ok = spec["not_contains"] not in text
    detail = spec["path"]
    if not ok:
        detail += f": missing {spec.get('contains')!r}"
        if spec.get("not_contains") and spec["not_contains"] in text:
            detail += f" (still contains {spec['not_contains']!r})"
    checks.append({"name": "expect_file", "ok": ok, "detail": detail})
    return checks


def _eval_run(argv: list[str]) -> int:
    from ._cli import _cli_version

    source_name: str | None = None
    out_dir: str | None = None
    parallel = 1
    as_json = False
    only: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--source",) and i + 1 < len(argv):
            source_name = argv[i + 1]; i += 2
        elif a in ("--out",) and i + 1 < len(argv):
            out_dir = argv[i + 1]; i += 2
        elif a in ("--filter",) and i + 1 < len(argv):
            only = argv[i + 1]; i += 2
        elif a == "--parallel" and i + 1 < len(argv):
            try:
                parallel = max(1, int(argv[i + 1]))
            except ValueError:
                sys.stderr.write("nx eval: --parallel wants an integer\n")
                return 2
            i += 2
        elif a == "--json":
            as_json = True; i += 1
        else:
            sys.stderr.write(f"nx eval: unknown flag {a!r}\n")
            sys.stderr.write(_USAGE_TEXT)
            return 2
    if only and only not in SCENARIOS:
        sys.stderr.write(f"nx eval: unknown scenario {only!r} "
                         f"(known: {', '.join(_scenario_ids())})\n")
        return 2

    import nx as _m
    from . import _state as st
    from ._batch import run_batch

    out = Path(out_dir or os.path.join(os.getcwd(), ".nx", "eval")).resolve()
    ws = out / "workspace"
    out.mkdir(parents=True, exist_ok=True)
    if ws.exists():
        import shutil
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    ids = [only] if only else _scenario_ids()
    # Task file for the harness: one object per scenario, deterministic ids.
    tasks_path = out / "tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as f:
        for sid in ids:
            f.write(json.dumps({"id": sid, "instruction": SCENARIOS[sid]["instruction"],
                                "timeout_seconds": _EVAL_TASK_TIMEOUT}) + "\n")

    server, base_url = _stub_server()
    t = threading.Thread(target=server.serve_forever, daemon=True,
                         name="nx-eval-stub")
    t.start()

    # Shadow the active source with the stub; everything saved is restored in
    # the finally so an in-process caller (the test suite) keeps its state.
    saved_active, saved_client, saved_model = st.ACTIVE, st.client, st.MODEL
    saved_sources = dict(st.SOURCES)
    saved_order = list(st.SOURCE_ORDER)
    saved_env = {k: os.environ.get(k) for k in
                 ("NX_HTTP_TIMEOUT_READ", "NX_APPROVAL")}
    saved_yolo = st.YOLO
    saved_level = st.DEFAULT_APPROVE_LEVEL
    saved_approve = st.APPROVE_LEVEL
    saved_cwd = os.getcwd()
    code = 2
    results_lines: list[dict[str, Any]] = []
    try:
        os.environ["NX_HTTP_TIMEOUT_READ"] = str(_EVAL_READ_TIMEOUT)
        # The eval's contract is that the scripted tool calls EXECUTE (that's
        # what expect_file scores). Batch is headless — with the default
        # prompt-all posture every gated call is auto-denied and the corpus
        # would measure denial, not the harness. Run --yolo, but confined:
        # the batch runs inside the throwaway workspace, so the effects stay
        # out of the caller's cwd. Ambient NX_APPROVAL must not win (ambient
        # yolo is redundant; anything else would re-gate the corpus).
        os.environ["NX_APPROVAL"] = "yolo"
        src = st.SOURCES.get("eval-stub") or _m._sources.Source(
            "eval-stub", base_url, "sk-eval", "eval-stub")
        src.base_url = base_url
        # Recreate the client so the base_url (ephemeral port) is current.
        src.client = _m._sources.OpenAI(base_url=base_url, api_key="sk-eval")
        src._context_window = None
        src._context_window_probe_attempted = True  # no probe: this is a stub
        st.SOURCES["eval-stub"] = src
        if "eval-stub" not in st.SOURCE_ORDER:
            st.SOURCE_ORDER.append("eval-stub")
        st.ACTIVE = src
        st.client = src.client
        st.MODEL = "eval-stub"
        st.YOLO = True
        st.APPROVE_LEVEL = None
        st.DEFAULT_APPROVE_LEVEL = None

        # Seed the workspace, then run the batch INSIDE it: tasks run with
        # --yolo (headless), so confining them to the workspace keeps their
        # real tool effects out of the caller's cwd.
        for rel, content in _WORKSPACE_FILES.items():
            p = ws / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        os.chdir(ws)

        results_path = out / "results.jsonl"
        with open(results_path, "w", encoding="utf-8") as rf:
            with contextlib.redirect_stdout(rf):
                code = run_batch(["-f", str(tasks_path),
                                  "--parallel", str(parallel),
                                  "--continue-on-error", "--yolo", "--json"])
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                results_lines.append(json.loads(line))

        rows = _score_results(results_lines, ids, ws)
        summary = {"tool": "nx eval", "nx_version": _cli_version(),
                   "scenarios": len(ids), "passed": sum(1 for r in rows if r["pass"]),
                   "failed": sum(1 for r in rows if not r["pass"]),
                   "harness_exit_code": code, "out_dir": str(out), "results": rows}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
        if as_json:
            sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
        else:
            _print_report(summary)
        code = 0 if summary["failed"] == 0 and summary["passed"] > 0 else 1
    finally:
        os.chdir(saved_cwd)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        st.SOURCES.clear(); st.SOURCES.update(saved_sources)
        st.SOURCE_ORDER[:] = saved_order
        st.ACTIVE, st.client, st.MODEL = saved_active, saved_client, saved_model
        st.YOLO = saved_yolo
        st.APPROVE_LEVEL = saved_approve
        st.DEFAULT_APPROVE_LEVEL = saved_level
        server.shutdown()
        server.server_close()
    return code


def _score_results(results: list[dict[str, Any]], ids: list[str],
                   ws: Path) -> list[dict[str, Any]]:
    by_id = {r.get("id"): r for r in results}
    rows = []
    for sid in ids:
        sc = SCENARIOS[sid]
        res = by_id.get(sid, {"status": "missing", "error": "no result line"})
        checks = _score(res, sc["expect"])
        spec = sc.get("expect_file")
        if spec:
            checks.extend(_check_workspace(ws, spec))
        fa = res.get("final_answer") or ""
        rows.append({"id": sid, "title": sc["title"], "pass": all(c["ok"] for c in checks),
                     "status": res.get("status"), "exit_code": res.get("exit_code"),
                     "turns": res.get("turns"), "timed_out": res.get("timed_out"),
                     "elapsed_seconds": res.get("elapsed_seconds"),
                     "tokens": res.get("tokens"), "error": res.get("error"),
                     "final_answer": fa if len(fa) <= 200 else fa[:197] + "...",
                     "checks": checks})
    return rows


def _print_report(summary: dict[str, Any]) -> None:
    sys.stdout.write(f"nx eval: {summary['passed']}/{summary['scenarios']} passed\n\n")
    for r in summary["results"]:
        mark = "PASS" if r["pass"] else "FAIL"
        line = f"  {mark} {r['id']:<12} status={r['status']} turns={r['turns']}"
        if r.get("elapsed_seconds") is not None:
            line += f" {r['elapsed_seconds']:.2f}s"
        sys.stdout.write(line + "\n")
        if not r["pass"]:
            for c in r["checks"]:
                if not c["ok"]:
                    sys.stdout.write(f"       ✗ {c['name']}: {c['detail']}\n")
    sys.stdout.write(f"\nartifacts: {summary['out_dir']}"
                     " (results.jsonl, summary.json, workspace/)\n")


def _eval_main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_USAGE_TEXT)
        return 0 if argv else 2
    verb = argv[0]
    if verb == "list":
        return _eval_list()
    if verb == "run":
        return _eval_run(argv[1:])
    sys.stderr.write(f"nx eval: unknown verb {verb!r} (want list|run)\n")
    sys.stderr.write(_USAGE_TEXT)
    return 2
