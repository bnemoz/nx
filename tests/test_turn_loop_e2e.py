"""Turn-loop end-to-end regression (phase 2): the REAL loop over the REAL HTTP
client against a loopback stub SSE server.

Why this file exists: `nx/_turn.py` is the most bug-prone module in the tree
and the recurring bug class is streaming/SSE/retry/timeout (SSE line-splitting
truncating tool args, the fail-fast timeout layer, honest mid-stream timeout,
say-panel re-echo after a mid-stream tool call). Every other test fakes either
`model_turn` or `open_stream` piecemeal, so nothing drove the real loop
through the real transport and pinned the two invariants that matter:

  (a) message-history validity — every tool message answers a tool_calls id
      actually issued, roles well-formed, no half-parsed tool call survives a
      cut stream (a poisoned history re-opens a dead conversation);
  (b) the honest-status contract — a mid-stream cut must NEVER yield status
      "done" / TURN_DONE.

Three scenarios, all through the SAME stub-server helper (module-local, not a
conftest change; pattern lifted from tests/test_eval_corpus.py's harness):
  1. happy arc: tool call (run_bash) -> tool result -> final text.
     Pins history validity, the tool's REAL effect on disk (cwd = tmp_path),
     status TURN_DONE, and zero residue (wake_due() false, no _PENDING_* note,
     _TOOL_RESULT_CACHE unremarkable).
  2. mid-stream aborted connection: valid role delta + partial tool-call args,
     then the server closes WITHOUT [DONE]. Pins honest error status, the
     transport cause named in the error, and no poisoned re-open.
  3. mid-stream HTTP error frame: valid deltas then an error frame /
     HTTP-level failure consistent with nx/_stream_errors.py classification.
     Same honest-status assertions as 2.

No monkeypatching of open_stream/model_turn: the loop talks to the stub via
the real open_stream/httpx path. Runs offline on 127.0.0.1:0 (ephemeral).
"""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")
for _k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(_k, None)

import nx as m  # noqa: E402
from nx import _state as st  # noqa: E402
from nx._turn_status import TURN_DONE, TURN_GAVE_UP  # noqa: E402

INSTRUCTION = ("e2e-turn-loop: run `echo ok > e2e_marker.txt` with run_bash, "
               "then confirm the marker file exists.")


# ---------------------------------------------------------------------------
# SSE chunk builders (OpenAI-style wire shape, same as the eval-corpus stub)
# ---------------------------------------------------------------------------

def _chunk(delta=None, finish=None, usage=None):
    d = {"role": "assistant"} if delta is None else delta
    c = {"id": "chatcmpl-e2e", "object": "chat.completion.chunk",
         "model": "e2e-stub",
         "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


def _sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


def _text_frames(text):
    """One text turn: content deltas (split mid-word — the 0.56.1
    line-splitting class), stop, usage, [DONE]."""
    mid = max(1, len(text) // 2)
    return [_sse(_chunk({"content": text[:mid]})),
            _sse(_chunk({"content": text[mid:]})),
            _sse(_chunk(None, "stop")),
            _sse(_chunk(None, None, {"prompt_tokens": 100,
                                     "completion_tokens": 20})),
            "data: [DONE]\n\n"]


def _tool_call_frames(name, args, preamble=None):
    """One native tool call streamed the standard way: optional text deltas
    first (the say-panel-before-tool-call path — the re-echo bug class), then
    the header delta carrying id/name, args in three continuation fragments
    (exercises the real SSE line-reassembly path end to end),
    finish=tool_calls, usage, [DONE]."""
    a = json.dumps(args)
    p1, p2 = len(a) // 3, 2 * (len(a) // 3)
    head = [_sse(_chunk({"content": preamble}))] if preamble else []
    return head + [_sse(_chunk({"tool_calls": [{"index": 0, "id": "call_e2e_0",
                                         "type": "function",
                                         "function": {"name": name,
                                                      "arguments": ""}}]})),
            _sse(_chunk({"tool_calls": [{"index": 0, "function":
                                         {"arguments": a[:p1]}}]})),
            _sse(_chunk({"tool_calls": [{"index": 0, "function":
                                         {"arguments": a[p1:p2]}}]})),
            _sse(_chunk({"tool_calls": [{"index": 0, "function":
                                         {"arguments": a[p2:]}}],
                         "finish_reason": "tool_calls"})),
            _sse(_chunk(None, None, {"prompt_tokens": 120,
                                     "completion_tokens": 24})),
            "data: [DONE]\n\n"]


# ---------------------------------------------------------------------------
# The stub server: ThreadingHTTPServer on 127.0.0.1:0 serving a per-test
# scripted list of turns. Each turn is a dict:
#   {"frames": [...]}                  -> stream the frames, terminal chunk
#   {"cut_after": n}                   -> stream n frames then DROP the
#                                         connection mid-stream (no [DONE], no
#                                         terminal chunked marker)
#   {"http_error": (status, body)}     -> answer with an HTTP-level failure
# A script past its end re-serves its LAST turn, so a recovery loop can never
# deadlock against the stub.
# ---------------------------------------------------------------------------

class _StubServer:
    """Owns the ThreadingHTTPServer + its serve thread + the turn cursor."""

    def __init__(self, script):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    body = json.dumps({"object": "list", "data": []}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
                with outer.lock:
                    outer.requests.append(payload)
                    idx = min(outer.cursor, len(outer.script) - 1)
                    outer.cursor += 1
                turn = outer.script[idx]
                if "http_error" in turn:
                    status, body = turn["http_error"]
                    data = json.dumps(body).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                frames = turn["frames"]
                cut_at = turn.get("cut_after")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    out = frames if cut_at is None else frames[:cut_at]
                    for frame in out:
                        data = frame.encode("utf-8")
                        self.wfile.write(f"{len(data):x}\r\n".encode()
                                         + data + b"\r\n")
                        self.wfile.flush()
                    if cut_at is not None:
                        # Mid-stream abort WITHOUT [DONE]: close the socket
                        # with the chunked body unfinished — the client sees a
                        # transport-level failure (incomplete read), exactly
                        # what a proxy/upstream reset looks like.
                        self.close_connection = True
                        try:
                            self.connection.close()
                        except Exception:
                            pass
                        return
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    return  # client hung up; nothing useful left to serve

        class _Srv(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

            def handle_error(self, request, client_address):
                import errno
                exc = sys.exc_info()[1]
                if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
                    return
                if isinstance(exc, OSError) and exc.errno in (
                        errno.EPIPE, errno.ECONNRESET):
                    return

        self.script = script
        self.requests = []
        self.cursor = 0
        self.lock = threading.Lock()
        self.srv = _Srv(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.srv.serve_forever,
                                       daemon=True, name="nx-e2e-stub")

    def start(self):
        self.thread.start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


def _stub_env(server, tmp_path, monkeypatch):
    """Shadow the active source with the stub and run with cwd in tmp_path.
    Mirrors _eval_cmd._eval_run's shadow/restore, via monkeypatch so pytest
    unwinds every piece (env, cwd, st.ACTIVE/client/MODEL, sources, yolo)."""
    src = m._sources.Source("e2e-stub", server.url, "sk-e2e", "e2e-stub")
    src.base_url = server.url
    src.client = m._sources.OpenAI(base_url=server.url, api_key="sk-e2e")
    src._context_window = None
    src._context_window_probe_attempted = True  # no probe: this is a stub
    monkeypatch.setitem(st.SOURCES, "e2e-stub", src)
    monkeypatch.setattr(st, "SOURCE_ORDER", list(st.SOURCE_ORDER) + ["e2e-stub"])
    monkeypatch.setattr(st, "ACTIVE", src)
    monkeypatch.setattr(st, "client", src.client)
    monkeypatch.setattr(st, "MODEL", "e2e-stub")
    # Headless posture (the batch harness's own): tools execute, nobody to
    # approve, no spinner/stream/footer.
    monkeypatch.setattr(st, "YOLO", True)
    monkeypatch.setattr(st, "APPROVE_LEVEL", None)
    monkeypatch.setattr(st, "DEFAULT_APPROVE_LEVEL", None)
    from nx._subagent import _ASYNC_QUIET_SLOTS
    # raising=False: the TLS slot only exists once a quiet context set it
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)
    monkeypatch.setenv("NX_APPROVAL", "yolo")
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    monkeypatch.chdir(ws)
    return ws


def _drive_turn():
    """Drive the REAL loop (nx._run_model_turn_loop -> nx.model_turn ->
    open_stream -> the vendored httpx client) against the stub source.
    Returns (summary, messages)."""
    messages = m._initial_messages() + [{"role": "user", "content": INSTRUCTION}]
    summary = m._run_model_turn_loop(messages, on_step=None)
    return summary, messages


def _assert_history_valid(messages, start_idx):
    """(a) message-history validity for everything the turn appended: every
    tool message answers a tool_calls id actually issued by an assistant
    message before it; roles well-formed; tool_calls well-formed."""
    issued = set()
    for msg in messages[start_idx:]:
        role = msg.get("role")
        assert role in ("assistant", "tool", "user", "system"), msg
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                assert tc.get("id"), msg
                fn = tc.get("function") or {}
                assert fn.get("name"), msg
                json.loads(fn.get("arguments") or "{}")  # complete, parseable
                issued.add(tc["id"])
        elif role == "tool":
            # every observation answers a call that was actually issued
            assert msg.get("tool_call_id") in issued, msg
    return issued


# ---------------------------------------------------------------------------
# 1. happy arc through the real loop
# ---------------------------------------------------------------------------

def test_e2e_happy_arc_through_real_loop(tmp_path, monkeypatch):
    server = _StubServer([
        {"frames": _tool_call_frames(
            "run_bash", {"command": "echo ok > e2e_marker.txt"},
            preamble="Let me create the marker first.")},
        {"frames": _text_frames("The marker file exists: it holds 'ok'.")},
    ])
    server.start()
    try:
        ws = _stub_env(server, tmp_path, monkeypatch)
        start_idx = 2  # system + user
        summary, messages = _drive_turn()
        # honest happy path: the DONE status from _turn_status
        assert summary["status"] == TURN_DONE, summary
        # history validity over everything the turn appended
        _assert_history_valid(messages, start_idx)
        # the loop really issued the tool call and its observation
        tool_msgs = [x for x in messages[start_idx:] if x.get("role") == "tool"]
        assert tool_msgs, messages
        # the tool's REAL effect landed in the task's cwd
        marker = ws / "e2e_marker.txt"
        assert marker.is_file(), f"{marker} missing; ws={list(ws.iterdir())}"
        assert marker.read_text().strip() == "ok"
        # the final answer is the scripted text
        assert "marker file exists" in messages[-1]["content"]
        # --- no residue after the turn ---
        from nx._autowake import wake_due
        assert not wake_due()
        from nx._notices import get as notice_get
        from _nxtest import _pending_slots
        # _PENDING_* notice slots only: _BG_SCAN_DONE/_BG_NOTICE_SEEN are
        # sweep bookkeeping the completion scan legitimately flips, not
        # undelivered notes.
        for slot in _pending_slots(st):
            if not slot.startswith("_PENDING_"):
                continue
            assert not notice_get(slot), f"pending notice residue: {slot}"
        # tool-result cache state unremarkable: the happy arc reads nothing,
        # so no read-result entries were cached (run_bash results are not
        # cached by path).
        assert not st._TOOL_RESULT_CACHE
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# 2. mid-stream aborted connection -> honest error, never done
# ---------------------------------------------------------------------------

def _assert_honest_cut(summary, messages, start_idx):
    """The honest-status contract for a mid-stream cut: the task reports an
    ERROR naming the stream/timeout/transport cause — never done — and the
    history was not extended with a half-parsed assistant tool call that
    would poison a re-open."""
    assert summary["status"] != TURN_DONE, (
        "MID-STREAM CUT REPORTED AS DONE — the honest-status contract "
        f"(P3-2 lineage) is violated: {summary}")
    assert summary["status"] == "error", summary
    err = (summary.get("error") or "").lower()
    assert ("stream" in err or "timeout" in err or "connection" in err
            or "transport" in err or "incomplete" in err
            or "protocol" in err or "api request failed" in err), summary
    # No half-parsed assistant tool-call survived the cut (poisoned re-open).
    for msg in messages[start_idx:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                args = (tc.get("function") or {}).get("arguments")
                json.loads(args or "{}")  # must be COMPLETE, not truncated
                raise AssertionError(
                    f"cut stream appended a tool call to the history: {tc}")


# Pinned invariant (honest-result rule): a mid-stream connection abort is NOT
# an httpx timeout, so it never reaches the P3-2 TURN_TIMED_OUT path — it falls
# into model_turn()'s malformed-stream ladder, and once that ladder is exhausted
# the turn must NOT be reported as a completed one. Since 0.154.6 the ladder
# exhaustion returns the honest TURN_GAVE_UP and the loop reports status "error"
# naming the transport cause, so this test asserts directly (no xfail). The
# history is also NOT poisoned: no half-parsed tool call survives the cut.
def test_e2e_midstream_aborted_connection_honest_error(tmp_path, monkeypatch):
    # A valid stream start — the tool-call header delta + a PARTIAL args
    # fragment — then the connection dies WITHOUT [DONE]. If the loop
    # reported done here, a benchmark would score a truncated run as a pass.
    frames = [
        _sse(_chunk({"tool_calls": [{"index": 0, "id": "call_cut_0",
                                     "type": "function",
                                     "function": {"name": "run_bash",
                                                  "arguments": "{\"comm"}}]})),
        _sse(_chunk({"tool_calls": [{"index": 0, "function":
                                     {"arguments": "and\": \"echo hal"}}]})),
    ]
    server = _StubServer([{"frames": frames, "cut_after": len(frames)}])
    server.start()
    try:
        _stub_env(server, tmp_path, monkeypatch)
        summary, messages = _drive_turn()
        _assert_honest_cut(summary, messages, start_idx=2)
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# 4. REPL-side half of the principle: a finished turn is reported by the same
#    TURN_* vocabulary in BOTH modes, never by a headless-only branch.
# ---------------------------------------------------------------------------

def test_repl_side_exhausted_ladder_is_not_a_done_turn(tmp_path, monkeypatch):
    """ONE loop, ONE vocabulary: the honest status is produced by model_turn()
    itself, so the REPL caller only has to end the turn and return to the
    prompt (user takes over). The bug was that this gave-up state was reported
    as TURN_DONE — indistinguishable from a completed task — and the REPL only
    looked benign by accident.

    Exhausts the malformed-stream ladder (same cut script as scenario 2) and
    pins the vocabulary the REPL consumes:
      • model_turn() returns TURN_GAVE_UP, NOT TURN_DONE;
      • the turn loop reports a non-"done" summary (a REPL caller ends the
        turn either way — no new user-visible state);
      • the recorded cause names the transport failure, not silence.
    """
    from nx._turn import _LAST_GIVE_UP_CAUSE
    frames = [
        _sse(_chunk({"tool_calls": [{"index": 0, "id": "call_cut_0",
                                     "type": "function",
                                     "function": {"name": "run_bash",
                                                  "arguments": "{\"comm"}}]})),
        _sse(_chunk({"tool_calls": [{"index": 0, "function":
                                     {"arguments": "and\": \"echo hal"}}]})),
    ]
    server = _StubServer([{"frames": frames, "cut_after": len(frames)}])
    server.start()
    try:
        _stub_env(server, tmp_path, monkeypatch)
        # (a) the vocabulary model_turn() itself returns on the give-up call —
        # the retry budget is already spent (malformed_stream_cut_count at the
        # limit), which is exactly the loop's last attempt.
        limit = max(0, st.MALFORMED_STREAM_RETRY_LIMIT)
        messages = m._initial_messages() + [{"role": "user", "content": INSTRUCTION}]
        status = m.model_turn(messages, 0, limit)
        assert status != TURN_DONE, (
            "an exhausted-ladder give-up is not a completed turn — the REPL "
            f"would return to the prompt believing the task finished: {status!r}")
        assert status == TURN_GAVE_UP, status
        assert _LAST_GIVE_UP_CAUSE, "the give-up cause must be recorded for the summary"
        # (b) the same driver path the other scenarios use: the summary the
        # REPL loop consumes is not "done" either.
        summary, _msgs = _drive_turn()
        assert summary["status"] != TURN_DONE, summary
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# 3. mid-stream HTTP error frame -> honest error, never done
# ---------------------------------------------------------------------------

# Same pinned invariant as scenario 2: the error frame + stream drop exhausts
# the malformed-stream ladder, and since 0.154.6 that exhaustion returns the
# honest TURN_GAVE_UP (loop -> status "error"), never a done turn.
def test_e2e_midstream_http_error_frame(tmp_path, monkeypatch):
    # After valid deltas the stub emits an SSE error frame and drops the
    # stream without [DONE] — the shape nx/_stream_errors.py classifies as a
    # busy/broken upstream (not a request-field rejection: no fallback rung
    # can help, it must surface honestly).
    frames = (
        [_sse(_chunk({"content": "Working on it, flushing the "})),
         _sse(_chunk({"content": "payload now — "}))]
        + [_sse({"error": {"message": "upstream GPU OOM; stream aborted",
                           "type": "server_error", "code": "500"}})]
    )
    server = _StubServer([{"frames": frames, "cut_after": len(frames)}])
    server.start()
    try:
        _stub_env(server, tmp_path, monkeypatch)
        summary, messages = _drive_turn()
        _assert_honest_cut(summary, messages, start_idx=2)
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# 5. cap-rescue forced-final turn that answers NOTHING -> honest error, cap
# ---------------------------------------------------------------------------

# Same pinned invariant, at the forced-final rescue seam: the turn-limit
# rescue asks the model for its final answer and it emits an EMPTY
# final_answer (no answer text, no actionable call). The handler used to
# return TURN_DONE, so the loop reported the run as a clean "cap"; since
# 0.154.7 it returns the honest TURN_GAVE_UP and the loop maps it to status
# "error" with the stream-cut/give-up message — a silent non-answer is not a
# result. (The final_answer-present rescue still ends TURN_DONE -> "cap".)
def test_e2e_forced_final_empty_answer_honest_error(tmp_path, monkeypatch):
    server = _StubServer([
        # loop step 1: a real tool turn, so the loop walks into the turn cap
        {"frames": _tool_call_frames(
            "run_bash", {"command": "echo ok > e2e_ff_marker.txt"})},
        # the cap rescue (forced_final=True): final_answer with an EMPTY
        # answer and nothing else — the model produced no final answer and
        # no actionable call
        {"frames": _tool_call_frames("final_answer", {"answer": ""})},
    ])
    server.start()
    try:
        _stub_env(server, tmp_path, monkeypatch)
        messages = m._initial_messages() + [{"role": "user", "content": INSTRUCTION}]
        summary = m._run_model_turn_loop(messages, on_step=None,
                                         max_turns_override=1)
        # The old dishonesty: empty forced-final -> TURN_DONE -> status "cap"
        # (headless exit 3, a scored partial-but-real result). The honest
        # mapping is the give-up "error" (headless exit 4), never "cap".
        assert summary["status"] == "error", (
            "an empty forced-final rescue must not score cap/done: "
            f"{summary}")
        err = (summary.get("error") or "").lower()
        assert "stream cut" in err and "forced-final" in err, summary
        # the recorded cause names the actual failure, not silence
        cause = m._turn._LAST_GIVE_UP_CAUSE.lower()
        assert "no final_answer" in cause, m._turn._LAST_GIVE_UP_CAUSE
        # (b) the handler-level vocabulary on the same script: the rescue
        # call itself returns TURN_GAVE_UP, not TURN_DONE. The stub re-
        # serves its last turn, so this replays the empty final_answer.
        status = m.model_turn(messages, forced_final=True)
        assert status == TURN_GAVE_UP, status
    finally:
        server.stop()
