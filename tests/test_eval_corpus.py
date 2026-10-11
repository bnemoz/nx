"""Offline eval corpus (`nx eval list|run`, nx/_eval_cmd.py) — P3-14 seed.

Contract under test:
- the stub server serves scripted OpenAI-style SSE turns keyed by the task
  instruction in the request payload (per-scenario turn cursor, so one server
  serves a whole batch)
- `nx eval run` replays the corpus through the REAL `nx run` batch harness and
  scores every scenario: status / final answer / error shape / workspace file
  effects — and exits 0 only when all pass
- recovery outcomes are pinned: two silent empty turns recover to a final
  answer; a special-token collapse cuts + recovers; a mid-stream read timeout
  is scored as status "error" naming the timeout (P3-2's honest-timeout
  contract) — never "done"
- `nx eval list` lists every scenario; unknown verbs/filters are misuse (2)

Runs offline: the loopback stub server IS the endpoint; no nx module state is
left behind (the runner saves/restores sources, env, and cwd itself).
"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)

from nx import _eval_cmd as ec


def _capture(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(*a, **kw)
    return code, out.getvalue(), err.getvalue()


def test_stub_serves_scripted_turns_keyed_by_instruction():
    """The scenario is resolved from the request payload and the turn cursor
    is per scenario — two interleaved scenarios don't steal each other's
    turns (the shared-cursor bug that mis-served a whole batch)."""
    import threading
    import httpx

    srv, url = ec._stub_server()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with httpx.Client(base_url=url, timeout=5) as c:
            def post(instr):
                payload = {"model": "m", "stream": True,
                           "messages": [{"role": "user", "content": instr}]}
                resp = c.post("/chat/completions", json=payload)
                assert resp.status_code == 200
                frames = [ln[len("data:"):].strip() for ln in resp.text.splitlines()
                          if ln.startswith("data:")]
                resp.close()
                return frames

            a1 = post(ec.SCENARIOS["e-empty"]["instruction"])
            b1 = post(ec.SCENARIOS["e-happy"]["instruction"])
            a2 = post(ec.SCENARIOS["e-empty"]["instruction"])
            # e-empty's first two turns are empty (no content deltas); its
            # third is the recovery final. e-happy's first turn is a tool call.
            def content_of(frames):
                out = ""
                for f in frames:
                    if f == "[DONE]":
                        continue
                    ch = json.loads(f)
                    d = (ch.get("choices") or [{}])[0].get("delta") or {}
                    out += d.get("content") or ""
                    for tc in d.get("tool_calls") or []:
                        out += (tc.get("function") or {}).get("name") or ""
                return out
            assert content_of(a1) == "", content_of(a1)[:80]
            assert "run_bash" in content_of(b1), content_of(b1)[:80]
            assert content_of(a2) == "", content_of(a2)[:80]
            # A request with no scenario instruction is rejected, not guessed.
            r = c.post("/chat/completions", json={"messages": []})
            assert r.status_code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_eval_list_lists_every_scenario():
    code, out, _ = _capture(ec._eval_main, ["list"])
    assert code == 0
    for sid in ec._scenario_ids():
        assert sid in out, sid
    assert "7 scenarios" in out or f"{len(ec._scenario_ids())} scenarios" in out


def test_eval_run_scores_whole_corpus(tmp_path):
    """The full seed corpus passes end to end through the real batch harness —
    including the honest-timeout contract (a mid-stream read timeout scores
    status "error" naming the timeout, never "done")."""
    code, out, err = _capture(ec._eval_main, ["run", "--out", str(tmp_path / "eval")])
    assert code == 0, out + err
    summary = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert summary["failed"] == 0
    assert summary["passed"] == len(ec._scenario_ids())
    ids = {r["id"]: r for r in summary["results"]}
    # Recovery scenarios really recovered (their finals exist on the line).
    assert "Recovered after 2 empty turns" in ids["e-empty"]["final_answer"]
    assert "Recovered from the special-token collapse" in ids["e-garble"]["final_answer"]
    # The honest-timeout scenario: status error naming the timeout.
    assert ids["e-timeout"]["status"] == "error"
    assert "timeout" in (ids["e-timeout"]["error"] or "").lower()
    # Real tool effects landed in the workspace.
    ws = tmp_path / "eval" / "workspace"
    assert (ws / "eval_write.txt").is_file()
    util = (ws / "eval_src" / "util.py").read_text()
    assert "body_raw = load_payload()" in util
    assert "payload_raw = load_payload()" not in util
    # results.jsonl is verbatim harness JSONL (one line per task).
    lines = [json.loads(l) for l in
             (tmp_path / "eval" / "results.jsonl").read_text().splitlines() if l.strip()]
    assert {r["id"] for r in lines} == set(ec._scenario_ids())


def test_eval_run_filter_single_scenario(tmp_path):
    code, out, _ = _capture(
        ec._eval_main, ["run", "--filter", "e-happy", "--out", str(tmp_path / "eval")])
    assert code == 0, out
    summary = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert [r["id"] for r in summary["results"]] == ["e-happy"]


def test_eval_misuse_exit_codes():
    # Unknown verb → 2; unknown filter → 2; bare `nx eval` → 2.
    assert _capture(ec._eval_main, ["nope"])[0] == 2
    assert _capture(ec._eval_main, ["run", "--filter", "no-such"])[0] == 2
    assert _capture(ec._eval_main, [])[0] == 2
    assert _capture(ec._eval_main, ["--help"])[0] == 0


def test_timeout_scenario_uses_vendored_client_timeout():
    """P3-14 caught a real client bug: create() passed timeout=None to
    build_request, which DISABLES httpx timeouts — NX_HTTP_TIMEOUT_READ never
    applied and the mid-stream timeout ladder was unreachable. The fix: the
    per-request kwarg is attached only when actually set. Pinned against the
    vendored client over a REAL socket (MockTransport bypasses the timeout
    machinery, so it cannot catch this)."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from nx._openai_client import OpenAI

    class _HoldingHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            frame = 'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
            data = frame.encode()
            try:
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                time.sleep(3.0)  # hold the stream open, no more frames
                self.wfile.write(b"0\r\n\r\n")
            except Exception:
                pass

    os.environ["NX_HTTP_TIMEOUT_READ"] = "0.5"
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _HoldingHandler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        client = OpenAI(base_url=url, api_key="k")
        assert client._client.timeout.read == 0.5
        t0 = time.time()
        try:
            stream = client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "x"}], stream=True)
            n = sum(1 for _ in stream)
            raise AssertionError(f"expected a mid-stream read timeout, got {n} chunks")
        except Exception as e:
            assert type(e).__name__ in ("ReadTimeout", "TimeoutException",
                                        "httpx.ReadTimeout"), type(e).__name__
        took = time.time() - t0
        assert took < 2.0, f"read timeout fired at {took:.2f}s — not applied"
    finally:
        os.environ.pop("NX_HTTP_TIMEOUT_READ", None)
        srv.shutdown()
        srv.server_close()
