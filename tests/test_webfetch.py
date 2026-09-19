"""web_fetch (0.70.0): keyless DuckDuckGo-lite search + GET-only page fetch.

The suite is offline by construction, so every network touch is asserted
against a local HTTP server / monkeypatched _http_get; no test ever contacts
duckduckgo.com. Courtesy constraints (cache, search throttle, GET-only) are
the contract under test, not incidental behavior.
"""
import os
import sys
import tempfile
import threading
import urllib.parse
import unittest.mock
import http.server

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx  # noqa: E402
import nx._webfetch as w  # noqa: E402


# --- fake DDG-lite HTML ------------------------------------------------------

DDG_PAGE = """
<html><body>
<div class="result">
  <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.org%2Fa&amp;rut=x">Example A</a>
  <a class="result__snippet" href="#">The first snippet &amp; more</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/b">Example B</a>
  <a class="result__snippet" href="#">Second</a>
</div>
</body></html>
"""

PAGE = """
<html><head><title>T</title><style>.x{}</style></head>
<body>
<script>var junk = 1;</script>
<h1>Heading</h1>
<p>Some <b>bold</b> text with a <a href="https://x.example/1">link</a>.</p>
<p>Second para</p>
</body></html>
"""


def _clean():
    w._CACHE.clear()
    w._LAST_SEARCH[0] = 0.0


def test_schema_registered_and_gated():
    assert any(t["function"]["name"] == "web_fetch" for t in nx.TOOLS)
    import nx._dispatch as d
    assert "web_fetch" in d._TOOL_GATES
    # env defaults to off -> gate closed (conftest scrubbed NX_*)
    assert d._TOOL_GATES["web_fetch"]() is False


def test_exactly_one_of_query_url():
    _clean()
    assert "exactly one" in w.web_fetch()
    assert "exactly one" in w.web_fetch(query="q", url="https://x/")
    assert "ERROR" in w.web_fetch(url="ftp://nope/")


def test_search_parses_lite_html_and_unwraps_redirects():
    _clean()
    with unittest.mock.patch.object(w, "_http_get", return_value=DDG_PAGE):
        out = w.web_fetch(query="test")
    assert "Example A" in out
    assert "https://example.org/a" in out  # uddg= redirect unwrapped
    assert "https://example.com/b" in out
    assert "The first snippet & more" in out  # unescaped + tag-stripped


def test_search_is_cached():
    _clean()
    calls = []
    def fake_get(url):
        calls.append(url)
        return DDG_PAGE
    with unittest.mock.patch.object(w, "_http_get", side_effect=fake_get):
        w.web_fetch(query="cached")
        assert "[cached]" in w.web_fetch(query="cached")
    assert len(calls) == 1  # second call never touched the endpoint


def test_search_throttle_enforces_interval():
    _clean()
    sleeps = []
    real_sleep = w.time.sleep
    w.time.sleep = lambda s: sleeps.append(s)
    try:
        with unittest.mock.patch.object(w, "_http_get", return_value=DDG_PAGE):
            w.web_fetch(query="one")
            w.web_fetch(query="two")
    finally:
        w.time.sleep = real_sleep
    # second search was held back to >= the courtesy floor
    assert sleeps and 0 < sleeps[-1] <= w.DDG_MIN_INTERVAL_SECS


def test_search_failure_fails_soft():
    _clean()
    def boom(url):
        raise OSError("no network")
    with unittest.mock.patch.object(w, "_http_get", side_effect=boom):
        out = w.web_fetch(query="down")
    assert out.startswith("ERROR") and "retry" in out


def test_search_sends_get_with_polite_ua():
    _clean()
    captured = {}
    class Resp:
        def __init__(self): self.headers = {"Content-Type": "text/html"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1): return DDG_PAGE.encode()
    def fake_urlopen(req, timeout=0):
        captured["method"] = req.get_method()
        captured["ua"] = req.headers.get("User-agent")
        return Resp()
    with unittest.mock.patch.object(w.urllib.request, "urlopen", side_effect=fake_urlopen):
        w.web_fetch(query="ua")
    assert captured["method"] == "GET"
    assert "nx-coding-agent" in captured["ua"]


def test_fetch_strips_html_keeps_links():
    _clean()
    with unittest.mock.patch.object(w, "_http_get", return_value=PAGE):
        out = w.web_fetch(url="https://example.org/page")
    assert "Heading" in out and "bold" in out
    assert "[link](https://x.example/1)" in out
    assert "var junk" not in out and ".x{}" not in out


def test_fetch_truncates():
    _clean()
    big = "<p>" + ("word " * 20000) + "</p>"
    with unittest.mock.patch.object(w, "_http_get", return_value=big):
        out = w.web_fetch(url="https://example.org/big")
    assert "truncated" in out and len(out) < 30000


def test_fetch_error_is_actionable():
    _clean()
    def boom(url):
        raise OSError("refused")
    with unittest.mock.patch.object(w, "_http_get", side_effect=boom):
        out = w.web_fetch(url="https://example.org/x")
    assert out.startswith("ERROR") and "browser" in out


def test_url_scheme_fence():
    _clean()
    assert "http" in w.web_fetch(url="file:///etc/passwd")
    # nothing was fetched / cached
    assert not w._CACHE


class _GateServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_fetch_against_local_server():
    _clean()
    srv = http.server.HTTPServer(("127.0.0.1", 0), _GateServer)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        out = w.web_fetch(url=f"http://127.0.0.1:{srv.server_port}/real")
    finally:
        srv.shutdown()
    assert "Heading" in out and "[link](https://x.example/1)" in out


def test_withheld_tool_refusal_names_the_gate():
    import nx._dispatch as d
    msg = d._unknown_tool_error("web_fetch")
    assert "NX_WEBFETCH" in msg and "ask the user" in msg
