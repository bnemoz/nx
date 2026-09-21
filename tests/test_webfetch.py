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
    # on by default (0.71.0): the env default is 1, so the gate is OPEN with
    # NX_* scrubbed (conftest); NX_WEBFETCH=0 closes it for offline hosts.
    assert d._TOOL_GATES["web_fetch"]() is True
    # NX_WEBFETCH is a module attr (proxied via _STATE_NAMES), not re-read
    # from the env at gate time — set it the way tests set NX_SUBAGENTS.
    saved = nx.NX_WEBFETCH
    nx.NX_WEBFETCH = 0
    try:
        assert d._TOOL_GATES["web_fetch"]() is False
    finally:
        nx.NX_WEBFETCH = saved


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


def test_searx_preferred_when_configured():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "http://127.0.0.1:9"
    try:
        with unittest.mock.patch.object(w, "_http_get",
                                        return_value='{"results": [{"title": "S", "url": "https://s.example/", "content": "c"}]}'):
            out = w.web_fetch(query="searx first")
    finally:
        w.SEARX_URL = saved
    assert "https://s.example/" in out and "DuckDuckGo" not in out


def test_searx_unreachable_falls_back_to_ddg():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "http://127.0.0.1:9"  # nothing listens there
    try:
        with unittest.mock.patch.object(w, "_http_get", return_value=DDG_PAGE) as g:
            out = w.web_fetch(query="fallback")
    finally:
        w.SEARX_URL = saved
    # first GET hit searx (failed), second hit DDG
    assert g.call_count == 2 and "Example A" in out


def test_searx_throttle_exempt():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "http://127.0.0.1:9"
    sleeps = []
    real_sleep = w.time.sleep
    w.time.sleep = lambda s: sleeps.append(s)
    try:
        def fake(url):
            if "searx" not in url and "127.0.0.1:9" not in url:
                return DDG_PAGE
            raise OSError("down")
        with unittest.mock.patch.object(w, "_http_get", side_effect=fake):
            w.web_fetch(query="one")
            w.web_fetch(query="two")
    finally:
        w.time.sleep = real_sleep
        w.SEARX_URL = saved
    assert not sleeps  # searx-configured searches are never held back


def test_searx_bad_json_falls_back():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "http://searx.invalid"
    try:
        with unittest.mock.patch.object(w, "_http_get",
                                        side_effect=['<html>not json</html>', DDG_PAGE]) as g:
            out = w.web_fetch(query="badjson")
    finally:
        w.SEARX_URL = saved
    assert g.call_count == 2 and "Example A" in out


def test_searx_empty_results():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "http://searx.invalid"
    try:
        with unittest.mock.patch.object(w, "_http_get", return_value='{"results": []}'):
            out = w.web_fetch(query="nothing")
    finally:
        w.SEARX_URL = saved
    assert "no results" in out


def test_search_bot_challenge_is_actionable():
    _clean()
    challenge = "<html>Unfortunately, bots use DuckDuckGo too.</html>"
    with unittest.mock.patch.object(w, "_http_get", return_value=challenge):
        out = w.web_fetch(query="blocked")
    assert "blocked" in out and "SearXNG" in out


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
    with unittest.mock.patch.object(w, "_http_get_bytes",
                                    return_value=(PAGE.encode(), "text/html")):
        out = w.web_fetch(url="https://example.org/page")
    assert "Heading" in out and "bold" in out
    assert "[link](https://x.example/1)" in out
    assert "var junk" not in out and ".x{}" not in out


def test_fetch_truncates():
    _clean()
    big = "<p>" + ("word " * 20000) + "</p>"
    with unittest.mock.patch.object(w, "_http_get_bytes",
                                    return_value=(big.encode(), "text/html")):
        out = w.web_fetch(url="https://example.org/big")
    assert "truncated" in out and len(out) < 30000


def test_fetch_pdf_routed_to_extractor():
    """0.79.0: a URL that serves a PDF (arXiv/PMC download links) routes
    through extract_pdf instead of decode-as-html garbage."""
    _clean()
    pdf_bytes = b"%PDF-1.4 fake body"
    with unittest.mock.patch.object(w, "_http_get_bytes",
                                    return_value=(pdf_bytes, "application/pdf")), \
         unittest.mock.patch("nx._pdf.extract_pdf",
                             return_value="--- @pdf ---\nExtracted paper text") as ex:
        out = w.web_fetch(url="https://arxiv.org/pdf/2401.00001")
    assert ex.called
    assert "Extracted paper text" in out
    # magic-byte sniff when the server mislabels the content type
    _clean()
    with unittest.mock.patch.object(w, "_http_get_bytes",
                                    return_value=(pdf_bytes, "application/octet-stream")), \
         unittest.mock.patch("nx._pdf.extract_pdf",
                             return_value="--- @pdf ---\nExtracted paper text"):
        out = w.web_fetch(url="https://example.org/paper.pdf")
    assert "Extracted paper text" in out
    # HTML still goes through the normal text extractor
    _clean()
    with unittest.mock.patch.object(w, "_http_get_bytes",
                                    return_value=(PAGE.encode(), "text/html")):
        out = w.web_fetch(url="https://example.org/page2")
    assert "Heading" in out


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
    # 0.71.0: on by default, so this refusal is only reachable with NX_WEBFETCH=0
    os.environ["NX_WEBFETCH"] = "0"
    try:
        assert "NX_WEBFETCH" in msg
    finally:
        del os.environ["NX_WEBFETCH"]


def test_cache_persists_across_sessions(tmp_path=None):
    """0.71.0: the cache is global, persisted under <NX_HOME>/web-cache/ and
    reloaded by a fresh process/module state; default TTL is 7 days."""
    import shutil
    home = tempfile.mkdtemp(prefix="nx-wf-home-")
    old_home = os.environ.get("NX_HOME")
    os.environ["NX_HOME"] = home
    # conftest re-derives the SESSION_HOME mirror from NX_HOME before each
    # test, but never mid-test - mirror our change by hand.
    sys.modules["nx"].SESSION_HOME = home
    sys.modules["nx._state"].SESSION_HOME = home
    try:
        _clean()
        w._CACHE_LOADED[0] = False
        with unittest.mock.patch.object(w, "_http_get", return_value=DDG_PAGE):
            w.web_fetch(query="persist me")
        cache_file = os.path.join(home, "web-cache", "cache.jsonl")
        assert os.path.exists(cache_file) and os.path.getsize(cache_file) > 0
        assert "Example A" in open(cache_file).read()
        # simulate a NEW session: blank memory cache, reload from disk
        _clean()
        w._CACHE_LOADED[0] = False
        with unittest.mock.patch.object(w, "_http_get",
                                        side_effect=AssertionError("network hit")):
            out = w.web_fetch(query="persist me")
        assert "[cached]" in out and "Example A" in out
        # expired entries are dropped on load
        lines2 = open(cache_file).read().strip().split("\n")
        import json as _json
        obj = _json.loads(lines2[0])
        obj["t"] -= w.CACHE_TTL_SECS + 10  # just past expiry
        with open(cache_file, "w") as f:
            f.write(_json.dumps(obj) + "\n")
        _clean()
        w._CACHE_LOADED[0] = False
        with unittest.mock.patch.object(w, "_http_get", return_value=DDG_PAGE) as g:
            w.web_fetch(query="persist me")
        assert g.call_count == 1  # expired -> refetched
        # corrupt lines are skipped, not fatal
        with open(cache_file, "w") as f:
            f.write("not json\n")
            f.write(lines2[0] + "\n")
        _clean()
        w._CACHE_LOADED[0] = False
        w._ensure_cache()
        assert any("persist me" in k for k in w._CACHE)
    finally:
        if old_home is None:
            os.environ.pop("NX_HOME", None)
        else:
            os.environ["NX_HOME"] = old_home
        shutil.rmtree(home, ignore_errors=True)
        _clean()
        w._CACHE_LOADED[0] = False


def test_cache_save_failure_never_fails_the_call():
    _clean()
    # break the atomic rename INSIDE _save_cache (the real call site swallows
    # OSError, so a cache write failure must not fail the tool call)
    real_replace = w.os.replace
    def boom(src, dst):
        raise OSError("disk gone")
    w.os.replace = boom
    try:
        with unittest.mock.patch.object(w, "_http_get", return_value=DDG_PAGE):
            out = w.web_fetch(query="ok anyway")
    finally:
        w.os.replace = real_replace
    assert "Example A" in out
