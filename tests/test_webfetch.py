"""web_fetch (0.70.0): keyless DuckDuckGo-lite search + GET-only page fetch.

The suite is offline by construction, so every network touch is asserted
against a local HTTP server / monkeypatched _http_get; no test ever contacts
duckduckgo.com. Courtesy constraints (cache, search throttle, GET-only) are
the contract under test, not incidental behavior.
"""
import os
import sys
import tempfile
import socket
import threading
import time
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
    w._LAST_SEARCH[0] = time.monotonic() - w.DDG_MIN_INTERVAL_SECS  # throttle already elapsed
    # 0.113.0: robots.txt would do a REAL network call for example.org hosts;
    # the suite is offline, so tests that exercise robots do so explicitly.
    w.ROBOTS_ENABLED = 0
    w._ROBOTS.clear()
    w._ROBOTS_FAILED.clear()


def test_schema_registered_and_gated():
    assert any(t["function"]["name"] == "web_fetch" for t in nx.TOOLS)
    import nx._dispatch as d
    # 0.113.0: tier-1 — the gate entry is GONE; the schema is appended to
    # TOOLS only when WEBFETCH_ENABLED, so NX_WEBFETCH=0 removes the tool
    # from the catalog entirely instead of closing a gate.
    assert "web_fetch" not in d._TOOL_GATES
    assert d._unknown_tool_error("web_fetch").startswith("ERROR: web_fetch is not available")
    # the refusal text is the generic withheld wording (no _GATE_REFUSALS entry)
    assert "NX_WEBFETCH" not in d._unknown_tool_error("web_fetch")


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
    # the courtesy throttle fires before the cache lookup on the 2nd call;
    # silence the real sleep like test_search_throttle_enforces_interval does
    real_sleep = w.time.sleep
    w.time.sleep = lambda s: None
    try:
        calls = []
        def fake_get(url):
            calls.append(url)
            return DDG_PAGE
        with unittest.mock.patch.object(w, "_http_get", side_effect=fake_get):
            w.web_fetch(query="cached")
            assert "[cached]" in w.web_fetch(query="cached")
        assert len(calls) == 1  # second call never touched the endpoint
    finally:
        w.time.sleep = real_sleep


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
                                        return_value='{"results": [{"title": "S", "url": "https://s.example/", "content": "c"}]}'), \
             unittest.mock.patch.object(socket, "getaddrinfo",
                                        side_effect=socket.gaierror(-2, "offline")):
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
    w.ALLOW_PRIVATE = 1  # _searx_usable would refuse a remote http instance
    try:
        with unittest.mock.patch.object(w, "_http_get",
                                        side_effect=['<html>not json</html>', DDG_PAGE]) as g:
            out = w.web_fetch(query="badjson")
    finally:
        w.SEARX_URL = saved
        w.ALLOW_PRIVATE = 0
    assert g.call_count == 2 and "Example A" in out


def test_searx_empty_results():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "http://searx.invalid"
    w.ALLOW_PRIVATE = 1  # remote http instance refused by the guard otherwise
    try:
        with unittest.mock.patch.object(w, "_http_get", return_value='{"results": []}'):
            out = w.web_fetch(query="nothing")
    finally:
        w.SEARX_URL = saved
        w.ALLOW_PRIVATE = 0
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
    with unittest.mock.patch.object(w._OPENER, "open", side_effect=fake_urlopen):
        w.web_fetch(query="ua")
    assert captured["method"] == "GET"
    assert "nx-coding-agent" in captured["ua"]


def test_fetch_strips_html_keeps_links():
    _clean()
    with unittest.mock.patch.object(w, "_http_get_bytes",
                                    return_value=(PAGE.encode(), "text/html", "", "")):
        out = w.web_fetch(url="https://example.org/page")
    assert "Heading" in out and "bold" in out
    assert "[link](https://x.example/1)" in out
    assert "var junk" not in out and ".x{}" not in out


def test_fetch_truncates():
    _clean()
    big = "<p>" + ("word " * 20000) + "</p>"
    with unittest.mock.patch.object(w, "_http_get_bytes",
                                    return_value=(big.encode(), "text/html", "", "")):
        out = w.web_fetch(url="https://example.org/big")
    assert "truncated" in out and len(out) < 30000


def test_fetch_pdf_routed_to_extractor():
    """0.79.0: a URL that serves a PDF (arXiv/PMC download links) routes
    through extract_pdf instead of decode-as-html garbage."""
    _clean()
    # the sub-fetches hit the same host back-to-back; drop the per-host
    # courtesy sleep for the duration of this test
    old_interval = w.HOST_MIN_INTERVAL_SECS
    w.HOST_MIN_INTERVAL_SECS = 0.0
    try:
        pdf_bytes = b"%PDF-1.4 fake body"
        with unittest.mock.patch.object(w, "_http_get_bytes",
                                        return_value=(pdf_bytes, "application/pdf", "", "")), \
             unittest.mock.patch("nx._pdf.extract_pdf",
                                 return_value="--- @pdf ---\nExtracted paper text") as ex:
            out = w.web_fetch(url="https://arxiv.org/pdf/2401.00001")
        assert ex.called
        assert "Extracted paper text" in out
        # magic-byte sniff when the server mislabels the content type
        _clean()
        with unittest.mock.patch.object(w, "_http_get_bytes",
                                        return_value=(pdf_bytes, "application/octet-stream", "", "")), \
             unittest.mock.patch("nx._pdf.extract_pdf",
                                 return_value="--- @pdf ---\nExtracted paper text"):
            out = w.web_fetch(url="https://example.org/paper.pdf")
        assert "Extracted paper text" in out
        # HTML still goes through the normal text extractor
        _clean()
        with unittest.mock.patch.object(w, "_http_get_bytes",
                                        return_value=(PAGE.encode(), "text/html", "", "")):
            out = w.web_fetch(url="https://example.org/page2")
        assert "Heading" in out
    finally:
        w.HOST_MIN_INTERVAL_SECS = old_interval


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
    saved = w.ALLOW_PRIVATE
    w.ALLOW_PRIVATE = 1  # loopback test server; the guard is tested separately
    w.SEARX_URL = ""
    srv = http.server.HTTPServer(("127.0.0.1", 0), _GateServer)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        out = w.web_fetch(url=f"http://127.0.0.1:{srv.server_port}/real")
    finally:
        srv.shutdown()
        w.ALLOW_PRIVATE = saved
    assert "Heading" in out and "[link](https://x.example/1)" in out


def test_withheld_tool_refusal_names_the_gate():
    """0.113.0: web_fetch is tier-1; with the schema appended only when
    WEBFETCH_ENABLED, NX_WEBFETCH=0 leaves the name unknown-to-the-catalog,
    so the refusal is the unknown-tool wording, not a gate refusal."""
    import nx._dispatch as d
    msg = d._unknown_tool_error("web_fetch")
    assert msg.startswith("ERROR: web_fetch")


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


# --- SSRF guard (0.92.1) ------------------------------------------------------

def _no_guards(monkeypatch_like=None):
    return None


def test_ssrf_blocks_private_url():
    _clean()
    saved = w.ALLOW_PRIVATE
    w.ALLOW_PRIVATE = 0
    try:
        for u in ("http://127.0.0.1:8888/search", "http://169.254.169.254/latest/meta-data",
                  "http://10.0.0.5/x", "http://[::1]:8080/"):
            out = w.web_fetch(url=u)
        assert "refused" in out and "SSRF" in out
    finally:
        w.ALLOW_PRIVATE = saved


def test_ssrf_allow_private_opt_in():
    _clean()
    saved = w.ALLOW_PRIVATE
    w.ALLOW_PRIVATE = 1
    try:
        with unittest.mock.patch.object(
                w, "_http_get_bytes", return_value=(b"<html>internal</html>", "text/html", "", "")):
            out = w.web_fetch(url="http://127.0.0.1:8888/x")
    finally:
        w.ALLOW_PRIVATE = saved
    assert "internal" in out


def test_ssrf_blocks_redirect_to_private():
    _clean()
    saved = w.ALLOW_PRIVATE
    w.ALLOW_PRIVATE = 0
    try:
        import urllib.error
        # a public page 302s at loopback: the redirect handler must refuse
        class FakeResp:
            headers = {"Location": "http://127.0.0.1:8888/search?q=x"}
            def getcode(self): return 302
            def info(self): return unittest.mock.Mock(get=lambda k, d=None: "http://127.0.0.1:8888/search?q=x")
            def read(self, n=-1): return b""
            def __enter__(self): return self
            def __exit__(self, *a): return False
        def boom(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 302, "moved", {}, None)
        # simpler: directly exercise the handler
        h = w._SsrfRedirectHandler()
        try:
            h.redirect_request(unittest.mock.Mock(), None, 302, "moved",
                               {"Location": "http://127.0.0.1:8888/x"},
                               "http://127.0.0.1:8888/x")
            assert False, "should have raised"
        except urllib.error.HTTPError:
            pass
        # a public hop still passes (use a real Request for super())
        req = urllib.request.Request("https://example.com/")
        new = h.redirect_request(req, None, 302, "moved",
                                 {"Location": "https://example.com/"},
                                 "https://example.com/")
        assert new is not None and new.full_url == "https://example.com/"
    finally:
        w.ALLOW_PRIVATE = saved


def test_searx_nonloopback_http_refused():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "http://searx.example.com"  # plain http, remote host
    try:
        with unittest.mock.patch.object(w, "_http_get", return_value=DDG_PAGE) as g:
            out = w.web_fetch(query="no cleartext")
    finally:
        w.SEARX_URL = saved
    # searx never tried; straight to DDG
    assert g.call_count == 1 and "Example A" in out


def test_searx_remote_https_ok():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "https://searx.example.com"
    try:
        with unittest.mock.patch.object(w, "_http_get",
                                        return_value='{"results": [{"title": "S", "url": "https://s.example/", "content": "c"}]}'), \
             unittest.mock.patch.object(socket, "getaddrinfo",
                                        side_effect=socket.gaierror(-2, "offline")):
            out = w.web_fetch(query="https ok")
    finally:
        w.SEARX_URL = saved
    assert "https://s.example/" in out


def test_searx_results_private_urls_dropped():
    _clean()
    saved = w.SEARX_URL
    w.SEARX_URL = "http://127.0.0.1:8888"
    try:
        with unittest.mock.patch.object(w, "_http_get", return_value=(
            '{"results": ['
            '{"title": "public", "url": "https://ok.example/", "content": "c"},'
            '{"title": "pivot", "url": "http://127.0.0.1:8080/admin", "content": "c"},'
            '{"title": "metadata", "url": "http://169.254.169.254/", "content": "c"}]}')), \
             unittest.mock.patch.object(socket, "getaddrinfo",
                                        side_effect=socket.gaierror(-2, "offline")):
            out = w.web_fetch(query="filter")
    finally:
        w.SEARX_URL = saved
    assert "ok.example" in out and "127.0.0.1" not in out and "169.254" not in out


def test_cache_dir_is_private():
    _clean()
    w._CACHE_LOADED[0] = True
    w._save_cache()
    d = w._cache_dir()
    mode = os.stat(d).st_mode & 0o777
    assert mode == 0o700, f"web-cache dir mode {oct(mode)} != 0700"


# --- 0.113.0: tier-1 promotion, revalidation, politeness ----------------------

def test_tier1_schema_in_active_tools():
    _clean()
    names = {t["function"]["name"] for t in nx._active_tools()}
    assert "web_fetch" in names


def test_ua_is_version_accurate():
    _clean()
    assert "0.72" not in w.UA  # the old hardcoded version never drifted again
    assert "nx-coding-agent/" in w.UA and "GET-only" in w.UA


def test_conditional_revalidation_uses_validators():
    """A stale entry with validators revalidates: If-None-Match is sent and a
    304 keeps serving the cached text without a body transfer."""
    _clean()
    key = "u:https://example.org/stale"
    w._CACHE[key] = (time.monotonic() - w.CACHE_TTL_SECS - 5,
                     "old page text", "etag-1", "Tue, 01 Jan 2030 00:00:00 GMT")
    w._CACHE_LOADED[0] = True
    captured = {}
    def fake(url, etag="", lm=""):
        captured["etag"] = etag
        captured["lm"] = lm
        raise w._NotModified()
    with unittest.mock.patch.object(w, "_http_get_bytes", side_effect=fake):
        out = w.web_fetch(url="https://example.org/stale")
    assert captured["etag"] == "etag-1"
    assert "[cached] old page text" in out
    # the hit refreshed the timestamp (still fresh, validators kept)
    assert w._cache_get(key) == ("old page text", "etag-1",
                                 "Tue, 01 Jan 2030 00:00:00 GMT")


def test_stale_entry_without_validators_refetches_plain():
    _clean()
    key = "u:https://example.org/novalid"
    w._CACHE[key] = (time.monotonic() - w.CACHE_TTL_SECS - 5, "old", "", "")
    w._CACHE_LOADED[0] = True
    with unittest.mock.patch.object(w, "_http_get_bytes",
                                    return_value=(b"<p>fresh</p>", "text/html", "", "")):
        out = w.web_fetch(url="https://example.org/novalid")
    assert "fresh" in out and "[cached]" not in out


def test_429_sets_backoff_and_reports():
    _clean()
    import urllib.error
    def boom(url, etag="", lm=""):
        raise urllib.error.HTTPError(url, 429, "slow down",
                                     {"Retry-After": "5"}, None)
    with unittest.mock.patch.object(w, "_http_get_bytes", side_effect=boom):
        out = w.web_fetch(url="https://example.org/busy")
    assert "429" in out and "back off" in out
    assert w._BACKOFF_UNTIL.get("example.org", 0) > time.monotonic()


def test_per_host_throttle_enforces_interval():
    _clean()
    w.HOST_MIN_INTERVAL_SECS = 1.0
    w._LAST_HOST.clear()
    sleeps = []
    real_sleep = w.time.sleep
    w.time.sleep = lambda s: sleeps.append(s)
    try:
        with unittest.mock.patch.object(w, "_http_get_bytes",
                                        return_value=(b"<p>x</p>", "text/html", "", "")):
            w.web_fetch(url="https://example.org/a")
            w.web_fetch(url="https://example.org/b")
    finally:
        w.time.sleep = real_sleep
        w.HOST_MIN_INTERVAL_SECS = 2.0
    # the second fetch of the SAME host was held back; loopback is exempt
    assert sleeps and 0 < sleeps[-1] <= 1.0


def test_robots_disallow_refuses_fetch():
    _clean()
    w.ROBOTS_ENABLED = 1
    try:
        class FakeRp:
            def set_url(self, u): pass
            def read(self): pass
            def crawl_delay(self, ua): return None
            def can_fetch(self, ua, url): return False
        with unittest.mock.patch.object(w, "RobotFileParser", return_value=FakeRp()), \
             unittest.mock.patch.object(w, "_http_get_bytes") as g, \
             unittest.mock.patch.object(socket, "getaddrinfo",
                                        side_effect=socket.gaierror(-2, "offline")):
            out = w.web_fetch(url="https://blocked.example/private")
            fetched = g.called
        assert "robots.txt" in out and "ERROR" in out
        assert not fetched  # nothing was fetched
        assert not w._CACHE  # nothing was cached either
    finally:
        w.ROBOTS_ENABLED = 0
        w._ROBOTS.clear()
        w._ROBOTS_FAILED.clear()


def test_robots_failure_means_allowed():
    _clean()
    w.ROBOTS_ENABLED = 1
    try:
        def boom(self):
            raise OSError("no dns")
        with unittest.mock.patch.object(w.RobotFileParser, "read", boom), \
             unittest.mock.patch.object(w, "_http_get_bytes",
                                        return_value=(b"<p>ok</p>", "text/html", "", "")):
            out = w.web_fetch(url="https://example.org/robots-down")
        assert "ok" in out
    finally:
        w.ROBOTS_ENABLED = 0
        w._ROBOTS.clear()
        w._ROBOTS_FAILED.clear()


def test_single_flight_shares_one_fetch():
    """Two parallel calls for the same URL share one network fetch."""
    _clean()
    calls = []
    started = threading.Event()
    release = threading.Event()
    def slow(url, etag="", lm=""):
        calls.append(url)
        started.set()
        release.wait(5)
        return (b"<p>shared</p>", "text/html", "", "")
    def worker(results, i):
        results[i] = w.web_fetch(url="https://example.org/same")
    results = [None, None]
    with unittest.mock.patch.object(w, "_http_get_bytes", side_effect=slow):
        t1 = threading.Thread(target=worker, args=(results, 0))
        t1.start()
        started.wait(5)
        t2 = threading.Thread(target=worker, args=(results, 1))
        t2.start()
        time.sleep(0.2)  # t2 blocks on the per-key lock
        release.set()
        t1.join(5); t2.join(5)
    assert len(calls) == 1  # one fetch served both
    assert all("shared" in r for r in results)
