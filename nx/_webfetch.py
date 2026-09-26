"""web_fetch (0.70.0; default-on + persistent cache in 0.71.0): keyless, free
web access — no paid API.

Two actions in one tier-2 tool:

- ``query=``  DuckDuckGo *lite* HTML search (html.duckduckgo.com/html/): a
  human-shaped GET of a page meant for browsers, parsed with the stdlib. No
  key, no account, no SDK. SearXNG (self-hosted metasearch, ``?format=json``)
  is the designated backup once DDG has been exercised enough to trust the
  parsing — see the MEMORY note "web_fetch SearXNG backup plan" before wiring
  it in.
- ``url=``    fetch a page and strip it to readable text with the stdlib
  (html.parser). Images/scripts/style are dropped, links keep their hrefs.

Courtesy of a free resource is a design constraint, not a nice-to-have:
every response is cached globally for 7 days (persisted under
``<SESSION_HOME>/web-cache/cache.jsonl``, so a fresh session hits the cache
too), a minimum interval is enforced between requests to duckduckgo.com, a
small polite User-Agent is sent, and only ``GET`` requests are ever issued.
Failures fail soft with a message the model can act on (retry later, narrower
query, different URL) — never an exception into the turn loop.
"""
from __future__ import annotations

import html as _html
import ipaddress
import json
import os
import re
import socket
import tempfile
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from ._env import _env_int

WEBFETCH_ENABLED = _env_int("NX_WEBFETCH", 1)
FETCH_MAX_BYTES = _env_int("NX_WEBFETCH_MAX_BYTES", 20000)
CACHE_TTL_SECS = _env_int("NX_WEBFETCH_CACHE_TTL", 7 * 24 * 3600)
# Courtesy floor between requests to duckduckgo.com (search OR the fetch of a
# page hosted anywhere is NOT throttled — only the search endpoint is, since
# that's the shared free resource).
DDG_MIN_INTERVAL_SECS = _env_int("NX_WEBFETCH_SEARCH_INTERVAL", 3.0)

UA = "nx-coding-agent/0.72 (+local CLI; respectful, cached, GET-only)"
SEARCH_URL = "https://html.duckduckgo.com/html/"
# SearXNG (self-hosted metasearch) is the PREFERRED search backend (0.72.0):
# set NX_SEARX_URL to the instance root (e.g. http://localhost:8080) and
# searches go to {url}/search?q=...&format=json. No key; the instance's own
# engines do the upstream querying, so nx adds exactly one request. DDG lite
# stays as the fallback when NX_SEARX_URL is unset or unreachable.
SEARX_URL = os.environ.get("NX_SEARX_URL", "").rstrip("/")
TIMEOUT_SECS = 15
MAX_SEARCH_RESULTS = 10
# SSRF guard (0.92.1): web_fetch is an approval-free tool, so `url=` must never
# become a pivot to internal services (loopback SearXNG, cloud metadata
# 169.254.169.254, RFC1918). Set NX_WEBFETCH_ALLOW_PRIVATE=1 to lift this for
# an explicitly trusted network. NX_SEARX_URL itself is the ONE sanctioned
# loopback target and is always exempt (it is the user's own instance).
ALLOW_PRIVATE = _env_int("NX_WEBFETCH_ALLOW_PRIVATE", 0)


def _host_is_private(host: str) -> bool:
    """True if `host` (IP literal or name) resolves to a private/loopback/
    link-local/reserved address. Unresolvable names return False (the fetch
    will simply fail); IP literals are checked without DNS."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return not addr.is_global
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    for fam, _, _, _, sockaddr in infos:
        try:
            if not ipaddress.ip_address(sockaddr[0]).is_global:
                return True
        except ValueError:
            continue
    return False


def _url_is_private(url: str) -> bool:
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except ValueError:
        return True
    if not host:
        return True
    return _host_is_private(host)


class _SsrfRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects but re-validate each hop against the SSRF guard —
    without this, a public page can bounce the GET at loopback/RFC1918."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        if not ALLOW_PRIVATE and _url_is_private(newurl):
            raise urllib.error.HTTPError(
                newurl, code, "redirect to a private/loopback address blocked",
                headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

# --- session state (module-level on purpose: conftest rewinds nx.* dicts) ---
_CACHE: dict[str, tuple[float, str]] = {}  # key -> (monotonic ts, text)
_LAST_SEARCH: list[float] = [0.0]
_CACHE_LOADED: list[bool] = [False]

# Web-cache persistence (0.71.0): one JSON object per line
# {"k", "t" (epoch secs), "v"} under <SESSION_HOME>/web-cache/cache.jsonl.
# SESSION_HOME (NX_HOME) is set up by __init__.py after this module imports
# and tests point NX_HOME at a tmpdir, so resolve it lazily.
def _cache_dir() -> str:
    from ._state import SESSION_HOME
    return os.path.join(SESSION_HOME, "web-cache")


def _ensure_cache() -> None:
    """Best-effort load of the persistent cache on first use per process;
    expired entries are dropped, corrupt lines skipped, missing dir a no-op."""
    if _CACHE_LOADED[0]:
        return
    _CACHE_LOADED[0] = True
    try:
        path = os.path.join(_cache_dir(), "cache.jsonl")
        if not os.path.exists(path):
            return
        now = time.time()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    age = now - float(obj["t"])
                    if 0 <= age < CACHE_TTL_SECS:
                        _CACHE[obj["k"]] = (time.monotonic() - age, str(obj["v"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass


def _save_cache() -> None:
    """Best-effort atomic write of the whole cache (tmp+rename, 0600). A cache
    write failure must never fail the tool call."""
    try:
        d = _cache_dir()
        # 0700: the cache holds full fetched-page text (potentially from
        # internal hosts), so it must not be group/other-traversable.
        os.makedirs(d, mode=0o700, exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
        now = time.time()
        lines = [json.dumps({"k": k, "t": now - (time.monotonic() - v[0]), "v": v[1]})
                 for k, v in _CACHE.items()]
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, os.path.join(d, "cache.jsonl"))
    except OSError:
        pass


# --- HTML -> text -----------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Stdlib-only readable-text extraction. Keeps link hrefs as markdown-ish
    ``[text](url)`` so the model can follow references with another fetch."""

    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5",
              "h6", "section", "article", "blockquote", "pre", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._link_href: str | None = None
        self._link_text: list[str] = []
        self._out: list[str] = []
        self._links: list[tuple[int, str]] = []  # (index into _out, href)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "a":
            href = dict(attrs).get("href") or ""
            self._link_href = href if href.startswith(("http://", "https://")) else None
            self._link_text = []
        elif tag in self._BLOCK:
            self._out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "a" and self._link_href is not None:
            text = "".join(self._link_text).strip()
            if text:
                self._links.append((len(self._out), self._link_href))
                self._out.append(f"[{text}]({self._link_href})")
            self._link_href = None
        elif tag in self._BLOCK:
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._link_href is not None:
            self._link_text.append(data)
        self._out.append(data)

    def result(self) -> str:
        text = "".join(self._out)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


def _html_to_text(body: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(body)
        parser.close()
        return parser.result()
    except Exception:
        # fail soft: crude tag strip rather than losing the fetch entirely
        text = re.sub(r"<[^>]+>", " ", body)
        return re.sub(r"\s+", " ", _html.unescape(text)).strip()


# --- HTTP (GET only, always) -------------------------------------------------

# One opener shared by all fetches: GET-only by construction (every call site
# passes method="GET"), with redirect hops re-validated against the SSRF guard.
_OPENER = urllib.request.build_opener(_SsrfRedirectHandler())


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    with _OPENER.open(req, timeout=TIMEOUT_SECS) as resp:  # noqa: S310
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read(FETCH_MAX_BYTES * 4).decode(charset, errors="replace")


def _http_get_bytes(url: str) -> tuple[bytes, str]:
    """GET with the body as raw bytes plus the declared content type. For PDF
    routing: extracting a PDF as latin-1 text would yield garbage, so the
    caller must know the content type before decoding."""
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    with _OPENER.open(req, timeout=TIMEOUT_SECS) as resp:  # noqa: S310
        ctype = (resp.headers.get_content_type() or "").lower()
        return resp.read(FETCH_MAX_BYTES * 4), ctype


def _pdf_sniff(body: bytes, ctype: str) -> bool:
    """Content-type says PDF, or the magic bytes say it anyway (some servers
    serve PDFs as application/octet-stream)."""
    if "pdf" in ctype:
        return True
    return body[:5] == b"%PDF-"


def _cache_get(key: str) -> str | None:
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_SECS:
        return hit[1]
    return None


def _cache_put(key: str, value: str) -> None:
    if len(_CACHE) > 64:  # bounded: drop everything past the TTL is overkill
        oldest = sorted(_CACHE, key=lambda k: _CACHE[k][0])[:-32]
        for k in oldest:
            _CACHE.pop(k, None)
    _CACHE[key] = (time.monotonic(), value)


# --- search (DuckDuckGo lite) ------------------------------------------------

_RESULT_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIP_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")
# DDG lite wraps result hrefs in a redirect (/l/?uddg=<urlencoded>).
_UDDG_RE = re.compile(r"[?&]uddg=([^&]+)")


def _unwrap_ddg(href: str) -> str:
    href = _html.unescape(href)
    m = _UDDG_RE.search(href)
    if m:
        href = urllib.parse.unquote(m.group(1))
    return href


def _parse_results(body: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    snips = _SNIP_RE.findall(body)
    for i, (href, title) in enumerate(_RESULT_SNIPPET_RE.findall(body)):
        url = _unwrap_ddg(href)
        if not url.startswith("http"):
            continue
        item = {
            "title": _html.unescape(_TAG_RE.sub("", title)).strip(),
            "url": url,
            "snippet": _html.unescape(_TAG_RE.sub("", snips[i])).strip() if i < len(snips) else "",
        }
        out.append(item)
        if len(out) >= MAX_SEARCH_RESULTS:
            break
    return out


def _searx_search(query: str) -> str | None:
    """SearXNG JSON-API search; None when unset/unreachable/broken (caller
    falls back to DDG lite). One GET to the user's own instance — the DDG
    throttle does not apply (it exists to spare a SHARED free endpoint)."""
    if not SEARX_URL or not _searx_usable():
        return None
    qs = urllib.parse.urlencode({"q": query, "format": "json"})
    try:
        body = _http_get(f"{SEARX_URL}/search?{qs}")
        data = json.loads(body)
        results = data.get("results") or []
    except Exception:
        return None  # unreachable or bad JSON: fall through to DDG lite
    out = []
    for r in results[:MAX_SEARCH_RESULTS]:
        url = str(r.get("url") or "")
        if not url.startswith(("http://", "https://")):
            continue
        if not ALLOW_PRIVATE and _url_is_private(url):
            continue  # a compromised engine must not seed an SSRF pivot
        title = str(r.get("title") or "").strip()
        content = str(r.get("content") or "").strip()
        out.append(f"- {title} — {url}" + (f"\n  {content}" if content else ""))
    if not out:
        return "(SearXNG returned no results for this query.)"
    return "\n".join(out)


def _web_search(query: str) -> str:
    # courtesy throttle for the shared free endpoint (DDG lite); the user's
    # own SearXNG instance, when configured, is exempt.
    if not SEARX_URL:
        now = time.monotonic()
        wait = _LAST_SEARCH[0] + DDG_MIN_INTERVAL_SECS - now
        if wait > 0:
            time.sleep(wait)
        _LAST_SEARCH[0] = time.monotonic()

    cache_key = "q:" + query.lower().strip()
    _ensure_cache()
    if (hit := _cache_get(cache_key)) is not None:
        return "[cached] " + hit

    # SearXNG first (preferred); DDG lite is the fallback.
    searx = _searx_search(query)
    if searx is not None:
        _cache_put(cache_key, searx)
        _save_cache()
        return searx

    qs = urllib.parse.urlencode({"q": query, "kl": "wt-wt"})
    try:
        body = _http_get(f"{SEARCH_URL}?{qs}")
    except Exception as e:
        hint = (" (NX_SEARX_URL is set but unreachable — fell back to DuckDuckGo)"
                if SEARX_URL else "")
        return (f"ERROR: search failed ({type(e).__name__}){hint}; duckduckgo may be "
                "rate-limiting — wait a few seconds and retry, or ask the user "
                "for a URL to fetch directly.")
    results = _parse_results(body)
    if not results:
        if "bots use DuckDuckGo" in body:
            return ("Search blocked: DuckDuckGo is bot-challenging this IP right "
                    "now. Options: retry in a few minutes (a cached result may "
                    "exist), ask the user for a URL to `url`-fetch directly, or "
                    "set NX_SEARX_URL to a SearXNG instance (the preferred search "
                    "backend — see the README's NX_WEBFETCH entry).")
        return ("No results (the lite endpoint's HTML may have changed — check "
                "the layout before assuming the query matched nothing).")
    lines = [
        f"- {r['title']} — {r['url']}" + (f"\n  {r['snippet']}" if r["snippet"] else "")
        for r in results
    ]
    text = "\n".join(lines)
    _cache_put(cache_key, text)
    _save_cache()
    return text


# --- fetch -------------------------------------------------------------------

def _searx_usable() -> bool:
    """Guard the configured instance: a NON-loopback NX_SEARX_URL over plain
    http leaks every query in cleartext and hands an arbitrary remote server
    control of what enters model context — refuse it (fall back to DDG) unless
    it is https or the private-address guard is lifted."""
    if SEARX_URL.startswith("https://"):
        return True
    if ALLOW_PRIVATE:
        return True
    host = urllib.parse.urlsplit(SEARX_URL).hostname or ""
    return host in ("127.0.0.1", "localhost", "::1", "[::1]")


def _web_fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    if not ALLOW_PRIVATE and _url_is_private(url):
        return ("ERROR: refused to fetch a loopback/private-address URL "
                "(SSRF guard; set NX_WEBFETCH_ALLOW_PRIVATE=1 if the URL is "
                "deliberately internal).")
    cache_key = "u:" + url
    _ensure_cache()
    if (hit := _cache_get(cache_key)) is not None:
        return "[cached] " + hit
    try:
        body, ctype = _http_get_bytes(url)
    except Exception as e:
        return (f"ERROR: fetch failed ({type(e).__name__}); the site may block "
                "scripts or need a browser — try a different URL or ask the user.")
    if _pdf_sniff(body, ctype):
        # A PDF (arXiv/PMC download links, journal PDFs): decode-as-text would
        # yield garbage. Same extractor read_file routes PDFs through —
        # page-capped via NX_PDF_MAX_PAGES, never raises.
        from ._pdf import extract_pdf
        fd, path = tempfile.mkstemp(suffix=".pdf", prefix="nx-webfetch-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(body)
            text = extract_pdf(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    else:
        text = _html_to_text(body.decode("utf-8", errors="replace"))
    if len(text) > FETCH_MAX_BYTES:
        text = text[:FETCH_MAX_BYTES] + f"\n\n[truncated at {FETCH_MAX_BYTES} bytes; fetch a narrower URL for more]"
    _cache_put(cache_key, text)
    _save_cache()
    return text


# --- tool body ---------------------------------------------------------------

def web_fetch(query: str = "", url: str = "") -> str:
    """Tool body for the web_fetch tier-2 tool. Exactly one of query/url."""
    if bool(query) == bool(url):
        return "ERROR: pass exactly one of `query` (DuckDuckGo search) or `url` (fetch a page)"
    return _web_search(query) if query else _web_fetch_url(url)


WEB_FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Keyless web access, GET-only and rate-limited to stay polite to free endpoints (results are cached globally for 7 days). Pass `query` for a web search (SearXNG via NX_SEARX_URL when configured, else DuckDuckGo-lite) returning title/URL/snippet results, or `url` to fetch a page as readable text (links kept as [text](url)). Never both. For the sandbox's own docs, prefer the local files — this is for the outside world.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search string (DuckDuckGo lite); returns up to 10 title/URL/snippet results to follow up with a url fetch"},
                "url": {"type": "string", "description": "Absolute http(s) URL to fetch and convert to readable text; truncated at NX_WEBFETCH_MAX_BYTES"},
            },
            # The body has always required exactly one operation.  Encode the
            # same contract for native/schema-aware callers; the text grammar
            # expands these branches into equivalent argument alternatives.
            "oneOf": [
                {"required": ["query"], "not": {"required": ["url"]}},
                {"required": ["url"], "not": {"required": ["query"]}},
            ],
        },
    },
}
