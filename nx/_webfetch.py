"""web_fetch (0.70.0; persistent cache 0.71.0; tier-1 always-on in 0.113.0):
keyless, free web access — no paid API.

Two actions in one tier-1 tool:

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
too), stale pages are revalidated with ``ETag``/``If-None-Match`` and
``Last-Modified``/``If-Modified-Since`` (a 304 costs the origin nothing), a
minimum interval is enforced between requests to duckduckgo.com AND between
page fetches of the same host, ``robots.txt`` is consulted for page fetches
(cached per host; a ``Crawl-delay`` widens that host's interval), ``429``/
``503`` responses put the host on backoff per its ``Retry-After``, a small
polite (version-accurate) User-Agent is sent, ``GET`` is the only method, and
parallel calls for the same key share one in-flight fetch. Failures fail soft
with a message the model can act on (retry later, narrower query, different
URL) — never an exception into the turn loop.
"""
from __future__ import annotations

import base64
import html as _html
import ipaddress
import json
import os
import re
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any
from urllib.robotparser import RobotFileParser

from ._env import _env_int

WEBFETCH_ENABLED = _env_int("NX_WEBFETCH", 1)
FETCH_MAX_BYTES = _env_int("NX_WEBFETCH_MAX_BYTES", 20000)
CACHE_TTL_SECS = _env_int("NX_WEBFETCH_CACHE_TTL", 7 * 24 * 3600)
# Courtesy floor between requests to duckduckgo.com (the shared free
# resource). Page fetches of arbitrary sites get their own per-host floor
# (0.113.0) so repeated fetches of one site stay one polite client.
DDG_MIN_INTERVAL_SECS = float(os.environ.get("NX_WEBFETCH_SEARCH_INTERVAL", "3.0"))
HOST_MIN_INTERVAL_SECS = float(os.environ.get("NX_WEBFETCH_HOST_INTERVAL", "2.0"))
_MAX_BACKOFF_SECS = 60.0
# robots.txt (0.113.0) for page fetches; since 0.120.0 the default is OFF when
# a self-hosted SearXNG is configured (your own instance implies a trusted
# setup, and blanket robots refusal blocked most of its results) — ON otherwise
# (the free DDG endpoints stay polite by default). 1/0 forces it either way.
ROBOTS_ENABLED = _env_int("NX_WEBFETCH_ROBOTS",
                          0 if os.environ.get("NX_SEARX_URL") else 1)


def _ua() -> str:
    """Version-accurate polite UA (0.113.0 — the old hardcoded "0.72" drifted)."""
    try:
        from importlib.metadata import version
        ver = version("nx")
    except Exception:
        ver = "unknown"
    return (f"nx-coding-agent/{ver} (+local CLI; respectful, cached, "
            "GET-only)")


UA = _ua()
SEARCH_URL = "https://html.duckduckgo.com/html/"
# SearXNG (self-hosted metasearch) is the PREFERRED search backend (0.72.0):
# set NX_SEARX_URL to the instance root (e.g. http://localhost:8080) and
# searches go to {url}/search?q=...&format=json. No key; the instance's own
# engines do the upstream querying, so nx adds exactly one request. DDG lite
# stays as the fallback when NX_SEARX_URL is unset or unreachable.
SEARX_URL = os.environ.get("NX_SEARX_URL", "").rstrip("/")
# Optional Basic-auth credentials ("user:pass") for instances that run the
# limiter or sit behind auth — without them such instances 403 on the JSON
# format and nx silently falls back to DDG (0.148.9).
SEARX_AUTH = os.environ.get("NX_SEARX_AUTH", "")
TIMEOUT_SECS = 15
MAX_SEARCH_RESULTS = _env_int("NX_SEARX_MAX_RESULTS", 10)

# Auto-routing heuristics (0.148.9): shape the query for the instance's
# engines instead of always hitting the general pool. `categories=it` pins
# github/stackoverflow/pypi/npm/docker_hub; recency words get time_range=day.
_SEARX_IT_RE = re.compile(
    r"\b(pip|pypi|npm|package|library|dependency|import|module|github|gitlab|"
    r"api|sdk|docker|cargo|crate|stack\s?trace|traceback|cli|venv)\b", re.I)
_SEARX_FRESH_RE = re.compile(
    r"\b(latest|newest|current version|release|released|changelog|upgrade|"
    r"breaking|cve|security advisory|patch|this week|today)\b", re.I)
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


class _NotModified(Exception):
    """Raised by _http_get_bytes when a conditional GET comes back 304."""


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
# key -> (monotonic ts, text, etag, last-modified) — the validators (0.113.0)
# make an expired entry revalidatable instead of garbage past the TTL.
_CACHE: dict[str, tuple[float, str, str, str]] = {}
_LAST_SEARCH: list[float] = [0.0]
_CACHE_LOADED: list[bool] = [False]
# 0.113.0 politeness state: per-host last-request time, per-host backoff-until
# (set by a 429/503 + Retry-After), cached robots.txt parsers, and one lock per
# cache key so parallel calls for the same key share a single fetch.
_LAST_HOST: dict[str, float] = {}
_BACKOFF_UNTIL: dict[str, float] = {}
_ROBOTS: dict[str, object] = {}
_ROBOTS_FAILED: set[str] = set()
_INFLIGHT: dict[str, threading.Lock] = {}
_STATE_LOCK = threading.Lock()

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
                        _CACHE[obj["k"]] = (time.monotonic() - age, str(obj["v"]),
                                            str(obj.get("e", "")),
                                            str(obj.get("m", "")))
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
        lines = [json.dumps({"k": k, "t": now - (time.monotonic() - v[0]),
                             "v": v[1], "e": v[2], "m": v[3]})
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


def _http_get(url: str, headers: dict[str, str] | None = None) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, **(headers or {})}, method="GET")
    with _OPENER.open(req, timeout=TIMEOUT_SECS) as resp:  # noqa: S310
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read(FETCH_MAX_BYTES * 4).decode(charset, errors="replace")


def _http_get_bytes(url: str, etag: str = "",
                    lm: str = "") -> tuple[bytes, str, str, str]:
    """GET with the body as raw bytes plus the declared content type. For PDF
    routing: extracting a PDF as latin-1 text would yield garbage, so the
    caller must know the content type before decoding. ``etag``/``lm`` turn the
    GET conditional (If-None-Match / If-Modified-Since, 0.113.0): a 304 raises
    _NotModified so the caller can keep serving its cache — the politest fetch
    there is, and fresher than a blind long TTL. Returns (body, ctype, etag,
    last-modified) so the validators can be stored with the cache entry."""
    headers = {"User-Agent": UA}
    if etag:
        headers["If-None-Match"] = etag
    if lm:
        headers["If-Modified-Since"] = lm
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with _OPENER.open(req, timeout=TIMEOUT_SECS) as resp:  # noqa: S310
            ctype = (resp.headers.get_content_type() or "").lower()
            return (resp.read(FETCH_MAX_BYTES * 4), ctype,
                    resp.headers.get("ETag") or "",
                    resp.headers.get("Last-Modified") or "")
    except urllib.error.HTTPError as e:
        if e.code == 304 and (etag or lm):
            raise _NotModified() from e
        raise


def _pdf_sniff(body: bytes, ctype: str) -> bool:
    """Content-type says PDF, or the magic bytes say it anyway (some servers
    serve PDFs as application/octet-stream)."""
    if "pdf" in ctype:
        return True
    return body[:5] == b"%PDF-"


def _cache_get(key: str) -> tuple[str, str, str] | None:
    """(text, etag, last-modified) while fresh, else None."""
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_SECS:
        return hit[1], hit[2], hit[3]
    return None


def _cache_stale_entry(key: str) -> tuple[str, str, str] | None:
    """An EXPIRED entry worth revalidating (0.113.0): (text, etag,
    last-modified); the validators must be present or a conditional GET is
    pointless and a plain refetch is simpler."""
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] >= CACHE_TTL_SECS and (hit[2] or hit[3]):
        return hit[1], hit[2], hit[3]
    return None


def _cache_put(key: str, value: str, etag: str = "", lm: str = "") -> None:
    if len(_CACHE) > 64:  # bounded: drop everything past the TTL is overkill
        oldest = sorted(_CACHE, key=lambda k: _CACHE[k][0])[:-32]
        for k in oldest:
            _CACHE.pop(k, None)
    _CACHE[key] = (time.monotonic(), value, etag, lm)


def _host_of(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url).hostname or ""
    except ValueError:
        return ""


def _throttle_host(host: str) -> None:
    """Per-host courtesy floor (0.113.0) plus any outstanding 429/503 backoff.
    Loopback (the user's own SearXNG, test servers) is exempt."""
    if not host or host in ("127.0.0.1", "localhost", "::1", "[::1]"):
        return
    now = time.monotonic()
    wait = max(_LAST_HOST.get(host, 0.0) + HOST_MIN_INTERVAL_SECS - now,
               _BACKOFF_UNTIL.get(host, 0.0) - now)
    if wait > 0:
        time.sleep(wait)
    _LAST_HOST[host] = time.monotonic()


def _backoff_host(host: str, retry_after: str) -> None:
    try:
        secs = min(float(retry_after), _MAX_BACKOFF_SECS)
    except (TypeError, ValueError):
        secs = 10.0
    _BACKOFF_UNTIL[host] = max(_BACKOFF_UNTIL.get(host, 0.0),
                               time.monotonic() + secs)


def _robots_allowed(url: str) -> bool:
    """robots.txt for page fetches (0.113.0; OFF by default under SearXNG):
    parsers cached per host;
    unreachable/broken robots.txt means allowed (the standard polite
    default); loopback and the user's own SearXNG host are exempt. A
    ``Crawl-delay`` widens that host's fetch interval."""
    if not ROBOTS_ENABLED or ALLOW_PRIVATE:
        return True
    host = _host_of(url)
    if not host or host in ("127.0.0.1", "localhost", "::1", "[::1]"):
        return True
    if SEARX_URL and host == _host_of(SEARX_URL):
        return True
    if host in _ROBOTS_FAILED:
        return True
    cached = _ROBOTS.get(host)
    rp: Any = cached if callable(getattr(cached, "can_fetch", None)) else None
    if rp is None:
        fresh = RobotFileParser()
        try:
            scheme = urllib.parse.urlsplit(url).scheme or "https"
            fresh.set_url(f"{scheme}://{host}/robots.txt")
            fresh.read()  # urllib under the hood; a failure is allowed-by-default
        except Exception:
            _ROBOTS_FAILED.add(host)
            return True
        _ROBOTS[host] = fresh
        rp = fresh
    if not hasattr(rp, "crawl_delay") or not hasattr(rp, "can_fetch"):
        return True  # unreachable; narrows the Optional for ty
    try:
        delay = rp.crawl_delay(UA) or rp.crawl_delay("*")
        if delay:
            _LAST_HOST[host] = max(_LAST_HOST.get(host, 0.0),
                                   time.monotonic() - float(delay))
    except Exception:
        pass
    try:
        return bool(rp.can_fetch(UA, url))
    except Exception:
        return True


def _single_flight(key: str, fn):
    """Run `fn` holding a per-key lock (0.113.0): two parallel web_fetch calls
    for the same URL share one network fetch instead of doubling the load.
    `fn` must re-check the cache on entry."""
    with _STATE_LOCK:
        lock = _INFLIGHT.setdefault(key, threading.Lock())
    with lock:
        return fn()


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


def _searx_params(query: str, engines: str, categories: str,
                  time_range: str, pageno: int) -> dict[str, Any]:
    """Build the /search params: explicit tool arguments win; otherwise the
    query shape picks categories (it) and a time_range (day) automatically."""
    params: dict[str, Any] = {"q": query, "format": "json"}
    if engines:
        params["engines"] = engines
    if categories:
        params["categories"] = categories
    elif _SEARX_IT_RE.search(query):
        params["categories"] = "it"
    if time_range:
        params["time_range"] = time_range
    elif not engines and _SEARX_FRESH_RE.search(query):
        params["time_range"] = "day"
    if pageno > 1:
        params["pageno"] = pageno
    return params


def _searx_search(query: str, engines: str = "", categories: str = "",
                  time_range: str = "", pageno: int = 1) -> str | None:
    """SearXNG JSON-API search; None when unset/unreachable/broken (caller
    falls back to DDG lite). One GET to the user's own instance — the DDG
    throttle does not apply (it exists to spare a SHARED free endpoint).
    Beyond `results`, instant answers, infoboxes and query suggestions from
    the response are surfaced when present (0.148.9)."""
    if not SEARX_URL or not _searx_usable():
        return None
    params = _searx_params(query, engines, categories, time_range, pageno)
    qs = urllib.parse.urlencode(params)
    headers: dict[str, str] = {}
    if SEARX_AUTH:
        token = base64.b64encode(SEARX_AUTH.encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    try:
        body = _http_get(f"{SEARX_URL}/search?{qs}", headers=headers)
        data = json.loads(body)
        results = data.get("results") or []
    except Exception:
        return None  # unreachable or bad JSON: fall through to DDG lite
    limit = 20 if pageno > 1 else MAX_SEARCH_RESULTS
    out = []
    for r in results[:limit]:
        url = str(r.get("url") or "")
        if not url.startswith(("http://", "https://")):
            continue
        if not ALLOW_PRIVATE and _url_is_private(url):
            continue  # a compromised engine must not seed an SSRF pivot
        title = str(r.get("title") or "").strip()
        content = str(r.get("content") or "").strip()
        out.append(f"- {title} — {url}" + (f"\n  {content}" if content else ""))
    # Instant answers (calculator, special engines) lead the result list;
    # infoboxes are free definitional context (wikipedia/wikidata).
    for ans in (data.get("answers") or [])[:3]:
        text = ans.get("answer") if isinstance(ans, dict) else str(ans)
        if text:
            out.insert(0, f"= {str(text).strip()}")
    for box in (data.get("infoboxes") or [])[:1]:
        inf = str(box.get("content") or box.get("infobox") or "").strip()
        if inf:
            out.insert(0, f"[info] {inf[:600]}")
    sugg = [str(s) for s in (data.get("suggestions") or [])[:5] if s]
    if sugg:
        out.append("Related queries: " + "; ".join(sugg))
    n_total = data.get("number_of_results")
    unresp = data.get("unresponsive_engines") or []
    notes = []
    if isinstance(n_total, int) and n_total > 0:
        notes.append(f"~{n_total} total results")
    if unresp:
        names = ", ".join(
            (u[0] if isinstance(u, (list, tuple)) else str(u)) for u in unresp)
        notes.append(f"unresponsive engines: {names}")
    if notes:
        out.append("(" + "; ".join(notes) + ")")
    if not out:
        return "(SearXNG returned no results for this query.)"
    return "\n".join(out)


def _web_search(query: str, engines: str = "", categories: str = "",
                time_range: str = "", pageno: int = 1) -> str:
    # courtesy throttle for the shared free endpoint (DDG lite); the user's
    # own SearXNG instance, when configured, is exempt.
    if not SEARX_URL:
        now = time.monotonic()
        wait = _LAST_SEARCH[0] + DDG_MIN_INTERVAL_SECS - now
        if wait > 0:
            time.sleep(wait)
        _LAST_SEARCH[0] = time.monotonic()

    cache_key = ("q:" + query.lower().strip()
                 + f"|{engines}|{categories}|{time_range}|{pageno}")
    _ensure_cache()
    if (hit := _cache_get(cache_key)) is not None:
        return "[cached] " + hit[0]

    # SearXNG first (preferred); DDG lite is the fallback. A fallback while
    # NX_SEARX_URL is set usually means the instance is misconfigured (json
    # format off) or down — say so instead of failing silently (0.113.0).
    searx = _searx_search(query, engines=engines, categories=categories,
                          time_range=time_range, pageno=pageno)
    searx_hint = ""
    if searx is None and SEARX_URL:
        searx_hint = (" (note: NX_SEARX_URL is set but the instance returned "
                      "nothing — check it is up and that settings.yml has "
                      "search: formats: [html, json])")
    if searx is not None:
        _cache_put(cache_key, searx)
        _save_cache()
        return searx

    qs = urllib.parse.urlencode({"q": query, "kl": "wt-wt"})
    try:
        body = _http_get(f"{SEARCH_URL}?{qs}")
    except Exception as e:
        return (f"ERROR: search failed ({type(e).__name__}){searx_hint}; "
                "duckduckgo may be rate-limiting — wait a few seconds and "
                "retry, or ask the user for a URL to fetch directly.")
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


def _fetch_network(url: str, cache_key: str,
                   stale: tuple[str, str, str] | None) -> str:
    """The network leg of a page fetch, under the per-key single-flight lock;
    re-checks the cache on entry (a parallel call may have just filled it).
    Handles conditional revalidation of an expired entry and 429/503 backoff."""
    if (hit := _cache_get(cache_key)) is not None:
        return "[cached] " + hit[0]
    host = _host_of(url)
    etag, lm = (stale[1], stale[2]) if stale else ("", "")
    try:
        _throttle_host(host)
        body, ctype, etag, lm = _http_get_bytes(url, etag=etag, lm=lm)
    except _NotModified:
        # 304: the origin says the cached page is still good — refresh the
        # timestamp (restarts the TTL) and serve it. No body was transferred.
        text = stale[0] if stale else ""
        _cache_put(cache_key, text, etag, lm)
        _save_cache()
        return "[cached] " + text
    except urllib.error.HTTPError as e:
        if e.code in (429, 503) and host:
            ra = e.headers.get("Retry-After", "") if e.headers else ""
            _backoff_host(host, ra)
            return (f"ERROR: fetch failed (HTTP {e.code}); the host is asking "
                    "clients to back off — wait a bit before retrying this URL.")
        return (f"ERROR: fetch failed ({type(e).__name__}); the site may block "
                "scripts or need a browser — try a different URL or ask the user.")
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
    _cache_put(cache_key, text, etag, lm)
    _save_cache()
    return text


def _web_fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    if not ALLOW_PRIVATE and _url_is_private(url):
        return ("ERROR: refused to fetch a loopback/private-address URL "
                "(SSRF guard; set NX_WEBFETCH_ALLOW_PRIVATE=1 if the URL is "
                "deliberately internal).")
    if not _robots_allowed(url):
        return ("ERROR: the site's robots.txt disallows fetching this URL — "
                "ask the user to fetch it themselves.")
    cache_key = "u:" + url
    _ensure_cache()
    stale = _cache_stale_entry(cache_key)
    if (hit := _cache_get(cache_key)) is not None:
        return "[cached] " + hit[0]
    return _single_flight(cache_key,
                          lambda: _fetch_network(url, cache_key, stale))


# --- tool body ---------------------------------------------------------------

def web_fetch(query: str = "", url: str = "", engines: str = "",
              categories: str = "", time_range: str = "", pageno: int = 1) -> str:
    """Tool body for the web_fetch tier-1 tool. Exactly one of query/url.
    The SearXNG-only knobs (engines/categories/time_range/pageno) scope the
    search when an instance is configured; ignored by the DDG fallback."""
    if bool(query) == bool(url):
        return "ERROR: pass exactly one of `query` (web search) or `url` (fetch a page)"
    if query:
        return _web_search(query, engines=engines, categories=categories,
                           time_range=time_range, pageno=pageno)
    return _web_fetch_url(url)


WEB_FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Keyless web access, GET-only and rate-limited to stay polite to free endpoints (results are cached globally for 7 days, stale pages revalidated via ETag/Last-Modified, robots.txt honored). Pass `query` for a web search (SearXNG via NX_SEARX_URL when configured, else DuckDuckGo-lite) returning title/URL/snippet results, or `url` to fetch a page as readable text (links kept as [text](url)). Never both. For the sandbox's own docs, prefer the local files — this is for the outside world.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search string (SearXNG via NX_SEARX_URL when configured, else DuckDuckGo lite); returns title/URL/snippet results to follow up with a url fetch"},
                "engines": {"type": "string", "description": "SearXNG only: comma-separated engines to pin (e.g. 'github,stackoverflow,pypi')"},
                "categories": {"type": "string", "description": "SearXNG only: comma-separated categories (e.g. 'it', 'science', 'news'); 'it' is auto-picked for package/API queries"},
                "time_range": {"type": "string", "enum": ["day", "month", "year"], "description": "SearXNG only: freshness filter; 'day' is auto-picked for release/CVE queries"},
                "pageno": {"type": "integer", "description": "SearXNG only: result page number (1-based), for digging past the first page"},
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
