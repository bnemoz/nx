"""web_fetch (0.70.0): keyless, free web access — no paid API.

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
every response is cached in-process for the session, a minimum interval is
enforced between requests to duckduckgo.com, a small polite User-Agent is
sent, and only ``GET`` requests are ever issued. Failures fail soft with a
message the model can act on (retry later, narrower query, different URL) —
never an exception into the turn loop.
"""
from __future__ import annotations

import html as _html
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from ._env import _env_int

WEBFETCH_ENABLED = _env_int("NX_WEBFETCH", 1)
FETCH_MAX_BYTES = _env_int("NX_WEBFETCH_MAX_BYTES", 20000)
CACHE_TTL_SECS = _env_int("NX_WEBFETCH_CACHE_TTL", 600)
# Courtesy floor between requests to duckduckgo.com (search OR the fetch of a
# page hosted anywhere is NOT throttled — only the search endpoint is, since
# that's the shared free resource).
DDG_MIN_INTERVAL_SECS = _env_int("NX_WEBFETCH_SEARCH_INTERVAL", 3.0)

UA = "nx-coding-agent/0.70 (+local CLI; respectful, cached, GET-only)"
SEARCH_URL = "https://html.duckduckgo.com/html/"
TIMEOUT_SECS = 15
MAX_SEARCH_RESULTS = 10

# --- session state (module-level on purpose: conftest rewinds nx.* dicts) ---
_CACHE: dict[str, tuple[float, str]] = {}
_LAST_SEARCH: list[float] = [0.0]


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

def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:  # noqa: S310
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read(FETCH_MAX_BYTES * 4).decode(charset, errors="replace")


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


def _web_search(query: str) -> str:
    # courtesy throttle: never hammer the free endpoint
    now = time.monotonic()
    wait = _LAST_SEARCH[0] + DDG_MIN_INTERVAL_SECS - now
    if wait > 0:
        time.sleep(wait)
    _LAST_SEARCH[0] = time.monotonic()

    cache_key = "q:" + query.lower().strip()
    if (hit := _cache_get(cache_key)) is not None:
        return "[cached] " + hit

    qs = urllib.parse.urlencode({"q": query, "kl": "wt-wt"})
    try:
        body = _http_get(f"{SEARCH_URL}?{qs}")
    except Exception as e:
        return (f"ERROR: search failed ({type(e).__name__}); duckduckgo may be "
                "rate-limiting — wait a few seconds and retry, or ask the user "
                "for a URL to fetch directly.")
    results = _parse_results(body)
    if not results:
        return ("No results (the lite endpoint's HTML may have changed — check "
                "the layout before assuming the query matched nothing).")
    lines = [
        f"- {r['title']} — {r['url']}" + (f"\n  {r['snippet']}" if r["snippet"] else "")
        for r in results
    ]
    text = "\n".join(lines)
    _cache_put(cache_key, text)
    return text


# --- fetch -------------------------------------------------------------------

def _web_fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    cache_key = "u:" + url
    if (hit := _cache_get(cache_key)) is not None:
        return "[cached] " + hit
    try:
        body = _http_get(url)
    except Exception as e:
        return (f"ERROR: fetch failed ({type(e).__name__}); the site may block "
                "scripts or need a browser — try a different URL or ask the user.")
    text = _html_to_text(body)
    if len(text) > FETCH_MAX_BYTES:
        text = text[:FETCH_MAX_BYTES] + f"\n\n[truncated at {FETCH_MAX_BYTES} bytes; fetch a narrower URL for more]"
    _cache_put(cache_key, text)
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
        "description": "Keyless web access, GET-only and rate-limited to stay polite to the free DuckDuckGo endpoints (results are cached per session). Pass `query` for a web search returning title/URL/snippet results, or `url` to fetch a page as readable text (links kept as [text](url)). Never both. For the sandbox's own docs, prefer the local files — this is for the outside world.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search string (DuckDuckGo lite); returns up to 10 title/URL/snippet results to follow up with a url fetch"},
                "url": {"type": "string", "description": "Absolute http(s) URL to fetch and convert to readable text; truncated at NX_WEBFETCH_MAX_BYTES"},
            },
            "required": [],
        },
    },
}
