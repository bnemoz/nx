"""nx talks to any OpenAI-compatible endpoint. A "source" bundles a
base_url, api_key, and model name. Define sources with env vars:

  NX_SOURCES=local,zai
  NX_SOURCE_LOCAL_BASE_URL=http://localhost:8080/v1
  NX_SOURCE_ZAI_BASE_URL=https://api.z.ai/api/paas/v4
  NX_SOURCE_ZAI_API_KEY=$zai_test        <- $name = look up env/file key
  NX_SOURCE_ZAI_MODEL=glm-x-preview

If no NX_SOURCE_* vars are present, a single "local" source is built
from the legacy NX_BASE_URL / NX_API_KEY / NX_MODEL vars
(same defaults as before, so existing setups keep working).
Switch at runtime with /source.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from typing import Any

from ._openai_client import OpenAI

from ._ansi import RESET, C
from . import _state as st

# --- model sources ----------------------------------------------------------
# nx talks to any OpenAI-compatible endpoint. A "source" bundles a
# base_url, api_key, and model name. Define sources with env vars:
#
#   NX_SOURCES=local,zai
#   NX_SOURCE_LOCAL_BASE_URL=http://localhost:8080/v1
#   NX_SOURCE_ZAI_BASE_URL=https://api.z.ai/api/paas/v4
#   NX_SOURCE_ZAI_API_KEY=$zai_test        ← $name = look up env/file key
#   NX_SOURCE_ZAI_MODEL=glm-x-preview
#
# If no NX_SOURCE_* vars are present, a single "local" source is built
# from the legacy NX_BASE_URL / NX_API_KEY / NX_MODEL vars
# (same defaults as before, so existing setups keep working).
# Switch at runtime with /source.

class Source:
    def __init__(self, name: str, base_url: str | None, api_key: str | None,
                 model: str | None = None, extra_body: dict[str, Any] | None = None,
                 http_headers: dict[str, str] | None = None):
        self.name = name
        self.base_url = base_url
        if base_url and not re.search(r"/v\d+/?$", base_url):
            sys.stderr.write(
                f"nx: warning: source {name!r} base_url {base_url!r} has no "
                f"/v1 suffix — OpenAI-compat endpoints (/chat/completions, "
                f"/models) live under /v1; requests will likely 404\n"
            )
        self.api_key = api_key or "sk-noop"
        self.model = model or None  # None → ask the server at resolve time
        # Default HTTP headers sent on every request. Used by aggregators like
        # OpenRouter which read HTTP-Referer (your site URL) and X-Title (your
        # app name) to identify the calling app in their dashboard. Ignored by
        # non-consumers (llama.cpp, Together, …) so it's safe to always pass.
        self.http_headers = http_headers or None
        client_kwargs: dict[str, Any] = {"base_url": base_url, "api_key": self.api_key}
        if self.http_headers:
            client_kwargs["default_headers"] = self.http_headers
        self.client = OpenAI(**client_kwargs)
        # Lazily-resolved max context window (in tokens) for the active model,
        # or None when no probe could determine it. Both a value and a miss are
        # cached: an over-max_tokens probe can be *accepted* by servers which
        # silently clamp it, and retrying that request can generate a very large
        # response on every turn.
        # Call resolve_context_window(force=True) to re-probe (e.g. after a
        # /source <name> <model> override pins a different model id).
        self._context_window = None
        self._context_window_probe_attempted = False
        # Whether this source's /tokenize endpoint has been confirmed working
        # (True), confirmed absent/broken (False), or not yet tried (None).
        self._tokenize_ok = None
        # Tool-calling protocol this source's model actually uses, one of
        # _capability.CAPABILITY_NATIVE / _GRAMMAR / _TEXT, or None before the
        # startup probe (nx/_capability.py: ensure_capability_probe()) lands.
        # Cached here (not a global) so switching back to a previously-probed
        # source doesn't need to re-learn it. Reset in switch_source().
        self._capability = None
        # Per-request body fields the OpenAI SDK doesn't accept as kwargs, merged
        # into every chat.completions.create() call via extra_body. The canonical
        # use is OpenRouter provider routing — a `provider` object whose `order`
        # picks which upstream serves a given model id (different providers back
        # the same endpoint at different prices / precisions / latencies):
        #
        #   {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false,
        #                 "require_parameters": true, "data_collection": "deny"}}
        #
        # For local servers (llama.cpp) this is empty and contributes nothing;
        # unknown keys are silently ignored by non-consumers, so it's safe to
        # always pass it through. Set via NX_SOURCE_<NAME>_EXTRA_BODY (a JSON
        # string) or the /provider slash command (ephemeral, per-session).
        self.extra_body = extra_body or None

    def extra_request_kwargs(self) -> dict[str, Any]:
        """extra_body kwarg to merge into every chat.completions.create(), or
        {} when none is configured. Non-consumers ignore unknown body keys, so
        this is a no-op for backends that don't understand it (llama.cpp,
        Together, …)."""
        if not self.extra_body:
            return {}
        return {"extra_body": dict(self.extra_body)}

    # Server-advertised ids that carry no real identity — when /v1/models
    # returns one of these (bare llama.cpp with no --alias, or a generic
    # OpenAI-compat stub), fall through to a /props probe before giving up so
    # saved sessions don't all collapse into a meaningless "local-model".
    _GENERIC_MODEL_IDS = {"local-model", "model", "auto", "unknown", "gpt-3.5-turbo"}

    @staticmethod
    def _clean_model_id(advertised: str | None) -> str | None:
        """Shorten a server-advertised model id into a concise display name.

        Only touches local-server ids that look like file paths (llama.cpp,
        Ollama, …) where /v1/models returns the full GGUF/HF file path. Remote
        API ids in org/model form (e.g. "zai-org/GLM-5.2") are returned
        unchanged.

        Examples:
          /media/h/.../GLM-5.2-GGUF/UD-IQ4_NL/GLM-5.2-UD-IQ4_NL-00001-of-00009.gguf
            → GLM-5.2-UD-IQ4_NL
          /models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
            → Meta-Llama-3-8B-Instruct-Q4_K_M
        Anything that doesn't look like a local model file path is returned
        as-is so we never mangle a legitimate remote model id.
        """
        if not advertised:
            return advertised
        # Only clean up local file paths — ones that end in a model file
        # extension. Anything else (org/model, bare names) is passed through.
        lower = advertised.lower()
        if not any(lower.endswith(ext) for ext in (".gguf", ".bin", ".safetensors")):
            return advertised
        name = advertised.rsplit("/", 1)[-1]
        # Strip shard suffix on multi-file quantizations
        # (e.g. "-00001-of-00009")
        name = re.sub(r"-\d{5}-of-\d{5}(?=$|\.)", "", name)
        # Strip extension
        for ext in (".gguf", ".bin", ".safetensors"):
            if name.lower().endswith(ext):
                name = name[: -len(ext)]
                break
        return name

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        advertised = None
        try:
            advertised = self.client.models.list(timeout=10).data[0].id
        except Exception:
            advertised = None
        if advertised and advertised.strip().lower() not in self._GENERIC_MODEL_IDS:
            return self._clean_model_id(advertised) or advertised
        # The endpoint gave us nothing useful. Many local servers (llama.cpp's
        # llama-server) expose the actually-loaded model file at /props even
        # when /v1/models only advertises a generic id — recover the real name
        # (incl. quant) from there. Best-effort; any failure keeps the old value.
        probed = self._probe_loaded_model()
        return probed or advertised or "local-model"

    def _probe_loaded_model(self) -> str | None:
        """Best-effort GET <server>/props for the loaded model path/alias.
        llama.cpp serves /props at the root (not under /v1). Returns None on
        any failure — never raises, never required, makes no assumption about
        what (if anything) consumes nx's session metrics."""
        try:
            root = re.sub(r"/v\d+/?$", "", (self.base_url or "").rstrip("/"))
            with urllib.request.urlopen(root + "/props", timeout=5) as r:
                props = json.loads(r.read().decode("utf-8"))
        except TimeoutError:
            print(f"nx: warning: /props model-name probe on {self.name} timed out", file=sys.stderr)
            return None
        except Exception:
            return None
        for key in ("model_alias", "model_path", "model"):
            val = props.get(key)
            if isinstance(val, str) and val.strip() and \
                    val.strip().lower() not in self._GENERIC_MODEL_IDS:
                return val.strip()
        return None

    def display_model(self) -> str:
        return self.model or "auto"

    def tokenize_count(self, text: str) -> int | None:
        """Best-effort exact token count via a local server's /tokenize
        endpoint (llama.cpp convention: POST {"content": text} -> {"tokens":
        [...]}"). Only attempted for local/LAN sources — remote aggregators
        (Together, Z.ai, OpenRouter) don't expose this and shouldn't eat a
        multi-second TLS round-trip just to 404. Caches a negative result on
        self._tokenize_ok so a single unsupported server doesn't pay the
        timeout again for each of --tokens' several segments; never raises."""
        if self._tokenize_ok is False or not self._is_local():
            return None
        try:
            root = re.sub(r"/v\d+/?$", "", (self.base_url or "").rstrip("/"))
            req = urllib.request.Request(
                root + "/tokenize",
                data=json.dumps({"content": text}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))
            tokens = data.get("tokens")
            if not isinstance(tokens, list):
                self._tokenize_ok = False
                return None
            self._tokenize_ok = True
            return len(tokens)
        except Exception:
            self._tokenize_ok = False
            return None

    # Regex for the provider-agnostic context-limit probe: Together (and any
    # OpenAI-compat backend that mirrors OpenAI's error wording) rejects an
    # over-large request with "This model's maximum context length is N tokens,
    # but the request …". We send a deliberately over-sized max_tokens with a
    # one-token prompt so the server tells us N without us ever generating.
    _CTX_LIMIT_RE = re.compile(r"maximum context length is (\d+) tokens", re.IGNORECASE)

    def _is_local(self) -> bool:
        """True if this source points at a host on the local machine or LAN
        (llama.cpp, Ollama, etc.) rather than a remote API (Together, Z.ai,
        OpenAI, …). Local servers expose /v1/models meta and /props cheaply
        (sub-millisecond LAN round-trips); remote hosts return empty lists or
        404 there and waste a multi-second TLS round-trip, so the probe order
        is flipped for them (over-max_tokens first)."""
        host = (self.base_url or "").lower()
        # strip the scheme
        if "://" in host:
            host = host.split("://", 1)[1]
        host = host.split("/", 1)[0].split(":", 1)[0]
        if not host:
            return True  # blank → assume local
        if host in ("localhost", "0.0.0.0", "::1"):
            return True
        # 127.x.x.x / 10.x / 192.168.x / 169.254.x / 172.16-31.x are LAN
        if re.match(r"^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2[0-9]|3[01])\.)", host):
            return True
        return False

    def resolve_context_window(self, model: str | None = None, force: bool = False) -> int | None:
        """Best-effort max context window (tokens) for `model` on this source.

        Returns the cached value if present (unless force=True), else probes
        and caches. Never raises — any failure returns None and the footer
        just omits the "/<max>" suffix. Probe order depends on host locality:

          • Local (llama.cpp etc.) — /v1/models meta, then /props, then the
            over-max_tokens chat probe. Local metadata endpoints are cheap
            (sub-ms LAN) and llama.cpp stashes n_ctx in data[0].meta, so they
            win on the first try.

          • Remote (Together, Z.ai, OpenAI, …) — /v1/models FIRST, then the
            over-max_tokens chat probe as a fallback. The chat probe is cheap
            and universal but some backends clamp-and-generate instead of
            rejecting with a 400, which would bill real tokens and add
            latency on every newly-activated paid endpoint; /v1/models never
            risks that, so it leads even though it costs a multi-second
            TLS round-trip and often returns nothing.

        `model` defaults to the source's configured model, falling back to a
        live resolve_model() so an auto source is probed against the model it
        will actually use. The result is keyed to that model id, so a
        /source <name> <other-model> override should call with force=True.

        A failed attempt is also cached. In particular, LM Studio-like
        backends can silently clamp the deliberately oversized completion
        instead of returning the 400 this probe needs; repeatedly issuing that
        request can hang or consume tokens indefinitely. Use force=True to
        explicitly retry after a transient failure (for example, once a local
        server has finished loading)."""
        if not force and (self._context_window is not None or
                          self._context_window_probe_attempted):
            return self._context_window
        try:
            mid = model or self.model or self.resolve_model()
            if self._is_local():
                n = self._ctx_from_models(mid) or self._ctx_from_props() \
                    or self._ctx_from_lmstudio(mid)
                if n is None:
                    n = self._ctx_from_overrun_probe(mid)
            else:
                # Remote: try /v1/models first. It costs a multi-second
                # round-trip but never risks real tokens/billing; the
                # over-max_tokens chat probe is cheap and universal but some
                # backends clamp-and-generate instead of rejecting with 400,
                # which would otherwise bill/incur latency on every
                # newly-activated paid endpoint. Only fall back to it if the
                # model metadata doesn't expose a context length.
                n = self._ctx_from_models(mid) or self._ctx_from_props()
                if n is None:
                    n = self._ctx_from_overrun_probe(mid)
            if isinstance(n, int) and n > 0:
                self._context_window = n
        except Exception:
            pass  # context metadata is always best-effort
        finally:
            # Cache misses too; see the method docstring. force=True remains
            # an explicit escape hatch for local-server cold starts.
            self._context_window_probe_attempted = True
        return self._context_window

    def _ctx_from_models(self, mid: str | None) -> int | None:
        """Pull n_ctx / context_length from GET /v1/models. llama.cpp stashes
        it under data[0].meta.n_ctx; some generic hosts expose context_length
        as a top-level model field. Returns int or None."""
        try:
            data = self.client.models.list(timeout=10).data
        except Exception:
            return None
        if not data:
            return None
        # Prefer the entry whose id matches the active model (multi-model
        # hosts), else the first. Match loosely on a trailing token so a full
        # path id ("…/GLM-5.2-…gguf") still hits a short alias ("GLM-5.2").
        pick = None
        if mid:
            mtail = mid.rsplit("/", 1)[-1].lower()
            for m in data:
                if (m.id or "").lower() == mid.lower() or \
                   (m.id or "").rsplit("/", 1)[-1].lower() == mtail:
                    pick = m
                    break
        if pick is None and data:
            pick = data[0]
        if pick is None:
            return None
        # Model objects carry unknown/extra fields (e.g. llama.cpp's "meta")
        # straight in their raw dict.
        extra = pick.model_dump() or {}
        meta = extra.get("meta") if isinstance(extra, dict) else None
        if isinstance(meta, dict):
            for k in ("n_ctx", "context_length", "context_window",
                      "max_context_length", "max_input_tokens"):
                v = meta.get(k)
                if isinstance(v, int) and v > 0:
                    return v
        # Some hosts put it directly on the model object (OpenAI-compat extra).
        for k in ("context_length", "context_window", "max_context_length",
                  "max_input_tokens", "max_tokens", "max_model_len"):
            v = extra.get(k) if isinstance(extra, dict) else None
            if v is None:
                v = getattr(pick, k, None)
            if isinstance(v, int) and v > 0:
                return v
        return None

    def _ctx_from_props(self) -> int | None:
        """llama.cpp /props fallback: default_generation_settings.n_ctx."""
        try:
            root = re.sub(r"/v\d+/?$", "", (self.base_url or "").rstrip("/"))
            with urllib.request.urlopen(root + "/props", timeout=5) as r:
                props = json.loads(r.read().decode("utf-8"))
        except TimeoutError:
            print(f"nx: warning: /props ctx-window probe on {self.name} timed out", file=sys.stderr)
            return None
        except Exception:
            return None
        dgs = props.get("default_generation_settings")
        if isinstance(dgs, dict):
            for k in ("n_ctx", "context_length", "context_window"):
                v = dgs.get(k)
                if isinstance(v, int) and v > 0:
                    return v
        # Some builds put n_ctx at the top level of /props.
        for k in ("n_ctx", "context_length", "context_window"):
            v = props.get(k)
            if isinstance(v, int) and v > 0:
                return v
        return None

    def _ctx_from_lmstudio(self, mid: str | None) -> int | None:
        """LM Studio's native GET /api/v0/models exposes loaded_context_length
        (the context the model is actually running with) per entry — unlike
        the OpenAI-compat /v1/models, which omits any length field, and unlike
        the overrun probe, which doesn't work: LM Studio silently clamps an
        over-large max_tokens instead of returning a 400. Returns int or
        None."""
        try:
            root = re.sub(r"/v\d+/?$", "", (self.base_url or "").rstrip("/"))
            with urllib.request.urlopen(root + "/api/v0/models", timeout=5) as r:
                data = json.loads(r.read().decode("utf-8")).get("data") or []
        except Exception:
            return None
        pick = None
        if mid:
            mtail = mid.rsplit("/", 1)[-1].lower()
            for m in data:
                mid_field = (m.get("id") or "").lower()
                if mid_field == mid.lower() or \
                   mid_field.rsplit("/", 1)[-1] == mtail:
                    pick = m
                    break
        if pick is None and data:
            pick = data[0]
        if not isinstance(pick, dict):
            return None
        v = pick.get("loaded_context_length") or pick.get("max_context_length")
        return v if isinstance(v, int) and v > 0 else None

    def _ctx_from_overrun_probe(self, mid: str | None) -> int | None:
        """Send a chat completion with max_tokens far beyond any real context
        and a 1-token prompt; parse the 400's "maximum context length is N
        tokens" message. Works for Together and any host mirroring OpenAI's
        over-limit wording. No tokens are generated (the request is rejected
        before inference). Returns int or None."""
        if not mid:
            return None
        try:
            self.client.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10_000_000,
                stream=False,
                timeout=30,
            )
        except Exception as e:
            msg = str(e)
            body = getattr(e, "body", None)
            if isinstance(body, dict):
                # Together nests it under body["message"]["message"]; OpenAI
                # under body["error"]["message"]. Flatten to a string search.
                msg = json.dumps(body) + " " + msg
            if m := self._CTX_LIMIT_RE.search(msg):
                return int(m.group(1))
        return None


def _resolve_api_key(val: str | None) -> str | None:
    """$name → look up a non-empty env var; otherwise return the literal.

    A missing reference is a configuration error. Returning an empty string
    here would let ``Source`` replace it with its local-server ``sk-noop``
    placeholder, making a typo in a remote API key look valid until a request
    fails much later.
    """
    if val and val.startswith("$"):
        name = val[1:]
        resolved = os.environ.get(name)
        if not resolved:
            raise ValueError(
                f"nx: API key reference {val!r} is unresolved: "
                f"environment variable {name!r} is not set or empty")
        return resolved
    return val


def _parse_extra_body(raw: str | None) -> dict[str, Any] | None:
    """Parse a NX_SOURCE_<NAME>_EXTRA_BODY env value into a dict, or None.

    Accepts a JSON object string and merges it into every chat request as the
    OpenAI SDK's `extra_body` — the canonical channel for fields the SDK doesn't
    model as top-level kwargs. The primary use case is OpenRouter provider
    routing (a `provider` object whose `order` picks which upstream serves a
    model id). Returns None on a blank/invalid value (with a stderr warning on
    bad JSON) so a typo never silently drops routing — it fails loudly at load.

    Example:
        NX_SOURCE_OR_EXTRA_BODY={"provider":{"order":["parasail/fp8"],"allow_fallbacks":false}}
    """
    if not raw or not raw.strip():
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.stderr.write(
            f"nx: ignoring bad NX_*_EXTRA_BODY JSON ({e.msg} at "
            f"char {e.pos}); provider routing is NOT set\n")
        return None
    if isinstance(obj, dict) and obj:
        return obj
    if isinstance(obj, dict):
        return None  # empty object {} → no routing, same as unset
    sys.stderr.write(
        "nx: NX_*_EXTRA_BODY must be a JSON object, ignoring "
        f"(got {type(obj).__name__}); provider routing is NOT set\n")
    return None


def _build_http_headers(app_name: str | None = None,
                         app_url: str | None = None) -> dict[str, str] | None:
    """Build default HTTP headers for API requests, for aggregators like
    OpenRouter which read HTTP-Referer (your site URL) and X-Title (your app
    name) to identify the calling app in their dashboard.

    Set via NX_SOURCE_<NAME>_APP_NAME and NX_SOURCE_<NAME>_APP_URL.
    Returns None when neither is provided so the OpenAI st.client is constructed
    without default_headers (no overhead for non-aggregator sources).
    """
    headers = {}
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_name:
        headers["X-Title"] = app_name
    return headers or None


def _discover_sources() -> None:
    """Build st.SOURCES + st.SOURCE_ORDER from NX_SOURCE_* env vars, falling
    back to a single 'local' source from the legacy NX_* vars."""
    names = []
    raw = os.environ.get("NX_SOURCES", "")
    if raw:
        names = [n.strip() for n in raw.split(",") if n.strip()]
    # auto-discover from NX_SOURCE_<NAME>_BASE_URL if NX_SOURCES absent
    if not names:
        prefix = "NX_SOURCE_"
        found = []
        for k in os.environ:
            if k.startswith(prefix) and k.endswith("_BASE_URL"):
                found.append(k[len(prefix):-len("_BASE_URL")].lower())
        names = sorted(found)
    # Warn (don't fail) on case-insensitive name collisions in NX_SOURCES —
    # e.g. "local,Local" both resolve to NX_SOURCE_LOCAL_* and would
    # otherwise silently clobber one entry in st.SOURCES with the other.
    seen_lower = {}
    for name in names:
        lower = name.lower()
        if lower in seen_lower and seen_lower[lower] != name:
            sys.stderr.write(
                f"nx: NX_SOURCES has {seen_lower[lower]!r} and {name!r} — "
                f"both resolve to NX_SOURCE_{lower.upper()}_*, "
                f"only one will be kept\n")
        seen_lower.setdefault(lower, name)
    for name in names:
        p = f"NX_SOURCE_{name.upper()}_"
        base_url = os.environ.get(p + "BASE_URL")
        if not base_url:
            continue
        api_key = _resolve_api_key(os.environ.get(p + "API_KEY"))
        model = os.environ.get(p + "MODEL")
        extra_body = _parse_extra_body(os.environ.get(p + "EXTRA_BODY"))
        # HTTP headers for aggregators like OpenRouter: HTTP-Referer (site
        # URL) and X-Title (app name) identify the calling app in their
        # dashboard. Set via NX_SOURCE_<NAME>_APP_NAME /
        # NX_SOURCE_<NAME>_APP_URL.
        http_headers = _build_http_headers(
            os.environ.get(p + "APP_NAME"),
            os.environ.get(p + "APP_URL"))
        src = Source(name, base_url, api_key, model, extra_body=extra_body,
                     http_headers=http_headers)
        st.SOURCES[name] = src
        st.SOURCE_ORDER.append(name)
    if not st.SOURCES:
        # legacy fallback: one source from NX_BASE_URL etc.
        _register_builtin(
            "local",
            os.environ.get("NX_BASE_URL", "http://localhost:8080/v1"),
            api_key=os.environ.get("NX_API_KEY", "sk-noop"),
            model=os.environ.get("NX_MODEL"),
            require_key=False,
        )

    # Built-in convenience source: Together AI. It's a multi-model host (you
    # point it at any model by id), so unlike the local fallback it only
    # appears when a key is actually available — and only when the user
    # hasn't already defined a "together" source themselves (so an explicit
    # NX_SOURCE_TOGETHER_* config always wins). Registered last, so the
    # user's first/source-listed source stays the default at startup and
    # `/source together` (or `--source together`) is purely opt-in.
    _register_builtin(
        "together",
        "https://api.together.xyz/v1",
        api_key=_resolve_api_key(os.environ.get("TOGETHER_API_KEY", "")),
        model="zai-org/GLM-5.2",  # default model; override per-switch via /source together <model>
    )

    # Built-in convenience source: OpenRouter. Like Together it's a multi-model
    # host (you address any model by id), so it only registers when a key is
    # available and only when the user hasn't defined an "openrouter" source
    # themselves (an explicit NX_SOURCE_OPENROUTER_* config always wins).
    # Registered last, so it never displaces the user's default startup source —
    # opt in with `/source openrouter` (or `--source openrouter`).
    #
    # The difference from Together is provider routing: OpenRouter fronts many
    # upstream providers behind one model id, each with its own price / precision
    # / latency / data-collection policy. Routing rides in the request body as a
    # `provider` object (extra_body), so the built-in seeds it from
    # NX_SOURCE_OPENROUTER_EXTRA_BODY (a JSON string) if present, and the
    # ephemeral /provider command can override it per-session. The default model
    # is z-ai/glm-5.2; the default routing pins parasail/fp8 — cheap, fast, and a
    # deliberate single-provider choice so you know exactly what's serving you.
    or_body = _parse_extra_body(
        os.environ.get("NX_SOURCE_OPENROUTER_EXTRA_BODY", ""))
    if or_body is None:
        or_body = {"provider": {"order": ["parasail/fp8"],
                                "allow_fallbacks": False}}
    _register_builtin(
        "openrouter",
        "https://openrouter.ai/api/v1",
        api_key=_resolve_api_key(os.environ.get("OPENROUTER_API_KEY", "")),
        model="z-ai/glm-5.2",  # default model; override via /source openrouter <model>
        extra_body=or_body,
        http_headers=_build_http_headers(
            os.environ.get("NX_SOURCE_OPENROUTER_APP_NAME", "Nx"),
            os.environ.get("NX_SOURCE_OPENROUTER_APP_URL",
                           "https://github.com/Sentdex/nx")),
    )


def _register_builtin(name: str, base_url: str, api_key: str | None,
                       model: str | None = None,
                       extra_body: dict[str, Any] | None = None,
                       http_headers: dict[str, str] | None = None,
                       require_key: bool = True) -> None:
    """Register a built-in convenience source (local/together/openrouter) into
    st.SOURCES/st.SOURCE_ORDER, unless the user already defined one of that
    name via NX_SOURCE_<NAME>_* (an explicit config always wins) or, for
    multi-model hosts, no API key is available (`require_key=True`, the
    default — the legacy `local` fallback passes require_key=False since it
    has no key to gate on)."""
    if name in st.SOURCES:
        return
    if require_key and not api_key:
        return
    st.SOURCES[name] = Source(
        name, base_url, api_key, model,
        extra_body=extra_body, http_headers=http_headers)
    st.SOURCE_ORDER.append(name)


def switch_source(name: str, model_override: str | None = None) -> bool:
    """Swap the active source. Reassigns st.client + st.MODEL globals. Returns True
    on success, False (with a message) if the name is unknown.

    `model_override` (optional) pins st.MODEL to a specific model id for this
    switch instead of resolving the source's default — used by
    `/source <name> <model>` so a multi-model host (e.g. Together) can be
    pointed at any of its models without a config edit. A bare switch (no
    override) always falls back to the source's configured/default model,
    so `/source together` returns to GLM-5.2 even after an override."""
    src = st.SOURCES.get(name)
    if not src:
        print(f"{C.ERROR}  ✗ unknown source {name!r}{RESET}")
        return False
    st.ACTIVE = src
    st.client = src.client
    st.MODEL = model_override if model_override else src.resolve_model()
    # A model change (different source, or a per-switch model override on a
    # multi-model host like Together) means the cached max-context may no longer
    # apply — drop it so the footer re-probes against the new model next turn.
    src._context_window = None
    src._context_window_probe_attempted = False
    # Different endpoint/model may not confirm native tool-calling the same
    # way — let _grammar_extra_body() re-derive it from this source's actual
    # behavior instead of carrying over the previous source's confirmation.
    st._NATIVE_TOOL_CALLS_CONFIRMED = False
    # A model override (e.g. /source together <other-model>) may point the
    # same Source at a different model with different behavior — drop the
    # cached capability so it's re-probed; a bare switch back to an
    # already-probed source's default model keeps its prior answer.
    if model_override:
        src._capability = None
    from ._capability import ensure_capability_probe
    ensure_capability_probe(src)
    return True


def restore_source(source_name: str | None, model: str | None = None) -> bool:
    """Best-effort restore of the source (and optional model) a session was
    started on, used when resuming. Switches source if it's configured and
    differs from the active one; if `model` is set and not equal to the
    source's configured default, pins st.MODEL to it so a `/source <name>
    <model>` override survives a resume. Returns True if the source is now
    the requested one."""
    if not source_name or source_name not in st.SOURCES:
        return False
    src = st.SOURCES[source_name]
    pin = model if model and model != src.model else None
    if st.ACTIVE is not None and source_name == st.ACTIVE.name and st.MODEL == (pin or src.model):
        return True
    switch_source(source_name, model_override=pin)
    return True


def _set_provider_order(src: Source, order: list[str],
                         allow_fallbacks: bool | None = None) -> None:
    """Set the OpenRouter `provider.order` routing for `src` in place.

    `order` is a list of provider names (e.g. ["parasail/fp8", "Together"]); a
    deliberate, non-empty list pins routing to exactly those providers and —
    unless `allow_fallbacks` is explicitly True — disables OpenRouter's usual
    fallback to other upstreams, so you know precisely what's serving the model.
    Pass `allow_fallbacks=True` to keep falling through the rest of OpenRouter's
    pool after the ordered list is exhausted. An empty `order` clears routing
    entirely (sets src.extra_body to None), letting OpenRouter pick freely.

    Only mutates `provider`-style routing; any other keys already in
    `extra_body` are preserved. Non-routing sources (local llama.cpp, Together,
    …) accept the field harmlessly — they ignore unknown request-body keys — so
    this is safe to call on any source, though it only meaningfully routes on
    OpenRouter."""
    if order:
        body = dict(src.extra_body or {})
        prov = dict(body.get("provider") or {})
        prov["order"] = list(order)
        if allow_fallbacks is not None:
            prov["allow_fallbacks"] = allow_fallbacks
        else:
            # Pinning an explicit order means "I chose these specifically" →
            # don't silently fall through to other providers. The caller can
            # override by passing allow_fallbacks=True.
            prov["allow_fallbacks"] = False
        body["provider"] = prov
        src.extra_body = body
    else:
        src.extra_body = None


def _provider_order(src: Source) -> list[str]:
    """The current `provider.order` list on `src`, or [] when unset. Read-only
    accessor for the /provider listing; never raises."""
    body = src.extra_body if isinstance(src.extra_body, dict) else None
    if not body:
        return []
    prov = body.get("provider")
    if not isinstance(prov, dict):
        return []
    order = prov.get("order")
    return [str(x) for x in order] if isinstance(order, list) else []

