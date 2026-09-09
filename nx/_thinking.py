"""Separate-thinking auto-configuration.

Some OpenAI-compatible servers run a hybrid reasoning model with its thinking
phase enabled but *not* surfaced as a distinct field: the private reasoning
streams as plain `content` and runs straight into the visible answer with no
separator at all. Observed with glm-5.3-flash behind a vLLM-backed proxy:

    {"delta": {"content": "Let me analyze this greeting … maintaining
    appropriate boundaries for a professional interaction.Hello! I'm GLM,
    trained by Z.ai. …"}}

There is nothing to parse client-side — the reasoning and the answer arrive as
one undifferentiated string, so nx's reasoning-block rendering (and the
reasoning-only stall guards, which key off `reasoning_content`) never fire.

The fix is server-side: vLLM-style servers accept `chat_template_kwargs`
(literally `extra_body={"chat_template_kwargs": {...}}` — top-level body keys
the OpenAI SDK does not model) to flip the template's thinking toggle, which
makes the server emit `reasoning_content` as its own delta field. The exact
toggle name is template-dependent: GLM's template calls it `thinking`, Qwen's
`enable_thinking`.

Rather than hardcoding model-name guesses, we PROBE once per (source, model):
fire a tiny non-streaming request with each candidate toggle and check whether
`reasoning_content` comes back separated. The probe result is cached on the
Source keyed by model id, so a thinking-capable model gets the kwarg attached
to every subsequent request (via Source.extra_request_kwargs) and everything
else is untouched. Safety properties:

  • A server that already separates reasoning (glm-5.3, DeepSeek-R1, …) never
    gets any kwarg attached — the baseline probe confirms separation, done.
  • A server that rejects the extra body field (strict OpenAI-compat APIs)
    answers the probe with an error → no kwarg attached, behavior unchanged.
  • A non-thinking model never gets the kwarg attached: forcing its template's
    thinking toggle yields no reasoning_content, so the probe records nothing.

NX_THINKING=off disables the probe and the attach entirely (e.g. for a
benchmark where you want the server's stock behavior, merged or not).
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from . import _state as st
from ._openai_client import probe_scope

if TYPE_CHECKING:
    from ._sources import Source

# Candidate body fields, tried in order. Each is one probe request, so keep
# this list short. All are merged as extra_body top-level keys — servers that
# don't understand them reject or ignore the request, both of which the probe
# treats as "no".
_THINKING_KWARG_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"chat_template_kwargs": {"thinking": True}},
    {"chat_template_kwargs": {"enable_thinking": True}},
)

_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
_PROBE_MAX_TOKENS = 48


def _probe_kwargs_for(src: "Source", mid: str) -> dict[str, Any] | None:
    """Return the body kwargs that make `mid` separate its reasoning, or None.

    Never raises. A baseline probe first: if the server already emits
    reasoning_content unprompted, nothing needs attaching. Otherwise each
    candidate toggle is tried until one produces a separated reasoning delta.
    """
    def _has_reasoning(body: dict[str, Any]) -> bool:
        try:
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return False
        return bool(msg.get("reasoning_content") or msg.get("reasoning"))

    def _create(extra_body: dict[str, Any] | None):
        resp = src.client.chat.completions.create(
            model=mid, messages=_PROBE_MESSAGES, stream=False,
            max_tokens=_PROBE_MAX_TOKENS, timeout=20, extra_body=extra_body)
        from ._metrics import _record_api_call
        _record_api_call(resp, fallback_chars=_PROBE_MAX_TOKENS,
                         fallback_prompt_chars=64)
        return resp

    try:
        # Baseline: does this model already separate its reasoning?
        resp = _create(None)
        if _has_reasoning(resp.model_dump()):
            return None  # already separated — attaching anything is noise
    except Exception:
        pass  # unreachable/erroring server: fall through and try the toggles
    for kwarg in _THINKING_KWARG_CANDIDATES:
        try:
            resp = _create(kwarg)
        except Exception:
            continue  # server rejects the field (strict OpenAI-compat) → skip
        if _has_reasoning(resp.model_dump()):
            return kwarg
    return None


# (id(Source), model id) -> PROBE_GENERATION the probe was armed at —
# model-keyed, not just Source-keyed, so a startup probe for the source's
# default model doesn't silently swallow a /source <name> <model>-override
# probe requested a moment later while the first is still running. Gen-valued
# so a switch past a still-backing-off probe can arm a fresh one immediately
# (see _capability._PROBE_STARTED).
_THINKING_STARTED: dict[tuple[int, str], int] = {}


def thinking_kwargs(src: "Source", mid: str) -> dict[str, Any] | None:
    """The kwargs recorded for (src, mid), or None if not yet probed. Read-only;
    used by Source.extra_request_kwargs(), which must stay synchronous."""
    cache = src._thinking_kwargs  # dict: model id → kwargs | None
    return cache.get(mid)


def ensure_thinking_probe(src: "Source | None", mid: str | None = None) -> None:
    """Background-probe `src` (for model `mid`, defaulting to the currently
    active st.MODEL) unless NX_THINKING=off. Result lands in
    src._thinking_kwargs[mid]; extra_request_kwargs() picks it up on the next
    request after that. Idempotent per (Source, model) while a probe is in
    flight."""
    if src is None or st.THINKING_MODE == "off":
        return
    model_id = mid or st.MODEL or src.model
    if not model_id:
        return  # nothing sensible to probe yet (no active model id)
    if model_id in src._thinking_kwargs:
        return  # already resolved by an earlier probe
    key = (id(src), model_id)
    gen = st.PROBE_GENERATION
    if _THINKING_STARTED.get(key) == gen:
        return
    _THINKING_STARTED[key] = gen

    def _work():
        try:
            # Share the startup-probe lock with the capability/context probes:
            # a single-slot local server serves requests FIFO, so unserialized
            # background probes would queue ahead of the user's first prompt.
            from ._capability import STARTUP_PROBE_LOCK
            with STARTUP_PROBE_LOCK:
                if model_id in src._thinking_kwargs:
                    return
                with probe_scope(lambda: st.PROBE_GENERATION != gen):
                    src._thinking_kwargs[model_id] = _probe_kwargs_for(src, model_id)
        except Exception:
            pass  # never raise from a background probe; stays unprobed, re-armed
        finally:
            if _THINKING_STARTED.get(key) == gen:
                _THINKING_STARTED.pop(key, None)

    t = threading.Thread(target=_work, daemon=True, name="nx-thinking-probe")
    t.start()
