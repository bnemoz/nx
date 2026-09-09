"""Startup capability probe: fingerprint a Source's actual tool-calling
protocol instead of leaving the choice to reactive per-turn recovery or an
env-var opt-in (NX_GRAMMAR).

Fires a couple of small, cheap, non-streaming requests against a trivial
one-tool schema at startup and classifies the result into one of:

  CAPABILITY_NATIVE  — the server returns structured tool_calls with valid
                       JSON arguments. Nothing else needed; llama.cpp (and
                       any other native-tool-call server) already constrains
                       argument generation itself once it recognizes the
                       chat template as tool-call-capable.
  CAPABILITY_GRAMMAR — no native tool_calls, but a GBNF-constrained
                       <tool_call> tag (see _grammar.py) comes back well-formed
                       in the XML dialect nx teaches (SYSTEM + _grammar.py).
                       Worth attaching build_tool_call_grammar() on every turn.
  CAPABILITY_TEXT    — neither works (or the probe itself failed/timed out);
                       fall back to plain-text <tool_call> tag parsing
                       (_text_fallback.py) with no grammar constraint.

The result caches on the Source instance (src._capability), mirroring
resolve_context_window()'s cache-on-Source pattern in _sources.py — reset on
every switch_source() since a different endpoint/model may behave
differently. Never raises: any probe failure degrades to CAPABILITY_TEXT
after the delay budget, matching the "degrade, don't demand" convention.
"""
from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from ._grammar import build_tool_call_grammar
from ._text_fallback import parse_text_calls_with_errors

if TYPE_CHECKING:
    from ._sources import Source

CAPABILITY_NATIVE = "native"
CAPABILITY_GRAMMAR = "grammar"
CAPABILITY_TEXT = "text"

# A minimal, throwaway tool schema — deliberately not one of nx's real TOOLS,
# so the probe never touches the filesystem/shell and its result can't be
# confused with a real tool call surfacing in the session transcript.
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "nx_probe",
        "description": "Call this with echo set to the string \"ok\".",
        "parameters": {
            "type": "object",
            "properties": {"echo": {"type": "string"}},
            "required": ["echo"],
        },
    },
}
_PROBE_MESSAGES = [
    {"role": "user", "content": "Call the nx_probe tool now with echo set to \"ok\". "
                                "Respond with nothing else."},
]
_PROBE_GRAMMAR, _PROBE_TRIGGER = build_tool_call_grammar([_PROBE_TOOL])


def _valid_probe_tool_call(name: str | None, raw_args: str | None) -> bool:
    if name != "nx_probe":
        return False
    try:
        args = json.loads(raw_args or "{}")
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(args, dict) and "echo" in args


def _probe_native(src: "Source", mid: str) -> bool:
    """One non-streaming request with tools=[_PROBE_TOOL]. Returns True if the
    server answers with a structurally valid native tool_calls entry."""
    try:
        resp = src.client.chat.completions.create(
            model=mid, messages=_PROBE_MESSAGES, tools=[_PROBE_TOOL],
            stream=False, max_tokens=64, timeout=20)
    except Exception:
        return False
    from ._metrics import _record_api_call
    _record_api_call(resp, fallback_chars=64, fallback_prompt_chars=64)
    try:
        msg = resp.choices[0].message
    except (AttributeError, IndexError):
        return False
    tcs = getattr(msg, "tool_calls", None) or []
    for tc in tcs:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        if _valid_probe_tool_call(getattr(fn, "name", None), getattr(fn, "arguments", None)):
            return True
    return False


# P3-4 (0.50.0): the probe validates with the REAL parser rather than a
# probe-only regex. The grammar constrains the dialect SYSTEM teaches
# (<name>..</name><arguments><k>v</k>..</arguments>), so "the grammar works
# here" means exactly "parse_text_calls_with_errors() reads back a real
# nx_probe call from the constrained output" -- a bespoke pattern here would
# drift from the parser and could classify a source GRAMMAR-capable on a
# shape a turn would then fail to parse (which is how this probe worked
# before: it accepted JSON-in-tag, the grammar now refuses to emit).


def _probe_grammar(src: "Source", mid: str) -> bool:
    """One non-streaming request with the GBNF grammar for _PROBE_TOOL
    attached via extra_body. Returns True if the reply parses, with the
    real parser, as a well-formed call in the dialect SYSTEM teaches."""
    if _PROBE_GRAMMAR is None:
        return False
    extra_body = {
        "grammar": _PROBE_GRAMMAR,
        "grammar_triggers": [{"type": "word", "value": _PROBE_TRIGGER}],
    }
    try:
        resp = src.client.chat.completions.create(
            model=mid, messages=_PROBE_MESSAGES, stream=False, max_tokens=128,
            timeout=20, extra_body=extra_body)
    except Exception:
        return False
    from ._metrics import _record_api_call
    _record_api_call(resp, fallback_chars=128, fallback_prompt_chars=64)
    try:
        text = resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return False
    calls, _errors = parse_text_calls_with_errors(text)
    if not calls:
        return False
    name, args = calls[0]
    return _valid_probe_tool_call(name, json.dumps(args if isinstance(args, dict)
                                            else {"echo": args}))


def probe_capability(src: "Source", mid: str) -> str:
    """Classify src's tool-calling protocol for model `mid`. Never raises;
    worst case (both probes fail/error) returns CAPABILITY_TEXT."""
    if _probe_native(src, mid):
        return CAPABILITY_NATIVE
    if _probe_grammar(src, mid):
        return CAPABILITY_GRAMMAR
    return CAPABILITY_TEXT


_PROBE_STARTED: set[int] = set()  # id(Source) currently being probed (avoid dupe threads)

# Shared with _turn.py's context-window probe: a local, single-slot inference
# server serializes requests FIFO, so firing the capability probe and the
# ctx-window probe as two independent concurrent threads means both requests
# can land in the server's queue ahead of the user's actual first prompt,
# needlessly delaying the real response behind two throwaway probe calls.
# Serializing the two background probes on this lock caps that to one probe
# request in flight at a time instead of two.
STARTUP_PROBE_LOCK = threading.Lock()


def ensure_capability_probe(src: "Source | None") -> None:
    """Start a background thread to classify `src`'s tool-calling protocol if
    it isn't already known and no probe is in flight for it. The result lands
    in src._capability; callers (_grammar_extra_body, the native/text branch
    in model_turn) fall back to the existing reactive behavior while it's
    still None. Idempotent via _PROBE_STARTED (cleared when the thread
    finishes)."""
    if src is None or src._capability is not None:
        return
    sid = id(src)
    if sid in _PROBE_STARTED:
        return
    _PROBE_STARTED.add(sid)

    def _work():
        try:
            if src._capability is not None:
                return  # resolved by another path (e.g. a /source switch)
            with STARTUP_PROBE_LOCK:
                if src._capability is not None:
                    return
                mid = src.model or src.resolve_model()
                src._capability = probe_capability(src, mid)
        except Exception:
            pass  # never raise from a background probe; stays None, re-armed later
        finally:
            _PROBE_STARTED.discard(sid)

    t = threading.Thread(target=_work, daemon=True, name="nx-capability-probe")
    t.start()
