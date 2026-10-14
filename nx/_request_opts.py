"""Request shaping: the recovery sampler, grammar-constrained tool calls,
the /toolcalls mode override, and the max_tokens clamp.

Split out of `_compress.py` (which keeps open_stream — the one funnel that
consumes these — and the compression machinery). Every model call rides the
options this module derives: the loop-breaking sampler (NX_RECOVERY_*), the
`<tool_call>`-tag grammar (NX_GRAMMAR + the capability probe), the
tool_choice posture (NX_NATIVE_TOOLS, /toolcalls), and the context-window
aware max_tokens clamp (NX_MAX_TOKENS)."""
from __future__ import annotations

from typing import Any

from ._env import _env_int
from ._grammar import build_tool_call_grammar
from . import _state as st

MAX_COMPLETION_TOKENS = _env_int("NX_MAX_TOKENS", 16000)


def _prompt_estimate(messages: list[dict[str, Any]],
                     active_source: Any = None) -> int:
    """Best-effort prompt size in tokens for the max_tokens clamp: the exact
    server /tokenize count when the source is local and has one (never a
    remote round-trip), else the chars/4 estimate — the same estimator /ctx
    reports with, plus the per-turn tools-schema overhead it adds.

    `active_source` is the source the CALL will ride (0.156.0): a routed target
    (batch task, sub-agent) must be estimated against ITS source, not the
    process default. None → ``st.ACTIVE``, i.e. the pre-0.156.0 behavior."""
    try:
        from ._ctxreport import _message_tokens, _tokenize_total, _turn_overhead_est
        src = active_source if active_source is not None else st.ACTIVE
        if st.TOKENIZE_PREFLIGHT and src is not None:
            n = _tokenize_total(messages)
            if isinstance(n, int) and n > 0:
                return n
        body = sum(_message_tokens(m) for m in messages)
        return body + (_turn_overhead_est() if src is not None else 0)
    except Exception:  # noqa: BLE001 — an estimate must never break the request
        return 0


def _clamp_max_tokens(messages: list[dict[str, Any]], requested: int,
                      active_source: Any = None) -> int:
    """Clamp `requested` to what the source's context window can actually
    answer: window - prompt_estimate - margin, never above `requested`.

    The window comes from the source's startup probe (resolve_context_window);
    when it is unknown — remote aggregator, probes skipped — the clamp is a
    no-op and the request is exactly what it was before 0.41. The floor keeps
    a degenerate (room <= 0) request well-formed: an 8k window with a 7.9k
    prompt still asks for a minimum-sized answer and lets the overflow-retry
    machinery handle the refusal, instead of sending max_tokens=0.

    `active_source` (0.156.0) is the source the CALL will ride: a routed target
    must be clamped by ITS OWN window, or a task/sub-agent on a smaller endpoint
    is clamped by the run's — the `_request_opts` wart. None (the default) means
    ``st.ACTIVE``, so unrouted callers clamp exactly as before."""
    if requested <= 0:
        return requested  # 0 = "omit the field" contract in open_stream
    src = active_source if active_source is not None else st.ACTIVE
    if src is None:
        return requested
    window = None
    try:
        window = src.resolve_context_window()
    except Exception:  # noqa: BLE001 — probe failures must not break requests
        window = None
    if not isinstance(window, int) or window <= 0:
        return requested
    room = window - _prompt_estimate(messages, active_source) - st.TOKENS_MARGIN
    floor = max(1, st.MIN_COMPLETION_TOKENS)
    return max(floor, min(requested, room))


# tool-name tuple -> extra_body dict, built lazily/once per tool set. Bounded
# (0.108.4): the key is every tool name in the request, so a swarm churning
# sub-agent tool sets could grow it without limit — cap at 128, drop-oldest.
_GRAMMAR_CACHE_CAP = 128
_grammar_cache: dict[tuple[str | None, ...], Any] = {}

# 0.30.0: /toolcalls text|native|auto — the escalation ladder's rung PAST the
# breaker. Once native tool-calling is confirmed, _grammar_extra_body() stops
# attaching (above), so an endpoint whose NATIVE args keep truncating has no
# mid-session route back to grammar-constrained text protocol; this override
# ("text") un-confirms the flag for the session so the text-protocol ladder
# (grammar + client-side repair) takes over again. "auto" clears the override
# (the reactive default); "native" re-arms the flag so native is used
# immediately without waiting for a confirming turn. None = no override.
_TOOLCALLS_MODE_OVERRIDE: str | None = None


def toolcalls_mode() -> str:
    """The effective tool-calling mode: the /toolcalls override when set,
    else the reactive default ('auto')."""
    return _TOOLCALLS_MODE_OVERRIDE or "auto"


def set_toolcalls_mode(mode: str | None) -> bool:
    """Apply a /toolcalls override; returns False for an unknown mode.
    None (or 'auto') clears the override and re-derives the flag reactively."""
    global _TOOLCALLS_MODE_OVERRIDE
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "text", "native"):
        return False
    _TOOLCALLS_MODE_OVERRIDE = None if mode == "auto" else mode
    if mode == "text":
        # Un-confirm native so _grammar_extra_body() re-attaches and
        # tool_choice goes back to "none" (the text-protocol posture).
        st._NATIVE_TOOL_CALLS_CONFIRMED = False
    elif mode == "native":
        st._NATIVE_TOOL_CALLS_CONFIRMED = True
    return True


def _grammar_extra_body(tools: list[dict[str, Any]] | None,
                        active_source: Any = None) -> dict[str, Any] | None:
    """Grammar extra_body for this tool set, or None when the source must send
    its tool_calls natively.

    `active_source` (0.156.0) is the source the CALL will ride, so a routed
    target's CAPABILITY probe verdict (the per-Source `_capability` cache) is
    what decides the attach — previously a task/sub-agent on another endpoint
    was grammar-checked by the run's source. None → ``st.ACTIVE``, unchanged."""
    if not tools:
        return None
    # /toolcalls text override (0.30.0): the user forced the text protocol for
    # this session — the grammar attaches even past the two vetoes below
    # (native-confirmed, or a CAPABILITY_NATIVE probe verdict), because that
    # rung exists exactly for endpoints whose NATIVE args keep truncating.
    if toolcalls_mode() != "text":
        if st._NATIVE_TOOL_CALLS_CONFIRMED:
            return None
        # The startup capability probe (nx/_capability.py) takes precedence over
        # the NX_GRAMMAR opt-in once it lands: CAPABILITY_NATIVE means llama.cpp
        # is already constraining tool_calls itself (skip, same as the reactive
        # _NATIVE_TOOL_CALLS_CONFIRMED check above); CAPABILITY_GRAMMAR means the
        # grammar demonstrably helps on this source, so attach it even without
        # NX_GRAMMAR set; CAPABILITY_TEXT means it didn't help in the probe, so
        # skip it (plain text-protocol parsing only). While the probe is still
        # pending (None), fall back to the old NX_GRAMMAR env-var gate.
        src = active_source if active_source is not None else st.ACTIVE
        capability = getattr(src, "_capability", None) if src else None
        from . import _compress as _c  # lazy: _compress imports this module
        if capability is not None:
            from ._capability import CAPABILITY_GRAMMAR
            if capability != CAPABILITY_GRAMMAR:
                return None
        elif not _c.NX_GRAMMAR:
            return None
    key = tuple(t.get("function", {}).get("name") for t in tools)
    if key not in _grammar_cache:
        if len(_grammar_cache) >= _GRAMMAR_CACHE_CAP:
            _grammar_cache.pop(next(iter(_grammar_cache)))
        grammar, trigger = build_tool_call_grammar(tools)
        _grammar_cache[key] = None if grammar is None else {
            "grammar": grammar,
            "grammar_triggers": [{"type": "word", "value": trigger}],
        }
    return _grammar_cache[key]


def _recovery_sampling_opts() -> dict[str, Any]:
    """Loop-breaking sampler for recovery retries (see RECOVERY_* constants).

    temperature/top_p are standard OpenAI params and go top-level. min_p,
    repeat_penalty, repeat_last_n and the DRY family are llama.cpp extensions the
    OpenAI SDK won't accept as kwargs, so they ride in extra_body — llama.cpp's
    server merges that into the request body; non-llama.cpp endpoints ignore
    unknown keys. Set any RECOVERY_* value negative to omit it.  When
    NX_BACKEND=vllm the llama.cpp-only knobs are set to None instead of
    their defaults, so the checks below miss them automatically."""
    opts: dict[str, Any] = {}
    if st.RECOVERY_TEMPERATURE is not None and st.RECOVERY_TEMPERATURE >= 0:
        opts["temperature"] = st.RECOVERY_TEMPERATURE
    if st.RECOVERY_TOP_P is not None and st.RECOVERY_TOP_P >= 0:
        opts["top_p"] = st.RECOVERY_TOP_P
    extra: dict[str, Any] = {}
    # RECOVERY_* constants live on _compress (recomputed by its reload when
    # NX_BACKEND flips); read them at call time so a mid-session backend
    # change and the tests' module reload both take effect.
    from . import _compress as _c  # lazy: _compress imports this module
    if _c.RECOVERY_MIN_P is not None and _c.RECOVERY_MIN_P >= 0:
        extra["min_p"] = _c.RECOVERY_MIN_P
    if _c.RECOVERY_REPEAT_PENALTY is not None and _c.RECOVERY_REPEAT_PENALTY >= 0:
        extra["repeat_penalty"] = _c.RECOVERY_REPEAT_PENALTY
        if _c.RECOVERY_REPEAT_LAST_N is not None:
            extra["repeat_last_n"] = _c.RECOVERY_REPEAT_LAST_N
    if _c.RECOVERY_DRY_MULTIPLIER is not None and _c.RECOVERY_DRY_MULTIPLIER > 0:
        extra["dry_multiplier"] = _c.RECOVERY_DRY_MULTIPLIER
        if _c.RECOVERY_DRY_BASE is not None:
            extra["dry_base"] = _c.RECOVERY_DRY_BASE
        if _c.RECOVERY_DRY_ALLOWED_LENGTH is not None:
            extra["dry_allowed_length"] = _c.RECOVERY_DRY_ALLOWED_LENGTH
        if _c.RECOVERY_DRY_SEQUENCE_BREAKERS is not None:
            extra["dry_sequence_breakers"] = _c.RECOVERY_DRY_SEQUENCE_BREAKERS
    if extra:
        opts["extra_body"] = extra
    return opts


def _parallel_tool_calls_opts(tool_choice: Any, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """0.103.2: {"parallel_tool_calls": True} when the request actually rides
    native tool_calls — tools attached, tool_choice == "auto", and the source
    already CONFIRMED native (_NATIVE_TOOL_CALLS_CONFIRMED, the same flag the
    grammar attach and /toolcalls read). Until native is confirmed the source
    may be a text-protocol model and the field is noise; the flag is set
    reactively the first time a turn returns real tool_calls, so this attach
    is exactly as conservative as the tool_choice="auto" posture itself.
    Gated by NX_PARALLEL_TOOL_CALLS (default on; 0 disables — defined in
    _compress and read lazily here so a mid-session env change and the
    tests' module reload both take effect)."""
    from . import _compress as _c  # lazy: _compress imports this module
    if not getattr(_c, "NX_PARALLEL_TOOL_CALLS", False):
        return {}
    if not tools or tool_choice != "auto":
        return {}
    if not st._NATIVE_TOOL_CALLS_CONFIRMED:
        return {}
    return {"parallel_tool_calls": True}


def _default_tool_choice() -> str:
    """The tool_choice open_stream() applies when the caller didn't force one:
    "auto" unless (a) the session explicitly forced the text protocol
    (/toolcalls text — the rung past the malformed-args breaker) or (b)
    NX_NATIVE_TOOLS=0 opted the source out of native tool_calls entirely.
    A stall/empty-escalated retry (st._STALL_FORCE_NATIVE, set by
    _turn._maybe_arm_native_escalation) overrides (b): the stall is evidence
    the opt-out took away the model's only action channel. See open_stream's
    long comment for why "auto" is the default."""
    from . import _compress as _c  # lazy: _compress imports this module
    if toolcalls_mode() == "text":
        return "none"
    if st._STALL_FORCE_NATIVE:
        return "auto"
    if _c.NX_NATIVE_TOOLS:
        return "none"
    return "auto"
