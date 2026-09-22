"""Streaming completion requests and context compression.

open_stream() is the one place every model call funnels through (normal
turns and the loop-breaking "recovery" retries alike). compress() (manual
/compress) and _maybe_autocompress() (automatic, triggered by
NX_AUTOCOMPRESS_PERCENT) fold older turns of `messages` into a single
summary turn, keeping the system prompt and the most recent turns verbatim
— freeing context without losing the model's grip on what it was just
doing. Unlike manual /compress (which keeps only COMPRESS_KEEP=2 turns),
auto-compress is conservative: it keeps roughly the last third of the
conversation so in-progress work isn't disturbed.

Split (0.88.1): the runtime-note contract lives in `nx/_runtime_notes.py`
(nudges, note flushing, keep-window sizing), request shaping in
`nx/_request_opts.py` (recovery sampler, grammar, /toolcalls mode, the
max_tokens clamp), and error classification in `nx/_stream_errors.py`.
Names moved there are re-exported below so the existing import surface
(`nx/__init__.py`, `_turn.py`, tests patching `nx._compress.*`) keeps
working unchanged."""
from __future__ import annotations

import os
import re
from typing import Any

import httpx
from ._openai_client import APIConnectionError

from ._ansi import C, DIM, RESET
from ._env import _env_int, _env_float
from ._logging import _log_event, _llog
from ._sessions_io import _prune_empty_assistant_messages, _drop_leading_dangling_tool_turns
from ._ctxreport import _pinned_indexes
from ._todo import _latest_todo_turn
from ._plan import _latest_plan_turn
from ._metrics import _abbr, _record_usage
from . import _state as st
from . import TOOLS

# --- re-exports from the split-out layers (single import surface preserved) ---
from ._runtime_notes import (  # noqa: F401 (re-exports)
    RUNTIME_NOTE_PREFIX, _flush_pending_background_notice,
    _is_runtime_note, _keep_past_notes, _last_is_dangling_tool,
    _nudge_current_user_turn,
)
from ._request_opts import (  # noqa: F401 (re-exports)
    _clamp_max_tokens, _default_tool_choice, _grammar_extra_body,
    _prompt_estimate, _recovery_sampling_opts, set_toolcalls_mode,
    toolcalls_mode,
)
from ._stream_errors import (  # noqa: F401 (re-exports)
    _CTX_OVERFLOW_RE, _is_context_overflow_error, _is_request_field_rejection,
)

# --- env-derived request flags (owned here: tests reload this module and
# patch these names — _request_opts reads them lazily at call time) ----------
MAX_COMPLETION_TOKENS = _env_int("NX_MAX_TOKENS", 16000)

# Backend hint. Set NX_BACKEND=vllm to disable llama.cpp-only recovery
# knobs (min_p, repeat_penalty, DRY) that vLLM's speculative decoder rejects.
RECOVERY_BACKEND = os.environ.get("NX_BACKEND", "").strip().lower()

# Recovery sampler. A gibberish/repetition collapse is a LOW-ENTROPY attractor:
# the model is copying long n-grams already in context (paths, flags). Escaping
# it needs MORE entropy (raise temperature, lower min_p) plus anti-repetition
# penalties that lower the looping token's logit — the OPPOSITE of sharpening
# toward greedy. So recovery now RAISES temperature (was 0.2, which made the
# loop more deterministic) and adds DRY + repeat_penalty. The non-OpenAI knobs
# ride in extra_body (llama.cpp reads them; other backends ignore unknown keys).
st.RECOVERY_TEMPERATURE = _env_float("NX_RECOVERY_TEMPERATURE", 1.0)
st.RECOVERY_TOP_P = _env_float("NX_RECOVERY_TOP_P", 0.95)
RECOVERY_MIN_P = None if RECOVERY_BACKEND == "vllm" else _env_float("NX_RECOVERY_MIN_P", 0.02)
RECOVERY_REPEAT_PENALTY = None if RECOVERY_BACKEND == "vllm" else _env_float("NX_RECOVERY_REPEAT_PENALTY", 1.2)
RECOVERY_REPEAT_LAST_N = None if RECOVERY_BACKEND == "vllm" else _env_int("NX_RECOVERY_REPEAT_LAST_N", 512)
RECOVERY_DRY_MULTIPLIER = None if RECOVERY_BACKEND == "vllm" else _env_float("NX_RECOVERY_DRY_MULTIPLIER", 0.8)
RECOVERY_DRY_BASE = None if RECOVERY_BACKEND == "vllm" else _env_float("NX_RECOVERY_DRY_BASE", 1.75)
RECOVERY_DRY_ALLOWED_LENGTH = None if RECOVERY_BACKEND == "vllm" else _env_int("NX_RECOVERY_DRY_ALLOWED_LENGTH", 2)
# Treat path/code punctuation as DRY sequence breakers so a long file path the
# agent must emit verbatim is never penalized as "repetition".
RECOVERY_DRY_SEQUENCE_BREAKERS = None if RECOVERY_BACKEND == "vllm" else ["\n", ":", "\"", "*", "/", "\\", "`", "'"]

# Grammar-constrained tool calls: opt-in (NX_GRAMMAR=1) since the request
# fields involved (llama.cpp's `grammar` + `grammar_triggers`) are a
# llama.cpp server extension whose exact shape has shifted across versions,
# unlike the long-stable min_p/DRY knobs above. It's skipped for vLLM (which
# has its own, differently-shaped guided-decoding fields) — an unsupported
# server just gets the field ignored or the request rejected, in which case
# open_stream()'s existing no-tools/no-extra fallback ladder drops it.
#
# Meaningful only for the text-protocol fallback, NOT once native tool-calling
# is confirmed (st._NATIVE_TOOL_CALLS_CONFIRMED, set by
# _handle_native_tool_calls() in _turn.py the first time a turn actually comes
# back with structured tool_calls). llama.cpp already builds its own
# argument-constraining grammar internally from `tools=` for chat templates it
# recognizes as native-tool-call-capable, with no separate client-observable
# trigger to hook a second grammar off of — sending our own `grammar` field
# on top of that would either be redundant or override/fight llama.cpp's own
# native grammar (some server versions reject `tools` + explicit `grammar`
# together outright). So once a session is confirmed native, we stop
# attaching ours and just let llama.cpp's built-in constraint do the job;
# right up until that first confirmation (and for servers/templates that
# never go native, i.e. permanently text-protocol), our <tool_call>-tag
# grammar is the only thing constraining malformed JSON, so it stays on.
# NX_GRAMMAR is now only a fallback gate: once the startup capability probe
# (nx/_capability.py, run at each switch_source()) lands on a source, its
# verdict (CAPABILITY_NATIVE/GRAMMAR/TEXT) decides whether the grammar gets
# attached, regardless of this env var — see _grammar_extra_body() below.
NX_GRAMMAR = RECOVERY_BACKEND != "vllm" and os.environ.get("NX_GRAMMAR", "") not in ("", "0")

# Native tool_calls (server-side JSON-args generation): since 0.54.0 the
# default posture is tool_choice="auto" (see open_stream) — a server MAY hand
# back structured tool_calls. Before that the default was "none" because
# several servers/models (notably glm) advertise structured tool_calls
# support but routinely emit malformed JSON in the `arguments` field with no
# repair path available client-side (unlike our own <tool_call> tag body,
# which we parse ourselves and can patch up). The "none" posture turned out
# to be the worse failure: on hybrid reasoning models behind vLLM it left the
# model with NO action path at all (see open_stream's comment), so "auto" won
# and malformed native args are handled by the recovery ladder instead
# (arg repair -> truncated-args salvage -> malformed-args breaker -> the
# /toolcalls text escape hatch).
#
# NX_NATIVE_TOOLS keeps its name for compatibility but flips meaning:
# unset / "1" / "auto" (the default) -> tool_choice="auto";
# "0" -> the old "none" posture for a source/model whose native args are
# unusable, without giving up the text-protocol fallback.
NX_NATIVE_TOOLS = os.environ.get("NX_NATIVE_TOOLS", "") not in ("", "0", "auto", "1")
# Renamed mirror with the post-0.54 semantics (True = tool_choice="auto" is
# the default posture). Kept as a separate name so the config viewer/doctor
# and the vision gate read intent, not a double negative.
NATIVE_TOOLS_DEFAULT = not NX_NATIVE_TOOLS


def _compress_for_overrun(messages: list[dict[str, Any]], tier: int = 0) -> bool:
    """Attempt to free context after a context-length rejection.

    Tier 0 (the first refusal): fold the older turns into a summary
    (auto=True keeps ~1/3 verbatim). Tier 1+: the kept tail itself is too big
    (a second refusal proves it) — halve the kept tail each rung (floored at
    OVERFLOW_FOLD_KEEP_FLOOR) and, from OVERFLOW_FOLD_TIERS-2 on, also spill-
    evict the largest tool results. Returns True iff anything changed
    (messages mutated) — the caller retries the open once per tier and falls
    back to the terminal-error path when the ladder is exhausted.
    """
    # Via the top-level package so a test stubbing nx.compress takes effect,
    # mirroring _maybe_autocompress below.
    import nx as _m
    # Image parts first (0.38.0): auto-compress keeps ~1/3 of turns verbatim,
    # so a read_image blob that CAUSED the overrun would survive the fold and
    # the retry would 400 again — the turn died with "could not free enough
    # room" and the only recovery was the user hand-pruning the session.
    # Swap every image_url part for its one-line text marker (in place, so the
    # live history — and any later save — loses the megabytes, not just the
    # retry), then fold. The marker keeps the model's memory of what it saw.
    from ._images import _strip_image_parts
    stripped = _strip_image_parts(messages)
    if tier <= 0:
        folded = _m.compress(messages, auto=True) is not None
    else:
        # The refusal came back AFTER a fold, so the conservative auto keep
        # (~1/3) is provably too much: halve it per tier. compress()'s auto
        # raise (`max(keep, body_len // 3)`) would defeat a small explicit
        # keep, which is why tiers pass keep= explicitly with auto=True.
        sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        body_len = len(messages) - (1 if sys_msg else 0)
        keep = max(st.OVERFLOW_FOLD_KEEP_FLOOR, body_len // (3 * (2 ** tier)))
        folded = _m.compress(messages, keep=keep, auto=True) is not None
    evicted = _evict_largest_tool_results(messages) if tier >= 2 else 0
    return folded or stripped > 0 or evicted > 0


def _evict_largest_tool_results(messages: list[dict[str, Any]]) -> int:
    """Spill the biggest tool results to disk (tier 0.41): replace-in-place
    with the on-disk stub, exactly like the arrival-time cap does, but driven
    by a context overflow instead of a single result's size. Overflow-tier
    thresholds; the fold path uses the parameterized _spill_tool_results.

    Since PF-14 the append-time floor (_text_fallback.APPEND_SPILL_MIN_CHARS,
    default 4000) already stubs results this large the moment they are
    produced — before they are ever sent — so in practice this now finds only
    content injected bypassing run_tool() and returns 0. It is kept as a
    safety net for that case, and because a mid-flight lowering of the floor
    should not silently drop the eviction path. It is the LAST-resort, cache-
    busting spill; prefer spilling at append time (never mutate sent bytes)."""
    return _spill_tool_results(messages, set(_pinned_indexes()),
                               st.OVERFLOW_EVICT_MIN_CHARS, st.OVERFLOW_EVICT_MAX)


def open_stream(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] = TOOLS,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
    recovery_sampling: bool = False,
):
    """Open a streaming completion. Retries without tools= if the server rejects
    that param; returns None (after a friendly message) on connection/API failure.
    A terminal failure also latches st._LAST_API_ERROR ("Type: msg") so
    model_turn() can distinguish a request that never happened from a
    legitimately empty turn — headless callers turn that into status "error"."""
    # _LoggingStream lives in _turn.py, which itself imports open_stream from
    # this module — a lazy import here (instead of at module scope) avoids a
    # circular import between the two.
    from ._turn import _LoggingStream
    # 0.86.0: sub-agent loops ride their own (client, model, extra) when
    # NX_SUBAGENT_SOURCE/_MODEL are set — resolved per-thread from the
    # agent's _SubagentContext, so concurrent agents never swap globals.
    # Everything downstream (window clamp, ladder pinning, the request)
    # uses this target instead of the parent's st.ACTIVE/st.MODEL.
    target = None
    from ._subagent import _subagent_ctx
    ctx = _subagent_ctx()
    if ctx is not None:
        target = ctx.model_target
    if target is not None and st.client is not None and st.MODEL is not None:
        call_client, call_model, call_extra, call_src = target
    else:
        call_client, call_model, call_extra, call_src = st.client, st.MODEL, None, st.ACTIVE
    assert call_client is not None and call_model is not None
    if tool_choice is None and tools:
        # Default posture (0.54.0): tool_choice="auto". The pre-0.54 default
        # was "none" — send tools= so the schema renders into the prompt, but
        # suppress server-side native tool_calls so the model could only act
        # through the text-protocol <tool_call> tag (which nx parses and can
        # repair itself; several models, notably glm, advertise structured
        # tool_calls but emit malformed JSON arguments nx can't patch up).
        # That posture had a fatal interaction with hybrid reasoning models
        # behind vLLM (glm-5.3, qwen3.8-flash-next): they only ever act via the
        # native tool_calls machinery, so "none" didn't fall back to the text
        # protocol — it produced reasoning-only turns with no action path, and
        # the stall ladder climbed to a forced final whose salvage printed the
        # model's reasoning as the answer. It also raced the startup capability
        # probe: the first prompt of a fresh session regularly beat the probe
        # and burned its whole stall ladder under the "none" posture.
        # "auto" permits native calls without forcing them — a model that
        # prefers the text protocol still emits text-protocol tags as content,
        # and nx parses/repairs those exactly as before. Malformed native args
        # keep their existing recovery (arg repair, salvage, breaker, and the
        # /toolcalls text escape hatch that pins "none" for the session).
        # A caller that forces tool_choice (the forced-final-answer path) is
        # left untouched.
        tool_choice = _default_tool_choice()
    try:
        _prune_empty_assistant_messages(messages)
        token_limit = MAX_COMPLETION_TOKENS if max_tokens is None else max_tokens
        token_limit = _clamp_max_tokens(messages, token_limit)
        request_opts: dict[str, Any] = {}
        if token_limit > 0:
            request_opts["max_tokens"] = token_limit
        if recovery_sampling:
            request_opts.update(_recovery_sampling_opts())
        elif st.TEMPERATURE is not None and st.TEMPERATURE >= 0:
            request_opts["temperature"] = st.TEMPERATURE
        elif st.TOP_P is not None and st.TOP_P >= 0:
            request_opts["top_p"] = st.TOP_P
        if tool_choice is not None:
            request_opts["tool_choice"] = tool_choice
        # Per-request body fields (e.g. OpenRouter provider routing) ride in
        # extra_body. Merged here once so both the tools and no-tools fallback
        # branches below carry it through unchanged.
        if call_extra:
            request_opts.update(call_extra)
        grammar_extra = _grammar_extra_body(tools)
        if grammar_extra:
            # Merge (not overwrite) — a source's own extra_body (e.g.
            # OpenRouter provider routing) must survive alongside grammar.
            request_opts.setdefault("extra_body", {}).update(grammar_extra)
        # PF-11 (0.59.0): start at the rung this (source, model) already told us
        # it accepts, instead of re-deriving the shape by failing every turn.
        # A pinned rung means one request where the old code paid up to three
        # full-prompt POSTs against an endpoint that had already rejected the
        # first one. pinned_index is a dict read; on a fresh source/model it is
        # 0 and this loop walks the ladder exactly as before, then remembers.
        from ._manifest import LADDER, RUNG_TAG, pin as _pin_shape, pinned_index
        rung = pinned_index(call_src, call_model)
        stream = None
        while True:
            tools_ok, extra_ok = LADDER[rung]
            # A rung that drops extra_body drops it wholesale (grammar, a
            # source's own body, the recovery samplers) — the pre-0.59.0 ladder
            # stripped the whole key too. A rung that drops tools drops
            # tool_choice with it (it is only meaningful alongside tools).
            opts = {k: v for k, v in request_opts.items()
                    if extra_ok or k != "extra_body"}
            if not tools_ok:
                opts.pop("tool_choice", None)
            _tag = RUNG_TAG[rung]
            _log_event("req", {"model": call_model, "messages": messages,
                               "tools": tools if tools_ok else None, "stream": True,
                               **({"_fallback": _tag} if _tag else {}), **opts})
            try:
                stream = call_client.chat.completions.create(
                    model=call_model, messages=messages,
                    tools=tools if tools_ok else None, stream=True,
                    stream_options={"include_usage": True}, **opts)
                break  # succeeded — remember the shape for the next turn
            except (APIConnectionError, httpx.HTTPError):
                raise  # transport failure — retrying without tools won't help
            except Exception as e:
                # Context-window overrun: the prompt + tools no longer fit.
                # Retrying without tools (or without extra_body) only shrinks
                # the request by a few KB of schema — it cannot fit an
                # over-length prompt, and it would burn seconds on fallback
                # attempts that must also 400. Signal the caller instead
                # (TURN_CONTEXT_OVERRUN → compress+retry).
                if _is_context_overflow_error(e):
                    st._LAST_CONTEXT_OVERRUN = str(e)
                    return None
                # Only a FIELD rejection (server parsed the request and objected
                # to tools=/tool_choice/extra_body) justifies re-sending.
                # Anything else — a 429/5xx APIError that _send already
                # exhausted NX_HTTP_RETRIES on, or an unexpected error shape —
                # propagates immediately: the fallback rungs are fresh
                # multi-attempt _send cycles, so re-sending here would amplify
                # one degraded turn into up to 9 POSTs against an endpoint that
                # is already failing for reasons no body-shedding can fix.
                # (Tradeoff: a server that signals field rejection only via 5xx
                # now fails fast instead of being rescued.)
                if not _is_request_field_rejection(e):
                    raise
                if rung >= len(LADDER) - 1:
                    raise  # all rungs exhausted
                rung += 1  # try the next reduced shape
        _pin_shape(call_src, call_model, rung)
        # Wrap the stream so every chunk is captured to the log on its way out.
        return _LoggingStream(stream, _llog)
    except (APIConnectionError, httpx.HTTPError) as e:
        st._LAST_API_ERROR = f"{type(e).__name__}: {e}"
        from ._subagent import _subagent_quiet
        if not _subagent_quiet():
            print(f"{C.ERROR}  ✗ can't reach {call_client.base_url} — is the server up? "
                  f"Set NX_BASE_URL (and NX_MODEL) to point at it.{RESET}")
    except Exception as e:
        # A context-length rejection can also surface on a fallback attempt
        # (no-tools / no-tools-no-extra) or be re-raised by the ladder above
        # once every level has been tried — same recovery signal as above.
        if _is_context_overflow_error(e):
            st._LAST_CONTEXT_OVERRUN = str(e)
            return None
        st._LAST_API_ERROR = f"{type(e).__name__}: {e}"
        from ._subagent import _subagent_quiet
        if not _subagent_quiet():
            print(f"{C.ERROR}  ✗ API error: {type(e).__name__}: {e}{RESET}")
    return None


# --- context compression ----------------------------------------------------
# Summarize the older turns of `messages` into a single user-role turn, keeping
# the system prompt and the last K turns verbatim. Frees context without losing
# the model's grip on what it was just doing.
st.COMPRESS_KEEP = 2  # how many recent turns to leave untouched

# Digest turns open with the same header _sessions_fs._SUMMARY_HEADER_RE pins
# for the session-merge contract (see _is_summary_msg below).
_SUMMARY_HEADER_RE = re.compile(r"^\[Compressed context — \d+ earlier turns summarized")


def _is_summary_msg(msg: Any) -> bool:
    """True for the digest turn compress() itself produces — role user, string
    content opening with the same header _sessions_fs._SUMMARY_HEADER_RE pins.
    Marked with an `_digest` key when freshly built and re-detected by content
    shape afterwards so a summary survives a save/load round-trip."""
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    if msg.get("_digest"):
        return True
    content = msg.get("content")
    return isinstance(content, str) and bool(_SUMMARY_HEADER_RE.match(content))


def _keep_target_tokens(auto: bool) -> int:
    """Token floor for the kept tail (2.4a): NX_COMPRESS_KEEP_TOKENS when set
    (any fold), else — auto folds with a known window — a quarter of the
    context window. 0 → message-count semantics only."""
    if st.COMPRESS_KEEP_TOKENS > 0:
        return st.COMPRESS_KEEP_TOKENS
    if not auto or st.ACTIVE is None:
        return 0
    try:
        window = st.ACTIVE.resolve_context_window()
    except Exception:  # noqa: BLE001 — probe failures must not break the fold
        window = None
    if not isinstance(window, int) or window <= 0:
        return 0
    return window // 4


def _compress_call_target() -> tuple[Any, str, dict[str, Any]]:
    """(client, model, extra_request_kwargs) for the summary call (2.4d).
    NX_COMPRESS_SOURCE[/NX_COMPRESS_MODEL] routes the fold to a cheaper
    endpoint; anything unset/unknown falls back to the active source. The
    fold target is never probed or switched-to — it is used as-is."""
    name = (st.COMPRESS_SOURCE or "").strip()
    if name:
        src = st.SOURCES.get(name)
        if src is not None:
            model = st.COMPRESS_MODEL or src.model
            if model:
                return src.client, model, src.extra_request_kwargs()
        from ._subagent import _subagent_quiet
        if not _subagent_quiet() and not st._COMPRESS_SOURCE_WARNED:
            st._COMPRESS_SOURCE_WARNED = True
            print(f"{C.WARNING}  ↳ NX_COMPRESS_SOURCE={name!r} unusable — folding "
                  f"on {getattr(st.ACTIVE, 'name', 'the active source')}{RESET}")
    return (st.client, st.MODEL,
            st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {})


def _spill_tool_results(messages: list[dict[str, Any]], pins: set[int],
                        min_chars: int, max_count: int) -> int:
    """Core of the 0.41 spill-evictor, parameterized (2.4c): replace the
    `max_count` biggest unpinned string tool results with on-disk stubs.
    Replace-in-place ONLY — a tool message must stay beside its assistant
    tool_calls pair or the chat template can't render the history at all.
    Skips anything already carrying a spill note (idempotent)."""
    if max_count <= 0 or min_chars <= 0:
        return 0
    from ._overflow import build_overflow_result, spill_to_disk
    cands = sorted(
        ((len(m["content"]), i) for i, m in enumerate(messages)
         if m.get("role") == "tool" and isinstance(m.get("content"), str)
         and i not in pins and "saved to:" not in m["content"]),
        reverse=True)
    n = 0
    for size, i in cands:
        if size < min_chars or n >= max_count:
            break
        full = messages[i]["content"]
        path = spill_to_disk(full)  # never raises; None on I/O error
        if path:
            messages[i]["content"] = build_overflow_result(full, path)
            n += 1
    return n


def compress(
    messages: list[dict[str, Any]], keep: int | None = None, auto: bool = False
) -> tuple[int, int, int] | None:
    """Ask the model to summarize everything except system + last `keep` turns.

    Mutates `messages` in place on success: replaces the middle slice with a
    single user-role summary turn. Returns (kept_n, summarized_n, summary_chars)
    or None on failure (in which case `messages` is untouched).

    Non-streaming on purpose — we want the whole summary before splicing it in,
    and a spinner for a one-shot summary would be visual noise.

    When auto=True the keep count is raised above COMPRESS_KEEP so
    auto-compression (triggered when the context window fills past
    AUTOCOMPRESS_PERCENT) is more conservative than a manual /compress —
    it keeps roughly the last third of the conversation verbatim so
    in-progress work and recent tool results survive the fold.
    """
    assert st.client is not None and st.MODEL is not None
    # An explicit keep= with auto=True means "the conservative fold, but with
    # THIS keep" — the overflow tiers rely on it (0.41); auto's raise-to-1/3
    # would otherwise defeat a halved keep on a second refusal.
    from ._prefix import is_prefix_note
    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    # The prefix block (AGENTS.md / memory / orientation / delegation notes) is
    # PERMANENT (PERF.md I-3/I-4, PF-15): it defines the pinned bytes every
    # later request must still open with, and it carries the project rules,
    # cwd, and memory index the model needs after the fold. It is carved out
    # before the foldable region and re-inserted unchanged in the rebuild
    # below — a fold must never swallow it (F-1 in PROMPT_CACHE_AUDIT.md).
    prefix_block = []
    body = messages[1:] if sys_msg else messages
    while body and is_prefix_note(body[0]):
        prefix_block.append(body.pop(0))
    # 2.4b: a compress() summary turn at the top of the body is prior art, not
    # material — it is reused verbatim below and the fresh digest is APPENDED
    # to it (bytes only ever grow, the same append-only discipline as the 1.1
    # recovery nudges), so a fold no longer re-bills the prefix cache.
    prior_summary = body[0] if (body and _is_summary_msg(body[0])) else None
    fold_body = body[1:] if prior_summary is not None else body
    if keep is None:
        keep = st.COMPRESS_KEEP
        # Auto-compress is conservative: keep roughly the last third of the
        # conversation verbatim so in-progress work and recent tool results
        # survive the fold. (fold_body excludes the reused summary turn.)
        if auto:
            keep = max(keep, len(fold_body) // 3)
        # 2.4a: a count-keep can still leave a tail that alone overruns the
        # window; grow it to a token target (any fold when NX_COMPRESS_KEEP_
        # TOKENS is set; auto folds derive ~window/4 from the probe).
        target = _keep_target_tokens(auto)
        if target > 0:
            from ._ctxreport import _message_tokens
            acc = 0
            size = 0
            for m in reversed(fold_body):
                acc += _message_tokens(m)
                size += 1
                if acc >= target:
                    break
            if size > keep:
                keep = size
    # keep counts MESSAGES; injected runtime notes are not turns (0.40)
    keep = _keep_past_notes(fold_body, keep)
    if len(fold_body) <= keep:
        return None  # nothing to compress (reused summary at body[0] doesn't count)

    head, tail = fold_body[:-keep], fold_body[-keep:]
    summarized_n = len(head)
    # fold_off: where `head` starts inside `body` (the reused summary turn
    # sits at body[0] and is neither folded nor re-summarized).
    fold_off = 1 if prior_summary is not None else 0

    # The tail must start on a turn the chat template can render — see
    # _drop_leading_dangling_tool_turns for why. Do not simply discard the
    # leading turns it removes: they are real context, and calling them
    # "summarized" without putting them in the summary prompt loses data.
    # Move them into `head` so they are actually folded into the summary.
    safe_tail, n_dropped = _drop_leading_dangling_tool_turns(tail)
    # Dangling leading tail turns are real context: fold them into the summary
    # prompt rather than discarding them (see _drop_leading_dangling_tool_turns).
    moved_into_head = list(tail[:n_dropped])
    if n_dropped:
        summarized_n += n_dropped
    tail = safe_tail
    # 2.4c — spill before summarizing: the bulkiest tool results in the
    # foldable region become on-disk stubs (readable back via read_file) so
    # the summary prompt carries pointers instead of megabytes and the fresh
    # digest summarizes prose. Replace-in-place, so tool_call/tool-result
    # pairs stay renderable; pins are honored after this (they move to the
    # tail below), and already-spilled results ("saved to:") are skipped.
    _spill_tool_results(head, set(), st.COMPRESS_SPILL_MIN_CHARS, st.COMPRESS_SPILL_MAX)

    # Auto-pin the latest todo_write turn (if it's about to be folded): the
    # todo list is the model's live task plan, so losing it mid multi-step
    # work makes the model re-derive or forget what it was doing. Reuses the
    # /ctx pin machinery (below) instead of a separate splice path.
    # Dangling-move turns count too (they are folded in this same call).
    todo_pin = _latest_todo_turn(head + moved_into_head)

    # Auto-pin the latest plan_edit turn the same way: under plan mode the plan
    # file is the gate contract — losing its most recent edit turn makes the
    # post-fold transcript's last-known plan stale (the model may re-attempt a
    # mutation the plan was just edited to forbid).
    plan_pin = _latest_plan_turn(head + moved_into_head)

    # Pinned turns (/ctx pin N) survive every fold: pinned messages move into
    # the kept tail (appended in index order, AFTER the verbatim tail) instead
    # of being summarized away. A chat template can't render a tool result
    # without its preceding assistant tool_call turn (see
    # _drop_leading_dangling_tool_turns), so pins expand to complete pairs and
    # an incomplete pair is dropped from the pins with a note rather than
    # corrupting the tail.
    pins = _pinned_indexes()
    if todo_pin is not None:
        # pins are messages-indexes; todo_pin is a foldable index → shift it
        # past system + the reused summary turn
        pins = set(pins) | {todo_pin + (1 if sys_msg else 0) + fold_off}
    if plan_pin is not None:
        pins = set(pins) | {plan_pin + (1 if sys_msg else 0) + fold_off}
    if pins:
        body_start = (1 if sys_msg else 0) + fold_off  # foldable region start
        split = len(body) - keep           # body indexes < split are foldable
        pin_idx = {i - body_start for i in pins if body_start <= i < split}
        # expand to complete tool_call/tool-result pairs, then drop orphans
        expanded = set(pin_idx)
        for i in pin_idx:
            m = body[i]
            tcs = m.get("tool_calls")
            if tcs:
                for j in range(i + 1, min(i + 1 + len(tcs), len(body))):
                    if body[j].get("role") == "tool":
                        expanded.add(j)
            elif m.get("role") == "tool":
                for j in range(i - 1, max(i - 4, -1), -1):
                    if body[j].get("tool_calls"):
                        expanded.add(j)
                        break
        orphans = []
        for i in sorted(expanded):
            m = body[i]
            if m.get("tool_calls"):
                if not any(body[j].get("role") == "tool" for j in expanded if j > i):
                    orphans.append(i)
            elif m.get("role") == "tool":
                if not any(body[j].get("tool_calls") for j in expanded if j < i):
                    orphans.append(i)
        expanded -= set(orphans)
        if orphans:
            from ._subagent import _subagent_quiet
            if not _subagent_quiet():
                print(f"{C.WARNING}  ↳ {len(orphans)} pinned turn(s) skipped: an "
                      f"assistant-tool-call/tool-result pair must be pinned together{RESET}")
        if expanded:
            pinned_msgs = [body[i] for i in sorted(expanded)]
            head = [m for i, m in enumerate(fold_body[:-keep]) if i not in expanded]
            summarized_n = len(head)
            tail = tail + pinned_msgs
            # The re-derived head lost its spilled stubs — spill again on the
            # new material (spill is idempotent on already-stubbed results).
            _spill_tool_results(head, set(), st.COMPRESS_SPILL_MIN_CHARS,
                                st.COMPRESS_SPILL_MAX)
        # Fold the dangling-move turns into the summary prompt material.
        head = head + moved_into_head
    else:
        # No pins: the pre-spill head carries into the summary prompt as-is.
        head = head + moved_into_head

    # Render the head as plain text the model can summarize. Tool outputs are
    # the bulkiest part of a real session — include them but truncate each one
    # so a single huge read_file doesn't blow up the summary prompt itself.
    def _render(msgs: list[dict[str, Any]]) -> str:
        out = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content")
            if content is None and m.get("tool_calls"):
                # assistant tool-call turn — show the calls so the summary knows what ran
                calls = ", ".join(
                    f"{c['function']['name']}({c['function']['arguments']})"
                    for c in m["tool_calls"]
                )
                out.append(f"[{role}] → {calls}")
            elif isinstance(content, list):
                # some servers return content as a list of parts; flatten it
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                out.append(f"[{role}] {content[:2000]}")
            else:
                out.append(f"[{role}] {(content or '')[:2000]}")
        return "\n\n".join(out)

    summary_prompt = (
        "Summarize the following conversation history for context retention. "
        "Preserve: the original user goal/task, key decisions made, file paths "
        "and identifiers touched, current state of any in-progress work, and "
        "any unresolved questions. Drop: raw tool outputs, full file contents, "
        "and verbose back-and-forth — keep it dense and information-rich. "
        "Write in the same language as the conversation. Output ONLY the "
        "summary, no preamble.\n\n"
        f"---\n{_render(head)}\n---"
    )

    payload = [{"role": "user", "content": summary_prompt}]
    # 2.4d: route the fold to a cheaper endpoint when configured; the summary
    # answer itself is bounded (2.4e) by NX_COMPRESS_SUMMARY_TOKENS, clamped
    # by the same window-aware rule open_stream uses (1.8).
    _client, _model, _extra = _compress_call_target()
    _max_tokens = _clamp_max_tokens(payload, st.COMPRESS_SUMMARY_TOKENS)
    try:
        _log_event("req", {"model": _model, "messages": payload, "stream": False,
                           "max_tokens": _max_tokens, "_purpose": "compress"})
        resp = _client.chat.completions.create(
            model=_model, messages=payload, stream=False, max_tokens=_max_tokens,
            **_extra)
        try:
            _log_event("resp", {"_purpose": "compress", "data": resp.model_dump()})
        except Exception:
            pass  # never let logging break the summary call
    except APIConnectionError:
        from ._subagent import _subagent_quiet
        if not _subagent_quiet():
            print(f"{C.ERROR}  ✗ can't reach {st.client.base_url} — context unchanged{RESET}")
        return None
    except Exception as e:
        from ._subagent import _subagent_quiet
        if not _subagent_quiet():
            print(f"{C.ERROR}  ✗ compress failed: {type(e).__name__}: {e}{RESET}")
        return None

    _record_usage(
        getattr(resp, "usage", None),
        fallback_chars=len(resp.choices[0].message.content or ""),
        fallback_prompt_chars=sum(len(str(m.get("content") or "")) for m in payload),
    )

    summary = (resp.choices[0].message.content or "").strip()
    if not summary:
        from ._subagent import _subagent_quiet
        if not _subagent_quiet():
            print(f"{C.ERROR}  ✗ compress returned empty summary — context unchanged{RESET}")
        return None

    if prior_summary is not None:
        # 2.4b — append-only reuse: the fresh digest extends the existing
        # summary turn instead of replacing it. The message's earlier bytes
        # stay identical (cache-safe); the frozen first line keeps the
        # _SUMMARY_HEADER_RE session-merge contract; each continuation carries
        # its own counts line for the model's benefit.
        continuation = (f"[Compressed context — {summarized_n} more turns "
                        f"summarized; last {len(tail)} turns kept verbatim]\n{summary}")
        prior_summary["content"] = f"{prior_summary['content']}\n\n{continuation}"
        new_mid = [prior_summary]
    else:
        header = f"[Compressed context — {summarized_n} earlier turns summarized; last {len(tail)} turns kept verbatim]"
        new_mid = [{"role": "user", "content": f"{header}\n\n{summary}", "_digest": True}]
    messages[:] = (([sys_msg] if sys_msg else []) + prefix_block
                   + new_mid + tail)
    return len(tail), summarized_n, len(summary)


def _maybe_autocompress(messages: list[dict[str, Any]], prompt_tokens: int) -> bool:
    """Silently compress when context usage crosses the AUTOCOMPRESS_PERCENT
    threshold. Called after each settled model turn with the last turn's
    `prompt_tokens` (the size of the context the model just saw).

    Returns True if a compression happened, False otherwise. Guards:
      - AUTOCOMPRESS_PERCENT == 0 → disabled
      - ACTIVE context window unknown / ≤ 0 → can't compute a ratio
      - ratio < threshold → nothing to do
      - compress() itself bails when there's too little to fold
    """
    if st.AUTOCOMPRESS_PERCENT <= 0:
        return False
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return False
    if st.ACTIVE is None:
        return False
    mx = st.ACTIVE.resolve_context_window()
    if not isinstance(mx, int) or mx <= 0:
        return False
    ratio = prompt_tokens / mx
    if ratio * 100 < st.AUTOCOMPRESS_PERCENT:
        return False
    # Via the top-level package (not a bare same-module reference) so a test
    # that monkeypatches `nx.compress` (to stub out the network call, or
    # to assert it's never invoked) takes effect.
    import nx as _m
    result = _m.compress(messages, auto=True)
    if result is None:
        return False
    kept_n, summarized_n, summary_chars = result
    pct = int(ratio * 100)
    from ._subagent import _subagent_quiet
    if not _subagent_quiet():
        print(f"{DIM}  ↻ auto-compressed {summarized_n} turns → 1 summary "
              f"({summary_chars} chars), kept last {kept_n} verbatim "
              f"(context was at {pct}% of {_abbr(mx)}){RESET}")
    return True
