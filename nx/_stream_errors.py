"""Error classification for streaming completion requests.

Split out of `_compress.py`. open_stream()'s fallback ladder needs to know,
from a raised exception alone, whether re-sending a reduced request shape can
possibly help: a server that parsed the request and rejected a FIELD (tools=,
tool_choice, extra_body) is worth another rung; a transport failure, a
busy 429/5xx, or a genuine context-window overrun is not."""
from __future__ import annotations

import re

# Match the context-length rejection wording the startup probe already relies
# on (_sources._CTX_LIMIT_RE: OpenAI/Together-style "This model's maximum
# context length is N tokens …"). Lowered/kept broad on purpose: llama.cpp,
# vLLM and proxies vary the surrounding text ("maximum context length",
# "context length exceeded", "reduce the length of the messages"); a relaxed
# pattern over a 400 response is specific enough that false positives are
# effectively limited to other genuine request-size rejections.
_CTX_OVERFLOW_RE = re.compile(
    r"(maximum context length|context length(?:\s+\w+)?\s+(?:is|exceed|exceeded)|"
    r"reduce the (?:length|size) of)", re.IGNORECASE)


def _is_request_field_rejection(e: BaseException) -> bool:
    """True if `e` is a server response that rejected a FIELD of the request
    (bad tools=/extra_body content) rather than a transport/status failure —
    the only case where open_stream()'s no-tools / no-extra fallback ladder
    should re-send.

    The signal: a deterministic validation-shaped status (400/422) from a
    server that parsed the request. 429/5xx mean busy or broken upstream —
    _send has already spent its NX_HTTP_RETRIES on them, and the fallback
    ladder would only multiply that amplification (each rung is a fresh
    multi-attempt _send cycle against a degraded endpoint). Anything without
    an int status_code (unexpected error shapes) also propagates: no evidence
    the server understood and objected to a field, so no evidence a retry
    without it would help."""
    status = getattr(e, "status_code", None)
    if not isinstance(status, int):
        return False
    if status not in (400, 422):
        return False
    text = " ".join(str(x) for x in (
        getattr(e, "message", ""), getattr(e, "body", ""), str(e)) if x)
    return bool(text)  # any body at all — validation errors always carry one


def _is_context_overflow_error(e: BaseException) -> bool:
    """True if exception `e` looks like a context-window-length rejection (an
    HTTP 400 whose body/message matches the over-limit wording). Needs the
    APIError.status_code added by nx's own client; any other error shape is
    not an overrun (returns False → normal terminal-failure handling)."""
    status = getattr(e, "status_code", None)
    if not isinstance(status, int) or status != 400:
        return False
    text = " ".join(str(x) for x in (
        getattr(e, "message", ""), getattr(e, "body", ""), str(e)) if x)
    return bool(_CTX_OVERFLOW_RE.search(text))
