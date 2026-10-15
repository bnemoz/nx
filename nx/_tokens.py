"""The one token estimator nx shares (stdlib only + optional tiktoken).

Before this module, five call sites carried five different char→token
heuristics (chars//4, utf-8-bytes//4, round(chars/3.5), ...) so /ctx, /cost,
the footer and the compaction gate quietly disagreed about how big the same
prompt was. This module is the single source of truth for that conversion:

  estimate_tokens(text)        exact via tiktoken/cl100k_base when the
                               optional dependency is importable, else the
                               chars/3.5 heuristic below.
  estimate_message_tokens(msg) one OpenAI-shaped message: string or
                               list-of-parts `content` plus native
                               `tool_calls` (function name + arguments).

Import cost is the whole design: tiktoken is imported INSIDE the first
estimate_tokens() call, never at module import. `import nx` must stay snappy
and work offline; tiktoken's first real use may download a vocab file, which
is fine for a live turn but not for an import. A failed import/lookup latches
a sentinel so the process falls back permanently instead of retrying the
import on every call (this runs in per-message loops).

The fallback divisor: natural-language code and prose tokenize at roughly
3.3–3.5 characters per token for cl100k-family BPEs (the older chars//4 rule
of thumb was a rounder, looser version of the same fact). Counted on
CHARACTERS, never UTF-8 bytes — a multibyte payload would otherwise be
overcounted by up to 4x against a tokenizer that splits on text, not bytes.
"""
from __future__ import annotations

import math
from typing import Any

# Documented fallback rate (see module docstring). Kept as a constant so tests
# and call sites reference one number instead of re-typing a magic divisor.
CHARS_PER_TOKEN = 3.5

# Lazy encoding singleton: None = not tried yet, False = tried and unavailable
# (latched for the process), anything else is the encoding object.
_encoding: Any = None


def _get_encoding() -> Any:
    """The cl100k_base encoding, or None when tiktoken is unusable.

    First call imports tiktoken and resolves the encoding — which may fetch the
    vocab file over the network — so this is deliberately never reached at
    import time. Any failure (missing dependency, no network, a broken
    install) latches `False`: the estimator falls back for the rest of the
    process rather than paying for a failed import per message.
    """
    global _encoding
    if _encoding is None:
        try:
            import tiktoken  # ty: ignore[unresolved-import]  # pyrefly: ignore[missing-import]  # optional dep, guarded by this try/except
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 — any failure means "use the heuristic"
            _encoding = False
    return _encoding or None


def encoding_available() -> bool:
    """True when the exact tiktoken estimator is usable in this process
    (resolves the lazy singleton, so it may be the call that pays for the
    vocab download)."""
    return _get_encoding() is not None


def estimate_tokens_from_chars(n_chars: int) -> int:
    """The chars/3.5 heuristic on a character COUNT.

    Same math the no-tiktoken branch of estimate_tokens() uses, exposed for
    call sites that only ever kept a running character count (the per-turn
    footer and _metrics' usage fallback) and must not allocate a string of
    that size to convert it.
    """
    if n_chars <= 0:
        return 0
    return max(1, math.ceil(n_chars / CHARS_PER_TOKEN))


def estimate_tokens(text: str) -> int:
    """Tokens in `text`: exact (tiktoken cl100k_base) when available, else
    ceil(chars / 3.5). Empty/None-ish text is 0."""
    if not text:
        return 0
    if isinstance(text, list):  # content-parts form (the shape /ctx feeds us)
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        if not text:
            return 0
    elif not isinstance(text, str):
        text = str(text)
    enc = _get_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:  # noqa: BLE001 — a tokenizer that breaks mid-call must not break a turn
            pass
    return estimate_tokens_from_chars(len(text))


def message_content_text(msg: dict[str, Any]) -> str:
    """The plain text of a message's `content`: str as-is, list-of-parts
    joined over its text parts, anything else as ''."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimated tokens for one message, including native tool_calls args."""
    n = estimate_tokens(message_content_text(msg))
    tcs = msg.get("tool_calls") or []
    for tc in tcs:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        n += estimate_tokens(str(fn.get("name") or "")) + estimate_tokens(str(fn.get("arguments") or ""))
    return n
