"""Evidence-Preserving Reducer (SoL-Pi port, 0.67.0): for large diagnostic
outputs (build/test commands), delegate a first-read summary to a second
model call, archive the raw log, and accept the summary ONLY if every quoted
line verifies byte-for-byte against the archive; otherwise fail open with the
original output.

Failure posture is always fail-open: any validation miss returns None and the
caller leaves the original tool result untouched.

Env knobs (nx convention replaces SoL-Pi's JSON config):
  NX_REDUCER=1                 enable (default 0)
  NX_REDUCER_SOURCE            source name for the summary model (default: active source)
  NX_REDUCER_MODEL             model name for the summary model (default: active model)
  NX_REDUCER_MIN_BYTES         4096  — below this, never bother
  NX_REDUCER_MAX_CHARS         600000 — above this, fail open (source-over-max-chars)
  NX_REDUCER_MAX_TOKENS        2048  — summary answer cap
  NX_REDUCER_TIMEOUT           90    — seconds for the summary call
  NX_REDUCER_MAX_ITEMS         12    — evidence items accepted in a receipt
  NX_REDUCER_MAX_QUOTE_CHARS   600   — per-quote cap

Receipt contract (schema sol-pi-evidence-receipt/1) is byte-identical to
SoL-Pi's so the two harnesses can read each other's archives; the receipt's
first line (sol_pi_evidence_receipt_v1) is the cross-mechanism marker the
observation pack never packs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from ._env import _env_int

REDUCER_ENABLED = _env_int("NX_REDUCER", 0)
MIN_BYTES = _env_int("NX_REDUCER_MIN_BYTES", 4096)
MAX_CHARS = _env_int("NX_REDUCER_MAX_CHARS", 600_000)
MAX_OUTPUT_TOKENS = _env_int("NX_REDUCER_MAX_TOKENS", 2048)
TIMEOUT_SECONDS = _env_int("NX_REDUCER_TIMEOUT", 90)
MAX_EVIDENCE_ITEMS = _env_int("NX_REDUCER_MAX_ITEMS", 12)
MAX_QUOTE_CHARS = _env_int("NX_REDUCER_MAX_QUOTE_CHARS", 600)

EVENT_SCHEMA = "sol-pi-evidence-preserving-reducer/1"
RECEIPT_SCHEMA = "sol-pi-evidence-receipt/1"
RECEIPT_PREFIX = "sol_pi_evidence_receipt_v1"

ALLOWED_KINDS = ("fatal", "failure", "warning", "target", "summary")

DIAGNOSTIC_COMMAND = re.compile(
    r"(?:^|[;&|()\s])(?:lake\s+build|lake\s+env\s+lean|lean|coq|"
    r"cargo(?:\s+(?:build|test|check))?|zig\s+build|pytest|"
    r"python(?:3)?\s+-m\s+(?:pytest|unittest|py_compile)|ctest|"
    r"cmake\s+--build|ninja|make|npm\s+test|pnpm\s+test|yarn\s+test|"
    r"go\s+test|bazel\s+test)(?:\s|$)", re.IGNORECASE)
FAILURE_SIGNAL = re.compile(
    r"error|failed|failure|fatal|exception|panic|timeout|unsolved|"
    r"type mismatch|assert", re.IGNORECASE)
LIKELY_SECRET = re.compile(
    r"(?:api[_-]?key|authorization|bearer|access[_-]?token|secret)"
    r"[^\n]{0,32}[=:][^\n]+", re.IGNORECASE)

REDUCER_SYSTEM_PROMPT = (
    "You reduce long diagnostic logs (build/test output) to a compact receipt. "
    "Reply with ONLY one JSON object, no markdown fence:\n"
    '{"schema":"sol-pi-evidence-receipt/1","source_sha256":"<hex>",'
    '"status":"success|failure","uncertain":<bool>,"evidence":['
    '{"kind":"fatal|failure|warning|target|summary","quote":"<exact text>",'
    '"uncertain":<bool>}...]}\n'
    "Rules: every quote MUST be copied byte-for-byte from the log (it is "
    f"verified mechanically; any mismatch rejects the whole receipt). At most "
    f"{MAX_EVIDENCE_ITEMS} evidence items, each quote at most {MAX_QUOTE_CHARS} "
    "characters. kinds: fatal = the error that ended the run, failure = a "
    "failed test/build step, warning = compiler/linter warnings worth seeing, "
    "target = which file/test target failed, summary = one line of what the "
    "run was. Set uncertain=true when the log does not clearly show the "
    "outcome. Do not diagnose, do not propose fixes, do not adjudicate "
    "pass/fail beyond what the log states."
)


def reducer_enabled() -> bool:
    return REDUCER_ENABLED == 1


def store_root() -> str:
    from . import _state as st
    return os.path.join(st.SESSION_HOME, "evidence-preserving-reducer")


def run_id() -> str:
    return hashlib.sha256(store_root().encode("utf-8")).hexdigest()[:16]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_diagnostic_command(command: str) -> bool:
    return bool(DIAGNOSTIC_COMMAND.search(command or ""))


def extract_then_run_body(result: str) -> tuple[str, str, bool]:
    """For a fused edit result, the reducible body is the text after the
    [then_run:...] marker. Returns (body, marker, is_error) or ("", "", False)
    when no marker is present."""
    for marker, is_err in (("[then_run:failed]", True),
                           ("[then_run:succeeded]", False)):
        idx = result.find(marker)
        if idx >= 0:
            return result[idx + len(marker):].lstrip("\n"), marker, is_err
    return "", "", False


def archive_body(body: str) -> tuple[str | None, str]:
    """Content-addressed archive; returns (path, sha256) or (None, hash) on
    I/O failure (fail-open). Existing identical objects are left in place;
    an existing object with different content is an integrity failure -> None."""
    h = _sha256(body)
    try:
        root = os.path.join(store_root(), "objects", h[:2])
        os.makedirs(root, mode=0o700, exist_ok=True)
        path = os.path.join(root, h + ".txt")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
        except FileExistsError:
            with open(path, "r", encoding="utf-8") as f:
                if f.read() != body:
                    return None, h
        return path, h
    except OSError:
        return None, h


def _validate_receipt(payload: Any, body: str, expected_sha: str,
                      is_error: bool) -> tuple[dict[str, Any] | None, str]:
    """Returns (receipt, reason). reason is "" on success."""
    if not isinstance(payload, dict):
        return None, "invalid-json"
    if payload.get("schema") != RECEIPT_SCHEMA:
        return None, "schema-mismatch"
    if payload.get("source_sha256") != expected_sha:
        return None, "schema-mismatch"
    want_status = "failure" if is_error else "success"
    if payload.get("status") != want_status:
        return None, "schema-mismatch"
    if not isinstance(payload.get("uncertain"), bool):
        return None, "schema-mismatch"
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE_ITEMS:
        return None, "schema-mismatch"
    verified = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        if not isinstance(item, dict):
            return None, "unverifiable-quote"
        kind = item.get("kind")
        quote = item.get("quote")
        if kind not in ALLOWED_KINDS or not isinstance(quote, str):
            return None, "unverifiable-quote"
        if not quote or len(quote) > MAX_QUOTE_CHARS:
            return None, "unverifiable-quote"
        key = (kind, quote)
        if key in seen:
            continue
        seen.add(key)
        if quote not in body:
            return None, "unverifiable-quote"
        line_no = body[:body.index(quote)].count("\n") + 1
        verified.append({"kind": kind, "line": line_no,
                         "quote_sha256": _sha256(quote), "quote": quote})
    if is_error and FAILURE_SIGNAL.search(body) and not any(
            v["kind"] in ("fatal", "failure") for v in verified):
        return None, "missing-failure-evidence"
    receipt = {"schema": RECEIPT_SCHEMA,
               "source_sha256": expected_sha,
               "status": want_status,
               "uncertain": payload["uncertain"],
               "evidence": verified}
    return receipt, ""


def receipt_text(receipt: dict[str, Any], archive_path: str,
                 source_bytes: int, source_lines: int,
                 command_sha: str, provider: str, model: str,
                 total_tokens: int) -> str:
    lines = [
        RECEIPT_PREFIX,
        f"status={receipt['status']}",
        f"uncertain={str(receipt['uncertain']).lower()}",
        f"command_sha256={command_sha}",
        f"source_sha256={receipt['source_sha256']}",
        f"source_bytes={source_bytes}",
        f"source_lines={source_lines}",
        f"source_artifact={archive_path}",
        f"reducer_provider={provider}",
        f"reducer_model={model}",
        f"reducer_total_tokens={total_tokens}",
        "verified_evidence:",
    ]
    ev = receipt["evidence"]
    if not ev:
        lines.append("- none")
    else:
        for item in ev:
            lines.append(
                f"- kind={item['kind']} line={item['line']} "
                f"quote_sha256={item['quote_sha256']} "
                f"quote={json.dumps(item['quote'], ensure_ascii=False)}")
    lines.append("authority=Sol retains diagnosis, repair, rerun, and "
                 "pass/fail adjudication")
    lines.append("readback=use read_file or a byte/line range on "
                 "source_artifact when exact context is needed")
    return "\n".join(lines)


def _call_reducer_model(body: str, source_sha: str, source_bytes: int,
                        source_lines: int, is_error: bool
                        ) -> tuple[str | None, str, str, int]:
    """One non-streaming summary call. Returns (content, provider, model,
    total_tokens); content None on any failure."""
    import nx as _m
    from . import _state as st
    src_name = (os.environ.get("NX_REDUCER_SOURCE") or "").strip()
    model_name = (os.environ.get("NX_REDUCER_MODEL") or "").strip()
    src = st.SOURCES.get(src_name) if src_name else st.ACTIVE
    if src is None:
        src = st.ACTIVE
    if src is None or st.client is None:
        return None, "", "", 0
    model = model_name or src.model or st.MODEL or ""
    client = src.client if src is not st.ACTIVE else st.client
    if not model:
        return None, "", "", 0
    user = (f"command_sha256={_sha256(src_name or 'active')}\n"
            f"source_sha256={source_sha}\n"
            f"source_bytes={source_bytes}\n"
            f"source_lines={source_lines}\n"
            f"is_error={str(is_error).lower()}\n"
            f"<untrusted_log>\n{body}\n</untrusted_log>")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": REDUCER_SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            max_tokens=min(MAX_OUTPUT_TOKENS,
                           _m.MAX_COMPLETION_TOKENS
                           if hasattr(_m, "MAX_COMPLETION_TOKENS") else MAX_OUTPUT_TOKENS),
            timeout=TIMEOUT_SECONDS,
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        total = int(getattr(usage, "total_tokens", 0) or 0)
        return content, src_name or getattr(src, "name", "active"), model, total
    except Exception:  # noqa: BLE001 — fail open
        return None, "", "", 0


def reduce_tool_result(tool_name: str, command: str, result: str,
                       is_error: bool = False) -> str | None:
    """The tool_result seam. Returns the receipt text to REPLACE the result
    with, or None to leave the original untouched (fail-open at every step)."""
    if not reducer_enabled():
        return None
    try:
        body = result
        marker = ""
        if tool_name in ("write_file", "edit_file", "multi_edit"):
            body, marker, ferr = extract_then_run_body(result)
            if not marker:
                return None
            is_error = ferr or is_error
        elif tool_name != "run_bash":
            return None
        if not is_diagnostic_command(command) or not body:
            return None
        raw = body.encode("utf-8", "replace")
        if len(raw) < MIN_BYTES:
            return None
        if len(body) > MAX_CHARS:
            return None
        if LIKELY_SECRET.search(body):
            return None
        archive_path, sha = archive_body(body)
        if archive_path is None:
            return None
        content, provider, model, total_tokens = _call_reducer_model(
            body, sha, len(raw), body.count("\n") + 1, is_error)
        if not content:
            return None
        # strip a markdown fence if the model added one anyway
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        try:
            payload = json.loads(stripped)
        except ValueError:
            return None
        receipt, reason = _validate_receipt(payload, body, sha, is_error)
        if receipt is None:
            return None
        text = receipt_text(receipt, archive_path, len(raw),
                            body.count("\n") + 1, _sha256(command or ""),
                            provider, model, total_tokens)
        if len(text.encode("utf-8", "replace")) >= len(raw):
            return None  # receipt-not-smaller
        return text
    except Exception:  # noqa: BLE001
        return None
