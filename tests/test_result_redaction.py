#!/usr/bin/env python3
"""Redaction pass in _sanitize_tool_result (roadmap 1.6, 0.40): known secret
shapes are masked BEFORE the result is shown, capped, spilled to disk or
persisted anywhere (sessions, sub-agent reports, bg logs).

The patterns are the same token shapes the approval-side gate
(_bash_jobs._SECRET_PATTERNS) already forces a prompt for on the way IN — this
covers the way OUT (a tool that printed a secret it was handed, an env dump, a
config file read). Placeholder keeps the token KIND so the model can still
reason about what it saw, just not the value.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _text_fallback as tf  # noqa: E402


def test_known_token_shapes_are_redacted():
    text = ("ok\n"
            "OPENAI_KEY=sk-abcdefghijklmnop123456\n"
            "GH=ghp_ABCDEFGHIJKLMNOPQRSTUVWX\n"
            "AWS=AKIAIOSFODNN7EXAMPLE\n"
            "SLACK=xoxb-123456789012-abcdefghijkl\n"
            "GL=glpat-abcdefghijklmnopqr\n")
    out = m._sanitize_tool_result(text)
    assert "sk-abcdefghijklmnop123456" not in out
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWX" not in out
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "xoxb-123456789012-abcdefghijkl" not in out
    assert "glpat-abcdefghijklmnopqr" not in out
    assert "[redacted:api-key]" in out
    assert "[redacted:github-token]" in out
    assert "[redacted:aws-access-key]" in out
    assert "[redacted:slack-token]" in out
    assert "[redacted:gitlab-token]" in out


def test_headers_and_pem_are_redacted_prefix_kept():
    text = ("POST /v1/x\n"
            "Authorization: Bearer AbCdEf12345678901234567890\n"
            "x-api-key: ZzYyXxWwVuTsRqPoNmLkJiHh 1234\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA7\n")
    out = m._sanitize_tool_result(text)
    assert "AbCdEf12345678901234567890" not in out
    assert "Authorization: Bearer [redacted:bearer-token]" in out
    assert "x-api-key: [redacted:api-key]" in out
    assert "-----BEGIN RSA PRIVATE KEY-----" not in out
    assert "[redacted:private-key]" in out


def test_redaction_runs_before_the_disk_spill(monkeypatch):
    monkeypatch.setattr(tf, "TOOL_RESULT_CHARS", 200)
    secret = "sk-abcdefghijklmnop123456"
    body = "noise\n" + "\n".join(f"attempt-{i} {secret}" for i in range(40))
    out = m._sanitize_tool_result(body)
    assert secret not in out, "the model-visible result must be masked"
    spill_path = out.rsplit("saved to: ", 1)[1].split(" —", 1)[0].strip()
    with open(spill_path, encoding="utf-8") as f:
        spilled = f.read()
    assert secret not in spilled, "the SPILL must not hold the raw secret either"


def test_redact_off_switch_restores_passthrough(monkeypatch):
    monkeypatch.setattr(tf, "TOOL_RESULT_REDACT", 0)
    secret = "sk-abcdefghijklmnop123456"
    out = m._sanitize_tool_result(f"key={secret}")
    assert secret in out


def test_short_results_still_get_redacted():
    # <1000 chars used to early-return before any content transform; redaction
    # must apply on that path too (env dumps and config cats are usually short).
    out = m._sanitize_tool_result("key: sk-shortexamplekey12345")
    assert "sk-shortexamplekey12345" not in out
    assert "[redacted:api-key]" in out


def test_ordinary_output_is_untouched():
    text = ("def f(x):\n"
            "    return x + 1\n"
            "task-sk_model_selection=done\n"   # 'sk-' shaped but not a key
            "wsk-node-12 and disk-busy-4\n")
    out = m._sanitize_tool_result(text)
    assert out == text
