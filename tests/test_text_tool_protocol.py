#!/usr/bin/env python3
"""Regression tests for Nx's text-mode tool-call protocol."""
import os
import sys
import tempfile


_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def test_standalone_text_tool_call_parses():
    content = (
        '\n[nx_tool_call]{"name": "read_file", '
        '"arguments": {"path": "nx.py"}}[/nx_tool_call]\n'
    )

    assert m.parse_text_calls(content) == [
        ("read_file", {"path": "nx.py"}),
    ]


def test_multiple_standalone_text_tool_calls_parse():
    content = (
        '[nx_tool_call]{"name": "list_files", "arguments": {"path": "."}}[/nx_tool_call]\n'
        '[nx_tool_call]{"name": "read_file", "arguments": {"path": "README.md"}}[/nx_tool_call]'
    )

    assert m.parse_text_calls(content) == [
        ("list_files", {"path": "."}),
        ("read_file", {"path": "README.md"}),
    ]


def test_legacy_tool_call_tag_still_parses():
    content = (
        '<tool_call>{"name": "read_file", '
        '"arguments": {"path": "nx.py"}}</tool_call>'
    )

    assert m.parse_text_calls(content) == [
        ("read_file", {"path": "nx.py"}),
    ]


def test_tool_call_inside_code_block_is_plain_text():
    content = '''I found the system prompt:
```python
SYSTEM = """If your runtime does NOT support native tool calls, emit:
[nx_tool_call]{"name": "read_file", "arguments": {"path": "foo.py"}}[/nx_tool_call]
"""
```
'''

    assert m.parse_text_calls(content) == []


def test_tool_call_with_surrounding_prose_still_parses():
    # Small models often add a sentence of chatter before/after a call despite
    # SYSTEM asking for the tag alone; a leading/trailing preamble shouldn't
    # disqualify an otherwise well-formed call.
    content = (
        'Sure, let me check that file for you: '
        '[nx_tool_call]{"name": "read_file", "arguments": {"path": "foo.py"}}[/nx_tool_call]'
    )

    assert m.parse_text_calls(content) == [
        ("read_file", {"path": "foo.py"}),
    ]


def test_parse_text_calls_with_errors_reports_malformed_json():
    content = '<tool_call>{"name": "read_file", "arguments": {path: "foo.py"}}</tool_call>'

    calls, errors = m.parse_text_calls_with_errors(content)

    assert calls == []
    assert len(errors) == 1
    assert "invalid JSON" in errors[0]


def test_parse_text_calls_with_errors_salvages_good_call_alongside_bad():
    content = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {bad json}}</tool_call>'
    )

    calls, errors = m.parse_text_calls_with_errors(content)

    assert calls == [("read_file", {"path": "a.py"})]
    assert len(errors) == 1


def test_malformed_tool_call_triggers_one_corrective_retry():
    messages = [{"role": "system", "content": "sys"}]
    text = '<tool_call>{"name": "read_file", "arguments": {bad json}}</tool_call>'

    status = m._handle_text_tool_calls(messages, text, malformed_call_count=0)

    assert status == m.TURN_MALFORMED_CALL
    assert messages[-2] == {"role": "assistant", "content": text}
    assert messages[-1]["role"] == "user"
    assert "did not parse" in messages[-1]["content"]


def test_correction_nudge_teaches_live_schema_names_and_args():
    """0.28.1: the corrective nudge hardcodes example tool calls, so a schema
    rename (run_bash's `cmd` -> `command`) or tool rename would silently teach
    the model a call that dies with a TypeError. Parse every example in the
    nudge and demand each name/arg pair exists in the live TOOLS schemas."""
    import re as _re

    messages = [{"role": "system", "content": "sys"}]
    text = '<tool_call>{"name": "read_file", "arguments": {bad json}}</tool_call>'

    status = m._handle_text_tool_calls(messages, text, malformed_call_count=0)

    assert status == m.TURN_MALFORMED_CALL
    nudge = messages[-1]["content"]
    assert "did not parse" in nudge  # pre-existing wording (pinned above too)
    # run_bash's declared key is `command`; the nudge must not teach `cmd`
    assert '"command"' in nudge
    assert '"cmd"' not in nudge
    # the primary example must be the dialect SYSTEM itself teaches
    assert "<name>read_file</name>" in nudge
    assert "<arguments>" in nudge

    schemas = {}
    for t in m.TOOLS:
        fn = t["function"]
        schemas[fn["name"]] = set(fn["parameters"]["properties"])
    # every tool-call example in the nudge (legacy JSON + XML dialect) parses
    calls, errors = m.parse_text_calls_with_errors(nudge)
    assert not errors
    assert len(calls) >= 3
    for name, args in calls:
        assert name in schemas, f"nudge teaches a tool TOOLS doesn't declare: {name}"
        unknown = set(args) - schemas[name]
        assert not unknown, f"nudge teaches non-schema args for {name}: {unknown}"
    # the XML-dialect example's argument tags must exist in read_file's schema
    xml_args = _re.findall(r"<arguments>(.*?)</arguments>", nudge, _re.DOTALL)
    assert xml_args
    for tag in xml_args:
        unknown = set(_re.findall(r"<(\w+)>", tag)) - schemas["read_file"]
        assert not unknown, f"XML example teaches non-schema args: {unknown}"


def test_malformed_tool_call_gives_up_cleanly_after_one_retry(capsys):
    messages = [{"role": "system", "content": "sys"}]
    text = '<tool_call>{"name": "read_file", "arguments": {bad json}}</tool_call>'

    status = m._handle_text_tool_calls(messages, text, malformed_call_count=1)

    # 0.154.6: the invalid call was discarded and nothing ran — honest
    # TURN_GAVE_UP (was TURN_DONE, which headless callers scored as a pass).
    assert status == m.TURN_GAVE_UP
    assert "correction retry also failed" in capsys.readouterr().out
    assert messages[-1]["role"] == "assistant"
    assert "Discarded an invalid tool call" in messages[-1]["content"]
    assert text not in messages[-1]["content"]


def test_tool_result_sanitizer_escapes_legacy_tool_tags():
    content = (
        'line 1\n'
        '<tool_call>{"name": "write_file", "arguments": {"path": "x", "content": "y"}}</tool_call>\n'
    )

    safe = m._sanitize_tool_result(content)

    assert safe.startswith("[nx note:")
    assert "<tool_call>" not in safe
    assert "</tool_call>" not in safe
    assert "&lt;tool_call&gt;" in safe
    assert "&lt;/tool_call&gt;" in safe


def test_tool_result_sanitizer_escapes_nx_tool_tags():
    content = (
        '[nx_tool_call]{"name": "write_file", '
        '"arguments": {"path": "x", "content": "y"}}[/nx_tool_call]'
    )

    safe = m._sanitize_tool_result(content)

    assert safe.startswith("[nx note:")
    assert "[nx_tool_call]" not in safe
    assert "[/nx_tool_call]" not in safe
    assert "&#91;nx_tool_call&#93;" in safe
    assert "&#91;/nx_tool_call&#93;" in safe


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
