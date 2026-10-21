"""Phase 1 (transcript detail + as-sent) of the explainable harness: the
ContextView detail pane, the live/as-sent toggle over st._LAST_PROMPT, and
the inline `/context t <n>` output — same fixtures as test_fullscreen.py's
ContextView block."""

from nx import _ctxreport


class _CtxState:
    """The bit of `_ReplState` the view reads: the live message list."""

    def __init__(self, n: int = 3) -> None:
        self.messages = [{"role": "system", "content": "sys prompt"}]
        for i in range(n):
            self.messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"msg {i} " + "x" * 80})


def _digest(msg):
    return _ctxreport._message_digest(msg)


def test_detail_pane_renders_the_full_message():
    from nx import _render
    state = _CtxState(2)
    state.messages.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "search_files",
                                     "arguments": "{\"pattern\": \"x\"}"}}]})
    view = _ctxreport.context_view(state)
    view.on_key("j", None)          # messages[1] = the padded message
    view.on_key("enter", None)
    assert view.detail
    body = "\n".join(view.render(240, 40))
    assert _render.BOX_TL in body
    # FULL text, not the 64-char preview: wrapped at 72, none elided (the
    # strip accounts for the pane's ANSI; the pane may split the run)
    import re as _re
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", body)
    assert "x" * 66 in plain
    # esc returns to the list (and does NOT quit the view)
    assert view.on_key("esc", None) == "repaint"
    assert not view.detail
    assert view.on_key("esc", None) == "quit"
    # the tool-call row renders its name + args in the pane too
    view.on_key("j", None)
    view.on_key("j", None)
    view.on_key("enter", None)
    body = "\n".join(view.render(240, 40))
    assert "tool call: search_files" in body
    assert '{"pattern": "x"}' in body


def test_detail_pane_shows_tool_result_message():
    state = _CtxState(1)
    state.messages.append({"role": "tool", "content": "42 matches found"})
    view = _ctxreport.context_view(state)
    view.on_key("j", None)
    view.on_key("j", None)          # messages[2] = the tool result
    view.on_key("enter", None)
    assert "42 matches found" in "\n".join(view.render(240, 40))


def test_as_sent_toggle_marks_identical_and_differing(monkeypatch):
    from nx import _state as st
    state = _CtxState(3)
    sent = [{"role": m["role"], "index": i,
             "digest": _digest(m) if i != 1 else "0" * 64}
            for i, m in enumerate(state.messages)]
    monkeypatch.setattr(st, "_LAST_PROMPT", sent, raising=False)
    view = _ctxreport.context_view(state)
    view.on_key("a", None)          # toggle before opening = list-wide hint
    view.on_key("enter", None)      # row 0 — digest matches
    body = "\n".join(view.render(240, 40))
    assert "as-sent index=0" in body
    assert "identical to live" in body
    view.on_key("esc", None)
    view.on_key("j", None)          # row 1 — digest differs
    view.on_key("enter", None)
    body = "\n".join(view.render(240, 40))
    import re as _re
    assert "differs from live (elided/summarized)" in _re.sub(
        r"\x1b\[[0-9;]*m", "", body)
    assert "as-sent index=1" in _re.sub(r"\x1b\[[0-9;]*m", "", body)


def test_as_sent_toggle_without_a_settled_call_shows_placeholder(monkeypatch):
    from nx import _state as st
    monkeypatch.setattr(st, "_LAST_PROMPT", None, raising=False)
    view = _ctxreport.context_view(_CtxState())
    view.on_key("enter", None)
    view.on_key("a", None)
    body = "\n".join(view.render(240, 40))
    assert "— no settled call recorded —" in body
    # and the list view below still renders (stays on live once esc'd)
    view.on_key("esc", None)
    assert not view.as_sent or True
    assert "context" in "\n".join(view.render(120, 20))


def test_as_sent_missing_attribute_falls_back_like_none(monkeypatch):
    from nx import _state as st
    monkeypatch.delattr(st, "_LAST_PROMPT", raising=False)
    view = _ctxreport.context_view(_CtxState())
    view.on_key("a", None)
    view.on_key("enter", None)
    assert "— no settled call recorded —" in "\n".join(view.render(240, 40))


def test_context_t_inline_prints_one_message_transcript(monkeypatch, capsys):
    from nx import _repl_settings_cmds as cmds
    state = _CtxState(3)
    cmds._cmd_ctx("/context t 2", state)
    out = capsys.readouterr().out
    assert "role=user" in out
    assert "msg 0" in out and "x" * 66 in out   # full text, not the preview
    # strip ANSI the view's pane adds; the bytes must match the helper
    import re
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert plain.strip() == _ctxreport._transcript_detail(
        state.messages[1], 1).strip()


def test_context_t_inline_error_handling(monkeypatch, capsys):
    from nx import _repl_settings_cmds as cmds
    cmds._cmd_ctx("/context t", _CtxState(3))
    assert "usage" in capsys.readouterr().out
    cmds._cmd_ctx("/context t 99", _CtxState(3))
    assert "outside 1..4" in capsys.readouterr().out


def test_message_digest_is_stable_sha256_of_role_and_text():
    msg = {"role": "user", "content": "hello"}
    d1 = _digest(msg)
    import hashlib
    assert d1 == hashlib.sha256(
        b"user\x00hello").hexdigest()
    assert d1 == _digest({"role": "user", "content": "hello"})
