"""Phase 4 (tool-gate transparency, /toolgates) of the explainable harness:
the _toolview report over the tiered registry, the ToolsView surface, and the
command wiring — same fixture patterns as test_turnlog.py."""

import json

from nx import _toolview


def _schema(name, text="x"):
    return {"type": "function", "function": {
        "name": name, "description": text, "parameters": {"type": "object"}}}


def test_report_lists_active_and_withheld(monkeypatch):
    import nx as m
    monkeypatch.setattr(m, "TOOLS", [_schema("search_files"),
                                     _schema("spawn_agent")],
                        raising=False)
    import nx._tool_gates as gates
    monkeypatch.setattr(gates, "_active_tools",
                        lambda: [_schema("search_files")], raising=False)
    lines = _toolview._tools_report_lines()
    assert "1 active tool(s)" in lines[0]
    joined = "\n".join(lines)
    assert "spawn_agent" in joined
    assert "tok to unlock" in joined


def test_unlock_cost_uses_the_shared_estimator(monkeypatch):
    import nx as m
    schema = _schema("spawn_agent", "y" * 700)
    monkeypatch.setattr(m, "TOOLS", [schema], raising=False)
    import nx._tool_gates as gates
    monkeypatch.setattr(gates, "_active_tools", lambda: [], raising=False)
    lines = _toolview._tools_report_lines()
    from nx import _tokens
    assert f"~{_tokens.estimate_tokens(json.dumps(schema))} tok to unlock" \
        in lines[2]


def test_refusal_wording_shown_when_known(monkeypatch):
    import nx as m
    monkeypatch.setattr(m, "TOOLS", [_schema("plan_edit")], raising=False)
    import nx._tool_gates as gates
    monkeypatch.setattr(gates, "_active_tools", lambda: [], raising=False)
    lines = _toolview._tools_report_lines()
    assert any("no plan is active" in ln for ln in lines)   # _GATE_REFUSALS


def test_nothing_withheld_placeholder(monkeypatch):
    import nx as m
    monkeypatch.setattr(m, "TOOLS", [_schema("search_files")], raising=False)
    import nx._tool_gates as gates
    monkeypatch.setattr(gates, "_active_tools",
                        lambda: [_schema("search_files")], raising=False)
    lines = _toolview._tools_report_lines()
    assert any("nothing withheld" in ln for ln in lines)


def test_garbage_schemas_do_not_raise(monkeypatch):
    import nx as m
    monkeypatch.setattr(m, "TOOLS", [{"function": {}}, None, "junk"],
                        raising=False)
    import nx._tool_gates as gates
    monkeypatch.setattr(gates, "_active_tools", lambda: [], raising=False)
    lines = _toolview._tools_report_lines()
    assert any("?" in ln for ln in lines)


def test_view_paints_the_same_lines_and_is_registered():
    from nx import _fullscreen as fs
    view = _toolview.tools_view()
    painted = "\n".join(view.render(240, 40))
    assert painted and fs.make_view("tools") is not None
    assert view.on_key("r", None) == "repaint"
    assert view.on_key("esc", None) == "quit"


def test_context_view_g_jumps_to_tools_view():
    from nx import _ctxreport
    pushed = []

    class _Shell:
        def push(self, v):
            pushed.append(v)

    class _State:
        messages = [{"role": "user", "content": "hi"}]

    view = _ctxreport.context_view(_State())
    assert view.on_key("g", _Shell()) == "repaint"
    assert len(pushed) == 1 and type(pushed[0]).__name__ == "ToolsView"
    assert view.on_key("g", None) == "repaint"     # shell=None degrades


def test_cmd_toolgates_inline_prints_the_report(monkeypatch, capsys):
    import nx as m
    monkeypatch.setattr(m, "TOOLS", [_schema("search_files")], raising=False)
    import nx._tool_gates as gates
    monkeypatch.setattr(gates, "_active_tools",
                        lambda: [_schema("search_files")], raising=False)
    monkeypatch.setattr("nx._fullscreen.wants_fullscreen", lambda: False)
    _toolview._cmd_toolgates("")
    out = capsys.readouterr().out
    assert "active tool(s)" in out
