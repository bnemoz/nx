"""Smoke-test the /todo slash command without a running model.

/todo is a read-only view of the model's current todo_write list, which lives
in st._TODOS and is rendered by render_todos() (see _todo.py). We seed the
list, run the REPL with /todo, and assert the rendered items appear in the
output; a second run with an empty list asserts the "(todo list empty)"
fallback. Network calls are stubbed out.
"""
import io
import sys
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m
from nx import _state as st


# --- stubs -------------------------------------------------------------------
class FakeStream:
    def __iter__(self): return iter([])


def fake_open_stream(msgs):
    return FakeStream()


def fake_model_turn(msgs, reasoning_loop_cut_count=0, malformed_stream_cut_count=0,
                    empty_turn_count=0, forced_final=False, recovery_sampling=False,
                    malformed_call_count=0):
    return False  # no tool calls → REPL moves to next prompt


captured = io.StringIO()
real_print = print


def fake_print(*a, **kw):
    captured.write(" ".join(str(x) for x in a) + (kw.get("end", "\n")))


def run_with(cmd):
    """Run the REPL with a small seeded conversation then `cmd`, return output."""
    captured.truncate(0); captured.seek(0)
    import builtins
    builtins.print = fake_print
    try:
        m.open_stream = fake_open_stream
        m.model_turn = fake_model_turn
        m.YOLO = True
        prompts = iter(["hello", "world", cmd, "/quit"])
        m.read_multiline = lambda history=None, initial=None: next(prompts)
        m.main()
    finally:
        builtins.print = real_print
    return captured.getvalue()


# --- assertions --------------------------------------------------------------
st._TODOS = [
    {"text": "inspect repository", "status": "done"},
    {"text": "add /todo command", "status": "in_progress"},
    {"text": "write tests", "status": "pending"},
]
out = run_with("/todo")

assert "(todo list empty)" not in out, \
    f"/todo should have shown the seeded list, got:\n{out}"
assert "inspect repository" in out, \
    f"/todo should render each item text, got:\n{out}"
assert "add /todo command" in out, \
    f"/todo should render each item text, got:\n{out}"
assert "write tests" in out, \
    f"/todo should render each item text, got:\n{out}"
# status marks from render_todos must come through (done=[x], in_progress=[~])
assert "[x]" in out and "[~]" in out and "[ ]" in out, \
    f"/todo should show status marks, got:\n{out}"

# empty-list fallback
st._TODOS = []
out_empty = run_with("/todo")
assert "(todo list empty)" in out_empty, \
    f"/todo on an empty list should say so, got:\n{out_empty}"

print("OK — /todo renders the model's current todo_write list")
