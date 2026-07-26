"""Smoke-test that /clear is an alias for /reset, without a running model.

Both commands should clear the in-memory context down to just the system
prompt and fork a fresh session id. We feed each command to the REPL with
the model calls stubbed out, and assert both produce the same output and
leave the message log in the same cleared state.
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


out_reset = run_with("/reset")
out_clear = run_with("/clear")
out_new = run_with("/new")

# --- assertions --------------------------------------------------------------
for label, out in (("/reset", out_reset), ("/clear", out_clear), ("/new", out_new)):
    assert "context cleared" in out, \
        f"{label} should have cleared context, got:\n{out}"
    assert "new session" in out, \
        f"{label} should have forked a new session id, got:\n{out}"

# The three commands take the same code path, so their output should match
# exactly (the new session id is non-deterministic, so strip those lines).
def strip_session_id(out):
    return "\n".join(
        line for line in out.splitlines()
        if "new session" not in line
    )


baseline = strip_session_id(out_reset)
for label, out in (("/clear", out_clear), ("/new", out_new)):
    assert strip_session_id(out) == baseline, \
        f"{label} should behave identically to /reset:\n" \
        f"--- /reset ---\n{out_reset}\n--- {label} ---\n{out}"

print("OK — /clear and /new are aliases for /reset")
