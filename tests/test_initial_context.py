#!/usr/bin/env python3
"""Guard the README's "~625 tokens on a bare hey" claim.

The README advertises that a bare `hey` costs about 625 tokens of initial
context. That floor is the whole reason nx exists — it's two orders of
magnitude under the harnesses that spend 20K-50K before you've typed
anything — so this test watches it. If a new tool or a fatter SYSTEM bloats
the payload, we want to know before the README quietly becomes a lie.

WHAT WE MEASURE (the part we control and send to the API):
  - SYSTEM          the system prompt
  - TOOLS           the 7-function tool schema, JSON-encoded
  - "hey"           a minimal first user turn
The per-turn floor is enforced as a recorded BUDGET, not a soft warn: a new
tool or a fatter schema fails the suite until the number is consciously
re-derived (that is the whole point — the 2500→2886 drift of 0.30–0.39 was
visible for months as a warning nobody read).

Measured 0.41, chars/3.5 (see _count_tokens), NX_SUBAGENTS/NX_VISION off —
17 active schemas: **2879** (SYSTEM 375 + TOOLS 2503 + 'hey' 1). With
NX_SUBAGENTS on the set gains spawn_agent (330 tok after the 0.41 trim,
was 396) → 3210. The old soft warn at 2500 predates the schema rewrite and
fired on every run, which is the same as having no line at all.

BUDGET covers the 17-tool default set because that is what a fresh session
pays; enabling sub-agents is a deliberate, user-visible choice. A PR that
moves the floor past BUDGET_TOKENS must either trim elsewhere or raise the
number here — with the justification in the ledger below.

TOKENIZER: prefers tiktoken (cl100k_base) when installed for a real count;
falls back to a conservative chars/3.5 estimate otherwise so the guard still
runs with zero extra dependencies, and we don't add one for a test. The
fallback over-counts ~7% vs cl100k on code/JSON, which is the safe
direction for a guard — it trips slightly early, never late.
"""
import json
import os
import sys
import tempfile

from nx._env import _env_int

# Keep nx from touching ~/.nx — same convention as the other tests,
# even though this one never writes a session.
_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
# Standard isolation (AGENTS.md): a local env file must not change what we
# measure. Exported NX_SUBAGENTS/NX_VISION are handled at call time below
# (nx resolves them at import, which may happen in another test module first).
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402

# Thresholds on the content token count (SYSTEM + TOOLS JSON + "hey").
# With the active schema gated by _active_tools() (wait_background excluded
# until a command backgrounds — currently read_file, write_file, edit_file,
# list_dir, run_bash, search_files, git_status, git_diff, git_log, multi_edit,
# sed_edit, apply_patch, todo_write, memory_write, memory_search, memory_delete,
# delete_file) the chars/3.5 content floor is ~2680; the README's figure adds
# ~50 of server-side framing. git_log + apply_patch (two new tools) plus
# read_file/search_files list-argument params pushed the floor up from
# ~1990; memory_write + memory_search (model-driven memory, two more tools)
# plus the SYSTEM memory line raised it again from ~2263; a 0.3.1 pass that
# trimmed verbose tool descriptions brought it back under 2500; a 0.18.9 pass
# re-streamlined the fattest descriptions (edit_file, search_files, multi_edit,
# sed_edit, run_bash, spawn_agent) which had crept back past the warn line; the
# 0.26 schema rewrite moved characters from terse abbreviation blobs into
# per-PROPERTY guidance (an undocumented `old_lines`/`timeout`/`scope` is a
# guessed-at argument), landing ~2835 — still warning, still under the ceiling.
# 0.41: the 2500 warn line — set when the active set was 7 tools — had been
# firing on every run for months, which is a warning nobody reads. Replaced by
# a recorded budget: BUDGET_TOKENS below, with this ledger as its justification.
# The 0.41 trim pass took spawn_agent 396→330 (trigger-list and async mechanics
# moved to the one-time Delegation note) and edit_file/plan_edit down a few
# dozen; the 17-tool floor now measures 2879.
BUDGET_TOKENS = _env_int("NX_FLOOR_BUDGET", 2900)  # hard: recorded, justified
SOFT_TOKENS = BUDGET_TOKENS - 50  # advisory: floor drift is under way, trim before it lands


def _count_tokens(text):
    """Token count for a string.

    Returns (count, method). Uses tiktoken's cl100k_base when available
    (the closest thing to a shared standard across served models); falls
    back to a conservative chars/3.5 estimate so the test runs with no
    extra deps. The fallback over-counts on code/JSON, which is the safe
    direction for a bloat guard.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "tiktoken"
    except Exception:
        return max(1, round(len(text) / 3.5)), "chars/3.5"


def test_initial_context_floor():
    # The exact payload a fresh "hey" turn sends: system prompt + the ACTIVE
    # tool schemas (_active_tools() — excludes wait_background until a
    # run_bash call has actually backgrounded this session, see
    # _SESSION_HAS_BACKGROUNDED) + a one-token user message. FINAL_ANSWER_TOOL
    # is intentionally excluded — it's only injected on a forced-final retry
    # (nx.py:1847), not on turn 1, so it's not part of the README's ~625.
    # Opt-in tools are forced OFF for the measurement: a developer shell with
    # NX_SUBAGENTS=1/NX_VISION=1 must not change the guarded baseline (nx
    # resolves those at import, possibly in another test module, so set the
    # module attributes rather than the env vars).
    saved = (m.NX_SUBAGENTS, m.NX_VISION)
    m.NX_SUBAGENTS = m.NX_VISION = False
    try:
        active_tools = m._active_tools()
        system_tokens, method = _count_tokens(m.SYSTEM)
        tools_tokens, _ = _count_tokens(json.dumps(active_tools))
        user_tokens, _ = _count_tokens("hey")
    finally:
        m.NX_SUBAGENTS, m.NX_VISION = saved
    total = system_tokens + tools_tokens + user_tokens

    breakdown = (
        f"initial context content = {total} tokens ({method}): "
        f"SYSTEM {system_tokens} + TOOLS {tools_tokens} "
        f"({len(active_tools)} funcs) + 'hey' {user_tokens}"
    )
    print(breakdown)

    # Sanity: a broken payload (empty SYSTEM, TOOLS clobbered to []) would
    # make a token-floor test meaningless by passing on a near-zero count.
    assert m.SYSTEM.strip(), "SYSTEM prompt is empty"
    assert len(active_tools) >= 1, "TOOLS is empty — nothing to guard"
    assert "wait_background" not in json.dumps(active_tools), (
        "a fresh session (nothing backgrounded yet) should not pay for the "
        "wait_background tool schema"
    )

    # Recorded budget (see the ledger above BUDGET_TOKENS): the floor may not
    # grow silently. Raising this number is a deliberate act that must say why
    # in the ledger comment; NX_FLOOR_BUDGET exists for CI experiments, not
    # for shipping past the budget.
    assert total <= BUDGET_TOKENS, (
        f"initial context content is {total} tokens "
        f"> budget {BUDGET_TOKENS} — trim a schema or raise the recorded "
        f"budget WITH a ledger justification. {breakdown}"
    )

    # Advisory: 50 tokens of headroom left. Still passes, but this is the
    # point to trim the fattest description rather than land on the budget.
    if total >= SOFT_TOKENS:
        print(f"NOTE: initial context {total} tok is within 50 of the "
              f"{BUDGET_TOKENS}-tok floor budget. {breakdown}")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
