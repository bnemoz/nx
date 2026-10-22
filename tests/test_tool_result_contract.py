"""Tool-result type contract (audit P3-2).

Nothing used to enumerate `nx._dispatch.DISPATCH` and assert what each body
returns: the per-tool isinstance checks lived scattered across the per-tool
files, so a body that started returning `bytes`/`dict`/`None` first surfaced
deep inside the turn loop (or as a malformed session row), with the failing
test some unrelated file. This is the missing single oracle, in the repo's
schema-invariant idiom:

1. **Annotation contract** (every entry, no exclusions-by-omission): each
   registered body's declared return annotation must be `str`, except names
   in `_DECLARED_NON_STR` — whose deviation is *stated here with its reason*
   and pinned against the live annotation, so an entry can neither silently
   drop its annotation nor quietly change shape.
2. **Anti-vacuity**: the exception table must have no dead keys, and the
   probe/skip tables must cover every DISPATCH name exactly once. A DISPATCH
   entry nobody probed and nobody declared is a failure — that is the whole
   point of the file.
3. **isinstance probes** for the subset of bodies that answer with a real
   result from a *harmless* stub call (empty/garbage args, or paths inside
   pytest's tmp dir): they must actually return the declared shape. Tools
   with side effects (writes outside the test tmp dir, subprocesses, network,
   sub-agent loops) are DECLARED-SKIPPED with a reason, never invoked.

Offline, no threads, no network; every probe is an error path or a read.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nx  # noqa: E402
from nx._dispatch import DISPATCH  # noqa: E402

_EDIT_TOOLS_SRC = os.path.join(os.path.dirname(nx.__file__), "_edit_tools.py")


def _ret_ann(fn) -> str | None:
    """Declared return annotation as a normalized string, or None.

    nx modules use `from __future__ import annotations`, so the stored value
    is the source text (`'str'`, `'str | Any'`); a module without the future
    import stores the object (`str`). Normalize both to one spelling.
    """
    raw = getattr(fn, "__annotations__", {}).get("return", None)
    if raw is None:
        return None
    text = raw if isinstance(raw, str) else getattr(raw, "__name__", str(raw))
    return text.replace("'", "").replace('"', "").replace(" ", "")


# Names whose bodies legitimately do NOT declare `-> str`, with the reason.
# Every entry is re-pinned against the LIVE annotation by
# test_non_str_exceptions_still_match_their_annotation, so this table cannot
# rot into a place where a body dropped its annotation.
_DECLARED_NON_STR = {
    # Content-parts result (the tool result a client sends back as a message
    # array): [text part, image_url part], or an ERROR str on every failure
    # path. `Any` is the honest spelling — typing has no "str | parts".
    "read_image": "Any",
    # Project (MCP-style) tools are third-party: their declared shape is
    # "str, or whatever the remote schema's content parts are".
    "call_project_tool": "str|Any",
    # write_file / edit_file / multi_edit are `_fused_body` wrappers around
    # annotated `-> str` bodies (nx/_edit_tools.py). The wrapper must accept
    # `then_run` as a real parameter, so it declares `(*pos, then_run, **kw)`
    # and inherits no return annotation. Their contract is pinned twice below
    # (source-level annotation of the bodies + the wrapper's then_run shape),
    # because the wrapper is what run_tool actually calls.
    "write_file": None,
    "edit_file": None,
    "multi_edit": None,
    # `search_tools` is registered as a lambda wrapper (so the tiered registry
    # stays lazily looked up); a lambda cannot carry a return annotation, so
    # the contract is pinned through the wrapped body by
    # test_lambda_entry_wraps_an_annotated_body.
    "search_tools": None,
}

_FUSION_WRAPPED = ("write_file", "edit_file", "multi_edit")


def test_dispatch_table_is_not_empty():
    # Anti-vacuity for everything below: if DISPATCH ever moved/renamed, this
    # file would otherwise pass by iterating nothing.
    assert isinstance(DISPATCH, dict)
    assert len(DISPATCH) >= 25, f"suspiciously small DISPATCH: {sorted(DISPATCH)}"


def test_every_body_is_callable():
    for name, fn in DISPATCH.items():
        assert callable(fn), f"DISPATCH[{name!r}] is not callable: {fn!r}"


def test_every_body_declares_str_or_a_declared_exception():
    for name, fn in sorted(DISPATCH.items()):
        ann = _ret_ann(fn)
        if name in _DECLARED_NON_STR:
            assert ann == _DECLARED_NON_STR[name], (
                f"DISPATCH[{name!r}] annotation changed shape: declared "
                f"{_DECLARED_NON_STR[name]!r}, found {ann!r}"
            )
            continue
        assert ann == "str", (
            f"DISPATCH[{name!r}] must declare `-> str` (it is the tool-result "
            f"contract run_tool's sanitize and the transcript depend on); "
            f"found {ann!r}. If the shape really changed, add the name to "
            f"_DECLARED_NON_STR in tests/test_tool_result_contract.py with a "
            f"reason — do not delete the annotation."
        )


def test_lambda_entry_wraps_an_annotated_body():
    # `search_tools` is registered as a lambda (so the tiered registry stays
    # lazily looked up); a lambda cannot carry an annotation, so the contract
    # for this one entry is pinned through the body it wraps.
    fn = DISPATCH["search_tools"]
    wrapped = getattr(fn, "__globals__", {}).get("search_tools")
    assert wrapped is not None, "lambda's globals lost `search_tools`"
    assert _ret_ann(wrapped) == "str", (
        f"the body behind DISPATCH['search_tools'] must declare `-> str`, "
        f"found {_ret_ann(wrapped)!r}"
    )


def test_non_str_exceptions_still_match_their_annotation():
    # The exception table's own anti-vacuity: no dead keys (a name that went
    # back to `-> str`, or left DISPATCH, must be pruned here), and no name
    # added to the table that actually does declare `-> str`.
    for name, expected in sorted(_DECLARED_NON_STR.items()):
        assert name in DISPATCH, (
            f"_DECLARED_NON_STR declares {name!r}, which DISPATCH no longer "
            f"registers — prune the entry"
        )
        ann = _ret_ann(DISPATCH[name])
        assert ann == expected, f"DISPATCH[{name!r}] annotation is {ann!r}"
        if name not in _FUSION_WRAPPED and name != "search_tools":
            assert ann != "str", (
                f"{name!r} declares `-> str` again — remove it from "
                f"_DECLARED_NON_STR so the general rule covers it"
            )


def test_fusion_wrappers_keep_the_then_run_shape():
    # The three edit tools reach run_tool as `_fused_body` wrappers; if a
    # refactor swaps in the bare body, the schema's `then_run` property would
    # describe a parameter the body rejects. Pin the wrapper identity here.
    for name in _FUSION_WRAPPED:
        fn = DISPATCH[name]
        params = set(getattr(fn, "__wrapped_signature_params__", ()))
        assert "then_run" in params, (
            f"DISPATCH[{name!r}] is no longer a _fused_body wrapper "
            f"(params: {sorted(params)})"
        )


def test_fused_edit_bodies_are_annotated_str_at_the_source():
    # The wrappers carry no return annotation, so pin the annotation of the
    # bodies they wrap by parsing the defining module — the docs-parity
    # idiom (regex over source), no import gymnastics.
    with open(_EDIT_TOOLS_SRC, encoding="utf-8") as f:
        src = f.read()
    for name in _FUSION_WRAPPED:
        m = re.search(rf"^def {name}\((?:.|\n)*?\) -> (str)\b", src, re.M)
        assert m, (
            f"nx/_edit_tools.py::{name} must declare `-> str` — it backs "
            f"DISPATCH[{name!r}] through the _fused_body wrapper, which "
            f"cannot carry an annotation of its own"
        )


# --------------------------------------------------------------------------
# Runtime probes: bodies that answer harmlessly from stub args.
# --------------------------------------------------------------------------

def _png(path: str) -> None:
    """Smallest thing `_images` accepts: ext-map hit + non-empty bytes. The
    payload never leaves the probe (we assert shape, never contents)."""
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)


# name -> callable(DISPATCH) -> value. Every entry is an error path or a read
# of a file the test itself created under tmp_path. Nothing here writes
# outside tmp_path, spawns a process, opens a socket, or starts an agent.
_SAFE_PROBES = {
    "read_file": lambda d, w: d["read_file"](""),
    "list_dir": lambda d, w: d["list_dir"](os.path.join(w, "no-such-dir")),
    "search_files": lambda d, w: d["search_files"]("", path=w),
    "git_status": lambda d, w: d["git_status"](w),
    "git_log": lambda d, w: d["git_log"](w),
    "git_diff": lambda d, w: d["git_diff"](w),
    "edit_file": lambda d, w: d["edit_file"](""),
    "multi_edit": lambda d, w: d["multi_edit"](os.path.join(w, "seed.txt"), []),
    "sed_edit": lambda d, w: d["sed_edit"]("a", "b", os.path.join(w, "nope.txt")),
    "delete_file": lambda d, w: d["delete_file"](os.path.join(w, "nope.txt")),
    "kill_background": lambda d, w: d["kill_background"]("not-a-pid"),
    "wait_background": lambda d, w: d["wait_background"]("not-a-pid"),
    # `search_tools` is the lambda above — covered by its own body-annotation
    # test, and its probe here checks the runtime shape of the lambda itself.
    "search_tools": lambda d, w: d["search_tools"](""),
    "search_project_tools": lambda d, w: d["search_project_tools"](""),
    "call_project_tool": lambda d, w: d["call_project_tool"]("no-such-tool", {}),
    "run_tools": lambda d, w: d["run_tools"](""),
    "obs_recall": lambda d, w: d["obs_recall"](""),
    "plan_edit": lambda d, w: d["plan_edit"](),
    "memory_search": lambda d, w: d["memory_search"](""),
    "recall_lessons": lambda d, w: d["recall_lessons"](""),
    "duckdb": lambda d, w: d["duckdb"](""),
}

# Bodies NOT called, and why. Declared rather than absent, so
# test_probe_and_skip_tables_cover_dispatch makes the gap visible.
_DECLARED_SKIPS = {
    # write_file is NOT probed: every non-error path passes through
    # _confirm()'s interactive approval prompt (stdin under pytest is a
    # tripwire, and forcing it True would make a contract test write files).
    # Its shape is pinned by the two source-level tests above.
    "write_file": "approval prompt on every non-error path; shape pinned at source",
    "run_bash": "spawns a real shell",
    "run_tests": "spawns a real pytest/xdist run",
    "lint_check": "spawns a real ruff run",
    "git_commit": "stages and commits; needs a scratch repo",
    "todo_write": "mutates the session's todo list",
    "memory_write": "writes the memory store",
    "memory_delete": "deletes from the memory store",
    "spawn_agent": "runs a sub-agent turn loop",
    "critique": "runs a sub-agent turn loop",
    "web_fetch": "network",
}


@pytest.fixture
def probe_workspace(tmp_path, monkeypatch):
    """A scratch dir + a private NX_HOME so the few bodies that touch the
    stores (memory_search, obs_recall) land in tmp dirs, never in ~/.nx. The
    autouse reset restores the env afterwards; this mirrors the file-level
    pattern the other store-touching tests use."""
    monkeypatch.setenv("NX_HOME", str(tmp_path / "nxhome"))
    monkeypatch.setenv("NX_SESSIONS_DIR", str(tmp_path / "sessions"))
    (tmp_path / "seed.txt").write_text("hello\n", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("name", sorted(_SAFE_PROBES))
def test_safe_probe_returns_the_declared_shape(name, probe_workspace):
    value = _SAFE_PROBES[name](DISPATCH, probe_workspace)
    assert isinstance(value, str), (
        f"DISPATCH[{name!r}] returned {type(value).__name__} (expected str): "
        f"{value!r}"
    )


def test_read_image_returns_str_or_content_parts(probe_workspace):
    # read_image is the one declared non-str shape, so it gets the real
    # isinstance pin rather than an annotation-only promise.
    bad = DISPATCH["read_image"]("")
    assert isinstance(bad, str) and bad.startswith("ERROR"), bad

    good = os.path.join(probe_workspace, "dot.png")
    _png(good)
    parts = DISPATCH["read_image"](good)
    assert isinstance(parts, list) and len(parts) == 2, parts
    assert parts[0]["type"] == "text" and isinstance(parts[0]["text"], str)
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_probe_and_skip_tables_cover_dispatch():
    # Every registered tool is either probed or declared-skipped, exactly one
    # way. This is the assertion that keeps the file honest as tools are
    # added: a new tool with no probe forces a deliberate decision.
    probed = set(_SAFE_PROBES) | {"read_image"}
    skipped = set(_DECLARED_SKIPS)
    unknown = set(DISPATCH) - probed - skipped
    assert not unknown, (
        f"tools with neither a safe probe nor a declared skip reason: "
        f"{sorted(unknown)} — call it safely or record why it can't be"
    )
    stale = (probed | skipped) - set(DISPATCH)
    assert not stale, f"probe/skip tables name unregistered tools: {sorted(stale)}"
    assert not (probed & skipped), f"declared both probed and skipped: {sorted(probed & skipped)}"


def test_skip_reasons_are_declared():
    # Weak spot check on the skip table: the reasons are prose, but the names
    # must be tools that genuinely could mutate (i.e. registered). Pinned so
    # the table can't become a dumping ground for "couldn't be bothered".
    for name, reason in sorted(_DECLARED_SKIPS.items()):
        assert name in DISPATCH
        assert reason.strip() and len(reason) > 3, f"{name}: empty skip reason"
