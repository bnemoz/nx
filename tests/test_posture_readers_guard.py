"""Grep guard (docs/YOLO_POSTURE_SPEC.md §8): every direct `st.YOLO` /
`st.APPROVE_LEVEL` attribute access in `nx/` must be on the explicit allowlist
of PARENT-facing sites. A new direct read is a decision-path read that skipped
`_state._posture()` — inside a sub-agent body it would silently resolve the
PARENT's posture instead of the agent's.

The allowlist is the census of §3.3's "stay on the process globals" list: the
UI/display reads, the posture WRITERS (/yolo, /approval, the CLI init, `nx
eval`, toggle_yolo) and the Shift+Tab handler. Everything else routes through
`st._posture()`.

The scan is AST-based (not grep): a comment or docstring that mentions
`st.YOLO` is documentation, not a read of the posture.
"""

import ast
import os

# path -> why this file may touch the posture globals directly (parent-facing).
_ALLOWED_FILES = {
    # --- posture writers: these ARE the parent-posture store -----------------
    "nx/__init__.py": "CLI init (--yolo/--approval/NX_APPROVAL) + the proxy",
    "nx/_eval_cmd.py": "`nx eval` saves/restores the parent posture per scenario",
    "nx/_approval.py": "_apply_approval / toggle_yolo / Shift+Tab (parent writer)",
    "nx/_repl_settings_cmds.py": "/yolo and /approval (parent writer)",
    # --- parent-facing display reads (the UI tells the user THEIR posture) ---
    "nx/_chatbox.py": "chatbox border/footer read the parent posture",
    "nx/_repl.py": "startup banner + prompt hint show the parent posture",
}

# The store itself: it DEFINES the globals (module-level names, not attribute
# access) and hosts `_posture()`. Exempt from the stale-entry check below,
# since there is no attribute read to find.
_STORE_FILES = frozenset({"nx/_state.py"})

# Aliases modules use for `nx._state` (and `nx._state.X` attribute chains).
_ALIASES = frozenset({"st", "_st", "state", "_state", "st_", "nx", "self"})
_POSTURE_NAMES = frozenset({"YOLO", "APPROVE_LEVEL"})


def _nx_root():
    """The LIVE package directory (resolved through the import, not through this
    test file's path — under worktrees the two disagree)."""
    import nx
    return os.path.dirname(os.path.abspath(nx.__file__))


def _hits(path):
    """[(lineno, source)] for every `<alias>.YOLO` / `<alias>.APPROVE_LEVEL`
    attribute access in the file (comments and docstrings are ignored)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in _POSTURE_NAMES):
            continue
        base = node.value
        name = None
        if isinstance(base, ast.Name):
            name = base.id
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            name = base.value.id          # e.g. nx._state.YOLO
        if name in _ALIASES:
            found.append((node.lineno, ast.unparse(node).strip()))
    return found


def _iter_nx_files():
    root = _nx_root()
    base = os.path.dirname(root)
    for dirpath, _dirs, files in os.walk(root):
        _dirs[:] = [d for d in _dirs if not d.startswith(".")]
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                yield os.path.relpath(path, base).replace(os.sep, "/"), path


def test_only_allowlisted_modules_touch_the_posture_globals():
    offenders = {}
    for rel, path in _iter_nx_files():
        if rel in _ALLOWED_FILES or rel in _STORE_FILES:
            continue
        for lineno, src in _hits(path):
            offenders.setdefault(rel, []).append(f"{rel}:{lineno}: {src}")
    assert not offenders, (
        "direct posture-global access outside the parent-facing allowlist — "
        "route decision reads through `_state._posture()` (see "
        "docs/YOLO_POSTURE_SPEC.md §3.3/§8). Either migrate the new site or add "
        "its file to _ALLOWED_FILES with a note saying why it is parent-facing.\n"
        + "\n".join(line for lines in offenders.values() for line in lines))


def test_allowlist_has_no_stale_entries():
    """Every allowlisted file must still exist and still touch a global —
    otherwise the allowlist rots into a blanket exemption."""
    checked = 0
    for rel, why in _ALLOWED_FILES.items():
        path = os.path.join(os.path.dirname(_nx_root()), rel)
        assert os.path.exists(path), f"allowlisted file vanished: {rel} ({why})"
        assert _hits(path), (
            f"{rel} is allowlisted ({why}) but no longer touches "
            "YOLO/APPROVE_LEVEL — drop it from _ALLOWED_FILES")
        checked += 1
    assert checked == len(_ALLOWED_FILES)
    for rel in _STORE_FILES:
        assert os.path.exists(os.path.join(os.path.dirname(_nx_root()), rel))
