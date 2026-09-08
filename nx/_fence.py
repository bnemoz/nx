"""The hard write fence (2.1): a deterministic, approval-independent check on
every path-carrying tool call, enforced in run_tool() *before* the tool body.

Why a fence when there is already a risk classifier and an approval prompt:
both are judgement. The classifier is a model call keyed off the approval
ACTION string (`_extract_action_path` only parses "edit/delete/write <path>"
prefixes — sed_edit/apply_patch had no path judgement at all), and YOLO
skips the prompt entirely. A committed `config -> ~/.ssh/authorized_keys`
symlink converts an approved in-project edit into an out-of-project write,
because `_atomic_write` deliberately follows symlinks (link-farm dotfile
management) and the judgement realpath'd at approval time while the write
resolves fresh (TOCTOU). A fence is a boundary, not an opinion: roots are
computed by realpath, candidate paths are realpath'd at the same moment the
body would resolve them, and the answer is in/out — no model, no prompt.

Downstream layers stay: the classifier shapes approval (low/medium/high),
the user confirms, allow-rules still short-circuit. The fence only decides
where a write may land, the one thing judgement must never be asked.

Env:
  NX_WRITE_ROOTS   os.pathsep-separated extra absolute roots, empty default
  NX_WRITE_FENCE   0 disables the check entirely (documented escape hatch)
"""
from __future__ import annotations

import os
from typing import Any

from . import _state as st
from ._env import _env_int

# Tools whose primary write target is `path` (same list run_tool() already
# special-cases for undo snapshots and cache invalidation).
_TOOL_PATH_TOOLS = ("write_file", "edit_file", "multi_edit", "delete_file")

# Tools with extra write surfaces beyond `path`.
#   sed_edit: `path` is a glob — the body expands it against the real cwd,
#     so fence each matched file plus the literal (pre-expansion) arg.
#   apply_patch: `diff` headers name files relative to `path`; also fence
#     each extracted +++/--- target joined against the patch base.
#   run_bash: no path args to fence; the fence would need a full shell
#     parser and produce either bypasses or false refusals. The sandbox tier
#     (P3-1) is the correct future layer for it — one important consequence:
#     a model can reach ANY path with run_bash. The classifier plus approval
#     remain the protection for shell; the fence covers the structured write
#     tools where a deterministic check is actually possible.
_SHELL_ONLY = "run_bash"

def _writable_roots() -> list[str]:
    """Realpath'd writable roots: the project root (git root or cwd), plus
    NX_WRITE_ROOTS extras, plus the session home (~/.nx or NX_HOME) so nx's
    own sinks (sessions, memories, images, overflow, agents) stay writable
    without every caller having to add them by hand."""
    roots = [os.path.realpath(getattr(st, "PROJECT_ROOT", None) or os.getcwd())]
    session_home = getattr(st, "SESSION_HOME", None) or os.path.expanduser(
        os.environ.get("NX_HOME", "~/.nx"))
    roots.append(os.path.realpath(session_home))
    raw = os.environ.get("NX_WRITE_ROOTS", "")
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        part = os.path.expanduser(part)
        if not os.path.isabs(part):
            part = os.path.join(roots[0], part)
        roots.append(os.path.realpath(part))
    return roots


def _under_any(path: str, roots: list[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except (OSError, ValueError):
            continue
    return False


# Committed-symlink targets that convert an approved in-project edit into an
# out-of-project write: the classic dotfile link farm, ~/.ssh authorized_keys
# chroots, shell rc files, and git hooks (self-executing on commit/checkout).
# The target being IN the denylist wins even when the link itself sits inside
# a root — _atomic_write writes THROUGH symlinks by design, so the real file
# is what gets modified and the real file is what gets judged.
_DENIED_TARGET_REALS = frozenset({
    os.path.realpath(os.path.expanduser("~/.ssh")),
    os.path.realpath(os.path.expanduser("~/.aws")),
    os.path.realpath(os.path.expanduser("~/.gnupg")),
    os.path.realpath(os.path.expanduser("~/.kube")),
    os.path.realpath(os.path.expanduser("~/.docker")),
    os.path.realpath(os.path.expanduser("~/.config/gcloud")),
    os.path.realpath(os.path.expanduser("~/.netrc")),
    os.path.realpath(os.path.expanduser("~/.gitconfig")),
})
_DENIED_TARGET_BASENAMES = {
    ".bashrc", ".bash_profile", ".profile", ".zshrc", ".zprofile",
    ".zshenv", ".cshrc", ".tcshrc", ".gitconfig", ".gitignore_global",
    ".netrc", ".wgetrc", ".curlrc", ".ssh", ".aws", ".gnupg", ".kube",
    ".docker", ".npmrc", ".pypirc", ".netrc", ".hgrc", ".pgpass",
    "authorized_keys", "authorized_keys2", "config", "id_rsa", "id_ed25519",
    "id_ecdsa", "identity",
}
_DENIED_TARGET_SUFFIXES = (".pem", ".key",)


def _denied_target(path: str) -> bool:
    """True for real targets that stay fenced even from inside a root:
    credential dirs/files and the home shell rc files."""
    if path in _DENIED_TARGET_REALS:
        return True
    base = os.path.basename(path)
    low = base.lower()
    if low in _DENIED_TARGET_BASENAMES:
        return True
    if low.endswith(_DENIED_TARGET_SUFFIXES):
        return True
    return False


def _fence_candidates(name: str, args: dict[str, Any]) -> list[tuple[str, str]]:
    """(display, realpath) pairs every write surface a tool call could touch.
    Ordered list of (candidate path as written, realpath)."""
    cands: list[tuple[str, str]] = []

    def add(raw: Any) -> None:
        if not isinstance(raw, str) or not raw.strip():
            return
        cands.append((raw, os.path.realpath(os.path.expanduser(raw))))

    if name in _TOOL_PATH_TOOLS:
        add(args.get("path"))
    elif name == "sed_edit":
        # The body globs `path` itself; fence the literal and every file the
        # glob would match (same _BINARY_EXTS skip is the body's business —
        # we fence before the body, so a broad glob is judged on what it
        # would touch, not on what the body later filtered out).
        import glob as _glob
        raw = args.get("path")
        if isinstance(raw, str) and raw.strip():
            add(raw)
            if _glob.has_magic(raw):
                try:
                    for m in sorted(_glob.glob(raw, recursive=True)):
                        add(m)
                    # A globlexists-but-no-file case is fenced via the literal
                    # above; no matches at all means nothing to write.
                except Exception:
                    pass
    elif name == "apply_patch":
        base = args.get("path") or "."
        add(base)
        diff = args.get("diff")
        if isinstance(diff, str):
            import re as _re
            for m in _re.finditer(r"^\+\+\+ (?:b/)?(\S+)", diff, _re.M):
                tgt = m.group(1)
                if tgt == "/dev/null":
                    continue
                add(os.path.join(base, tgt))
    return cands


def check_write_fence(name: str, args: dict[str, Any]) -> str:
    """The fence verdict for a tool call: "" = allow, else the ERROR refusal.

    Never raises, never prompts, never calls the model. Called from run_tool()
    for every tool with a structured write surface (the `path` tools,
    sed_edit, apply_patch). run_bash is deliberately NOT fenced here — no
    static analysis covers shell without either bypasses or false refusals;
    it keeps its classifier+approval story (see P3-1 for the real fix).
    """
    if not _env_int("NX_WRITE_FENCE", 1):
        return ""
    roots = _writable_roots()
    outside: list[str] = []
    denied: list[str] = []
    for raw, real in _fence_candidates(name, args):
        if _denied_target(real):
            denied.append(raw)
            continue
        if _under_any(real, roots):
            continue
        outside.append(raw)
    if outside:
        return (
            "ERROR: write fence: " + ", ".join(sorted(set(outside)))
            + " resolves outside every writable root (project root "
            + roots[0] + (", plus NX_WRITE_ROOTS" if len(roots) > 2 else "")
            + "; session home is always writable). No part of the call ran."
            " Write inside the project, add a root via NX_WRITE_ROOTS (colon-"
            "separated, ~ expands), or set NX_WRITE_FENCE=0 to disable the"
            " fence for this environment."
        )
    if denied:
        return (
            "ERROR: write fence: refusing to write through a path that lands in a"
            " protected location even though it may sit inside a writable root"
            " (committed symlink, mount point, or direct write): " + ", ".join(sorted(set(denied)))
            + ". nx writes THROUGH symlinks (link-farm dotfile management), so the"
            " link's target is what gets modified. Ask the user to perform this"
            " write themselves, or add the target directory via NX_WRITE_ROOTS"
            " if it is genuinely meant to be writable."
        )
    return ""
