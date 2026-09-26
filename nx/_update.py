"""`nx update` — self-update: detect how nx was installed, run the matching
upgrade command, and report the version delta.

Like `nx config`, this verb never touches the configured endpoints: it is
purely a local/package-manager operation (its only network access is the
package index query made by the underlying upgrade command itself).
nx/__init__.py skips switch_source()'s startup probes when argv[1] == "update"
so the verb stays instant against a dead or unconfigured endpoint.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from ._ansi import DIM, RESET

_USAGE = """\
usage: nx update              upgrade nx in place via its package manager
       nx update --check      print the installed version and install method,
                              without upgrading"""

# Install-method detectors, checked in order. Editable installs (uv pip
# install -e ., pip -e) are reported but never "upgraded" — the upgrade is
# `git pull` in the repo, which nx should not run silently.
_UV_TOOL_PREFIX = os.path.join("uv", "tools")


def _install_method() -> tuple[str, str]:
    """Classify how the running nx was installed.

    Returns (method, detail) where method is one of "uv-tool", "pipx",
    "editable", "pip" — the order matters: uv-tool/pipx installs also look
    like pip installs from site-packages, so the tool-specific markers are
    checked first."""
    try:
        import nx  # noqa: F401 — resolve the real install path
        root = os.path.dirname(os.path.dirname(os.path.abspath(nx.__file__)))
    except Exception:
        root = ""
    # uv tool: <uv tools dir>/nx/... — the parent path contains uv/tools
    if _UV_TOOL_PREFIX in root.replace(os.sep, "/"):
        return "uv-tool", root
    # pipx: ~/.local/pipx/venvs/nx/...
    if os.path.join("pipx", "venvs") in root.replace(os.sep, "/"):
        return "pipx", root
    # editable: nx resolves outside site-packages (a repo checkout)
    if "site-packages" not in root.replace(os.sep, "/"):
        return "editable", root
    return "pip", root


def _upgrade_command(method: str) -> list[str]:
    if method == "uv-tool":
        return ["uv", "tool", "upgrade", "nx"]
    if method == "pipx":
        return ["pipx", "upgrade", "nx"]
    # plain pip install --upgrade works for both pip and (as a last resort)
    # editable checkouts' venv; editable gets a guidance message instead.
    return [sys.executable, "-m", "pip", "install", "--upgrade", "nx"]


def _cmd_check(_args: list[str]) -> int:
    from ._cli import _cli_version
    method, root = _install_method()
    print(f"nx {_cli_version()}")
    print(f"  install: {method}")
    print(f"  prefix:  {root or '(unknown)'}")
    if method == "editable":
        print("  editable install — upgrade with `git pull` in the repo, "
              "not `nx update`")
    return 0


def _cmd_update() -> int:
    from ._cli import _cli_version
    method, _root = _install_method()
    if method == "editable":
        print("nx update: this is an editable install — run `git pull` in the "
              "repo instead")
        return 1
    cmd = _upgrade_command(method)
    tool = cmd[0]
    if shutil.which(tool) is None and tool != sys.executable:
        print(f"nx update: {tool!r} not found on PATH — how was nx installed?")
        return 1
    old = _cli_version()
    print(f"{DIM}  upgrading via {tool} ({method} install)...{RESET}")
    try:
        rc = subprocess.call(cmd)
    except OSError as e:
        print(f"nx update: {tool} failed to start: {e}")
        return 1
    if rc != 0:
        print(f"nx update: {tool} exited {rc}")
        return 1
    new = _cli_version()
    if new == old:
        print(f"nx {old} — already up to date")
    else:
        print(f"nx {old} -> {new} — restart any running nx session "
              "to pick it up")
    return 0



def _cli_update(args: list[str]) -> int:
    """`nx update` verb — returns the process exit code (0 ok, 1 failure,
    2 usage)."""
    if args and args[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0
    rest = [a for a in args if not a.startswith("-")]
    flags = [a for a in args if a.startswith("-")]
    if rest:
        sys.stderr.write(f"nx update: unexpected argument {rest[0]!r}\n"
                         + _USAGE + "\n")
        return 2
    if "--check" in flags or "check" in args:
        return _cmd_check(flags)
    if flags:
        sys.stderr.write(f"nx update: unknown flag {flags[0]!r}\n"
                         + _USAGE + "\n")
        return 2
    return _cmd_update()
