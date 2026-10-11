#!/usr/bin/env python3
"""One-shot: regenerate the README generated-commands block from the registry."""
import os
import re
import sys
import tempfile

os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_HOME"] = tempfile.mkdtemp()
os.environ["NX_SESSIONS_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nx import _cli  # noqa: E402

table = _cli._commands_markdown()
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
src = open(path, encoding="utf-8").read()
new = re.sub(r"(<!-- BEGIN GENERATED COMMANDS[^\n]*-->\n).*?(\n<!-- END GENERATED COMMANDS -->)",
             lambda m: m.group(1) + table + m.group(2), src, flags=re.DOTALL)
assert new != src, "markers not found"
open(path, "w", encoding="utf-8").write(new)
print("README command table regenerated")
