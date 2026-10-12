"""The duckdb tool (0.87.0): analytical SQL over files, in one fixed shape.

DuckDB is the better default for data-science sessions — it reads CSV/
Parquet/JSON directly from SQL (``SELECT * FROM 'data/*.parquet'``), runs
in-process, and outperforms pandas on the aggregations a coding session
actually does. This tool wraps the DuckDB CLI so the model can reach it
without free-text shell: the argv is always ``duckdb -batch -noheader -box
<cfile>`` (or ``duckdb <db> ...``), never an arbitrary command — that is
run_bash's job.

Approval contract: pure ``SELECT/WITH/DESCRIBE/SHOW/EXPLAIN/PRAGMA/`` etc.
read-only queries skip approval entirely (like search_files/git_status);
anything that mutates (CREATE/INSERT/UPDATE/DELETE/COPY/DROP/ATTACH/INSTALL/
EXPORT/BEGIN/CALL/SET/LOAD/ALTER/ATTACH) goes through ``_confirm()`` with the
action string ``run duckdb: <first line of sql>`` — nx/_risk.py's local
fast path scores exactly that shape ("contained SQL run, argv is fixed")
so a mutating query never costs a classifier round-trip.

Session memory (the idea worth stealing from duckdb-skills): a plain SQL
notes file, ``NX_DUCKDB_NOTES`` (default ``.nx-duckdb-notes.sql`` under the
project root), is auto-loaded as a ``-init`` prologue before every query so
macro/view/type definitions from earlier sessions are present — recall
without a config file. The model appends to it with write_file/edit_file.

Identity resolution (the idea worth stealing from the "context engine"
pattern): name/logon filters are the top cause of silently-empty org-data
results — "Peter" never matches ``author = 'puery'``. The model should
resolve human names to canonical ids BEFORE filtering, and record what it
learned in the notes file so later sessions skip the step:

    -- id-map: peter -> puery; anisha -> anisha.r

A comment line is enough (the model reads the whole prologue each turn);
views built on top of a mapping table also work. Resolve-then-filter, and
say so when a resolved name had no match instead of returning zero rows.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from . import _state as st
from ._env import _env_int

# Hard wall-clock ceiling for one duckdb call. Analytical queries can be slow
# on real data, but a wedged CLI holds the whole turn; run_bash (which
# backgrounds) is the named escape hatch for genuinely long jobs.
_DEFAULT_TIMEOUT = 300

# Result folding: a wide/long result table is capped and the full output
# spills to a temp file for read_file, mirroring _testrunner.
_MAX_RESULT_LINES = 120

# Output format for the CLI box renderer. `-box` gives aligned tables the
# model reads well; CSV (`format="csv"`) switches to -csv for downstream
# parsing/piping by the model.
_FORMAT_FLAGS = {
    "box": ["-box"],
    "csv": ["-csv"],
    "json": ["-json"],
}

# Statement types that never mutate data or state — safe to auto-run without
# an approval prompt (the whole SQL text is inspected, so a CTE-leading WITH
# is covered and a sneaky `SELECT ...; DROP TABLE x` is NOT).
_READ_ONLY_RE = re.compile(
    r"\b(select|with|describe|show|explain|pragma|summarize|from|values|"
    r"table\s+\w)\b",
    re.IGNORECASE,
)
# Anything on this list anywhere in the SQL makes the whole call mutating —
# cheaper and safer than parsing SQL properly.
_MUTATING_RE = re.compile(
    r"\b(insert|update|delete|create|drop|alter|copy|export|import|attach|"
    r"detach|install|load|begin|commit|rollback|checkpoint|vacuum|analyze|"
    r"set|reset|call|checkpoint)\b",
    re.IGNORECASE,
)


def _notes_path() -> str:
    """The auto-loaded session-notes SQL file (created on demand)."""
    default = os.path.join(st.PROJECT_ROOT or ".", ".nx-duckdb-notes.sql")
    return os.environ.get("NX_DUCKDB_NOTES") or default


def _is_read_only(sql: str) -> bool:
    """True when *sql* provably cannot mutate: every statement is a read.

    Cheap two-sided check — a read keyword must be present and no mutating
    keyword anywhere (in comments or string literals included). Not a SQL
    parser; the conservative direction is to demand approval.
    """
    return bool(_READ_ONLY_RE.search(sql)) and not _MUTATING_RE.search(sql)


def _fold_result(out: str) -> tuple[str, bool]:
    """Cap a result table to _MAX_RESULT_LINES; report whether folding
    happened (the caller spills the full text and points read_file at it)."""
    lines = out.splitlines()
    if len(lines) <= _MAX_RESULT_LINES:
        return out, False
    head, tail = lines[:100], lines[-15:]
    folded = len(lines) - 115
    return ("\n".join(head)
            + f"\n... [{folded} result lines folded — full output below]\n"
            + "\n".join(tail), True)


def duckdb_query(sql: str, database: str = "", format: str = "box",
                 **_: Any) -> str:
    """Run analytical SQL with the DuckDB CLI and return the result table.

    Read-only SQL (SELECT/WITH/DESCRIBE/...) runs without approval; anything
    mutating asks first. When the CLI is absent the result names the install
    URL instead of failing opaque.
    """
    if not sql or not sql.strip():
        return "ERROR: duckdb: `sql` is empty — pass the query text"
    binary = shutil.which("duckdb")
    if not binary:
        return ("ERROR: duckdb: 'duckdb' not found on PATH — install it with "
                "`curl https://install.duckdb.org | sh` "
                "(https://duckdb.org/docs/installation)")
    fmt = (format or "box").strip().lower()
    if fmt not in _FORMAT_FLAGS:
        return (f"ERROR: duckdb: unknown format {format!r} — "
                "expected box, csv, or json")
    argv = [binary, "-batch", "-noheader"] + _FORMAT_FLAGS[fmt]
    notes = _notes_path()
    if notes and os.path.isfile(notes):
        argv += ["-init", notes]
    if database:
        db = os.path.expanduser(database)
        if not os.path.isfile(db):
            return (f"ERROR: duckdb: database file not found: {db} — omit "
                    "`database` to query files directly (read_parquet/"
                    "read_csv/read_json) or create it first")
        argv.append(db)
    # SQL rides a temp file passed with -f: no quoting/injection surface, and
    # multi-statement scripts (a CTE plus a follow-up SELECT) work as-is.
    # (-f is required since DuckDB 1.5.x — a positional file is opened as a
    # DATABASE there, which fails with "not a valid DuckDB database file".)
    fd, sql_path = tempfile.mkstemp(prefix="nx-duckdb-", suffix=".sql")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(sql if sql.endswith("\n") else sql + "\n")
        argv += ["-f", sql_path]
        if not _is_read_only(sql):
            from ._bash_jobs import _confirm  # lazy: avoid an import cycle
            first = sql.strip().splitlines()[0][:120]
            if not _confirm(f"run duckdb: {first}"):
                return "DENIED by user"
        timeout = _env_int("NX_DUCKDB_TIMEOUT", _DEFAULT_TIMEOUT)
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=timeout or None,
                               cwd=st.PROJECT_ROOT or None)
        except subprocess.TimeoutExpired:
            return (f"[duckdb] TIMED OUT after {timeout}s — run the query "
                    "with run_bash (backgrounds) to investigate instead")
        except (OSError, FileNotFoundError):
            return (f"ERROR: duckdb: {binary!r} not found on PATH — install it "
                    "with `curl https://install.duckdb.org | sh`")
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0 and not out.strip():
            out = f"(duckdb exited {r.returncode} with no output)"
        body, folded = _fold_result(out)
        spill = ""
        if folded and out.strip():
            try:
                fd2, spill_path = tempfile.mkstemp(
                    prefix="nx-duckdb-result-", suffix=".txt")
                with os.fdopen(fd2, "w", encoding="utf-8",
                               errors="replace") as f:
                    f.write(out)
                spill = (f"\nfull result ({len(out.splitlines())} lines): "
                         f"read_file('{spill_path}')")
            except OSError:
                spill = ""
        head = "[duckdb]" if r.returncode == 0 else f"[duckdb exit {r.returncode}]"
        return f"{head}\n{body}{spill}"
    finally:
        try:
            os.unlink(sql_path)
        except OSError:
            pass


# Tool-body alias: the schema/tool name is `duckdb` (short, and what a model
# reaches for), the body is duckdb_query. Registered under BOTH names in
# DISPATCH so `DISPATCH[name] is <body>` introspection holds for either.
duckdb = duckdb_query
