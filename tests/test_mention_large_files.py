"""0.54.1: large-file mention gate — a @mention of a big file no longer inlines
the whole thing. CSV/TSV >1 KB get a header + first-10-rows preview (the columns
are what a CSV mention is about); any other file over NX_MENTION_INLINE_LIMIT
gets a stub naming the size and pointing at read_file. Small files inline as
before.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_VISION", "NX_MENTION_INLINE_LIMIT"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    yield nx, tmp_path


def _expand(nx, text):
    from nx._mentions import expand_at_mentions
    out = expand_at_mentions(text)
    assert isinstance(out, str)
    return out


def test_large_csv_gets_head_preview(nx_fresh):
    nx, tmp = nx_fresh
    rows = ["id,name,value"] + [f"{i},item{i},{i * 2}" for i in range(2000)]
    (tmp / "big.csv").write_text("\n".join(rows) + "\n")
    out = _expand(nx, "look at @big.csv")
    assert "large CSV" in out and "1999 data rows" in out
    assert "id,name,value" in out
    assert "item5," in out            # one of the first 10 rows
    assert "item50," not in out       # later rows are not inlined
    assert "NOT inlined" in out


def test_small_csv_still_inlines_whole(nx_fresh):
    nx, tmp = nx_fresh
    (tmp / "tiny.csv").write_text("a,b\n1,2\n")
    out = _expand(nx, "look at @tiny.csv")
    assert "NOT inlined" not in out and "1,2" in out


def test_large_tsv_gets_preview(nx_fresh):
    nx, tmp = nx_fresh
    rows = ["a\tb"] + [f"{i}\tx{i}" for i in range(500)]
    (tmp / "big.tsv").write_text("\n".join(rows) + "\n")
    out = _expand(nx, "look at @big.tsv")
    assert "large CSV" in out and "a\tb" in out
    assert "x400" not in out


def test_large_other_file_gets_stub(nx_fresh):
    nx, tmp = nx_fresh
    (tmp / "big.log").write_text("line\n" * 5000)
    out = _expand(nx, "look at @big.log")
    assert "NOT inlined" in out and "25000 bytes" in out
    assert "line\nline" not in out.split("--- @big.log")[1]


def test_small_other_file_still_inlines(nx_fresh):
    nx, tmp = nx_fresh
    (tmp / "small.py").write_text("x = 1\n")
    out = _expand(nx, "look at @small.py")
    assert "x = 1" in out


def test_inline_limit_env_override(nx_fresh):
    nx, tmp = nx_fresh
    os.environ["NX_MENTION_INLINE_LIMIT"] = "10"
    try:
        (tmp / "mid.txt").write_text("hello world\n")
        out = _expand(nx, "look at @mid.txt")
        assert "NOT inlined" in out and "12 bytes" in out
    finally:
        os.environ.pop("NX_MENTION_INLINE_LIMIT", None)


def test_oversized_csv_skips_row_count_scan(nx_fresh):
    nx, tmp = nx_fresh
    # past NX_MENTION_CSV_SCAN_BYTES the preview must not read the whole file
    # to count rows: it reports "10+" instead
    rows = ["id,name"] + [f"{i},item{i}" for i in range(300)]
    (tmp / "huge.csv").write_text("\n".join(rows) + "\n")
    os.environ["NX_MENTION_CSV_SCAN_BYTES"] = "10"
    try:
        from nx._mentions import expand_at_mentions
        out = expand_at_mentions("look at @huge.csv")
    finally:
        os.environ.pop("NX_MENTION_CSV_SCAN_BYTES", None)
    assert isinstance(out, str)
    assert "10+ data rows" in out
    assert "id,name" in out


def test_malformed_csv_does_not_inline_binary(nx_fresh):
    nx, tmp = nx_fresh
    # not valid utf-8 text: csv preview bails, read_file's own error names it
    (tmp / "bin.csv").write_bytes(b"\xff\xfe\x00" * 500)
    out = _expand(nx, "look at @bin.csv")
    assert "not a valid UTF-8" in out
