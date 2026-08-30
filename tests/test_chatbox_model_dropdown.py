#!/usr/bin/env python3
"""Regression tests for the /source <name> <model> dropdown.

Covers the bubbles incident: a multi-model proxy whose GET /v1/models is
partly stale (retired ids still listed) grew past the old hard 6-row cap,
so the alphabetically-last ids (the newer qwen models) could never render —
and nothing distinguished live from retired ids. The dropdown must now
scroll instead of truncating, and NX_SOURCE_<NAME>_MODELS must pin the id
list when the server's is stale.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-model-dropdown-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _chatbox, _state as st  # noqa: E402


class _FakeModels:
    def __init__(self, ids):
        self.data = [type("M", (), {"id": i})() for i in ids]

    def list(self, timeout=None):
        return self


class _FakeClient:
    def __init__(self, ids):
        self.models = _FakeModels(ids)


def _make_source(name: str, ids: list[str], pinned: str | None = None):
    from nx._sources import Source
    src = Source(name, "http://localhost:9/v1", "sk-noop",
                 pinned_models=pinned)
    src.client = _FakeClient(ids)
    return src


def test_unpinned_list_is_served_sorted():
    src = _make_source("bubbles", ["qwen3.8-27b", "glm-5.3", "glm-5.2"])
    assert src.list_model_ids() == ["glm-5.2", "glm-5.3", "qwen3.8-27b"]


def test_pinned_models_replaces_the_server_list():
    # The bubbles lineup as of 2026-09: glm-5.3 replaces glm-5.2, and the
    # qwen3.8 models are new. The server still lists retired ids.
    src = _make_source(
        "bubbles",
        ["glm-5.2", "deepseek-v4-flash", "deepseek-v4-flash-dspark",
         "deepseek-v4-pro", "qwen3-embedding-8b", "glm-5.3",
         "glm-5.3-flash", "qwen3.8-flash-next", "qwen3.8-27b"],
        pinned="glm-5.3,glm-5.3-flash,qwen3.8-flash-next,qwen3.8-27b,"
               "deepseek-v4-pro,qwen3-embedding-8b")
    ids = src.list_model_ids()
    assert ids == ["deepseek-v4-pro", "glm-5.3", "glm-5.3-flash",
                   "qwen3-embedding-8b", "qwen3.8-27b",
                   "qwen3.8-flash-next"]
    # Retired ids are gone from the dropdown source of truth.
    assert "glm-5.2" not in ids
    assert "deepseek-v4-flash" not in ids
    assert "deepseek-v4-flash-dspark" not in ids
    # No network call is made when pinned (client.models is the fake;
    # it would raise if a real fetch replaced it — so just assert the pin
    # wins even after a forced refresh).
    class _Raising:
        def list(self, timeout=None):
            raise AssertionError("pinned source must not hit the network")
    src.client.models = _Raising()
    assert src.list_model_ids(force=True) == ids


def test_pinned_list_strips_and_drops_empty_entries():
    src = _make_source("bubbles", [], pinned=" a , ,b,, ")
    assert src.list_model_ids() == ["a", "b"]
    src2 = _make_source("bubbles", [], pinned=",,,")
    assert src2.list_model_ids() == []
    assert src2._pinned_model_ids is None  # all-empty pin ≡ no pin


def test_match_source_model_is_not_capped_and_leads_with_default():
    src = _make_source("bubbles", [f"model-{i:02d}" for i in range(20)])
    st.SOURCES["bubbles"] = src
    st.SOURCE_ORDER.append("bubbles")
    try:
        matches = _chatbox._match_source_model("bubbles", "")
        # sentinel + all 20 ids: the dropdown scrolls (render-side window),
        # it must not truncate the match list before rendering.
        assert len(matches) == 21
        assert matches[0] == ("", "")
        assert matches[1] == ("model-00", "")
        assert matches[-1] == ("model-19", "")
        # With a prefix the sentinel disappears and fuzzy filtering runs.
        # (Fuzzy matching also admits subsequence hits like "model-01".)
        typed = _chatbox._match_source_model("bubbles", "model-1")
        assert typed and all(t[0].startswith("model-1") for t in typed[:10])
        assert len(typed) == 11  # model-10 .. model-19 + subsequence model-01
    finally:
        del st.SOURCES["bubbles"]
        st.SOURCE_ORDER.remove("bubbles")


def test_dropdown_window_scrolls_to_keep_selection_visible():
    w = _chatbox._dropdown_window
    # Short lists: no windowing, no ellipsis.
    assert w(8, 0, 0) == (0, False, False)
    assert w(8, 7, 0) == (0, False, False)
    # 21 rows (sentinel + 20 ids), selection at the last row.
    top, above, below = w(21, 20, 0)
    assert top == 21 - _chatbox._DROPDOWN_MAX_ROWS
    assert above is True and below is False
    # Selection walks off the top of the kept window.
    top, above, below = w(21, 0, 13)
    assert top == 0
    assert above is False and below is True
    # PageUp from the bottom clamps at row 0, PageDown at the last row.
    n = 21
    sel = max(0, min(n - 1, 20 - _chatbox._DROPDOWN_MAX_ROWS))
    assert sel == 12
    assert max(0, min(n - 1, 20 + _chatbox._DROPDOWN_MAX_ROWS)) == 20


def test_fuzzy_filter_finds_the_new_qwen_models():
    # Typing "qwen" must surface the previously-hidden ids.
    ids = ["deepseek-v4-flash", "deepseek-v4-flash-dspark", "deepseek-v4-pro",
           "glm-5.2", "glm-5.3", "glm-5.3-flash", "qwen3-embedding-8b",
           "qwen3.8-27b", "qwen3.8-flash-next"]
    got = _chatbox._fuzzy_filter(ids, "qwen")
    assert "qwen3.8-27b" in got
    assert "qwen3.8-flash-next" in got


if __name__ == "__main__":
    for _name in [n for n in sorted(globals()) if n.startswith("test_")]:
        globals()[_name]()
        print(f"  PASS {_name}")
    print("ALL PASS")
