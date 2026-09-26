#!/usr/bin/env python3
"""Smoke test for nx's multi-source system. No live server needed."""
import os
import sys
import importlib

# Add project root (parent of this tests/ dir) to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_nx_env():
    for k in list(os.environ):
        if k.startswith("NX_"):
            del os.environ[k]
    # Scrub the built-in sources' trigger keys so tests that want a known,
    # minimal source set aren't perturbed by a real ~/.env / shell export of
    # TOGETHER_API_KEY / OPENROUTER_API_KEY. (Tests that exercise a built-in
    # source set it explicitly.)
    os.environ.pop("TOGETHER_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)
    # Prevent ~/.env from leaking real config into the tests. _load_env_file()
    # re-reads it on every reload, so point it at a nonexistent file.
    os.environ["NX_ENV_FILE"] = "/dev/null"

# --- Test 1: legacy fallback (no NX_SOURCE_* vars) ---
clear_nx_env()
sys.argv = ["nx.py"]
import nx
# Keep the ~14 reloads below offline: the built-in sources (openrouter,
# together) point at real URLs and every switch_source() fires resolve_model()
# plus the capability/thinking background probes. Stub them once at class /
# module level — importlib.reload(nx) re-executes nx/__init__.py but NOT these
# submodules, so the stubs survive every reload below. The configured-model
# assertions (nx.MODEL == ...) are unaffected: the stub returns self.model.
# pytest imports every test module into ONE process, so the stubs MUST be
# undone before this file finishes — else later modules (test_thinking.py
# drives ensure_thinking_probe for real) see the no-ops. Restored at the end.
import nx._capability as _cap_mod
import nx._thinking as _think_mod
_real_resolve_model = nx._sources.Source.resolve_model
_real_ensure_cap = _cap_mod.ensure_capability_probe
_real_ensure_think = _think_mod.ensure_thinking_probe
nx._sources.Source.resolve_model = lambda self: self.model or "local-model"
_cap_mod.ensure_capability_probe = lambda src: None
_think_mod.ensure_thinking_probe = lambda src, mid=None: None
importlib.reload(nx)
print("TEST 1: legacy fallback")
print(f"  SOURCES: {list(nx.SOURCES.keys())}")
print(f"  ACTIVE.name: {nx.ACTIVE.name}")
print(f"  client.base_url: {nx.client.base_url}")
assert list(nx.SOURCES.keys()) == ["local"]
assert nx.ACTIVE.name == "local"
print("  PASS\n")

# --- Test 2: multi-source discovery ---
clear_nx_env()
os.environ["NX_SOURCES"] = "local,zai"
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
os.environ["NX_SOURCE_LOCAL_API_KEY"] = "sk-noop"
os.environ["ZAI_TEST_KEY"] = "fake-zai-key-12345"
os.environ["NX_SOURCE_ZAI_BASE_URL"] = "https://api.z.ai/api/paas/v4"
os.environ["NX_SOURCE_ZAI_API_KEY"] = "$ZAI_TEST_KEY"
os.environ["NX_SOURCE_ZAI_MODEL"] = "glm-x-preview"
sys.argv = ["nx.py"]
importlib.reload(nx)
print("TEST 2: multi-source discovery")
print(f"  SOURCE_ORDER: {nx.SOURCE_ORDER}")
print(f"  ACTIVE.name: {nx.ACTIVE.name}")
print(f"  zai.api_key: {nx.SOURCES['zai'].api_key}")
print(f"  zai.model: {nx.SOURCES['zai'].model}")
print(f"  local.model: {nx.SOURCES['local'].model}")
assert nx.SOURCE_ORDER == ["local", "zai"]
assert nx.ACTIVE.name == "local"
assert nx.SOURCES["zai"].api_key == "fake-zai-key-12345", "$indirection failed"
assert nx.SOURCES["zai"].model == "glm-x-preview"
print("  PASS\n")

# --- Test 2b: unresolved API-key indirection must not become sk-noop ---
os.environ.pop("NX_MISSING_API_KEY", None)
try:
    nx._resolve_api_key("$NX_MISSING_API_KEY")
except ValueError as e:
    assert "NX_MISSING_API_KEY" in str(e)
    assert "not set or empty" in str(e)
else:
    raise AssertionError("unresolved API-key reference should fail loudly")
print("TEST 2b: unresolved API-key reference fails loudly")
print("  PASS\n")

# --- Test 3: switch_source ---
print("TEST 3: switch_source('zai')")
old_client_id = id(nx.client)
old_model = nx.MODEL
nx.switch_source("zai")
print(f"  ACTIVE.name: {nx.ACTIVE.name}")
print(f"  MODEL: {nx.MODEL}")
print(f"  client.base_url: {nx.client.base_url}")
assert nx.ACTIVE.name == "zai"
assert nx.MODEL == "glm-x-preview"
assert id(nx.client) != old_client_id, "client object should have changed"
print("  PASS\n")

# --- Test 4: switch back ---
nx.switch_source("local")
print("TEST 4: switch back to local")
assert nx.ACTIVE.name == "local"
print("  PASS\n")

# --- Test 5: unknown source ---
print("TEST 5: unknown source")
result = nx.switch_source("nonexistent")
assert result is False
print("  PASS\n")

# --- Test 6: --source flag ---
clear_nx_env()
os.environ["NX_SOURCES"] = "local,zai"
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
os.environ["NX_SOURCE_ZAI_BASE_URL"] = "https://api.z.ai/api/paas/v4"
os.environ["NX_SOURCE_ZAI_MODEL"] = "glm-x-preview"
sys.argv = ["nx.py", "--source", "zai"]
importlib.reload(nx)
print("TEST 6: --source zai flag")
print(f"  ACTIVE.name: {nx.ACTIVE.name}")
assert nx.ACTIVE.name == "zai"
print("  PASS\n")

# --- Test 7: auto-discover without NX_SOURCES ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
os.environ["NX_SOURCE_ZAI_BASE_URL"] = "https://api.z.ai/api/paas/v4"
importlib.reload(nx)
print("TEST 7: auto-discover from BASE_URL vars")
print(f"  SOURCE_ORDER: {nx.SOURCE_ORDER}")
assert set(nx.SOURCE_ORDER) == {"local", "zai"}
print("  PASS\n")

# --- Test 8: _banner() reflects active source ---
nx.switch_source("zai")
banner = nx._banner()
print("TEST 8: _banner() with multiple sources")
print(f"  banner: {banner}")
assert "zai" in banner
print("  PASS\n")

# --- Test 9: built-in `together` source auto-registers with a key ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
os.environ["TOGETHER_API_KEY"] = "fake-together-key"
importlib.reload(nx)
print("TEST 9: built-in together source auto-register")
print(f"  SOURCE_ORDER: {nx.SOURCE_ORDER}")
print(f"  together.base_url: {nx.SOURCES['together'].base_url}")
print(f"  together.model: {nx.SOURCES['together'].model}")
print(f"  together.api_key: {nx.SOURCES['together'].api_key}")
assert "together" in nx.SOURCES, "together source should auto-register with a key"
assert nx.SOURCES["together"].base_url == "https://api.together.xyz/v1"
assert nx.SOURCES["together"].model == "zai-org/GLM-5.3", "default model should be GLM-5.3"
assert nx.SOURCES["together"].api_key == "fake-together-key"
# Registered last, so the first-defined source stays the startup default.
assert nx.SOURCE_ORDER[0] == "local"
assert nx.SOURCE_ORDER[-1] == "together"
assert nx.ACTIVE.name == "local", "local should remain the default on load"
print("  PASS\n")

# --- Test 10: no together source without a key ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
importlib.reload(nx)
print("TEST 10: no together source without TOGETHER_API_KEY")
print(f"  SOURCE_ORDER: {nx.SOURCE_ORDER}")
assert "together" not in nx.SOURCES, "no together source without a key"
print("  PASS\n")

# --- Test 11: explicit together config wins over the built-in ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCES"] = "together"
os.environ["NX_SOURCE_TOGETHER_BASE_URL"] = "https://my-proxy.example/v1"
os.environ["NX_SOURCE_TOGETHER_API_KEY"] = "custom-key"
os.environ["NX_SOURCE_TOGETHER_MODEL"] = "my-org/my-model"
os.environ["TOGETHER_API_KEY"] = "fake-together-key"
importlib.reload(nx)
print("TEST 11: explicit together config overrides built-in")
print(f"  together.base_url: {nx.SOURCES['together'].base_url}")
print(f"  together.model: {nx.SOURCES['together'].model}")
assert nx.SOURCES["together"].base_url == "https://my-proxy.example/v1"
assert nx.SOURCES["together"].model == "my-org/my-model"
assert nx.SOURCES["together"].api_key == "custom-key"
print("  PASS\n")

# --- Test 12: switch_source model override (per-switch model pin) ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCES"] = "together"
os.environ["NX_SOURCE_TOGETHER_BASE_URL"] = "https://api.together.xyz/v1"
os.environ["NX_SOURCE_TOGETHER_API_KEY"] = "k"
os.environ["NX_SOURCE_TOGETHER_MODEL"] = "zai-org/GLM-5.3"
importlib.reload(nx)
print("TEST 12: switch_source model_override")
nx.switch_source("together")
assert nx.MODEL == "zai-org/GLM-5.3", "bare switch → source default"
# Override pins a different model for this switch.
nx.switch_source("together", model_override="zai-org/GLM-4.6")
assert nx.MODEL == "zai-org/GLM-4.6", "override should pin MODEL"
# A bare switch afterward falls back to the configured default (non-sticky).
nx.switch_source("together")
assert nx.MODEL == "zai-org/GLM-5.3", "bare switch should restore default"
print("  PASS\n")

# --- Test 13: built-in `openrouter` source auto-registers with a key ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
os.environ["OPENROUTER_API_KEY"] = "fake-or-key"
importlib.reload(nx)
print("TEST 13: built-in openrouter source auto-register")
print(f"  SOURCE_ORDER: {nx.SOURCE_ORDER}")
or_src = nx.SOURCES.get("openrouter")
assert or_src is not None, "openrouter source should auto-register with a key"
assert or_src.base_url == "https://openrouter.ai/api/v1"
assert or_src.model == "z-ai/glm-5.3", "default model should be z-ai/glm-5.3"
assert or_src.api_key == "fake-or-key"
# Default routing pins io-net/fp8 with no fallback.
assert or_src.extra_body == {"provider": {"order": ["io-net/fp8"],
                                          "allow_fallbacks": False}}, \
    f"default routing should pin io-net/fp8, got {or_src.extra_body!r}"
# Registered last, so the first-defined source stays the startup default.
assert nx.SOURCE_ORDER[0] == "local"
assert nx.SOURCE_ORDER[-1] == "openrouter"
assert nx.ACTIVE.name == "local", "local should remain the default on load"
print("  PASS\n")

# --- Test 14: no openrouter source without a key ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
importlib.reload(nx)
print("TEST 14: no openrouter source without OPENROUTER_API_KEY")
print(f"  SOURCE_ORDER: {nx.SOURCE_ORDER}")
assert "openrouter" not in nx.SOURCES, "no openrouter source without a key"
print("  PASS\n")

# --- Test 15: explicit openrouter config overrides the built-in ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCES"] = "openrouter"
os.environ["NX_SOURCE_OPENROUTER_BASE_URL"] = "https://my-or-proxy.example/v1"
os.environ["NX_SOURCE_OPENROUTER_API_KEY"] = "custom-or-key"
os.environ["NX_SOURCE_OPENROUTER_MODEL"] = "my-org/my-model"
os.environ["NX_SOURCE_OPENROUTER_EXTRA_BODY"] = (
    '{"provider":{"order":["Together"],"allow_fallbacks":true}}')
os.environ["OPENROUTER_API_KEY"] = "fake-or-key"
importlib.reload(nx)
print("TEST 15: explicit openrouter config overrides built-in")
print(f"  or.base_url: {nx.SOURCES['openrouter'].base_url}")
print(f"  or.model: {nx.SOURCES['openrouter'].model}")
assert nx.SOURCES["openrouter"].base_url == "https://my-or-proxy.example/v1"
assert nx.SOURCES["openrouter"].model == "my-org/my-model"
assert nx.SOURCES["openrouter"].api_key == "custom-or-key"
assert nx.SOURCES["openrouter"].extra_body == \
    {"provider": {"order": ["Together"], "allow_fallbacks": True}}
print("  PASS\n")

# --- Test 16: bad EXTRA_BODY JSON is ignored loudly (no silent routing) ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCES"] = "openrouter"
os.environ["NX_SOURCE_OPENROUTER_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["NX_SOURCE_OPENROUTER_API_KEY"] = "k"
os.environ["NX_SOURCE_OPENROUTER_MODEL"] = "z-ai/glm-5.3"
os.environ["NX_SOURCE_OPENROUTER_EXTRA_BODY"] = "not json{"
importlib.reload(nx)
print("TEST 16: bad EXTRA_BODY JSON ignored")
# Built-in default routing does NOT fill in for a malformed explicit config —
# a typo must fail loudly (stderr warning) rather than silently route to the
# default, so extra_body stays None.
assert nx.SOURCES["openrouter"].extra_body is None, \
    "malformed EXTRA_BODY should yield no routing, not the built-in default"
print("  PASS\n")

# --- Test 17: _set_provider_order / _provider_order helpers ---
clear_nx_env()
sys.argv = ["nx.py"]
# Auto-register the built-in openrouter source (the path that seeds the
# io-net/fp8 default routing). An explicit NX_SOURCES=openrouter without
# EXTRA_BODY would NOT inject the default — explicit config means user-control.
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
os.environ["OPENROUTER_API_KEY"] = "k"
importlib.reload(nx)
print("TEST 17: _set_provider_order / _provider_order")
src = nx.SOURCES["openrouter"]
# Starts with the built-in default routing.
assert nx._provider_order(src) == ["io-net/fp8"]
# Pin a new ordered list — defaults to no fallback (deliberate choice).
nx._set_provider_order(src, ["Together", "DeepInfra"])
assert nx._provider_order(src) == ["Together", "DeepInfra"]
assert src.extra_body["provider"]["allow_fallbacks"] is False
# Explicit allow_fallbacks=True is honored.
nx._set_provider_order(src, ["Together"], allow_fallbacks=True)
assert src.extra_body["provider"]["allow_fallbacks"] is True
# Empty order clears routing entirely.
nx._set_provider_order(src, [])
assert nx._provider_order(src) == []
assert src.extra_body is None
print("  PASS\n")

# --- Test 18: extra_request_kwargs plumbing ---
clear_nx_env()
sys.argv = ["nx.py"]
os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
os.environ["OPENROUTER_API_KEY"] = "k"
importlib.reload(nx)
print("TEST 18: extra_request_kwargs plumbing")
# PF-10 (0.59.0): extra_request_kwargs() also carries a per-session routing
# hint (user / prompt_cache_key) whenever a session recorder exists. This test
# asserts the extra_body plumbing exactly, so pin the recorder off — an earlier
# test module in the same pytest process may have left one installed.
nx.st._METRICS = None
src = nx.SOURCES["openrouter"]
# Routed source yields an extra_body kwarg.
kw = src.extra_request_kwargs()
assert kw == {"extra_body": {"provider": {"order": ["io-net/fp8"],
                                          "allow_fallbacks": False}}}, kw
# Clearing routing yields nothing — non-routing sources contribute no kwarg.
nx._set_provider_order(src, [])
assert src.extra_request_kwargs() == {}
# Re-pin and confirm the active source's kwarg matches.
nx._set_provider_order(src, ["io-net/fp8"])
nx.switch_source("openrouter")
assert nx.ACTIVE.extra_request_kwargs() == \
    {"extra_body": {"provider": {"order": ["io-net/fp8"],
                                 "allow_fallbacks": False}}}
print("  PASS\n")

print("=" * 50)
# undo the offline stubs (see the note at the top) before pytest moves on
nx._sources.Source.resolve_model = _real_resolve_model
_cap_mod.ensure_capability_probe = _real_ensure_cap
_think_mod.ensure_thinking_probe = _real_ensure_think
print("ALL TESTS PASSED")
