"""Regression: conftest's per-test module rewind vs. daemon-thread imports.

The suite's background workers (nx._subagent tier jobs, nx._tool_prefetch)
lazily do `from ._worktree import ...` on daemon threads that outlive the test
which started them. conftest's `_restore()` deletes every `nx.*` module missing
from its post-import baseline — and `nx._worktree` is missing (nx/__init__
doesn't import it). Deleting it mid-import is fatal in a way that's easy to
miss: cpython's `_load_unlocked` ends with an UNGUARDED
`module = sys.modules.pop(spec.name)` (importlib/_bootstrap.py:946 on 3.12; the
failure-path `del` a few lines above it catches KeyError, this one doesn't), so
the importing thread dies with `KeyError: 'nx._worktree'` and pytest reports it
out of context as PytestUnhandledThreadExceptionWarning — the 0.122.0 warning.

The fix (conftest `_restore`) skips modules whose `__spec__._initializing` is
still set; the next restore pass reaps them. This test forces that exact
interleaving deterministically instead of leaving it to thread timing.
"""
import importlib.machinery
import sys
import threading

import pytest

_MODULE = "nx._worktree"
_TIMEOUT = 15.0


class _GatedLoader:
    """Wraps the real loader and blocks inside exec_module, after the body ran.

    The window we need is *after* exec_module and *before* importlib's closing
    unguarded `sys.modules.pop(...)` — the importer is still `_initializing`,
    holding the entry that `_restore` used to delete out from under it.
    """

    def __init__(self, inner):
        self._inner = inner
        self.in_exec = threading.Event()
        self.release = threading.Event()

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self.in_exec.set()
        if not self.release.wait(_TIMEOUT):
            raise RuntimeError("test gate never released")

    def __getattr__(self, item):  # spec_from_file_location / finder protocols
        return getattr(self._inner, item)


class _GatedFinder:
    """meta_path finder that hands out nx._worktree wrapped in _GatedLoader."""

    def __init__(self):
        self.loader = None
        self._reentrant = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _MODULE or self._reentrant or self.loader is not None:
            return None
        self._reentrant = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            self._reentrant = False
        if spec is None:
            return None
        self.loader = _GatedLoader(spec.loader)
        spec.loader = self.loader
        return spec


def _cold_import(outcomes):
    """What a worker thread does: cold-import nx._worktree, record crash/result."""
    try:
        from nx._worktree import _worktree_enabled  # noqa: F401
        outcomes.append(sys.modules.get(_MODULE))
    except BaseException as exc:  # noqa: BLE001 - the thread's crash IS the signal
        outcomes.append(exc)


def _conftest():
    try:
        import conftest
    except ImportError:  # pragma: no cover - pytest imports tests/conftest.py as "conftest"
        pytest.skip("tests/conftest.py not importable as 'conftest'")
    return conftest


def test_restore_does_not_yank_a_module_out_from_under_a_running_import():
    conftest = _conftest()
    assert _MODULE not in conftest._BASELINE_MODS, (
        "precondition: nx._worktree must be absent from the baseline snapshot "
        "(that's what made _restore() delete it)"
    )

    sys.modules.pop(_MODULE, None)
    finder = _GatedFinder()
    sys.meta_path.insert(0, finder)
    outcomes = []
    try:
        thread = threading.Thread(target=_cold_import, args=(outcomes,), daemon=True)
        thread.start()
        try:
            gate = _await_gate(finder)
            gate.in_exec.wait(_TIMEOUT)

            # The race: rewind module state while that import is mid-flight.
            conftest._restore(str(__file__))
        finally:
            if finder.loader is not None:
                finder.loader.release.set()
        thread.join(_TIMEOUT)
        assert not thread.is_alive(), "importing thread hung on the gate"
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        sys.modules.pop(_MODULE, None)

    assert len(outcomes) == 1, "import thread neither finished nor crashed"
    result = outcomes[0]
    assert not isinstance(result, BaseException), (
        f"import thread crashed under restore(): {result!r}"
    )
    assert result is not None, "importer's own bookkeeping lost the module"


def _await_gate(finder, settle=0.002):
    """Poll until the finder has installed its gated loader."""
    waited = 0.0
    while finder.loader is None and waited < _TIMEOUT:
        threading.Event().wait(settle)
        waited += settle
    assert finder.loader is not None, "importer never reached the gated loader"
    return finder.loader
