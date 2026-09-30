"""Steering queue + /tabs surface (0.105.0): queue/drain semantics, injection
into the turn loop's messages, and the dashboard snapshot shape."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import nx as _m  # noqa: E402
from nx import _surfaces  # noqa: E402


class TestSteeringQueue(unittest.TestCase):
    def setUp(self):
        _surfaces._STEER_QUEUE.clear()
        _surfaces.set_steer_partial("")

    def test_queue_drain_fifo(self):
        _m.queue_steering("use rg instead")
        _m.queue_steering("  skip the tests  ")
        _m.queue_steering("   ")  # blank ignored
        self.assertEqual(_m.drain_steering(), ["use rg instead", "skip the tests"])
        self.assertEqual(_m.drain_steering(), [])
        self.assertEqual(_m.steer_pending(), 0)

    def test_partial_roundtrip(self):
        _surfaces.set_steer_partial("hal")
        _surfaces.set_steer_partial("half typed")
        self.assertEqual(_m.steer_partial(), "half typed")
        _surfaces.set_steer_partial("")
        self.assertEqual(_m.steer_partial(), "")

    def test_inject_steering_appends_user_messages(self):
        msgs = [{"role": "user", "content": "task"}]
        _m.queue_steering("prefer duckdb")
        _m.queue_steering("then commit")
        ok = _m.inject_steering(msgs)
        self.assertTrue(ok)
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("prefer duckdb", msgs[1]["content"])
        self.assertIn("[Steering note", msgs[1]["content"])
        self.assertIn("then commit", msgs[2]["content"])
        self.assertFalse(_m.inject_steering(msgs))
        self.assertEqual(len(msgs), 3)


class TestTabsSnapshot(unittest.TestCase):
    def test_snapshot_has_main_row(self):
        rows = _surfaces._surfaces_snapshot()
        self.assertTrue(any(r["kind"] == "main" for r in rows))
        main = rows[0]
        self.assertEqual(main["id"], "main")
        self.assertIn("alive", main)

    def test_fallback_print_runs(self):
        # exercises the non-tty path end to end; must not raise
        _surfaces.tabs_print_fallback()


class TestTurnLoopSteeringContinuation(unittest.TestCase):
    def test_done_with_pending_steering_continues(self):
        """TURN_DONE + queued steering → loop injects and keeps going."""
        _surfaces._STEER_QUEUE.clear()
        _m.queue_steering("also run the linter")
        turns = ["done", "done"]  # first model_turn → done; continuation after steering
        _turn_messages = [{"role": "user", "content": "go"}]
        with mock.patch.object(_m, "model_turn", side_effect=turns) as _turn_mock, \
             mock.patch.object(_m, "open_stream"), \
             mock.patch("nx._turn._launch_verify_if_settled"):
            summary = _m._run_model_turn_loop(_turn_messages)
        self.assertEqual(summary["status"], "done")
        # queue must be drained and the steering note present in the transcript
        self.assertEqual(_m.drain_steering(), [])
        self.assertTrue(any("also run the linter" in str(m.get("content", ""))
                            for m in _turn_messages))
        self.assertEqual(_turn_mock.call_count, 2)  # continued past the DONE

    def test_done_without_steering_returns(self):
        _surfaces._STEER_QUEUE.clear()
        with mock.patch.object(_m, "model_turn", return_value="done"), \
             mock.patch.object(_m, "open_stream"):
            summary = _m._run_model_turn_loop([{"role": "user", "content": "go"}])
        self.assertEqual(summary["status"], "done")


class TestAgentSteering(unittest.TestCase):
    """0.106.0: per-agent steering queues, thread-scoped drain, breadcrumbs."""

    def setUp(self):
        _surfaces._STEER_QUEUE.clear()
        _surfaces._AGENT_QUEUES.clear()
        _surfaces.set_steer_partial("")

    def _register_agent(self, job_id="agent-1-abc", status="running"):
        from nx import _subagent as sa
        with sa._ASYNC_LOCK:
            sa._ASYNC_AGENTS[job_id] = {
                "instruction": "test", "status": status, "answer": None,
                "started": 0.0, "announced": False,
            }
        return job_id

    def _drop_agent(self, job_id="agent-1-abc"):
        from nx import _subagent as sa
        with sa._ASYNC_LOCK:
            sa._ASYNC_AGENTS.pop(job_id, None)

    def test_queue_agent_steering_routes_and_validates(self):
        jid = self._register_agent()
        self.assertTrue(_m.queue_agent_steering(jid, "use rg"))
        self.assertFalse(_m.queue_agent_steering("agent-9-none", "nope"))
        self._register_agent("agent-2-dead", status="done")
        self.assertFalse(_m.queue_agent_steering("agent-2-dead", "nope"))
        self.assertEqual(_m.drain_agent_steering(jid), ["use rg"])
        self.assertEqual(_m.drain_agent_steering(jid), [])
        self.assertEqual(_m.drain_steering(), [])  # main queue untouched

    def test_agent_pending_count(self):
        jid = self._register_agent()
        _m.queue_agent_steering(jid, "a")
        _m.queue_agent_steering(jid, "b")
        self.assertEqual(_m.agent_steering_pending(jid), 2)
        _m.drain_agent_steering(jid)
        self.assertEqual(_m.agent_steering_pending(jid), 0)

    def test_agent_loop_drains_own_queue_not_main(self):
        """Agent turn loop (TLS ctx with job_id) drains its own queue; the
        parent's global queue stays intact — the pre-0.106.0 steal race."""
        from nx import _subagent_tiers as st_tiers
        from nx import _subagent_tls
        jid = self._register_agent()
        _m.queue_agent_steering(jid, "prefer duckdb")
        _m.queue_steering("parent line")  # must NOT be stolen by the agent

        ctx = st_tiers._SubagentContext({}, job_id=jid)
        token = _subagent_tls._SUBAGENT_TLS
        prev = getattr(token, "ctx", None)
        token.ctx = ctx
        try:
            with mock.patch.object(_m, "model_turn", return_value="done"), \
                 mock.patch.object(_m, "open_stream"):
                msgs = [{"role": "user", "content": "go"}]
                summary = _m._run_model_turn_loop(msgs)
        finally:
            token.ctx = prev
        self.assertEqual(summary["status"], "done")
        self.assertTrue(any("prefer duckdb" in str(m.get("content", ""))
                            for m in msgs))
        self.assertEqual(_m.drain_steering(), ["parent line"])  # not stolen
        self.assertEqual(_m.drain_agent_steering(jid), [])

    def test_agent_loop_stamps_steered_breadcrumb(self):
        from nx import _subagent as sa
        from nx import _subagent_tiers as st_tiers
        from nx import _subagent_tls
        jid = self._register_agent()
        _m.queue_agent_steering(jid, "also check the schema")
        ctx = st_tiers._SubagentContext({}, job_id=jid)
        prev = getattr(_subagent_tls._SUBAGENT_TLS, "ctx", None)
        _subagent_tls._SUBAGENT_TLS.ctx = ctx
        try:
            with mock.patch.object(_m, "model_turn", return_value="done"), \
                 mock.patch.object(_m, "open_stream"):
                _m._run_model_turn_loop([{"role": "user", "content": "go"}])
        finally:
            _subagent_tls._SUBAGENT_TLS.ctx = prev
        with sa._ASYNC_LOCK:
            self.assertEqual(sa._ASYNC_AGENTS[jid].get("steered"), 1)

    def test_main_loop_does_not_steal_agent_queue(self):
        """Main loop (no TLS ctx) ignores per-agent queues."""
        jid = self._register_agent()
        _m.queue_agent_steering(jid, "agent-only note")
        with mock.patch.object(_m, "model_turn", return_value="done"), \
             mock.patch.object(_m, "open_stream"):
            _m._run_model_turn_loop([{"role": "user", "content": "go"}])
        self.assertEqual(_m.drain_agent_steering(jid), ["agent-only note"])
        self._drop_agent(jid)


if __name__ == "__main__":
    unittest.main()
