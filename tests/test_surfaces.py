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


if __name__ == "__main__":
    unittest.main()
