#!/usr/bin/env python3
"""Offline tests for the wrist-camera-only marker centre logged by insertion.py."""
import collections
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import insertion
from std_msgs.msg import String

A = insertion.AngledInserter


def _node():
    params = {"wrist_camera_name": "kinova", "world_frame": "world"}
    n = SimpleNamespace(
        _marker_obs=collections.deque(maxlen=2000),
        get_parameter=lambda k: SimpleNamespace(value=params[k]),
        get_logger=lambda: Mock(),
        _run_info=Mock())
    n._wrist_centre = lambda t, ids=(0, 1, 2, 3): A._wrist_centre(n, t, ids)
    return n


def _obs(cam, markers, frame="world"):
    return String(data=json.dumps({"camera": cam, "frame": frame, "markers": [
        {"id": i, "pos_ref": list(p)} for i, p in markers.items()]}))


SQUARE = {0: (0.55, -0.05, -0.025), 1: (0.59, -0.05, -0.025),
          2: (0.59, -0.09, -0.025), 3: (0.55, -0.09, -0.025)}


class WristCentre(unittest.TestCase):
    def test_four_markers_give_square_centre(self):
        n = _node()
        for _ in range(5):
            A._marker_obs_cb(n, _obs("kinova", SQUARE))
        (x, y), counts = A._wrist_centre(n, 0.0)
        self.assertAlmostEqual(x, 0.57, places=6)
        self.assertAlmostEqual(y, -0.07, places=6)
        self.assertEqual(counts, {0: 5, 1: 5, 2: 5, 3: 5})

    def test_front_camera_and_stale_messages_ignored(self):
        n = _node()
        A._marker_obs_cb(n, _obs("realsense", {i: (9, 9, 9) for i in SQUARE}))
        A._marker_obs_cb(n, _obs("kinova", SQUARE))
        n._marker_obs[-1] = (-1.0,) + n._marker_obs[-1][1:]   # older than t_start
        A._marker_obs_cb(n, _obs("kinova", SQUARE, frame="base_link"))
        centre, counts = A._wrist_centre(n, 0.0)
        self.assertIsNone(centre)
        self.assertEqual(counts, {})

    def test_missing_corner_refuses(self):
        n = _node()
        A._marker_obs_cb(n, _obs("kinova", {i: SQUARE[i] for i in (0, 1, 3)}))
        centre, counts = A._wrist_centre(n, 0.0)
        self.assertIsNone(centre)
        self.assertEqual(sorted(counts), [0, 1, 3])

    def test_outlier_rejected_by_median(self):
        n = _node()
        for _ in range(4):
            A._marker_obs_cb(n, _obs("kinova", SQUARE))
        bad = dict(SQUARE)
        bad[0] = (1.0, 1.0, 0.0)       # one false detection, 424 mm-style jump
        A._marker_obs_cb(n, _obs("kinova", bad))
        (x, y), _ = A._wrist_centre(n, 0.0)
        self.assertAlmostEqual(x, 0.57, places=6)

    def test_malformed_message_ignored(self):
        n = _node()
        A._marker_obs_cb(n, String(data="not json"))
        A._marker_obs_cb(n, String(data=json.dumps({"camera": "kinova", "markers": [{"x": 1}]})))
        self.assertEqual(len(n._marker_obs), 0)

    def test_log_event_has_diff(self):
        n = _node()
        for _ in range(3):
            A._marker_obs_cb(n, _obs("kinova", SQUARE))
        A._log_wrist_vs_fused(n, 0.0, (0.565, -0.078))
        ev, data = n._run_info.call_args[0][0], n._run_info.call_args[1]
        self.assertEqual(ev, "wrist_centre")
        self.assertAlmostEqual(data["diff_x_mm"], 5.0)
        self.assertAlmostEqual(data["diff_y_mm"], 8.0)
        json.dumps(data)   # must be serialisable for /insertion/run_info

    def test_log_event_without_wrist(self):
        n = _node()
        A._log_wrist_vs_fused(n, 0.0, (0.5, 0.1))
        data = n._run_info.call_args[1]
        self.assertNotIn("wrist_x", data)
        json.dumps(data)


if __name__ == "__main__":
    unittest.main()
