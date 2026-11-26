#!/usr/bin/env python3
"""Offline tests for the wrist-camera look pose helper in insertion.py."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import insertion

A = insertion.AngledInserter

# Hardware numbers (bag insertion_20261125_174017): ee_link->camera offset
# in world at q_search = camera - bracelet; q_search is tool-down.
Q_DOWN = (1.0, 0.0, 0.0, 0.0)   # 180 deg about X: tool z -> world -z
LOCAL_CAM = (0.001, 0.056, -0.058)


def _tf(v):
    return SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=v[0], y=v[1], z=v[2])))


def _node(offset=LOCAL_CAM):
    n = SimpleNamespace(
        get_parameter=lambda k: SimpleNamespace(value="bracelet_link"),
        get_logger=lambda: Mock(),
        _get_tf=Mock(return_value=_tf(offset) if offset else None),
        _ee_for_tip=Mock(return_value=("tip", 0, 0)))
    n._get_frame_offset = lambda f: A._get_frame_offset(n, f)
    return n


class LookPose(unittest.TestCase):
    def test_camera_lands_over_target(self):
        n = _node()
        tgt = (0.28, -0.07, 0.595)
        ee, fb = A._ee_for_frame(n, tgt, Q_DOWN, "camera_color_frame")
        self.assertFalse(fb)
        rot = insertion.rotate_vector_by_quat(LOCAL_CAM, Q_DOWN)
        for i in range(3):
            self.assertAlmostEqual(ee[i] + rot[i], tgt[i], places=9)
        # camera is offset from the wrist, so EE must differ from the target
        self.assertGreater(math.hypot(ee[0] - tgt[0], ee[1] - tgt[1]), 0.04)

    def test_tf_missing_falls_back_to_tip(self):
        n = _node(offset=None)
        ee, fb = A._ee_for_frame(n, (1, 2, 3), Q_DOWN, "camera_color_frame",
                                 fallback_tip_xyz=(0.28, -0.07, 0.3))
        self.assertTrue(fb)
        self.assertEqual(ee, ("tip", 0, 0))
        n._ee_for_tip.assert_called_once_with((0.28, -0.07, 0.3), Q_DOWN)

    def test_offset_rotated_by_q(self):
        n = _node(offset=(0.0, 0.1, 0.0))
        s = math.sqrt(0.5)
        q = (0.0, 0.0, s, s)   # +90 deg yaw: local y -> world -x
        ee, fb = A._ee_for_frame(n, (0.5, 0.0, 0.6), q, "camera_color_frame")
        self.assertFalse(fb)
        self.assertAlmostEqual(ee[0], 0.6, places=9)
        self.assertAlmostEqual(ee[1], 0.0, places=9)
        self.assertAlmostEqual(ee[2], 0.6, places=9)

    def test_offset_cached(self):
        n = _node()
        A._ee_for_frame(n, (0, 0, 0), Q_DOWN, "f")
        A._ee_for_frame(n, (0, 0, 0), Q_DOWN, "f")
        self.assertEqual(n._get_tf.call_count, 1)


if __name__ == "__main__":
    unittest.main()
