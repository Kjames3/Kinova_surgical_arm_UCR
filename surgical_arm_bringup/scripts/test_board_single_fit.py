#!/usr/bin/env python3
"""Offline test for single-marker board fitting in combine_cameras.py.

  source /opt/ros/humble/setup.bash && python3 test_board_single_fit.py
"""
import json, math, os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combine_cameras as cc

H = 0.115  # half of the 0.23 m board
# Learned board: corner id -> (x, y) from the centre, marker yaws in the same frame
TEMPLATE = {0: (-H, H), 1: (H, H), 2: (H, -H), 3: (-H, -H)}
TEMPLATE_YAW = {0: 0.02, 1: 0.01, 2: -0.015, 3: 0.03}   # slightly unequal on purpose


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def world_board(angle, centre):
    R = rot(angle)
    xy = {i: R @ np.array(p) + centre for i, p in TEMPLATE.items()}
    yaw = {i: cc.wrap_angle(TEMPLATE_YAW[i] + angle) for i in TEMPLATE}
    return xy, yaw


class TestSingleFit(unittest.TestCase):
    def test_recovers_board_from_each_marker(self):
        centre = np.array([0.43, -0.12])
        xy, yaw = world_board(math.radians(30), centre)
        for m in TEMPLATE:
            c, corners = cc.fit_board_single(TEMPLATE, TEMPLATE_YAW, m, xy[m], yaw[m])
            self.assertLess(np.linalg.norm(c - centre), 1e-9, m)
            for i in TEMPLATE:
                self.assertLess(np.linalg.norm(corners[i] - xy[i]), 1e-9)

    def test_agrees_with_template_fit(self):
        centre = np.array([0.43, -0.12])
        xy, yaw = world_board(math.radians(30), centre)
        c2, k2 = cc.fit_board_template(TEMPLATE, {0: xy[0], 1: xy[1]})
        c1, k1 = cc.fit_board_single(TEMPLATE, TEMPLATE_YAW, 0, xy[0], yaw[0])
        self.assertLess(np.linalg.norm(c1 - c2), 1e-9)
        for i in TEMPLATE:
            self.assertLess(np.linalg.norm(k1[i] - k2[i]), 1e-9)

    def test_yaw_wraparound(self):
        centre = np.array([0.3, 0.2])
        for deg in (179.0, 181.0, -179.5, 170.0):
            xy, yaw = world_board(math.radians(deg), centre)
            for m in TEMPLATE:
                c, _ = cc.fit_board_single(TEMPLATE, TEMPLATE_YAW, m, xy[m], yaw[m])
                self.assertLess(np.linalg.norm(c - centre), 1e-9, (deg, m))
        self.assertAlmostEqual(cc.wrap_angle(3 * math.pi), math.pi)
        a = [math.radians(179), math.radians(-179), math.radians(180)]
        self.assertLess(abs(abs(cc.circular_median(a)) - math.pi), math.radians(1.5))

    def test_old_format_template(self):
        old = {"0": [-H, H], "1": [H, H], "2": [H, -H], "3": [-H, -H]}
        tmpl, yaws = cc.parse_board_template(old)
        self.assertEqual(yaws, {})
        self.assertIsNone(cc.fit_board_single(tmpl, yaws, 0, (0, 0), 0.0))
        # still works for >= 2 markers
        xy, _ = world_board(0.5, np.array([0.4, 0.1]))
        self.assertIsNotNone(cc.fit_board_template(tmpl, {0: xy[0], 1: xy[1]}))
        # round trip through JSON in both new forms
        new = {str(i): list(TEMPLATE[i]) + [TEMPLATE_YAW[i]] for i in TEMPLATE}
        t2, y2 = cc.parse_board_template(json.loads(json.dumps(new)))
        self.assertEqual(y2, TEMPLATE_YAW)
        t3, y3 = cc.parse_board_template({"0": {"xy": [1, 2], "yaw": 0.3}})
        self.assertEqual((t3[0], y3[0]), ((1.0, 2.0), 0.3))

    def test_half_degree_yaw_error(self):
        centre = np.array([0.43, -0.12])
        xy, yaw = world_board(math.radians(30), centre)
        worst = 0.0
        for m in TEMPLATE:
            for sgn in (1, -1):
                c, _ = cc.fit_board_single(TEMPLATE, TEMPLATE_YAW, m, xy[m],
                                           yaw[m] + sgn * math.radians(0.5))
                worst = max(worst, np.linalg.norm(c - centre))
        print(f"0.5 deg yaw error -> centre error {1000 * worst:.2f} mm")
        self.assertLess(worst, 0.002)
        self.assertGreater(worst, 0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
