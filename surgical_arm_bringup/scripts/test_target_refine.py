#!/usr/bin/env python3
"""Offline tests for the refine_target checkpoints in insertion.py."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import insertion
from insertion_geometry import insertion_axis_geometry

A = insertion.AngledInserter
MM = 1e-3


def _node(params=None, centre=None, lin_ok=True):
    p = {"refine_target": True, "target_source": "markers",
         "refine_min_mm": 1.0, "refine_max_mm": 15.0}
    p.update(params or {})
    n = SimpleNamespace(
        get_parameter=lambda k: SimpleNamespace(value=p[k]),
        get_logger=lambda: Mock(),
        _run_info=Mock(),
        _fresh_marker_centre=Mock(return_value=centre),
        _wait_for_user=Mock(return_value=True),
        _axis_lin=Mock(return_value=lin_ok),
        _get_tip_offset=lambda: {"x": 0.108, "y": -0.008, "z": -0.411})
    return n


Q = (0.0, 0.0, 0.0, 1.0)
EE = (0.5, 0.0, 0.3)


def _run(n, execute=True):
    return A._refine_checkpoint(n, "C1", (0.55, -0.07), Q, EE, 0.02, execute)


class Decision(unittest.TestCase):
    def test_none(self):
        self.assertEqual(insertion.refine_decision((0, 0), None, 1, 15)[0], "none")

    def test_below_min(self):
        self.assertEqual(insertion.refine_decision((0, 0), (0.0005, 0.0), 1, 15)[0], "skip")

    def test_apply(self):
        v, dx, dy = insertion.refine_decision((0.5, 0.1), (0.505, 0.097), 1, 15)
        self.assertEqual(v, "apply")
        self.assertAlmostEqual(dx, 0.005)
        self.assertAlmostEqual(dy, -0.003)

    def test_above_max(self):
        self.assertEqual(insertion.refine_decision((0, 0), (0.02, 0.0), 1, 15)[0], "abort")

    def test_boundaries_inclusive_apply(self):
        self.assertEqual(insertion.refine_decision((0, 0), (0.001, 0), 1, 15)[0], "apply")
        self.assertEqual(insertion.refine_decision((0, 0), (0.015, 0), 1, 15)[0], "apply")


class Checkpoint(unittest.TestCase):
    def test_disabled_never_measures(self):
        for params in ({"refine_target": False}, {"target_source": "params"}):
            n = _node(params, centre=(0.56, -0.07))
            self.assertEqual(_run(n)[0], "inactive")
            n._fresh_marker_centre.assert_not_called()
            n._axis_lin.assert_not_called()
            n._run_info.assert_not_called()

    def test_dry_run_inactive(self):
        n = _node(centre=(0.56, -0.07))
        self.assertEqual(_run(n, execute=False)[0], "inactive")
        n._fresh_marker_centre.assert_not_called()

    def test_none_keeps_target(self):
        n = _node(centre=None)
        self.assertEqual(_run(n), ("skip", 0.0, 0.0))
        n._axis_lin.assert_not_called()

    def test_below_min_no_move(self):
        n = _node(centre=(0.5505, -0.07))
        self.assertEqual(_run(n)[0], "skip")
        n._axis_lin.assert_not_called()

    def test_apply_moves_xy_only(self):
        n = _node(centre=(0.555, -0.072))
        st, dx, dy = _run(n)
        self.assertEqual(st, "applied")
        self.assertAlmostEqual(dx, 0.005)
        self.assertAlmostEqual(dy, -0.002)
        (ee, q, label, speed, _rot), _ = n._axis_lin.call_args
        self.assertAlmostEqual(ee[0], EE[0] + 0.005)
        self.assertAlmostEqual(ee[1], EE[1] - 0.002)
        self.assertEqual(ee[2], EE[2])
        self.assertEqual((q, label, speed), (Q, "RefineC1", 0.02))
        ev = n._run_info.call_args
        self.assertEqual(ev.args[0], "target_refine")
        self.assertTrue(ev.kwargs["applied"])

    def test_above_max_aborts_without_moving(self):
        n = _node(centre=(0.58, -0.07))
        self.assertEqual(_run(n)[0], "abort")
        n._axis_lin.assert_not_called()

    def test_lin_failure_aborts(self):
        n = _node(centre=(0.555, -0.07), lin_ok=False)
        self.assertEqual(_run(n)[0], "abort")

    def test_declined_step_does_not_shift(self):
        n = _node(centre=(0.555, -0.07))
        n._wait_for_user.return_value = False
        self.assertEqual(_run(n)[0], "skip")
        n._axis_lin.assert_not_called()


class Translation(unittest.TestCase):
    def test_equals_recompute(self):
        n = _node()
        tilt, az, ready_z, tz = math.radians(45), math.radians(120), 0.2, 0.05
        qv = (0.7071, 0.7071, 0.0, 0.0)
        qt = insertion._tilt_quaternion(qv, az, tilt)
        ee = lambda tip, q: A._ee_for_tip(n, tip, q)

        def build(cx, cy):
            tgt = (cx, cy, tz)
            hov, _ax, _D = insertion_axis_geometry(cx, cy, ready_z, tz, tilt, az)
            return dict(target=tgt, hover=hov, ee_hover=ee(hov, qv),
                        end=ee(hov, qt), ready=ee((cx, cy, ready_z), qv))

        old, new = build(0.55, -0.07), build(0.5563, -0.0641)
        shifted = insertion.translate_waypoints(old, 0.0063, 0.0059)
        for k in old:
            for a, b in zip(shifted[k], new[k]):
                self.assertAlmostEqual(a, b, places=9)


if __name__ == "__main__":
    unittest.main()
