#!/usr/bin/env python3
"""Offline tests for trajectory_table_check.py. No robot, no live ROS graph.

    source /opt/ros/humble/setup.bash
    python3 test_trajectory_table_check.py
    TABLE_CHECK_BAGS=~/robot-logs/insertion_bags python3 test_trajectory_table_check.py

With TABLE_CHECK_BAGS set, every planned trajectory in every archived insertion bag
(all phases, including the insertion itself) is replayed through the check. Those
motions ran on hardware without touching the table, so all of them must pass.
"""
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from trajectory_table_check import (TableClearanceChecker, TableCheckRejected,  # noqa: E402
                                    TABLE_Z, LINKS)

URDF = (HERE.parent / "config" / "curobo" / "gen3_surgical.urdf").read_text()
HOME = np.array([0.0, -0.3049, -3.1416, -1.6607, 0.0, -1.7928, -0.0006])
J = [f"joint_{i}" for i in range(1, 8)]


def ramp(q0, q1, n=50):
    return [q0 + (q1 - q0) * s for s in np.linspace(0, 1, n)]


class Checker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TableClearanceChecker(URDF, "thesis_ee")

    def tip_clear(self, q):
        return self.c.clearances(q)[1]

    def test_home_passes(self):
        rep = self.c.check(J, [HOME])
        self.assertGreater(rep["min_clear_m"], 0.2)

    def test_dive_into_table_rejected(self):
        # Pitch the shoulder forward until the tool tip is below the table.
        q1 = HOME.copy()
        while self.tip_clear(q1) > -0.05:
            q1[1] += 0.02
        with self.assertRaisesRegex(TableCheckRejected, "Trajectory NOT sent"):
            self.c.check(J, ramp(HOME, q1))

    def test_densification_catches_a_dip_between_waypoints(self):
        # Two waypoints that both clear the table, but the straight joint-space
        # path between them dips through it. Only densification can see that.
        q_mid = HOME.copy()
        while self.tip_clear(q_mid) > -0.03:
            q_mid[1] += 0.02
        rng = np.random.default_rng(3)
        for _ in range(2000):
            d = rng.normal(size=7); d *= rng.uniform(0.3, 1.5) / np.linalg.norm(d)
            x, y = q_mid - d, q_mid + d
            if min(self.tip_clear(x), self.tip_clear(y),
                   self.c.clearances(x)[0].min(), self.c.clearances(y)[0].min()) > 0.03:
                break
        else:
            self.fail("could not construct endpoints that clear the table")
        self.c.check(J, [x]); self.c.check(J, [y])        # each endpoint alone is fine
        with self.assertRaisesRegex(TableCheckRejected, "NOT sent"):
            self.c.check(J, [x, y])

    def test_started_inside_margin(self):
        low = HOME.copy()
        while self.tip_clear(low) > 0.006:
            low[1] += 0.002
        rep = self.c.check(J, ramp(low, HOME))          # moving up and away: allowed
        self.assertTrue(rep["started_inside_margin"])
        deeper = low.copy()
        while self.tip_clear(deeper) > 0.002:
            deeper[1] += 0.002
        with self.assertRaisesRegex(TableCheckRejected, "may not get closer"):
            self.c.check(J, ramp(low, deeper))           # getting closer: refused

    def test_margin_boundary(self):
        q1 = HOME.copy()
        while self.tip_clear(q1) > 0.005:
            q1[1] += 0.002
        # Approaching from outside the margin and ending 5 mm above the table:
        c0 = TableClearanceChecker(URDF, "thesis_ee", margin=0.0)
        c0.check(J, ramp(HOME, q1))                       # 0 mm margin: allowed
        with self.assertRaisesRegex(TableCheckRejected, "limit is 10.0 mm"):
            self.c.check(J, ramp(HOME, q1))               # 10 mm margin: refused

    def test_joint_order_and_extra_joints(self):
        names = list(reversed(J)) + ["finger_joint"]
        pos = [list(reversed(HOME)) + [0.3]]
        self.assertAlmostEqual(self.c.check(names, pos)["min_clear_m"],
                               self.c.check(J, [HOME])["min_clear_m"])

    def test_missing_joint_and_nan(self):
        with self.assertRaises(TableCheckRejected):
            self.c.check(J[:6], [HOME[:6]])
        bad = HOME.copy(); bad[2] = math.nan
        with self.assertRaises(TableCheckRejected):
            self.c.check(J, [bad])

    def test_tool_policy(self):
        with self.assertRaisesRegex(TableCheckRejected, "no tool hull"):
            TableClearanceChecker(URDF, "2f140")
        bare = TableClearanceChecker(URDF, "none")
        link, tool, _ = bare.clearances(HOME)
        self.assertEqual(tool, math.inf)
        # With no tool modelled, a pose that buries the needle still passes
        # (only links are guarded) -- which is exactly why 2f140 is refused.
        q1 = HOME.copy()
        while self.tip_clear(q1) > -0.05:
            q1[1] += 0.02
        if bare.clearances(q1)[0].min() > bare.margin:
            bare.check(J, [q1])

    def test_hull_vertex_is_assembly_tip(self):
        from arm_kinematics import KinDynModel
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(URDF)
        try:
            tip = KinDynModel(f.name, "base_link", "assembly_tip")
        finally:
            os.unlink(f.name)
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = HOME + rng.uniform(-0.6, 0.6, 7)
            p, _ = tip.fk(q)
            _, tool, _ = self.c.clearances(q)
            self.assertLessEqual(tool, p[2] - TABLE_Z + 1e-12)   # tip is a hull vertex

    def test_links_match_impedance_barrier(self):
        """Same per-link heights as the impedance barrier's height_barrier_terms."""
        try:
            from arm_kinematics import KinDynModel, DEFAULT_URDF
        except ImportError:
            self.skipTest("arm_kinematics unavailable")
        imp = KinDynModel(DEFAULT_URDF, "gen3_base_link", "gen3_end_effector_link")
        segs = [imp.segment_index("gen3_" + n) for n in LINKS]
        rng = np.random.default_rng(1)
        for _ in range(50):
            q = HOME + rng.uniform(-1.0, 1.0, 7)
            z_imp, _, _ = imp.height_barrier_terms(q, np.zeros(7), segs)
            mine, _, _ = self.c.clearances(q)
            np.testing.assert_allclose(mine, z_imp - imp.CAPSULE_RADIUS - TABLE_Z, atol=1e-6)


class InsertionWiring(unittest.TestCase):
    """Drive insertion.py's real execute functions with a stub node: a rejected
    trajectory must never be published or sent to the controller."""

    def setUp(self):
        try:
            import insertion
            from moveit_msgs.msg import RobotTrajectory
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError:
            self.skipTest("needs ROS sourced")
        from unittest.mock import MagicMock
        self.ins = insertion
        cls = next(v for v in vars(insertion).values()
                   if isinstance(v, type) and hasattr(v, "_execute_fjt_inner"))
        self.cls = cls
        params = {"table_check": True, "table_check_tool": "thesis_ee",
                  "table_check_margin": 0.01}
        node = MagicMock()
        node.get_parameter = lambda n: type("P", (), {"value": params[n]})()
        node._robot_description_xml = URDF
        node._table_checker = None
        node._live_joints = dict(zip(J, HOME))
        node._unwrap_trajectory = lambda traj, ref_joints=None: traj
        node._clamp_traj = lambda traj: traj
        node._get_table_checker = lambda: cls._get_table_checker(node)
        node._table_check_ok = lambda traj, label: cls._table_check_ok(node, traj, label)
        self.node, self.params = node, params

        def traj(qs):
            tr = RobotTrajectory(); tr.joint_trajectory.joint_names = J
            for i, q in enumerate(qs):
                pt = JointTrajectoryPoint(); pt.positions = [float(x) for x in q]
                pt.time_from_start.sec = i
                tr.joint_trajectory.points.append(pt)
            return tr
        self.traj = traj
        low = HOME.copy()
        c = TableClearanceChecker(URDF, "thesis_ee")
        while c.clearances(low)[1] > -0.05:
            low[1] += 0.02
        self.bad = traj(ramp(HOME, low, 20))
        self.good = traj(ramp(HOME, HOME + 0.1, 20))

    def test_fjt_rejects_before_publish(self):
        self.assertFalse(self.cls._execute_fjt_inner(self.node, self.bad, "t", 1.0))
        self.node._traj_pub.publish.assert_not_called()
        self.node._fjt_cli.send_goal_async.assert_not_called()

    def test_moveit_rejects_before_publish(self):
        self.assertFalse(self.cls._execute_moveit_inner(self.node, self.bad, "t", 1.0))
        self.node._traj_pub.publish.assert_not_called()
        self.node._execute_cli.send_goal_async.assert_not_called()

    def test_good_trajectory_reaches_publish(self):
        self.node._fjt_cli.server_is_ready.return_value = False   # stop right after publish
        self.cls._execute_fjt_inner(self.node, self.good, "t", 1.0)
        self.node._traj_pub.publish.assert_called_once()

    def test_unguardable_tool_refused(self):
        self.params["table_check_tool"] = "2f140"
        self.assertRaises(TableCheckRejected, self.cls._get_table_checker, self.node)

    def test_off_switch(self):
        self.params["table_check"] = False
        self.node._fjt_cli.server_is_ready.return_value = False
        self.cls._execute_fjt_inner(self.node, self.bad, "t", 1.0)
        self.node._traj_pub.publish.assert_called_once()

    def test_kortex_lin_refused_while_check_on(self):
        self.node._motion_blocked = lambda label: False
        self.assertFalse(self.cls._execute_kortex_lin(self.node, 0.3, 0., 0.3, (0, 0, 0, 1), "lin"))
        self.node._kortex_base.ExecuteAction.assert_not_called()


class Archive(unittest.TestCase):
    def test_every_archived_trajectory_passes(self):
        root = os.environ.get("TABLE_CHECK_BAGS")
        if not root:
            self.skipTest("TABLE_CHECK_BAGS not set")
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from moveit_msgs.msg import RobotTrajectory
        c = TableClearanceChecker(URDF, "thesis_ee")
        n, worst = 0, (math.inf, None)
        for meta in sorted(Path(root).expanduser().glob("*/metadata.yaml")):
            r = rosbag2_py.SequentialReader()
            r.open(rosbag2_py.StorageOptions(uri=str(meta.parent), storage_id="sqlite3"),
                   rosbag2_py.ConverterOptions("", ""))
            r.set_filter(rosbag2_py.StorageFilter(topics=["/insertion/planned_trajectory"]))
            k = 0
            while r.has_next():
                _, data, _ = r.read_next()
                jt = deserialize_message(data, RobotTrajectory).joint_trajectory
                with self.subTest(bag=meta.parent.name, traj=k):
                    rep = c.check(jt.joint_names, [p.positions for p in jt.points])
                    if rep["min_clear_m"] < worst[0]:
                        worst = (rep["min_clear_m"], f"{meta.parent.name}#{k} {rep['worst']['body']}")
                n += 1; k += 1
        print(f"\n  {n} archived trajectories; closest approach {worst[0] * 1000:.1f} mm ({worst[1]})")
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
