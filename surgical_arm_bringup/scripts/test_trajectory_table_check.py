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
                                    TABLE_Z, LINKS, target_tip_floor)

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
            TableClearanceChecker(URDF, "some_new_gripper")
        bare = TableClearanceChecker(URDF, "none")
        link, tool, _ = bare.clearances(HOME)
        self.assertEqual(tool, math.inf)
        # With no tool modelled, a pose that buries the needle still passes
        # (only links are guarded) -- which is exactly why unknown tools are refused.
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


class TipFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TableClearanceChecker(URDF, "thesis_ee")

    @classmethod
    def tip_model(cls):
        if not hasattr(cls, "_tip"):
            from arm_kinematics import KinDynModel
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
                f.write(URDF)
            try:
                cls._tip = KinDynModel(f.name, "base_link", "assembly_tip")
            finally:
                os.unlink(f.name)
        return cls._tip

    def lower_until(self, clear):
        """Bring the tip straight down, tool kept vertical (resolved-rate IK),
        until it is `clear` above the table -- the insertion's geometry."""
        m, q = self.tip_model(), HOME.copy()
        for _ in range(5000):
            tip = self.c._split(q)[2]
            if tip <= clear:
                return q
            step = min(0.002, tip - clear + 1e-4)
            J = m.jacobian(q)
            q = q + np.linalg.pinv(J) @ np.array([0, 0, -step, 0, 0, 0])
        self.fail("could not lower the tip")

    def test_tip_vertex_is_found(self):
        self.assertIsNotNone(self.c.tip_idx)
        np.testing.assert_allclose(self.c.hull[self.c.tip_idx], (0.108, -0.008, -0.411))

    def test_tip_may_reach_its_floor_but_not_below(self):
        q6 = self.lower_until(0.006)                     # tip 6 mm above the table
        with self.assertRaises(TableCheckRejected):
            self.c.check(J, ramp(HOME, q6))              # no floor: 10 mm margin applies
        rep = self.c.check(J, ramp(HOME, q6), tip_floor=0.005)
        self.assertEqual(rep["tip_floor_m"], 0.005)
        with self.assertRaisesRegex(TableCheckRejected, "tool_tip"):
            self.c.check(J, ramp(HOME, q6), tip_floor=0.007)

    def test_floor_never_raises_the_limit(self):
        q = self.lower_until(0.012)
        self.c.check(J, ramp(HOME, q), tip_floor=0.05)  # capped at the 10 mm margin

    def test_other_hull_vertices_keep_the_margin(self):
        # Tip lifted out of the way, rest of the tool driven low: the floor
        # must not help, because only the tip vertex is relaxed.
        rng = np.random.default_rng(5)
        for _ in range(4000):
            q = HOME + rng.uniform(-2.0, 2.0, 7)
            b, name, tip = self.c._split(q)
            if name == "tool_hull" and 0.0 < b < 0.008 and tip > 0.02:
                with self.assertRaisesRegex(TableCheckRejected, "tool_hull"):
                    self.c.check(J, [HOME, q], tip_floor=0.0)
                return
        self.skipTest("no pose with a low non-tip vertex found")

    def test_target_tip_floor(self):
        top = TABLE_Z + 0.005 + 0.086                    # board + container
        self.assertAlmostEqual(target_tip_floor(top - 0.030, TABLE_Z, 0.005, 0.003), 0.056)
        with self.assertRaisesRegex(TableCheckRejected, "before any motion"):
            target_tip_floor(TABLE_Z + 0.006, TABLE_Z, 0.005, 0.003)


URDF_2F140 = Path("/tmp/gen3_2f140.urdf")


class Gripper2F140(unittest.TestCase):
    """2F-140 hull. Uses the xacro output named in generate_tool_hull_2f140.py."""

    @classmethod
    def setUpClass(cls):
        if not URDF_2F140.exists():
            raise unittest.SkipTest(f"{URDF_2F140} missing; see generate_tool_hull_2f140.py")
        cls.c = TableClearanceChecker(URDF_2F140.read_text(), "2f140")

    def test_hull_covers_the_finger_pads_at_every_opening(self):
        from scipy.spatial import Delaunay
        import generate_tool_hull_2f140 as g
        from urdf_parser_py.urdf import URDF
        tri = Delaunay(self.c.hull)
        robot = URDF.from_xml_string(URDF_2F140.read_text())
        for f in np.linspace(0, 0.7, 8):
            T = g.link_transforms(robot, f)
            for pad in ("left_inner_finger_pad", "right_inner_finger_pad"):
                p = T[pad][:3, 3]
                self.assertGreaterEqual(tri.find_simplex(p), 0, f"{pad} at finger {f:.2f} outside hull")

    def test_no_tip_vertex_so_tip_floor_refused(self):
        self.assertIsNone(self.c.tip_idx)
        with self.assertRaisesRegex(TableCheckRejected, "tip vertex"):
            self.c.check(J, [HOME], tip_floor=0.02)

    def test_home_clears_and_dive_rejected(self):
        rep = self.c.check(J, [HOME])
        self.assertGreater(rep["min_clear_m"], 0.1)
        q = HOME.copy()
        while self.c._split(q)[0] > -0.03:
            q[1] += 0.02
        with self.assertRaisesRegex(TableCheckRejected, "NOT sent"):
            self.c.check(J, ramp(HOME, q))

    def test_gripper_is_what_hits_first_when_lowered_tool_down(self):
        # Resolved-rate descent with the flange orientation held: the fingers,
        # not the links, must be what the check stops.
        from arm_kinematics import KinDynModel
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(URDF_2F140.read_text())
        try:
            m = KinDynModel(f.name, "base_link", "bracelet_link")
        finally:
            os.unlink(f.name)
        q = HOME.copy()
        for _ in range(3000):
            b, name, _ = self.c._split(q)
            if b < 0.005:
                break
            q = q + np.linalg.pinv(m.jacobian(q)) @ np.array([0, 0, -0.002, 0, 0, 0])
        self.assertEqual(name, "tool_hull")
        with self.assertRaisesRegex(TableCheckRejected, "tool_hull"):
            self.c.check(J, ramp(HOME, q))
        bare = TableClearanceChecker(URDF_2F140.read_text(), "none")
        bare.check(J, ramp(HOME, q))           # links alone would have let it through

    def test_fingers_reach_lower_than_the_bare_flange(self):
        bare = TableClearanceChecker(URDF_2F140.read_text(), "none")
        b_bare = bare._split(HOME)[0]
        b_grip = self.c._split(HOME)[0]
        self.assertLess(b_grip, b_bare - 0.05)    # ~80 mm below the lowest link surface at home


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
                  "table_check_margin": 0.01, "table_z": TABLE_Z}
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
        self.params["table_check_tool"] = "some_new_gripper"
        self.assertRaises(TableCheckRejected, self.cls._get_table_checker, self.node)

    def test_off_switch(self):
        self.params["table_check"] = False
        self.node._fjt_cli.server_is_ready.return_value = False
        self.cls._execute_fjt_inner(self.node, self.bad, "t", 1.0)
        self.node._traj_pub.publish.assert_called_once()

    def test_floor_only_for_insertion_labels(self):
        from unittest.mock import patch
        import trajectory_table_check as ttc
        self.node._insert_tip_floor = 0.04
        seen = {}
        real = ttc.check_joint_trajectory

        def spy(checker, jt, tip_floor=None):
            seen["floor"] = tip_floor
            return real(checker, jt, tip_floor)
        with patch.object(ttc, "check_joint_trajectory", spy):
            self.cls._table_check_ok(self.node, self.good, "Phase4b_insert")
            self.assertEqual(seen["floor"], 0.04)
            self.cls._table_check_ok(self.node, self.good, "Phase0_approach")
            self.assertIsNone(seen["floor"])
            self.cls._table_check_ok(self.node, self.good, "Phase3_tip_rotate")
            self.assertIsNone(seen["floor"])

    def test_low_target_refused_before_any_motion(self):
        import types
        vals = dict(transit_velocity_scaling=0.15, max_velocity_scaling=0.15, insert_speed_mps=0.01,
                    insert_rot_speed_dps=5., blend_transit=True, axis_speed_mps=0.02,
                    insert_slow_zone_m=0.01, real_robot=False, world_frame="world",
                    tip_link="assembly_tip", ee_link="end_effector_link", move_group_name="manipulator",
                    table_z=TABLE_Z, container_height=0.086, approach_clearance=0.18,
                    return_to_start=True, post_insert_wait=0., table_check=True,
                    container_base_offset=0.005, target_depth_mm=85.0,   # past the 39 mm wall limit
                    table_check_tip_allowance=0.005, table_check_tip_min_floor=0.003)
        self.node.get_parameter = lambda n: types.SimpleNamespace(value=vals[n])
        gh = types.SimpleNamespace(request=types.SimpleNamespace(dry_run=False, hover_above_top=0.03))
        with self.assertRaisesRegex(ValueError, "wall-clearance limit"):
            self.cls._run_phases(self.node, gh, lambda *a: None)
        # A short container: depth is legal for the wall, but the tip floor
        # would sit 0 mm above the table -> the 3 mm table minimum refuses it.
        vals.update(container_height=0.030, target_depth_mm=30.0)   # floor lands on the table
        with self.assertRaisesRegex(TableCheckRejected, "before any motion"):
            self.cls._run_phases(self.node, gh, lambda *a: None)
        vals.update(container_height=0.086, target_depth_mm=30.0)   # today's task
        self.node._get_table_checker = lambda: (_ for _ in ()).throw(RuntimeError("stop here"))
        try:
            self.cls._run_phases(self.node, gh, lambda *a: None)
        except Exception:
            pass
        self.assertAlmostEqual(self.node._insert_tip_floor, 0.056, places=6)
        for m in ("_plan", "_execute_moveit", "_execute_fjt", "_plan_sequence", "_blended_transit"):
            getattr(self.node, m).assert_not_called()

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
