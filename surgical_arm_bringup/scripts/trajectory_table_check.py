#!/usr/bin/env python3
"""Pre-execution table-clearance check for planned (position-controlled) motion.

The impedance table barrier (impedance.TableSafetyFilter) only protects torque
mode. Everything insertion.py sends -- Pilz, OMPL, cuRobo -- goes out as a
joint trajectory to the position controller, and until now the only thing
between it and the table was MoveIt's plan-time collision check against a tool
mesh known to differ from the CAD. This module is an independent backstop: it
runs forward kinematics along the whole trajectory and refuses it if any
guarded body would come closer than `margin` to the table.

Same geometry as the impedance barrier, deliberately:
  * links: the six distal links (impedance.DEFAULT_TABLE_LINKS), each guarded
    as a sphere of KinDynModel.CAPSULE_RADIUS about its origin;
  * tool: the convex hull from impedance.TOOL_HULLS, in the bracelet frame.
    A plane's closest approach to a convex body is a hull vertex, so checking
    the vertices is exact for the tool;
  * table: z = -0.03 in base frame (setup_planning_scene.py).

Tool policy is fail-closed: a tool with a hull is checked; "none" checks the
links only; any other tool (e.g. "2f140") is REFUSED, because the hull that
would make it safe does not exist yet. Pass the right `tool`, not "none", to
get past that.

If the arm STARTS inside the margin (e.g. parked low after a jog), refusing
every motion would trap it there, including the move up and away. The rule is
then "never closer than the start": the threshold drops to the start's own
clearance minus `start_tol` (2 mm). That is reported as started_inside_margin.
Two archived 2026-10-10 home moves started 5.3 and 8.3 mm above the table.

Between waypoints the path is densified so that no joint moves more than
`max_step_rad` per sample (default 0.01 rad, which is 4 mm at the 0.41 m tool
tip). Sampled, not a swept-volume certificate.

ROS-free: PyKDL + urdf_parser_py only (via arm_kinematics.KinDynModel).
"""
import math
import os
import tempfile

import numpy as np

from arm_kinematics import KinDynModel

TABLE_Z = -0.03
LINKS = ("half_arm_2_link", "forearm_link", "spherical_wrist_1_link",
         "spherical_wrist_2_link", "bracelet_link", "end_effector_link")
HULL_LINK = "bracelet_link"
_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "config")
TOOL_HULLS = {"thesis_ee": os.path.join(_CONFIG, "thesis_ee_hull_bracelet.csv")}
LINKS_ONLY_TOOLS = ("none",)


class TableCheckRejected(ValueError):
    pass


def load_hull(tool):
    if tool in LINKS_ONLY_TOOLS:
        return None
    path = TOOL_HULLS.get(tool)
    if path is None:
        raise TableCheckRejected(
            f"table check: no tool hull for '{tool}', so the tool cannot be guarded. "
            f"Known: {sorted(TOOL_HULLS)} (or 'none' for a bare flange).")
    pts = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if pts.shape[1] != 3 or not len(pts):
        raise ValueError(f"{path}: expected rows of x,y,z")
    return pts


class TableClearanceChecker:
    def __init__(self, urdf_xml, tool, *, base_link="base_link",
                 tip_link="end_effector_link", table_z=TABLE_Z, margin=0.01,
                 link_radius=KinDynModel.CAPSULE_RADIUS, max_step_rad=0.01,
                 links=LINKS, start_tol=0.002):
        self.start_tol = start_tol
        self.hull = load_hull(tool)
        self.tool = tool
        self.table_z, self.margin = table_z, margin
        self.link_radius, self.max_step_rad = link_radius, max_step_rad
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(urdf_xml)
            path = f.name
        try:
            self.model = KinDynModel(path, base_link, tip_link)
        finally:
            os.unlink(path)
        names = self.model.segment_names()
        missing = [n for n in links if n not in names]
        if missing:
            raise TableCheckRejected(f"table check: links {missing} not on {base_link}->{tip_link}")
        self.links = list(links)
        self.link_segs = [self.model.segment_index(n) for n in links]
        self.hull_seg = self.model.segment_index(HULL_LINK)
        self.joint_names = [j.getName() for j in
                            (self.model.chain.getSegment(s).getJoint()
                             for s in range(self.model.chain.getNrOfSegments()))
                            if j.getTypeName() not in ("None", "Fixed")]

    def clearances(self, q):
        """(per-link surface clearance [6], tool clearance or inf, lowest hull index)."""
        kdl = self.model.kdl
        jnt = self.model._to_jnt(q)
        fr = kdl.Frame()
        link = []
        for s in self.link_segs:
            self.model._fk.JntToCart(jnt, fr, s)
            link.append(fr.p[2] - self.link_radius - self.table_z)
        tool, idx = math.inf, -1
        if self.hull is not None:
            self.model._fk.JntToCart(jnt, fr, self.hull_seg)
            r_z = np.array([fr.M[2, j] for j in range(3)])
            z = fr.p[2] + self.hull @ r_z - self.table_z
            idx = int(np.argmin(z))
            tool = float(z[idx])
        return np.array(link), tool, idx

    def check(self, joint_names, positions, times=None):
        """positions: (N, n) in joint_names order. Returns a report dict, or raises."""
        P = np.asarray(positions, float)
        if P.ndim != 2 or len(P) < 1:
            raise TableCheckRejected("table check: empty trajectory")
        try:
            order = [list(joint_names).index(n) for n in self.joint_names]
        except ValueError as e:
            raise TableCheckRejected(f"table check: trajectory lacks a chain joint ({e})")
        P = P[:, order]
        if not np.isfinite(P).all():
            raise TableCheckRejected("table check: non-finite joint position")
        worst = {"clear_m": math.inf}
        n_samples = 0
        l0, t0, _ = self.clearances(P[0])
        start_clear = min(float(l0.min()), t0)
        threshold = self.margin
        if start_clear < self.margin:
            threshold = start_clear - self.start_tol
        for k in range(len(P)):
            seg = [P[k]] if k == 0 else None
            if k:
                n = max(1, int(math.ceil(np.max(np.abs(P[k] - P[k - 1])) / self.max_step_rad)))
                seg = [P[k - 1] + (P[k] - P[k - 1]) * (i / n) for i in range(1, n + 1)]
            for q in seg:
                n_samples += 1
                link, tool, idx = self.clearances(q)
                j = int(np.argmin(link))
                for body, c in ((self.links[j], float(link[j])),
                                (f"tool_hull[{idx}]", tool)):
                    if c < worst["clear_m"]:
                        worst = {"clear_m": c, "body": body, "waypoint": k,
                                 "t": None if times is None else float(times[k])}
        report = {"tool": self.tool, "margin_m": self.margin, "samples": n_samples,
                  "min_clear_m": worst["clear_m"], "worst": worst,
                  "start_clear_m": start_clear, "threshold_m": threshold,
                  "started_inside_margin": start_clear < self.margin}
        if worst["clear_m"] < threshold:
            where = f"waypoint {worst['waypoint']}" + (
                f" (t={worst['t']:.2f} s)" if worst.get("t") is not None else "")
            raise TableCheckRejected(
                f"table check: {worst['body']} comes {worst['clear_m'] * 1000:.1f} mm from the "
                f"table at {where}; limit is {threshold * 1000:.1f} mm"
                + (f" (started inside the {self.margin * 1000:.0f} mm margin at "
                   f"{start_clear * 1000:.1f} mm, so it may not get closer)"
                   if start_clear < self.margin else "")
                + ". Trajectory NOT sent.")
        return report


def check_joint_trajectory(checker, jt):
    """Convenience for a trajectory_msgs/JointTrajectory."""
    pts = jt.points
    return checker.check(jt.joint_names, [p.positions for p in pts],
                         [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts])
