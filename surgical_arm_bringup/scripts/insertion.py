#!/usr/bin/env python3
"""
angled_insert.py — Rotate the assembly tip about its own point then descend
                   at the computed angle to reach a target inside the container.

Overview
--------
The script takes the container centre position and a user-specified target
point INSIDE the container (given in mm from the container centre).  It then:

  Phase 0 — Approach (Pilz PTP)
    Navigate assembly_tip to hover position directly above container centre,
    same as insert_to_container.py Phase 0.

  Phase 1 — Descend vertical (Pilz LIN)
    Drop straight down to hover_z (just above container top), tip still vertical.

  Phase 2 — Compute tilt geometry
    From the hover position and target point, calculate:
      azimuth  : compass direction to tilt toward (rotation about world Z)
      tilt     : angle from vertical (0° = straight down, 30° max)
      descent  : straight-line distance from hover_z to target

  Phase 3 — CIRC rotation about tip (Pilz CIRC)
    Rotate the EE in a circular arc that keeps assembly_tip FIXED at hover_z
    while the wrist reorients to the computed tilt + azimuth.
    The via-point is at half the rotation angle (required by Pilz CIRC).

  Phase 4 — Angled descent (Pilz LIN along new tool axis)
    Descend in a straight line along the tilted tool axis from hover_z to
    the target point.  Distance = computed descent length.

  Phase 5 — Hold
    Wait for user to press Enter.

  Phase 6 — Reverse (Phases 4→3 in reverse order)
    Ascend along tool axis back to hover_z.
    CIRC arc back to vertical orientation.

  Phase 7 — Ascend vertical + Return
    LIN back to approach height, PTP return to home joints.

Coordinate convention for target point
---------------------------------------
The user specifies the target in millimetres from the container centre:
  offset_x_mm : + = away from robot base  (world +X direction)
  offset_y_mm : + = left, - = right        (world +Y direction)
  depth_mm    : depth below container top  (always positive, converted to -Z)

Container dimensions: 90 × 90 × 86 mm.  The target must be within:
  |offset_x| ≤ 40 mm, |offset_y| ≤ 40 mm (inner radius with 5 mm wall margin)
  0 < depth ≤ 80 mm

Safety limits
-------------
  MAX_TILT_DEG = 25°  — beyond this the tool may hit the container wall
  The script checks wall clearance at the target depth before executing.

Usage
-----
  # Standalone (default): runs ONE insertion from parameters, then exits.
  # Dry-run first (plans, no motion):
  ros2 run surgical_arm_bringup insert_to_container.py \\
      --ros-args -p real_robot:=true \\
      -p target_x:=0.45 -p target_y:=-0.20 \\
      -p insertion_angle_deg:=45.0

  # Execute for real (add execute_motion:=true):
  ros2 run surgical_arm_bringup insert_to_container.py \\
      --ros-args -p real_robot:=true -p execute_motion:=true \\
      -p target_x:=0.45 -p target_y:=-0.20 \\
      -p insertion_angle_deg:=45.0 -p target_depth_mm:=30.0

  # Go to calibrated vertical home first (recommended for a true 45° from vertical):
  #   omit skip_home_move (defaults true) -> set it false:
  #   -p skip_home_move:=false

  # Action-server mode (legacy): wait for goals on /insert_container instead:
  ros2 run surgical_arm_bringup insert_to_container.py \\
      --ros-args -p use_action_server:=true -p real_robot:=true
"""

import math
from robot_model_parser import get_robot_info
from kortex_utils import *
import os
import signal
import sys
import threading
import time
import subprocess
import collections
import collections.abc
import json

import rclpy
import rclpy.time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse

try:
    from surgical_arm_bringup.action import InsertContainer
    _HAS_ACTION_INTERFACE = True
except ImportError:
    _HAS_ACTION_INTERFACE = False

import tf2_ros
from geometry_msgs.msg import Pose, Quaternion, Point
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetMotionPlan, GetMotionSequence, GetPositionIK
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from moveit_msgs.action import ExecuteTrajectory
from builtin_interfaces.msg import Duration as RosDuration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from moveit_msgs.msg import (
    RobotState, Constraints, OrientationConstraint,
    MotionPlanRequest, WorkspaceParameters,
    PositionConstraint, BoundingVolume, JointConstraint,
    RobotTrajectory, MotionSequenceItem,
)
from shape_msgs.msg import SolidPrimitive

# Kortex API 2.6.0 ships protobuf 3.5.1, which still imports these aliases
# from collections.  Python 3.10 moved them to collections.abc.
for _name in ("MutableMapping", "Mapping", "Sequence", "MutableSequence",
              "Callable", "Iterable"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

try:
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.messages import Session_pb2, Base_pb2
    _HAS_KORTEX_API = True
except ImportError:
    _HAS_KORTEX_API = False

def create_kortex_client(ip, user, pw):
    transport = TCPTransport()
    router = RouterClient(transport, RouterClient.basicErrorCallback)
    transport.connect(ip, 10000)
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = user
    session_info.password = pw
    session_info.session_inactivity_timeout = 60000
    session_info.connection_inactivity_timeout = 2000
    session_manager = SessionManager(router)
    session_manager.CreateSession(session_info)
    base = BaseClient(router)
    return transport, router, session_manager, base

# Re-use constants and helpers from insert_to_container
# (assumes both scripts are in the same package)




# Container physical dimensions (metres)
CONTAINER_WIDTH_M  = 0.090   # 90 mm
CONTAINER_HEIGHT_M = 0.086   # 86 mm
CONTAINER_WALL_M   = 0.003   # 3 mm safety margin from inner wall
CONTAINER_INNER_R  = (CONTAINER_WIDTH_M / 2.0) - CONTAINER_WALL_M  # 42 mm

MAX_TILT_DEG = 25.0   # hard limit — beyond this hits the wall at shallow depths

# ee_link (bracelet_link) → assembly_tip offset, expressed in the ee_link frame.
# Mirrors surgical_arm_description/thesis_ee/urdf/thesis_ee_macro.xacro:
#   <joint name="assembly_tip_joint"> <origin xyz="0.108 -0.008 -0.411"/>
# If you change the xacro, change this too.






# ---------------------------------------------------------------------------
# Quaternion helpers (duplicated from insert_to_container for standalone use)
# ---------------------------------------------------------------------------

def _quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])

def _quat_normalize(q):
    n = math.sqrt(sum(v*v for v in q))
    return tuple(v/n for v in q)

def _quat_slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    dot = sum(a*b for a, b in zip(q0, q1))
    # Ensure shortest path
    if dot < 0.0:
        q1 = tuple(-v for v in q1)
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        # Linear interpolation for nearly identical quaternions
        result = tuple(a + t*(b-a) for a, b in zip(q0, q1))
        return _quat_normalize(result)
    theta_0 = math.acos(dot)
    theta   = theta_0 * t
    sin_theta   = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize(tuple(s0*a + s1*b for a, b in zip(q0, q1)))

def _quat_from_axis_angle(axis_xyz, angle_rad):
    """Build unit quaternion from rotation axis and angle."""
    ax, ay, az = axis_xyz
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    ax, ay, az = ax/n, ay/n, az/n
    s = math.sin(angle_rad / 2.0)
    c = math.cos(angle_rad / 2.0)
    return (ax*s, ay*s, az*s, c)

def _tilt_quaternion(q_vertical, azimuth_rad, tilt_rad):
    """
    Compute the new EE quaternion after tilting the tool axis by tilt_rad
    toward azimuth_rad, starting from q_vertical (tool pointing straight down).

    The rotation is composed as two successive rotations applied to q_vertical:
      1. Rotate about world Z by azimuth (point the tilt direction)
      2. Rotate about the new X axis by tilt (lean the tool)

    This keeps the tip fixed while the wrist reorients.
    """
    # Step 1: rotation about world Z by azimuth
    q_az = _quat_from_axis_angle((0, 0, 1), azimuth_rad)
    # Step 2: tilt the (downward) tool axis toward the azimuth direction.
    # The tilt axis is horizontal, 90° clockwise from the azimuth heading, so
    # that azimuth=0 leans the tool toward world +X (matching the geometry in
    # _run_impl, which places hover on the −X side and descends toward +X).
    # The previous (-sin, +cos) axis leaned toward −X — the opposite way —
    # which put the wrist on the wrong side and made the EE pose unreachable.
    tilt_ax = (math.sin(azimuth_rad), -math.cos(azimuth_rad), 0.0)
    q_tilt  = _quat_from_axis_angle(tilt_ax, tilt_rad)
    # Compose: q_new = q_tilt * q_az * q_vertical
    q_new = quat_multiply(q_tilt, quat_multiply(q_az, q_vertical))
    return _quat_normalize(q_new)


# ---------------------------------------------------------------------------
# Insertion-axis geometry and approach-direction (azimuth) ranking
# ---------------------------------------------------------------------------

# Gen3 7-DOF position limits of the bounded joints [rad]; 1/3/5/7 are continuous.
GEN3_BOUNDED_LIMITS = {"joint_2": 2.24, "joint_4": 2.57, "joint_6": 2.09}


def insertion_axis_geometry(cont_x, cont_y, ready_z, target_z, tilt_rad, azimuth_rad):
    """Tool axis, descent length and hover point for one approach azimuth.

    The tip ends at (cont_x, cont_y, target_z).  It starts at the hover point,
    which lies back along the tilted tool axis at height ready_z.
    Returns (hover_xyz, tool_axis, descent_m).
    """
    axis = (math.sin(tilt_rad) * math.cos(azimuth_rad),
            math.sin(tilt_rad) * math.sin(azimuth_rad),
            -math.cos(tilt_rad))
    cos_t = math.cos(tilt_rad)
    descent = (ready_z - target_z) / cos_t if cos_t > 1e-3 else 0.0
    hover = (cont_x - axis[0] * descent, cont_y - axis[1] * descent, ready_z)
    return hover, axis, descent


def unwrap_to_seed(solution, seed, joint_names, continuous_joints):
    """Shift each continuous joint by 2*pi multiples to sit nearest the seed."""
    out = []
    for name, value, ref in zip(joint_names, solution, seed):
        if name in continuous_joints:
            value += 2.0 * math.pi * round((ref - value) / (2.0 * math.pi))
        out.append(value)
    return out


def rank_azimuth_candidates(start, solved, joint_names, continuous_joints,
                            limits=None, limit_margin=0.10):
    """Order approach azimuths by how little the arm has to move.

    start:  joint vector the arm is at now.
    solved: {azimuth_deg: [q_ready, q_hover, q_tilted, q_target]} -- the IK
            chain for that azimuth, or None where some waypoint has no solution.
    A candidate is dropped if any waypoint puts a bounded joint within
    limit_margin of its limit.  The rest are sorted by the largest travel of
    any single joint over start -> ready -> hover -> tilted -> target, then by
    total travel.  Continuous joints are compared modulo 2*pi.

    Returns [{"azimuth_deg", "max_travel", "total_travel", "limit_margin",
    "limit_joint"}].
    """
    limits = GEN3_BOUNDED_LIMITS if limits is None else limits
    ranked = []
    for azimuth_deg, chain in solved.items():
        if not chain:
            continue
        previous = list(start)
        travel = [0.0] * len(joint_names)
        margin, tightest = float("inf"), None
        for waypoint in chain:
            q = unwrap_to_seed(waypoint, previous, joint_names, continuous_joints)
            for i, name in enumerate(joint_names):
                travel[i] += abs(q[i] - previous[i])
                if name in limits and limits[name] - abs(q[i]) < margin:
                    margin, tightest = limits[name] - abs(q[i]), name
            previous = q
        if margin < limit_margin:
            continue
        ranked.append({"azimuth_deg": azimuth_deg, "max_travel": max(travel),
                       "total_travel": sum(travel), "limit_margin": margin,
                       "limit_joint": tightest})
    ranked.sort(key=lambda c: (round(c["max_travel"], 3), c["total_travel"]))
    return ranked


def robust_centre(samples, max_spread_m):
    """Median of repeated (x, y) centre measurements, or None if they disagree.

    Returns (x, y, spread) where spread is the largest distance of any sample
    from the median; None when there are no samples or spread > max_spread_m
    (the markers or the arm were still moving, or the detection is flickering
    between corner subsets).
    """
    if not samples:
        return None
    xs = sorted(p[0] for p in samples)
    ys = sorted(p[1] for p in samples)
    mid = len(samples) // 2
    if len(samples) % 2:
        mx, my = xs[mid], ys[mid]
    else:
        mx, my = (xs[mid - 1] + xs[mid]) / 2.0, (ys[mid - 1] + ys[mid]) / 2.0
    spread = max(math.hypot(x - mx, y - my) for x, y in samples)
    if spread > max_spread_m:
        return None
    return mx, my, spread


# ---------------------------------------------------------------------------
# Geometry: compute tilt from hover point to target
# ---------------------------------------------------------------------------

def compute_tilt_geometry(hover_xyz, target_xyz):
    """
    Given the hover position (tip directly above container centre) and
    the target point inside the container, compute:

      azimuth_rad  : rotation about world Z to face the target (0 = +X direction)
      tilt_rad     : angle from vertical toward target (0 = straight down)
      descent_m    : straight-line distance from hover to target
      tool_axis    : unit vector pointing from hover to target (world frame)

    Parameters
    ----------
    hover_xyz  : (x, y, z) world position of tip at hover height
    target_xyz : (x, y, z) world position of target inside container

    Returns dict with keys: azimuth_rad, tilt_rad, descent_m, tool_axis,
                             dx, dy, dz, horizontal_m
    """
    dx = target_xyz[0] - hover_xyz[0]
    dy = target_xyz[1] - hover_xyz[1]
    dz = target_xyz[2] - hover_xyz[2]   # always negative (going down)

    horizontal_m = math.sqrt(dx*dx + dy*dy)
    descent_m    = math.sqrt(dx*dx + dy*dy + dz*dz)

    if descent_m < 1e-4:
        return dict(azimuth_rad=0.0, tilt_rad=0.0, descent_m=0.0,
                    tool_axis=(0.0, 0.0, -1.0),
                    dx=dx, dy=dy, dz=dz, horizontal_m=0.0)

    # Azimuth: direction of horizontal offset in XY plane
    azimuth_rad = math.atan2(dy, dx)

    # Tilt: angle from world -Z toward target
    # dz is negative, so |dz| is the vertical drop
    tilt_rad = math.atan2(horizontal_m, abs(dz))

    # Unit vector pointing from hover to target
    tool_axis = (dx/descent_m, dy/descent_m, dz/descent_m)

    return dict(
        azimuth_rad  = azimuth_rad,
        tilt_rad     = tilt_rad,
        descent_m    = descent_m,
        tool_axis    = tool_axis,
        dx=dx, dy=dy, dz=dz,
        horizontal_m = horizontal_m,
    )


def check_wall_clearance(offset_x_m, offset_y_m, depth_m,
                          tilt_rad, tip_radius_m=0.003,
                          max_tilt_deg=MAX_TILT_DEG):
    """
    Check that the tilted assembly does not hit the container wall.

    At depth d below hover_z, the tip is offset (dx, dy) from centre.
    The assembly body at depth d has an additional lateral offset from tilt:
      lateral_at_depth = d * tan(tilt_rad)
    This must not exceed inner_radius - tip_radius.

    Returns (ok, message).
    """
    r_target = math.sqrt(offset_x_m**2 + offset_y_m**2)
    if r_target > CONTAINER_INNER_R:
        return False, (f"Target is {r_target*1000:.1f} mm from centre — "
                       f"outside inner radius {CONTAINER_INNER_R*1000:.1f} mm")

    if tilt_rad > math.radians(max_tilt_deg):
        return False, (f"Tilt {math.degrees(tilt_rad):.1f}° > "
                       f"max {max_tilt_deg}°")

    # At full depth, the assembly body's lateral extent
    lateral = depth_m * math.tan(tilt_rad)
    effective_r = r_target + lateral + tip_radius_m
    if effective_r > CONTAINER_INNER_R:
        return False, (
            f"Assembly hits wall at depth {depth_m*1000:.0f} mm: "
            f"effective radius {effective_r*1000:.1f} mm > "
            f"inner radius {CONTAINER_INNER_R*1000:.1f} mm.\n"
            f"  Reduce offset or depth.")

    return True, "OK"


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class EffortGuard:
    """Joint-torque contact guard for position-mode motions.

    Fed from /joint_states effort. The baseline is the mean effort over the
    first baseline_s of the motion (the arm starts from rest), each joint is
    low-pass filtered, and the guard trips when any joint's |filtered -
    baseline| stays above its threshold for hold_s. This is a CONTACT PROXY:
    gravity torque also changes with posture, which is why it is only armed
    on the short, fixed-orientation insertion-axis moves.
    """

    def __init__(self, thresholds_nm, filter_s=0.02, hold_s=0.03, baseline_s=0.05):
        self.thr = [float(x) for x in thresholds_nm]
        self.filter_s, self.hold_s, self.baseline_s = filter_s, hold_s, baseline_s
        self.t0 = None
        self.base_acc = []
        self.base = None
        self.filt = None
        self.t_prev = None
        self.over_since = None
        self.peak = [0.0] * len(self.thr)
        self.tripped = None           # (joint_index, delta_nm, t_rel)

    def update(self, t, tau):
        if self.t0 is None:
            self.t0 = t
        if self.base is None:
            self.base_acc.append(list(tau))
            if t - self.t0 >= self.baseline_s:
                n = len(self.base_acc)
                self.base = [sum(r[i] for r in self.base_acc) / n for i in range(len(tau))]
                self.filt = list(self.base)
                self.t_prev = t
            return False
        dt = max(t - self.t_prev, 0.0)
        self.t_prev = t
        a = dt / (self.filter_s + dt) if self.filter_s > 0 else 1.0
        over = None
        for i, x in enumerate(tau):
            self.filt[i] += a * (x - self.filt[i])
            d = abs(self.filt[i] - self.base[i])
            if d > self.peak[i]:
                self.peak[i] = d
            if d > self.thr[i] and (over is None or d / self.thr[i] > over[1] / self.thr[over[0]]):
                over = (i, d)
        if over is None:
            self.over_since = None
            return False
        if self.over_since is None:
            self.over_since = t
        if self.tripped is None and t - self.over_since >= self.hold_s:
            self.tripped = (over[0], over[1], t - self.t0)
            return True
        return False


class AngledInserter(Node):

    def __init__(self):
        super().__init__("angled_inserter")
        self._cb_group = ReentrantCallbackGroup()

        # Container position (world frame). In standalone mode these drive the
        # insertion directly; in action-server mode they are the goal fallback.
        self.declare_parameter("target_x",             0.260)  # m, world frame — container centre X
        self.declare_parameter("target_y",             0.000)  # m, world frame — container centre Y
        self.declare_parameter("table_z",             -0.030)  # m
        self.declare_parameter("container_height",     0.086)  # m — 86 mm
        self.declare_parameter("hover_above_top",      0.030)  # m above container top

        # Approach height — must match home tip height minus container_top.
        # Same rule as insert_to_container: ready_z ≈ home tip z.
        self.declare_parameter("approach_clearance",   0.18)   # m above container top

        # Target point inside container (mm from container centre, depth from top)
        # These are overridden by the interactive prompt when interactive_target=true.
        self.declare_parameter("insertion_angle_deg",  45.0)
        self.declare_parameter("insertion_azimuth_deg",  -90.0)
        self.declare_parameter("auto_azimuth", False)
        # How the approach direction is chosen:
        #   "fixed"      insertion_azimuth_deg as given
        #   "tangential" perpendicular to the base->container line (= auto_azimuth)
        #   "search"     try every azimuth_search_step_deg, keep the one that
        #                moves the joints least and still plans end to end
        # Empty keeps the old behaviour: tangential if auto_azimuth, else fixed.
        # Where the target centre comes from:
        #   "auto"    target_x/target_y, overridden by the last
        #             /fused_marker_square_center ever received (old behaviour)
        #   "params"  target_x/target_y only
        #   "markers" the centre of the four table markers, measured NOW: fresh
        #             messages only, median-filtered; aborts a real run if none.
        self.declare_parameter("target_source", "auto")
        # "markers": if no fresh centre arrives, first move the tool vertically
        # above target_x/target_y (the approach pose) so the wrist camera looks
        # straight down at the square, then measure.
        self.declare_parameter("marker_look_first", True)
        self.declare_parameter("marker_wait_s", 5.0)
        self.declare_parameter("marker_min_samples", 5)
        self.declare_parameter("marker_max_spread_mm", 5.0)
        self.declare_parameter("azimuth_mode", "")
        # Each IK call costs the solver's full kinematics_solver_timeout (0.2 s
        # with TRAC-IK solve_type Distance), so the step sets the search time:
        # 15 deg is 24 directions, ~73 calls, ~15 s.
        self.declare_parameter("azimuth_search_step_deg", 15.0)
        # Reject approach directions that carry the wrist closer than this to
        # the base axis (it would hang over the robot's own shoulder).
        self.declare_parameter("azimuth_search_min_base_radius", 0.15)
        # How many of the best-ranked directions to fully plan before giving up.
        self.declare_parameter("azimuth_search_confirm", 3)
        # Reject directions that bring a bounded joint (2/4/6) this close to its limit.
        self.declare_parameter("azimuth_search_limit_margin_deg", 5.7)
        # Extra random IK seeds tried for the vertical pose above the target.
        self.declare_parameter("azimuth_search_ready_seeds", 6)
        # Hard cap on tilt angle (deg). The geometric wall-clearance check still
        # runs regardless; this just rejects obviously-too-steep angles early.
        self.declare_parameter("max_tilt_deg",         45.0)
        self.declare_parameter("target_depth_mm",     30.0)    # mm below container top

        # If true, prompt user to enter target in mm at runtime
        self.declare_parameter("interactive_target",   False)
        self.declare_parameter("direct_to_angled_hover", False)

        # Motion parameters
        self.declare_parameter("max_velocity_scaling", 0.25)
        # Transit (free space: Phases 0/1/3/6b/7/8) and insertion (tip entering
        # or inside the container: Phases 4/6a) are tuned separately so the
        # transit can be fluid while the insertion stays rigid and slow.
        # transit_velocity_scaling < 0 falls back to max_velocity_scaling.
        self.declare_parameter("transit_velocity_scaling", -1.0)
        self.declare_parameter("insert_speed_mps",         0.01)  # m/s
        self.declare_parameter("insert_rot_speed_dps",     5.0)   # deg/s, Kortex LIN only
        # Pilz LIN speed = velocity scaling * max_trans_vel from
        # pilz_cartesian_limits.yaml (1.0 m/s). Measured on REAL-1 2026-09-14:
        # scale 0.05 -> 4.6 cm/s, 0.01 -> 9.4 mm/s average including ramps.
        self.declare_parameter("pilz_max_trans_vel",       1.0)
        # Native Kortex ReachPose for straight lines. OFF: switching servoing
        # mode under a running ros2_control driver stops its hardware interface
        # (WRONG_SERVOING_MODE, 2026-09-14), and the pose it sends is
        # bracelet_link while Kortex interprets it as the tool frame (+0.20 m).
        self.declare_parameter("use_kortex_lin",           False)
        # Record every executed run with `ros2 bag record` for offline analysis.
        # Phase boundaries go on /insertion/phase ("start:<label>" / "end:<label>:ok|fail").
        # Over-force guard on Phase 4a/4b (joint-torque contact proxy, see
        # EffortGuard). mode: abort | monitor (log only) | off. Thresholds are
        # per joint, Nm above the torque at the start of the move. Defaults are
        # a first guess -- calibrate from the "peak d-tau" line each run prints
        # and analyze_insertion_bag.py.
        self.declare_parameter("force_guard_mode",         "abort")
        self.declare_parameter("force_guard_nm",           [6.0, 6.0, 6.0, 6.0, 3.0, 3.0, 3.0])
        self.declare_parameter("force_guard_hold_s",       0.03)
        self.declare_parameter("force_guard_filter_s",     0.02)
        self.declare_parameter("record_bag",               True)
        self.declare_parameter("bag_dir",                  "~/insertion_bags")
        # combine_cameras.py publishes a 5 Hz JPEG copy of each camera on
        # /insertion/throttled<image topic>/compressed. Never subscribe to the
        # raw image topics from here: a second subscriber makes CycloneDDS
        # multicast the raw video onto the arm's Ethernet link, and the Kortex
        # driver's cyclic I/O times out (2026-10-09: the arm did not move).
        self.declare_parameter("bag_extra_topics",         ["/insertion/throttled/camera/color/image_raw/compressed",
                                                            "/insertion/throttled/realsense/front_cam/color/image_raw/compressed",
                                                            "/camera/color/camera_info",
                                                            "/realsense/front_cam/color/camera_info",
                                                            "/fused_corners",
                                                            "/marker_observations"])
        # Phase 4/6a travel the whole insertion axis from hover (~300 mm at 45 deg).
        # Only the stretch within insert_slow_zone_m (tip height above the
        # container top) runs at insert speed; the rest of the axis is a
        # straight LIN at axis_speed_mps. Both halves are rigid position moves.
        self.declare_parameter("insert_slow_zone_m",       0.02)
        self.declare_parameter("axis_speed_mps",           0.05)
        # Blend Phase 0 -> 1 and Phase 7 -> 8 into single motions via Pilz
        # /plan_sequence_path. Only free-space corners are blended: the tip
        # pivot (Phase 3) and the insertion stay exact. Falls back to the
        # separate phases if sequence planning fails.
        self.declare_parameter("blend_transit",            True)
        self.declare_parameter("blend_radius",             0.05)  # m, at ee_link
        self.declare_parameter("execute_motion",       False)
        # Run mode: False = standalone (run one insertion from params, then exit);
        # True = action server (wait for goals on /insert_container).
        self.declare_parameter("use_action_server",    False)
        self.declare_parameter("real_robot",           False)
        self.declare_parameter("skip_home_move",       True)
        self.declare_parameter("teach_mode",           False)
        self.declare_parameter("replay_mode",          False)
        self.declare_parameter("demo_file",            "insertion_demo.csv")
        self.declare_parameter("step_by_step",         False)
        self.declare_parameter("return_to_start",      True)
        self.declare_parameter("post_insert_wait",     2.0)
        self.declare_parameter("tip_link",             "assembly_tip")
        self.declare_parameter("ee_link",              "bracelet_link")
        self.declare_parameter("world_frame",          "world")
        self.declare_parameter("move_group_name",      "manipulator")

        # use_current_orientation: use live EE quaternion as pen-down target
        # (same meaning as insert_to_container — keep true unless you've
        #  measured the exact pen-down quaternion separately)
        self.declare_parameter("use_current_orientation", False)

        # Tool-down ("vertical") orientation of ee_link (bracelet_link), used
        # when use_current_orientation is False. Measured 2026-06-16 by jogging
        # assembly_tip straight down and reading `tf2_echo world assembly_tip`
        # (assembly_tip == bracelet_link orientation: fixed joint, rpy 0). At
        # tool-down, bracelet_link is ~aligned with world (near identity).
        # Re-measure if the arm/assembly is reconfigured.
        self.declare_parameter("vertical_quat_x", -0.003)
        self.declare_parameter("vertical_quat_y", -0.007)
        self.declare_parameter("vertical_quat_z",  0.000)
        self.declare_parameter("vertical_quat_w",  1.000)

        # Joint path-constraint half-width (rad) for the Phase 0 reorientation
        # PTP. Wider than the 1.2 used elsewhere because reaching tool-down from
        # a horizontal home can require a ~90° wrist swing.
        # 2026-09-14: 2.6 rejected target (0.45, -0.20) -- joint_6 needs -2.88 rad
        # from home to tool-down there. joint_6 is bounded (+/-2.58 rad), so 3.6
        # never binds it, while still keeping continuous joints off a 2*pi flip.
        self.declare_parameter("approach_joint_band", 3.6)

        # CIRC arc parameters
        # n_circ_via_points: number of intermediate via-points for the arc.
        # Pilz CIRC needs exactly ONE via-point at the midpoint of the arc.
        # This parameter is kept for documentation; do not change from 1.
        self.declare_parameter("n_circ_via_points", 1)
        
        self.arm_joint_names = self.declare_parameter("arm_joint_names", [
            "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6", "joint_7"
        ]).value
        
        c_joints, m_vel, h_joints = get_robot_info(self)
        self.continuous_joints = c_joints if c_joints else {"joint_1", "joint_3", "joint_5", "joint_7"}
        self.home_joints = h_joints if h_joints else {
            "joint_1":  0.0000, "joint_2": -0.3049, "joint_3": -3.1416,
            "joint_4": -1.6607, "joint_5":  0.0000, "joint_6": -1.7928, "joint_7": -0.0006,
        }
        self.max_joint_vel = m_vel if m_vel else {j: 0.8 for j in self.arm_joint_names}
        self.max_joint_acc = {j: 0.4 for j in self.arm_joint_names}
        self.fjt_topic = self.declare_parameter("fjt_topic", "/joint_trajectory_controller/follow_joint_trajectory").value
        self._assembly_tip_offset = None

        
        # Kortex API parameters
        self.declare_parameter("robot_ip", "192.168.1.10")
        self.declare_parameter("robot_user", "admin")
        self.declare_parameter("robot_password", "admin")
        self._kortex_transport = None
        self._kortex_router = None
        self._kortex_session = None
        self._kortex_base = None
        
        if _HAS_KORTEX_API and self.get_parameter("real_robot").value:
            try:
                ip = self.get_parameter("robot_ip").value
                usr = self.get_parameter("robot_user").value
                pwd = self.get_parameter("robot_password").value
                t, r, s, b = create_kortex_client(ip, usr, pwd)
                self._kortex_transport = t
                self._kortex_router = r
                self._kortex_session = s
                self._kortex_base = b
                self.get_logger().info(f"Connected to Kortex API at {ip} successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to connect to Kortex API: {e}")
        elif not _HAS_KORTEX_API:
            self.get_logger().warn("kortex_api module not found. Native cartesian control will fail.")

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)

        # Joint state snapshot
        self._start_joints = {}
        # Always-fresh live joints (updated even after _start_joints is frozen).
        # Used to rewind planned trajectories onto the arm's actual joint
        # winding before execution — see _unwrap_trajectory.
        self._live_joints  = {}
        self._guard        = None
        self._guard_lock   = threading.Lock()
        self._js_frozen    = False
        self._js_sub       = self.create_subscription(
            JointState, "/joint_states", self._js_cb, 10,
            callback_group=self._cb_group)

        # Clients
        self._plan_cli = self.create_client(
            GetMotionPlan, "/plan_kinematic_path",
            callback_group=self._cb_group)
        self._ik_cli = self.create_client(
            GetPositionIK, "/compute_ik",
            callback_group=self._cb_group)
        # Dry runs never move the arm, so each phase must be planned from where
        # the previous one would have ended: {joint: position}, None when live.
        self._dry_chain = None
        self._step_declined = False
        self._phase_pub = self.create_publisher(String, "/insertion/phase", 10)
        # Bag-only topics: run_info is JSON, one message per event (params,
        # target, measured_offset, outcome); planned_trajectory is every
        # trajectory as sent, between that phase's start/end markers.
        self._info_pub = self.create_publisher(String, "/insertion/run_info", 10)
        self._traj_pub = self.create_publisher(
            RobotTrajectory, "/insertion/planned_trajectory", 10)
        self._run_log = {}
        self._seq_cli = self.create_client(
            GetMotionSequence, "/plan_sequence_path",
            callback_group=self._cb_group)
        self._execute_cli = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory",
            callback_group=self._cb_group)
        self._fjt_cli = ActionClient(
            self, FollowJointTrajectory,
            self.fjt_topic,
            callback_group=self._cb_group)

        self._active_gh  = None
        self._arm_moved  = False
        self._run_done   = threading.Event()
        self._action_goal_handle = None

        if self.get_parameter("use_action_server").value:
            if _HAS_ACTION_INTERFACE:
                self._action_server = ActionServer(
                    self,
                    InsertContainer,
                    "insert_container",
                    execute_callback   = self._action_execute_cb,
                    goal_callback      = self._action_goal_cb,
                    cancel_callback    = self._action_cancel_cb,
                    callback_group     = self._cb_group,
                )
                self.get_logger().info("Action server ready on /insert_container")
            else:
                self.get_logger().warn("InsertContainer action interface not found — Action Server disabled.")

        self.add_on_set_parameters_callback(self._parameter_callback)
        self._collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)
        self._camera_target = None
        self._marker_centres = collections.deque(maxlen=300)   # (monotonic, x, y, frame)
        self.create_subscription(PoseStamped, "/fused_marker_square_center", self._camera_target_cb, 10)
        self._done = False

    def destroy_node(self):
        if hasattr(self, '_kortex_session') and self._kortex_session:
            try:
                self._kortex_session.CloseSession()
            except Exception:
                pass
        if hasattr(self, '_kortex_transport') and self._kortex_transport:
            try:
                self._kortex_transport.disconnect()
            except Exception:
                pass
        super().destroy_node()

    # ------------------------------------------------------------------


    def _run_info(self, event, **data):
        self._info_pub.publish(String(data=json.dumps(dict(event=event, **data), default=list)))

    def _camera_target_cb(self, msg: PoseStamped):
        self._camera_target = msg
        self._marker_centres.append((time.monotonic(), msg.pose.position.x,
                                     msg.pose.position.y, msg.header.frame_id))

    def _fresh_marker_centre(self):
        """Median marker-square centre from messages arriving from now on, or None."""
        timeout = float(self.get_parameter("marker_wait_s").value)
        need = int(self.get_parameter("marker_min_samples").value)
        max_spread = float(self.get_parameter("marker_max_spread_mm").value) / 1000.0
        world = self.get_parameter("world_frame").value
        t_start = time.monotonic()
        fresh = []
        while time.monotonic() - t_start < timeout:
            fresh = [m for m in list(self._marker_centres) if m[0] >= t_start]
            if len(fresh) >= need:
                break
            time.sleep(0.1)
        log = self.get_logger()
        if len(fresh) < need:
            log.warn(f"  [Markers] only {len(fresh)} fresh centre message(s) in "
                     f"{timeout:.0f} s (need {need}) -- is combine_cameras.py running "
                     "and are the markers in view?")
            return None
        frames = {m[3] for m in fresh}
        if frames != {world}:
            log.error(f"  [Markers] centre is published in {sorted(frames)}, "
                      f"expected '{world}'.")
            return None
        centre = robust_centre([(m[1], m[2]) for m in fresh], max_spread)
        if centre is None:
            log.warn(f"  [Markers] {len(fresh)} centre messages disagree by more than "
                     f"{max_spread*1000:.0f} mm -- not using them.")
            return None
        log.info(f"  [Markers] centre ({centre[0]:.4f}, {centre[1]:.4f}) from "
                 f"{len(fresh)} fresh messages, spread {centre[2]*1000:.1f} mm")
        return centre[0], centre[1]
    def _parameter_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name in ["container_height", "insert_depth_from_top", "target_x", "target_y", "hover_above_top", "approach_clearance"]:
                self.get_logger().info(f"Dynamically updated parameter {p.name} to {p.value}")
        return SetParametersResult(successful=True)

    def _js_cb(self, msg):
        # Track the live arm state unconditionally so trajectory unwrapping can
        # anchor to where the arm physically is, even while _start_joints is frozen.
        for name, pos in zip(msg.name, msg.position):
            self._live_joints[name] = pos
        guard = self._guard
        if guard is not None and len(msg.effort) == len(msg.name):
            try:
                tau = [msg.effort[msg.name.index(j)] for j in self.arm_joint_names]
            except ValueError:
                tau = None
            if tau is not None:
                t = time.monotonic()   # header stamps can be zero or sim time
                with self._guard_lock:
                    tripped = guard.update(t, tau)
                if tripped and self.get_parameter("force_guard_mode").value == "abort":
                    gh = self._active_gh
                    if gh is not None:
                        gh.cancel_goal_async()
        if self._js_frozen:
            return
        for name, pos in zip(msg.name, msg.position):
            self._start_joints[name] = pos

    def _nearest_equiv_angle(self, joint_name, target):
        """Shift a target joint angle by multiples of 2π so it is the nearest
        equivalent to the arm's current measured position.

        Kinova Gen3 joints 1, 3, 5, 7 are continuous. A hardcoded goal can sit a
        full 2π from where the arm physically rests (e.g. home wants joint_3=-π
        but the arm reports +π — the SAME pose). Without this, MoveIt plans a 2π
        "unwind" (a wide pointless circle) and the joint_trajectory_controller
        aborts with a ~6.28 rad state-tolerance violation (error -4 /
        PATH_TOLERANCE_VIOLATED). Wrapping the target removes the phantom motion.
        """
        cur = self._start_joints.get(joint_name)
        if cur is None:
            return target
        return target + 2.0 * math.pi * round((cur - target) / (2.0 * math.pi))

    
    def _get_tip_offset(self):
        if self._assembly_tip_offset is None:
            tf = self._get_tf(self.get_parameter("ee_link").value, self.get_parameter("tip_link").value)
            if tf:
                self._assembly_tip_offset = {
                    "x": tf.transform.translation.x,
                    "y": tf.transform.translation.y,
                    "z": tf.transform.translation.z
                }
            else:
                self.get_logger().warn("Could not lookup tf ee_link -> tip_link, using default offset.")
                self._assembly_tip_offset = {"x": 0.108, "y": -0.008, "z": -0.411}
        return self._assembly_tip_offset

    def _get_tf(self, parent, child, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                return self.tf_buffer.lookup_transform(
                    parent, child, rclpy.time.Time())
            except Exception:
                time.sleep(0.1)
        return None

    @staticmethod
    def _wait_for_future(future, timeout_sec):
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.05)
        return future.result()

    def _build_workspace(self):
        ws = WorkspaceParameters()
        ws.header.frame_id = self.get_parameter("world_frame").value
        ws.min_corner.x = -1.1; ws.min_corner.y = -1.1; ws.min_corner.z = -0.1
        ws.max_corner.x =  1.1; ws.max_corner.y =  1.1; ws.max_corner.z =  1.2
        return ws

    # ------------------------------------------------------------------
    # Trajectory clamping (mirrors insert_to_container)
    # ------------------------------------------------------------------
    
    def _clamp_traj(self, traj):
        pts = traj.joint_trajectory.points
        names = traj.joint_trajectory.joint_names
        for i, pt in enumerate(pts):
            if not pt.velocities and not pt.accelerations:
                continue
            worst = 1.0
            for j, v in enumerate(pt.velocities):
                jname = names[j] if j < len(names) else self.arm_joint_names[0]
                max_v = self.max_joint_vel.get(jname, 0.8)
                if abs(v) > max_v:
                    worst = max(worst, abs(v) / max_v)
            for j, a in enumerate(pt.accelerations):
                jname = names[j] if j < len(names) else self.arm_joint_names[0]
                max_a = self.max_joint_acc.get(jname, 0.4)
                if abs(a) > max_a:
                    worst = max(worst, (abs(a) / max_a) ** 0.5)
            if worst <= 1.0:
                continue
            prev_ns = (pts[i-1].time_from_start.sec * 1_000_000_000
                       + pts[i-1].time_from_start.nanosec) if i > 0 else 0
            cur_ns  = (pt.time_from_start.sec * 1_000_000_000
                       + pt.time_from_start.nanosec)
            delta   = int((cur_ns - prev_ns) * worst) - (cur_ns - prev_ns)
            for j in range(i, len(pts)):
                ns = (pts[j].time_from_start.sec * 1_000_000_000
                      + pts[j].time_from_start.nanosec) + delta
                pts[j].time_from_start.sec     = ns // 1_000_000_000
                pts[j].time_from_start.nanosec = ns %  1_000_000_000
            pt.velocities    = [v / worst for v in pt.velocities]
            pt.accelerations = [a / (worst*worst) for a in pt.accelerations]
        return traj

    # ------------------------------------------------------------------
    # Trajectory unwrapping (prevents 2pi joint wraps for FJT)
    # ------------------------------------------------------------------
    def _unwrap_trajectory(self, traj, ref_joints=None):
        """Rewind a planned trajectory onto a continuous joint winding.

        Pilz (and OMPL) can return a goal IK solution a full 2π from where a
        continuous Gen3 joint physically rests (joints 1,3,5,7 are continuous).
        The joint_trajectory_controller then sees a ~6.28 rad step it cannot
        follow and aborts with a state-tolerance violation (MoveIt error -4).

        Two passes:
          1. If ref_joints is given (the arm's live joint positions), shift the
             FIRST waypoint to the nearest 2π-equivalent of where the arm
             actually is. This kills the case where the planner's start point
             is canonicalised a full turn from the physical pose.
          2. Make every subsequent waypoint within π of its predecessor, so no
             internal 2π jump survives.
        """
        pts = traj.joint_trajectory.points
        if not pts: return traj

        import math

        names = list(traj.joint_trajectory.joint_names)

        # Pass 1 — anchor the first point to the arm's real winding.
        if ref_joints:
            p0 = list(pts[0].positions)
            for j, name in enumerate(names):
                cur = ref_joints.get(name)
                if cur is None:
                    continue
                p0[j] += 2.0 * math.pi * round((cur - p0[j]) / (2.0 * math.pi))
            pts[0].positions = p0

        # Pass 2 — ensure no jump > pi between consecutive waypoints
        for i in range(1, len(pts)):
            prev_pt = pts[i-1]
            curr_pt = pts[i]
            new_positions = list(curr_pt.positions)
            for j in range(len(new_positions)):
                diff = new_positions[j] - prev_pt.positions[j]
                new_positions[j] -= 2.0 * math.pi * round(diff / (2.0 * math.pi))
            curr_pt.positions = new_positions

        return traj

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _quat_to_euler_deg(self, q):
        import math
        x, y, z, w = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(t3, t4))
        return roll, pitch, yaw

    def _execute_kortex_lin(self, ee_x, ee_y, ee_z, q_xyzw, label,
                            speed_mps=0.05, rot_speed_dps=15.0):
        """Execute a straight Cartesian line via native Kortex API (bypassing MoveIt/IK)"""
        if self._motion_blocked(label):
            return False
        if not _HAS_KORTEX_API or not self._kortex_base:
            self.get_logger().error(f"  [{label}] Kortex API not connected!")
            return False
            
        action = Base_pb2.Action()
        action.name = label
        action.application_data = ""
        
        pose = action.reach_pose.target_pose
        pose.x = ee_x
        pose.y = ee_y
        pose.z = ee_z
        r, p, yaw = self._quat_to_euler_deg(q_xyzw)
        pose.theta_x = r
        pose.theta_y = p
        pose.theta_z = yaw
        
        self.get_logger().info(f"  [EXEC] native Kortex API ReachPose: {label}")
        
        mode = Base_pb2.ServoingModeInformation()
        
        # Define the speed constraint
        speed = Base_pb2.CartesianSpeed()
        speed.translation = speed_mps
        speed.orientation = rot_speed_dps

        action.reach_pose.constraint.speed.CopyFrom(speed)
        # Timeout scales with the move so a slow insertion is not cut off.
        tf0 = self._get_tf(self.get_parameter("world_frame").value,
                           self.get_parameter("ee_link").value, timeout=0.5)
        dist0 = 0.3
        if tf0:
            t0 = tf0.transform.translation
            dist0 = math.sqrt((t0.x-ee_x)**2 + (t0.y-ee_y)**2 + (t0.z-ee_z)**2)
        action_timeout = max(30.0, 3.0 * dist0 / max(speed_mps, 1e-3) + 10.0)

        try:
            # Switch to High-Level servoing to accept API commands
            mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self._kortex_base.SetServoingMode(mode)
            
            import threading
            e = threading.Event()
            error_details = []
            
            def check_for_end_or_abort(e):
                def check(notification, e=e):
                    if notification.action_event == Base_pb2.ACTION_END:
                        e.set()
                    elif notification.action_event == Base_pb2.ACTION_ABORT:
                        error_details.append("ACTION_ABORT")
                        e.set()
                return check

            notification_handle = self._kortex_base.OnNotificationActionTopic(
                check_for_end_or_abort(e),
                Base_pb2.NotificationOptions()
            )
            
            self._kortex_base.ExecuteAction(action)
            
            finished = e.wait(action_timeout)
            self._kortex_base.Unsubscribe(notification_handle)
            
            if finished and not error_details:
                # Wait for arm to settle
                time.sleep(1.0)
                tf = self._get_tf(self.get_parameter("world_frame").value, self.get_parameter("ee_link").value, timeout=0.5)
                if tf:
                    cx = tf.transform.translation.x
                    cy = tf.transform.translation.y
                    cz = tf.transform.translation.z
                    dist = math.sqrt((cx-ee_x)**2 + (cy-ee_y)**2 + (cz-ee_z)**2)
                    self.get_logger().info(f"  [PASS] {label} (dist={dist:.3f}m)")
                else:
                    self.get_logger().info(f"  [PASS] {label} (no tf)")
                self._arm_moved = True
                return True
            else:
                self.get_logger().warn(f"  [WARN] {label} failed: finished={finished} errors={error_details}")
                return False
        except Exception as e:
            self.get_logger().error(f"  Kortex API Error: {e}")
            return False
        finally:
            # ros2_control's position controller owns the arm in single-level
            # servoing.  LOW_LEVEL_SERVOING is reserved for cyclic torque control.
            try:
                mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
                self._kortex_base.SetServoingMode(mode)
            except Exception as e2:
                self.get_logger().error(f"  Failed to restore SINGLE_LEVEL_SERVOING: {e2}")

    def _execute_moveit(self, traj, label, timeout=120.0):
        """Execute via MoveIt /execute_trajectory, publishing phase markers for the bag."""
        if self._motion_blocked(label):
            return False
        self._mark_phase(f"start:{label}")
        ok = self._execute_moveit_inner(traj, label, timeout)
        self._mark_phase(f"end:{label}:{'ok' if ok else 'fail'}")
        return ok

    def _execute_moveit_inner(self, traj, label, timeout):
        """Execute via MoveIt /execute_trajectory (OMPL / Pilz PTP paths)."""
        # Rewind onto the arm's actual joint winding: Pilz PTP-to-pose goals
        # (e.g. Phase 0) can sit 2π from the physical pose on a continuous joint,
        # which the controller aborts on (error -4 / state-tolerance violation).
        self._unwrap_trajectory(traj, ref_joints=self._live_joints)
        self._clamp_traj(traj)
        self._traj_pub.publish(traj)
        if not self._execute_cli.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"  /execute_trajectory not available ({label}).")
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        f  = self._execute_cli.send_goal_async(goal)
        gh = self._wait_for_future(f, 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().error(f"  Goal rejected ({label}).")
            return False
        self._active_gh = gh
        rf     = gh.get_result_async()
        result = self._wait_for_future(rf, timeout)
        self._active_gh = None
        if result is None:
            self.get_logger().error(f"  Timed out ({label}).")
            return False
        if result.result.error_code.val == 1:
            self.get_logger().info(f"  [PASS] {label}")
            self._arm_moved = True
            return True
        self.get_logger().warn(f"  [WARN] {label} MoveIt error {result.result.error_code.val}")
        return False

    def _mark_phase(self, text):
        self._phase_pub.publish(String(data=text))
        if text.startswith("end:"):
            label, _, status = text[len("end:"):].rpartition(":")
            self._run_log.setdefault("phases", []).append([label, status])
        elif text.startswith("guard_trip:"):
            self._run_log.setdefault("guard_trips", []).append(text[len("guard_trip:"):])

    def _execute_fjt(self, traj, label, timeout=60.0):
        """Execute via direct FJT, publishing phase start/end markers for the bag."""
        if self._motion_blocked(label):
            return False
        self._mark_phase(f"start:{label}")
        ok = self._execute_fjt_inner(traj, label, timeout)
        self._mark_phase(f"end:{label}:{'ok' if ok else 'fail'}")
        return ok

    def _execute_fjt_inner(self, traj, label, timeout):
        """Execute via direct FJT (Cartesian paths — overrides 0.1 rad path tol)."""
        self._unwrap_trajectory(traj, ref_joints=self._live_joints)
        self._clamp_traj(traj)
        self._traj_pub.publish(traj)
        deadline = time.time() + 5.0
        while not self._fjt_cli.server_is_ready() and time.time() < deadline:
            time.sleep(0.05)
        if not self._fjt_cli.server_is_ready():
            self.get_logger().error(f"  FJT not available ({label}).")
            return False
        # Stamp header so Kortex driver accepts it
        now_ns   = self.get_clock().now().nanoseconds
        start_ns = now_ns + int(0.3e9)
        traj.joint_trajectory.header.stamp.sec     = start_ns // 1_000_000_000
        traj.joint_trajectory.header.stamp.nanosec = start_ns %  1_000_000_000
        fjt = FollowJointTrajectory.Goal()
        fjt.trajectory = traj.joint_trajectory
        # Gen3 7DOF joints 1,3,5,7 are CONTINUOUS. The home pose parks joint_3 at
        # exactly -pi (the +/-pi wrap seam), so the encoder resolves it to +pi one
        # cycle and -pi the next. A trajectory planned at +pi then shows a transient
        # 2*pi (=6.283 rad) desired-vs-measured error the instant the reading flips,
        # which trips the path tolerance and aborts Phase 0 with FollowJointTrajectory
        # error -4 (PATH_TOLERANCE_VIOLATED). A 2*pi error on a continuous joint is
        # the SAME physical pose, so it is safe to ignore along the path: ERASE the
        # path tolerance for the continuous joints and keep the 5.0 rad guard on the
        # limited joints (2,4,6 — which can never be 2*pi off). Goal tolerance (0.1)
        # still enforces the endpoint (which sits off the seam).
        #
        # NOTE: per ros2_controllers resolve_tolerance_source(), the special path
        # tolerance value is -1.0 = "erase" (unrestricted). 0.0 would silently fall
        # back to the YAML default (5.0), so it MUST be -1.0 to actually disable it.
        
        for name in self.arm_joint_names:
            pt = JointTolerance(); pt.name = name
            pt.position = -1.0 if name in self.continuous_joints else 5.0
            pt.velocity = 0.0; pt.acceleration = 0.0
            fjt.path_tolerance.append(pt)
            gt = JointTolerance(); gt.name = name
            gt.position = -1.0 if name in self.continuous_joints else 0.2
            gt.velocity = -1.0; gt.acceleration = 0.0
            fjt.goal_tolerance.append(gt)
        fjt.goal_time_tolerance = RosDuration(sec=30, nanosec=0)
        f  = self._fjt_cli.send_goal_async(fjt)
        gh = self._wait_for_future(f, 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().error(f"  FJT goal rejected ({label}).")
            return False
        self._active_gh = gh
        rf     = gh.get_result_async()
        result = self._wait_for_future(rf, timeout)
        self._active_gh = None
        if result is None:
            self.get_logger().error(f"  FJT timed out ({label}).")
            return False
        if result.status == GoalStatus.STATUS_CANCELED:
            # error_code is 0 (== SUCCESSFUL) on a cancel, so check status first.
            self.get_logger().warn(f"  [STOP] {label} cancelled.")
            return False
        code = result.result.error_code
        if code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
            # -5 also comes back when the arm never moved (e.g. the hardware
            # interface stopped, 2026-09-14), so check before calling it success.
            time.sleep(0.5)
            final = traj.joint_trajectory.points[-1].positions
            errs = []
            for name, want in zip(traj.joint_trajectory.joint_names, final):
                have = self._live_joints.get(name)
                if have is None:
                    continue
                d = want - have
                if name in self.continuous_joints:
                    d = (d + math.pi) % (2 * math.pi) - math.pi
                errs.append(abs(d))
            worst = max(errs) if errs else float("inf")
            if worst > 0.05:
                self.get_logger().error(
                    f"  [FAIL] {label} FJT error -5 and arm is {math.degrees(worst):.1f} deg "
                    f"from the goal -- it did not get there.")
                return False
            self.get_logger().info(
                f"  [INFO] {label} FJT error -5 ignored: arm within {math.degrees(worst):.2f} deg of goal.")
        if code == FollowJointTrajectory.Result.SUCCESSFUL or code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
            self.get_logger().info(f"  [PASS] {label}")
            self._arm_moved = True
            return True
        self.get_logger().warn(f"  [WARN] {label} FJT error {code}")
        return False

    def set_admittance_mode(self, enable=True):
        """Enable or disable Joint Admittance Mode for kinesthetic teaching."""
        if not _HAS_KORTEX_API or not self._kortex_base:
            self.get_logger().warn("Kortex API not connected, cannot toggle Admittance.")
            return
        from kortex_api.autogen.messages import Base_pb2
        my_admittance = Base_pb2.Admittance()
        my_admittance.admittance_mode = 2 if enable else 0
        try:
            self._kortex_base.SetAdmittance(my_admittance)
        except Exception as e:
            self.get_logger().error(f"Failed to set admittance: {e}")

    def _teach_and_record(self):
        import csv, time, threading
        demo_file = self.get_parameter("demo_file").value
        trajectory_data = []
        is_recording = [True]
        
        def record_loop():
            start_time = time.time()
            while is_recording[0]:
                t = time.time() - start_time
                row = [t] + [self._live_joints.get(j, 0.0) for j in self.arm_joint_names]
                trajectory_data.append(row)
                time.sleep(0.02)
        
        self.get_logger().info("Activating Admittance Mode...")
        self.set_admittance_mode(True)
        
        rec_thread = threading.Thread(target=record_loop)
        rec_thread.start()
        
        self.get_logger().info("Admittance mode active. Physically guide the arm.")
        try:
            input("Press Enter when finished teaching to stop and save...")
        except EOFError:
            pass
            
        is_recording[0] = False
        rec_thread.join()
        
        self.set_admittance_mode(False)
        self.get_logger().info("Deactivated Admittance Mode.")
        
        try:
            with open(demo_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['time'] + self.arm_joint_names
                writer.writerow(header)
                writer.writerows(trajectory_data)
            self.get_logger().info(f"Saved {len(trajectory_data)} points to {demo_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to save {demo_file}: {e}")

    def _replay_trajectory(self):
        import csv
        from trajectory_msgs.msg import JointTrajectoryPoint
        demo_file = self.get_parameter("demo_file").value
        try:
            with open(demo_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                traj = RobotTrajectory()
                traj.joint_trajectory.joint_names = self.arm_joint_names
                
                for row in reader:
                    if not row: continue
                    t = float(row[0])
                    pos = [float(x) for x in row[1:8]]
                    pt = JointTrajectoryPoint()
                    pt.positions = pos
                    pt.time_from_start.sec = int(t)
                    pt.time_from_start.nanosec = int((t % 1) * 1e9)
                    traj.joint_trajectory.points.append(pt)
            
            if not traj.joint_trajectory.points:
                self.get_logger().error("No points in trajectory.")
                return
                
            self.get_logger().info(f"Replaying trajectory from {demo_file}")
            self._execute_fjt(traj, "replay_trajectory", timeout=120.0)
            
        except Exception as e:
            self.get_logger().error(f"Failed to load/replay {demo_file}: {e}")

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _chain_start(self, req):
        """Dry run: start this request where the previous planned phase ended."""
        if self._dry_chain and not req.start_state.joint_state.name:
            req.start_state.is_diff = True
            req.start_state.joint_state.name = list(self._dry_chain)
            req.start_state.joint_state.position = list(self._dry_chain.values())

    def _chain_advance(self, traj):
        if self._dry_chain is not None and traj.joint_trajectory.points:
            jt = traj.joint_trajectory
            self._dry_chain = dict(zip(jt.joint_names, jt.points[-1].positions))

    def _plan(self, req, timeout=45.0, quiet=False):
        """Plan via /plan_kinematic_path, publishing plan_start/plan_end markers for the bag."""
        self._mark_phase("plan_start")
        ok, traj = self._plan_inner(req, timeout, quiet)
        self._mark_phase(f"plan_end:{'ok' if ok else 'fail'}")
        return ok, traj

    def _plan_inner(self, req, timeout, quiet):
        if not self._plan_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("  /plan_kinematic_path not available.")
            return False, None
        self._chain_start(req)
        svc = GetMotionPlan.Request()
        svc.motion_plan_request = req
        f      = self._plan_cli.call_async(svc)
        result = self._wait_for_future(f, timeout)
        if result is None:
            self.get_logger().error("  Planning timed out.")
            return False, None
        resp = result.motion_plan_response
        if resp.error_code.val == 1:
            self._chain_advance(resp.trajectory)
            return True, resp.trajectory
        if not quiet:
            self.get_logger().error(f"  Planning failed (code {resp.error_code.val}).")
        return False, None

    def _guarded(self, label, fn):
        """Run fn() with the effort guard armed. Returns (ok, tripped_info_or_None)."""
        mode = self.get_parameter("force_guard_mode").value
        if mode == "off":
            return fn(), None
        thr = list(self.get_parameter("force_guard_nm").value)
        guard = EffortGuard(thr, self.get_parameter("force_guard_filter_s").value,
                            self.get_parameter("force_guard_hold_s").value)
        with self._guard_lock:
            self._guard = guard
        try:
            ok = fn()
        finally:
            with self._guard_lock:
                self._guard = None
        if guard.base is None:
            self.get_logger().warn(f"  [guard] {label}: no joint effort received -- guard was blind.")
            return ok, None
        peaks = " ".join(f"j{i+1}={p:.2f}" for i, p in enumerate(guard.peak))
        margin = min(t - p for t, p in zip(thr, guard.peak))
        self.get_logger().info(f"  [guard] {label} peak d-tau Nm: {peaks} (min margin {margin:.2f})")
        if guard.tripped is None:
            return ok, None
        j, d, t_rel = guard.tripped
        self._mark_phase(f"guard_trip:{label}:joint_{j+1}:{d:.2f}")
        msg = (f"  [guard] {label}: joint_{j+1} torque changed {d:.2f} Nm "
               f"(limit {thr[j]:.2f}) at {t_rel:.2f} s")
        if mode == "abort":
            self.get_logger().error(msg + " -- motion STOPPED.")
            return False, guard.tripped
        self.get_logger().warn(msg + " -- monitor mode, not stopping.")
        return ok, None

    def _use_kortex_lin(self):
        return bool(self._kortex_base) and self.get_parameter("use_kortex_lin").value

    def _axis_lin(self, ee_xyz, q, label, speed_mps, rot_speed_dps):
        """Rigid straight line along the insertion axis at speed_mps (Pilz LIN via FJT)."""
        if self._use_kortex_lin():
            return self._execute_kortex_lin(*ee_xyz, (q.x, q.y, q.z, q.w), label,
                                            speed_mps=speed_mps,
                                            rot_speed_dps=rot_speed_dps)
        scale = min(1.0, max(1e-3, speed_mps / self.get_parameter("pilz_max_trans_vel").value))
        ok, traj = self._plan(self._build_pilz_lin(*ee_xyz, q, scale))
        if not ok:
            self.get_logger().error(f"  {label} LIN planning failed.")
            return False
        return self._execute_fjt(traj, label)

    def _plan_sequence(self, reqs, blend_radii, timeout=45.0):
        self._mark_phase("plan_start")
        ok, trajs = self._plan_sequence_inner(reqs, blend_radii, timeout)
        self._mark_phase(f"plan_end:{'ok' if ok else 'fail'}")
        return ok, trajs

    def _plan_sequence_inner(self, reqs, blend_radii, timeout):
        """Plan several Pilz requests as one blended motion via /plan_sequence_path.

        reqs[0] keeps its start state; later items must leave it empty (Pilz
        chains them). The final blend radius is forced to 0 so the motion ends
        at rest on the last goal. Returns (ok, [RobotTrajectory, ...]).
        """
        if not self._seq_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("  /plan_sequence_path not available.")
            return False, None
        self._chain_start(reqs[0])
        svc = GetMotionSequence.Request()
        for i, (req, r) in enumerate(zip(reqs, blend_radii)):
            if i > 0:
                req.start_state = RobotState()
            item = MotionSequenceItem()
            item.req = req
            item.blend_radius = 0.0 if i == len(reqs) - 1 else float(r)
            svc.request.items.append(item)
        result = self._wait_for_future(self._seq_cli.call_async(svc), timeout)
        if result is None:
            self.get_logger().warn("  Sequence planning timed out.")
            return False, None
        resp = result.response
        if resp.error_code.val != 1 or not resp.planned_trajectories:
            self.get_logger().warn(
                f"  Sequence planning failed (code {resp.error_code.val}).")
            return False, None
        self._chain_advance(resp.planned_trajectories[-1])
        return True, list(resp.planned_trajectories)

    def _blended_transit(self, reqs, seg_lengths, label, execute):
        """Plan and run a blended free-space transit.

        Returns "done", "fallback" (planning failed, nothing moved -- caller
        runs the separate phases) or "failed" (execution failed mid-motion).
        """
        r = self.get_parameter("blend_radius").value
        # Pilz rejects a blend sphere that swallows a segment endpoint.
        r_max = 0.4 * min(seg_lengths)
        if r > r_max:
            self.get_logger().warn(
                f"  blend_radius {r*1000:.0f} mm too large for a "
                f"{min(seg_lengths)*1000:.0f} mm segment; using {r_max*1000:.0f} mm.")
            r = r_max
        self.get_logger().info(f"\n--- [{label}] Blended transit (r={r*1000:.0f} mm) ---")
        ok, trajs = self._plan_sequence(reqs, [r] * len(reqs))
        if not ok:
            self.get_logger().warn(f"  {label}: falling back to separate phases.")
            return "fallback"
        self.get_logger().info(f"  [PASS] {label} planned ({len(trajs)} trajectory segment(s)).")
        if not execute:
            return "done"
        if not self._wait_for_user(label):
            return "done"
        for i, traj in enumerate(trajs):
            if not self._execute_fjt(traj, f"{label}[{i}]"):
                return "failed"
        return "done"

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def _recovery_return(self, p=None):
        if not self._arm_moved:
            return
        if not self._start_joints:
            self.get_logger().warn("  No start_joints recorded; cannot return home.")
            return

        self.get_logger().info("\n--- Recovery: returning to start joints ---")
        
        req = MotionPlanRequest()
        req.group_name   = self.get_parameter("move_group_name").value
        req.planner_id   = "PTP"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor     = self.get_parameter("max_velocity_scaling").value
        req.max_acceleration_scaling_factor = self.get_parameter("max_velocity_scaling").value * 0.5
        req.workspace_parameters = self._build_workspace()
        req.start_state.is_diff = True
        
        goal_c = Constraints()
        for name, pos in self._start_joints.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            goal_c.joint_constraints.append(jc)
        req.goal_constraints.append(goal_c)
        
        ok, traj = self._plan(req)
        if ok:
            if self._execute_fjt(traj, "recovery_return"):
                self.get_logger().info("  [PASS] recovery_return")
                return
        self.get_logger().error("  [FAIL] recovery_return")

    def _build_pilz_ptp(self, ee_x, ee_y, ee_z, pen_q, vel_scale,
                         start_state=None, joint_band=1.2):
        """Pilz PTP to a single EE pose."""
        req = MotionPlanRequest()
        req.group_name   = self.get_parameter("move_group_name").value
        req.planner_id   = "PTP"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor     = vel_scale
        req.max_acceleration_scaling_factor = vel_scale * 0.5
        req.workspace_parameters = self._build_workspace()
        if start_state:
            req.start_state = start_state
        else:
            req.start_state.is_diff = True

        # Joint path constraints to keep IK local
        if self._start_joints:
            path_c = Constraints()
            for name in self.arm_joint_names:
                cur = self._start_joints.get(name)
                if cur is None:
                    continue
                jc = JointConstraint()
                jc.joint_name = name; jc.position = cur
                jc.tolerance_above = joint_band; jc.tolerance_below = joint_band
                jc.weight = 1.0
                path_c.joint_constraints.append(jc)
            req.path_constraints = path_c

        sphere = SolidPrimitive(); sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.005]
        bv_pose = Pose()
        bv_pose.position.x = ee_x; bv_pose.position.y = ee_y
        bv_pose.position.z = ee_z; bv_pose.orientation = pen_q
        bv = BoundingVolume()
        bv.primitives.append(sphere); bv.primitive_poses.append(bv_pose)
        pos_c = PositionConstraint()
        pos_c.header.frame_id = self.get_parameter("world_frame").value
        pos_c.link_name       = self.get_parameter("ee_link").value
        pos_c.constraint_region = bv; pos_c.weight = 1.0
        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self.get_parameter("world_frame").value
        ori_c.link_name       = self.get_parameter("ee_link").value
        ori_c.orientation = pen_q
        ori_c.absolute_x_axis_tolerance = 0.05
        ori_c.absolute_y_axis_tolerance = 0.05
        ori_c.absolute_z_axis_tolerance = 0.05
        ori_c.weight = 1.0
        goal_c = Constraints()
        goal_c.position_constraints.append(pos_c)
        goal_c.orientation_constraints.append(ori_c)
        req.goal_constraints.append(goal_c)
        return req

    def _build_pilz_ptp_joints(self, joints, vel_scale):
        """Pilz PTP to a joint configuration (list in arm_joint_names order)."""
        req = MotionPlanRequest()
        req.group_name   = self.get_parameter("move_group_name").value
        req.planner_id   = "PTP"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor     = vel_scale
        req.max_acceleration_scaling_factor = vel_scale * 0.5
        req.start_state.is_diff = True
        goal_c = Constraints()
        for name, position in zip(self.arm_joint_names, joints):
            jc = JointConstraint(); jc.joint_name = name
            jc.position = float(position)
            jc.tolerance_above = 0.01; jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal_c.joint_constraints.append(jc)
        req.goal_constraints.append(goal_c)
        return req

    def _build_pilz_lin(self, ee_x, ee_y, ee_z, pen_q, vel_scale,
                         start_state=None, link=None):
        """Pilz LIN straight-line Cartesian move to a pose of `link` (default ee_link).

        With link=tip_link and the tip position unchanged, this is a pure
        rotation about the tip (verified 2026-09-14: 0.01 mm tip drift). The
        tip frame shares bracelet_link's orientation, so the same quaternions apply.
        """
        link = link or self.get_parameter("ee_link").value
        req = MotionPlanRequest()
        req.group_name   = self.get_parameter("move_group_name").value
        req.planner_id   = "LIN"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor     = vel_scale
        req.max_acceleration_scaling_factor = vel_scale * 0.5
        req.workspace_parameters = self._build_workspace()
        if start_state:
            req.start_state = start_state
        else:
            req.start_state.is_diff = True
        sphere = SolidPrimitive(); sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.005]
        bv_pose = Pose()
        bv_pose.position.x = ee_x; bv_pose.position.y = ee_y
        bv_pose.position.z = ee_z; bv_pose.orientation = pen_q
        bv = BoundingVolume()
        bv.primitives.append(sphere); bv.primitive_poses.append(bv_pose)
        pos_c = PositionConstraint()
        pos_c.header.frame_id = self.get_parameter("world_frame").value
        pos_c.link_name       = link
        pos_c.constraint_region = bv; pos_c.weight = 1.0
        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self.get_parameter("world_frame").value
        ori_c.link_name       = link
        ori_c.orientation = pen_q
        ori_c.absolute_x_axis_tolerance = 0.01
        ori_c.absolute_y_axis_tolerance = 0.01
        ori_c.absolute_z_axis_tolerance = 0.01
        ori_c.weight = 1.0
        goal_c = Constraints()
        goal_c.position_constraints.append(pos_c)
        goal_c.orientation_constraints.append(ori_c)
        req.goal_constraints.append(goal_c)
        return req

    def _build_pilz_circ(self, ee_start, ee_via, ee_end,
                          q_start, q_via, q_end,
                          vel_scale, start_state=None):
        """
        Pilz CIRC arc from ee_start through ee_via to ee_end.

        Pilz CIRC in MoveIt2 Humble is specified as:
          - Goal constraints: the END pose (position + orientation)
          - Path constraints: the VIA pose (position only — Pilz ignores
            via orientation in path constraints)

        The arc keeps the TCP on a circular path through all three points.
        When the three EE positions trace a circle, the TCP (assembly_tip)
        remains fixed at the pivot point while the wrist reorients.

        Parameters
        ----------
        ee_start : (x,y,z) EE position at arc start (current pose)
        ee_via   : (x,y,z) EE position at arc midpoint
        ee_end   : (x,y,z) EE position at arc end
        q_start/via/end : Quaternion EE orientation at each point
        vel_scale : velocity scaling factor
        """
        world_frame = self.get_parameter("world_frame").value
        ee_link     = self.get_parameter("ee_link").value
        group_name  = self.get_parameter("move_group_name").value

        req = MotionPlanRequest()
        req.group_name   = group_name
        req.planner_id   = "CIRC"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 15.0
        req.max_velocity_scaling_factor     = vel_scale
        req.max_acceleration_scaling_factor = vel_scale * 0.5
        if start_state:
            req.start_state = start_state
        else:
            req.start_state.is_diff = True

        # Goal: the END pose (position + orientation)
        sphere_end = SolidPrimitive(); sphere_end.type = SolidPrimitive.SPHERE
        sphere_end.dimensions = [0.005]
        end_pose = Pose()
        end_pose.position.x = ee_end[0]; end_pose.position.y = ee_end[1]
        end_pose.position.z = ee_end[2]; end_pose.orientation = q_end
        bv_end = BoundingVolume()
        bv_end.primitives.append(sphere_end)
        bv_end.primitive_poses.append(end_pose)
        pos_end = PositionConstraint()
        pos_end.header.frame_id   = world_frame
        pos_end.link_name         = ee_link
        pos_end.constraint_region = bv_end
        pos_end.weight            = 1.0
        ori_end = OrientationConstraint()
        ori_end.header.frame_id           = world_frame
        ori_end.link_name                 = ee_link
        ori_end.orientation               = q_end
        ori_end.absolute_x_axis_tolerance = 0.05
        ori_end.absolute_y_axis_tolerance = 0.05
        ori_end.absolute_z_axis_tolerance = 0.05
        ori_end.weight                    = 1.0
        goal_c = Constraints()
        goal_c.position_constraints.append(pos_end)
        goal_c.orientation_constraints.append(ori_end)
        req.goal_constraints.append(goal_c)

        # Via-point: position only in path constraints (Pilz CIRC convention)
        sphere_via = SolidPrimitive(); sphere_via.type = SolidPrimitive.SPHERE
        sphere_via.dimensions = [0.005]
        via_pose = Pose()
        via_pose.position.x = ee_via[0]; via_pose.position.y = ee_via[1]
        via_pose.position.z = ee_via[2]; via_pose.orientation = q_via
        bv_via = BoundingVolume()
        bv_via.primitives.append(sphere_via)
        bv_via.primitive_poses.append(via_pose)
        pos_via = PositionConstraint()
        pos_via.header.frame_id   = world_frame
        pos_via.link_name         = ee_link
        pos_via.constraint_region = bv_via
        pos_via.weight            = 1.0
        path_c = Constraints()
        # Pilz only accepts a CIRC path constraint named "interim" or "center";
        # unnamed, every CIRC was rejected and Phase 3/6b fell back to two LINs.
        path_c.name = "interim"
        path_c.position_constraints.append(pos_via)
        req.path_constraints = path_c

        return req


    def _motion_blocked(self, label):
        """True once a step was declined: every phase assumes the previous one
        ran, so after a skip the arm is not where the next plan starts from."""
        if self._step_declined:
            self.get_logger().error(
                f"  {label} NOT executed: an earlier step was declined, so the arm "
                "is not where this motion expects. Rerun to continue.")
        return self._step_declined

    def _wait_for_user(self, label):
        if not self.get_parameter("step_by_step").value:
            return True
        if self._step_declined:
            return False
        try:
            ans = input(f"  [Step] Execute {label}? [Y/n]: ").strip().lower()
            if ans in ('', 'y', 'yes'):
                return True
            self.get_logger().warn(
                f"  {label} declined -- no further motion will run in this insertion.")
        except EOFError:
            # Seen 2026-10-08 under tmux: the prompt returned at once, the
            # approach was skipped silently and the next phase was planned from
            # the wrong pose.
            self.get_logger().error(
                f"  No terminal answered the {label} prompt -- no further motion "
                "will run in this insertion.")
        self._step_declined = True
        return False

    # ------------------------------------------------------------------
    # Approach-direction search
    # ------------------------------------------------------------------
    def _ik(self, xyz, q_xyzw, seed, timeout_s=0.05):
        """Collision-aware IK for ee_link at a world pose, seeded; None if unsolved."""
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = self.get_parameter("move_group_name").value
        r.ik_link_name = self.get_parameter("ee_link").value
        r.avoid_collisions = True
        r.robot_state.is_diff = True
        r.robot_state.joint_state.name = list(self.arm_joint_names)
        r.robot_state.joint_state.position = [float(v) for v in seed]
        r.pose_stamped.header.frame_id = self.get_parameter("world_frame").value
        pose = r.pose_stamped.pose
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = (float(v) for v in q_xyzw)
        r.timeout = RosDuration(sec=0, nanosec=int(timeout_s * 1e9))
        result = self._wait_for_future(self._ik_cli.call_async(req), 2.0)
        if result is None or result.error_code.val != 1:
            return None
        js = result.solution.joint_state
        try:
            return [js.position[js.name.index(j)] for j in self.arm_joint_names]
        except ValueError:
            return None

    def _azimuth_waypoints(self, cont_x, cont_y, ready_z, target_z, tilt_rad,
                           azimuth_rad, q_vertical_xyzw):
        """EE poses [(xyz, quat)] for ready, hover, tilted-at-hover, target."""
        hover, _, _ = insertion_axis_geometry(
            cont_x, cont_y, ready_z, target_z, tilt_rad, azimuth_rad)
        q_tilted = _tilt_quaternion(q_vertical_xyzw, azimuth_rad, tilt_rad)
        return [
            (self._ee_for_tip((cont_x, cont_y, ready_z), q_vertical_xyzw), q_vertical_xyzw),
            (self._ee_for_tip(hover, q_vertical_xyzw), q_vertical_xyzw),
            (self._ee_for_tip(hover, q_tilted), q_tilted),
            (self._ee_for_tip((cont_x, cont_y, target_z), q_tilted), q_tilted),
        ], hover, q_tilted

    def _confirm_azimuth(self, waypoints, hover, q_tilted_xyzw, start, q_ready, vel_scale):
        """Plan approach, tip-pivot and descent for real, each from the previous end."""
        def quat(q):
            return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        (_, q_v), (ee_hover, _), _, (ee_target, _) = waypoints
        steps = [
            self._build_pilz_ptp_joints(q_ready, vel_scale),
            self._build_pilz_lin(*ee_hover, quat(q_v), vel_scale),
            self._build_pilz_lin(*hover, quat(q_tilted_xyzw), vel_scale,
                                 link=self.get_parameter("tip_link").value),
            self._build_pilz_lin(*ee_target, quat(q_tilted_xyzw), vel_scale),
        ]
        names, positions = list(self.arm_joint_names), list(start)
        for req in steps:
            req.start_state.is_diff = True
            req.start_state.joint_state.name = names
            req.start_state.joint_state.position = [float(v) for v in positions]
            ok, traj = self._plan(req, quiet=True)
            if not ok:
                return False
            jt = traj.joint_trajectory
            names, positions = list(jt.joint_names), list(jt.points[-1].positions)
        return True

    def _search_azimuth(self, cont_x, cont_y, ready_z, target_z, tilt_rad,
                        q_vertical_xyzw, vel_scale):
        """Azimuth [rad] needing the least joint motion that plans end to end, or None."""
        log = self.get_logger()
        start = [self._live_joints.get(j) for j in self.arm_joint_names]
        if any(v is None for v in start):
            log.error("  Azimuth search: no joint state yet.")
            return None
        if not self._ik_cli.wait_for_service(timeout_sec=5.0):
            log.error("  Azimuth search: /compute_ik not available.")
            return None
        step = max(1.0, float(self.get_parameter("azimuth_search_step_deg").value))
        min_r = float(self.get_parameter("azimuth_search_min_base_radius").value)
        t0 = time.monotonic()
        stages = ("ready", "hover", "tilted", "target")
        dropped = collections.Counter()
        solved, plans, n_ik = {}, {}, 0

        def solve(xyz, q, seed):
            solution = self._ik(xyz, q, seed)
            return None if solution is None else unwrap_to_seed(
                solution, seed, self.arm_joint_names, self.continuous_joints)

        # The ready pose (vertical, above the centre) is the same for every
        # direction: solve it once.
        ready_pose = self._azimuth_waypoints(
            cont_x, cont_y, ready_z, target_z, tilt_rad, 0.0, q_vertical_xyzw)[0][0]
        margin = math.radians(float(
            self.get_parameter("azimuth_search_limit_margin_deg").value))
        # The arm is redundant, and the solver returns the posture nearest its
        # seed -- which for a far target is often one with joint_6 on its limit
        # (seen 2026-10-08).  Seed from several postures and keep the nearest one
        # that leaves room on the bounded joints.
        rng = __import__("random").Random(0)
        seeds = [start, [self.home_joints.get(j, 0.0) for j in self.arm_joint_names]]
        for _ in range(int(self.get_parameter("azimuth_search_ready_seeds").value)):
            seeds.append([rng.uniform(-math.pi, math.pi) if j in self.continuous_joints
                          else rng.uniform(-0.8, 0.8) * GEN3_BOUNDED_LIMITS.get(j, 2.0)
                          for j in self.arm_joint_names])
        postures = {}
        for ready_seed in seeds:
            n_ik += 1
            solution = self._ik(*ready_pose, ready_seed)
            if solution is not None:
                # travel is always measured from where the arm really is
                q = unwrap_to_seed(solution, start, self.arm_joint_names,
                                   self.continuous_joints)
                postures[tuple(round(v, 2) for v in q)] = q
        if not postures:
            log.error("  Azimuth search: the vertical pose above the target has no "
                      "IK solution -- the target itself is out of reach.")
            return None
        ready_ranked = rank_azimuth_candidates(
            start, {i: [q] for i, q in enumerate(postures.values())},
            self.arm_joint_names, self.continuous_joints, limit_margin=-math.inf)
        roomy = [c for c in ready_ranked if c["limit_margin"] >= margin]
        if not roomy:
            best = max(ready_ranked, key=lambda c: c["limit_margin"])
            log.error(
                f"  Azimuth search: even vertically above the target, {best['limit_joint']} "
                f"is within {math.degrees(best['limit_margin']):.1f} deg of its limit in "
                f"every posture found ({len(postures)}). The target is too far out; "
                "move it closer to the robot.")
            return None
        q_ready = list(postures.values())[roomy[0]["azimuth_deg"]]
        log.info(f"  ready posture: {len(postures)} found, using the nearest with "
                 f"{math.degrees(roomy[0]['limit_margin']):.1f} deg of joint-limit room "
                 f"(largest joint move {math.degrees(roomy[0]['max_travel']):.1f} deg)")
        azimuth_deg = -180.0
        while azimuth_deg < 180.0 - 1e-6:
            waypoints, hover, q_tilted = self._azimuth_waypoints(
                cont_x, cont_y, ready_z, target_z, tilt_rad,
                math.radians(azimuth_deg), q_vertical_xyzw)
            plans[azimuth_deg] = (waypoints, hover, q_tilted)
            chain, seed = [q_ready], q_ready
            if min(math.hypot(xyz[0], xyz[1]) for xyz, _ in waypoints) < min_r:
                dropped["wrist over the base"] += 1
                chain = None
            else:
                for stage, (xyz, q) in zip(stages[1:], waypoints[1:]):
                    n_ik += 1
                    seed = solve(xyz, q, seed)
                    if seed is None:
                        dropped[f"no IK at {stage}"] += 1
                        chain = None
                        break
                    chain.append(seed)
            solved[azimuth_deg] = chain
            azimuth_deg += step
        with_ik = rank_azimuth_candidates(start, solved, self.arm_joint_names,
                                          self.continuous_joints, limit_margin=-math.inf)
        ranked = [c for c in with_ik if c["limit_margin"] >= margin]
        for c in with_ik:
            if c["limit_margin"] < margin:
                dropped["joint near its limit"] += 1
                log.info(f"  az {c['azimuth_deg']:+7.1f} deg rejected: {c['limit_joint']} "
                         f"comes within {math.degrees(c['limit_margin']):.1f} deg of its limit")
        log.info(f"\n--- [Azimuth search] {len(solved)} directions, {n_ik} IK calls, "
                 f"{time.monotonic() - t0:.2f} s: {len(ranked)} usable ---")
        if dropped:
            log.info("  rejected: " + ", ".join(f"{n} x {why}" for why, n in dropped.items()))
        for c in ranked[:8]:
            log.info(f"  az {c['azimuth_deg']:+7.1f} deg | largest joint move "
                     f"{math.degrees(c['max_travel']):6.1f} deg | total "
                     f"{math.degrees(c['total_travel']):6.1f} deg | limit margin "
                     f"{math.degrees(c['limit_margin']):5.1f} deg")
        # IK at the waypoints does not prove the straight segments between
        # them are feasible, so the best few are planned for real.
        chain_backup, self._dry_chain = self._dry_chain, None
        try:
            for c in ranked[:int(self.get_parameter("azimuth_search_confirm").value)]:
                waypoints, hover, q_tilted = plans[c["azimuth_deg"]]
                if self._confirm_azimuth(waypoints, hover, q_tilted, start, q_ready, vel_scale):
                    log.info(f"  [PASS] az {c['azimuth_deg']:+.1f} deg plans end to end "
                             f"-- selected ({time.monotonic() - t0:.2f} s in all).")
                    return math.radians(c["azimuth_deg"]), q_ready
                log.warn(f"  az {c['azimuth_deg']:+.1f} deg has IK but does not plan; "
                         "trying the next.")
        finally:
            self._dry_chain = chain_backup
        return None

    # ------------------------------------------------------------------
    # EE pose computation
    # ------------------------------------------------------------------
    def _ee_from_tip(self, tip_xyz, q_xyzw, ee_world_offset):
        """Compute EE position from desired tip position and world-frame offset."""
        wx, wy, wz = ee_world_offset
        return (tip_xyz[0] + wx, tip_xyz[1] + wy, tip_xyz[2] + wz)

    def _ee_for_tip(self, tip_xyz, q_xyzw):
        """EE (ee_link) position so assembly_tip lands at tip_xyz with EE orientation q.

        tip = EE + R(q) * local_offset  →  EE = tip - R(q) * local_offset.

        Unlike _ee_from_tip (which assumes a fixed world-frame offset measured at
        one pose), this rotates the URDF EE→tip offset by the *target* orientation,
        so it is correct for any orientation — matching how Phases 3/4 compute EE.
        """
        local = (self._get_tip_offset()["x"],
                 self._get_tip_offset()["y"],
                 self._get_tip_offset()["z"])
        rot = rotate_vector_by_quat(local, q_xyzw)
        return (tip_xyz[0] - rot[0], tip_xyz[1] - rot[1], tip_xyz[2] - rot[2])

    # ------------------------------------------------------------------
    # User prompt
    # ------------------------------------------------------------------
    def _prompt_target(self):
        """
        Prompt user for target point inside container in mm from centre.
        Returns (offset_x_mm, offset_y_mm, depth_mm) or None to use defaults.
        """
        container_half = (CONTAINER_WIDTH_M / 2.0 - CONTAINER_WALL_M) * 1000

        print()
        print("  ┌──────────────────────────────────────────────────────────┐")
        print("  │  Target point inside container                           │")
        print("  │                                                          │")
        print("  │  Specify offset from container CENTRE (mm):             │")
        print(f"  │    offset_x : +ve away from robot  (max ±{container_half:.0f} mm)  │")
        print(f"  │    offset_y : +ve left, -ve right  (max ±{container_half:.0f} mm)  │")
        print(f"  │    depth    : mm below container top (max {CONTAINER_HEIGHT_M*1000-6:.0f} mm)   │")
        print("  │                                                          │")
        print(f"  │  Container: {CONTAINER_WIDTH_M*1000:.0f}×{CONTAINER_WIDTH_M*1000:.0f}×{CONTAINER_HEIGHT_M*1000:.0f} mm   max tilt: {MAX_TILT_DEG:.0f}°         │")
        print("  │                                                          │")
        print("  │  Format:  offset_x, offset_y, depth                    │")
        print("  │  Example: 10, -15, 40  (10mm fwd, 15mm right, 40mm deep)│")
        print("  │  Press Enter for straight-down (0, 0, 30 mm default)   │")
        print("  └──────────────────────────────────────────────────────────┘")

        defaults = (
            self.get_parameter("target_offset_x_mm").value,
            self.get_parameter("target_offset_y_mm").value,
            self.get_parameter("target_depth_mm").value,
        )

        while True:
            try:
                raw = input(
                    f"  Target [offset_x, offset_y, depth mm] "
                    f"(default {defaults[0]:.0f}, {defaults[1]:.0f}, {defaults[2]:.0f}): "
                ).strip()

                if not raw:
                    return defaults

                parts = raw.replace(",", " ").split()
                if len(parts) == 3:
                    ox, oy, d = float(parts[0]), float(parts[1]), float(parts[2])
                elif len(parts) == 1:
                    # Just depth
                    ox, oy, d = 0.0, 0.0, float(parts[0])
                else:
                    print("  Enter 3 values: offset_x, offset_y, depth (mm)")
                    continue

                # Validate
                if d <= 0:
                    print(f"  Depth must be > 0 mm")
                    continue
                if d > (CONTAINER_HEIGHT_M * 1000 - 6):
                    print(f"  Depth {d:.0f} mm exceeds container ({CONTAINER_HEIGHT_M*1000:.0f} mm - 6 mm margin).")
                    continue

                ox_m = ox / 1000.0
                oy_m = oy / 1000.0
                d_m  = d  / 1000.0
                ok, msg = check_wall_clearance(ox_m, oy_m, d_m,
                                               math.atan2(math.sqrt(ox_m**2+oy_m**2), d_m))
                if not ok:
                    print(f"  Wall clearance check: {msg}")
                    print("  Try a smaller offset or shallower depth.")
                    continue

                tilt = math.degrees(math.atan2(
                    math.sqrt(ox_m**2 + oy_m**2), d_m))
                azim = math.degrees(math.atan2(oy_m, ox_m))
                print(f"  Target: ({ox:.1f}, {oy:.1f}) mm offset, {d:.1f} mm deep")
                print(f"  Computed: tilt={tilt:.1f}°, azimuth={azim:.1f}°")
                return (ox, oy, d)

            except ValueError:
                print("  Invalid input — enter numbers, e.g.  10, -15, 40")
            except EOFError:
                return defaults

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
        if not self._arm_moved:
            return
        if not self._start_joints:
            return
        self.get_logger().info("\n--- Recovery: returning to start joints ---")
        req = MotionPlanRequest()
        req.group_name   = p["group_name"]
        req.planner_id   = "PTP"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor     = p["vel_scale"]
        req.max_acceleration_scaling_factor = p["vel_scale"] * 0.5
        req.start_state.is_diff = True
        goal_c = Constraints()
        for name, pos in self._start_joints.items():
            if name not in self.arm_joint_names:
                continue
            jc = JointConstraint()
            jc.joint_name = name; jc.position = self._nearest_equiv_angle(name, pos)
            jc.tolerance_above = 0.05; jc.tolerance_below = 0.05
            jc.weight = 1.0
            goal_c.joint_constraints.append(jc)
        req.goal_constraints.append(goal_c)
        ok, traj = self._plan(req)
        if ok:
            p_copy = dict(p); p_copy["execute"] = True
            self._execute_moveit(traj, "recovery_return")

    # ------------------------------------------------------------------
    # Action Server Callbacks
    # ------------------------------------------------------------------
    def _action_goal_cb(self, goal_request):
        self.get_logger().info("Received goal request")
        return GoalResponse.ACCEPT


    def _cancel_active(self):
        from std_msgs.msg import Empty as EmptyMsg
        import time
        import subprocess
        try:
            stop_pub = self.create_publisher(EmptyMsg, "/stop_kortex_cmd", 1)
            stop_pub.publish(EmptyMsg())
            self.get_logger().warn("  Published /stop_kortex_cmd (hardware stop).")
        except Exception as e:
            self.get_logger().warn(f"  Could not publish /stop_kortex_cmd: {e}")

        gh = getattr(self, '_active_gh', None)
        if gh is not None:
            self.get_logger().warn("  Sending trajectory cancellation...")
            try:
                import rclpy
                cancel_f = gh.cancel_goal_async()
                deadline = time.monotonic() + 3.0
                while not cancel_f.done() and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.05)
            except Exception as e:
                self.get_logger().warn(f"  Cancel request failed: {e}")
            self._active_gh = None

        self.get_logger().warn("  Waiting 2s for controller to settle...")
        time.sleep(2.0)

    def _action_cancel_cb(self, goal_handle):
        self.get_logger().warn("Action server: cancel requested — stopping arm.")
        self._cancel_active()
        return CancelResponse.ACCEPT

    def _action_execute_cb(self, goal_handle):
        self._action_goal_handle = goal_handle
        result = InsertContainer.Result()
        result.success = False

        def _pub_fb(phase, prog):
            fb = InsertContainer.Feedback()
            fb.current_phase = phase
            fb.progress = float(prog)
            goal_handle.publish_feedback(fb)

        try:
            self._run_impl(goal_handle, _pub_fb)
            result.success = True
            result.message = "Insertion complete."
            goal_handle.succeed()
        except Exception as e:
            self.get_logger().error(f"Execution failed: {e}")
            result.message = str(e)
            goal_handle.abort()
        finally:
            self._action_goal_handle = None
            self._arm_moved = False
            self._js_frozen = False

        return result

    _BAG_TOPICS = [
        "/joint_states",                                   # positions + efforts
        "/joint_trajectory_controller/controller_state",   # desired vs actual
        "/tf", "/tf_static",
        "/insertion/phase",
        "/insertion/run_info",
        "/insertion/planned_trajectory",
        "/fused_marker_square_center",
    ]

    def _start_bag(self):
        bag_dir = os.path.expanduser(self.get_parameter("bag_dir").value)
        os.makedirs(bag_dir, exist_ok=True)
        path = os.path.join(bag_dir, time.strftime("insertion_%Y%m%d_%H%M%S"))
        try:
            proc = subprocess.Popen(
                ["ros2", "bag", "record", "-o", path] + self._BAG_TOPICS
                + [t for t in self.get_parameter("bag_extra_topics").value if t],
                # Not the terminal: the recorder's keyboard handler switches a
                # shared stdin to non-blocking, and every input() prompt in this
                # script then returns EOF at once (reproduced 2026-10-08).
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.get_logger().warn(f"Failed to start rosbag: {e}")
            return None
        time.sleep(1.5)   # let the recorder discover topics before motion starts
        if proc.poll() is not None:
            self.get_logger().warn(f"rosbag exited immediately (code {proc.returncode}).")
            return None
        self.get_logger().info(f"Recording bag: {path}")
        return proc

    def _stop_bag(self, proc):
        if proc is None or proc.poll() is not None:
            return
        proc.send_signal(signal.SIGINT)   # SIGINT lets the recorder write its metadata
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        self.get_logger().info("Bag recording stopped.")

    def _run_impl(self, goal_handle, pub_fb):
        record = (self.get_parameter("record_bag").value
                  and not goal_handle.request.dry_run)
        bag = self._start_bag() if record else None
        self._run_log = {}
        self._run_info("params", **{name: prm.value for name, prm in self._parameters.items()})
        try:
            self._run_phases(goal_handle, pub_fb)
        finally:
            log = self._run_log
            phases = log.get("phases", [])
            self._run_info(
                "outcome",
                success=bool(log.get("reached_end") and log.get("insert_state") == "done"
                             and all(st == "ok" for _, st in phases)),
                reached_end=bool(log.get("reached_end")),
                insert_state=log.get("insert_state"),
                step_declined=self._step_declined,
                phases=phases, guard_trips=log.get("guard_trips", []))
            time.sleep(0.2)   # let the recorder take the outcome before SIGINT
            self._stop_bag(bag)

    def _run_phases(self, goal_handle, pub_fb):
        vel_scale  = self.get_parameter("transit_velocity_scaling").value
        if vel_scale <= 0.0:
            vel_scale = self.get_parameter("max_velocity_scaling").value
        ins_speed  = self.get_parameter("insert_speed_mps").value
        ins_rot    = self.get_parameter("insert_rot_speed_dps").value
        blend      = self.get_parameter("blend_transit").value
        axis_speed = self.get_parameter("axis_speed_mps").value
        slow_zone  = self.get_parameter("insert_slow_zone_m").value
        execute    = not goal_handle.request.dry_run
        real_robot = self.get_parameter("real_robot").value
        world_frame= self.get_parameter("world_frame").value
        tip_link   = self.get_parameter("tip_link").value
        ee_link    = self.get_parameter("ee_link").value
        group_name = self.get_parameter("move_group_name").value
        table_z    = self.get_parameter("table_z").value
        cont_h     = self.get_parameter("container_height").value
        hover_top  = goal_handle.request.hover_above_top
        approach   = self.get_parameter("approach_clearance").value
        ret        = self.get_parameter("return_to_start").value
        post_wait  = self.get_parameter("post_insert_wait").value

        p = dict(group_name=group_name, vel_scale=vel_scale,
                 execute=execute, world_frame=world_frame,
                 tip_link=tip_link, ee_link=ee_link)
        self._dry_chain = None if execute else {}
        self._step_declined = False


        pub_fb("home_move", 0.0)
        # Home move
        if real_robot and not goal_handle.request.skip_home_move:
            self.get_logger().info("Moving arm to Home...")
            req = MotionPlanRequest()
            req.group_name = group_name
            req.planner_id = "PTP"
            req.pipeline_id = "pilz_industrial_motion_planner"
            req.num_planning_attempts = 1
            req.allowed_planning_time = 10.0
            req.max_velocity_scaling_factor     = max(vel_scale, 0.15)
            req.max_acceleration_scaling_factor = max(vel_scale, 0.15) * 0.5
            req.start_state.is_diff = True
            goal_c = Constraints()
            for name, position in self.home_joints.items():
                jc = JointConstraint(); jc.joint_name = name
                jc.position = self._nearest_equiv_angle(name, position)
                jc.tolerance_above = 0.05; jc.tolerance_below = 0.05
                jc.weight = 1.0
                goal_c.joint_constraints.append(jc)
            req.goal_constraints.append(goal_c)
            ok, traj = self._plan(req)
            if not ok:
                self.get_logger().error("Home move planning failed.")
                return
            if execute and self._wait_for_user("home_move"):
                if not self._execute_moveit(traj, "home_move", timeout=90.0):
                    self.get_logger().error("Home move failed.")
                    return
            time.sleep(1.5)
            self.get_logger().info("Arm at Home.")

        self._js_frozen = True
        if real_robot:
            self.get_logger().info("Arm is ready.")

        container_top = table_z + cont_h
        hover_z  = container_top + hover_top
        ready_z  = container_top + approach

        # Tool-down orientation, from the measured parameters or the live pose
        if self.get_parameter("use_current_orientation").value:
            ee_now = self._get_tf(world_frame, ee_link)
            if ee_now is None:
                self.get_logger().error("  TF lookup failed.")
                return
            r_now = ee_now.transform.rotation
            q_search = (r_now.x, r_now.y, r_now.z, r_now.w)
        else:
            q_search = _quat_normalize(tuple(
                self.get_parameter(f"vertical_quat_{a}").value for a in "xyzw"))

        # Get container position
        cont_x = goal_handle.request.target_x
        cont_y = goal_handle.request.target_y
        target_source = self.get_parameter("target_source").value
        if target_source == "markers":
            self.get_logger().info("\n--- [Markers] Locating the marker square ---")
            centre = self._fresh_marker_centre()
            if centre is None and self.get_parameter("marker_look_first").value:
                ee_look = self._ee_for_tip((cont_x, cont_y, ready_z), q_search)
                self.get_logger().info(
                    f"  Moving above the nominal target ({cont_x:.3f}, {cont_y:.3f}) so "
                    f"the wrist camera looks down at it: EE ({ee_look[0]:.3f}, "
                    f"{ee_look[1]:.3f}, {ee_look[2]:.3f})")
                ok, traj = self._plan(self._build_pilz_ptp(
                    *ee_look, Quaternion(x=q_search[0], y=q_search[1],
                                         z=q_search[2], w=q_search[3]), vel_scale,
                    joint_band=self.get_parameter("approach_joint_band").value))
                if not ok:
                    self.get_logger().error("  Look move planning failed. Aborting.")
                    return
                if execute:
                    if not (self._wait_for_user("marker_look")
                            and self._execute_fjt(traj, "marker_look")):
                        self.get_logger().error("  Look move did not run. Aborting.")
                        return
                    time.sleep(1.0)   # let the image settle after the stop
                    centre = self._fresh_marker_centre()
                else:
                    self.get_logger().info(
                        "  Dry run -- the arm does not move, so nothing new can be seen.")
            if centre is not None:
                cont_x, cont_y = centre
            elif execute:
                self.get_logger().error(
                    "  No marker centre available -- refusing to insert at the "
                    "nominal parameters. Aborting.")
                return
            else:
                self.get_logger().warn(
                    f"  Dry run without a marker centre: planning for the nominal "
                    f"target ({cont_x:.3f}, {cont_y:.3f}) instead.")
        elif target_source == "auto":
            if getattr(self, '_camera_target', None) is not None:
                cont_x = self._camera_target.pose.position.x
                cont_y = self._camera_target.pose.position.y
                self.get_logger().info(f"[Vision] Dynamic target from /fused_marker_square_center: ({cont_x:.3f}, {cont_y:.3f})")
        elif target_source != "params":
            self.get_logger().error(
                f"  Unknown target_source '{target_source}' (auto | params | markers).")
            return

        # Phase 2 — Compute fixed 45 degree tilt geometry
        d_m  = self.get_parameter("target_depth_mm").value / 1000.0
        angle_deg = self.get_parameter("insertion_angle_deg").value
        
        import math
        tilt_rad = math.radians(angle_deg)
        # Tip must end exactly in the center
        target_xyz = (cont_x, cont_y, container_top - d_m)
        self._run_info("target", source=target_source, frame=self.get_parameter("world_frame").value,
                       x=cont_x, y=cont_y, z=target_xyz[2], container_top=container_top)

        azimuth_mode = self.get_parameter("azimuth_mode").value or (
            "tangential" if self.get_parameter("auto_azimuth").value else "fixed")
        azimuth_rad, search_ready = None, None
        if azimuth_mode == "search":
            found = self._search_azimuth(
                cont_x, cont_y, ready_z, target_xyz[2], tilt_rad, q_search, vel_scale)
            if found is None:
                self.get_logger().error(
                    "  Azimuth search found no approach direction that plans. Aborting.")
                return
            # Phase 0 must go to the posture the search ranked, not to whatever
            # posture a pose-goal PTP would pick for the same Cartesian pose.
            azimuth_rad, search_ready = found
        elif azimuth_mode == "tangential":
            # Tangential insertion keeps the wrist at a comfortable radius
            azimuth_rad = math.atan2(cont_y, cont_x) + math.pi / 2.0
        elif azimuth_mode != "fixed":
            self.get_logger().warn(
                f"  Unknown azimuth_mode '{azimuth_mode}'; using insertion_azimuth_deg.")
        if azimuth_rad is None:
            azimuth_rad = math.radians(self.get_parameter("insertion_azimuth_deg").value)

        hover_xyz, (axis_x, axis_y, axis_z), D = insertion_axis_geometry(
            cont_x, cont_y, ready_z, target_xyz[2], tilt_rad, azimuth_rad)

        geo = dict(
            azimuth_rad=azimuth_rad, 
            tilt_rad=tilt_rad, 
            descent_m=D,
            tool_axis=(axis_x, axis_y, axis_z)
        )
        ox_mm = 0.0
        oy_mm = 0.0
        d_mm = d_m * 1000.0
        ox_m = 0.0
        oy_m = 0.0

        # Removed dynamic container CollisionObject publication. 
        # A solid BOX primitive prevents MoveIt from planning the insertion.
        # The hollow container mesh should be published by setup_planning_scene.py instead.

        

        self.get_logger().info("=" * 62)
        self.get_logger().info("angled_insert — geometry")
        self.get_logger().info(
            f"  Container centre: ({cont_x:.3f}, {cont_y:.3f})")
        self.get_logger().info(
            f"  Hover position:   ({hover_xyz[0]:.3f}, {hover_xyz[1]:.3f}, "
            f"{hover_xyz[2]:.3f})")
        self.get_logger().info(
            f"  Target:           ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, "
            f"{target_xyz[2]:.3f})")
        self.get_logger().info(
            f"  Offset:           ({ox_mm:.1f} mm fwd, {oy_mm:.1f} mm side, "
            f"{d_mm:.1f} mm deep)")
        self.get_logger().info(
            f"  Tilt:             {math.degrees(geo['tilt_rad']):.2f}°")
        self.get_logger().info(
            f"  Azimuth:          {math.degrees(geo['azimuth_rad']):.2f}°")
        self.get_logger().info(
            f"  Descent:          {geo['descent_m']*1000:.1f} mm")
        self.get_logger().info(f"  execute={execute}")
        

        self.get_logger().info("=" * 62)

        # Wall clearance check
        ok, msg = check_wall_clearance(
            ox_m, oy_m, d_m, geo["tilt_rad"],
            max_tilt_deg=self.get_parameter("max_tilt_deg").value)
        if not ok:
            self.get_logger().error(f"  Wall clearance FAIL: {msg}\n  Aborting.")
            return
        self.get_logger().info(f"  Wall clearance: {msg}")

        # Get live TF — tip and EE positions
        tip_tf = self._get_tf(world_frame, tip_link)
        ee_tf  = self._get_tf(world_frame, ee_link)
        if tip_tf is None or ee_tf is None:
            self.get_logger().error("  TF lookup failed.")
            return

        tip_w = tip_tf.transform.translation
        ee_w  = ee_tf.transform.translation
        cur_tip = (tip_w.x, tip_w.y, tip_w.z)

        # World-frame EE→tip offset at current arm pose
        ee_world_offset = (ee_w.x - tip_w.x,
                           ee_w.y - tip_w.y,
                           ee_w.z - tip_w.z)
        self.get_logger().info(
            f"  EE→ tip world offset: "
            f"({ee_world_offset[0]:.4f}, {ee_world_offset[1]:.4f}, "
            f"{ee_world_offset[2]:.4f})  "
            f"mag={math.sqrt(sum(v**2 for v in ee_world_offset))*100:.1f} cm")

        # Current orientation
        r = ee_tf.transform.rotation
        if self.get_parameter("use_current_orientation").value:
            q_vertical_xyzw = (r.x, r.y, r.z, r.w)
        else:
            q_vertical_xyzw = _quat_normalize((
                self.get_parameter("vertical_quat_x").value,
                self.get_parameter("vertical_quat_y").value,
                self.get_parameter("vertical_quat_z").value,
                self.get_parameter("vertical_quat_w").value,
            ))

        q_vertical = Quaternion(x=q_vertical_xyzw[0], y=q_vertical_xyzw[1],
                                z=q_vertical_xyzw[2], w=q_vertical_xyzw[3])

        # Compute tilted quaternion
        q_tilted_xyzw = _tilt_quaternion(
            q_vertical_xyzw, geo["azimuth_rad"], geo["tilt_rad"])
        q_tilted = Quaternion(x=q_tilted_xyzw[0], y=q_tilted_xyzw[1],
                              z=q_tilted_xyzw[2], w=q_tilted_xyzw[3])

        # Intermediate quaternion at half tilt (for CIRC via-point)
        q_via_xyzw = _quat_slerp(q_vertical_xyzw, q_tilted_xyzw, 0.5)
        q_via = Quaternion(x=q_via_xyzw[0], y=q_via_xyzw[1],
                           z=q_via_xyzw[2], w=q_via_xyzw[3])

        self.get_logger().info(
            f"  Vertical quat:  ({q_vertical_xyzw[0]:.4f}, {q_vertical_xyzw[1]:.4f}, "
            f"{q_vertical_xyzw[2]:.4f}, {q_vertical_xyzw[3]:.4f})")
        self.get_logger().info(
            f"  Tilted quat:    ({q_tilted_xyzw[0]:.4f}, {q_tilted_xyzw[1]:.4f}, "
            f"{q_tilted_xyzw[2]:.4f}, {q_tilted_xyzw[3]:.4f})")

        # Precompute EE poses for hover and rotation
        ee_hover = self._ee_for_tip(hover_xyz, q_vertical_xyzw)
        ee_start_circ = ee_hover
        ee_via_circ   = self._ee_for_tip(hover_xyz, q_via_xyzw)
        ee_end_circ   = self._ee_for_tip(hover_xyz, q_tilted_xyzw)

        ee_ready = self._ee_for_tip((cont_x, cont_y, ready_z), q_vertical_xyzw)
        transit_01 = "fallback"
        if not self.get_parameter("direct_to_angled_hover").value and blend:
            seg0 = math.dist((ee_w.x, ee_w.y, ee_w.z), ee_ready)
            seg1 = math.dist(ee_ready, ee_hover)
            transit_01 = self._blended_transit(
                [self._build_pilz_ptp_joints(search_ready, vel_scale)
                 if search_ready is not None else
                 self._build_pilz_ptp(
                     *ee_ready, q_vertical, vel_scale,
                     joint_band=self.get_parameter("approach_joint_band").value),
                 self._build_pilz_lin(*ee_hover, q_vertical, vel_scale)],
                [seg0, seg1], "Phase0-1_blended", execute)
            if transit_01 == "failed":
                self._recovery_return(p); return

        if not self.get_parameter("direct_to_angled_hover").value and transit_01 == "fallback":
            self.get_logger().info(
                f"\n--- [Phase 0] Approach above container ---")
            self.get_logger().info(
                f"  EE target: ({ee_ready[0]:.3f}, {ee_ready[1]:.3f}, {ee_ready[2]:.3f})")
            req = (self._build_pilz_ptp_joints(search_ready, vel_scale)
                   if search_ready is not None else self._build_pilz_ptp(
                ee_ready[0], ee_ready[1], ee_ready[2], q_vertical, vel_scale,
                joint_band=self.get_parameter("approach_joint_band").value))
            ok, traj = self._plan(req)
            if not ok:
                self.get_logger().error("  Phase 0 planning failed.")
                return
            if execute and self._wait_for_user("Phase0_approach") and not self._execute_fjt(traj, "Phase0_approach"):
                self._recovery_return(p); return

            pub_fb("Phase1_vertical_descent", 0.0)
            self.get_logger().info(f"\n--- [Phase 1] Vertical descent to hover_z ---")
            self.get_logger().info(
                f"  EE hover: ({ee_hover[0]:.3f}, {ee_hover[1]:.3f}, {ee_hover[2]:.3f})")
            if execute and self._wait_for_user("Phase1_vertical_descent"):
                if self._use_kortex_lin():
                    if not self._execute_kortex_lin(ee_hover[0], ee_hover[1], ee_hover[2], (q_vertical.x, q_vertical.y, q_vertical.z, q_vertical.w), "Phase1_vertical_descent"):
                        self._recovery_return(p); return
                else:
                    req = self._build_pilz_lin(ee_hover[0], ee_hover[1], ee_hover[2], q_vertical, vel_scale)
                    ok, traj = self._plan(req)
                    if not ok:
                        self.get_logger().error("  Phase 1 planning failed.")
                        return
                    if not self._execute_fjt(traj, "Phase1_vertical_descent"):
                        self._recovery_return(p); return

        if self.get_parameter("teach_mode").value:
            self.get_logger().info("\n--- [TEACH MODE] ---")
            self._teach_and_record()
            if ret: self._recovery_return(p)
            return

        if self.get_parameter("replay_mode").value:
            self.get_logger().info("\n--- [REPLAY MODE] ---")
            self._replay_trajectory()
            if ret: self._recovery_return(p)
            return

        if not self.get_parameter("direct_to_angled_hover").value:
            # ── Phase 2: Print geometry summary and optionally confirm ─────────
            self.get_logger().info(
                f"\n--- [Phase 2] Tilt geometry ---\n"
                f"  Tip fixed at: ({hover_xyz[0]:.3f}, {hover_xyz[1]:.3f}, "
                f"{hover_xyz[2]:.3f})\n"
                f"  Tilt:    {math.degrees(geo['tilt_rad']):.2f}°\n"
                f"  Azimuth: {math.degrees(geo['azimuth_rad']):.2f}°\n"
                f"  Descent: {geo['descent_m']*1000:.1f} mm along tilted axis")
    
            # ── Phase 3: CIRC rotation keeping tip fixed ──────────────────────
            self.get_logger().info(f"\n--- [Phase 3] CIRC rotation about tip ---")

            # EE sweeps a circular arc while tip stays fixed at hover_xyz.
            # The arc is defined by three EE positions corresponding to:
            #   start: vertical orientation  (current after Phase 1)
            #   via:   half-tilt orientation (at mid-arc)
            #   end:   full-tilt orientation (target)
            #
            # For each EE position, tip is fixed at hover_xyz:
            #   EE = tip + R(q) * tip_to_ee_local
            # Since we track the world-frame offset directly:
            #   At start: ee_hover (already computed above)
            #   At via:   EE when orientation is q_via and tip at hover_xyz
            #   At end:   EE when orientation is q_tilted and tip at hover_xyz
            #
            # The EE→tip offset IN WORLD FRAME changes with orientation.
            # We recompute it by rotating the local offset by each quaternion.
            # Local EE→tip offset (in EE frame, from URDF):
            local_tip_offset = (
                self._get_tip_offset()["x"],
                self._get_tip_offset()["y"],
                self._get_tip_offset()["z"],
            )

            self.get_logger().info(
                f"  EE arc start: ({ee_start_circ[0]:.4f}, {ee_start_circ[1]:.4f}, "
                f"{ee_start_circ[2]:.4f})\n"
                f"  EE arc via:   ({ee_via_circ[0]:.4f},  {ee_via_circ[1]:.4f},  "
                f"{ee_via_circ[2]:.4f})\n"
                f"  EE arc end:   ({ee_end_circ[0]:.4f},  {ee_end_circ[1]:.4f},  "
                f"{ee_end_circ[2]:.4f})")
    
            # Arc radius check — should equal the EE→tip distance
            arc_r = math.sqrt(sum((a-b)**2 for a,b in
                                  zip(ee_start_circ, hover_xyz)))
            self.get_logger().info(
                f"  Arc radius (EE from tip): {arc_r*100:.1f} cm  "
                f"(expected {math.sqrt(sum(v**2 for v in local_tip_offset))*100:.1f} cm)")
    
            # Primary: LIN on the tip link -- the tip holds still and the wrist
            # swings. Pilz CIRC through /plan_kinematic_path is rejected in
            # Humble (its "interim" via-point is re-checked as a path constraint).
            ok, traj = self._plan(self._build_pilz_lin(
                *hover_xyz, q_tilted, vel_scale, link=tip_link))
            tip_rotated = ok
            if ok:
                self.get_logger().info("  [PASS] tip-pivot rotation planned.")
                if execute and self._wait_for_user("Phase3_tip_rotate") and not self._execute_fjt(traj, "Phase3_tip_rotate"):
                    self._recovery_return(p); return
            else:
                req = self._build_pilz_circ(
                    ee_start_circ, ee_via_circ, ee_end_circ,
                    q_vertical, q_via, q_tilted,
                    vel_scale)
                ok, traj = self._plan(req, timeout=20.0)
    
            if tip_rotated:
                pass
            elif not ok:
                self.get_logger().warn(
                    "  Pilz CIRC failed — falling back to two sequential Pilz LIN "
                    "moves (via half-tilt, then full tilt).\n"
                    "  Tip will move slightly during rotation (< 2mm for small tilts).")
                # Fallback: LIN to via then LIN to end
                req_via = self._build_pilz_lin(*ee_via_circ, q_via, vel_scale)
                ok_via, traj_via = self._plan(req_via)
                req_end = self._build_pilz_lin(*ee_end_circ, q_tilted, vel_scale)
                if ok_via:
                    rs = req_end.start_state
                    rs.is_diff = True
                    rs.joint_state.name = traj_via.joint_trajectory.joint_names
                    rs.joint_state.position = traj_via.joint_trajectory.points[-1].positions
                ok_end, traj_end = self._plan(req_end)
                if not ok_via or not ok_end:
                    self.get_logger().error("  Rotation fallback planning failed.")
                    self._recovery_return(p); return
                if execute:
                    if self._wait_for_user("Phase3a_rotate_via") and not self._execute_fjt(traj_via, "Phase3a_rotate_via"):
                        self._recovery_return(p); return
                    if self._wait_for_user("Phase3b_rotate_end") and not self._execute_fjt(traj_end, "Phase3b_rotate_end"):
                        self._recovery_return(p); return
                traj = None   # signal that CIRC was not used
            else:
                self.get_logger().info("  [PASS] CIRC rotation planned.")
                if execute and self._wait_for_user("Phase3_circ_rotate") and not self._execute_fjt(traj, "Phase3_circ_rotate"):
                    self._recovery_return(p); return
        else:
            self.get_logger().info(f"\n--- [Phase 0_direct] Direct approach to angled hover ---")
            self.get_logger().info(f"  EE target: ({ee_end_circ[0]:.3f}, {ee_end_circ[1]:.3f}, {ee_end_circ[2]:.3f})")
            req = self._build_pilz_ptp(
                ee_end_circ[0], ee_end_circ[1], ee_end_circ[2], q_tilted, vel_scale,
                joint_band=self.get_parameter("approach_joint_band").value)
            ok, traj = self._plan(req)
            if not ok:
                self.get_logger().error("  Phase 0_direct planning failed.")
                return
            if execute and self._wait_for_user("Phase0_direct_approach") and not self._execute_fjt(traj, "Phase0_direct_approach"):
                self._recovery_return(p); return

        # ── Phase 4: Angled descent to target ─────────────────────────────
        self.get_logger().info(f"\n--- [Phase 4] Angled descent to target ---")

        # EE position at target: tip is at target_xyz, EE has tilted orientation
        ee_target = self._ee_for_tip(target_xyz, q_tilted_xyzw)

        self.get_logger().info(
            f"  Tip target:  ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, "
            f"{target_xyz[2]:.3f})\n"
            f"  EE target:   ({ee_target[0]:.3f}, {ee_target[1]:.3f}, "
            f"{ee_target[2]:.3f})\n"
            f"  Descent:     {geo['descent_m']*1000:.1f} mm along "
            f"{math.degrees(geo['tilt_rad']):.1f}° axis")

        # Pre-insert point: on the insertion axis, tip slow_zone above the top.
        # Clamped so it never sits above the hover point.
        pre_d = min((d_m + slow_zone) / max(math.cos(geo["tilt_rad"]), 1e-3),
                    geo["descent_m"])
        ax = geo["tool_axis"]
        pre_tip = (target_xyz[0] - ax[0] * pre_d, target_xyz[1] - ax[1] * pre_d,
                   target_xyz[2] - ax[2] * pre_d)
        ee_pre = self._ee_for_tip(pre_tip, q_tilted_xyzw)
        self.get_logger().info(
            f"  Fast axis:   {(geo['descent_m'] - pre_d)*1000:.1f} mm at {axis_speed*100:.1f} cm/s\n"
            f"  Slow insert: {pre_d*1000:.1f} mm at {ins_speed*1000:.1f} mm/s")

        insert_state = "done"      # done | aborted_4a | aborted_4b
        if not execute:
            for label, pose in (("Phase4a_axis_approach", ee_pre),
                                ("Phase4b_insert", ee_target)):
                ok, _ = self._plan(self._build_pilz_lin(*pose, q_tilted, vel_scale))
                self.get_logger().info(f"  [{'PASS' if ok else 'FAIL'}] {label} planned (dry run).")
        if execute and self._wait_for_user("Phase4_angled_descent"):
            if geo["descent_m"] - pre_d > 1e-3:
                ok, trip = self._guarded("Phase4a_axis_approach", lambda: self._axis_lin(
                    ee_pre, q_tilted, "Phase4a_axis_approach", axis_speed, 15.0))
                if trip:
                    # Contact before the container top: back straight out to hover,
                    # then take the normal reverse path.
                    insert_state = "aborted_4a"
                    if not self._axis_lin(ee_end_circ, q_tilted, "Phase4a_guard_retreat",
                                          ins_speed, ins_rot):
                        self.get_logger().error("  Guard retreat failed -- arm left where it stopped.")
                        return
                elif not ok:
                    self._recovery_return(p); return
            if insert_state == "done":
                ok, trip = self._guarded("Phase4b_insert", lambda: self._axis_lin(
                    ee_target, q_tilted, "Phase4b_insert", ins_speed, ins_rot))
                if trip:
                    insert_state = "aborted_4b"      # Phase 6a extracts along the axis
                elif not ok:
                    # Never PTP home with the tip possibly in the container:
                    # extract along the axis first.
                    if (self._axis_lin(ee_pre, q_tilted, "Phase4b_fail_extract", ins_speed, ins_rot)
                            and self._axis_lin(ee_end_circ, q_tilted, "Phase4b_fail_retreat",
                                               axis_speed, 15.0)):
                        self._recovery_return(p)
                    else:
                        self.get_logger().error("  Extraction failed -- arm left where it stopped.")
                    return

        # ── Phase 5: Hold ──────────────────────────────────────────────────
        self._run_log["insert_state"] = insert_state
        self.get_logger().info(f"\n--- [Phase 5] Hold at target ---")
        if insert_state != "done":
            self.get_logger().warn(f"  Insertion {insert_state} by the force guard -- skipping hold.")
        elif execute:
            try:
                ans = input(f"  Tip at target ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, "
                            f"{target_xyz[2]:.3f}).  Measured tip offset from the centre "
                            "'dx dy' in mm (world X Y), or just ENTER, to reverse ... ")
                try:
                    dx, dy = (float(v) for v in ans.replace(",", " ").split())
                    self._run_info("measured_offset", dx_mm=dx, dy_mm=dy)
                    self.get_logger().info(f"  Recorded tip offset ({dx:+.1f}, {dy:+.1f}) mm.")
                except ValueError:
                    if ans.strip():
                        self.get_logger().warn(f"  '{ans}' is not 'dx dy' -- no offset recorded.")
            except EOFError:
                time.sleep(post_wait)
        else:
            self.get_logger().info("  Dry-run — skipping hold.")

        # ── Phase 6a: Reverse angled ascent ───────────────────────────────
        self.get_logger().info(f"\n--- [Phase 6a] Reverse angled ascent ---")
        if not execute:
            for label, pose in (("Phase6a_extract", ee_pre),
                                ("Phase6a_axis_retreat", ee_end_circ)):
                ok, _ = self._plan(self._build_pilz_lin(*pose, q_tilted, vel_scale))
                self.get_logger().info(f"  [{'PASS' if ok else 'FAIL'}] {label} planned (dry run).")
        if execute and insert_state != "aborted_4a" and self._wait_for_user("Phase6a_angled_ascent"):
            if not self._axis_lin(ee_pre, q_tilted, "Phase6a_extract",
                                  ins_speed, ins_rot):
                self._recovery_return(p); return
            if geo["descent_m"] - pre_d > 1e-3 and not self._axis_lin(
                    ee_end_circ, q_tilted, "Phase6a_axis_retreat", axis_speed, 15.0):
                self._recovery_return(p); return

        if not self.get_parameter("direct_to_angled_hover").value:
            # ── Phase 6b: Reverse CIRC rotation back to vertical ──────────────
            self.get_logger().info(f"\n--- [Phase 6b] Reverse rotation to vertical ---")
            ok, traj_rot = self._plan(self._build_pilz_lin(
                *hover_xyz, q_vertical, vel_scale, link=tip_link))
            tip_rotated = ok
            if ok:
                self.get_logger().info("  [PASS] reverse tip-pivot rotation planned.")
                if execute and self._wait_for_user("Phase6b_tip_rotate") and not self._execute_fjt(traj_rot, "Phase6b_tip_rotate"):
                    self._recovery_return(p); return
            else:
                req = self._build_pilz_circ(
                    ee_end_circ, ee_via_circ, ee_start_circ,
                    q_tilted, q_via, q_vertical,
                    vel_scale)
                ok, traj_circ_rev = self._plan(req, timeout=20.0)
            if tip_rotated:
                pass
            elif not ok:
                # Fallback: two LIN moves back
                self.get_logger().warn("  Reverse CIRC failed — using LIN fallback.")
                req_via = self._build_pilz_lin(*ee_via_circ, q_via, vel_scale)
                req_vert= self._build_pilz_lin(*ee_start_circ, q_vertical, vel_scale)
                ok_v, t_v = self._plan(req_via)
                if ok_v and execute:
                    if self._wait_for_user("Phase6b_via"): self._execute_fjt(t_v, "Phase6b_via")
                ok_vert, t_vert = self._plan(req_vert)
                if ok_vert and execute:
                    if self._wait_for_user("Phase6b_vertical"): self._execute_fjt(t_vert, "Phase6b_vertical")
            else:
                self.get_logger().info("  [PASS] Reverse CIRC planned.")
                if execute:
                    if self._wait_for_user("Phase6b_circ_reverse"): self._execute_fjt(traj_circ_rev, "Phase6b_circ_reverse")

        req_return = None
        if ret and self._start_joints:
            req_return = MotionPlanRequest()
            req_return.group_name = group_name
            req_return.planner_id = "PTP"
            req_return.pipeline_id = "pilz_industrial_motion_planner"
            req_return.num_planning_attempts = 1
            req_return.allowed_planning_time = 10.0
            req_return.max_velocity_scaling_factor = vel_scale
            req_return.max_acceleration_scaling_factor = vel_scale * 0.5
            req_return.start_state.is_diff = True
            goal_c = Constraints()
            for name, pos in self._start_joints.items():
                if name not in self.arm_joint_names:
                    continue
                jc = JointConstraint(); jc.joint_name = name
                # Wrap to nearest equivalent of the current winding so this
                # return doesn't hit the same 2pi abort as Phase 0
                # (mirrors _recovery_return and home_move).
                jc.position = self._nearest_equiv_angle(name, pos)
                jc.tolerance_above = 0.05; jc.tolerance_below = 0.05
                jc.weight = 1.0; goal_c.joint_constraints.append(jc)
            req_return.goal_constraints.append(goal_c)

        transit_78 = "fallback"
        if not self.get_parameter("direct_to_angled_hover").value and blend and req_return is not None:
            # Phase 7's LIN starts where Phase 6b ended (vertical at hover).
            seg7 = math.dist(ee_hover, ee_ready)
            transit_78 = self._blended_transit(
                [self._build_pilz_lin(*ee_ready, q_vertical, vel_scale), req_return],
                [seg7, seg7], "Phase7-8_blended", execute)
            if transit_78 == "failed":
                self._recovery_return(p); return

        if not self.get_parameter("direct_to_angled_hover").value and transit_78 == "fallback":
            # ── Phase 7: Vertical ascent + return ─────────────────────────────
            self.get_logger().info(f"\n--- [Phase 7] Vertical ascent to approach height ---")
            if execute and self._wait_for_user("Phase7_vertical_ascent"):
                if self._use_kortex_lin():
                    if not self._execute_kortex_lin(ee_ready[0], ee_ready[1], ee_ready[2], (q_vertical.x, q_vertical.y, q_vertical.z, q_vertical.w), "Phase7_vertical_ascent"):
                        self._recovery_return(p); return
                else:
                    req = self._build_pilz_lin(*ee_ready, q_vertical, vel_scale)
                    ok, traj_up = self._plan(req)
                    if ok:
                        if not self._execute_fjt(traj_up, "Phase7_vertical_ascent"):
                            self._recovery_return(p); return
                    else:
                        self.get_logger().warn("  Vertical ascent failed — skipping.")

        if req_return is not None and transit_78 == "fallback":
            self.get_logger().info(f"\n--- [Phase 8] Return to start joints ---")
            ok, traj = self._plan(req_return)
            if ok and execute and self._wait_for_user("Phase8_return"):
                self._execute_moveit(traj, "Phase8_return", timeout=90.0)

        

        self._run_log["reached_end"] = True
        self.get_logger().info("=" * 62)
        self.get_logger().info("angled_insert complete.")
        

        self.get_logger().info("=" * 62)


def main(args=None):
    rclpy.init(args=args)
    node = AngledInserter()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    _interrupted = threading.Event()

    def _sigint(sig, frame):
        if _interrupted.is_set():
            sys.exit(1)
        _interrupted.set()
        node._run_done.set()

    signal.signal(signal.SIGINT, _sigint)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    if node.get_parameter("use_action_server").value:
        # Action-server mode: wait for goals on /insert_container.
        node.get_logger().info("Running as action server — waiting for goals on /insert_container.")
        node._run_done.wait()
    else:
        # Standalone mode: run a single insertion from parameters, then exit.
        import types
        node.get_logger().info(
            "Running standalone — one insertion from parameters "
            "(set use_action_server:=true to wait for action goals instead).")
        req = types.SimpleNamespace(
            target_x        = node.get_parameter("target_x").value,
            target_y        = node.get_parameter("target_y").value,
            hover_above_top = node.get_parameter("hover_above_top").value,
            dry_run         = not node.get_parameter("execute_motion").value,
            skip_home_move  = node.get_parameter("skip_home_move").value,
        )
        gh = types.SimpleNamespace(request=req)

        def _pub_fb(phase, prog):
            node.get_logger().info(f"  [feedback] {phase}: {float(prog)*100:.0f}%")

        try:
            node._run_impl(gh, _pub_fb)
            node.get_logger().info("Standalone insertion finished.")
        except Exception as e:
            node.get_logger().error(f"Standalone insertion error: {e}")
        node._run_done.set()

    executor.shutdown(timeout_sec=2.0)
    spin_thread.join(timeout=3.0)

    if _interrupted.is_set():
        print("\n[Ctrl+C] Stopping arm...")
        try:
            node._recovery_return(dict(
                group_name=node.get_parameter("move_group_name").value,
                vel_scale=node.get_parameter("max_velocity_scaling").value,
                execute=True,
            ))
        except Exception as e:
            print(f"Recovery error: {e}")

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
