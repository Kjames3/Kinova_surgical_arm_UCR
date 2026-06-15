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
  ros2 run surgical_arm_bringup angled_insert.py \\
      --ros-args -p real_robot:=true -p execute_motion:=true \\
      -p skip_home_move:=true

  # Specify target at launch (mm from container centre, depth below top):
  ros2 run surgical_arm_bringup angled_insert.py \\
      --ros-args -p real_robot:=true -p execute_motion:=true \\
      -p skip_home_move:=true \\
      -p target_offset_x_mm:=10.0 \\
      -p target_offset_y_mm:=-15.0 \\
      -p target_depth_mm:=40.0

  # Or use interactive prompt (default):
  ros2 run surgical_arm_bringup angled_insert.py \\
      --ros-args -p real_robot:=true -p execute_motion:=true \\
      -p skip_home_move:=true -p interactive_target:=true
"""

import math
import signal
import sys
import threading
import time

import rclpy
import rclpy.time
from rclpy.node import Node
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
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.action import ExecuteTrajectory
from builtin_interfaces.msg import Duration as RosDuration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from moveit_msgs.msg import (
    RobotState, Constraints, OrientationConstraint,
    MotionPlanRequest, WorkspaceParameters,
    PositionConstraint, BoundingVolume, JointConstraint,
    RobotTrajectory,
)
from shape_msgs.msg import SolidPrimitive

# Re-use constants and helpers from insert_to_container
# (assumes both scripts are in the same package)
_GEN3_JOINTS = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7",
]

_GEN3_HOME_JOINTS = {
    "joint_1":  0.0000,
    "joint_2": -0.3049,
    "joint_3": -3.1416,
    "joint_4": -1.6607,
    "joint_5":  0.0000,
    "joint_6": -1.7928,
    "joint_7": -0.0006,
}

# Container physical dimensions (metres)
CONTAINER_WIDTH_M  = 0.090   # 90 mm
CONTAINER_HEIGHT_M = 0.086   # 86 mm
CONTAINER_WALL_M   = 0.003   # 3 mm safety margin from inner wall
CONTAINER_INNER_R  = (CONTAINER_WIDTH_M / 2.0) - CONTAINER_WALL_M  # 42 mm

MAX_TILT_DEG = 25.0   # hard limit — beyond this hits the wall at shallow depths

_MAX_JOINT_VEL_RAD_S  = 0.8
_MAX_JOINT_ACC_RAD_S2 = 0.4


# ---------------------------------------------------------------------------
# Quaternion helpers (duplicated from insert_to_container for standalone use)
# ---------------------------------------------------------------------------

def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )

def _quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])

def _quat_normalize(q):
    n = math.sqrt(sum(v*v for v in q))
    return tuple(v/n for v in q)

def _rotate_vector_by_quat(v, q):
    qv  = (v[0], v[1], v[2], 0.0)
    qc  = _quat_conjugate(q)
    res = _quat_multiply(_quat_multiply(q, qv), qc)
    return res[0], res[1], res[2]

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
    # Step 2: rotation about world X (after azimuth rotation) by tilt
    # The tilt axis in world frame is perpendicular to azimuth in XY plane
    tilt_ax = (-math.sin(azimuth_rad), math.cos(azimuth_rad), 0.0)
    q_tilt  = _quat_from_axis_angle(tilt_ax, tilt_rad)
    # Compose: q_new = q_tilt * q_az * q_vertical
    q_new = _quat_multiply(q_tilt, _quat_multiply(q_az, q_vertical))
    return _quat_normalize(q_new)


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
                          tilt_rad, tip_radius_m=0.003):
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

    if tilt_rad > math.radians(MAX_TILT_DEG):
        return False, (f"Tilt {math.degrees(tilt_rad):.1f}° > "
                       f"max {MAX_TILT_DEG}°")

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

class AngledInserter(Node):

    def __init__(self):
        super().__init__("angled_inserter")
        self._cb_group = ReentrantCallbackGroup()

        # Container position (same defaults as insert_to_container)
        self.declare_parameter("container_x",          0.260)  # m, world frame
        self.declare_parameter("container_y",          0.000)  # m, world frame
        self.declare_parameter("table_z",             -0.030)  # m
        self.declare_parameter("container_height",     0.086)  # m — 86 mm
        self.declare_parameter("hover_above_top",      0.030)  # m above container top

        # Approach height — must match home tip height minus container_top.
        # Same rule as insert_to_container: ready_z ≈ home tip z.
        self.declare_parameter("approach_clearance",   0.18)   # m above container top

        # Target point inside container (mm from container centre, depth from top)
        # These are overridden by the interactive prompt when interactive_target=true.
        self.declare_parameter("target_offset_x_mm",  10.0)    # +X = away from robot
        self.declare_parameter("target_offset_y_mm",  -30.0)   # +Y = left, -Y = right
        self.declare_parameter("target_depth_mm",     30.0)    # mm below container top

        # If true, prompt user to enter target in mm at runtime
        self.declare_parameter("interactive_target",   False)

        # Motion parameters
        self.declare_parameter("max_velocity_scaling", 0.15)
        self.declare_parameter("execute_motion",       False)
        self.declare_parameter("real_robot",           False)
        self.declare_parameter("skip_home_move",       True)
        self.declare_parameter("return_to_start",      True)
        self.declare_parameter("post_insert_wait",     2.0)
        self.declare_parameter("tip_link",             "assembly_tip")
        self.declare_parameter("ee_link",              "bracelet_link")
        self.declare_parameter("world_frame",          "world")
        self.declare_parameter("move_group_name",      "manipulator")

        # use_current_orientation: use live EE quaternion as pen-down target
        # (same meaning as insert_to_container — keep true unless you've
        #  measured the exact pen-down quaternion separately)
        self.declare_parameter("use_current_orientation", True)

        # CIRC arc parameters
        # n_circ_via_points: number of intermediate via-points for the arc.
        # Pilz CIRC needs exactly ONE via-point at the midpoint of the arc.
        # This parameter is kept for documentation; do not change from 1.
        self.declare_parameter("n_circ_via_points", 1)

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)

        # Joint state snapshot
        self._start_joints = {}
        self._js_frozen    = False
        self._js_sub       = self.create_subscription(
            JointState, "/joint_states", self._js_cb, 10,
            callback_group=self._cb_group)

        # Clients
        self._plan_cli = self.create_client(
            GetMotionPlan, "/plan_kinematic_path",
            callback_group=self._cb_group)
        self._execute_cli = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory",
            callback_group=self._cb_group)
        self._fjt_cli = ActionClient(
            self, FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self._cb_group)

        self._active_gh  = None
        self._arm_moved  = False
        self._run_done   = threading.Event()
        self._action_goal_handle = None

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

        self._done = False

    # ------------------------------------------------------------------
    def _js_cb(self, msg):
        if self._js_frozen:
            return
        for name, pos in zip(msg.name, msg.position):
            self._start_joints[name] = pos

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
    @staticmethod
    def _clamp_traj(traj):
        pts = traj.joint_trajectory.points
        for i, pt in enumerate(pts):
            if not pt.velocities and not pt.accelerations:
                continue
            worst = 1.0
            for v in pt.velocities:
                if abs(v) > _MAX_JOINT_VEL_RAD_S:
                    worst = max(worst, abs(v) / _MAX_JOINT_VEL_RAD_S)
            for a in pt.accelerations:
                if abs(a) > _MAX_JOINT_ACC_RAD_S2:
                    worst = max(worst, (abs(a) / _MAX_JOINT_ACC_RAD_S2) ** 0.5)
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
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute_moveit(self, traj, label, timeout=120.0):
        """Execute via MoveIt /execute_trajectory (OMPL / Pilz PTP paths)."""
        self._clamp_traj(traj)
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
        self.get_logger().warn(f"  [WARN] {label} error {result.result.error_code.val}")
        return False

    def _execute_fjt(self, traj, label, timeout=60.0):
        """Execute via direct FJT (Cartesian paths — overrides 0.1 rad path tol)."""
        self._clamp_traj(traj)
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
        for name in _GEN3_JOINTS:
            pt = JointTolerance(); pt.name = name
            pt.position = 5.0; pt.velocity = 0.0; pt.acceleration = 0.0
            fjt.path_tolerance.append(pt)
            gt = JointTolerance(); gt.name = name
            gt.position = 0.05; gt.velocity = 0.0; gt.acceleration = 0.0
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
        if result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(f"  [PASS] {label}")
            self._arm_moved = True
            return True
        self.get_logger().warn(f"  [WARN] {label} FJT error {result.result.error_code}")
        return False

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _plan(self, req, timeout=45.0):
        if not self._plan_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("  /plan_kinematic_path not available.")
            return False, None
        svc = GetMotionPlan.Request()
        svc.motion_plan_request = req
        f      = self._plan_cli.call_async(svc)
        result = self._wait_for_future(f, timeout)
        if result is None:
            self.get_logger().error("  Planning timed out.")
            return False, None
        resp = result.motion_plan_response
        if resp.error_code.val == 1:
            return True, resp.trajectory
        self.get_logger().error(f"  Planning failed (code {resp.error_code.val}).")
        return False, None

    def _build_pilz_ptp(self, ee_x, ee_y, ee_z, pen_q, vel_scale,
                         start_state=None):
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
            for name in _GEN3_JOINTS:
                cur = self._start_joints.get(name)
                if cur is None:
                    continue
                jc = JointConstraint()
                jc.joint_name = name; jc.position = cur
                jc.tolerance_above = 1.2; jc.tolerance_below = 1.2
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

    def _build_pilz_lin(self, ee_x, ee_y, ee_z, pen_q, vel_scale,
                         start_state=None):
        """Pilz LIN straight-line Cartesian move to EE pose."""
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
        pos_c.link_name       = self.get_parameter("ee_link").value
        pos_c.constraint_region = bv; pos_c.weight = 1.0
        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self.get_parameter("world_frame").value
        ori_c.link_name       = self.get_parameter("ee_link").value
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
        path_c.position_constraints.append(pos_via)
        req.path_constraints = path_c

        return req

    # ------------------------------------------------------------------
    # EE pose computation
    # ------------------------------------------------------------------
    def _ee_from_tip(self, tip_xyz, q_xyzw, ee_world_offset):
        """Compute EE position from desired tip position and world-frame offset."""
        wx, wy, wz = ee_world_offset
        return (tip_xyz[0] + wx, tip_xyz[1] + wy, tip_xyz[2] + wz)

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
    def _recovery_return(self, p):
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
            if name not in _GEN3_JOINTS:
                continue
            jc = JointConstraint()
            jc.joint_name = name; jc.position = pos
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

    def _action_cancel_cb(self, goal_handle):
        self.get_logger().info("Received cancel request")
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

    def _run_impl(self, goal_handle, pub_fb):
        vel_scale  = self.get_parameter("max_velocity_scaling").value
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
            for name, position in _GEN3_HOME_JOINTS.items():
                jc = JointConstraint(); jc.joint_name = name
                jc.position = position
                jc.tolerance_above = 0.05; jc.tolerance_below = 0.05
                jc.weight = 1.0
                goal_c.joint_constraints.append(jc)
            req.goal_constraints.append(goal_c)
            ok, traj = self._plan(req)
            if not ok:
                self.get_logger().error("Home move planning failed.")
                return
            if execute:
                if not self._execute_moveit(traj, "home_move", timeout=90.0):
                    self.get_logger().error("Home move failed.")
                    return
            time.sleep(1.5)
            self.get_logger().info("Arm at Home.")

        self._js_frozen = True
        if real_robot:
            self.get_logger().info("Arm is ready.")

        # Get container position
        cont_x = goal_handle.request.target_x
        cont_y = goal_handle.request.target_y
        container_top = table_z + cont_h
        hover_z  = container_top + hover_top
        ready_z  = container_top + approach

        # Get target inside container
        if self.get_parameter("interactive_target").value:
            ox_mm, oy_mm, d_mm = self._prompt_target()
        else:
            ox_mm = self.get_parameter("target_offset_x_mm").value
            oy_mm = self.get_parameter("target_offset_y_mm").value
            d_mm  = self.get_parameter("target_depth_mm").value

        ox_m = ox_mm / 1000.0
        oy_m = oy_mm / 1000.0
        d_m  = d_mm  / 1000.0

        # Target point in world frame
        target_xyz = (cont_x + ox_m, cont_y + oy_m, container_top - d_m)
        hover_xyz  = (cont_x, cont_y, hover_z)

        # Compute tilt geometry
        geo = compute_tilt_geometry(hover_xyz, target_xyz)

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
        ok, msg = check_wall_clearance(ox_m, oy_m, d_m, geo["tilt_rad"])
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
            f"  EE→tip world offset: "
            f"({ee_world_offset[0]:.4f}, {ee_world_offset[1]:.4f}, "
            f"{ee_world_offset[2]:.4f})  "
            f"mag={math.sqrt(sum(v**2 for v in ee_world_offset))*100:.1f} cm")

        # Current orientation
        r = ee_tf.transform.rotation
        if self.get_parameter("use_current_orientation").value:
            q_vertical_xyzw = (r.x, r.y, r.z, r.w)
        else:
            q_vertical_xyzw = (0.5, 0.5, 0.5, 0.5)  # pen-down default

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

        # ── Phase 0: Approach above container ────────────────────────────
        self.get_logger().info(
            f"\n--- [Phase 0] Approach above container ---")
        ee_ready = self._ee_from_tip(
            (cont_x, cont_y, ready_z), q_vertical_xyzw, ee_world_offset)
        self.get_logger().info(
            f"  EE target: ({ee_ready[0]:.3f}, {ee_ready[1]:.3f}, {ee_ready[2]:.3f})")
        req = self._build_pilz_ptp(*ee_ready, q_vertical, vel_scale)
        ok, traj = self._plan(req)
        if not ok:
            self.get_logger().error("  Phase 0 planning failed.")
            return
        if execute and not self._execute_moveit(traj, "Phase0_approach"):
            self._recovery_return(p); return

        # ── Phase 1: Vertical descent to hover_z ─────────────────────────
        self.get_logger().info(f"\n--- [Phase 1] Vertical descent to hover_z ---")
        ee_hover = self._ee_from_tip(
            hover_xyz, q_vertical_xyzw, ee_world_offset)
        self.get_logger().info(
            f"  EE hover: ({ee_hover[0]:.3f}, {ee_hover[1]:.3f}, {ee_hover[2]:.3f})")
        req = self._build_pilz_lin(*ee_hover, q_vertical, vel_scale)
        ok, traj = self._plan(req)
        if not ok:
            self.get_logger().error("  Phase 1 planning failed.")
            return
        if execute and not self._execute_fjt(traj, "Phase1_vertical_descent"):
            self._recovery_return(p); return

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
            _ASSEMBLY_TIP_OFFSET["x"],
            _ASSEMBLY_TIP_OFFSET["y"],
            _ASSEMBLY_TIP_OFFSET["z"],
        )

        def _ee_at_orientation(q_xyzw):
            """EE position when tip is at hover_xyz and EE has orientation q."""
            # tip = EE + R * local_offset  →  EE = tip - R * local_offset
            rot_offset = _rotate_vector_by_quat(local_tip_offset, q_xyzw)
            return (hover_xyz[0] - rot_offset[0],
                    hover_xyz[1] - rot_offset[1],
                    hover_xyz[2] - rot_offset[2])

        ee_start_circ = ee_hover                          # = _ee_at_orientation(q_vertical_xyzw)
        ee_via_circ   = _ee_at_orientation(q_via_xyzw)
        ee_end_circ   = _ee_at_orientation(q_tilted_xyzw)

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

        req = self._build_pilz_circ(
            ee_start_circ, ee_via_circ, ee_end_circ,
            q_vertical, q_via, q_tilted,
            vel_scale)
        ok, traj = self._plan(req, timeout=20.0)

        if not ok:
            self.get_logger().warn(
                "  Pilz CIRC failed — falling back to two sequential Pilz LIN "
                "moves (via half-tilt, then full tilt).\n"
                "  Tip will move slightly during rotation (< 2mm for small tilts).")
            # Fallback: LIN to via then LIN to end
            req_via = self._build_pilz_lin(*ee_via_circ, q_via, vel_scale)
            ok_via, traj_via = self._plan(req_via)
            req_end = self._build_pilz_lin(*ee_end_circ, q_tilted, vel_scale)
            ok_end, traj_end = self._plan(req_end)
            if not ok_via or not ok_end:
                self.get_logger().error("  Rotation fallback planning failed.")
                self._recovery_return(p); return
            if execute:
                if not self._execute_fjt(traj_via, "Phase3a_rotate_via"):
                    self._recovery_return(p); return
                if not self._execute_fjt(traj_end, "Phase3b_rotate_end"):
                    self._recovery_return(p); return
            traj = None   # signal that CIRC was not used
        else:
            self.get_logger().info("  [PASS] CIRC rotation planned.")
            if execute and not self._execute_fjt(traj, "Phase3_circ_rotate"):
                self._recovery_return(p); return

        # ── Phase 4: Angled descent to target ─────────────────────────────
        self.get_logger().info(f"\n--- [Phase 4] Angled descent to target ---")

        # EE position at target: tip is at target_xyz, EE has tilted orientation
        ee_target = _ee_at_orientation(q_tilted_xyzw)
        ee_target = (target_xyz[0] - _rotate_vector_by_quat(
                         local_tip_offset, q_tilted_xyzw)[0],
                     target_xyz[1] - _rotate_vector_by_quat(
                         local_tip_offset, q_tilted_xyzw)[1],
                     target_xyz[2] - _rotate_vector_by_quat(
                         local_tip_offset, q_tilted_xyzw)[2])

        self.get_logger().info(
            f"  Tip target:  ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, "
            f"{target_xyz[2]:.3f})\n"
            f"  EE target:   ({ee_target[0]:.3f}, {ee_target[1]:.3f}, "
            f"{ee_target[2]:.3f})\n"
            f"  Descent:     {geo['descent_m']*1000:.1f} mm along "
            f"{math.degrees(geo['tilt_rad']):.1f}° axis")

        req = self._build_pilz_lin(*ee_target, q_tilted, vel_scale)
        ok, traj_descent = self._plan(req)
        if not ok:
            self.get_logger().error("  Phase 4 angled descent planning failed.")
            self._recovery_return(p); return
        if execute and not self._execute_fjt(traj_descent, "Phase4_angled_descent"):
            self._recovery_return(p); return

        # ── Phase 5: Hold ──────────────────────────────────────────────────
        self.get_logger().info(f"\n--- [Phase 5] Hold at target ---")
        if execute:
            try:
                input(f"  Tip at target ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, "
                      f"{target_xyz[2]:.3f}).  Press ENTER to reverse ...")
            except EOFError:
                time.sleep(post_wait)
        else:
            self.get_logger().info("  Dry-run — skipping hold.")

        # ── Phase 6a: Reverse angled ascent ───────────────────────────────
        self.get_logger().info(f"\n--- [Phase 6a] Reverse angled ascent ---")
        req = self._build_pilz_lin(*ee_end_circ, q_tilted, vel_scale)
        ok, traj_asc = self._plan(req)
        if ok:
            if execute and not self._execute_fjt(traj_asc, "Phase6a_angled_ascent"):
                self._recovery_return(p); return
        else:
            self.get_logger().warn("  Angled ascent planning failed — attempting recovery.")
            self._recovery_return(p); return

        # ── Phase 6b: Reverse CIRC rotation back to vertical ──────────────
        self.get_logger().info(f"\n--- [Phase 6b] Reverse CIRC rotation to vertical ---")
        req = self._build_pilz_circ(
            ee_end_circ, ee_via_circ, ee_start_circ,
            q_tilted, q_via, q_vertical,
            vel_scale)
        ok, traj_circ_rev = self._plan(req, timeout=20.0)
        if not ok:
            # Fallback: two LIN moves back
            self.get_logger().warn("  Reverse CIRC failed — using LIN fallback.")
            req_via = self._build_pilz_lin(*ee_via_circ, q_via, vel_scale)
            req_vert= self._build_pilz_lin(*ee_start_circ, q_vertical, vel_scale)
            ok_v, t_v = self._plan(req_via)
            ok_r, t_r = self._plan(req_vert)
            if execute:
                if ok_v:
                    self._execute_fjt(t_v, "Phase6b_via")
                if ok_r:
                    self._execute_fjt(t_r, "Phase6b_vertical")
        else:
            self.get_logger().info("  [PASS] Reverse CIRC planned.")
            if execute:
                self._execute_fjt(traj_circ_rev, "Phase6b_circ_reverse")

        # ── Phase 7: Vertical ascent + return ─────────────────────────────
        self.get_logger().info(f"\n--- [Phase 7] Vertical ascent to approach height ---")
        req = self._build_pilz_lin(*ee_ready, q_vertical, vel_scale)
        ok, traj_up = self._plan(req)
        if ok:
            if execute:
                self._execute_fjt(traj_up, "Phase7_vertical_ascent")
        else:
            self.get_logger().warn("  Vertical ascent failed — skipping.")

        if ret and self._start_joints:
            self.get_logger().info(f"\n--- [Phase 8] Return to start joints ---")
            req = MotionPlanRequest()
            req.group_name = group_name
            req.planner_id = "PTP"
            req.pipeline_id = "pilz_industrial_motion_planner"
            req.num_planning_attempts = 1
            req.allowed_planning_time = 10.0
            req.max_velocity_scaling_factor = vel_scale
            req.max_acceleration_scaling_factor = vel_scale * 0.5
            req.start_state.is_diff = True
            goal_c = Constraints()
            for name, pos in self._start_joints.items():
                if name not in _GEN3_JOINTS:
                    continue
                jc = JointConstraint(); jc.joint_name = name; jc.position = pos
                jc.tolerance_above = 0.05; jc.tolerance_below = 0.05
                jc.weight = 1.0; goal_c.joint_constraints.append(jc)
            req.goal_constraints.append(goal_c)
            ok, traj = self._plan(req)
            if ok and execute:
                self._execute_moveit(traj, "Phase8_return", timeout=90.0)

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
    node._run_done.wait()
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