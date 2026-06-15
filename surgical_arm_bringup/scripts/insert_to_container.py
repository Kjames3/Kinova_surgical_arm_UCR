#!/usr/bin/env python3
"""
Insert the assembly tip into the center of the glass container.

The glass container is 50 cm in front of the robot base and 20 cm to its left.
The goal is to position the assembly_tip (physical TCP of the writing assembly)
at the container's centre axis, hover_above_top metres above the container top,
while keeping the assembly vertical (tip-down orientation).

TCP frame: assembly_tip  (fixed joint from bracelet_link, offset=-0.027,0,-0.414 m)
EE frame:  bracelet_link (live TF frame — end_effector_link is URDF-static only)

Four-phase operation
--------------------
Phase 0 — Approach (Pilz PTP)
  Navigate to assembly_tip at (target_x, target_y, ready_z) with orientation locked.
  ready_z = container_top + approach_clearance ≈ home tip height (no Z descent during PTP).

Phase 1 — Descend (Pilz LIN)
  Straight-line TCP descent from ready_z to hover_target. Orientation locked.
  Equivalent to Xbox controller twist-linear mode.

Phase 2 — Hold
  Remain at hover position for post_insert_wait seconds.

Phase 2.5 — Ascend (Pilz LIN)
  Retract straight back up to ready_z.

Phase 3 — Return (Pilz PTP)
  Return to the joint configuration captured at startup.


Phase 1 — Descend (Cartesian)
  Move straight down from the approach height into the container to the
  insert depth (insert_depth_from_top below the container's top surface).
  Dense waypoints + jump threshold prevent joint flips.

Phase 2 — Hold at insert position
  Remain at the insert position for post_insert_wait seconds.
  (Extend this phase later for fluid dispensing.)

Phase 2.5 — Ascend (Cartesian)
  Retract straight back up to the approach height.

Phase 3 — Return (OMPL)
  Return to the joint configuration captured at startup.

Parameters
----------
  target_x              (float, default 0.50)  : container centre X in world frame
                                                  (50 cm in front of robot base)
  target_y              (float, default -0.20) : container centre Y in world frame
                                                  (20 cm to the left from operator's perspective facing the robot)
  table_z               (float, default -0.03) : table surface Z in world frame
  container_height      (float, default 0.10)  : height of the glass container [m]
                                                  Measure and set before running.
  insert_depth_from_top (float, default 0.010) : how far below the container's
                                                  top surface to insert the tip [m]
  approach_clearance    (float, default 0.05)  : approach height above container
                                                  top before descending [m]
  post_insert_wait      (float, default 2.0)   : seconds to hold at insert position

  tip_ee_offset_x       (float, default -0.11726) : x-component of EE→pen_tip
  tip_ee_offset_y       (float, default  0.32815) : y-component (negated joint xyz)
  tip_ee_offset_z       (float, default  0.00288) : z-component
                          These represent the vector from pen_tip back to EE origin,
                          expressed in EE frame:  -pen_tip_joint_xyz.
                          Update if the pen_tip joint offset changes in the URDF.

  execute_motion        (bool,  default false) : false = dry-run | true = MOVES ARM
  return_to_start       (bool,  default true)  : return to start joints after insert
  max_velocity_scaling  (float, default 0.08)  : joint velocity limit fraction 0-1
  tip_link              (str,   default pen_tip)
  ee_link               (str,   default end_effector_link)
  world_frame           (str,   default world)
  move_group_name       (str,   default manipulator)
  pen_down_qx/y/z/w     Pen-vertical quaternion (world -> end_effector_link).
                         Re-measure with: ros2 run tf2_ros tf2_echo world end_effector_link
  vertical_tilt_tol     (float, default 0.05)  : orientation tolerance [rad]
  joint_delta_rad       (float, default 1.2)   : OMPL path constraint half-width [rad]
  descend_step          (float, default 0.01)  : vertical waypoint spacing [m]
  descend_jump_threshold(float, default 3.0)   : max joint delta between IK samples [rad]
  add_pole_collision    (bool,  default false) : add pole to scene before planning
  pole_x/y/radius/height                       : pole parameters (if add_pole_collision)

Usage
-----
  ros2 run surgical_arm_bringup insert_to_container.py                      # dry-run
  ros2 run surgical_arm_bringup insert_to_container.py \\
      --ros-args -p execute_motion:=true                              # MOVES ARM
  ros2 run surgical_arm_bringup insert_to_container.py \\
      --ros-args -p execute_motion:=true -p container_height:=0.085  # set container height
"""

import math
from kortex_utils import *
import signal
import sys
import threading
import time
import concurrent.futures

import rclpy
import rclpy.time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from rclpy.duration import Duration as RclpyDuration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse

import tf2_ros
from geometry_msgs.msg import Pose, Quaternion, Point
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan, ApplyPlanningScene
from moveit_msgs.action import ExecuteTrajectory
from builtin_interfaces.msg import Duration as RosDuration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from moveit_msgs.msg import (
    RobotState, Constraints, OrientationConstraint,
    MotionPlanRequest, WorkspaceParameters,
    PositionConstraint, BoundingVolume, JointConstraint,
    RobotTrajectory, PlanningScene, CollisionObject,
    DisplayTrajectory,
)
from shape_msgs.msg import SolidPrimitive

# Action interface — defined inline using a SimpleNamespace so no separate
# .action file is required.  A calling script creates a goal dict and sends
# it via the /insert_container ROS2 action topic.
# For production use, generate a proper surgical_arm_bringup/action/InsertContainer.action.
try:
    from surgical_arm_bringup.action import InsertContainer
    _HAS_ACTION_INTERFACE = True
except ImportError:
    _HAS_ACTION_INTERFACE = False


_GEN3_JOINTS = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7",
]

# Home position — calibrated 2025-05-06, assembly_tip vertical to <0.5°.
# RPY at home: (-0.000°, 0.403°, -0.038°) — essentially zero tilt.
# assembly_tip world at home: (0.150, 0.001, 0.234)
# Joint values remapped from /joint_states (topic order: j1,j2,j4,j5,j3,j6,j7).
_GEN3_HOME_JOINTS = {
    "joint_1":  0.0000,   #    0.0°
    "joint_2": -0.3049,   #  -17.5°  shoulder raised for clearance
    "joint_3": -3.1416,   # -180.0°  (note: negative, matches measured value)
    "joint_4": -1.6607,   #  -95.2°
    "joint_5":  0.0000,   #    0.0°
    "joint_6": -1.7928,   # -102.7°  wrist tuned for vertical tip
    "joint_7": -0.0006,   #    0.0°
}

# Measured assembly tip offset from bracelet_link (in bracelet_link local frame).
# Derived from: tip_z = bracelet_z + offset_z  (when arm points straight down)
#   offset_z = table_z - bracelet_z_at_touch = -0.030 - 0.383 = -0.413 m
#   offset_x ≈ -0.027 m (from 3.8° tilt × 0.413 m)
#   offset_y ≈  0.000 m
# Update the URDF assembly_tip joint xyz to these values and rebuild.
# Then use: ros2 run tf2_ros tf2_echo world assembly_tip to verify.
_ASSEMBLY_TIP_OFFSET = {
    "x": -0.027,   # m — lateral offset due to bracelet tilt
    "y":  0.000,   # m
    "z": -0.414,   # m — main drop length (average of two measurements)
}





# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Quaternion math helpers
# ---------------------------------------------------------------------------

def _tip_to_ee_pose(tip_x, tip_y, tip_z, q_xyzw, ee_offset_xyz):
    """
    Given a desired pen_tip world position and the EE quaternion
    (world -> end_effector_link), return the end_effector_link origin in world.

    Geometry:
      pen_tip = EE_origin + R * pen_tip_joint_xyz
      => EE_origin = pen_tip - R * pen_tip_joint_xyz
                   = pen_tip + R * ee_offset_xyz

    ee_offset_xyz = -pen_tip_joint_xyz (negated, in EE local frame).
    IMPORTANT: these values must match your actual URDF pen_tip joint xyz.
    Verify with: ros2 run tf2_ros tf2_echo end_effector_link pen_tip
    The translation shown is pen_tip_joint_xyz; negate it for ee_offset_xyz.
    """
    wx, wy, wz = rotate_vector_by_quat(ee_offset_xyz, q_xyzw)
    return tip_x + wx, tip_y + wy, tip_z + wz


# ---------------------------------------------------------------------------

class ContainerInserter(Node):

    def __init__(self):
        super().__init__("container_inserter")
        self._cb_group = ReentrantCallbackGroup()

        # --- Container target parameters ---
        # Container position — measured 2025-05-06 with assembly_tip physically
        # hovering 2cm above container centre while arm points down:
        #   ros2 run tf2_ros tf2_echo world assembly_tip → (0.299, -0.192, 0.084)
        # Container is 43cm forward and 20cm to the right of the robot base.
        # Negative Y = right side from robot's perspective.
        # Update these if the container is moved — jog tip over centre, re-echo.
        self.declare_parameter("target_x",              0.299)
        self.declare_parameter("target_y",             -0.192)

        # If true, prompt the user to enter container position in centimetres
        # at runtime instead of using the target_x / target_y parameters above.
        # The user types distances from the robot base:
        #   forward (cm) : positive = away from robot  (becomes target_x in metres)
        #   sideways (cm): negative = right, positive = left  (becomes target_y)
        # Example: "43, -20" → target_x=0.430, target_y=-0.200
        # This lets you reposition the container without restarting the script
        # or passing command-line parameters each time.
        self.declare_parameter("interactive_target",    False)
        self.declare_parameter("table_z",              -0.03)   # table surface Z
        # Container dimensions — physically measured:
        #   Square 90mm × 90mm base, height 86mm.
        #   container_top = table_z + container_height = -0.030 + 0.086 = 0.056 m
        self.declare_parameter("container_height",      0.086)  # 86mm measured

        # approach_clearance: calibrated from new level home position 2025-05-06.
        #   home tip Z = 0.234 m, container_top = 0.056 m
        #   clearance = 0.234 - 0.056 = 0.178 → 0.18
        self.declare_parameter("approach_clearance",    0.18)
        self.declare_parameter("insert_depth_from_top", 0.010)  # 10 mm from top
        # approach_clearance: how far above container_top to position the tip
        # before descending via Pilz LIN.
        #
        # MUST equal (home_tip_z - container_top) so ready_z matches the tip
        # height at home — Phase 0b PTP then only moves XY with no Z descent.
        #
        # Calibrated from log 2025-05-06:
        #   home tip Z    = 0.233 m  (from pre-flight log after home move)
        #   container_top = table_z + container_height = -0.030 + 0.100 = 0.070 m
        #   approach_clearance = 0.233 - 0.070 = 0.163 → rounded to 0.16
        self.declare_parameter("hover_above_top",       0.025)
        self.declare_parameter("lateral_rise",          0.20)

        # --- Motion parameters ---
        self.declare_parameter("post_insert_wait",     2.0)
        self.declare_parameter("max_velocity_scaling", 0.15)
        # transition_velocity_scaling: speed used for long-distance travel phases
        # (Phase 0 approach PTP and Phase 3 return PTP/OMPL).  Kept separate from
        # max_velocity_scaling so the physical insertion / retraction (Pilz LIN)
        # always runs at the safe, slow vel_scale.
        # Professor requested faster transitions — raise to 0.5 (50%) for approach
        # and return while keeping insertion at max_velocity_scaling (15%).
        self.declare_parameter("transition_velocity_scaling", 0.50)
        self.declare_parameter("execute_motion",       False)
        self.declare_parameter("return_to_start",      True)
        # assembly_tip is the new measured TCP frame in the URDF.
        # pen_tip has been removed from thesis_ee_macro.xacro.
        # bracelet_link is the live EE TF frame (end_effector_link is URDF-static only).
        self.declare_parameter("tip_link",             "assembly_tip")
        self.declare_parameter("ee_link",              "bracelet_link")
        self.declare_parameter("world_frame",          "world")
        self.declare_parameter("move_group_name",      "manipulator")

        # Tip-vertical quaternion (world -> end_effector_link, assembly tip pointing straight down).
        # EE +Y must point world +Z; exact value = RPY[90°,0°,90°] = (0.5, 0.5, 0.5, 0.5).
        # Re-measure with: ros2 run tf2_ros tf2_echo world end_effector_link
        # (run AFTER manually commanding arm to tip-down pose, not at ros2_control startup)
        self.declare_parameter("pen_down_qx",  0.5)
        self.declare_parameter("pen_down_qy",  0.5)
        self.declare_parameter("pen_down_qz",  0.5)
        self.declare_parameter("pen_down_qw",  0.5)
        # If true, use the arm's CURRENT EE orientation as the pen-down target
        # instead of the pen_down_qx/y/z/w parameters.
        # Use this when the arm is already in the correct insertion orientation
        # at startup — avoids the large reorientation move that causes joint_1
        # to swing 177° to reach a different IK solution family.
        # Measure the correct quaternion with:
        #   ros2 run tf2_ros tf2_echo world end_effector_link
        # then set pen_down_qx/y/z/w to those values and set this to false.
        self.declare_parameter("use_current_orientation", True)
        self.declare_parameter("vertical_tilt_tol", 0.05)
        self.declare_parameter("joint_delta_rad",   1.2)

        # Assembly tip offset from bracelet_link — calibrated 2025-05-06.
        # Measured by touching tip to table and reading bracelet_link Z from TF:
        #   offset_z = table_z - bracelet_z = -0.030 - 0.383 = -0.413 m
        # Cross-checked at 20cm above table: -0.416 m (3mm error = acceptable)
        # offset_x from 3.8° bracelet tilt × 0.414 m length = -0.027 m
        # These are negated in the script (ee_offset = -pen_tip_joint_xyz):
        #   tip_ee_offset_x = -(-0.027) = NOT applicable here — offset IS from bracelet
        # For bracelet_link as ee_link, the offset vector IS the local tip position:
        #   (x=-0.027, y=0.000, z=-0.414) in bracelet_link frame
        # Since use_tf_offset=true reads this from live TF subtraction, these
        # parameters are only used as fallback if TF lookup fails.
        self.declare_parameter("tip_ee_offset_x", -0.027)
        self.declare_parameter("tip_ee_offset_y",  0.000)
        self.declare_parameter("tip_ee_offset_z", -0.414)

        # If true, look up the EE→pen_tip offset from TF at runtime instead of
        # using the tip_ee_offset_x/y/z parameters above.  This is more reliable
        # when the URDF offset is uncertain.  Requires robot_state_publisher running.
        self.declare_parameter("use_tf_offset", True)

        self.declare_parameter("real_robot",             False)
        self.declare_parameter("skip_home_move",         False)
        self.declare_parameter("use_dynamic_tf",         False)
        self.declare_parameter("descend_step",          0.01)
        self.declare_parameter("descend_jump_threshold", 3.0)
        self.declare_parameter("add_pole_collision", False)
        self.declare_parameter("pole_x",      0.0)
        self.declare_parameter("pole_y",      0.0)
        self.declare_parameter("pole_radius", 0.025)
        self.declare_parameter("pole_height", 0.50)

        # --- Parallel planner parameters ---
        # Number of planning attempts to run in parallel for Phase 0b approach.
        # Each attempt uses a different OMPL random seed. The trajectory with the
        # lowest joint displacement score is selected and presented for confirmation.
        # Set to 1 to disable parallel planning (single attempt, original behaviour).
        self.declare_parameter("parallel_plans",   15)
        # Timeout for the parallel planning pool (seconds).
        # All n attempts must finish within this window.
        self.declare_parameter("parallel_timeout", 30.0)
        # Weight on joint_1 displacement in the scoring function (0.0–1.0).
        # Higher values penalise base rotation more heavily — reduces circular arcs.
        self.declare_parameter("j1_weight",        0.4)
        # Weight on trajectory duration in the scoring function (0.0–1.0).
        # 0.2 means 20% of the score is duration — breaks ties between paths
        # with similar displacement but different efficiency.
        # Increase to 0.4 if short paths are consistently better in your setup.
        self.declare_parameter("duration_weight",  0.2)

        # --- TF ---
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)

        # --- Joint state snapshot ---
        self._start_joints: dict = {}
        self._js_frozen = False
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._js_cb, 10,
            callback_group=self._cb_group)

        # --- Service / action clients ---
        self._plan_cli = self.create_client(
            GetMotionPlan, "/plan_kinematic_path",
            callback_group=self._cb_group)

        self._cartesian_cli = self.create_client(
            GetCartesianPath, "/compute_cartesian_path",
            callback_group=self._cb_group)

        self._execute_cli = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory",
            callback_group=self._cb_group)

        # Direct controller client used by _move_to_home to bypass MoveIt's
        # ExecuteTrajectory (which cannot override the controller's 0.1 rad
        # path tolerance configured in ros2_controllers.yaml).
        self._fjt_cli = ActionClient(
            self, FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self._cb_group)

        self._active_gh = None
        self._arm_moved = False
        self._run_done  = threading.Event()
        self._action_goal_handle = None   # tracks active action server goal

        from rclpy.qos import QoSProfile, DurabilityPolicy
        latched_qos = QoSProfile(depth=1,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._marker_pub = self.create_publisher(
            MarkerArray, "/insert_preview", latched_qos)

        # RViz "Motion Planning" display reads this topic to show the ghost
        # trajectory preview before the arm actually moves.
        self._display_traj_pub = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", latched_qos)

        # --- Action server (callable from other scripts) ---
        # Exposes the insertion sequence as a ROS2 action on /insert_container.
        # Goal fields (all optional — defaults come from node parameters):
        #   float64 target_x        container centre X (metres, world frame)
        #   float64 target_y        container centre Y (metres, world frame)
        #   float64 hover_above_top how far above container top to hover (metres)
        #   bool    dry_run         plan but do not move arm
        #   bool    skip_home_move  skip home move pre-phase
        # Feedback: string current_phase, float64 progress (0–1)
        # Result:   bool success, string message
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
            self.get_logger().info(
                "Action server ready on /insert_container\n"
                "  Call from another script:\n"
                "    ros2 action send_goal /insert_container "
                "surgical_arm_bringup/action/InsertContainer "
                "'{target_x: 0.299, target_y: -0.192, hover_above_top: 0.03}'")
        else:
            self.get_logger().warn(
                "InsertContainer action interface not found — "
                "action server disabled.\n"
                "  To enable: add InsertContainer.action to surgical_arm_bringup "
                "and rebuild.\n"
                "  Running in standalone timer mode.")
            self.create_timer(2.0, self._run_once, callback_group=self._cb_group)

        self.add_on_set_parameters_callback(self._parameter_callback)
        self._collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)
        self._camera_target = None
        self.create_subscription(PoseStamped, "/fused_marker_square_center", self._camera_target_cb, 10)
        self._done = False

    # ------------------------------------------------------------------


    def _camera_target_cb(self, msg: PoseStamped):
        self._camera_target = msg
    def _parameter_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name in ["container_height", "insert_depth_from_top", "target_x", "target_y", "hover_above_top", "approach_clearance"]:
                self.get_logger().info(f"Dynamically updated parameter {p.name} to {p.value}")
        return SetParametersResult(successful=True)

    def _js_cb(self, msg: JointState):
        if self._js_frozen:
            return
        for name, pos in zip(msg.name, msg.position):
            self._start_joints[name] = pos

    def _get_params(self):
        return dict(
            target_x        = self.get_parameter("target_x").value,
            target_y        = self.get_parameter("target_y").value,
            table_z         = self.get_parameter("table_z").value,
            container_h     = self.get_parameter("container_height").value,
            insert_depth    = self.get_parameter("insert_depth_from_top").value,
            approach        = self.get_parameter("approach_clearance").value,
            hover_above_top = self.get_parameter("hover_above_top").value,
            lateral_rise    = self.get_parameter("lateral_rise").value,
            post_wait       = self.get_parameter("post_insert_wait").value,
            vel_scale       = self.get_parameter("max_velocity_scaling").value,
            transit_vel     = self.get_parameter("transition_velocity_scaling").value,
            execute         = self.get_parameter("execute_motion").value,
            ret             = self.get_parameter("return_to_start").value,
            tip_link        = self.get_parameter("tip_link").value,
            ee_link         = self.get_parameter("ee_link").value,
            world_frame     = self.get_parameter("world_frame").value,
            group_name      = self.get_parameter("move_group_name").value,
            qx              = self.get_parameter("pen_down_qx").value,
            qy              = self.get_parameter("pen_down_qy").value,
            qz              = self.get_parameter("pen_down_qz").value,
            qw              = self.get_parameter("pen_down_qw").value,
            tilt_tol        = self.get_parameter("vertical_tilt_tol").value,
            joint_delta     = self.get_parameter("joint_delta_rad").value,
            tip_ee_ox       = self.get_parameter("tip_ee_offset_x").value,
            tip_ee_oy       = self.get_parameter("tip_ee_offset_y").value,
            tip_ee_oz       = self.get_parameter("tip_ee_offset_z").value,
            descend_step    = self.get_parameter("descend_step").value,
            descend_jump    = self.get_parameter("descend_jump_threshold").value,
            parallel_plans  = self.get_parameter("parallel_plans").value,
            parallel_timeout= self.get_parameter("parallel_timeout").value,
            j1_weight       = self.get_parameter("j1_weight").value,
            duration_weight = self.get_parameter("duration_weight").value,
        )

    # ------------------------------------------------------------------
    # Shared helpers (unchanged from trace_circle.py)
    # ------------------------------------------------------------------

    def _extract_end_state(self, traj: RobotTrajectory):
        jt = traj.joint_trajectory
        if not jt.points:
            return None
        state = RobotState()
        state.joint_state.name     = list(jt.joint_names)
        state.joint_state.position = list(jt.points[-1].positions)
        return state

    @staticmethod
    def _rescale_cartesian_traj(solution: RobotTrajectory, vel_scale: float) -> RobotTrajectory:
        """Scale Cartesian-path trajectory timestamps/velocities/accels in-place.

        GetCartesianPath.Request has no max_velocity_scaling_factor in Humble;
        compute_cartesian_path plans at 100% speed. This post-processes the result
        so the arm moves at vel_scale fraction of maximum joint velocity.
        After time-scaling, a smooth-velocity pass is applied so the resulting
        trajectory has natural acceleration/deceleration rather than hard steps.
        """
        pts = solution.joint_trajectory.points
        if not pts or vel_scale <= 0.0 or vel_scale >= 1.0:
            return solution
        inv = 1.0 / vel_scale
        for pt in pts:
            ns = pt.time_from_start.sec * 1_000_000_000 + pt.time_from_start.nanosec
            scaled_ns = int(ns * inv)
            pt.time_from_start.sec     = scaled_ns // 1_000_000_000
            pt.time_from_start.nanosec = scaled_ns %  1_000_000_000
            pt.velocities    = [v * vel_scale          for v in pt.velocities]
            pt.accelerations = [a * vel_scale * vel_scale for a in pt.accelerations]
        # Re-derive smooth velocities/accelerations from the new timestamps
        smooth_traj_velocities(solution)
        return solution

    def _preflight_check(self, p: dict) -> bool:
        """
        Run before any motion begins on the real robot.

        Checks:
          1. TF is alive and pen_tip is being published
          2. Pen_tip is within a reasonable distance of the target
             (catches wrong container position parameters)
          3. Current EE orientation is not wildly different from pen-down
             (catches arm being in a completely wrong configuration)
          4. Joint states are being received
          5. All required services are available

        Returns True if safe to proceed, False to abort.
        """
        self.get_logger().info("\n─── Pre-flight safety check ───────────────────────")
        ok = True

        # 1. Joint states
        if not self._start_joints:
            self.get_logger().error(
                "  [PRE-FLIGHT] FAIL: No /joint_states received.\n"
                "  Is ros2_control_node running?  Check: ros2 topic hz /joint_states")
            ok = False
        else:
            js_str = {n: round(math.degrees(v), 1)
                      for n, v in self._start_joints.items()
                      if n in _GEN3_JOINTS}
            self.get_logger().info(f"  [PRE-FLIGHT] Joint states OK: {js_str}")

        # 2. TF
        tip_tf, ee_tf = self._get_tip_tf(p, timeout_sec=3.0)
        if tip_tf is None:
            self.get_logger().error(
                "  [PRE-FLIGHT] FAIL: Cannot read world→pen_tip TF.\n"
                "  Is robot_state_publisher running?\n"
                "  Check: ros2 run tf2_ros tf2_echo world pen_tip")
            ok = False
        else:
            t = tip_tf.transform.translation
            ee_t = ee_tf.transform.translation if ee_tf else None

            # 3. Distance to target + EE needed position
            dx = t.x - p["target_x"]
            dy = t.y - p["target_y"]
            dist_xy = math.sqrt(dx*dx + dy*dy)

            if ee_t is not None:
                # World-frame EE→tip offset at current arm pose
                cur_offset_x = ee_t.x - t.x
                cur_offset_y = ee_t.y - t.y
                cur_offset_z = ee_t.z - t.z
                ee_needed_x = p["target_x"] + cur_offset_x
                ee_needed_y = p["target_y"] + cur_offset_y
                offset_mag = math.sqrt(cur_offset_x**2 + cur_offset_y**2 + cur_offset_z**2)
                self.get_logger().info(
                    f"  [PRE-FLIGHT] pen_tip world:  ({t.x:.3f}, {t.y:.3f}, {t.z:.3f})\n"
                    f"  [PRE-FLIGHT] EE_origin world: ({ee_t.x:.3f}, {ee_t.y:.3f}, {ee_t.z:.3f})\n"
                    f"  [PRE-FLIGHT] EE→tip offset (world): "
                    f"({cur_offset_x:.3f}, {cur_offset_y:.3f}, {cur_offset_z:.3f})"
                    f"  magnitude={offset_mag*100:.1f} cm  (PDF: 34.8 cm expected)\n"
                    f"  [PRE-FLIGHT] Container target XY: ({p['target_x']:.3f}, {p['target_y']:.3f})\n"
                    f"  [PRE-FLIGHT] pen_tip→container distance: {dist_xy*100:.1f} cm\n"
                    f"  [PRE-FLIGHT] EE must reach: ({ee_needed_x:.3f}, {ee_needed_y:.3f}) "
                    f"for pen_tip to be over container")
            else:
                self.get_logger().info(
                    f"  [PRE-FLIGHT] pen_tip TF OK: ({t.x:.3f}, {t.y:.3f}, {t.z:.3f})\n"
                    f"  [PRE-FLIGHT] pen_tip→container distance: {dist_xy*100:.1f} cm")

            if dist_xy > MAX_TARGET_DIST_M:
                self.get_logger().warn(
                    f"  [PRE-FLIGHT] WARN: pen_tip is {dist_xy*100:.0f} cm from "
                    f"target XY — large approach move expected.\n"
                    f"  To minimise arm motion, place the container directly below\n"
                    f"  the current pen_tip position and update parameters:\n"
                    f"    -p target_x:={t.x:.3f} -p target_y:={t.y:.3f}\n"
                    f"  Or jog arm until pen_tip is over the container, then re-run\n"
                    f"  with those coordinates.")

        # 4. EE orientation check
        if ee_tf is not None:
            r = ee_tf.transform.rotation
            # Check if qw is near 0 — means arm is near a singularity or flipped
            if abs(r.w) < 0.05:
                self.get_logger().warn(
                    f"  [PRE-FLIGHT] WARN: EE quaternion w={r.w:.3f} is near 0.\n"
                    f"  The arm may be in a near-singular or flipped configuration.\n"
                    f"  Consider manually moving to a safer starting pose.")

        # 5. Services
        for svc_name, cli in [
            ("/plan_kinematic_path",   self._plan_cli),
            ("/compute_cartesian_path", self._cartesian_cli),
        ]:
            ready = cli.wait_for_service(timeout_sec=2.0)
            status = "OK" if ready else "FAIL — move_group not running?"
            level  = self.get_logger().info if ready else self.get_logger().error
            level(f"  [PRE-FLIGHT] {svc_name}: {status}")
            if not ready:
                ok = False

        self.get_logger().info("───────────────────────────────────────────────────")

        # Extra check: warn if approach_clearance is too low relative to current tip height.
        # If ready_z << cur_tip_z, Phase 0b PTP will descend in joint-space and may
        # clip the table. The fix is to increase approach_clearance.
        if tip_tf is not None:
            cur_tip_z = tip_tf.transform.translation.z
            container_top = p.get("table_z", -0.03) + p.get("container_h", 0.10)
            ready_z_val = container_top + p.get("approach", 0.16)
            correct_clearance = round(cur_tip_z - container_top, 2)
            gap = cur_tip_z - ready_z_val
            if abs(gap) > 0.05:
                self.get_logger().warn(
                    f"  [PRE-FLIGHT] WARN: ready_z={ready_z_val:.3f} m is "
                    f"{abs(gap)*100:.0f} cm {'below' if gap > 0 else 'above'} "
                    f"current tip z={cur_tip_z:.3f} m.\n"
                    f"  Phase 0b PTP will {'ascend' if gap > 0 else 'descend'} "
                    f"{abs(gap)*100:.0f} cm in joint-space — risk of table collision.\n"
                    f"  CORRECT approach_clearance for this home position: "
                    f"{correct_clearance:.2f} m\n"
                    f"  → Add: -p approach_clearance:={correct_clearance:.2f}")
            else:
                self.get_logger().info(
                    f"  [PRE-FLIGHT] approach_clearance OK: "
                    f"ready_z={ready_z_val:.3f} m ≈ tip z={cur_tip_z:.3f} m "
                    f"(gap={gap*100:.0f} cm)")
        return ok


    def _execute_traj(self, traj: RobotTrajectory, label: str,
                      timeout_sec: float = 30.0,
                      use_moveit_execute: bool = False) -> bool:
        """Execute a trajectory with safety clamping.

        use_moveit_execute=True  — MoveIt /execute_trajectory (OMPL joint-space).
        use_moveit_execute=False — Direct FJT (Cartesian, overrides 0.1 rad path tol).
        """
        clamp_traj_limits(traj)

        if use_moveit_execute:
            if not self._execute_cli.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(
                    f"  /execute_trajectory not available ({label}).")
                return False
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = traj
            f = self._execute_cli.send_goal_async(goal)
            gh = self._wait_for_future(f, timeout_sec=10.0)
            if gh is None or not gh.accepted:
                self.get_logger().error(f"  Execution goal rejected ({label}).")
                return False
            self._active_gh = gh
            rf = gh.get_result_async()
            result = self._wait_for_future(rf, timeout_sec=timeout_sec)
            self._active_gh = None
            if result is None:
                self.get_logger().error(
                    f"  Execution timed out after {timeout_sec:.0f}s ({label}).")
                return False
            ec = result.result.error_code.val
            if ec == 1:
                self.get_logger().info(f"  [PASS] {label} executed.")
                self._arm_moved = True
                return True
            self.get_logger().warn(f"  [WARN] {label} execution error code {ec}.")
            return False

        # Direct FJT path — bypasses MoveIt to override 0.1 rad path tolerance.
        deadline_srv = time.time() + 5.0
        while not self._fjt_cli.server_is_ready() and time.time() < deadline_srv:
            time.sleep(0.05)
        if not self._fjt_cli.server_is_ready():
            self.get_logger().error(
                f"  follow_joint_trajectory not available ({label}).")
            return False

        # Stamp header: Kortex driver rejects zero/stale stamps (error -5).
        now_ns   = self.get_clock().now().nanoseconds
        start_ns = now_ns + int(0.3 * 1e9)
        traj.joint_trajectory.header.stamp.sec     = start_ns // 1_000_000_000
        traj.joint_trajectory.header.stamp.nanosec = start_ns %  1_000_000_000

        fjt_goal = FollowJointTrajectory.Goal()
        fjt_goal.trajectory = traj.joint_trajectory
        for name in _GEN3_JOINTS:
            ptol = JointTolerance()
            ptol.name = name; ptol.position = 5.0
            ptol.velocity = 0.0; ptol.acceleration = 0.0
            fjt_goal.path_tolerance.append(ptol)
            gtol = JointTolerance()
            gtol.name = name; gtol.position = 0.05
            gtol.velocity = 0.0; gtol.acceleration = 0.0
            fjt_goal.goal_tolerance.append(gtol)
        fjt_goal.goal_time_tolerance = RosDuration(sec=30, nanosec=0)

        f = self._fjt_cli.send_goal_async(fjt_goal)
        gh = self._wait_for_future(f, timeout_sec=10.0)
        if gh is None or not gh.accepted:
            self.get_logger().error(f"  Execution goal rejected ({label}).")
            return False
        self._active_gh = gh
        rf = gh.get_result_async()
        result = self._wait_for_future(rf, timeout_sec=timeout_sec)
        self._active_gh = None
        if result is None:
            self.get_logger().error(
                f"  Execution timed out after {timeout_sec:.0f}s ({label}).")
            return False
        ec = result.result.error_code
        if ec == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(f"  [PASS] {label} executed.")
            self._arm_moved = True
            return True
        self.get_logger().warn(f"  [WARN] {label} execution error code {ec}.")
        return False

        return False

    def _cancel_active(self):
        from std_msgs.msg import Empty as EmptyMsg
        try:
            stop_pub = self.create_publisher(EmptyMsg, "/stop_kortex_cmd", 1)
            stop_pub.publish(EmptyMsg())
            self.get_logger().warn("  Published /stop_kortex_cmd (hardware stop).")
        except Exception as e:
            self.get_logger().warn(f"  Could not publish /stop_kortex_cmd: {e}")

        gh = self._active_gh
        if gh is not None:
            self.get_logger().warn("  Sending trajectory cancellation...")
            try:
                cancel_f = gh.cancel_goal_async()
                deadline = time.monotonic() + 3.0
                while not cancel_f.done() and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.05)
            except Exception as e:
                self.get_logger().warn(f"  Cancel request failed: {e}")
            self._active_gh = None

        self.get_logger().warn("  Waiting 2s for controller to settle...")
        time.sleep(2.0)

    def _recovery_return(self):
        if not self._arm_moved:
            self.get_logger().info("  Arm did not move -- no recovery needed.")
            return
        if not self._start_joints:
            self.get_logger().error(
                "  No start joints saved -- arm stays in current position.")
            return
        self.get_logger().info(
            "\n--- [Recovery] Returning arm to pre-script position ---")
        p = self._get_params()
        p["execute"] = True
        try:
            self._return_to_start(p, start_state=None)
        except Exception as e:
            self.get_logger().error(f"  Recovery return failed: {e}")

    @staticmethod
    def _wait_for_future(future, timeout_sec: float):
        """Poll a future without re-entering the executor.

        rclpy.spin_until_future_complete() called from inside a
        MultiThreadedExecutor callback causes concurrent wait-set access and
        crashes Thread-1 (rcl_action 'status subscription out of bounds').
        This pure-poll approach lets the background executor thread complete
        futures normally while we wait.
        """
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.05)
        return future.result()

    def _build_workspace(self, world_frame):
        ws = WorkspaceParameters()
        ws.header.frame_id = world_frame
        ws.min_corner.x = -1.1;  ws.min_corner.y = -1.1;  ws.min_corner.z = -0.1
        ws.max_corner.x =  1.1;  ws.max_corner.y =  1.1;  ws.max_corner.z =  1.2
        return ws

    def _apply_pole_collision(self, add: bool, p: dict) -> None:
        pole_x      = self.get_parameter("pole_x").value
        pole_y      = self.get_parameter("pole_y").value
        pole_radius = self.get_parameter("pole_radius").value
        pole_height = self.get_parameter("pole_height").value

        co = CollisionObject()
        co.header.frame_id = p["world_frame"]
        co.id        = "table_pole"
        co.operation = CollisionObject.ADD if add else CollisionObject.REMOVE

        if add:
            cyl = SolidPrimitive()
            cyl.type       = SolidPrimitive.CYLINDER
            cyl.dimensions = [pole_height, pole_radius]
            co.primitives.append(cyl)

            cyl_pose = Pose()
            cyl_pose.position.x   = pole_x
            cyl_pose.position.y   = pole_y
            cyl_pose.position.z   = pole_height / 2.0
            cyl_pose.orientation.w = 1.0
            co.primitive_poses.append(cyl_pose)

        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True

        cli = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene",
            callback_group=self._cb_group)
        if not cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                "  /apply_planning_scene unavailable — pole not added.")
            return
        req = ApplyPlanningScene.Request()
        req.scene = scene
        f = cli.call_async(req)
        self._wait_for_future(f, timeout_sec=5.0)
        verb = "added" if add else "removed"
        self.get_logger().info(f"  Pole collision object '{co.id}' {verb}.")

    def _joint_bounds_constraint(self, joint_delta_rad: float) -> Constraints:
        c = Constraints()
        for name in _GEN3_JOINTS:
            pos = self._start_joints.get(name)
            if pos is None:
                continue
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = joint_delta_rad
            jc.tolerance_below = joint_delta_rad
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        return c

    def _call_plan(self, motion_req: MotionPlanRequest):
        if not self._plan_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("  /plan_kinematic_path not available.")
            return False, None
        svc_req = GetMotionPlan.Request()
        svc_req.motion_plan_request = motion_req
        f = self._plan_cli.call_async(svc_req)
        svc_result = self._wait_for_future(f, timeout_sec=90.0)
        if svc_result is None:
            self.get_logger().error("  Planning timed out.")
            return False, None
        resp = svc_result.motion_plan_response
        ec = resp.error_code.val
        if ec == 1:
            return True, resp.trajectory
        self.get_logger().error(f"  Planning failed (error code {ec}).")
        return False, None

    def _get_tip_tf(self, p, timeout_sec=5.0):
        """Look up world→pen_tip and world→EE transforms with retry."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                tip_tf = self.tf_buffer.lookup_transform(
                    p["world_frame"], p["tip_link"], rclpy.time.Time())
                ee_tf = self.tf_buffer.lookup_transform(
                    p["world_frame"], p["ee_link"], rclpy.time.Time())
                return tip_tf, ee_tf
            except Exception:
                time.sleep(0.1)
        return None, None

    def _get_ee_to_tip_offset_world(self, p, ee_quat_xyzw,
                                    timeout_sec=3.0):
        """
        Compute the world-frame vector from EE_origin to pen_tip by subtracting
        the two world-frame TF positions directly.

        world_offset = pen_tip_world - EE_origin_world

        Then EE_origin = pen_tip_target + (-world_offset)
                       = pen_tip_target - (pen_tip_world - EE_origin_world)

        This approach does NOT require end_effector_link to exist as a live TF
        frame — it only needs world→pen_tip and world→EE, which the script
        already looks up in _get_tip_tf().  The offset is the current geometric
        displacement in world coordinates and is valid for any arm orientation.

        Returns (neg_wx, neg_wy, neg_wz) such that EE_origin = pen_tip + offset.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                tip_tf = self.tf_buffer.lookup_transform(
                    p["world_frame"], p["tip_link"], rclpy.time.Time())
                ee_tf  = self.tf_buffer.lookup_transform(
                    p["world_frame"], p["ee_link"],  rclpy.time.Time())

                tip_w = tip_tf.transform.translation
                ee_w  = ee_tf.transform.translation

                # Vector from pen_tip to EE_origin in world frame
                # EE_origin = pen_tip + (ee - tip)
                neg_wx = ee_w.x - tip_w.x
                neg_wy = ee_w.y - tip_w.y
                neg_wz = ee_w.z - tip_w.z

                dist = math.sqrt(neg_wx**2 + neg_wy**2 + neg_wz**2)
                self.get_logger().info(
                    f"  [TF offset] pen_tip world: ({tip_w.x:.4f}, {tip_w.y:.4f}, {tip_w.z:.4f})\n"
                    f"  [TF offset] EE_origin world: ({ee_w.x:.4f}, {ee_w.y:.4f}, {ee_w.z:.4f})\n"
                    f"  [TF offset] EE→tip offset (world): ({neg_wx:.4f}, {neg_wy:.4f}, {neg_wz:.4f})"
                    f"  magnitude={dist*100:.1f} cm\n"
                    f"  [TF offset] Expected from URDF: ~34.8 cm  (PDF confirmed: 348.48 mm)")
                return neg_wx, neg_wy, neg_wz

            except Exception as e:
                time.sleep(0.1)

        self.get_logger().warn(
            f"  [TF offset] Could not look up world→{p['tip_link']} or "
            f"world→{p['ee_link']} — falling back to tip_ee_offset parameters.\n"
            f"  Note: '{p['ee_link']}' may not exist in the live TF tree.\n"
            f"  Check available frames: ros2 run tf2_tools view_frames")
        return None, None, None



    # ------------------------------------------------------------------
    # Single-waypoint Cartesian step
    # ------------------------------------------------------------------
    def _cartesian_single_step(self, x, y, z, pen_q, p, start_state, label):
        """compute_cartesian_path to one waypoint. Returns (ok, end_state)."""
        wp = Pose()
        wp.position.x = x; wp.position.y = y; wp.position.z = z
        wp.orientation = pen_q

        if not self._cartesian_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                f"  compute_cartesian_path not available ({label}).")
            return False, None

        req = GetCartesianPath.Request()
        req.header.frame_id  = p["world_frame"]
        req.group_name       = p["group_name"]
        req.link_name        = p["tip_link"]
        req.waypoints        = [wp]
        req.max_step         = 0.005
        req.jump_threshold   = p["descend_jump"]
        req.avoid_collisions = True

        if start_state is not None:
            req.start_state = start_state
        else:
            req.start_state = RobotState()
            req.start_state.is_diff = True

        f = self._cartesian_cli.call_async(req)
        resp = self._wait_for_future(f, timeout_sec=20.0)

        if resp is None:
            self.get_logger().error(
                f"  compute_cartesian_path timed out ({label}).")
            return False, None

        fraction = resp.fraction
        self.get_logger().info(
            f"  [{label}] fraction={fraction:.3f}  "
            f"traj_pts={len(resp.solution.joint_trajectory.points)}")

        if fraction < 0.99:
            self.get_logger().error(
                f"  [{label}] path only {fraction*100:.1f}% reachable.")
            return False, None

        end_state = self._extract_end_state(resp.solution)
        if p["execute"]:
            self._rescale_cartesian_traj(resp.solution, p["vel_scale"])
            ok = self._execute_traj(resp.solution, label, timeout_sec=60.0)
            return ok, None
        return True, end_state

    # ------------------------------------------------------------------
    # OMPL request: move pen_tip to (cx, cy, safe_z) with loose EE orientation
    # ------------------------------------------------------------------
    def _build_ompl_approach_request(self, ee_x, ee_y, ee_z, pen_q, p,
                                     start_state=None):
        """
        OMPL goal: end_effector_link at (ee_x, ee_y, ee_z) within 2 cm,
        orientation within 0.3 rad of pen_q.

        Both constraints use end_effector_link — the IK tip MoveIt knows —
        so OMPL can use IK sampling to efficiently generate goal states.
        Using pen_tip for position + ee_link for orientation (different links)
        forces rejection sampling instead of IK sampling, causing error 99999.
        """
        req = MotionPlanRequest()
        req.group_name   = p["group_name"]
        req.planner_id   = "RRTConnectkConfigDefault"
        req.pipeline_id  = "ompl"
        req.num_planning_attempts = 20
        req.allowed_planning_time = 2.0
        req.max_velocity_scaling_factor     = p.get("transit_vel", p["vel_scale"])
        # Use 0.3× vel for accel — gentler ramp-in/out, natural-looking motion.
        req.max_acceleration_scaling_factor = p.get("transit_vel", p["vel_scale"]) * 0.3
        req.workspace_parameters = self._build_workspace(p["world_frame"])

        if start_state is not None:
            req.start_state = start_state
        else:
            req.start_state.is_diff = True

        # Position goal: EE origin inside a 2 cm sphere at (ee_x, ee_y, ee_z)
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.02]

        bv_pose = Pose()
        bv_pose.position.x  = ee_x
        bv_pose.position.y  = ee_y
        bv_pose.position.z  = ee_z
        bv_pose.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(bv_pose)

        pos_c = PositionConstraint()
        pos_c.header.frame_id   = p["world_frame"]
        pos_c.link_name         = p["ee_link"]
        pos_c.constraint_region = bv
        pos_c.weight            = 1.0

        # Orientation goal: EE within 0.3 rad of pen_q (tip roughly vertical)
        ori_c = OrientationConstraint()
        ori_c.header.frame_id = p["world_frame"]
        ori_c.link_name       = p["ee_link"]
        ori_c.orientation     = pen_q
        ori_c.absolute_x_axis_tolerance = 0.3
        ori_c.absolute_y_axis_tolerance = 0.3
        ori_c.absolute_z_axis_tolerance = 0.3
        ori_c.weight          = 1.0

        goal_c = Constraints()
        goal_c.position_constraints.append(pos_c)
        goal_c.orientation_constraints.append(ori_c)
        req.goal_constraints.append(goal_c)

        return req

    # ------------------------------------------------------------------
    # Pilz PTP request: joint-space move to an EE pose (deterministic IK)
    # ------------------------------------------------------------------
    def _build_pilz_ptp_request(self, ee_x, ee_y, ee_z, pen_q, p,
                                start_state=None):
        """
        Pilz PTP to a Cartesian EE goal.

        PTP is deterministic — it always picks the IK solution closest to the
        current joint configuration, eliminating the 90° wrist-flip problem
        that OMPL causes.  Orientation is respected exactly (within tilt_tol).

        Joint path constraints are added to keep the IK solver within
        joint_delta_rad of the current configuration — this prevents the
        planner from picking a distant IK solution that requires a large
        circular arc to reach.  Without this, PTP occasionally finds a valid
        but distant solution and the arm sweeps a wide arc to reach it.

        Use for: home move, approach navigate, return-to-start.
        """
        req = MotionPlanRequest()
        req.group_name   = p["group_name"]
        req.planner_id   = "PTP"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1          # Pilz is deterministic
        req.allowed_planning_time = 10.0
        # Transit phases (approach/return) run at transition_velocity_scaling
        # for speed; the LIN builders (descend/ascend) use vel_scale for safety.
        req.max_velocity_scaling_factor     = p.get("transit_vel", p["vel_scale"])
        # 0.3× vel → gentler S-curve ramp, eliminates abrupt start/stop jerk
        req.max_acceleration_scaling_factor = p.get("transit_vel", p["vel_scale"]) * 0.3
        if start_state is not None:
            req.start_state = start_state
        else:
            req.start_state.is_diff = True

        # Joint path constraints — keep IK within joint_delta_rad of current pose.
        # This prevents the IK solver from picking a distant solution that requires
        # a 150°+ swing of joint_1 OR a ±180° mirror flip of joint_3.
        # joint_3 sits near -180° at home; the IK solver can reach the equivalent
        # +180° solution via a full forearm spin.  A tighter tolerance (1.0 rad)
        # is applied to joint_1 and joint_3 specifically to block these spins.
        if self._start_joints:
            path_c = Constraints()
            delta = p.get("joint_delta", 1.2)
            # Joints that need tighter spin-prevention tolerances.
            _TIGHT_JOINTS = {"joint_1": 1.0, "joint_3": 1.0}
            for name in _GEN3_JOINTS:
                cur = self._start_joints.get(name)
                if cur is None:
                    continue
                jc = JointConstraint()
                jc.joint_name      = name
                jc.position        = cur
                tol = _TIGHT_JOINTS.get(name, delta)
                jc.tolerance_above = tol
                jc.tolerance_below = tol
                jc.weight          = 1.0
                path_c.joint_constraints.append(jc)
            req.path_constraints = path_c

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.005]   # 5 mm — Pilz needs a near-point goal

        bv_pose = Pose()
        bv_pose.position.x = ee_x; bv_pose.position.y = ee_y
        bv_pose.position.z = ee_z
        bv_pose.orientation.x = pen_q.x; bv_pose.orientation.y = pen_q.y
        bv_pose.orientation.z = pen_q.z; bv_pose.orientation.w = pen_q.w
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(bv_pose)

        pos_c = PositionConstraint()
        pos_c.header.frame_id   = p["world_frame"]
        pos_c.link_name         = p["ee_link"]
        pos_c.constraint_region = bv
        pos_c.weight            = 1.0

        ori_c = OrientationConstraint()
        ori_c.header.frame_id           = p["world_frame"]
        ori_c.link_name                 = p["ee_link"]
        ori_c.orientation               = pen_q
        ori_c.absolute_x_axis_tolerance = p["tilt_tol"]
        ori_c.absolute_y_axis_tolerance = p["tilt_tol"]
        ori_c.absolute_z_axis_tolerance = p["tilt_tol"]
        ori_c.weight                    = 1.0

        goal_c = Constraints()
        goal_c.position_constraints.append(pos_c)
        goal_c.orientation_constraints.append(ori_c)
        req.goal_constraints.append(goal_c)
        return req

    # ------------------------------------------------------------------
    # Pilz LIN request: straight Cartesian line with locked orientation
    # ------------------------------------------------------------------
    def _build_pilz_lin_request(self, ee_x, ee_y, ee_z, pen_q, p,
                                start_state=None):
        """
        Pilz LIN: straight-line TCP motion with orientation held constant.

        This is equivalent to Xbox controller twist-linear mode — the TCP
        moves in a straight line and the wrist does not rotate.  Zero IK
        ambiguity.  Use for descend, ascend, and Z-rise phases.

        Pilz LIN requires the goal as a position+orientation constraint
        on end_effector_link with tight (5 mm) position tolerance.
        """
        req = MotionPlanRequest()
        req.group_name   = p["group_name"]
        req.planner_id   = "LIN"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor     = p["vel_scale"]
        # LIN phases (descend/ascend) use lower accel for smooth insertion/retraction.
        req.max_acceleration_scaling_factor = p["vel_scale"] * 0.3
        if start_state is not None:
            req.start_state = start_state
        else:
            req.start_state.is_diff = True

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.005]

        bv_pose = Pose()
        bv_pose.position.x = ee_x; bv_pose.position.y = ee_y
        bv_pose.position.z = ee_z
        bv_pose.orientation.x = pen_q.x; bv_pose.orientation.y = pen_q.y
        bv_pose.orientation.z = pen_q.z; bv_pose.orientation.w = pen_q.w
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(bv_pose)

        pos_c = PositionConstraint()
        pos_c.header.frame_id   = p["world_frame"]
        pos_c.link_name         = p["ee_link"]
        pos_c.constraint_region = bv
        pos_c.weight            = 1.0

        ori_c = OrientationConstraint()
        ori_c.header.frame_id           = p["world_frame"]
        ori_c.link_name                 = p["ee_link"]
        ori_c.orientation               = pen_q
        ori_c.absolute_x_axis_tolerance = 0.01   # LIN locks orientation tightly
        ori_c.absolute_y_axis_tolerance = 0.01
        ori_c.absolute_z_axis_tolerance = 0.01
        ori_c.weight                    = 1.0

        goal_c = Constraints()
        goal_c.position_constraints.append(pos_c)
        goal_c.orientation_constraints.append(ori_c)
        req.goal_constraints.append(goal_c)
        return req

    def _pilz_single_step(self, ee_x, ee_y, ee_z, pen_q, p,
                          start_state, label, motion_type="LIN",
                          fallback_cartesian=True):
        """
        Plan and optionally execute a single Pilz LIN or PTP step.

        Falls back to compute_cartesian_path if Pilz fails (LIN only).
        Returns (ok, end_state).
        """
        if motion_type == "LIN":
            req = self._build_pilz_lin_request(ee_x, ee_y, ee_z, pen_q, p,
                                               start_state)
        else:
            req = self._build_pilz_ptp_request(ee_x, ee_y, ee_z, pen_q, p,
                                               start_state)

        ok, traj = self._call_plan(req)

        if not ok and motion_type == "LIN" and fallback_cartesian:
            self.get_logger().warn(
                f"  [{label}] Pilz LIN failed — falling back to compute_cartesian_path.")
            # Build a single-waypoint Cartesian request as fallback
            wp = Pose()
            wp.position.x = ee_x; wp.position.y = ee_y; wp.position.z = ee_z
            wp.orientation = pen_q
            if not self._cartesian_cli.wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f"  compute_cartesian_path not available ({label}).")
                return False, None
            creq = GetCartesianPath.Request()
            creq.header.frame_id  = p["world_frame"]
            creq.group_name       = p["group_name"]
            creq.link_name        = p["tip_link"]
            creq.waypoints        = [wp]
            creq.max_step         = 0.005
            creq.jump_threshold   = p["descend_jump"]
            creq.avoid_collisions = True
            if start_state is not None:
                creq.start_state = start_state
            else:
                creq.start_state = RobotState()
                creq.start_state.is_diff = True
            f = self._cartesian_cli.call_async(creq)
            resp = self._wait_for_future(f, timeout_sec=20.0)
            if resp is None or resp.fraction < 0.99:
                frac = resp.fraction if resp else 0.0
                self.get_logger().error(
                    f"  [{label}] Cartesian fallback also failed (fraction={frac:.2f}).")
                return False, None
            traj = resp.solution
            self._rescale_cartesian_traj(traj, p["vel_scale"])
            ok = True

        if not ok:
            self.get_logger().error(f"  [{label}] Pilz {motion_type} planning failed.")
            return False, None

        real = self.get_parameter("real_robot").value
        if not validate_trajectory(traj, label, self.get_logger(), self._display_traj_pub, real):
            return False, None

        end_state = self._extract_end_state(traj)
        if p["execute"]:
            exec_ok = self._execute_traj(traj, label, timeout_sec=120.0,
                                         use_moveit_execute=True)
            return exec_ok, None
        return True, end_state


    def _approach(self, cx, cy, ready_z, cur_x, cur_y, cur_z, p, live_q, pen_q,
                  ee_world_offset=None):
        """
        Phase 0a  Z rise   — Pilz LIN: lift straight up, orientation locked.
        Phase 0b  Navigate — Pilz PTP: deterministic IK, no wrist-flip.

        ee_world_offset: (wx, wy, wz) in world frame such that EE_origin = pen_tip + offset.
                         Computed from TF lookup (preferred) or tip_ee_offset params.
        """
        # safe_z: tip height BEFORE the XY navigate move.
        # When cur_z ≈ ready_z (arm already at approach height), skip the rise.
        # When cur_z > ready_z significantly, cap safe_z at cur_z + a small buffer
        # rather than adding lateral_rise — Pilz LIN fails on large vertical rises
        # because the IK can't find a solution far from the current configuration.
        # The PTP in Phase 0b handles any remaining height adjustment.
        if abs(cur_z - ready_z) <= 0.05:
            # Already at approach height — no rise needed, go straight to PTP
            safe_z = cur_z
        elif cur_z > ready_z:
            # Arm is above approach height — no rise, PTP will descend safely
            safe_z = cur_z
        else:
            # Arm is below approach height — rise to ready_z + 5cm buffer
            # Cap rise at 10cm max for Pilz LIN reliability
            rise_needed = ready_z - cur_z + 0.05
            safe_z = cur_z + min(rise_needed, 0.10)
        end_state = None

        need_rise = safe_z > cur_z + 0.02

        # Compute EE target at ready_z
        if ee_world_offset is not None:
            wx, wy, wz = ee_world_offset
            ee_x = cx + wx
            ee_y = cy + wy
            ee_z = ready_z + wz
        else:
            ee_x, ee_y, ee_z = _tip_to_ee_pose(
                cx, cy, ready_z,
                (pen_q.x, pen_q.y, pen_q.z, pen_q.w),
                (p["tip_ee_ox"], p["tip_ee_oy"], p["tip_ee_oz"]))

        # Sanity check: EE target must be within arm reach
        ee_dist = math.sqrt(ee_x**2 + ee_y**2 + ee_z**2)
        tip_to_ee_dist = math.sqrt((ee_x-cx)**2 + (ee_y-cy)**2 + (ee_z-ready_z)**2)
        if ee_dist > 0.95:
            self.get_logger().error(
                f"  [APPROACH] EE target ({ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f}) "
                f"is {ee_dist:.3f} m from base — OUTSIDE arm reach (max 0.902 m).\n"
                f"  The tip_ee_offset is wrong for the current arm orientation.\n"
                f"  Run: ros2 run tf2_ros tf2_echo end_effector_link pen_tip\n"
                f"  and set use_tf_offset:=true or update tip_ee_offset_x/y/z.")
            return False, None
        if tip_to_ee_dist > 0.50:
            self.get_logger().warn(
                f"  [APPROACH] WARN: EE is {tip_to_ee_dist*100:.0f} cm from pen_tip target "
                f"(expected ~{math.sqrt(p['tip_ee_ox']**2+p['tip_ee_oy']**2+p['tip_ee_oz']**2)*100:.0f} cm).\n"
                f"  Check tip_ee_offset vs URDF pen_tip joint xyz.")

        self.get_logger().info(
            f"\n--- [Phase 0] Approach (Pilz PTP+LIN) ---\n"
            f"  current tip: ({cur_x:.3f}, {cur_y:.3f}, {cur_z:.3f})\n"
            f"  safe_z={safe_z:.3f} "
            f"({'rise ' + f'{(safe_z-cur_z)*100:.0f} cm' if need_rise else 'already clear'})"
            f"  ready_z={ready_z:.3f}\n"
            f"  pen_tip target: ({cx:.3f}, {cy:.3f}, {ready_z:.3f})\n"
            f"  EE   target:    ({ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f})"
            f"  dist_from_base={ee_dist:.3f} m  tip_to_ee={tip_to_ee_dist*100:.1f} cm")

        if need_rise:
            if ee_world_offset is not None:
                er_x, er_y, er_z = cur_x+wx, cur_y+wy, safe_z+wz
            else:
                er_x, er_y, er_z = _tip_to_ee_pose(
                    cur_x, cur_y, safe_z,
                    (live_q.x, live_q.y, live_q.z, live_q.w),
                    (p["tip_ee_ox"], p["tip_ee_oy"], p["tip_ee_oz"]))
            self.get_logger().info(
                f"  Phase 0a: Pilz LIN rise  z={cur_z:.3f} → {safe_z:.3f}")
            ok, end_state = self._pilz_single_step(
                er_x, er_y, er_z, live_q, p,
                end_state, "0a_z_rise", motion_type="LIN")
            if not ok:
                return False, None
        else:
            self.get_logger().info(
                f"  Phase 0a: skipped (already above safe_z={safe_z:.3f})")

        self.get_logger().info(
            f"  Phase 0b: Pilz PTP navigate → ({cx:.3f}, {cy:.3f}, {ready_z:.3f})")
        ok, ptp_end = self._pilz_single_step(
            ee_x, ee_y, ee_z, pen_q, p,
            end_state, "0b_navigate_ptp", motion_type="PTP",
            fallback_cartesian=False)

        if ok:
            end_state = ptp_end
        else:
            # Parallel OMPL fallback — run N attempts, pick lowest displacement.
            self.get_logger().warn(
                "  Phase 0b: Pilz PTP failed — falling back to parallel OMPL.\n"
                f"  Running {p.get('parallel_plans', 5)} attempts simultaneously "
                "and selecting the most direct path.")

            def _build_ompl():
                return self._build_ompl_approach_request(
                    ee_x, ee_y, ee_z, pen_q, p, start_state=end_state)

            ok, traj = self._plan_parallel(_build_ompl, p, label="0b_ompl")

            if not ok:
                self.get_logger().error(
                    "  Phase 0b failed (Pilz PTP and parallel OMPL both failed).\n"
                    "  Verify: ros2 run tf2_ros tf2_echo world assembly_tip")
                return False, None
            end_state = self._extract_end_state(traj)
            if p["execute"]:
                real = self.get_parameter("real_robot").value
                if not validate_trajectory(traj, "0b_navigate_ompl", self.get_logger(), self._display_traj_pub, real):
                    return False, None
                ok = self._execute_traj(traj, "0b_navigate_ompl",
                                        timeout_sec=120.0, use_moveit_execute=True)
                if not ok:
                    return False, None
                end_state = None
            else:
                self.get_logger().info("  [PASS] Phase 0b parallel OMPL planned.")

        self.get_logger().info("  [PASS] Approach complete.")
        return True, end_state

    def _descend(self, cx, cy, ready_z, target_z, p, pen_q,
                 start_state=None, ee_world_offset=None):
        """Phase 1: Pilz LIN straight down — orientation locked, no IK ambiguity."""
        self.get_logger().info(
            f"\n--- [Phase 1] Descend (Pilz LIN) -- "
            f"z={ready_z:.3f} → z={target_z:.3f} m ---")
        if ee_world_offset is not None:
            wx, wy, wz = ee_world_offset
            ee_x, ee_y, ee_z = cx+wx, cy+wy, target_z+wz
        else:
            ee_x, ee_y, ee_z = _tip_to_ee_pose(
                cx, cy, target_z,
                (pen_q.x, pen_q.y, pen_q.z, pen_q.w),
                (p["tip_ee_ox"], p["tip_ee_oy"], p["tip_ee_oz"]))
        ok, end_state = self._pilz_single_step(
            ee_x, ee_y, ee_z, pen_q, p, start_state, "Descend", motion_type="LIN")
        if not ok:
            return False, None
        self.get_logger().info("  [PASS] Descend complete.")
        return True, end_state

    # ------------------------------------------------------------------
    # Phase 2 -- Hold at insert position
    # ------------------------------------------------------------------
    def _hold_at_target(self, cx, cy, target_z, p):
        self.get_logger().info(
            f"\n--- [Phase 2] Hold at insert position -- "
            f"x={cx:.3f}  y={cy:.3f}  z={target_z:.3f} ---")

        if p["execute"]:
            self.get_logger().info(
                f"  Holding for {p['post_wait']:.1f} s...")
            time.sleep(p["post_wait"])
            self.get_logger().info("  [PASS] Hold complete.")
        else:
            self.get_logger().info(
                "  Dry-run -- hold skipped.  "
                "Add -p execute_motion:=true to move.")

        return True, None

    # ------------------------------------------------------------------
    # Phase 2.5 -- Ascend (Cartesian, mirror of descend)
    # ------------------------------------------------------------------

    def _ascend(self, cx, cy, target_z, ready_z, p, pen_q,
                start_state=None, ee_world_offset=None):
        """Phase 2.5: Pilz LIN straight up — orientation locked, mirror of descend."""
        self.get_logger().info(
            f"\n--- [Phase 2.5] Ascend (Pilz LIN) -- "
            f"z={target_z:.3f} → z={ready_z:.3f} m ---")
        if ee_world_offset is not None:
            wx, wy, wz = ee_world_offset
            ee_x, ee_y, ee_z = cx+wx, cy+wy, ready_z+wz
        else:
            ee_x, ee_y, ee_z = _tip_to_ee_pose(
                cx, cy, ready_z,
                (pen_q.x, pen_q.y, pen_q.z, pen_q.w),
                (p["tip_ee_ox"], p["tip_ee_oy"], p["tip_ee_oz"]))
        ok, end_state = self._pilz_single_step(
            ee_x, ee_y, ee_z, pen_q, p, start_state, "Ascend", motion_type="LIN")
        if not ok:
            self.get_logger().warn(
                "  Ascend failed — proceeding to return from current height.")
            return True, None
        self.get_logger().info("  [PASS] Ascend complete.")
        return True, end_state

    def _return_to_start(self, p, start_state=None):
        self.get_logger().info("\n--- [Phase 3] Return to start joints ---")

        goal_joints = {n: self._start_joints.get(n) for n in _GEN3_JOINTS}
        missing = [n for n, v in goal_joints.items() if v is None]
        if missing:
            self.get_logger().warn(
                f"  Missing joint states for {missing} -- skipping return.")
            return

        current_joints = {}
        if start_state is not None and start_state.joint_state.name:
            for name, pos in zip(start_state.joint_state.name,
                                 start_state.joint_state.position):
                if name in _GEN3_JOINTS:
                    current_joints[name] = pos
        else:
            current_joints = dict(self._start_joints)

        def _build_path_c(delta_rad):
            c = Constraints()
            for name in _GEN3_JOINTS:
                cur  = current_joints.get(name, goal_joints.get(name, 0.0))
                goal = goal_joints.get(name, cur)
                mid  = (cur + goal) / 2.0
                half_range = abs(cur - goal) / 2.0 + delta_rad
                jc = JointConstraint()
                jc.joint_name      = name
                jc.position        = mid
                jc.tolerance_above = half_range
                jc.tolerance_below = half_range
                jc.weight          = 1.0
                c.joint_constraints.append(jc)
            return c

        def _build_ompl_return_req(attempts, plan_time, path_constraints=None):
            req = MotionPlanRequest()
            req.group_name   = p["group_name"]
            req.planner_id   = "RRTConnectkConfigDefault"
            req.pipeline_id  = "ompl"
            req.num_planning_attempts = attempts
            req.allowed_planning_time = plan_time
            # Return-to-start uses transition speed (50%) — safe because it is a
            # free-space joint move with no constraints near the container.
            req.max_velocity_scaling_factor     = p.get("transit_vel", p["vel_scale"])
            # Lower accel scaling → smooth ramp-out after the insertion is complete.
            req.max_acceleration_scaling_factor = p.get("transit_vel", p["vel_scale"]) * 0.3
            req.workspace_parameters = self._build_workspace(p["world_frame"])
            if path_constraints is not None:
                req.path_constraints = path_constraints
            if start_state is not None:
                req.start_state = start_state
            else:
                req.start_state.is_diff = True
            goal_c = Constraints()
            for name, position in goal_joints.items():
                jc = JointConstraint()
                jc.joint_name      = name
                jc.position        = position
                jc.tolerance_above = 0.01
                jc.tolerance_below = 0.01
                jc.weight          = 1.0
                goal_c.joint_constraints.append(jc)
            req.goal_constraints.append(goal_c)
            return req

        def _build_pilz_ptp_return_req():
            """Pilz PTP to return to start joints — deterministic, minimal motion."""
            req = MotionPlanRequest()
            req.group_name   = p["group_name"]
            req.planner_id   = "PTP"
            req.pipeline_id  = "pilz_industrial_motion_planner"
            req.num_planning_attempts = 1
            req.allowed_planning_time = 10.0
            # Return-to-start uses transition speed (50%) — safe free-space PTP.
            req.max_velocity_scaling_factor     = p.get("transit_vel", p["vel_scale"])
            # Gentle accel for smooth return to home.
            req.max_acceleration_scaling_factor = p.get("transit_vel", p["vel_scale"]) * 0.3
            if start_state is not None:
                req.start_state = start_state
            else:
                req.start_state.is_diff = True
            goal_c = Constraints()
            for name, position in goal_joints.items():
                jc = JointConstraint()
                jc.joint_name      = name
                jc.position        = position
                jc.tolerance_above = 0.05
                jc.tolerance_below = 0.05
                jc.weight          = 1.0
                goal_c.joint_constraints.append(jc)
            req.goal_constraints.append(goal_c)
            return req

        # Try Pilz PTP first — deterministic, minimal motion, no random IK flips.
        self.get_logger().info("  Planning return (Pilz PTP — deterministic)...")
        ok, traj = self._call_plan(_build_pilz_ptp_return_req())
        if ok:
            self.get_logger().info("  [PASS] Return planned via Pilz PTP.")
        else:
            self.get_logger().warn(
                "  Pilz PTP return failed — falling back to OMPL.")
            self.get_logger().info(
                "  Planning return (OMPL midpoint-centred constraint, +/-1.5 rad)...")
            ok, traj = self._call_plan(
                _build_ompl_return_req(20, 3.0, _build_path_c(1.5)))

        if ok:
            self.get_logger().info("  [PASS] Return planned (constrained, 1.5 rad).")
        else:
            self.get_logger().warn("  Attempt 1 failed -- loosening to +/-2.5 rad.")
            ok, traj = self._call_plan(
                _build_ompl_return_req(20, 3.0, _build_path_c(2.5)))
            if ok:
                self.get_logger().info(
                    "  [PASS] Return planned (constrained, 2.5 rad).")
            else:
                self.get_logger().warn(
                    "  Attempt 2 failed -- planning without path constraint.")
                ok, traj = self._call_plan(_build_ompl_return_req(20, 3.0, None))
                if not ok:
                    self.get_logger().warn(
                        "  Return planning failed -- arm stays at current position.")
                    return
                self.get_logger().info("  [PASS] Return planned (unconstrained OMPL).")

        if p["execute"]:
            real = self.get_parameter("real_robot").value
            if validate_trajectory(traj, "Return", self.get_logger(), self._display_traj_pub, real):
                self._execute_traj(traj, "Return", timeout_sec=120.0,
                                   use_moveit_execute=True)

    # ------------------------------------------------------------------
    # Pre-phase: move to MoveIt Home (real robot gate)
    # ------------------------------------------------------------------
    def _normalize_home_joints(self) -> dict:
        """
        Normalize _GEN3_HOME_JOINTS so each target is within π rad of the
        current joint position.

        The canonical issue: joint_3 in _GEN3_HOME_JOINTS is -π (-180°), but
        the arm reports +π (+180°) from /joint_states — both are the same
        physical pose.  Normalizing only joint_3 removes the ambiguity in how
        that pose is expressed. Other joints are left absolute to prevent shifting
        to unreachable/invalid 2π wraps.
        """
        normalized = {}
        for name, target in _GEN3_HOME_JOINTS.items():
            cur = self._start_joints.get(name)
            if cur is None:
                normalized[name] = target
                continue
            
            if name == "joint_3":
                # Only normalize joint_3 because it's at the -π boundary (-180°)
                diff = (target - cur + math.pi) % (2 * math.pi) - math.pi
                best = cur + diff
                if abs(best - target) > 1e-4:
                    self.get_logger().info(
                        f"  [home-norm] {name}: {math.degrees(target):.1f}° → "
                        f"{math.degrees(best):.1f}° (normalized to current "
                        f"{math.degrees(cur):.1f}°)")
                normalized[name] = best
            else:
                normalized[name] = target
        return normalized

    def _move_to_home(self, p: dict) -> bool:
        """
        Move the arm to _GEN3_HOME_JOINTS using the best available path.

        Strategy (three attempts in order):
          1. Pilz PTP — deterministic, fastest to plan.
             Accepted if the result passes _is_suspicious_trajectory.
          2. Parallel OMPL — if Pilz PTP is suspicious (large joint_1 swing,
             mid-path jump, etc.), run p['parallel_plans'] OMPL attempts
             simultaneously and pick the lowest-scoring valid result.
          3. Unconstrained OMPL fallback — if all parallel attempts fail or
             are filtered, plan once without path constraints as last resort.

        This mirrors the parallel planning used in Phase 0b so the home move
        benefits from the same path quality guarantees when starting from an
        unknown arm position.
        """
        self.get_logger().info("\n--- [Pre-phase] Move to Home position ---")
        
        # Ensure joint states are populated before doing any normalization or planning
        self.get_logger().info("  Waiting for joint states to be populated...")
        start_wait = time.time()
        timeout = 5.0
        while len(self._start_joints) < len(_GEN3_JOINTS) and (time.time() - start_wait) < timeout:
            time.sleep(0.1)
            
        if len(self._start_joints) < len(_GEN3_JOINTS):
            self.get_logger().error(
                f"  Failed to populate joint states within {timeout}s!\n"
                f"  Current joints in cache: {list(self._start_joints.keys())}")
            return False

        home_vel = max(p["vel_scale"], 0.15)

        # Normalise home joint targets relative to the current arm position.
        norm_home = self._normalize_home_joints()
        self.get_logger().info(
            f"  Normalised home targets (°): "
            + ", ".join(f"{n}={math.degrees(v):.1f}" for n, v in norm_home.items()))

        def _build_joint_goal_req(use_ompl: bool = False,
                                  n_attempts: int = 1) -> MotionPlanRequest:
            req = MotionPlanRequest()
            req.group_name = p["group_name"]
            if use_ompl:
                req.planner_id   = "RRTConnectkConfigDefault"
                req.pipeline_id  = "ompl"
                req.num_planning_attempts = n_attempts
                req.allowed_planning_time = 15.0
                # No path constraints for the home move.
                #
                # Path constraints centred at the current configuration (±1.2 rad)
                # prevent OMPL from reaching home when the arm starts far away.
                # Example: current joint_6 = +55° (0.96 rad), home = -102.7°
                # (-1.79 rad) → 2.75 rad apart, well outside the ±1.2 rad window.
                # OMPL would "succeed" but settle at the closest reachable point
                # (e.g. joint_6 ≈ 64°, joint_7 ≈ 74°) rather than true home.
                #
                # The home move is purely joint-space so OMPL doesn't need
                # path constraints to avoid Cartesian-space hazards.
                # _is_suspicious_trajectory still filters out 360° spins / arcs.
            else:
                req.planner_id   = "PTP"
                req.pipeline_id  = "pilz_industrial_motion_planner"
                req.num_planning_attempts = 1
                req.allowed_planning_time = 10.0
            req.max_velocity_scaling_factor     = home_vel
            # Gentle acceleration keeps home move smooth even at 15%+ velocity.
            req.max_acceleration_scaling_factor = home_vel * 0.3
            req.workspace_parameters = self._build_workspace(p["world_frame"])
            req.start_state.is_diff = True
            goal_c = Constraints()
            for name, position in norm_home.items():
                jc = JointConstraint()
                jc.joint_name      = name
                jc.position        = position
                jc.tolerance_above = 0.05
                jc.tolerance_below = 0.05
                jc.weight          = 1.0
                goal_c.joint_constraints.append(jc)
            req.goal_constraints.append(goal_c)
            return req

        # ── Attempt 1: Pilz PTP ───────────────────────────────────────────
        self.get_logger().info("  Planning home move via Pilz PTP...")
        ok, traj = self._call_plan(_build_joint_goal_req(use_ompl=False))
        chosen_method = "Pilz PTP"

        if ok and traj:
            suspicious, reason = self._is_suspicious_trajectory(traj, p, 0.0)
            if suspicious:
                self.get_logger().warn(
                    f"  Pilz PTP home path rejected — {reason}.\n"
                    f"  Falling back to parallel OMPL for home move.")
                ok = False   # force fallback

        # ── Attempt 2: Parallel OMPL ──────────────────────────────────────
        if not ok:
            n = int(p.get("parallel_plans", 5))
            self.get_logger().info(
                f"  Planning home move via parallel OMPL "
                f"({n} attempts)...")
            ok, traj = self._plan_parallel(
                lambda: _build_joint_goal_req(use_ompl=True, n_attempts=1),
                p, label="home_ompl")
            chosen_method = f"parallel OMPL ({n} attempts)"

        # ── Attempt 3: Unconstrained OMPL fallback ────────────────────────
        if not ok:
            self.get_logger().warn(
                "  Parallel OMPL home also failed — "
                "trying unconstrained OMPL as last resort.")
            req = _build_joint_goal_req(use_ompl=True, n_attempts=20)
            req.path_constraints = Constraints()   # clear constraints
            req.allowed_planning_time = 30.0
            ok, traj = self._call_plan(req)
            chosen_method = "unconstrained OMPL (last resort)"

        if not ok or traj is None:
            self.get_logger().error("  Home move planning failed on all attempts.")
            return False

        # ── Validate and report ───────────────────────────────────────────
        real = self.get_parameter("real_robot").value
        if not validate_trajectory(traj, "move_to_home", self.get_logger(), self._display_traj_pub, real):
            return False

        pts = traj.joint_trajectory.points
        dur = (pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9
               ) if pts else 0.0
        _, stats = self._score_trajectory(
            traj, p.get("j1_weight", 0.4), p.get("duration_weight", 0.2))
        self.get_logger().info(
            f"  [PASS] Home move planned via {chosen_method}\n"
            f"    duration  = {dur:.1f} s at {home_vel*100:.0f}% velocity\n"
            f"    joint_1   = {math.degrees(stats.get('j1_disp', 0)):.1f}°\n"
            f"    total disp= {stats.get('total_disp', 0):.3f} rad")

        if not p["execute"]:
            return True

        # ── Execute via direct FJT (loose path tolerance, avoids -4 CONTROL_FAILED) ──
        # MoveIt's /execute_trajectory enforces the 0.1 rad path tolerance from
        # ros2_controllers.yaml.  If the arm falls even slightly behind the commanded
        # trajectory (common near joint limits or at velocity peaks), the controller
        # trips this tolerance and aborts with error -4 — the "moves halfway then
        # stops" symptom.  The direct FJT path sets path_tolerance = 5.0 rad
        # (effectively disabled) so the arm tracks to completion.
        clamp_traj_limits(traj)
        ok = self._execute_traj(traj, "move_to_home",
                                timeout_sec=dur + 60.0,
                                use_moveit_execute=False)
        if ok:
            self.get_logger().info("  [PASS] move_to_home executed.")
            self._arm_moved = True
            
            # Settle and verify that the physical joints actually reached the home position
            self.get_logger().info("  Verifying physical joint positions at home...")
            time.sleep(1.0)
            current_joints = dict(self._start_joints)
            reached_home = True
            tolerance = 0.05 # rad (~2.8 degrees)
            
            for name, target in _GEN3_HOME_JOINTS.items():
                cur = current_joints.get(name)
                if cur is None:
                    self.get_logger().error(f"  [VERIFY] Missing joint state for {name} after home move.")
                    reached_home = False
                    break
                    
                if name == "joint_3":
                    # For joint_3, check target and target + 2π equivalents
                    diff = (target - cur + math.pi) % (2 * math.pi) - math.pi
                    err = abs(diff)
                else:
                    err = abs(target - cur)
                    
                if err > tolerance:
                    self.get_logger().error(
                        f"  [VERIFY] Joint {name} failed to reach home!\n"
                        f"    Target: {math.degrees(target):.2f}°, Current: {math.degrees(cur):.2f}° "
                        f"(Err: {math.degrees(err):.2f}° > {math.degrees(tolerance):.1f}°)")
                    reached_home = False
                    
            if not reached_home:
                self.get_logger().error("  [VERIFY] Robot did not reach calibrated home position. Aborting.")
                return False
                
            self.get_logger().info("  [VERIFY] Arm successfully verified at calibrated home position.")
            
        return ok

    # ------------------------------------------------------------------
    # Insert target preview marker (RViz)
    # ------------------------------------------------------------------
    def _publish_target_marker(self, cx, cy, target_z, ready_z,
                               container_h, table_z, world_frame):
        """
        Publish a MarkerArray to /insert_preview for RViz verification.

        id=0  CYLINDER  — approximate container outline (semi-transparent)
        id=1  SPHERE    — insert target point (yellow)
        id=2  ARROW     — approach: from ready_z down to target_z (green)
        id=3  TEXT      — label with key numbers
        """
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        def _base(mid, mtype, r, g, b, a=1.0):
            m = Marker()
            m.header.frame_id = world_frame
            m.header.stamp    = now
            m.ns              = "insert_preview"
            m.id              = mid
            m.type            = mtype
            m.action          = Marker.ADD
            m.color           = ColorRGBA(r=r, g=g, b=b, a=a)
            m.pose.orientation.w = 1.0
            m.lifetime.sec    = 0
            m.lifetime.nanosec = 0
            return m

        # id 0: container outline (cylinder)
        cont = _base(0, Marker.CYLINDER, 0.2, 0.6, 1.0, 0.25)
        cont.pose.position.x = cx
        cont.pose.position.y = cy
        cont.pose.position.z = table_z + container_h / 2.0
        cont.scale.x = 0.08   # approximate 8 cm diameter
        cont.scale.y = 0.08
        cont.scale.z = container_h
        markers.markers.append(cont)

        # id 1: insert target sphere
        target = _base(1, Marker.SPHERE, 1.0, 1.0, 0.0)  # yellow
        target.pose.position.x = cx
        target.pose.position.y = cy
        target.pose.position.z = target_z
        target.scale.x = 0.015
        target.scale.y = 0.015
        target.scale.z = 0.015
        markers.markers.append(target)

        # id 2: approach arrow (from ready_z down to target_z)
        arrow = _base(2, Marker.ARROW, 0.0, 1.0, 0.2)   # green
        arrow.scale.x = 0.008
        arrow.scale.y = 0.016
        arrow.scale.z = 0.020
        arrow.points = [
            Point(x=cx, y=cy, z=ready_z),
            Point(x=cx, y=cy, z=target_z),
        ]
        markers.markers.append(arrow)

        # id 3: text label
        label = _base(3, Marker.TEXT_VIEW_FACING, 1.0, 1.0, 1.0)
        label.pose.position.x = cx
        label.pose.position.y = cy
        label.pose.position.z = table_z + container_h + 0.06
        label.scale.z = 0.035
        label.text = (
            f"insert target ({cx:.3f}, {cy:.3f}, {target_z:.3f})\n"
            f"container_height={container_h*100:.0f} cm\n"
            f"approach z={ready_z:.3f} m"
        )
        markers.markers.append(label)

        self._marker_pub.publish(markers)
        self.get_logger().info(
            f"  [Marker] Insert preview published to /insert_preview\n"
            f"           In RViz2: Add -> By topic -> /insert_preview -> MarkerArray\n"
            f"           Fixed Frame must be '{world_frame}'"
        )

    # ------------------------------------------------------------------
    # Parallel planner — run N planning attempts simultaneously, pick best
    # ------------------------------------------------------------------

    def _score_trajectory(self, traj: RobotTrajectory,
                          j1_weight:       float = 0.4,
                          duration_weight: float = 0.2) -> tuple:
        """
        Score a trajectory for path quality selection.  Returns a tuple
        (score, stats_dict) where lower score = more preferred path.

        Scoring formula (all components normalised to comparable units):
          displacement_score = (1 - j1_weight) * total_disp
                             + j1_weight * 3.0  * j1_disp
          duration_score     = duration / 30.0   (normalised; 30s ≈ typical long path)
          combined_score     = (1 - duration_weight) * displacement_score
                             + duration_weight       * duration_score

        Why include duration:
          Duration and joint displacement are correlated for normal paths — a
          short, direct path moves fewer joints and takes less time.  However,
          OMPL occasionally produces paths with low displacement but long
          duration (many waypoints, small steps, inefficient timing) or vice
          versa.  Including duration breaks ties between similar displacement
          scores and catches these edge cases.

        Parameters
        ----------
        j1_weight       : 0–1, weight on joint_1 vs other joints (default 0.4).
                          Higher = penalise base rotation more.
        duration_weight : 0–1, weight on duration vs displacement (default 0.2).
                          0.2 means 20% of the score comes from duration.
                          Increase to 0.4 if you observe fast but odd paths being
                          rejected in favour of slow normal ones.

        Returns
        -------
        (score, stats) — score is float (lower = better),
                         stats is dict with keys: total_disp, j1_disp, duration,
                         max_single_jump, waypoints, displacement_score, duration_score
        """
        pts   = traj.joint_trajectory.points
        names = traj.joint_trajectory.joint_names
        stats = dict(total_disp=0.0, j1_disp=0.0, duration=0.0,
                     max_single_jump=0.0, waypoints=len(pts),
                     displacement_score=float('inf'), duration_score=float('inf'))

        if len(pts) < 2:
            return float('inf'), stats

        j1_idx = names.index("joint_1") if "joint_1" in names else -1

        total_disp     = 0.0
        j1_disp        = 0.0
        max_jump       = 0.0   # largest single-step joint displacement anywhere

        for i in range(1, len(pts)):
            step_max = 0.0
            for j, _ in enumerate(names):
                if j >= len(pts[i].positions) or j >= len(pts[i-1].positions):
                    continue
                d = abs(pts[i].positions[j] - pts[i-1].positions[j])
                total_disp += d
                step_max    = max(step_max, d)
                if j == j1_idx:
                    j1_disp += d
            max_jump = max(max_jump, step_max)

        last = pts[-1].time_from_start
        duration = last.sec + last.nanosec * 1e-9

        disp_score = (1.0 - j1_weight) * total_disp + j1_weight * 3.0 * j1_disp
        dur_score  = duration / 30.0   # normalise to [0, 1] range for typical paths

        combined = (1.0 - duration_weight) * disp_score + duration_weight * dur_score

        stats.update(dict(
            total_disp        = total_disp,
            j1_disp           = j1_disp,
            duration          = duration,
            max_single_jump   = max_jump,
            waypoints         = len(pts),
            displacement_score= disp_score,
            duration_score    = dur_score,
        ))
        return combined, stats

    def _is_suspicious_trajectory(self, traj: RobotTrajectory,
                                   p: dict,
                                   ready_z: float) -> tuple:
        """
        Filter out geometrically suspicious trajectories before scoring.

        Returns (is_suspicious, reason_string).  Suspicious trajectories are
        discarded from the parallel results pool rather than scored — this
        prevents a low-displacement-but-odd path from being selected.

        Checks
        ------
        1. Max single-step joint displacement > SUSPICIOUS_STEP_RAD.
           Normal smooth paths have steps < 0.1 rad per waypoint.
        2. Joint_1 total displacement > 120° (2.09 rad).
           Any path requiring > 120° base rotation is a circular arc.
        3. Joint_3 total displacement > 120° (2.09 rad).
           Joint_3 sits near ±180° at home; the IK solver can pick the mirror
           solution causing a full forearm spin.  Same limit as joint_1.
        4. Duration is suspiciously short (< 1.5 s) for a path with many
           waypoints — indicates velocity scaling was ignored.
        5. Trajectory is excessively long (> 60 s) — planning artefact.
        """
        pts   = traj.joint_trajectory.points
        names = traj.joint_trajectory.joint_names
        if len(pts) < 2:
            return True, "empty trajectory"

        j1_idx    = names.index("joint_1") if "joint_1" in names else -1
        j3_idx    = names.index("joint_3") if "joint_3" in names else -1
        j1_total  = 0.0
        j3_total  = 0.0
        max_step  = 0.0

        for i in range(1, len(pts)):
            for j, _ in enumerate(names):
                if j >= len(pts[i].positions) or j >= len(pts[i-1].positions):
                    continue
                d = abs(pts[i].positions[j] - pts[i-1].positions[j])
                if d > max_step:
                    max_step = d
                if j == j1_idx:
                    j1_total += d
                if j == j3_idx:
                    j3_total += d

        last     = pts[-1].time_from_start
        duration = last.sec + last.nanosec * 1e-9

        if max_step > SUSPICIOUS_STEP_RAD:
            return True, (f"large mid-path jump {math.degrees(max_step):.1f}°/step "
                          f"> {math.degrees(SUSPICIOUS_STEP_RAD):.1f}°")
        if j1_idx >= 0 and j1_total > math.radians(120):
            return True, f"joint_1 swings {math.degrees(j1_total):.1f}° > 120° (circular arc)"
        if j3_idx >= 0 and j3_total > math.radians(120):
            return True, f"joint_3 spins {math.degrees(j3_total):.1f}° > 120° (IK mirror flip)"
        if len(pts) > 10 and duration < 1.5:
            return True, f"suspiciously short duration {duration:.2f}s for {len(pts)} waypoints"
        if duration > 60.0:
            return True, f"excessively long duration {duration:.1f}s"

        return False, ""

    def _call_plan_threadsafe(self, motion_req: MotionPlanRequest):
        """
        Thread-safe planning call using a per-call service client.

        The shared self._plan_cli cannot be called from multiple threads
        simultaneously (the Future object is not thread-safe).  Creating a
        new client per call is safe because MoveIt's /plan_kinematic_path
        service handles concurrent requests correctly on the server side.
        """
        cli = self.create_client(
            GetMotionPlan, "/plan_kinematic_path",
            callback_group=self._cb_group)
        if not cli.wait_for_service(timeout_sec=5.0):
            return False, None
        svc_req = GetMotionPlan.Request()
        svc_req.motion_plan_request = motion_req
        f = cli.call_async(svc_req)
        result = self._wait_for_future(f, timeout_sec=45.0)
        if result is None:
            return False, None
        resp = result.motion_plan_response
        if resp.error_code.val == 1:
            return True, resp.trajectory
        return False, None

    def _plan_parallel(self, build_req_fn, p: dict, label: str = "parallel"):
        """
        Run p['parallel_plans'] planning calls simultaneously, filter suspicious
        results, score the rest, and return the best trajectory.

        Each attempt uses a fresh MotionPlanRequest so OMPL uses a different
        random seed.  Results are filtered through _is_suspicious_trajectory
        before scoring — this prevents geometrically odd paths (circular arcs,
        velocity-scaling violations) from being selected even if their raw
        displacement score happens to be low.

        The scoring table is printed for every run so you can see why each
        path was accepted/rejected and what was selected.

        Parameters (from p dict)
        ------------------------
        parallel_plans   : number of simultaneous attempts (default 5, use 10-15)
        parallel_timeout : seconds to wait for all attempts (default 30)
        j1_weight        : joint_1 penalty weight in scorer (default 0.4)
        duration_weight  : duration penalty weight in scorer (default 0.2)
        """
        n            = int(p.get("parallel_plans",   5))
        timeout      = float(p.get("parallel_timeout", 30.0))
        j1_w         = float(p.get("j1_weight",       0.4))
        dur_w        = float(p.get("duration_weight",  0.2))

        if n <= 1:
            return self._call_plan(build_req_fn())

        self.get_logger().info(
            f"  [Parallel] Starting {n} attempts "
            f"(j1_weight={j1_w:.1f}, duration_weight={dur_w:.1f})...")

        # (score, attempt_idx, trajectory, stats_dict)
        results   = []
        discarded = []   # (attempt_idx, reason)
        lock      = threading.Lock()

        def _attempt(idx):
            req = build_req_fn()
            req.num_planning_attempts = idx + 1   # different OMPL seed per attempt
            ok, traj = self._call_plan_threadsafe(req)
            if not ok or not traj:
                with lock:
                    discarded.append((idx, "planning failed"))
                return

            suspicious, reason = self._is_suspicious_trajectory(traj, p, 0.0)
            if suspicious:
                with lock:
                    discarded.append((idx, f"filtered: {reason}"))
                return

            score, stats = self._score_trajectory(traj, j1_w, dur_w)
            with lock:
                results.append((score, idx, traj, stats))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_attempt, i) for i in range(n)]
            concurrent.futures.wait(futures, timeout=timeout)

        # Print summary table
        all_rows = []
        for score, idx, _, stats in results:
            all_rows.append((idx, "✓ accepted", score, stats))
        for idx, reason in discarded:
            all_rows.append((idx, reason, float('inf'),
                             dict(j1_disp=0, duration=0, total_disp=0,
                                  max_single_jump=0, waypoints=0)))
        all_rows.sort(key=lambda r: r[0])

        header = (f"  {'#':>2}  {'status':<26}  "
                  f"{'score':>7}  {'dur(s)':>6}  "
                  f"{'j1(°)':>6}  {'tot(rad)':>8}  {'pts':>4}")
        self.get_logger().info(f"\n  [Parallel] Results ({len(results)} valid / {n} total):\n"
                               + header + "\n" + "  " + "-"*72)
        for idx, status, score, stats in all_rows:
            j1_deg = math.degrees(stats.get("j1_disp", 0))
            dur    = stats.get("duration", 0)
            tot    = stats.get("total_disp", 0)
            pts    = stats.get("waypoints", 0)
            sc_str = f"{score:.3f}" if score < float('inf') else "  —  "
            self.get_logger().info(
                f"  {idx:>2}  {status:<26}  "
                f"{sc_str:>7}  {dur:>6.1f}  "
                f"{j1_deg:>6.1f}  {tot:>8.3f}  {pts:>4}")

        if not results:
            # All attempts filtered — relax filtering and try once more
            self.get_logger().warn(
                "  [Parallel] All attempts were filtered as suspicious.\n"
                "  Relaxing filter and accepting best unfiltered result...")
            fallback_results = []
            for idx, reason in discarded:
                if "planning failed" not in reason:
                    # Re-plan this attempt without filtering
                    req = build_req_fn()
                    req.num_planning_attempts = idx + 1
                    ok, traj = self._call_plan_threadsafe(req)
                    if ok and traj:
                        score, stats = self._score_trajectory(traj, j1_w, dur_w)
                        fallback_results.append((score, idx, traj, stats))
            if not fallback_results:
                self.get_logger().error(
                    f"  [Parallel] All {n} planning attempts failed.")
                return False, None
            fallback_results.sort(key=lambda r: r[0])
            best = fallback_results[0]
            self.get_logger().warn(
                f"  [Parallel] Using fallback attempt {best[1]} "
                f"score={best[0]:.3f} — verify trajectory in RViz before executing.")
            return True, best[2]

        results.sort(key=lambda r: r[0])
        best_score, best_idx, best_traj, best_stats = results[0]
        worst_score = results[-1][0]
        improvement = (1 - best_score / worst_score) * 100 if worst_score > 0 else 0

        self.get_logger().info(
            f"\n  [Parallel] ★ Selected attempt {best_idx}:\n"
            f"    score       = {best_score:.3f}\n"
            f"    duration    = {best_stats['duration']:.1f} s\n"
            f"    joint_1     = {math.degrees(best_stats['j1_disp']):.1f}°\n"
            f"    total disp  = {best_stats['total_disp']:.3f} rad\n"
            f"    waypoints   = {best_stats['waypoints']}\n"
            f"    improvement = {improvement:.0f}% vs worst accepted path")

        return True, best_traj

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------

    def _action_goal_cb(self, goal_request):
        """Accept or reject incoming action goals."""
        if self._action_goal_handle is not None:
            self.get_logger().warn(
                "Action server: rejecting new goal — insertion already in progress.")
            return GoalResponse.REJECT
        self.get_logger().info(
            f"Action server: accepted goal "
            f"target=({goal_request.target_x:.3f}, {goal_request.target_y:.3f})"
            f"  hover={goal_request.hover_above_top:.3f} m"
            f"  dry_run={goal_request.dry_run}")
        return GoalResponse.ACCEPT

    def _action_cancel_cb(self, goal_handle):
        """Handle cancellation — stop the arm immediately."""
        self.get_logger().warn("Action server: cancel requested — stopping arm.")
        self._cancel_active()
        return CancelResponse.ACCEPT

    def _action_execute_cb(self, goal_handle):
        """
        Execute the full insertion sequence for an action goal.

        The goal overrides target_x, target_y, hover_above_top, and dry_run.
        All other parameters come from the node's ROS parameters.
        Uses the same _run_once_impl logic but publishes feedback at each phase
        and returns a structured result instead of logging only.
        """
        self._action_goal_handle = goal_handle
        goal = goal_handle.request

        # Build feedback and result objects
        if _HAS_ACTION_INTERFACE:
            feedback = InsertContainer.Feedback()
            result   = InsertContainer.Result()
        else:
            return  # should not reach here

        def fb(phase: str, progress: float):
            if goal_handle.is_active:
                feedback.current_phase = phase
                feedback.progress      = float(progress)
                goal_handle.publish_feedback(feedback)
            self.get_logger().info(
                f"  [Action] Phase: {phase}  progress={progress*100:.0f}%")

        def abort(msg: str):
            result.success = False
            result.message = msg
            self.get_logger().error(f"  [Action] Aborting: {msg}")
            self._recovery_return()
            goal_handle.abort(result)
            self._action_goal_handle = None
            return result

        # Override parameters from goal
        p = self._get_params()
        # Goal can specify target in metres (target_x/y) OR centimetres (forward_cm/sideways_cm).
        # Centimetre inputs take priority if non-zero — this lets calling scripts
        # use the same natural cm convention as the interactive prompt.
        if hasattr(goal, 'forward_cm') and goal.forward_cm != 0.0:
            p["target_x"] = goal.forward_cm  / 100.0
            p["target_y"] = goal.sideways_cm / 100.0
            self.get_logger().info(
                f"  [Action] Using cm input: "
                f"{goal.forward_cm:.0f} cm fwd, {goal.sideways_cm:.0f} cm side → "
                f"({p['target_x']:.3f}, {p['target_y']:.3f}) m")
        else:
            p["target_x"] = goal.target_x
            p["target_y"] = goal.target_y
        p["hover_above_top"] = goal.hover_above_top
        p["execute"]         = not goal.dry_run

        # --- Home move (optional) ---
        fb("home_move", 0.0)
        if not goal.skip_home_move and not goal.dry_run:
            if not self._move_to_home(p):
                return abort("Home move failed")
            time.sleep(1.5)

        # Freeze joints
        self._js_frozen = True

        # --- Pre-flight ---
        fb("preflight", 0.05)
        if not self._preflight_check(p):
            return abort("Pre-flight check failed")

        # --- Compute geometry (same as _run_once_impl) ---
        container_top = p["table_z"] + p["container_h"]
        target_z      = container_top + p["hover_above_top"]
        ready_z       = container_top + p["approach"]
        cx, cy        = p["target_x"], p["target_y"]
        p["ready_z"]  = ready_z
        p["target_z"] = target_z

        self._publish_target_marker(
            cx, cy, target_z, ready_z,
            p["container_h"], p["table_z"], p["world_frame"])

        tip_tf, ee_tf = self._get_tip_tf(p, timeout_sec=5.0)
        if tip_tf is None:
            return abort("TF lookup failed — is robot_state_publisher running?")

        t = tip_tf.transform.translation
        cur_x, cur_y, cur_z = t.x, t.y, t.z
        r = ee_tf.transform.rotation
        live_q = Quaternion(x=r.x, y=r.y, z=r.z, w=r.w)
        pen_q  = live_q if self.get_parameter("use_current_orientation").value \
                       else Quaternion(x=p["qx"], y=p["qy"], z=p["qz"], w=p["qw"])

        # Compute EE world offset from TF
        ee_world_offset = None
        try:
            wx = ee_tf.transform.translation.x - cur_x
            wy = ee_tf.transform.translation.y - cur_y
            wz = ee_tf.transform.translation.z - cur_z
            ee_world_offset = (wx, wy, wz)
        except Exception:
            pass

        # --- Phase 0: Approach ---
        fb("approach", 0.10)
        if goal_handle.is_cancel_requested:
            return abort("Cancelled before approach")
        ok, level_end = self._approach(cx, cy, ready_z, cur_x, cur_y, cur_z,
                                       p, live_q, pen_q,
                                       ee_world_offset=ee_world_offset)
        if not ok:
            return abort("Approach failed")

        # --- Phase 1: Descend ---
        fb("descend", 0.35)
        if goal_handle.is_cancel_requested:
            return abort("Cancelled before descend")
        ok, descend_end = self._descend(cx, cy, ready_z, target_z, p, pen_q,
                                        start_state=level_end,
                                        ee_world_offset=ee_world_offset)
        if not ok:
            return abort("Descend failed")

        # --- Phase 2: Hold ---
        fb("hold", 0.60)
        ok, _ = self._hold_at_target(cx, cy, target_z, p)
        if not ok:
            return abort("Hold failed")

        # --- Phase 2.5: Ascend ---
        fb("ascend", 0.75)
        if goal_handle.is_cancel_requested:
            return abort("Cancelled before ascend")
        ok, ascend_end = self._ascend(cx, cy, target_z, ready_z, p, pen_q,
                                      start_state=descend_end,
                                      ee_world_offset=ee_world_offset)
        if not ok:
            return abort("Ascend failed")

        # --- Phase 3: Return ---
        fb("return", 0.90)
        if p["ret"]:
            return_from = ascend_end if ascend_end is not None else descend_end
            self._return_to_start(p, start_state=return_from)

        fb("complete", 1.0)
        result.success = True
        result.message = f"Insertion complete at ({cx:.3f}, {cy:.3f})"
        goal_handle.succeed(result)
        self._action_goal_handle = None
        self.get_logger().info(f"  [Action] {result.message}")
        return result

    # ------------------------------------------------------------------
    # Interactive container position prompt
    # ------------------------------------------------------------------

    def _prompt_target_position(self, default_x: float,
                                 default_y: float) -> tuple:
        """
        Ask the user to enter the container centre position in centimetres
        from the robot base.  Returns (x_metres, y_metres).

        Coordinate convention (matches the robot world frame):
          forward_cm  : distance in front of the robot base along X axis.
                        Always positive.  Example: 43 cm → x = 0.430 m
          sideways_cm : distance left (+) or right (−) along Y axis.
                        Right is negative.  Example: -20 cm → y = −0.200 m

        The current default values (from parameters) are shown so the user
        can just press Enter to keep them.

        Input format accepted:
          "43, -20"    →  x=0.430, y=-0.200
          "43 -20"     →  x=0.430, y=-0.200  (space separator also works)
          "43"         →  x=0.430, y=default_y  (only X provided)
          ""  (Enter)  →  use defaults unchanged
        """
        default_fwd_cm  = default_x * 100.0
        default_side_cm = default_y * 100.0

        print()
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │  Container position (cm from robot base)                │")
        print("  │                                                         │")
        print("  │  Coordinate convention:                                 │")
        print("  │    forward_cm   : +ve = away from robot  (→ target_x)  │")
        print("  │    sideways_cm  : -ve = right, +ve = left (→ target_y)  │")
        print("  │                                                         │")
        print(f"  │  Current defaults: {default_fwd_cm:.0f} cm fwd, "
              f"{default_side_cm:.0f} cm side               │")
        print("  │  Enter two values: e.g.  43, -20                       │")
        print("  │  Press Enter to keep current defaults.                  │")
        print("  └─────────────────────────────────────────────────────────┘")

        while True:
            try:
                raw = input("  Container position [forward_cm, sideways_cm]: ").strip()

                if not raw:
                    # Keep defaults
                    self.get_logger().info(
                        f"  Using default position: "
                        f"{default_fwd_cm:.0f} cm fwd, "
                        f"{default_side_cm:.0f} cm side → "
                        f"({default_x:.3f}, {default_y:.3f}) m")
                    return default_x, default_y

                # Accept comma or space as separator
                parts = raw.replace(",", " ").split()
                if len(parts) == 1:
                    fwd_cm  = float(parts[0])
                    side_cm = default_side_cm
                elif len(parts) == 2:
                    fwd_cm  = float(parts[0])
                    side_cm = float(parts[1])
                else:
                    print("  Please enter one or two numbers, e.g.  43  or  43, -20")
                    continue

                x_m = fwd_cm  / 100.0
                y_m = side_cm / 100.0

                # Basic sanity check — Gen3 reach is 902 mm
                dist = math.sqrt(x_m**2 + y_m**2)
                if dist > 0.90:
                    print(f"  Warning: {fwd_cm:.0f} cm fwd, {side_cm:.0f} cm side "
                          f"is {dist*100:.0f} cm from base — near/outside arm reach "
                          f"(max ~90 cm).")
                    confirm = input("  Continue anyway? [y/N]: ").strip().lower()
                    if confirm != "y":
                        continue

                self.get_logger().info(
                    f"  Container position set: "
                    f"{fwd_cm:.0f} cm fwd, {side_cm:.0f} cm side → "
                    f"({x_m:.3f}, {y_m:.3f}) m")
                return x_m, y_m

            except ValueError:
                print("  Invalid input — please enter numbers, e.g.  43, -20")
            except EOFError:
                # Non-interactive environment — use defaults
                return default_x, default_y


    def _run_once(self):
        if self._done:
            return
        self._done = True
        try:
            self._run_once_impl()
        finally:
            self._run_done.set()

    def _run_once_impl(self):
        p = self._get_params()

        if self.get_parameter("real_robot").value:
            skip_home = self.get_parameter("skip_home_move").value
            if skip_home:
                self.get_logger().info(
                    "\n[Real robot] skip_home_move:=true — skipping home move.\n"
                    "  Ensure arm is in a safe upright pose before continuing.")
                input("[Real robot] Press ENTER when arm is ready ...")
            else:
                input(
                    "\n[Real robot] Press ENTER to move arm to Home position ...\n"
                    "  (If already near Home, re-run with -p skip_home_move:=true)")
                if not self._move_to_home(p):
                    self.get_logger().error("Home move failed — aborting.")
                    return
                time.sleep(1.5)
                input("[Real robot] Arm is at Home. Press ENTER to begin insertion ...")

            self._js_frozen = True
            if not self._start_joints:
                self.get_logger().warn(
                    "No joint states received yet -- return-to-start may skip.")
            else:
                self.get_logger().info(
                    "  Start joints frozen — recovery will return here.")

        # --- Container position ---
        # If interactive_target is set, prompt the user to enter the container
        # position in centimetres.  Otherwise use the target_x / target_y params.
        if self.get_parameter("interactive_target").value:
            new_x, new_y = self._prompt_target_position(
                p["target_x"], p["target_y"])
            p["target_x"] = new_x
            p["target_y"] = new_y

        # If use_dynamic_tf is set to True, look up the target center from the TF tree
        if self.get_parameter("use_dynamic_tf").value:
            try:
                # Look up latest transform from world to marker_square_center
                tf_target = self.tf_buffer.lookup_transform(
                    p["world_frame"],
                    "marker_square_center",
                    rclpy.time.Time(),
                    timeout=RclpyDuration(seconds=2.0)
                )
                p["target_x"] = tf_target.transform.translation.x
                p["target_y"] = tf_target.transform.translation.y
                self.get_logger().info(
                    f"[TF Lookup] Dynamically loaded container target from TF frame "
                    f"'marker_square_center': x={p['target_x']:.3f}, y={p['target_y']:.3f}")
            except Exception as e:
                self.get_logger().error(
                    f"[TF Lookup] Failed to look up 'marker_square_center' transform: {e}. "
                    f"Falling back to default target parameters.")
        if getattr(self, '_camera_target', None) is not None:
            p["target_x"] = self._camera_target.pose.position.x
            p["target_y"] = self._camera_target.pose.position.y
            self.get_logger().info(f"[Vision] Dynamic target from /fused_marker_square_center: ({p['target_x']:.3f}, {p['target_y']:.3f})")


        # Compute world-frame Z values from container parameters.
        #   container top  = table_z + container_height
        #   target_z       = container top + hover_above_top
        #   ready_z        = container top + approach_clearance
        container_top = p["table_z"] + p["container_h"]
        target_z      = container_top + p["hover_above_top"]  # hover above, not insert
        ready_z       = container_top + p["approach"]

        cx = p["target_x"]
        cy = p["target_y"]

        self.get_logger().info("=" * 60)
        self.get_logger().info("insert_to_container")
        self.get_logger().info(
            f"  container centre: x={cx:.3f}  y={cy:.3f}")
        self.get_logger().info(
            f"  container top:    z={container_top:.3f} m  "
            f"(table_z={p['table_z']:.3f} + height={p['container_h']*100:.0f} cm)")
        self.get_logger().info(
            f"  hover target:     z={target_z:.3f} m  "
            f"({p['hover_above_top']*1000:.0f} mm above container top)")
        self.get_logger().info(
            f"  approach height:  z={ready_z:.3f} m  "
            f"({p['approach']*100:.0f} cm above container top)")
        self.get_logger().info(
            f"  execute={p['execute']}")
        self.get_logger().info("=" * 60)

        if not self.get_parameter("real_robot").value:
            self._js_frozen = True
        if not self._start_joints:
            self.get_logger().warn(
                "No joint states received yet -- return-to-start skipped.")

        # Run pre-flight safety check before any planning or motion.
        if not self._preflight_check(p):
            self.get_logger().error(
                "Pre-flight check failed — fix the issues above before running.\n"
                "No motion will occur.")
            return

        # Inject ready_z and target_z into params dict for downstream methods
        p["ready_z"]  = ready_z
        p["target_z"] = target_z

        # Publish RViz marker for visual verification
        self._publish_target_marker(
            cx, cy, target_z, ready_z,
            p["container_h"], p["table_z"], p["world_frame"])

        if self.get_parameter("add_pole_collision").value:
            self.get_logger().info("  Adding pole collision object to planning scene...")
            self._apply_pole_collision(add=True, p=p)

        # Read current pen_tip position and actual EE orientation from TF.
        # Using the live TF orientation preserves whatever orientation the arm
        # is already in — avoids any large reorientation at startup.
        tip_tf, ee_tf = self._get_tip_tf(p, timeout_sec=5.0)
        if tip_tf is None or ee_tf is None:
            self.get_logger().error(
                "  Could not get TF (world→pen_tip / world→EE). "
                "Is robot_state_publisher running?")
            return
        t = tip_tf.transform.translation
        cur_x, cur_y, cur_z = t.x, t.y, t.z
        r = ee_tf.transform.rotation
        # live_q: the arm's actual current EE orientation from TF.
        live_q = Quaternion(x=r.x, y=r.y, z=r.z, w=r.w)

        use_live = self.get_parameter("use_current_orientation").value
        use_tf   = self.get_parameter("use_tf_offset").value

        if use_live:
            pen_q = live_q
            self.get_logger().info(
                f"  Current pen_tip: ({cur_x:.3f}, {cur_y:.3f}, {cur_z:.3f})\n"
                f"  EE orientation (live, used as pen-down): "
                f"qx={r.x:.4f}  qy={r.y:.4f}  qz={r.z:.4f}  qw={r.w:.4f}\n"
                f"  use_current_orientation:=true — arm must already point tip down.")
        else:
            pen_q = Quaternion(x=p["qx"], y=p["qy"], z=p["qz"], w=p["qw"])
            self.get_logger().info(
                f"  Current pen_tip: ({cur_x:.3f}, {cur_y:.3f}, {cur_z:.3f})\n"
                f"  EE orientation (live):  qx={r.x:.4f}  qy={r.y:.4f}  qz={r.z:.4f}  qw={r.w:.4f}\n"
                f"  Pen-down target quat:   qx={p['qx']:.4f}  qy={p['qy']:.4f}  "
                f"qz={p['qz']:.4f}  qw={p['qw']:.4f}")

        # Compute world-frame EE-to-tip offset.
        # EE_origin = pen_tip_target + world_offset
        # Best source: subtract world→EE from world→pen_tip (already have both from TF).
        ee_world_offset = None
        if use_tf:
            wx, wy, wz = self._get_ee_to_tip_offset_world(p, None)
            if wx is not None:
                ee_world_offset = (wx, wy, wz)
                self.get_logger().info(
                    f"  EE→tip world offset (TF-based, direct subtraction): "
                    f"({wx:.4f}, {wy:.4f}, {wz:.4f})")

        if ee_world_offset is None:
            # Fallback: compute from current world TF positions directly.
            # EE_origin_world - pen_tip_world gives the world-frame offset.
            # This is equivalent to the TF method but uses the already-looked-up transforms.
            try:
                ee_w_x = ee_tf.transform.translation.x
                ee_w_y = ee_tf.transform.translation.y
                ee_w_z = ee_tf.transform.translation.z
                wx = ee_w_x - cur_x
                wy = ee_w_y - cur_y
                wz = ee_w_z - cur_z
                ee_world_offset = (wx, wy, wz)
                dist = math.sqrt(wx**2 + wy**2 + wz**2)
                self.get_logger().info(
                    f"  EE→tip world offset (from existing TF lookups): "
                    f"({wx:.4f}, {wy:.4f}, {wz:.4f})  magnitude={dist*100:.1f} cm")
            except Exception:
                # Last resort: rotate local offset by pen_q
                q_for_offset = (pen_q.x, pen_q.y, pen_q.z, pen_q.w)
                wx, wy, wz = rotate_vector_by_quat(
                    (p["tip_ee_ox"], p["tip_ee_oy"], p["tip_ee_oz"]), q_for_offset)
                ee_world_offset = (wx, wy, wz)
                self.get_logger().warn(
                    f"  EE→tip world offset (rotated params — may be wrong for qw≈0): "
                    f"({wx:.4f}, {wy:.4f}, {wz:.4f})\n"
                    f"  If EE target looks wrong, check tip_ee_offset parameters.")

        # Phase 0 -- Pilz approach above container
        ok, level_end = self._approach(cx, cy, ready_z, cur_x, cur_y, cur_z,
                                       p, live_q, pen_q,
                                       ee_world_offset=ee_world_offset)
        if not ok:
            self._recovery_return()
            return

        # Phase 1 -- descend into container
        ok, descend_end = self._descend(
            cx, cy, ready_z, target_z, p, pen_q,
            start_state=level_end, ee_world_offset=ee_world_offset)
        if not ok:
            self._recovery_return()
            return

        # Phase 2 -- hold at insert depth
        ok, _ = self._hold_at_target(cx, cy, target_z, p)
        if not ok:
            self._recovery_return()
            return

        # Phase 2.5 -- ascend back to approach height
        ok, ascend_end = self._ascend(
            cx, cy, target_z, ready_z, p, pen_q,
            start_state=descend_end, ee_world_offset=ee_world_offset)
        if not ok:
            self._recovery_return()
            return

        # Phase 3 -- return to start
        if p["ret"]:
            return_from = ascend_end if ascend_end is not None else descend_end
            self._return_to_start(p, start_state=return_from)
        else:
            self.get_logger().info("Return skipped (return_to_start:=false).")

        if self.get_parameter("add_pole_collision").value:
            self._apply_pole_collision(add=False, p=p)

        self.get_logger().info("=" * 60)
        self.get_logger().info("insert_to_container complete.")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = ContainerInserter()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    _interrupted = threading.Event()

    def _sigint_handler(sig, frame):
        if _interrupted.is_set():
            print("\n[Ctrl+C] Force-quitting.", file=sys.stderr)
            sys.exit(1)
        _interrupted.set()
        node._run_done.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    node._run_done.wait()

    executor.shutdown(timeout_sec=2.0)
    spin_thread.join(timeout=3.0)

    if _interrupted.is_set():
        try:
            node.get_logger().warn(
                "\n[Ctrl+C] Interrupt received.\n"
                "  Stopping arm and returning to pre-script position...\n"
                "  Press Ctrl+C again to force-quit without returning.")
            node._cancel_active()
            time.sleep(0.5)
            node._recovery_return()
        except Exception as e:
            print(f"Recovery error: {e}", file=sys.stderr)

    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()