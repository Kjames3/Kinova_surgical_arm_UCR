#!/usr/bin/env python3
"""
Test script: verify pen_tip IK offset on the Kinova Gen3 7-DOF + Robotiq 2F-140.

What it checks
--------------
1. TF sanity  : pen_tip must be exactly 0.255 m along end_effector_link −X.
2. Current pose: prints pen_tip pose in world frame at the current joint state.
3. Cartesian  : plans (position only, orientation free) to a user-specified XYZ
                and optionally executes, then reads back pen_tip TF.

Usage
-----
  ros2 run surgical_arm_bringup test_pen_tip_offset.py

  # Override target position (metres in world frame):
  ros2 run surgical_arm_bringup test_pen_tip_offset.py \
    --ros-args -p target_x:=0.4 -p target_y:=0.0 -p target_z:=0.25

  # Also enforce pen-vertical throughout the PATH (not just at goal):
  ros2 run surgical_arm_bringup test_pen_tip_offset.py \
    --ros-args -p enforce_path_constraint:=true

  # Execute motion (moves the arm!):
  ros2 run surgical_arm_bringup test_pen_tip_offset.py \
    --ros-args -p execute_motion:=true

Tuning hint
-----------
  If pen_tip is consistently SHORT of the physical tip  → increase magnitude (e.g. -0.255 → -0.265).
  If pen_tip is consistently PAST  the physical tip     → decrease magnitude (e.g. -0.255 → -0.245).
  Edit kortex_description/grippers/robotiq_2f_140/urdf/robotiq_2f_140_macro.xacro,
  the <origin xyz="-0.255 0 0"> on the pen_tip_joint, then rebuild & re-source.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import tf2_ros
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    WorkspaceParameters,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from rclpy.action import ActionClient


# ---------------------------------------------------------------------------
# Helper: pretty-print a transform
# ---------------------------------------------------------------------------
def _fmt_tf(t):
    tr = t.transform.translation
    ro = t.transform.rotation
    return (
        f"  translation  x={tr.x:.4f}  y={tr.y:.4f}  z={tr.z:.4f}\n"
        f"  rotation     x={ro.x:.4f}  y={ro.y:.4f}  z={ro.z:.4f}  w={ro.w:.4f}"
    )


class PenTipTester(Node):
    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__("pen_tip_tester")

        # Allow callbacks to run concurrently (required for action clients
        # called from timer callbacks — avoids nested-spin deadlock).
        self._cb_group = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter("target_x", 0.4)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", 0.25)  # 25 cm above base — within pen-down workspace
        self.declare_parameter("pen_tip_offset", 0.255)   # metres; must match xacro
        self.declare_parameter("move_group_name", "manipulator")
        self.declare_parameter("tip_link", "pen_tip")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("ee_link", "end_effector_link")
        self.declare_parameter("execute_motion", False)    # safety: default dry-run
        # Pen-down orientation (quaternion xyzw).
        # The pen axis is end_effector_link +X.  "Pen vertical, tip down" means
        # end_effector_link +X points world -Z.
        # Quaternion: RPY=[90°, 0°, 90°] → (x=0.5, y=0.5, z=0.5, w=0.5).
        # Live measured values from tf2_echo: (x=0.497, y=0.476, z=0.501, w=0.525).
        # Verify by running: ros2 run tf2_ros tf2_echo world end_effector_link
        # with the arm in pen-down position, then update these four values if needed.
        self.declare_parameter("pen_down_qx",  0.5)
        self.declare_parameter("pen_down_qy",  0.5)
        self.declare_parameter("pen_down_qz",  0.5)
        self.declare_parameter("pen_down_qw",  0.5)
        # How tightly to enforce vertical (radians). ~0.05 rad ≈ 3°.
        self.declare_parameter("vertical_tilt_tolerance", 0.05)
        # Whether to enforce pen-vertical as a PATH constraint (pen stays vertical
        # throughout the entire motion) vs. only at the goal.  Path constraints
        # dramatically reduce the planner's search space and can cause failures when
        # moving to positions far from the current pose.  Default: goal-only (False).
        self.declare_parameter("enforce_path_constraint", False)

        self.tip_link    = self.get_parameter("tip_link").value
        self.world_frame = self.get_parameter("world_frame").value
        self.ee_link     = self.get_parameter("ee_link").value
        self.expected_offset = self.get_parameter("pen_tip_offset").value

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self,
                                                     spin_thread=False)

        # MoveGroup action client
        self._mg_client = ActionClient(self, MoveGroup, "/move_action",
                                       callback_group=self._cb_group)

        # Run after a short delay so TF is populated
        self.create_timer(2.0, self._run_once,
                          callback_group=self._cb_group)
        self._done = False

    # ------------------------------------------------------------------
    # Main routine (called once via timer)
    # ------------------------------------------------------------------
    def _run_once(self):
        if self._done:
            return
        self._done = True

        self.get_logger().info("=" * 60)
        self.get_logger().info("pen_tip offset verification")
        self.get_logger().info("=" * 60)

        self._check_static_offset()
        self._report_tip_in_world("current")
        self._plan_to_position()

        self.get_logger().info("=" * 60)
        self.get_logger().info(
            "Done.  Tune robotiq_2f_140_macro.xacro pen_tip_joint origin X "
            f"(currently -{self.expected_offset:.3f} m) if offset looks wrong."
        )
        self.get_logger().info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Check pen_tip vs end_effector_link static offset
    # ------------------------------------------------------------------
    def _check_static_offset(self):
        self.get_logger().info(
            "\n--- [1] Static TF: pen_tip relative to end_effector_link ---")
        try:
            t = self.tf_buffer.lookup_transform(
                self.ee_link, self.tip_link,
                rclpy.time.Time(), timeout=RclpyDuration(seconds=5)
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return

        tz = t.transform.translation.z
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        self.get_logger().info(_fmt_tf(t))

        # pen_tip_joint origin is now xyz="-0.255 0 0" so tx should be -0.255,
        # ty and tz should both be ~0.
        ok = (
            abs(tx + self.expected_offset) < 1e-4 and
            abs(ty) < 1e-4 and
            abs(tz) < 1e-4
        )
        if ok:
            self.get_logger().info(
                f"  [PASS] pen_tip X offset = {tx:.4f} m  "
                f"(expected -{self.expected_offset:.3f} m)"
            )
        else:
            self.get_logger().warn(
                f"  [FAIL] pen_tip offset mismatch!\n"
                f"         got  x={tx:.4f}  y={ty:.4f}  z={tz:.4f}\n"
                f"         want x=-{self.expected_offset:.4f}  y=0.0000  z=0.0000\n"
                "         → Check pen_tip_joint origin in robotiq_2f_140_macro.xacro"
            )

    # ------------------------------------------------------------------
    # 2. Report pen_tip pose in world frame (current joint state)
    # ------------------------------------------------------------------
    def _report_tip_in_world(self, label: str):
        self.get_logger().info(
            f"\n--- [2] pen_tip in world frame ({label} pose) ---")
        try:
            t = self.tf_buffer.lookup_transform(
                self.world_frame, self.tip_link,
                rclpy.time.Time(), timeout=RclpyDuration(seconds=5)
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return

        self.get_logger().info(_fmt_tf(t))
        tr = t.transform.translation
        dist = math.sqrt(tr.x**2 + tr.y**2 + tr.z**2)
        self.get_logger().info(
            f"  Distance from world origin: {dist:.4f} m"
        )

        # Also print end_effector_link → pen_tip in world to show pen direction
        try:
            tee = self.tf_buffer.lookup_transform(
                self.world_frame, self.ee_link,
                rclpy.time.Time(), timeout=RclpyDuration(seconds=2)
            )
            ro = tee.transform.rotation
            self.get_logger().info(
                f"  end_effector_link orientation in world: "
                f"x={ro.x:.4f}  y={ro.y:.4f}  z={ro.z:.4f}  w={ro.w:.4f}\n"
                f"  (Use this to determine which world axis the pen is aligned with)"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3. Plan (position only) to pen_tip target — no orientation lock
    # ------------------------------------------------------------------
    def _plan_to_position(self):
        self.get_logger().info(
            "\n--- [3] Position-only plan via MoveIt (pen_tip, Pilz PTP) ---")

        if not self._mg_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                "MoveGroup action server not available — skipping.")
            return

        tx = self.get_parameter("target_x").value
        ty = self.get_parameter("target_y").value
        tz = self.get_parameter("target_z").value
        execute = self.get_parameter("execute_motion").value

        qx = self.get_parameter("pen_down_qx").value
        qy = self.get_parameter("pen_down_qy").value
        qz = self.get_parameter("pen_down_qz").value
        qw = self.get_parameter("pen_down_qw").value
        tilt_tol = self.get_parameter("vertical_tilt_tolerance").value
        enforce_path = self.get_parameter("enforce_path_constraint").value

        self.get_logger().info(
            f"  Target pen_tip in world frame: "
            f"x={tx:.3f}  y={ty:.3f}  z={tz:.3f}\n"
            f"  Orientation: pen vertical (down), yaw free\n"
            f"  pen_down quaternion: x={qx:.3f} y={qy:.3f} z={qz:.3f} w={qw:.3f}\n"
            f"  Path constraint: {'ON (pen vertical throughout)' if enforce_path else 'OFF (goal only)'}"
        )

        # Build MotionPlanRequest
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()

        req.group_name  = self.get_parameter("move_group_name").value
        req.planner_id  = "RRTConnectkConfigDefault"
        req.pipeline_id = "ompl"
        req.num_planning_attempts = 5
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.1

        # Tell MoveIt to start from the current robot state
        req.start_state.is_diff = True

        req.workspace_parameters = WorkspaceParameters()
        req.workspace_parameters.header.frame_id = self.world_frame
        req.workspace_parameters.min_corner.x = -1.0
        req.workspace_parameters.min_corner.y = -1.0
        req.workspace_parameters.min_corner.z = -0.5
        req.workspace_parameters.max_corner.x =  1.0
        req.workspace_parameters.max_corner.y =  1.0
        req.workspace_parameters.max_corner.z =  1.5

        # Position constraint only — no orientation lock
        pos_c = PositionConstraint()
        pos_c.header.frame_id = self.world_frame
        pos_c.link_name = self.tip_link
        pos_c.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]   # 1 cm tolerance sphere

        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv_pose = Pose()
        bv_pose.position.x = tx
        bv_pose.position.y = ty
        bv_pose.position.z = tz
        bv_pose.orientation.w = 1.0
        bv.primitive_poses.append(bv_pose)
        pos_c.constraint_region = bv

        # Orientation constraint: pen vertical (Z axis pointing down), yaw free.
        # Applied as BOTH a goal constraint AND a path constraint so the pen
        # stays vertical throughout the entire trajectory, not just at the goal.
        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self.world_frame
        ori_c.link_name = self.tip_link
        ori_c.orientation.x = qx
        ori_c.orientation.y = qy
        ori_c.orientation.z = qz
        ori_c.orientation.w = qw
        ori_c.absolute_x_axis_tolerance = tilt_tol   # tight: ~3° tilt allowed
        ori_c.absolute_y_axis_tolerance = tilt_tol   # tight: ~3° tilt allowed
        ori_c.absolute_z_axis_tolerance = 3.14159    # free: pen can spin around its axis
        ori_c.weight = 1.0

        # Goal: reach position AND be vertical at the target
        constraints = Constraints()
        constraints.position_constraints.append(pos_c)
        constraints.orientation_constraints.append(ori_c)
        req.goal_constraints.append(constraints)

        # Path constraint: only applied when enforce_path_constraint=true.
        # Keeping pen vertical at EVERY waypoint greatly reduces the planner's
        # search space — use for actual writing motions, not simple reachability tests.
        if enforce_path:
            path_constraints = Constraints()
            path_constraints.orientation_constraints.append(ori_c)
            req.path_constraints = path_constraints

        goal_msg.request = req
        goal_msg.planning_options.plan_only = not execute
        goal_msg.planning_options.replan = False

        # Send goal — safe inside ReentrantCallbackGroup with MultiThreadedExecutor
        future = self._mg_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                "  Goal rejected by MoveGroup.\n"
                "  Check: is the target reachable? Is the arm in a fault state?")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        result = result_future.result()

        if result is None:
            self.get_logger().error("  No result received (timeout).")
            return

        ec = result.result.error_code.val
        if ec == 1:  # SUCCESS
            self.get_logger().info("  [PASS] Plan succeeded.")
            if execute:
                self.get_logger().info(
                    "  Motion executed — reading back pen_tip TF …")
                import time; time.sleep(2.0)
                self._report_tip_in_world("post-execution")
        else:
            self.get_logger().warn(
                f"  [WARN] Planning failed with error code {ec}.\n"
                "         Try adjusting target_x/y/z or check arm reachability."
            )


def main(args=None):
    rclpy.init(args=args)
    node = PenTipTester()
    # MultiThreadedExecutor lets action client callbacks run concurrently
    # with the timer callback — prevents nested-spin deadlock.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
