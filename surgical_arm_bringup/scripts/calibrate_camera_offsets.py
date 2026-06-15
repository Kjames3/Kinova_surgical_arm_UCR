#!/usr/bin/env python3
"""
calibrate_camera_offsets.py
────────────────────────────────────────────────────────────────────────────────
Automatically determine the x_offset / y_offset / z_offset parameters needed
by combine_cameras.py to correct residual hand-eye calibration errors.

The script is designed for **minimal user involvement**.  Three modes are
available; choose the one that matches your setup:

══════════════════════════════════════════════════════════════════════════════
 MODE 1 — "arm"  (RECOMMENDED — fully automatic, needs MoveIt)
══════════════════════════════════════════════════════════════════════════════
  The arm moves assembly_tip to hover directly above the ArUco board center
  at several heights.  Because the tip is vertically above the board, the TF
  position of assembly_tip gives the exact XY ground truth.  The Z ground
  truth is table_z + marker_height_above_table.

  The script computes: residual = TF_truth − camera_reported
  and then:            new_offset = current_offset + mean(residual)

  The arm visits calibration_poses number of heights between hover_z_min and
  hover_z_max (all with the same XY as the board center).

══════════════════════════════════════════════════════════════════════════════
 MODE 2 — "manual"  (semi-automatic — jog arm by hand)
══════════════════════════════════════════════════════════════════════════════
  The arm is jogged manually to position assembly_tip directly above the board
  center.  Press Enter at each measurement position.  The script reads TF for
  ground truth at that moment.  No MoveIt required; works with the Xbox
  controller or any jogging interface.

  Collect at least 4 positions (different heights) for a reliable calibration.

══════════════════════════════════════════════════════════════════════════════
 MODE 3 — "known"  (zero arm movement — fastest)
══════════════════════════════════════════════════════════════════════════════
  The user provides the exact world-frame board center position via parameters.
  The script averages combine_cameras output for average_seconds, then reports
  the offset needed.  No arm or TF required.

  Measure board center physically or jog the arm there once and read TF,
  then pass those coordinates to this mode.

══════════════════════════════════════════════════════════════════════════════
Usage
══════════════════════════════════════════════════════════════════════════════
  # ARM mode — arm moves automatically:
  ros2 run surgical_arm_bringup calibrate_camera_offsets.py \\
      --ros-args -p mode:=arm -p board_center_x:=0.299 -p board_center_y:=-0.192

  # MANUAL mode — you jog arm to hover over board, press Enter each time:
  ros2 run surgical_arm_bringup calibrate_camera_offsets.py --ros-args -p mode:=manual

  # KNOWN-POSITION mode — pass physical measurement directly:
  ros2 run surgical_arm_bringup calibrate_camera_offsets.py \\
      --ros-args -p mode:=known \\
                 -p known_x:=0.299 -p known_y:=-0.192 -p known_z:=0.055

══════════════════════════════════════════════════════════════════════════════
Output
══════════════════════════════════════════════════════════════════════════════
  - Printed calibration report with exact ros2 param set commands
  - Saved YAML config file (default: ~/calibration_offsets.yaml)
  - Live parameter update of /combine_cameras_node (if apply_live:=true)

Requirements:
  - combine_cameras.py must be running and publishing /fused_marker_square_center
  - robot_state_publisher running for TF  (arm + manual modes)
  - MoveIt running  (arm mode only)
  - ArUco board placed on table and visible to at least one camera
"""

import math
import sys
import time
import threading
import yaml
import os

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint,
    PositionConstraint, OrientationConstraint, WorkspaceParameters,
    BoundingVolume, RobotState, RobotTrajectory,
)
from moveit_msgs.action import ExecuteTrajectory
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from builtin_interfaces.msg import Duration as RosDuration
from shape_msgs.msg import SolidPrimitive
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
import tf2_ros
import numpy as np


# ─── Calibration target joints (same as insert_to_container.py home) ─────────
_GEN3_JOINTS = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7",
]

# Assembly-tip-vertical home joints (arm points tip straight down).
# Used as the starting configuration for ARM mode hover moves.
_GEN3_HOME_JOINTS = {
    "joint_1":  0.0000,
    "joint_2": -0.3049,
    "joint_3": -3.1416,
    "joint_4": -1.6607,
    "joint_5":  0.0000,
    "joint_6": -1.7928,
    "joint_7": -0.0006,
}

# ─── Kinematic safety limits ──────────────────────────────────────────────────
_MAX_JOINT_VEL_RAD_S  = 0.6
_MAX_JOINT_ACC_RAD_S2 = 0.3


# ══════════════════════════════════════════════════════════════════════════════
# Statistics helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mean_and_std(samples: list) -> tuple:
    """Return (mean, std) for a list of np.ndarray samples."""
    arr = np.array(samples)
    return arr.mean(axis=0), arr.std(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Main calibration node
# ══════════════════════════════════════════════════════════════════════════════

class CameraOffsetCalibrator(Node):

    def __init__(self):
        super().__init__("camera_offset_calibrator")
        self._cb = ReentrantCallbackGroup()

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("mode",                "manual")   # arm | manual | known

        # Board geometry (world frame)
        self.declare_parameter("board_center_x",      0.299)     # m — used in arm mode
        self.declare_parameter("board_center_y",     -0.192)     # m
        self.declare_parameter("table_z",            -0.030)     # m
        self.declare_parameter("marker_height_above_table", 0.005)  # m (board thickness)

        # ARM mode: hover heights above the board center
        self.declare_parameter("hover_z_min",         0.10)      # m above table
        self.declare_parameter("hover_z_max",         0.30)      # m above table
        self.declare_parameter("calibration_poses",   5)         # number of heights
        self.declare_parameter("settle_time",         2.0)       # s — wait after each move
        self.declare_parameter("samples_per_pose",    20)        # camera readings per pose

        # MANUAL mode
        self.declare_parameter("manual_samples",      6)         # how many Enter presses

        # KNOWN mode
        self.declare_parameter("known_x",             0.0)       # m
        self.declare_parameter("known_y",             0.0)       # m
        self.declare_parameter("known_z",             0.0)       # m
        self.declare_parameter("average_seconds",     15.0)      # s — integration time

        # Current combine_cameras offsets (to compute incremental correction)
        self.declare_parameter("current_x_offset",    0.0)
        self.declare_parameter("current_y_offset",    0.0)
        self.declare_parameter("current_z_offset",    0.0)

        # Output
        self.declare_parameter("output_yaml",         os.path.expanduser("~/calibration_offsets.yaml"))
        self.declare_parameter("apply_live",          True)      # set params on combine_cameras_node

        # MoveIt / motion
        self.declare_parameter("tip_link",            "assembly_tip")
        self.declare_parameter("ee_link",             "bracelet_link")
        self.declare_parameter("world_frame",         "world")
        self.declare_parameter("move_group_name",     "manipulator")
        self.declare_parameter("velocity_scaling",    0.20)

        # Pen-down quaternion (world → bracelet_link, tip pointing down)
        self.declare_parameter("pen_down_qx",  0.5)
        self.declare_parameter("pen_down_qy",  0.5)
        self.declare_parameter("pen_down_qz",  0.5)
        self.declare_parameter("pen_down_qw",  0.5)

        # ── Read parameters ───────────────────────────────────────────────────
        self._mode        = self.get_parameter("mode").value
        self._board_cx    = self.get_parameter("board_center_x").value
        self._board_cy    = self.get_parameter("board_center_y").value
        self._table_z     = self.get_parameter("table_z").value
        self._marker_h    = self.get_parameter("marker_height_above_table").value
        self._world       = self.get_parameter("world_frame").value
        self._tip_link    = self.get_parameter("tip_link").value
        self._ee_link     = self.get_parameter("ee_link").value
        self._group       = self.get_parameter("move_group_name").value
        self._vel         = self.get_parameter("velocity_scaling").value
        self._cur_offsets = np.array([
            self.get_parameter("current_x_offset").value,
            self.get_parameter("current_y_offset").value,
            self.get_parameter("current_z_offset").value,
        ])

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)

        # ── Camera measurement cache ──────────────────────────────────────────
        self._camera_lock   = threading.Lock()
        self._camera_latest = None   # np.array([x, y, z]) — last fused center

        self._cam_sub = self.create_subscription(
            PoseStamped,
            "/fused_marker_square_center",
            self._camera_cb,
            10,
            callback_group=self._cb,
        )

        # ── Joint states ──────────────────────────────────────────────────────
        self._joints: dict = {}
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._js_cb, 10, callback_group=self._cb)

        # ── MoveIt planning client ────────────────────────────────────────────
        self._plan_cli = self.create_client(
            GetMotionPlan, "/plan_kinematic_path", callback_group=self._cb)

        # ── Direct FJT client (bypasses MoveIt path tolerance) ────────────────
        self._fjt_cli = ActionClient(
            self, FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self._cb)

        # ── Kick off calibration after a brief warm-up ────────────────────────
        self.create_timer(3.0, self._start_calibration, callback_group=self._cb)
        self._started = False
        self.get_logger().info(
            f"\n═══ Camera Offset Calibrator ═══\n"
            f"  Mode : {self._mode}\n"
            f"  Warming up TF / camera feed for 3 s …")

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._joints[name] = pos

    def _camera_cb(self, msg: PoseStamped):
        with self._camera_lock:
            self._camera_latest = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])

    # ──────────────────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────────────────

    def _start_calibration(self):
        if self._started:
            return
        self._started = True

        mode = self._mode.lower()
        if mode == "arm":
            self._run_arm_mode()
        elif mode == "manual":
            self._run_manual_mode()
        elif mode == "known":
            self._run_known_mode()
        else:
            self.get_logger().error(
                f"Unknown mode '{mode}'.  Use arm | manual | known.")
        rclpy.shutdown()

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 1 — ARM (MoveIt)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_arm_mode(self):
        """
        Move assembly_tip to N hover heights above the board center.
        At each height, read assembly_tip TF (ground truth) and camera output.
        """
        self.get_logger().info("\n── ARM MODE ──────────────────────────────────────────")
        self.get_logger().info(
            f"  Board center: ({self._board_cx:.3f}, {self._board_cy:.3f})\n"
            f"  Will hover at {self.get_parameter('calibration_poses').value} heights "
            f"between {self.get_parameter('hover_z_min').value:.2f} m "
            f"and {self.get_parameter('hover_z_max').value:.2f} m above table.")

        if not self._wait_for_camera(timeout=10.0):
            self.get_logger().error("  combine_cameras is not publishing. Abort.")
            return

        n_poses    = self.get_parameter("calibration_poses").value
        z_min      = self.get_parameter("hover_z_min").value
        z_max      = self.get_parameter("hover_z_max").value
        settle     = self.get_parameter("settle_time").value
        n_samples  = self.get_parameter("samples_per_pose").value

        hover_heights = np.linspace(z_min, z_max, n_poses)
        residuals = []

        for idx, dz in enumerate(hover_heights):
            hover_z = self._table_z + dz
            self.get_logger().info(
                f"\n  [Pose {idx+1}/{n_poses}] Moving tip to "
                f"({self._board_cx:.3f}, {self._board_cy:.3f}, z={hover_z:.3f})")

            ok = self._move_tip_to(self._board_cx, self._board_cy, hover_z)
            if not ok:
                self.get_logger().warn(
                    f"  Move to pose {idx+1} failed — skipping.")
                continue

            self.get_logger().info(f"  Settling for {settle:.1f} s …")
            time.sleep(settle)

            # Ground truth: TF position of assembly_tip
            tip_world = self._get_tip_position()
            if tip_world is None:
                self.get_logger().warn("  TF lookup failed — skipping pose.")
                continue

            # Camera measurements — average n_samples readings
            cam_samples = []
            t_end = time.time() + (n_samples * 0.15)
            while time.time() < t_end:
                with self._camera_lock:
                    v = self._camera_latest
                if v is not None:
                    cam_samples.append(v.copy())
                time.sleep(0.12)

            if len(cam_samples) < 3:
                self.get_logger().warn(
                    f"  Only {len(cam_samples)} camera readings — skipping pose.")
                continue

            cam_mean, cam_std = _mean_and_std(cam_samples)

            # Ground truth XY = tip XY (tip is directly above board center)
            # Ground truth Z  = table_z + marker_height (board surface)
            truth = np.array([
                tip_world[0],
                tip_world[1],
                self._table_z + self._marker_h,
            ])

            residual = truth - cam_mean
            residuals.append(residual)

            self.get_logger().info(
                f"  TF truth:   ({truth[0]:.4f}, {truth[1]:.4f}, {truth[2]:.4f})\n"
                f"  Camera avg: ({cam_mean[0]:.4f}, {cam_mean[1]:.4f}, {cam_mean[2]:.4f})  "
                f"std=({cam_std[0]:.4f}, {cam_std[1]:.4f}, {cam_std[2]:.4f})\n"
                f"  Residual:   ({residual[0]:+.4f}, {residual[1]:+.4f}, {residual[2]:+.4f})")

        if not residuals:
            self.get_logger().error("  No valid measurements collected.")
            return

        self._compute_and_report(residuals)

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 2 — MANUAL
    # ──────────────────────────────────────────────────────────────────────────

    def _run_manual_mode(self):
        """
        User jogs arm so assembly_tip is directly above board center.
        Press Enter at each position to capture a measurement.
        """
        self.get_logger().info("\n── MANUAL MODE ───────────────────────────────────────")
        n = self.get_parameter("manual_samples").value
        self.get_logger().info(
            f"  Collect {n} measurements.\n"
            f"  At each: position assembly_tip DIRECTLY ABOVE the board center,\n"
            f"           then press Enter.")

        if not self._wait_for_camera(timeout=15.0):
            self.get_logger().error("  combine_cameras is not publishing. Abort.")
            return

        residuals = []
        for i in range(n):
            print(f"\n  [{i+1}/{n}] Hover tip directly above board center, then press Enter …",
                  flush=True)
            try:
                input()
            except EOFError:
                break

            tip_world = self._get_tip_position()
            if tip_world is None:
                print("  TF lookup failed — skipping.")
                continue

            # Collect 3 s of camera readings
            cam_samples = []
            t_end = time.time() + 3.0
            while time.time() < t_end:
                with self._camera_lock:
                    v = self._camera_latest
                if v is not None:
                    cam_samples.append(v.copy())
                time.sleep(0.12)

            if len(cam_samples) < 3:
                print("  Not enough camera readings — is combine_cameras running?")
                continue

            cam_mean, cam_std = _mean_and_std(cam_samples)
            truth = np.array([tip_world[0], tip_world[1], self._table_z + self._marker_h])
            residual = truth - cam_mean
            residuals.append(residual)

            print(
                f"  TF truth:   ({truth[0]:.4f}, {truth[1]:.4f}, {truth[2]:.4f})\n"
                f"  Camera avg: ({cam_mean[0]:.4f}, {cam_mean[1]:.4f}, {cam_mean[2]:.4f})\n"
                f"  Residual:   ({residual[0]:+.4f}, {residual[1]:+.4f}, {residual[2]:+.4f})")

        if not residuals:
            self.get_logger().error("  No valid measurements collected.")
            return

        self._compute_and_report(residuals)

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 3 — KNOWN POSITION
    # ──────────────────────────────────────────────────────────────────────────

    def _run_known_mode(self):
        """
        Compare combine_cameras output against a user-supplied ground truth.
        Requires no arm movement.
        """
        kx = self.get_parameter("known_x").value
        ky = self.get_parameter("known_y").value
        kz = self.get_parameter("known_z").value
        avg_sec = self.get_parameter("average_seconds").value

        self.get_logger().info(
            f"\n── KNOWN-POSITION MODE ───────────────────────────────\n"
            f"  Ground truth: ({kx:.4f}, {ky:.4f}, {kz:.4f})\n"
            f"  Averaging camera for {avg_sec:.0f} s …"
            f"  (Make sure ArUco board is in camera view)")

        if not self._wait_for_camera(timeout=15.0):
            self.get_logger().error("  combine_cameras not publishing. Abort.")
            return

        cam_samples = []
        t_end = time.time() + avg_sec
        while time.time() < t_end:
            with self._camera_lock:
                v = self._camera_latest
            if v is not None:
                cam_samples.append(v.copy())
            time.sleep(0.1)
            remaining = t_end - time.time()
            if int(remaining) % 5 == 0:
                self.get_logger().info(
                    f"  … {remaining:.0f} s remaining, {len(cam_samples)} samples so far",
                    throttle_duration_sec=4.5)

        if len(cam_samples) < 5:
            self.get_logger().error("  Too few camera samples. Is the board visible?")
            return

        cam_mean, cam_std = _mean_and_std(cam_samples)
        truth    = np.array([kx, ky, kz])
        residual = truth - cam_mean

        self.get_logger().info(
            f"  Ground truth: ({truth[0]:.4f}, {truth[1]:.4f}, {truth[2]:.4f})\n"
            f"  Camera mean:  ({cam_mean[0]:.4f}, {cam_mean[1]:.4f}, {cam_mean[2]:.4f})  "
            f"std=({cam_std[0]:.4f}, {cam_std[1]:.4f}, {cam_std[2]:.4f})\n"
            f"  Residual:     ({residual[0]:+.4f}, {residual[1]:+.4f}, {residual[2]:+.4f})")

        self._compute_and_report([residual])

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration result computation and reporting
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_and_report(self, residuals: list):
        """
        Given a list of residual vectors (truth − camera), compute the
        incremental correction to apply to combine_cameras offsets, then
        report results and save/apply them.

        new_x_offset = current_x_offset + mean(residual_x)
        """
        res_arr   = np.array(residuals)       # shape (N, 3)
        mean_res  = res_arr.mean(axis=0)
        std_res   = res_arr.std(axis=0)
        n         = len(residuals)

        new_offsets = self._cur_offsets + mean_res

        # 95% confidence interval (±2σ / √N)
        ci = 2.0 * std_res / max(math.sqrt(n), 1.0)

        sep = "═" * 70
        self.get_logger().info(f"\n{sep}")
        self.get_logger().info("  CALIBRATION RESULT")
        self.get_logger().info(sep)
        self.get_logger().info(
            f"  Measurements    : {n}\n"
            f"\n"
            f"  Current offsets : x={self._cur_offsets[0]:+.4f}  "
            f"y={self._cur_offsets[1]:+.4f}  z={self._cur_offsets[2]:+.4f}\n"
            f"  Mean residual   : x={mean_res[0]:+.4f}  "
            f"y={mean_res[1]:+.4f}  z={mean_res[2]:+.4f}\n"
            f"  Residual std    : x={std_res[0]:.4f}   "
            f"y={std_res[1]:.4f}   z={std_res[2]:.4f}\n"
            f"  95% CI (±)      : x={ci[0]:.4f}   y={ci[1]:.4f}   z={ci[2]:.4f}\n"
            f"\n"
            f"  ★ NEW OFFSETS (apply these to combine_cameras.py):\n"
            f"      x_offset = {new_offsets[0]:+.5f} m\n"
            f"      y_offset = {new_offsets[1]:+.5f} m\n"
            f"      z_offset = {new_offsets[2]:+.5f} m")

        if n < 3:
            self.get_logger().warn(
                "  WARN: Fewer than 3 measurements — result may be unreliable.\n"
                "        Collect more data points for confidence.")
        if np.any(ci > 0.010):
            self.get_logger().warn(
                f"  WARN: 95% CI > 1 cm on some axis — high variability.\n"
                f"        Check that ArUco board is stationary and well-lit.")

        self.get_logger().info(f"\n{sep}")
        self.get_logger().info("  APPLY IMMEDIATELY (copy-paste into your terminal):")
        self.get_logger().info(sep)
        self.get_logger().info(
            f"  ros2 param set /combine_cameras_node x_offset {new_offsets[0]:.5f}\n"
            f"  ros2 param set /combine_cameras_node y_offset {new_offsets[1]:.5f}\n"
            f"  ros2 param set /combine_cameras_node z_offset {new_offsets[2]:.5f}")

        self.get_logger().info(f"\n{sep}")
        self.get_logger().info("  PERSIST IN LAUNCH FILE (add these --ros-args):")
        self.get_logger().info(sep)
        self.get_logger().info(
            f"  -p x_offset:={new_offsets[0]:.5f} "
            f"-p y_offset:={new_offsets[1]:.5f} "
            f"-p z_offset:={new_offsets[2]:.5f}")

        # Save YAML
        yaml_path = self.get_parameter("output_yaml").value
        calib_data = {
            "combine_cameras_calibration": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": self._mode,
                "n_measurements": n,
                "x_offset": float(new_offsets[0]),
                "y_offset": float(new_offsets[1]),
                "z_offset": float(new_offsets[2]),
                "residual_mean": {
                    "x": float(mean_res[0]),
                    "y": float(mean_res[1]),
                    "z": float(mean_res[2]),
                },
                "residual_std": {
                    "x": float(std_res[0]),
                    "y": float(std_res[1]),
                    "z": float(std_res[2]),
                },
                "95pct_ci": {
                    "x": float(ci[0]),
                    "y": float(ci[1]),
                    "z": float(ci[2]),
                },
            }
        }
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(calib_data, f, default_flow_style=False)
            self.get_logger().info(
                f"\n  Saved to: {yaml_path}")
        except Exception as e:
            self.get_logger().warn(f"  Could not save YAML: {e}")

        # Apply live via ros2 param set (subprocess)
        if self.get_parameter("apply_live").value:
            self.get_logger().info("\n  Applying offsets to /combine_cameras_node …")
            import subprocess
            for axis, val in zip(["x", "y", "z"], new_offsets):
                cmd = [
                    "ros2", "param", "set",
                    "/combine_cameras_node",
                    f"{axis}_offset",
                    str(round(float(val), 6)),
                ]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
                    if result.returncode == 0:
                        self.get_logger().info(f"  ✓  {axis}_offset applied.")
                    else:
                        self.get_logger().warn(
                            f"  ✗  {axis}_offset: {result.stderr.strip()}")
                except Exception as e:
                    self.get_logger().warn(f"  ✗  {axis}_offset subprocess error: {e}")

        self.get_logger().info(f"\n{sep}")
        self.get_logger().info("  Calibration complete.")
        self.get_logger().info(sep)

    # ──────────────────────────────────────────────────────────────────────────
    # TF helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_tip_position(self, timeout_sec: float = 3.0):
        """Return world-frame position of assembly_tip as np.array([x,y,z])."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self._world, self._tip_link, rclpy.time.Time())
                t = tf.transform.translation
                return np.array([t.x, t.y, t.z])
            except Exception:
                time.sleep(0.05)
        self.get_logger().warn(
            f"  TF lookup world→{self._tip_link} timed out after {timeout_sec:.1f} s.")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Camera wait helper
    # ──────────────────────────────────────────────────────────────────────────

    def _wait_for_camera(self, timeout: float = 10.0) -> bool:
        """Block until at least one /fused_marker_square_center message arrives."""
        self.get_logger().info("  Waiting for /fused_marker_square_center …")
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._camera_lock:
                if self._camera_latest is not None:
                    self.get_logger().info("  Camera feed confirmed.")
                    return True
            time.sleep(0.2)
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # MoveIt helpers (ARM mode)
    # ──────────────────────────────────────────────────────────────────────────

    def _wait_for_future(self, future, timeout_sec: float):
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.05)
        return future.result()

    def _move_tip_to(self, cx: float, cy: float, tip_z: float) -> bool:
        """
        Plan and execute: move assembly_tip to world (cx, cy, tip_z) with
        tip-down orientation using Pilz PTP, falling back to OMPL.
        """
        qx = self.get_parameter("pen_down_qx").value
        qy = self.get_parameter("pen_down_qy").value
        qz = self.get_parameter("pen_down_qz").value
        qw = self.get_parameter("pen_down_qw").value

        # Compute EE origin from tip target using current TF offset
        ee_offset = self._get_ee_tip_offset()
        if ee_offset is None:
            self.get_logger().warn("  Cannot get EE→tip offset via TF.")
            return False

        ee_x = cx + ee_offset[0]
        ee_y = cy + ee_offset[1]
        ee_z = tip_z + ee_offset[2]

        from geometry_msgs.msg import Quaternion
        pen_q = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

        # Try Pilz PTP first
        req = self._build_ptp_req(ee_x, ee_y, ee_z, pen_q)
        ok, traj = self._call_plan(req)

        if not ok:
            self.get_logger().warn("  Pilz PTP failed — trying OMPL …")
            req = self._build_ompl_req(ee_x, ee_y, ee_z, pen_q)
            ok, traj = self._call_plan(req)

        if not ok or traj is None:
            self.get_logger().error("  Planning failed.")
            return False

        return self._execute_traj(traj)

    def _get_ee_tip_offset(self, timeout: float = 3.0):
        """Return world-frame vector from assembly_tip to ee_link origin."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                tip_tf = self.tf_buffer.lookup_transform(
                    self._world, self._tip_link, rclpy.time.Time())
                ee_tf  = self.tf_buffer.lookup_transform(
                    self._world, self._ee_link,  rclpy.time.Time())
                tip_t = tip_tf.transform.translation
                ee_t  = ee_tf.transform.translation
                return np.array([
                    ee_t.x - tip_t.x,
                    ee_t.y - tip_t.y,
                    ee_t.z - tip_t.z,
                ])
            except Exception:
                time.sleep(0.05)
        return None

    def _build_workspace(self):
        ws = WorkspaceParameters()
        ws.header.frame_id = self._world
        ws.min_corner.x = -1.1;  ws.min_corner.y = -1.1;  ws.min_corner.z = -0.1
        ws.max_corner.x =  1.1;  ws.max_corner.y =  1.1;  ws.max_corner.z =  1.2
        return ws

    def _build_ptp_req(self, ee_x, ee_y, ee_z, pen_q):
        from geometry_msgs.msg import Pose
        req = MotionPlanRequest()
        req.group_name   = self._group
        req.planner_id   = "PTP"
        req.pipeline_id  = "pilz_industrial_motion_planner"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor     = self._vel
        req.max_acceleration_scaling_factor = self._vel * 0.3
        req.workspace_parameters = self._build_workspace()
        req.start_state.is_diff  = True

        sphere = SolidPrimitive(); sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.005]
        from geometry_msgs.msg import Pose
        bv_pose = Pose()
        bv_pose.position.x  = ee_x; bv_pose.position.y = ee_y; bv_pose.position.z = ee_z
        bv_pose.orientation = pen_q
        bv = BoundingVolume(); bv.primitives.append(sphere); bv.primitive_poses.append(bv_pose)

        pos_c = PositionConstraint()
        pos_c.header.frame_id = self._world; pos_c.link_name = self._ee_link
        pos_c.constraint_region = bv; pos_c.weight = 1.0

        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self._world; ori_c.link_name = self._ee_link
        ori_c.orientation = pen_q
        ori_c.absolute_x_axis_tolerance = 0.05
        ori_c.absolute_y_axis_tolerance = 0.05
        ori_c.absolute_z_axis_tolerance = 0.05
        ori_c.weight = 1.0

        gc = Constraints()
        gc.position_constraints.append(pos_c)
        gc.orientation_constraints.append(ori_c)
        req.goal_constraints.append(gc)
        return req

    def _build_ompl_req(self, ee_x, ee_y, ee_z, pen_q):
        from geometry_msgs.msg import Pose
        req = MotionPlanRequest()
        req.group_name   = self._group
        req.planner_id   = "RRTConnectkConfigDefault"
        req.pipeline_id  = "ompl"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 30.0
        req.max_velocity_scaling_factor     = self._vel
        req.max_acceleration_scaling_factor = self._vel * 0.3
        req.workspace_parameters = self._build_workspace()
        req.start_state.is_diff  = True

        sphere = SolidPrimitive(); sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.02]
        from geometry_msgs.msg import Pose
        bv_pose = Pose()
        bv_pose.position.x = ee_x; bv_pose.position.y = ee_y; bv_pose.position.z = ee_z
        bv_pose.orientation.w = 1.0
        bv = BoundingVolume(); bv.primitives.append(sphere); bv.primitive_poses.append(bv_pose)

        pos_c = PositionConstraint()
        pos_c.header.frame_id = self._world; pos_c.link_name = self._ee_link
        pos_c.constraint_region = bv; pos_c.weight = 1.0

        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self._world; ori_c.link_name = self._ee_link
        ori_c.orientation = pen_q
        ori_c.absolute_x_axis_tolerance = 0.3
        ori_c.absolute_y_axis_tolerance = 0.3
        ori_c.absolute_z_axis_tolerance = 0.3
        ori_c.weight = 1.0

        gc = Constraints()
        gc.position_constraints.append(pos_c)
        gc.orientation_constraints.append(ori_c)
        req.goal_constraints.append(gc)
        return req

    def _call_plan(self, motion_req: MotionPlanRequest):
        if not self._plan_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("  /plan_kinematic_path not available.")
            return False, None
        svc_req = GetMotionPlan.Request()
        svc_req.motion_plan_request = motion_req
        f = self._plan_cli.call_async(svc_req)
        res = self._wait_for_future(f, 90.0)
        if res is None:
            return False, None
        resp = res.motion_plan_response
        if resp.error_code.val == 1:
            return True, resp.trajectory
        return False, None

    def _execute_traj(self, traj: RobotTrajectory,
                      timeout_sec: float = 60.0) -> bool:
        """Clamp and execute via direct FJT."""
        self._clamp_traj(traj)
        deadline_srv = time.time() + 5.0
        while not self._fjt_cli.server_is_ready() and time.time() < deadline_srv:
            time.sleep(0.05)
        if not self._fjt_cli.server_is_ready():
            self.get_logger().error("  FJT server not ready.")
            return False

        now_ns   = self.get_clock().now().nanoseconds
        start_ns = now_ns + int(0.3e9)
        traj.joint_trajectory.header.stamp.sec     = start_ns // 1_000_000_000
        traj.joint_trajectory.header.stamp.nanosec = start_ns %  1_000_000_000

        names = traj.joint_trajectory.joint_names
        fjt = FollowJointTrajectory.Goal()
        fjt.trajectory = traj.joint_trajectory
        for name in names:
            pt = JointTolerance(); pt.name = name; pt.position = 5.0
            fjt.path_tolerance.append(pt)
            gt = JointTolerance(); gt.name = name; gt.position = 0.05
            fjt.goal_tolerance.append(gt)
        fjt.goal_time_tolerance = RosDuration(sec=30, nanosec=0)

        f  = self._fjt_cli.send_goal_async(fjt)
        gh = self._wait_for_future(f, 10.0)
        if gh is None or not gh.accepted:
            return False
        rf = gh.get_result_async()
        res = self._wait_for_future(rf, timeout_sec)
        if res is None:
            return False
        return res.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

    def _clamp_traj(self, traj: RobotTrajectory):
        pts = traj.joint_trajectory.points
        if len(pts) < 2:
            return
        n = len(pts[0].positions) if pts[0].positions else 0
        times_ns = [
            pt.time_from_start.sec * 1_000_000_000 + pt.time_from_start.nanosec
            for pt in pts
        ]
        for i in range(1, len(pts)):
            min_dt = 0
            for j in range(n):
                if j >= len(pts[i].positions) or j >= len(pts[i-1].positions):
                    continue
                disp = abs(pts[i].positions[j] - pts[i-1].positions[j])
                if disp > 1e-9:
                    min_dt = max(min_dt, int(disp / _MAX_JOINT_VEL_RAD_S * 1e9) + 1)
            cur_dt = times_ns[i] - times_ns[i-1]
            if cur_dt < min_dt:
                stretch = min_dt - cur_dt
                for k in range(i, len(pts)):
                    times_ns[k] += stretch
        for i, pt in enumerate(pts):
            pt.time_from_start.sec     = times_ns[i] // 1_000_000_000
            pt.time_from_start.nanosec = times_ns[i] %  1_000_000_000


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = CameraOffsetCalibrator()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
