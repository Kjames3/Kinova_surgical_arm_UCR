#!/usr/bin/env python3
"""
Unified Automated Hand-Eye Calibration Script for Kinova Gen3 7DOF

This script automatically calibrates camera frames (Intel RealSense, OAK-D,
or Kinova Wrist Camera) relative to the robot base frame (base_link / world).
It supports:
  1. Eye-in-Hand Calibration (Moving camera on wrist, static marker on table)
  2. Eye-to-Hand Calibration (Static camera, moving marker on gripper)

The safe default is supervised manual capture: the operator positions the arm
and explicitly accepts each sample. A legacy MoveIt pose sweep is available
only behind an explicit automatic-motion acknowledgement. The script solves
with OpenCV hand-eye algorithms, rejects inconsistent results, and exports a
candidate static transform for independent validation.

Author: Antigravity AI
Date: 2026-06-02
"""

import sys
import os
import threading
import time
import argparse
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, JointState
from geometry_msgs.msg import Pose, Point, Quaternion
import tf2_ros

# Pre-defined joint poses (in radians) designed for eye-in-hand calibration.
EYE_IN_HAND_POSES = [
    [-0.0362,  0.6621, -1.6009, -0.0051, -3.1087, -0.7185,  1.5692],
    [ 0.1140,  0.6758, -1.5740, -0.0140, -3.1302, -0.7321,  1.7096],
    [-0.3068,  0.7043, -1.5116,  0.0081, -3.0695, -0.7722,  1.3172],
    [-0.2472,  0.9510, -0.9866, -0.0066, -3.0724, -1.0487,  1.3724],
    [-0.2803,  0.7136, -1.2297,  0.0046, -3.0699, -1.0437,  1.3454],
    [-0.0631,  0.6741, -1.3071, -0.0160, -3.0865, -1.0007,  1.5613],
    [-0.0630,  0.6901, -1.3319, -0.0167, -3.0872, -0.9599,  1.5617],
    [ 0.2117,  0.7650, -1.4222, -0.0484, -3.1144, -0.7973,  1.8393],
    [-0.0841,  0.3948, -2.0248, -0.0180, -3.0784, -0.5622,  1.5611],
    [-0.2344,  0.3807, -2.0491,  0.0085, -3.0627, -0.5549,  1.4040],
]

# Pre-defined joint poses (in radians) designed for eye-to-hand calibration.
EYE_TO_HAND_POSES = [
    [-0.0362,  0.6621, -3.1087, -1.6009, -0.0051, -0.7185,  1.5692],
    [ 0.1140,  0.6758, -3.1302, -1.5740, -0.0140, -0.7321,  1.7096],
    [-0.3068,  0.7043, -3.0695, -1.5116,  0.0081, -0.7722,  1.3172],
    [-0.2472,  0.9510, -3.0724, -0.9866, -0.0066, -1.0487,  1.3724],
    [-0.2803,  0.7136, -3.0699, -1.2297,  0.0046, -1.0437,  1.3454],
    [-0.0631,  0.6741, -3.0865, -1.3071, -0.0160, -1.0007,  1.5613],
    [-0.0630,  0.6901, -3.0872, -1.3319, -0.0167, -0.9599,  1.5617],
    [ 0.2117,  0.7650, -3.1144, -1.4222, -0.0484, -0.7973,  1.8393],
    [-0.0841,  0.3948, -3.0784, -2.0248, -0.0180, -0.5622,  1.5611],
    [-0.2344,  0.3807, -3.0627, -2.0491,  0.0085, -0.5549,  1.4040],
]

# Conservative bounded-joint limits used by this surgical-arm MoveIt package.
# Joints 1, 3, 5, and 7 are continuous. Pilz can return a trajectory endpoint
# outside these overrides when its pipeline-specific limits disagree with the
# robot model, so plan-only validates every endpoint independently.
BOUNDED_JOINT_LIMITS = {
    "joint_2": (-2.2689, 2.2689),
    "joint_4": (-2.4435, 2.4435),
    "joint_6": (-2.0944, 2.0944),
}
JOINT_LIMIT_MARGIN_RAD = 0.05
CONTINUOUS_JOINTS = {"joint_1", "joint_3", "joint_5", "joint_7"}
RETURN_POSITION_TOLERANCE_RAD = 0.02


def _quat_to_matrix(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("zero-length quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _matrix_to_quat(R_mat):
    """Convert a proper rotation matrix to normalized [x, y, z, w]."""
    M = np.asarray(R_mat, dtype=float)
    q = np.empty(4)
    trace = np.trace(M)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q[:] = [(M[2, 1] - M[1, 2]) / s,
                (M[0, 2] - M[2, 0]) / s,
                (M[1, 0] - M[0, 1]) / s,
                0.25 * s]
    else:
        i = int(np.argmax(np.diag(M)))
        if i == 0:
            s = 2.0 * np.sqrt(max(0.0, 1.0 + M[0, 0] - M[1, 1] - M[2, 2]))
            q[:] = [0.25 * s,
                    (M[0, 1] + M[1, 0]) / s,
                    (M[0, 2] + M[2, 0]) / s,
                    (M[2, 1] - M[1, 2]) / s]
        elif i == 1:
            s = 2.0 * np.sqrt(max(0.0, 1.0 + M[1, 1] - M[0, 0] - M[2, 2]))
            q[:] = [(M[0, 1] + M[1, 0]) / s,
                    0.25 * s,
                    (M[1, 2] + M[2, 1]) / s,
                    (M[0, 2] - M[2, 0]) / s]
        else:
            s = 2.0 * np.sqrt(max(0.0, 1.0 + M[2, 2] - M[0, 0] - M[1, 1]))
            q[:] = [(M[0, 2] + M[2, 0]) / s,
                    (M[1, 2] + M[2, 1]) / s,
                    0.25 * s,
                    (M[1, 0] - M[0, 1]) / s]
    return q / np.linalg.norm(q)


def _rotation_angle(R_mat):
    value = (np.trace(np.asarray(R_mat, dtype=float)) - 1.0) / 2.0
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def _matrix_to_euler_xyz(R_mat):
    """Return intrinsic XYZ Euler angles in radians (for reporting only)."""
    M = np.asarray(R_mat, dtype=float)
    sy = np.hypot(M[0, 0], M[1, 0])
    if sy > 1e-9:
        return np.array([np.arctan2(M[2, 1], M[2, 2]),
                         np.arctan2(-M[2, 0], sy),
                         np.arctan2(M[1, 0], M[0, 0])])
    return np.array([np.arctan2(-M[1, 2], M[1, 1]),
                     np.arctan2(-M[2, 0], sy), 0.0])


def _safe_destroy_windows():
    """Close OpenCV windows when HighGUI exists; no-op on headless builds."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

class UnifiedCameraCalibrationNode(Node):
    def __init__(self, args):
        super().__init__("unified_camera_calibration_node")
        self.args = args
        self.cb_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        
        self.latest_frame = None
        self._image_sequence = 0
        self._image_rx_monotonic = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.detected_camera_frame = None
        
        # TF2 listener to query robot pose (base_link to effector frame)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Resolve Topic Mapping based on selected camera profile
        self.resolve_camera_profile()
        
        # Setup ArUco detector
        try:
            dict_id = getattr(cv2.aruco, self.args.aruco_dict)
        except AttributeError:
            self.get_logger().error(f"Invalid ArUco dictionary: {self.args.aruco_dict}. Defaulting to DICT_4X4_50.")
            dict_id = cv2.aruco.DICT_4X4_50
            
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        
        # Define 3D coordinate points of a single marker in its local frame.
        msize = self.args.marker_size
        self.marker_obj_pts = np.array([
            [-msize/2,  msize/2, 0],
            [ msize/2,  msize/2, 0],
            [ msize/2, -msize/2, 0],
            [-msize/2, -msize/2, 0]
        ], dtype=np.float32)

        # Optional planar grid target. The IDs are supplied in physical
        # row-major order (top-left to bottom-right), because generated boards
        # do not necessarily number markers in row-major order. Keeping the
        # board frame at its geometric centre makes the target pose independent
        # of which subset of markers is visible.
        self.grid_obj_pts = {}
        if self.args.target_type == "grid":
            grid_ids = [
                int(x.strip()) for x in self.args.grid_marker_ids.split(",")
                if x.strip()
            ]
            expected = self.args.grid_cols * self.args.grid_rows
            if len(grid_ids) != expected:
                raise ValueError(
                    f"--grid-marker-ids must contain exactly {expected} IDs "
                    f"for a {self.args.grid_cols}x{self.args.grid_rows} board")
            if len(set(grid_ids)) != len(grid_ids):
                raise ValueError("--grid-marker-ids contains duplicate IDs")

            pitch = msize + self.args.marker_gap
            board_width = self.args.grid_cols * msize + (
                self.args.grid_cols - 1) * self.args.marker_gap
            board_height = self.args.grid_rows * msize + (
                self.args.grid_rows - 1) * self.args.marker_gap
            for index, marker_id in enumerate(grid_ids):
                row, col = divmod(index, self.args.grid_cols)
                cx = -board_width / 2.0 + msize / 2.0 + col * pitch
                cy = board_height / 2.0 - msize / 2.0 - row * pitch
                points = np.array([
                    [cx - msize/2, cy + msize/2, 0],
                    [cx + msize/2, cy + msize/2, 0],
                    [cx + msize/2, cy - msize/2, 0],
                    [cx - msize/2, cy - msize/2, 0],
                ], dtype=np.float32)
                # OpenCV returns corners in decoded-marker orientation. This
                # shift maps them to physical board corners when every marker
                # was printed with a common quarter-turn on the sheet.
                self.grid_obj_pts[marker_id] = np.roll(
                    points, self.args.grid_corner_shift, axis=0)

            self.get_logger().info(
                f"Grid target: {self.args.grid_cols}x{self.args.grid_rows}, "
                f"{len(grid_ids)} IDs, size={board_width:.5f}x"
                f"{board_height:.5f} m, corner shift="
                f"{self.args.grid_corner_shift}")

        # Build offset + rotation table for multi-marker calibration (from container board)
        Lx = self.args.rect_width
        Ly = self.args.rect_height
        _pos_in_ref = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([Lx,  0.0, 0.0]),
            2: np.array([Lx,  Ly,  0.0]),
            3: np.array([0.0, Ly,  0.0]),
        }
        Rz180 = np.array([[-1., 0., 0.],
                           [ 0., -1., 0.],
                           [ 0.,  0., 1.]])
        rotated_ids = {int(x.strip()) for x in self.args.rotated_180_ids.split(",") if x.strip()}
        self.R_ref2m = {
            mid: Rz180.copy() if mid in rotated_ids else np.eye(3)
            for mid in _pos_in_ref
        }
        ref_pos = _pos_in_ref.get(self.args.marker_id, np.zeros(3))
        self.offset_to_ref = {
            mid: self.R_ref2m[mid] @ (ref_pos - pos)
            for mid, pos in _pos_in_ref.items()
        }
        self.all_marker_ids = [int(x.strip()) for x in self.args.all_marker_ids.split(",")]
        
        # Select target joint configurations depending on calibration type
        if self.args.mode == "eye-in-hand":
            self.poses = EYE_IN_HAND_POSES
        else:
            self.poses = EYE_TO_HAND_POSES
            
        # Manual capture is the safe default: the operator positions the arm and
        # explicitly accepts each sample.  Construct a motion client only when
        # automatic motion was explicitly requested.
        self.moveit2 = None
        if self.args.capture_mode == "automatic":
            from pymoveit2 import MoveIt2, MoveIt2State
            self.get_logger().info("Initializing MoveIt2 arm controller...")
            joint_names = [f"joint_{i}" for i in range(1, 8)]
            self.moveit2 = MoveIt2(
                node=self,
                joint_names=joint_names,
                base_link_name=self.args.robot_base_frame,
                end_effector_name="end_effector_link",
                group_name="manipulator",
                callback_group=self.cb_group
            )
            # Planner IDs are pipeline-specific. Without pipeline_id, MoveIt
            # sends "PTP" to its default OMPL pipeline, which has no such
            # planner and silently falls back to a short RRTConnect attempt.
            self.moveit2.pipeline_id = "pilz_industrial_motion_planner"
            self.moveit2.planner_id = "PTP"
            self.moveit2.allowed_planning_time = 3.0
            self.moveit2.max_velocity = self.args.max_vel
            self.moveit2.max_acceleration = self.args.max_accel
            self._moveit2_idle_state = MoveIt2State.IDLE
            self._moveit2_executing_state = MoveIt2State.EXECUTING
        
        # Setup ROS 2 topic subscriptions
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Track receipt time independently of the message header. A dead
        # ros2_control process leaves the last JointState cached in pymoveit2,
        # and TF can likewise remain queryable at its last value. Monotonic
        # receipt time lets automatic mode distinguish live feedback from a
        # plausible-looking stale state without relying on clock sync.
        self._joint_state_rx_monotonic = None
        if self.args.capture_mode == "automatic":
            self.create_subscription(
                JointState,
                "/joint_states",
                self._joint_state_cb,
                camera_qos,
                callback_group=self.cb_group,
            )
        
        self.get_logger().info(
            f"Subscribing to image topic: {self.image_topic} "
            f"({self.image_transport} transport)")
        self.create_subscription(
            CompressedImage if self.image_transport == "compressed" else Image,
            self.image_topic,
            (self._compressed_image_cb if self.image_transport == "compressed"
             else self._image_cb),
            camera_qos,
            callback_group=self.cb_group
        )
        
        self.get_logger().info(f"Subscribing to camera info: {self.info_topic}")
        self.create_subscription(
            CameraInfo,
            self.info_topic,
            self._info_cb,
            camera_qos,
            callback_group=self.cb_group
        )
        
        # Calibration storage lists
        self.R_g2b = []  # Gripper to Base rotations
        self.t_g2b = []  # Gripper to Base translations
        self.R_t2c = []  # Target to Camera rotations
        self.t_t2c = []  # Target to Camera translations
        self.sample_count = 0
        self._automatic_motion_started = False
        self._automatic_return_allowed = True
        self._last_motion_execution_started = False
        
        # TF2 Broadcaster for live preview visualization in RViz
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        os.makedirs(self.args.save_dir, exist_ok=True)
        self.get_logger().info(f"Unified Calibration Node Ready (Mode: {self.args.mode.upper()}).")

    def resolve_camera_profile(self):
        """Map camera name to topic names and standard camera frames."""
        cam = self.args.camera.lower()
        if cam == "realsense":
            self.image_topic = "/realsense/camera/color/image_raw"
            self.info_topic = "/realsense/camera/color/camera_info"
            self.default_camera_frame = "realsense_camera_color_optical_frame"
        elif cam == "oakd":
            self.image_topic = "/global_camera/color/image_raw"
            self.info_topic = "/global_camera/color/camera_info"
            self.default_camera_frame = "global_camera_link"
        elif cam == "kinova":
            self.image_topic = "/camera/color/image_raw"
            self.info_topic = "/camera/color/camera_info"
            self.default_camera_frame = "camera_color_frame"
        else:
            # Custom profile
            self.image_topic = self.args.custom_image_topic
            self.info_topic = self.args.custom_info_topic
            self.default_camera_frame = self.args.custom_camera_frame
            if not self.image_topic or not self.info_topic or not self.default_camera_frame:
                self.get_logger().error("Custom camera selected but custom topics/frames not fully specified! Exiting.")
                sys.exit(1)

        # Allow either REAL-1 launch layout to be selected without inventing a
        # fake camera profile.  In particular, cameras.launch.py uses the
        # /global_camera/global_camera namespace for the D435i.
        if self.args.image_topic:
            self.image_topic = self.args.image_topic
        if self.args.info_topic:
            self.info_topic = self.args.info_topic
        if self.args.camera_frame:
            self.default_camera_frame = self.args.camera_frame

        self.image_transport = self.args.image_transport
        if self.image_transport == "auto":
            # REAL-1 receives the D435i raw RGB frames in bursts even though
            # the camera itself is producing a steady stream. JPEG transport
            # is a stable 15 Hz and avoids stale image/robot-pose pairings.
            self.image_transport = (
                "compressed" if cam == "realsense" else "raw")
        if (self.image_transport == "compressed"
                and not self.image_topic.endswith("/compressed")):
            self.image_topic = self.image_topic.rstrip("/") + "/compressed"

    def _store_frame(self, frame):
        with self.lock:
            self.latest_frame = frame
            self._image_sequence += 1
            self._image_rx_monotonic = time.monotonic()

    def _image_cb(self, msg):
        try:
            # Avoid cv_bridge here.  REAL-1's user Python environment carries a
            # NumPy/OpenCV ABI combination that has previously made cv_bridge
            # crash the process rather than raise a Python exception.
            channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(msg.encoding)
            if channels is None:
                raise ValueError(
                    f"unsupported calibration image encoding '{msg.encoding}'")
            row_bytes = msg.width * channels
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.step)
            packed = raw[:, :row_bytes].reshape(msg.height, msg.width, channels)
            if msg.encoding == "rgb8":
                frame = packed[:, :, ::-1].copy()
            elif msg.encoding == "mono8":
                frame = cv2.cvtColor(packed[:, :, 0], cv2.COLOR_GRAY2BGR)
            else:
                frame = packed.copy()
            self._store_frame(frame)
        except Exception as e:
            self.get_logger().error(f"Image decode error: {str(e)}")

    def _compressed_image_cb(self, msg):
        try:
            encoded = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("OpenCV could not decode the compressed image")
            self._store_frame(frame)
        except Exception as e:
            self.get_logger().error(f"Compressed image decode error: {str(e)}")

    def _info_cb(self, msg):
        with self.lock:
            if self.camera_matrix is None:
                self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
                self.dist_coeffs = np.array(msg.d, dtype=np.float64)
                self.detected_camera_frame = msg.header.frame_id
                self.get_logger().info("Dynamically initialized camera intrinsics successfully!")
                self.get_logger().info(f"Intrinsic Matrix K:\n{self.camera_matrix}")
                self.get_logger().info(f"Distortion Coefficients D: {self.dist_coeffs}")
                self.get_logger().info(f"Detected Camera Frame ID: {self.detected_camera_frame}")

    def _joint_state_cb(self, _msg):
        self._joint_state_rx_monotonic = time.monotonic()

    def joint_state_feedback_is_fresh(self, context, log_error=True):
        """Require recently received feedback before any automatic action."""
        received_at = self._joint_state_rx_monotonic
        age = (float("inf") if received_at is None
               else time.monotonic() - received_at)
        if age <= self.args.max_joint_state_age:
            return True
        if log_error:
            detail = ("no JointState has been received"
                      if received_at is None
                      else f"latest JointState is {age:.3f} s old")
            self.get_logger().error(
                f"Automatic calibration stopped during {context}: {detail} "
                f"(limit {self.args.max_joint_state_age:.3f} s). The robot "
                "controller is unavailable or unhealthy; no motion will be "
                "requested.")
        return False

    def get_camera_parameters(self):
        with self.lock:
            if self.camera_matrix is not None:
                return self.camera_matrix.copy(), self.dist_coeffs.copy()
        return None, None

    def detect_marker(self, frame):
        """
        Detects ArUco marker from frame. Computes the offset to the reference marker
        if another marker on the board is detected.
        """
        K, D = self.get_camera_parameters()
        if K is None:
            vis = frame.copy()
            cv2.putText(vis, "Waiting for CameraInfo", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return False, None, None, vis
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is None:
            vis = frame.copy()
            cv2.putText(vis, "No markers found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return False, None, None, vis

        vis = frame.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        ids_flat = ids.flatten().tolist()

        if self.args.target_type == "grid":
            object_points = []
            image_points = []
            visible_ids = []
            for marker_corners, marker_id in zip(corners, ids_flat):
                if marker_id not in self.grid_obj_pts:
                    continue
                object_points.extend(self.grid_obj_pts[marker_id])
                image_points.extend(marker_corners.reshape(4, 2))
                visible_ids.append(marker_id)

            if len(visible_ids) < self.args.min_visible_markers:
                cv2.putText(
                    vis,
                    f"Grid: {len(visible_ids)}/{self.args.min_visible_markers} markers",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
                return False, None, None, vis

            object_points = np.asarray(object_points, dtype=np.float32)
            image_points = np.asarray(image_points, dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, K, D,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                cv2.putText(vis, "Grid pose solve failed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return False, None, None, vis

            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, K, D)
            errors = projected.reshape(-1, 2) - image_points
            reproj_rms = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
            if reproj_rms > self.args.max_reprojection_error_px:
                cv2.putText(
                    vis, f"Grid rejected: RMS {reproj_rms:.2f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
                return False, None, None, vis

            cv2.drawFrameAxes(vis, K, D, rvec, tvec, self.args.marker_size)
            cv2.putText(
                vis,
                f"Grid OK: {len(visible_ids)} tags, RMS {reproj_rms:.2f}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
            cam_frame = (self.detected_camera_frame
                         if self.detected_camera_frame
                         else self.default_camera_frame)
            R_target2cam, _ = cv2.Rodrigues(rvec)
            self.broadcast_matrix_as_tf(
                R_target2cam, tvec.flatten(), cam_frame,
                "live_calib_marker")
            return True, rvec, tvec, vis

        ref_id = self.args.marker_id
        search_order = [ref_id] + [m for m in self.all_marker_ids if m != ref_id]

        for target_id in search_order:
            if target_id not in ids_flat:
                continue
            i = ids_flat.index(target_id)
            img_pts = corners[i][0].astype(np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                self.marker_obj_pts, img_pts, K, D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue

            cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.05)

            if target_id == ref_id:
                label = f"Marker {target_id} (ref) OK"
                cv2.putText(vis, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Broadcast live detected marker pose in camera frame
                cam_frame = self.detected_camera_frame if self.detected_camera_frame else self.default_camera_frame
                R_m2c, _ = cv2.Rodrigues(rvec)
                self.broadcast_matrix_as_tf(R_m2c, tvec.flatten(), cam_frame, "live_calib_marker")
                
                # Broadcast live estimated camera pose in base frame (for eye-to-hand preview)
                if self.args.mode == "eye-to-hand":
                    ok_tf, rot_g2b, t_g2b = self.get_robot_pose()
                    if ok_tf:
                        T_g2b = np.eye(4)
                        T_g2b[:3, :3] = rot_g2b
                        T_g2b[:3, 3] = t_g2b
                        
                        T_m2c = np.eye(4)
                        T_m2c[:3, :3] = R_m2c
                        T_m2c[:3, 3] = tvec.flatten()
                        
                        T_c2m = np.linalg.inv(T_m2c)
                        # Assume marker is at end effector (T_marker_to_gripper = Identity)
                        T_c2b = T_g2b @ T_c2m
                        self.broadcast_matrix_as_tf(T_c2b[:3, :3], T_c2b[:3, 3], self.args.robot_base_frame, "live_estimated_camera")
                
                return True, rvec, tvec, vis

            # Compensate for offset of non-reference marker
            R_det2cam, _ = cv2.Rodrigues(rvec)
            offset = self.offset_to_ref.get(target_id, np.zeros(3))
            R_ref2m = self.R_ref2m.get(target_id, np.eye(3))
            t_ref_in_cam = (R_det2cam @ offset + tvec.flatten()).reshape(3, 1)
            R_ref2cam = R_det2cam @ R_ref2m
            rvec_ref, _ = cv2.Rodrigues(R_ref2cam)

            label = f"Marker {target_id}->ref({ref_id}) OK"
            cv2.putText(vis, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            
            # Broadcast live detected marker pose in camera frame
            cam_frame = self.detected_camera_frame if self.detected_camera_frame else self.default_camera_frame
            self.broadcast_matrix_as_tf(R_ref2cam, t_ref_in_cam.flatten(), cam_frame, "live_calib_marker")
            
            # Broadcast live estimated camera pose in base frame (for eye-to-hand preview)
            if self.args.mode == "eye-to-hand":
                ok_tf, rot_g2b, t_g2b = self.get_robot_pose()
                if ok_tf:
                    T_g2b = np.eye(4)
                    T_g2b[:3, :3] = rot_g2b
                    T_g2b[:3, 3] = t_g2b
                    
                    T_m2c = np.eye(4)
                    T_m2c[:3, :3] = R_ref2cam
                    T_m2c[:3, 3] = t_ref_in_cam.flatten()
                    
                    T_c2m = np.linalg.inv(T_m2c)
                    # Assume marker is at end effector (T_marker_to_gripper = Identity)
                    T_c2b = T_g2b @ T_c2m
                    self.broadcast_matrix_as_tf(T_c2b[:3, :3], T_c2b[:3, 3], self.args.robot_base_frame, "live_estimated_camera")
            
            return True, rvec_ref, t_ref_in_cam, vis

        cv2.putText(vis, f"None of markers {self.all_marker_ids} found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False, None, None, vis

    def broadcast_matrix_as_tf(self, R_mat, t_vec, parent_frame, child_frame):
        """Helper to broadcast a rotation matrix and translation vector as a TF2 transform."""
        try:
            t = tf2_ros.TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent_frame
            t.child_frame_id = child_frame
            
            t.transform.translation.x = float(t_vec[0])
            t.transform.translation.y = float(t_vec[1])
            t.transform.translation.z = float(t_vec[2])
            
            quat = _matrix_to_quat(R_mat)
            t.transform.rotation.x = float(quat[0])
            t.transform.rotation.y = float(quat[1])
            t.transform.rotation.z = float(quat[2])
            t.transform.rotation.w = float(quat[3])
            
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().warn(f"Failed to broadcast live TF: {e}", throttle_duration_sec=5.0)

    def get_robot_pose(self):
        """Query TF for transform from base_link to the designated effector frame."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.robot_base_frame,
                self.args.robot_effector_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            rot_matrix = _quat_to_matrix([q.x, q.y, q.z, q.w])
            t_vector = np.array([t.x, t.y, t.z])
            return True, rot_matrix, t_vector
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {str(e)}")
            return False, None, None

    def execute_move(self, joints_rad):
        if self.moveit2 is None:
            raise RuntimeError("automatic motion was not enabled")
        start_joint_state = self.zero_velocity_current_joint_state()
        if start_joint_state is None:
            return False
        trajectory = self.plan_with_existing_executor(
            joint_positions=joints_rad,
            joint_names=[f"joint_{i}" for i in range(1, 8)],
            start_joint_state=start_joint_state,
        )
        if trajectory is None or not trajectory.points:
            return False
        self.moveit2.motion_suceeded = False
        self.moveit2.execute(trajectory)
        return self.wait_for_motion_with_existing_executor(trajectory)

    def plan_with_existing_executor(self, **kwargs):
        """Plan without letting pymoveit2 spin this already-spinning node."""
        future = self.moveit2.plan_async(**kwargs)
        if future is None:
            return None
        deadline = time.monotonic() + self.moveit2.allowed_planning_time + 5.0
        while rclpy.ok() and not future.done():
            if not self.joint_state_feedback_is_fresh("motion planning"):
                return None
            if time.monotonic() >= deadline:
                self.get_logger().error(
                    "Motion planning response timed out while the external "
                    "executor remained active.")
                return None
            time.sleep(0.01)
        if not future.done():
            return None
        return self.moveit2.get_trajectory(future)

    def wait_for_motion_with_existing_executor(self, trajectory):
        """Wait for execution while the node's MultiThreadedExecutor spins."""
        final_time = trajectory.points[-1].time_from_start
        expected_duration = (
            float(final_time.sec) + float(final_time.nanosec) * 1e-9)
        deadline = time.monotonic() + max(10.0, expected_duration + 10.0)
        while rclpy.ok():
            state = self.moveit2.query_state()
            if state == self._moveit2_idle_state:
                return bool(self.moveit2.motion_suceeded)
            if not self.joint_state_feedback_is_fresh("trajectory execution"):
                if state == self._moveit2_executing_state:
                    self.moveit2.cancel_execution()
                return False
            if time.monotonic() >= deadline:
                if state == self._moveit2_executing_state:
                    self.moveit2.cancel_execution()
                self.get_logger().error(
                    "Trajectory execution timed out; stop was requested.")
                return False
            time.sleep(0.01)
        return False

    def execute_pose_move(self, pose):
        """Plan, validate, then execute one bracelet trajectory through MoveIt."""
        if self.moveit2 is None:
            raise RuntimeError("automatic motion was not enabled")
        self._last_motion_execution_started = False
        try:
            if not self.joint_state_feedback_is_fresh("pre-planning check"):
                return False
            live_joint_state = self.moveit2.joint_state
            if live_joint_state is None or not live_joint_state.position:
                self.get_logger().error(
                    "Cannot plan motion: current joint state is unavailable.")
                return False
            # Pilz PTP requires an exactly stationary start state. The Gen3
            # occasionally publishes residual velocities on the order of
            # 1e-6 rad/s several seconds after settling. Preserve the live
            # name-to-position mapping but remove that measurement noise.
            start_joint_state = JointState()
            start_joint_state.header.stamp = self.get_clock().now().to_msg()
            start_joint_state.header.frame_id = live_joint_state.header.frame_id
            start_joint_state.name = list(live_joint_state.name)
            start_joint_state.position = list(live_joint_state.position)
            start_joint_state.velocity = [
                0.0 for _ in start_joint_state.position]
            trajectory = self.plan_with_existing_executor(
                pose=pose,
                target_link=self.args.robot_effector_frame,
                frame_id=self.args.robot_base_frame,
                start_joint_state=start_joint_state,
            )
            if trajectory is None or not trajectory.points:
                self.get_logger().error(
                    "Motion planning returned no executable trajectory.")
                return False
            violations = self.trajectory_limit_violations(trajectory)
            if violations:
                self.get_logger().error(
                    "Refusing to execute trajectory: " + "; ".join(violations))
                return False
            # Planning can take several seconds. Recheck immediately before
            # handing the trajectory to MoveIt so a controller failure during
            # planning cannot turn into an ambiguous execution request.
            if not self.joint_state_feedback_is_fresh(
                    "pre-execution check"):
                return False
            self._last_motion_execution_started = True
            self.moveit2.motion_suceeded = False
            self.moveit2.execute(trajectory)
            return self.wait_for_motion_with_existing_executor(trajectory)
        except Exception as exc:
            self.get_logger().error(
                f"Motion planning/execution failed: {exc}")
            return False

    def zero_velocity_current_joint_state(self, timeout_sec=2.0):
        """Copy the latest named joint positions with Pilz-safe zero velocity."""
        deadline = time.monotonic() + timeout_sec
        live_joint_state = None
        while rclpy.ok() and time.monotonic() < deadline:
            live_joint_state = self.moveit2.joint_state
            if (live_joint_state is not None
                    and live_joint_state.name
                    and live_joint_state.position
                    and len(live_joint_state.name) ==
                    len(live_joint_state.position)
                    and self.joint_state_feedback_is_fresh(
                        "joint-state acquisition", log_error=False)):
                break
            time.sleep(0.05)
        else:
            return None

        stationary = JointState()
        stationary.header.stamp = self.get_clock().now().to_msg()
        stationary.header.frame_id = live_joint_state.header.frame_id
        stationary.name = list(live_joint_state.name)
        stationary.position = list(live_joint_state.position)
        stationary.velocity = [0.0 for _ in stationary.position]
        return stationary

    def plan_joint_state_move(self, goal_joint_state, start_joint_state):
        """Plan a named joint-space move and apply calibration safety checks."""
        start_by_name = dict(zip(
            start_joint_state.name, start_joint_state.position))
        goal_positions = list(goal_joint_state.position)
        # Map continuous-joint goals to the nearest equivalent representation.
        # For example, +3.10 and -3.18 rad are the same physical joint angle;
        # this prevents a return plan from requesting an unnecessary full turn.
        for index, joint_name in enumerate(goal_joint_state.name):
            if (joint_name in CONTINUOUS_JOINTS
                    and joint_name in start_by_name):
                start_value = start_by_name[joint_name]
                delta = goal_positions[index] - start_value
                goal_positions[index] = start_value + np.arctan2(
                    np.sin(delta), np.cos(delta))
        trajectory = self.plan_with_existing_executor(
            joint_positions=goal_positions,
            joint_names=list(goal_joint_state.name),
            start_joint_state=start_joint_state,
        )
        if trajectory is None or not trajectory.points:
            return None, ["planner returned no trajectory"]
        return trajectory, self.trajectory_limit_violations(trajectory)

    def return_to_start(self, startup_joint_state):
        """Best-effort validated return after a controlled automatic run."""
        if not rclpy.ok():
            self.get_logger().error(
                "Return-to-start skipped because ROS is shutting down.")
            return False
        current_joint_state = self.zero_velocity_current_joint_state()
        if current_joint_state is None:
            self.get_logger().error(
                "Return-to-start skipped: current joint state is unavailable.")
            return False
        try:
            trajectory, violations = self.plan_joint_state_move(
                startup_joint_state, current_joint_state)
            if violations:
                self.get_logger().error(
                    "Return-to-start refused: " + "; ".join(violations))
                return False
            self.get_logger().warn(
                "Returning to the joint configuration recorded at calibration "
                "startup. Keep the workspace clear and remain at the E-stop.")
            self.moveit2.motion_suceeded = False
            self.moveit2.execute(trajectory)
            success = self.wait_for_motion_with_existing_executor(trajectory)
            if success:
                time.sleep(0.5)
                reached = self.zero_velocity_current_joint_state()
                if reached is None:
                    self.get_logger().error(
                        "Return completed but final joint state is unavailable.")
                    return False
                reached_by_name = dict(zip(reached.name, reached.position))
                errors = []
                for joint_name, target in zip(
                        startup_joint_state.name,
                        startup_joint_state.position):
                    if joint_name not in reached_by_name:
                        errors.append((joint_name, float("inf")))
                        continue
                    error = reached_by_name[joint_name] - target
                    if joint_name in CONTINUOUS_JOINTS:
                        error = np.arctan2(np.sin(error), np.cos(error))
                    errors.append((joint_name, abs(float(error))))
                max_joint, max_error = max(errors, key=lambda item: item[1])
                if max_error > RETURN_POSITION_TOLERANCE_RAD:
                    self.get_logger().error(
                        "Return trajectory completed but startup position "
                        f"verification failed: {max_joint} error="
                        f"{max_error:.4f} rad (limit "
                        f"{RETURN_POSITION_TOLERANCE_RAD:.4f} rad).")
                    return False
                self.get_logger().info(
                    "RETURN-TO-START COMPLETED AND VERIFIED "
                    f"(maximum joint error {max_error:.4f} rad).")
            else:
                self.get_logger().error(
                    "Return-to-start execution did not complete successfully.")
            return success
        except Exception as exc:
            self.get_logger().error(f"Return-to-start failed: {exc}")
            return False

    @staticmethod
    def trajectory_limit_violations(trajectory):
        """Return hard-path or endpoint-margin violations for a trajectory."""
        names = list(trajectory.joint_names)
        name_to_index = {name: index for index, name in enumerate(names)}
        violations = []
        for joint_name, (lower, upper) in BOUNDED_JOINT_LIMITS.items():
            if joint_name not in name_to_index:
                violations.append(f"{joint_name}=missing")
                continue
            joint_index = name_to_index[joint_name]
            for point_index, point in enumerate(trajectory.points):
                if joint_index >= len(point.positions):
                    violations.append(
                        f"{joint_name}=missing at trajectory point "
                        f"{point_index + 1}")
                    break
                value = point.positions[joint_index]
                if value < lower or value > upper:
                    violations.append(
                        f"{joint_name}={value:+.4f} outside hard range "
                        f"[{lower:+.4f}, {upper:+.4f}] at trajectory point "
                        f"{point_index + 1}")
                    break
            else:
                endpoint = trajectory.points[-1].positions[joint_index]
                safe_lower = lower + JOINT_LIMIT_MARGIN_RAD
                safe_upper = upper - JOINT_LIMIT_MARGIN_RAD
                if endpoint < safe_lower or endpoint > safe_upper:
                    violations.append(
                        f"{joint_name} endpoint={endpoint:+.4f} outside safe "
                        f"range [{safe_lower:+.4f}, {safe_upper:+.4f}] "
                        f"({JOINT_LIMIT_MARGIN_RAD:.2f} rad margin)")
        return violations

    def generate_relative_calibration_poses(self):
        """Generate a bounded 15-pose sweep from the current bracelet pose.

        Translations are expressed in base axes. Rotations are applied about
        the safer anchor bracelet's local axes. The anchor itself is defined
        relative to the live starting pose, so the sweep is not coupled to an
        old camera extrinsic.
        """
        try:
            tf_start = self.tf_buffer.lookup_transform(
                self.args.robot_base_frame,
                self.args.robot_effector_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
        except Exception as exc:
            self.get_logger().error(
                f"Cannot read starting effector pose: {exc}")
            return []

        translation_scale = self.args.auto_translation_scale
        angle_scale = self.args.auto_rotation_scale
        t = tf_start.transform.translation
        q = tf_start.transform.rotation
        p_start = np.array([t.x, t.y, t.z], dtype=float)
        R_start = _quat_to_matrix([q.x, q.y, q.z, q.w])

        # The live calibration posture can be close to joint_4's lower limit.
        # First move 40 mm outward and rotate +8 deg about local Y; this was
        # validated on REAL-1 to increase joint_4 margin. Remaining offsets are
        # relative to that anchor and deliberately bias local-Y rotation away
        # from the lower limit while retaining X/Y/Z translation and three-axis
        # rotation diversity. Pose 1 remains the exact live starting pose.
        anchor_dp = translation_scale * np.array([0.04, 0.0, 0.0])
        anchor_ry = np.radians(angle_scale * 8.0)
        cy_anchor, sy_anchor = np.cos(anchor_ry), np.sin(anchor_ry)
        R_anchor_delta = np.array([
            [cy_anchor, 0., sy_anchor],
            [0., 1., 0.],
            [-sy_anchor, 0., cy_anchor],
        ])
        p_anchor = p_start + anchor_dp
        R_anchor = R_start @ R_anchor_delta

        # dx, dy, dz [m], then local anchor rx, ry, rz [deg]. Including the
        # anchor itself, this set spans 120 mm in X and at least 19 degrees
        # about local Y; X and Z each span at least 24 degrees.
        offsets = [
            ( 0.00,  0.00,  0.00,   0,   0,   0),  # live pose
            ( 0.00,  0.00,  0.00,   0,   0,   0),  # safer anchor
            ( 0.00,  0.04,  0.00,   8,   0,   0),
            (-0.01,  0.00,  0.00,   0,   0,   0),
            ( 0.00, -0.04,  0.00,  -8,   0,   0),
            ( 0.06,  0.03,  0.02,  10,  10,   8),
            (-0.01,  0.03, -0.02, -10,  10,  -8),
            (-0.01, -0.03,  0.02,  10,   0,   8),
            ( 0.06, -0.03, -0.02, -10,  -4,  -8),
            ( 0.00,  0.00,  0.04,  15,   0,   0),
            ( 0.00,  0.00, -0.04, -15,   0,   0),
            ( 0.03,  0.00,  0.03,   0,  15,  10),
            (-0.01,  0.00,  0.03,   0,  -4, -10),
            ( 0.00,  0.03, -0.03,  12,   0,  15),
            ( 0.00, -0.03, -0.03, -12,   0, -15),
        ]
        offsets = offsets[:self.args.requested_samples]

        poses = []
        for index, values in enumerate(offsets):
            dx, dy, dz, rx_deg, ry_deg, rz_deg = values
            if index == 0:
                reference_p = p_start
                reference_R = R_start
            else:
                reference_p = p_anchor
                reference_R = R_anchor
            local_delta_p = translation_scale * np.array([dx, dy, dz])
            rx, ry, rz = np.radians(
                angle_scale * np.array([rx_deg, ry_deg, rz_deg]))
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            Rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
            Ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
            Rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
            R_target = reference_R @ (Rz @ Ry @ Rx)
            quat = _matrix_to_quat(R_target)

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (
                reference_p + local_delta_p).tolist()
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])
            total_dp = reference_p + local_delta_p - p_start
            total_rpy_deg = np.degrees(_matrix_to_euler_xyz(
                R_start.T @ R_target))
            description = (
                f"base dxyz=({total_dp[0]:+.3f},{total_dp[1]:+.3f},"
                f"{total_dp[2]:+.3f}) m, live-local rpy="
                f"({total_rpy_deg[0]:+.1f},{total_rpy_deg[1]:+.1f},"
                f"{total_rpy_deg[2]:+.1f}) deg")
            poses.append((pose, description))

        return poses

    def run_relative_automatic_capture(self):
        targets = self.generate_relative_calibration_poses()
        if len(targets) < self.args.min_samples:
            self.get_logger().error(
                "Not enough relative targets were generated for calibration.")
            return False

        startup_joint_state = self.zero_velocity_current_joint_state()
        if startup_joint_state is None:
            self.get_logger().error(
                "Cannot record the calibration startup joint configuration.")
            return False

        if self.args.automatic_preposition_only:
            if len(targets) < 2:
                self.get_logger().error(
                    "No safer anchor target is available for prepositioning.")
                return False
            self.get_logger().warn(
                "PREPOSITION-ONLY WILL MOVE THE PHYSICAL ARM to the safer "
                "calibration anchor, then exit without capturing samples. "
                "Keep the board path clear and remain at the E-stop.")
            for remaining in range(self.args.motion_countdown, 0, -1):
                if not self.joint_state_feedback_is_fresh(
                        "preposition countdown"):
                    return False
                self.get_logger().warn(
                    f"Preposition motion starts in {remaining} s "
                    "(Ctrl-C to abort)...")
                time.sleep(1.0)
            anchor_pose, anchor_description = targets[1]
            self.get_logger().info(
                f"Moving to safer calibration start: {anchor_description}")
            if not self.execute_pose_move(anchor_pose):
                self.get_logger().error(
                    "Preposition-only motion did not complete successfully.")
                return False
            self.get_logger().info(
                "PREPOSITION-ONLY COMPLETED. Leave the arm here, rerun "
                "--automatic-plan-only, and require the targets plus return "
                "leg to pass before calibration motion.")
            return True

        if self.args.automatic_plan_only:
            self.get_logger().info(
                "PLAN-ONLY: validating the complete sequential target sweep; "
                "the arm will not move and no samples will be captured.")
            start_joint_state = startup_joint_state
            for index, (pose, description) in enumerate(targets):
                if index == 0:
                    # Pose 1 is exactly the TF pose from which this target set
                    # was generated. MoveIt commonly reports a zero-length
                    # request as generic FAILURE instead of a valid empty plan.
                    self.get_logger().info(
                        f"Plan-only target 1/{len(targets)} passed: current "
                        "starting pose requires no trajectory.")
                    continue
                try:
                    trajectory = self.plan_with_existing_executor(
                        pose=pose,
                        frame_id=self.args.robot_base_frame,
                        target_link=self.args.robot_effector_frame,
                        start_joint_state=start_joint_state,
                    )
                except Exception as exc:
                    self.get_logger().error(
                        f"Plan-only target {index+1} raised an error: {exc}")
                    return False
                if trajectory is None or not trajectory.points:
                    self.get_logger().error(
                        f"Plan-only target {index+1} FAILED: {description}")
                    return False
                # Preserve the trajectory's explicit name-to-position mapping.
                # JointState and planned-trajectory ordering are not guaranteed
                # to match the MoveIt2 client's configured joint list (REAL-1
                # currently publishes 1,2,4,5,3,6,7). Passing positions alone
                # can therefore assign a continuous joint's value to a bounded
                # joint and make the next request INVALID_ROBOT_STATE.
                start_joint_state = JointState()
                start_joint_state.header.stamp = self.get_clock().now().to_msg()
                start_joint_state.name = list(trajectory.joint_names)
                start_joint_state.position = list(
                    trajectory.points[-1].positions)
                violations = self.trajectory_limit_violations(trajectory)
                if violations:
                    self.get_logger().error(
                        f"Plan-only target {index+1} REJECTED: planned "
                        "trajectory violates configured joint limits: "
                        + "; ".join(violations))
                    return False
                endpoint = ", ".join(
                    f"{name}={position:+.3f}"
                    for name, position in zip(
                        start_joint_state.name, start_joint_state.position))
                self.get_logger().info(
                    f"Plan-only target {index+1}/{len(targets)} passed "
                    f"({len(trajectory.points)} trajectory points): "
                    f"{description}; endpoint [{endpoint}]")
            if self.args.return_to_start:
                try:
                    return_trajectory, violations = self.plan_joint_state_move(
                        startup_joint_state, start_joint_state)
                except Exception as exc:
                    self.get_logger().error(
                        f"Plan-only return-to-start raised an error: {exc}")
                    return False
                if violations:
                    self.get_logger().error(
                        "Plan-only return-to-start REJECTED: "
                        + "; ".join(violations))
                    return False
                self.get_logger().info(
                    "Plan-only return-to-start passed "
                    f"({len(return_trajectory.points)} trajectory points).")
            validated_scope = (
                "all targets and the return leg"
                if self.args.return_to_start else "all targets")
            self.get_logger().info(
                f"PLAN-ONLY PASSED for {validated_scope}. This validates the "
                "modeled robot, not clearance for the unmodeled calibration "
                "board.")
            return True

        self.get_logger().warn(
            "AUTOMATIC HARDWARE MOTION WILL BEGIN. MoveIt checks the modeled "
            "robot/tool, but the calibration board is not in its collision "
            "model. Keep the full board sweep clear and remain at the E-stop.")
        self.get_logger().warn(
            f"Sweep bounds relative to the starting bracelet pose: X=[0.000, "
            f"+{0.10*self.args.auto_translation_scale:.3f}] m, Y=+/-"
            f"{0.04*self.args.auto_translation_scale:.3f} m, Z=+/-"
            f"{0.04*self.args.auto_translation_scale:.3f} m.")
        for remaining in range(self.args.motion_countdown, 0, -1):
            if not self.joint_state_feedback_is_fresh(
                    "motion countdown"):
                return False
            self.get_logger().warn(
                f"Motion starts in {remaining} s (Ctrl-C to abort)...")
            time.sleep(1.0)

        outcome = False
        self._automatic_motion_started = False
        self._automatic_return_allowed = True
        try:
            consecutive_capture_failures = 0
            sequence_complete = True
            for index, (pose, description) in enumerate(targets):
                if not rclpy.ok():
                    self._automatic_return_allowed = False
                    sequence_complete = False
                    break
                if not self.joint_state_feedback_is_fresh(
                        f"pose {index+1} pre-check"):
                    self._automatic_return_allowed = False
                    sequence_complete = False
                    break
                if index == 0:
                    self.get_logger().info(
                        f"Automatic pose 1/{len(targets)} is the current "
                        "starting pose; capturing without issuing a motion "
                        "request.")
                else:
                    self.get_logger().info(
                        f"Moving to automatic pose {index+1}/{len(targets)}: "
                        f"{description}")
                    if not self.execute_pose_move(pose):
                        if self._last_motion_execution_started:
                            # Once execution was requested, a failure can leave
                            # the physical state ambiguous. Do not compound it
                            # with an automatic recovery motion.
                            self._automatic_return_allowed = False
                            return_note = (
                                " Return-to-start is suppressed because the "
                                "execution state is uncertain.")
                        else:
                            # A planning/validation rejection did not move the
                            # arm; returning from the last completed pose is OK.
                            return_note = (
                                " The arm did not move for this leg; a validated "
                                "return-to-start will be attempted.")
                        self.get_logger().error(
                            "Automatic calibration aborted: MoveIt could not "
                            f"safely plan/execute pose {index+1}." + return_note)
                        sequence_complete = False
                        break
                    self._automatic_motion_started = True

                time.sleep(self.args.settle_time)
                if not self.joint_state_feedback_is_fresh(
                        f"pose {index+1} capture check"):
                    self._automatic_return_allowed = False
                    sequence_complete = False
                    break
                if self.capture_sample(index):
                    consecutive_capture_failures = 0
                else:
                    consecutive_capture_failures += 1
                    self.get_logger().warn(
                        f"Capture failure {consecutive_capture_failures}/"
                        f"{self.args.max_consecutive_capture_failures}; no "
                        "search motion will be attempted.")
                    if (consecutive_capture_failures >=
                            self.args.max_consecutive_capture_failures):
                        self.get_logger().error(
                            "Automatic calibration aborted after repeated "
                            "target detection failures.")
                        sequence_complete = False
                        break

            if sequence_complete:
                outcome = self.solve_calibration()
        except KeyboardInterrupt:
            self._automatic_return_allowed = False
            self.get_logger().warn(
                "Return-to-start suppressed because the operator interrupted "
                "the calibration.")
            raise

        if (self.args.return_to_start and self._automatic_motion_started
                and self._automatic_return_allowed):
            return_ok = self.return_to_start(startup_joint_state)
            outcome = outcome and return_ok
        return outcome

    def capture_sample(self, idx):
        """Verify sharp frame with detected marker, record gripper & marker transforms."""
        best_sharpness = -1.0
        best_frame_data = None
        start_time = time.time()
        with self.lock:
            last_processed_sequence = self._image_sequence
        new_frame_count = 0
        
        # Only evaluate frames received after capture begins. Reusing the last
        # cached pre-motion frame would pair an old camera pose with the new
        # robot TF and silently corrupt the hand-eye solution.
        while time.time() - start_time < 1.5:
            with self.lock:
                sequence = self._image_sequence
                frame = (self.latest_frame.copy()
                         if sequence > last_processed_sequence
                         and self.latest_frame is not None else None)
            
            if frame is not None:
                last_processed_sequence = sequence
                new_frame_count += 1
                ok_marker, rvec, tvec, vis = self.detect_marker(frame)
                if ok_marker:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame_data = (rvec, tvec, vis, frame)
            
            time.sleep(0.05)
            
        if best_frame_data is None:
            if new_frame_count == 0:
                self.get_logger().warn(
                    "  [Capture Fail] No new camera frame arrived during the "
                    "1.5 s capture window; refusing to reuse a stale frame.")
            else:
                self.get_logger().warn(
                    "  [Capture Fail] ArUco Marker was not detected in any "
                    f"of {new_frame_count} new frame(s) at this pose.")
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is not None:
                cv2.imwrite(os.path.join(self.args.save_dir, f"pose_{idx+1:02d}_NO_MARKER.png"), frame)
            return False
            
        rvec, tvec, vis, _ = best_frame_data
        ok_tf, rot_g2b, t_g2b = self.get_robot_pose()
        if not ok_tf:
            self.get_logger().error("  [Capture Fail] Could not query robot transform from TF.")
            return False
            
        # Record sample
        cv2.imwrite(os.path.join(self.args.save_dir, f"pose_{idx+1:02d}_ok.png"), vis)
        R_marker2cam, _ = cv2.Rodrigues(rvec)
        
        self.R_g2b.append(rot_g2b)
        self.t_g2b.append(t_g2b)
        self.R_t2c.append(R_marker2cam)
        self.t_t2c.append(tvec.flatten())
        
        self.sample_count += 1
        self.get_logger().info(f"  [Sample Captured #{self.sample_count}] Pose: {t_g2b[0]:.3f}, {t_g2b[1]:.3f}, {t_g2b[2]:.3f} | Sharpness = {best_sharpness:.1f}")
        return True

    @staticmethod
    def _transform(rot, trans):
        T = np.eye(4)
        T[:3, :3] = np.asarray(rot, dtype=float)
        T[:3, 3] = np.asarray(trans, dtype=float).reshape(3)
        return T

    def _solution_residual(self, R_solution, t_solution):
        """Return consistency RMS for the fixed transform implied per sample.

        A correct hand-eye result makes the unobserved marker-to-gripper
        transform (eye-to-hand) or marker-to-base transform (eye-in-hand)
        constant across every robot pose.  This provides a solver-independent
        score and catches plausible-looking but conventionally inverted output.
        """
        T_solution = self._transform(R_solution, t_solution)
        implied = []
        for R_g2b, t_g2b, R_t2c, t_t2c in zip(
                self.R_g2b, self.t_g2b, self.R_t2c, self.t_t2c):
            T_g2b = self._transform(R_g2b, t_g2b)
            T_t2c = self._transform(R_t2c, t_t2c)
            if self.args.mode == "eye-to-hand":
                # T_t2g = inv(T_g2b) * T_c2b * T_t2c
                T_const = np.linalg.inv(T_g2b) @ T_solution @ T_t2c
            else:
                # T_t2b = T_g2b * T_c2g * T_t2c
                T_const = T_g2b @ T_solution @ T_t2c
            implied.append(T_const)

        translations = np.asarray([T[:3, 3] for T in implied])
        t_mean = np.mean(translations, axis=0)
        t_rms = float(np.sqrt(np.mean(np.sum(
            (translations - t_mean) ** 2, axis=1))))

        quats = np.asarray([_matrix_to_quat(T[:3, :3]) for T in implied])
        # q and -q encode the same rotation. Align signs before averaging.
        quats[np.sum(quats * quats[0], axis=1) < 0.0] *= -1.0
        q_mean = np.mean(quats, axis=0)
        q_mean /= np.linalg.norm(q_mean)
        R_mean = _quat_to_matrix(q_mean)
        angles = np.asarray([
            _rotation_angle(R_mean.T @ T[:3, :3]) for T in implied])
        r_rms_deg = float(np.degrees(np.sqrt(np.mean(angles ** 2))))
        return t_rms, r_rms_deg

    def solve_calibration(self):
        """Solves hand-eye calibration equations and prints static TF publishers."""
        self.get_logger().info(f"Computing hand-eye calibration using {self.sample_count} samples...")
        if self.sample_count < self.args.min_samples:
            self.get_logger().error(
                f"Calibration requires at least {self.args.min_samples} valid, "
                f"well-distributed samples; got {self.sample_count}.")
            return False

        gripper_t = np.asarray(self.t_g2b)
        translation_span = float(np.linalg.norm(
            np.max(gripper_t, axis=0) - np.min(gripper_t, axis=0)))
        relative_angles = []
        for i in range(len(self.R_g2b)):
            for j in range(i + 1, len(self.R_g2b)):
                relative_angles.append(_rotation_angle(
                    np.asarray(self.R_g2b[i]).T @ self.R_g2b[j]))
        rotation_span_deg = (float(np.degrees(max(relative_angles)))
                             if relative_angles else 0.0)
        self.get_logger().info(
            f"Sample diversity: translation span={translation_span:.3f} m, "
            f"rotation span={rotation_span_deg:.1f} deg")
        if (translation_span < self.args.min_translation_span
                or rotation_span_deg < self.args.min_rotation_span_deg):
            self.get_logger().error(
                "Calibration rejected: robot poses do not span enough "
                "translation/orientation for a well-conditioned solve.")
            return False
            
        solvers = [
            ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
            ("PARK", cv2.CALIB_HAND_EYE_PARK),
            ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
            ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF)
        ]
        
        candidates = []
        
        # Run solvers
        for name, method in solvers:
            try:
                if self.args.mode == "eye-in-hand":
                    # Eye-in-hand (moving camera)
                    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                        self.R_g2b,
                        self.t_g2b,
                        self.R_t2c,
                        self.t_t2c,
                        method=method
                    )
                    R_candidate, t_candidate = R_cam2gripper, t_cam2gripper
                else:
                    # Eye-to-hand (static camera)
                    # We pass the inverse gripper poses and target poses to get camera-to-base
                    R_b2g = [R_g2b_i.T for R_g2b_i in self.R_g2b]
                    t_b2g = [-R_g2b_i.T @ t_g2b_i for R_g2b_i, t_g2b_i in zip(self.R_g2b, self.t_g2b)]
                    
                    R_cam2base, t_cam2base = cv2.calibrateHandEye(
                        R_b2g,
                        t_b2g,
                        self.R_t2c,
                        self.t_t2c,
                        method=method
                    )
                    R_candidate, t_candidate = R_cam2base, t_cam2base

                if (R_candidate is None or t_candidate is None
                        or not np.all(np.isfinite(R_candidate))
                        or not np.all(np.isfinite(t_candidate))):
                    raise ValueError("solver returned a non-finite result")
                t_rms, r_rms_deg = self._solution_residual(
                    R_candidate, t_candidate)
                # One degree is scaled to 1 cm for ranking only; acceptance is
                # checked against the two physical residuals independently.
                score = t_rms + 0.01 * r_rms_deg
                candidates.append((score, name, R_candidate, t_candidate,
                                   t_rms, r_rms_deg))
                self.get_logger().info(
                    f"  Solver {name}: translation RMS={t_rms*1000:.1f} mm, "
                    f"rotation RMS={r_rms_deg:.2f} deg")
            except Exception as e:
                self.get_logger().warn(f"  Solver {name}: FAILED - {str(e)}")
                
        if not candidates:
            self.get_logger().error("All hand-eye calibration solvers failed. Check your data.")
            return False

        _, best_name, best_R, best_t, best_t_rms, best_r_rms = min(
            candidates, key=lambda item: item[0])
        self.get_logger().info(
            f"Selected {best_name} by minimum consistency residual: "
            f"{best_t_rms*1000:.1f} mm, {best_r_rms:.2f} deg")

        if (best_t_rms > self.args.max_translation_rms
                or best_r_rms > self.args.max_rotation_rms_deg):
            self.get_logger().error(
                "Calibration rejected: consistency residual exceeds the "
                f"configured limits ({self.args.max_translation_rms*1000:.1f} "
                f"mm, {self.args.max_rotation_rms_deg:.2f} deg).")
            return False
            
        tf_trans = best_t.flatten()
        quat = _matrix_to_quat(best_R)  # [qx, qy, qz, qw]
        rpy = np.degrees(_matrix_to_euler_xyz(best_R))
        
        # Decide frame naming
        camera_frame = self.detected_camera_frame if self.detected_camera_frame else self.default_camera_frame
        
        # Validation bounds check (for static table cameras)
        approx_values = (self.args.approx_x, self.args.approx_y,
                         self.args.approx_z)
        if (self.args.mode == "eye-to-hand"
                and all(v is not None for v in approx_values)):
            approx = np.array(approx_values)
            error = np.linalg.norm(tf_trans - approx)
            self.get_logger().info("--------------------------------------------------------------------------------")
            self.get_logger().info(f"PHYSICAL DISPLACEMENT BOUNDS VALIDATION:")
            self.get_logger().info(f"  Physical manual measurement: X={approx[0]:.3f}, Y={approx[1]:.3f}, Z={approx[2]:.3f} m")
            self.get_logger().info(f"  Calibration solution:        X={tf_trans[0]:.3f}, Y={tf_trans[1]:.3f}, Z={tf_trans[2]:.3f} m")
            self.get_logger().info(f"  Euclidean discrepancy:       {error:.4f} m ({error*100.1:.1f} cm)")
            if error > 0.15:
                self.get_logger().warn(
                    "WARNING: Discrepancy between calibration and manual table measurements is high (> 15cm)!\n"
                    "Please double-check marker sizes, dictionary configuration, or tracking quality."
                )
            else:
                self.get_logger().info(
                    "Discrepancy is below the coarse 15 cm sanity threshold. "
                    "This is not the final workspace accuracy validation.")
        
        # Export yaml
        yaml_name = (f"{self.args.camera}_"
                     f"{self.args.mode.replace('-', '_')}_calib.yaml")
        yaml_path = os.path.join(self.args.save_dir, yaml_name)
        with open(yaml_path, "w") as f:
            f.write(f"# Auto-generated Hand-Eye Calibration Result\n")
            f.write(f"# Solver: {best_name} | Mode: {self.args.mode}\n")
            f.write(f"# Parent Frame: {self.args.robot_base_frame if self.args.mode=='eye-to-hand' else self.args.robot_effector_frame}\n")
            f.write(f"# Camera Frame: {camera_frame}\n\n")
            f.write("target:\n")
            f.write(f"  type: {self.args.target_type}\n")
            f.write(f"  aruco_dictionary: {self.args.aruco_dict}\n")
            f.write(f"  marker_size_m: {self.args.marker_size:.8f}\n")
            if self.args.target_type == "grid":
                f.write(f"  rows: {self.args.grid_rows}\n")
                f.write(f"  cols: {self.args.grid_cols}\n")
                f.write(f"  marker_gap_m: {self.args.marker_gap:.8f}\n")
                f.write(f"  marker_ids_row_major: [{self.args.grid_marker_ids}]\n")
                f.write(f"  corner_shift: {self.args.grid_corner_shift}\n")
            f.write("\n")
            f.write("consistency_residual:\n")
            f.write(f"  translation_rms_m: {best_t_rms:.8f}\n")
            f.write(f"  rotation_rms_deg: {best_r_rms:.6f}\n\n")
            f.write("translation:\n")
            f.write(f"  x: {tf_trans[0]:.8f}\n")
            f.write(f"  y: {tf_trans[1]:.8f}\n")
            f.write(f"  z: {tf_trans[2]:.8f}\n\n")
            f.write("rotation_quaternion:\n")
            f.write(f"  x: {quat[0]:.8f}\n")
            f.write(f"  y: {quat[1]:.8f}\n")
            f.write(f"  z: {quat[2]:.8f}\n")
            f.write(f"  w: {quat[3]:.8f}\n\n")
            f.write("rotation_euler_xyz_deg:\n")
            f.write(f"  r: {rpy[0]:.4f}\n")
            f.write(f"  p: {rpy[1]:.4f}\n")
            f.write(f"  y: {rpy[2]:.4f}\n")
            
        parent_f = self.args.robot_base_frame if self.args.mode == "eye-to-hand" else self.args.robot_effector_frame
        
        self.get_logger().info("================================================================================")
        self.get_logger().info("CALIBRATION CALCULATIONS COMPLETED!")
        self.get_logger().info(f"Target Transform: {parent_f} -> {camera_frame}")
        self.get_logger().info(f"Translation (meters) : X={tf_trans[0]:.5f}, Y={tf_trans[1]:.5f}, Z={tf_trans[2]:.5f}")
        self.get_logger().info(f"Euler Angles (deg)   : R={rpy[0]:.3f}, P={rpy[1]:.3f}, Y={rpy[2]:.3f}")
        self.get_logger().info(f"Quaternion (xyzw)    : [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        self.get_logger().info("--------------------------------------------------------------------------------")
        self.get_logger().info("Copy-paste ROS 2 static TF publisher command:")
        self.get_logger().info(f"ros2 run tf2_ros static_transform_publisher {tf_trans[0]:.6f} {tf_trans[1]:.6f} {tf_trans[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} {parent_f} {camera_frame}")
        self.get_logger().info("================================================================================")
        self.get_logger().info(f"Calibration configurations exported to: {yaml_path}")
        return True

    def run(self):
        # Wait for camera stream to become active
        self.get_logger().info(
            f"Awaiting image {self.image_topic} and CameraInfo "
            f"{self.info_topic}...")
        rate = self.create_rate(10)
        deadline = time.monotonic() + self.args.preflight_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            with self.lock:
                ready = (self.latest_frame is not None
                         and self.camera_matrix is not None)
            if ready:
                break
            rate.sleep()

        with self.lock:
            ready = (self.latest_frame is not None
                     and self.camera_matrix is not None)
            frame = (self.latest_frame.copy()
                     if self.latest_frame is not None else None)
        if not ready:
            with self.lock:
                got_image = self.latest_frame is not None
                got_info = self.camera_matrix is not None
            self.get_logger().error(
                "Calibration preflight timed out. Both a current image and "
                "matching CameraInfo are mandatory; fallback intrinsics are "
                "intentionally disabled. Received: "
                f"image={'yes' if got_image else 'NO'}, "
                f"CameraInfo={'yes' if got_info else 'NO'}. Start the selected "
                "camera driver and verify its topics in this ROS domain.")
            return False

        ok_marker, _, _, _ = self.detect_marker(frame)
        ok_tf, _, _ = self.get_robot_pose()
        self.get_logger().info(
            f"Preflight: frame={self.detected_camera_frame or self.default_camera_frame}, "
            f"marker={'visible' if ok_marker else 'NOT visible'}, "
            f"robot TF={'available' if ok_tf else 'unavailable'}")
        if self.args.preflight_only:
            return bool(ok_marker and ok_tf)

        if self.args.capture_mode == "manual":
            self.get_logger().info(
                "Manual eye-to-hand capture: place the calibration board "
                "rigidly on the selected robot effector, move to a distinct "
                "pose under local supervision, then press ENTER. Type q to "
                "finish early. No command in this mode moves the robot.")
            for i in range(self.args.requested_samples):
                try:
                    answer = input(
                        f"Sample {i + 1}/{self.args.requested_samples}: "
                        "ENTER=capture, q=finish > ").strip().lower()
                except EOFError:
                    self.get_logger().error(
                        "Manual capture needs an interactive terminal.")
                    return False
                if answer == "q":
                    break
                self.capture_sample(i)
            return self.solve_calibration()

        if self.args.automatic_pose_source == "relative-cartesian":
            return self.run_relative_automatic_capture()

        self.get_logger().info(
            "Using the legacy joint-pose automatic calibration sweep.")
        if self.args.show_gui:
            cv2.namedWindow("Unified Calibration Feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Unified Calibration Feed", 848, 480)
        
        total_poses = len(self.poses)
        
        if self.args.mode == "eye-in-hand":
            # Joint-based safe trajectory sweep for eye-in-hand
            self.get_logger().info(f"Visiting {total_poses} safe predefined configurations automatically...")
            for i, joints in enumerate(self.poses):
                if not rclpy.ok():
                    break
                    
                self.get_logger().info(f"Moving to Pose {i+1}/{total_poses}...")
                self.execute_move(joints)
                
                # Settle arm and draw status feed
                settle_start = time.time()
                while time.time() - settle_start < self.args.settle_time:
                    with self.lock:
                        curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                    if curr_frame is not None:
                        _, _, _, vis = self.detect_marker(curr_frame)
                        cv2.putText(vis, f"Pose {i+1}/{total_poses} settling...", (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Unified Calibration Feed", vis)
                        cv2.waitKey(30)
                        
                # Settle completed, attempt marker discovery
                self.get_logger().info(f"Capturing details at Pose {i+1}...")
                captured = self.capture_sample(i)
                
                # Active Search fallback for lost markers (micro-adjustments of wrist joints)
                if not captured:
                    self.get_logger().warn("  [Active Search] Marker not detected. Initiating micro-search sweeps...")
                    search_joint_offsets = [
                        ( 0.10, 0.0),  # joint_6 +6 deg
                        (-0.10, 0.0),  # joint_6 -6 deg
                        ( 0.0,  0.26), # joint_7 +15 deg
                        ( 0.0, -0.26), # joint_7 -15 deg
                        ( 0.10, 0.26), # joint_6 +6, joint_7 +15
                        (-0.10, -0.26),# joint_6 -6, joint_7 -15
                    ]
                    
                    for idx_s, (dj6, dj7) in enumerate(search_joint_offsets):
                        self.get_logger().info(f"    [Micro-Search {idx_s+1}/{len(search_joint_offsets)}] Offset: j6={np.degrees(dj6):.1f} deg, j7={np.degrees(dj7):.1f} deg...")
                        searched_joints = list(joints)
                        searched_joints[5] += dj6
                        searched_joints[6] += dj7
                        
                        self.execute_move(searched_joints)
                        time.sleep(1.0)
                        captured = self.capture_sample(i)
                        if captured:
                            self.get_logger().info("    [Micro-Search] SUCCESS! Marker captured.")
                            break
                            
                    if not captured:
                        self.get_logger().error("    [Micro-Search] FAILED - Marker could not be located at this pose.")
                        self.execute_move(joints)
                        
                # Visual capture confirmation
                with self.lock:
                    curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                if curr_frame is not None:
                    _, _, _, vis = self.detect_marker(curr_frame)
                    color = (0, 255, 0) if captured else (0, 0, 255)
                    status_str = f"Captured Sample {self.sample_count}" if captured else "CAPTURE FAILED"
                    cv2.putText(vis, status_str, (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow("Unified Calibration Feed", vis)
                    cv2.waitKey(1000)
                    
        else:
            # Joint-based safe trajectory sweep for eye-to-hand
            self.get_logger().info(f"Visiting {total_poses} safe predefined configurations automatically...")
            for i, joints in enumerate(self.poses):
                if not rclpy.ok():
                    break
                    
                self.get_logger().info(f"Moving to Pose {i+1}/{total_poses}...")
                self.execute_move(joints)
                
                # Settle arm and draw status feed
                settle_start = time.time()
                while time.time() - settle_start < self.args.settle_time:
                    with self.lock:
                        curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                    if curr_frame is not None:
                        _, _, _, vis = self.detect_marker(curr_frame)
                        cv2.putText(vis, f"Pose {i+1}/{total_poses} settling...", (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Unified Calibration Feed", vis)
                        cv2.waitKey(30)
                        
                # Settle completed, attempt capture
                self.get_logger().info(f"Capturing details at Pose {i+1}...")
                captured = self.capture_sample(i)
                
                # Active Search fallback for eye-to-hand (minor joint rotations)
                if not captured:
                    self.get_logger().warn("  [Active Search] Marker not detected. Initiating micro-search sweeps...")
                    search_joint_offsets = [
                        (0.08,  0.0),
                        (-0.08, 0.0),
                        (0.0,   0.20),
                        (0.0,  -0.20),
                    ]
                    for idx_s, (dj5, dj6) in enumerate(search_joint_offsets):
                        self.get_logger().info(f"    [Micro-Search {idx_s+1}] Adjusting joints: j5={np.degrees(dj5):.1f} deg, j6={np.degrees(dj6):.1f} deg...")
                        searched_joints = list(joints)
                        searched_joints[4] += dj5
                        searched_joints[5] += dj6
                        
                        self.execute_move(searched_joints)
                        time.sleep(1.0)
                        captured = self.capture_sample(i)
                        if captured:
                            self.get_logger().info("    [Micro-Search] SUCCESS! Marker captured.")
                            break
                            
                    if not captured:
                        self.get_logger().error("    [Micro-Search] FAILED.")
                        self.execute_move(joints)
                        
                # Visual capture confirmation
                with self.lock:
                    curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                if curr_frame is not None:
                    _, _, _, vis = self.detect_marker(curr_frame)
                    color = (0, 255, 0) if captured else (0, 0, 255)
                    status_str = f"Captured Sample {self.sample_count}" if captured else "CAPTURE FAILED"
                    cv2.putText(vis, status_str, (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow("Unified Calibration Feed", vis)
                    cv2.waitKey(1000)

        _safe_destroy_windows()
        return self.solve_calibration()

def main():
    parser = argparse.ArgumentParser(description="Unified Camera Calibration CLI Script")
    parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "oakd", "kinova", "custom"],
                        help="Camera profile to calibrate (default: realsense)")
    parser.add_argument("--mode", type=str, default="eye-to-hand", choices=["eye-in-hand", "eye-to-hand", "hand-to-eye"],
                        help="Calibration mode (default: eye-to-hand)")
    parser.add_argument("--marker-id", type=int, default=8,
                        help="ArUco reference marker ID (default: 8)")
    parser.add_argument("--marker-size", type=float, default=0.2032,
                        help="ArUco reference marker size in meters (default: 0.2032)")
    parser.add_argument("--aruco-dict", type=str, default="DICT_4X4_50",
                        help="ArUco dictionary name (default: DICT_4X4_50)")
    parser.add_argument("--robot-base-frame", type=str, default="base_link",
                        help="Robot base frame ID (default: base_link)")
    parser.add_argument("--robot-effector-frame", type=str, default="bracelet_link",
                        help="Rigid robot frame carrying the marker (default: bracelet_link)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save calibration results")
    parser.add_argument("--settle-time", type=float, default=3.0,
                        help="Time in seconds for robot arm to settle before capture (default: 3.0)")
    parser.add_argument("--max-vel", type=float, default=0.15,
                        help="MoveIt trajectory execution max velocity scale (default: 0.15)")
    parser.add_argument("--max-accel", type=float, default=0.10,
                        help="MoveIt trajectory execution max acceleration scale (default: 0.10)")
    parser.add_argument("--capture-mode", choices=["manual", "automatic"],
                        default="manual",
                        help="Manual never commands the arm (default). Automatic visits the legacy joint pose list.")
    parser.add_argument("--allow-automatic-motion", action="store_true",
                        help="Required acknowledgement before automatic mode may command robot motion.")
    parser.add_argument("--no-return-to-start", dest="return_to_start",
                        action="store_false", default=True,
                        help="Do not return to the recorded startup joint configuration after a controlled automatic run.")
    parser.add_argument("--automatic-pose-source",
                        choices=["relative-cartesian", "legacy-joint"],
                        default="relative-cartesian",
                        help="Automatic sweep strategy. Relative Cartesian is centered on the startup pose (default).")
    parser.add_argument("--automatic-plan-only", action="store_true",
                        help="Plan the complete relative sweep sequentially without moving or capturing")
    parser.add_argument("--automatic-preposition-only", action="store_true",
                        help="Move only to the safer relative calibration anchor, then exit without capturing")
    parser.add_argument("--auto-translation-scale", type=float, default=1.0,
                        help="Scale the bounded relative translation sweep (default: 1.0)")
    parser.add_argument("--auto-rotation-scale", type=float, default=1.0,
                        help="Scale the bounded relative orientation sweep (default: 1.0)")
    parser.add_argument("--motion-countdown", type=int, default=10,
                        help="Abortable countdown before automatic hardware motion (default: 10 s)")
    parser.add_argument("--max-joint-state-age", type=float, default=0.5,
                        help="Maximum JointState receipt age allowed during automatic mode (default: 0.5 s)")
    parser.add_argument("--max-consecutive-capture-failures", type=int,
                        default=2,
                        help="Abort automatic motion after this many consecutive failed captures")
    parser.add_argument("--show-gui", action="store_true",
                        help="Open the OpenCV preview for the legacy automatic sweep; rqt_image_view is preferred remotely.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check image, CameraInfo, marker detection, and robot TF without moving or saving.")
    parser.add_argument("--preflight-timeout", type=float, default=20.0,
                        help="Seconds to wait for image and CameraInfo (default: 20).")
    parser.add_argument("--requested-samples", type=int, default=15,
                        help="Manual capture opportunities (default: 15).")
    parser.add_argument("--min-samples", type=int, default=12,
                        help="Minimum valid samples required to solve (default: 12).")
    parser.add_argument("--min-translation-span", type=float, default=0.10,
                        help="Minimum gripper-position span in metres (default: 0.10).")
    parser.add_argument("--min-rotation-span-deg", type=float, default=25.0,
                        help="Minimum relative gripper rotation span (default: 25 deg).")
    parser.add_argument("--max-translation-rms", type=float, default=0.010,
                        help="Maximum accepted transform-consistency RMS in metres (default: 0.010).")
    parser.add_argument("--max-rotation-rms-deg", type=float, default=2.0,
                        help="Maximum accepted transform-consistency RMS in degrees (default: 2.0).")
    parser.add_argument("--image-topic", type=str, default="",
                        help="Override the selected profile's image topic.")
    parser.add_argument("--image-transport",
                        choices=["auto", "raw", "compressed"], default="auto",
                        help="Image transport. Auto uses compressed for RealSense and raw for other profiles (default: auto).")
    parser.add_argument("--info-topic", type=str, default="",
                        help="Override the selected profile's CameraInfo topic.")
    parser.add_argument("--camera-frame", type=str, default="",
                        help="Fallback frame if CameraInfo has no frame ID.")
    
    # Custom camera settings (only used if --camera custom is chosen)
    parser.add_argument("--custom-image-topic", type=str, default="",
                        help="Image topic for custom camera setup")
    parser.add_argument("--custom-info-topic", type=str, default="",
                        help="CameraInfo topic for custom camera setup")
    parser.add_argument("--custom-camera-frame", type=str, default="",
                        help="Camera frame name for custom camera setup")
    
    # Target calibration board parameters
    parser.add_argument("--target-type", choices=["single", "grid"],
                        default="single",
                        help="Use one/legacy markers or a measured planar grid (default: single)")
    parser.add_argument("--grid-cols", type=int, default=4,
                        help="Number of marker columns for --target-type grid")
    parser.add_argument("--grid-rows", type=int, default=3,
                        help="Number of marker rows for --target-type grid")
    parser.add_argument("--grid-marker-ids", type=str,
                        default="2,5,8,11,1,4,7,10,0,3,6,9",
                        help="Physical row-major marker IDs, top-left to bottom-right")
    parser.add_argument("--marker-gap", type=float, default=0.00505,
                        help="Edge-to-edge grid marker separation in meters")
    parser.add_argument("--grid-corner-shift", type=int, default=1,
                        choices=[0, 1, 2, 3],
                        help="Quarter-corner shift mapping decoded marker orientation to board geometry")
    parser.add_argument("--min-visible-markers", type=int, default=4,
                        help="Minimum grid markers needed for a valid frame (default: 4)")
    parser.add_argument("--max-reprojection-error-px", type=float, default=5.0,
                        help="Reject grid poses above this corner RMS in pixels (default: 5.0)")

    # Legacy sparse-marker container details (only for --target-type single)
    parser.add_argument("--all-marker-ids", type=str, default="8",
                        help="Comma-separated list of all marker IDs on the board")
    parser.add_argument("--rotated-180-ids", type=str, default="",
                        help="Marker IDs rotated 180 deg around normal axis")
    parser.add_argument("--rect-width", type=float, default=0.23,
                        help="Distance BL to BR center in meters")
    parser.add_argument("--rect-height", type=float, default=0.21,
                        help="Distance BL to TL center in meters")
    
    # Physics bounds validation parameters
    parser.add_argument("--approx-x", type=float, default=None,
                        help="Manual physical x distance from camera to base_link")
    parser.add_argument("--approx-y", type=float, default=None,
                        help="Manual physical y distance from camera to base_link")
    parser.add_argument("--approx-z", type=float, default=None,
                        help="Manual physical z distance from camera to base_link")

    args = parser.parse_args(sys.argv[1:])

    if args.save_dir is None:
        args.save_dir = os.path.expanduser(
            f"~/Calibration_data/{args.camera}_{args.mode.replace('-', '_')}")
    if args.requested_samples < args.min_samples:
        parser.error("--requested-samples must be >= --min-samples")
    if args.min_samples < 4:
        parser.error("--min-samples must be >= 4")
    if args.marker_size <= 0.0:
        parser.error("--marker-size must be positive")
    if args.target_type == "grid":
        if args.grid_cols < 1 or args.grid_rows < 1:
            parser.error("--grid-cols and --grid-rows must be positive")
        if args.marker_gap < 0.0:
            parser.error("--marker-gap cannot be negative")
        marker_count = args.grid_cols * args.grid_rows
        if not 1 <= args.min_visible_markers <= marker_count:
            parser.error(
                "--min-visible-markers must be between 1 and the number of grid markers")
    if args.capture_mode == "automatic":
        if args.automatic_plan_only and args.automatic_preposition_only:
            parser.error(
                "--automatic-plan-only and --automatic-preposition-only are "
                "mutually exclusive")
        if not args.automatic_plan_only and not args.allow_automatic_motion:
            parser.error(
                "automatic mode can move the real arm; add "
                "--allow-automatic-motion only during a supervised run")
        if args.motion_countdown < 5:
            parser.error("--motion-countdown must be at least 5 seconds")
        if args.max_joint_state_age <= 0.0:
            parser.error("--max-joint-state-age must be positive")
        if args.max_consecutive_capture_failures < 1:
            parser.error("--max-consecutive-capture-failures must be positive")
        if not 0.65 <= args.auto_translation_scale <= 1.0:
            parser.error("--auto-translation-scale must be between 0.65 and 1.0")
        if not 0.85 <= args.auto_rotation_scale <= 1.0:
            parser.error("--auto-rotation-scale must be between 0.85 and 1.0")
        if (args.automatic_pose_source == "relative-cartesian"
                and args.requested_samples > 15):
            parser.error(
                "the relative Cartesian sweep supports at most 15 samples")
        if (args.automatic_pose_source == "legacy-joint"
                and not args.show_gui):
            parser.error("the legacy joint sweep requires --show-gui")
        if (args.automatic_plan_only
                and args.automatic_pose_source != "relative-cartesian"):
            parser.error(
                "--automatic-plan-only supports only relative-cartesian poses")
        if (args.automatic_pose_source == "legacy-joint"
                and args.min_samples > len(EYE_TO_HAND_POSES)):
            parser.error(
                "automatic mode has only 10 legacy poses; use manual mode "
                "for the required 12+ samples, or explicitly lower "
                "--min-samples after lab approval")
    
    # Map hand-to-eye alias to eye-in-hand
    if args.mode == "hand-to-eye":
        args.mode = "eye-in-hand"
        
    # Fix effector link defaults depending on mode
    if args.mode == "eye-in-hand" and args.robot_effector_frame == "end_effector_link":
        args.robot_effector_frame = "bracelet_link"
        
    rclpy.init(args=None)
    node = UnifiedCameraCalibrationNode(args)
    
    # Spin in a separate thread so main execution loop isn't blocked by spin events
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    success = False
    interrupted = False
    try:
        success = bool(node.run())
    except KeyboardInterrupt:
        interrupted = True
        node.get_logger().info("Calibration script interrupted by keyboard request.")
    finally:
        _safe_destroy_windows()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)
    if not success and not interrupted:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
