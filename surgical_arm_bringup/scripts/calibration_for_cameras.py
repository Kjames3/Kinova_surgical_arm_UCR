#!/usr/bin/env python3
"""
Unified Automated Hand-Eye Calibration Script for Kinova Gen3 7DOF

This script automatically calibrates camera frames (Intel RealSense, OAK-D,
or Kinova Wrist Camera) relative to the robot base frame (base_link / world).
It supports:
  1. Eye-in-Hand Calibration (Moving camera on wrist, static marker on table)
  2. Eye-to-Hand Calibration (Static camera, moving marker on gripper)

The script executes collision-safe trajectories via MoveIt, captures data
points, solves using OpenCV hand-eye calibration algorithms, validates results
against manual physical table measurements, and exports final static TF publishers.

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
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from cv_bridge import CvBridge
from pymoveit2 import MoveIt2
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

class UnifiedCameraCalibrationNode(Node):
    def __init__(self, args):
        super().__init__("unified_camera_calibration_node")
        self.args = args
        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        self.latest_frame = None
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
        
        # Define 3D coordinate points of the marker corners in the marker-local frame
        msize = self.args.marker_size
        self.marker_obj_pts = np.array([
            [-msize/2,  msize/2, 0],
            [ msize/2,  msize/2, 0],
            [ msize/2, -msize/2, 0],
            [-msize/2, -msize/2, 0]
        ], dtype=np.float32)

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
            
        # Pymoveit2 robot arm controller
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
        self.moveit2.planner_id = "PTP"
        self.moveit2.max_velocity = self.args.max_vel
        self.moveit2.max_acceleration = self.args.max_accel
        
        # Setup ROS 2 topic subscriptions
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.get_logger().info(f"Subscribing to image topic: {self.image_topic}")
        self.create_subscription(
            Image,
            self.image_topic,
            self._image_cb,
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
            self.image_topic = "/oakd/oak/rgb/image_raw"
            self.info_topic = "/oakd/oak/rgb/camera_info"
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

    def _image_cb(self, msg):
        try:
            with self.lock:
                self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image bridge error: {str(e)}")

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

    def get_camera_parameters(self):
        with self.lock:
            if self.camera_matrix is not None:
                return self.camera_matrix.copy(), self.dist_coeffs.copy()
        # Fallback values if camera_info topic is not yet active
        k_fallback = np.array([[909.9, 0.0, 640.0],
                               [0.0, 909.8, 360.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
        d_fallback = np.zeros(5, dtype=np.float64)
        return k_fallback, d_fallback

    def detect_marker(self, frame):
        """
        Detects ArUco marker from frame. Computes the offset to the reference marker
        if another marker on the board is detected.
        """
        K, D = self.get_camera_parameters()
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is None:
            vis = frame.copy()
            cv2.putText(vis, "No markers found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return False, None, None, vis

        vis = frame.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        ids_flat = ids.flatten().tolist()

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
            
            quat = R.from_matrix(R_mat).as_quat()
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
            rot_matrix = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            t_vector = np.array([t.x, t.y, t.z])
            return True, rot_matrix, t_vector
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {str(e)}")
            return False, None, None

    def execute_move(self, joints_rad):
        self.moveit2.move_to_configuration(joints_rad)
        self.moveit2.wait_until_executed()

    def get_searched_pose(self, pose, rx, ry, rz):
        """Rotate pose orientation by rx, ry, rz radians in effector frame."""
        q_nom = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        R_nom = R.from_quat(q_nom)
        R_rel = R.from_euler('xyz', [rx, ry, rz])
        R_new = R_nom * R_rel
        q_new = R_new.as_quat()
        
        pose_searched = Pose()
        pose_searched.position = pose.position
        pose_searched.orientation.x = q_new[0]
        pose_searched.orientation.y = q_new[1]
        pose_searched.orientation.z = q_new[2]
        pose_searched.orientation.w = q_new[3]
        return pose_searched

    def capture_sample(self, idx):
        """Verify sharp frame with detected marker, record gripper & marker transforms."""
        best_sharpness = -1.0
        best_frame_data = None
        start_time = time.time()
        
        # Evaluate frames over a 1.5s window to locate the sharpest frame
        while time.time() - start_time < 1.5:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            
            if frame is not None:
                ok_marker, rvec, tvec, vis = self.detect_marker(frame)
                if ok_marker:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame_data = (rvec, tvec, vis, frame)
            
            time.sleep(0.05)
            
        if best_frame_data is None:
            self.get_logger().warn(f"  [Capture Fail] ArUco Marker was not detected at this pose.")
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

    def solve_calibration(self):
        """Solves hand-eye calibration equations and prints static TF publishers."""
        self.get_logger().info(f"Computing hand-eye calibration using {self.sample_count} samples...")
        if self.sample_count < 4:
            self.get_logger().error("Calibration requires at least 4 unique spatial samples. Calculation aborted.")
            return False
            
        solvers = [
            ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
            ("PARK", cv2.CALIB_HAND_EYE_PARK),
            ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
            ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF)
        ]
        
        best_R = None
        best_t = None
        best_name = None
        
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
                    if R_cam2gripper is not None:
                        best_R = R_cam2gripper
                        best_t = t_cam2gripper
                        best_name = name
                        self.get_logger().info(f"  Solver {name}: SUCCESS")
                        break
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
                    if R_cam2base is not None:
                        best_R = R_cam2base
                        best_t = t_cam2base
                        best_name = name
                        self.get_logger().info(f"  Solver {name}: SUCCESS")
                        break
            except Exception as e:
                self.get_logger().warn(f"  Solver {name}: FAILED - {str(e)}")
                
        if best_R is None:
            self.get_logger().error("All hand-eye calibration solvers failed. Check your data.")
            return False
            
        tf_trans = best_t.flatten()
        quat = R.from_matrix(best_R).as_quat()  # [qx, qy, qz, qw]
        rpy = R.from_matrix(best_R).as_euler("xyz", degrees=True)
        
        # Decide frame naming
        camera_frame = self.detected_camera_frame if self.detected_camera_frame else self.default_camera_frame
        
        # Validation bounds check (for static table cameras)
        if self.args.mode == "eye-to-hand" and any([self.args.approx_x, self.args.approx_y, self.args.approx_z]):
            approx = np.array([self.args.approx_x or 0.0, self.args.approx_y or 0.0, self.args.approx_z or 0.0])
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
                self.get_logger().info("Discrepancy is within safe validation bounds (< 15cm). Calibration is sane.")
        
        # Export yaml
        yaml_name = f"{self.args.mode.replace('-', '_')}_calib.yaml"
        yaml_path = os.path.join(self.args.save_dir, yaml_name)
        with open(yaml_path, "w") as f:
            f.write(f"# Auto-generated Hand-Eye Calibration Result\n")
            f.write(f"# Solver: {best_name} | Mode: {self.args.mode}\n")
            f.write(f"# Parent Frame: {self.args.robot_base_frame if self.args.mode=='eye-to-hand' else self.args.robot_effector_frame}\n")
            f.write(f"# Camera Frame: {camera_frame}\n\n")
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
        self.get_logger().info(f"Awaiting active image frames on topic: {self.image_topic}...")
        rate = self.create_rate(10)
        while rclpy.ok():
            with self.lock:
                frame_ready = self.latest_frame is not None
            if frame_ready:
                break
            rate.sleep()
            
        self.get_logger().info("Camera stream detected. Initializing calibration display windows...")
        
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

        cv2.destroyAllWindows()
        self.solve_calibration()

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
    parser.add_argument("--robot-effector-frame", type=str, default="end_effector_link",
                        help="Robot end effector frame ID (default: end_effector_link)")
    parser.add_argument("--save-dir", type=str, default=os.path.expanduser("~/Calibration_data/auto_run"),
                        help="Directory to save calibration results")
    parser.add_argument("--settle-time", type=float, default=3.0,
                        help="Time in seconds for robot arm to settle before capture (default: 3.0)")
    parser.add_argument("--max-vel", type=float, default=0.15,
                        help="MoveIt trajectory execution max velocity scale (default: 0.15)")
    parser.add_argument("--max-accel", type=float, default=0.10,
                        help="MoveIt trajectory execution max acceleration scale (default: 0.10)")
    
    # Custom camera settings (only used if --camera custom is chosen)
    parser.add_argument("--custom-image-topic", type=str, default="",
                        help="Image topic for custom camera setup")
    parser.add_argument("--custom-info-topic", type=str, default="",
                        help="CameraInfo topic for custom camera setup")
    parser.add_argument("--custom-camera-frame", type=str, default="",
                        help="Camera frame name for custom camera setup")
    
    # Target calibration board parameters (multi-marker container details)
    parser.add_argument("--all-marker-ids", type=str, default="0,1,2,3",
                        help="Comma-separated list of all marker IDs on the board")
    parser.add_argument("--rotated-180-ids", type=str, default="1,2",
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
    
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Calibration script interrupted by keyboard request.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)

if __name__ == "__main__":
    main()