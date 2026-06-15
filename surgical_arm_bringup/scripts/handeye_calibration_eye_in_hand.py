#!/usr/bin/env python3
"""
Automated Eye-in-Hand Calibration for Kinova Gen3 7DOF

This script automatically moves the robot arm to various configurations,
captures images of an ArUco marker from the wrist-mounted camera (on bracelet_link),
and solves the hand-eye calibration problem using OpenCV.

Author: Antigravity AI
Date: 2026-05-22
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
# Unscrambled to match the alphabetically sorted driver.
DEFAULT_POSES_RAD = [
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

class AutoCalibEyeInHandNode(Node):
    def __init__(self, args):
        super().__init__("auto_handeye_calib_eye_in_hand")
        self.args = args
        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        self.latest_frame = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # TF2 listener to query robot pose (base_link to bracelet_link)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
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

        # Build offset + rotation table for multi-marker calibration.
        # Layout (z-axis of all markers points up toward camera):
        #   TL(3) ---Lx--- TR(2)    markers 0,3: normal orientation
        #   BL(0) ---Lx--- BR(1)    markers 1,2: rotated 180° around z (upside-down)
        # x = rightward (BL→BR) and y = upward (BL→TL) in the REFERENCE frame.
        Lx = self.args.rect_width   # BL→BR = 0.23 m
        Ly = self.args.rect_height  # BL→TL = 0.21 m

        # Positions of each marker centre in the reference (marker 0 / BL) frame
        _pos_in_ref = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([Lx,  0.0, 0.0]),
            2: np.array([Lx,  Ly,  0.0]),
            3: np.array([0.0, Ly,  0.0]),
        }

        # Rotation from reference frame into each marker's local frame.
        # Normal markers (0, 3): identity.
        # 180°-rotated markers (1, 2): Rz(180°) = diag(-1, -1, +1).
        Rz180 = np.array([[-1., 0., 0.],
                           [ 0., -1., 0.],
                           [ 0.,  0., 1.]])
        rotated_ids = {int(x.strip())
                       for x in self.args.rotated_180_ids.split(",") if x.strip()}
        self.R_ref2m = {
            mid: Rz180.copy() if mid in rotated_ids else np.eye(3)
            for mid in _pos_in_ref
        }

        ref_pos = _pos_in_ref.get(self.args.marker_id, np.zeros(3))
        # offset_to_ref[m] = vector from marker m's centre to the reference centre,
        # expressed IN MARKER M'S OWN FRAME (needed for the solvePnP transform).
        self.offset_to_ref = {
            mid: self.R_ref2m[mid] @ (ref_pos - pos)
            for mid, pos in _pos_in_ref.items()
        }

        self.all_marker_ids = [int(x.strip())
                               for x in self.args.all_marker_ids.split(",")]
        
        # Poses definition
        self.poses = DEFAULT_POSES_RAD
        if self.args.poses_file:
            self.load_custom_poses(self.args.poses_file)
            
        # Pymoveit2 robot arm controller
        self.get_logger().info("Initializing MoveIt2 arm controller...")
        joint_names = [f"joint_{i}" for i in range(1, 8)]
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=joint_names,
            base_link_name=self.args.robot_base_frame,
            end_effector_name="end_effector_link", # MoveIt planning frame
            group_name="manipulator",
            callback_group=self.cb_group
        )
        self.moveit2.planner_id = "PTP"
        self.moveit2.max_velocity = self.args.max_vel
        self.moveit2.max_acceleration = self.args.max_accel
        
        # Setup topics
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to RGB image feed
        self.get_logger().info(f"Subscribing to image topic: {self.args.camera_topic}")
        self.create_subscription(
            Image,
            self.args.camera_topic,
            self._image_cb,
            camera_qos,
            callback_group=self.cb_group
        )
        
        # Subscribe to camera info to dynamically retrieve camera matrix (K) and distortion (D)
        self.get_logger().info(f"Subscribing to camera info: {self.args.camera_info_topic}")
        self.create_subscription(
            CameraInfo,
            self.args.camera_info_topic,
            self._info_cb,
            camera_qos,
            callback_group=self.cb_group
        )
        
        # Hand-eye calibration datasets
        self.R_g2b = []  # Gripper (bracelet) to Base rotations
        self.t_g2b = []  # Gripper (bracelet) to Base translations
        self.R_t2c = []  # Target (marker) to Camera rotations
        self.t_t2c = []  # Target (marker) to Camera translations
        self.sample_count = 0
        
        os.makedirs(self.args.save_dir, exist_ok=True)
        self.get_logger().info("Calibration Node ready.")

    def load_custom_poses(self, filepath):
        try:
            poses = []
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [float(x) for x in line.replace(",", " ").split()]
                    if len(parts) == 7:
                        poses.append(parts)
            if poses:
                self.poses = poses
                self.get_logger().info(f"Loaded {len(poses)} custom poses from {filepath}")
            else:
                self.get_logger().warn(f"No valid joint configurations found in {filepath}. Using defaults.")
        except Exception as e:
            self.get_logger().error(f"Failed to load poses from {filepath}: {str(e)}. Using defaults.")

    def _image_cb(self, msg):
        try:
            with self.lock:
                self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image bridge error: {str(e)}")

    def _info_cb(self, msg):
        if self.camera_matrix is not None:
            return  # Already initialized
        with self.lock:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().info("Dynamically initialized camera parameters successfully!")
            self.get_logger().info(f"Intrinsic Matrix K:\n{self.camera_matrix}")
            self.get_logger().info(f"Distortion Coefficients D: {self.dist_coeffs}")

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
        Try each marker in all_marker_ids (reference ID first).
        For any detected marker, compute the equivalent pose of the reference marker
        by applying the known geometric offset between them on the shared flat board.
        This lets any of the 4 corner markers serve as a valid calibration target.
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

        # Search order: reference marker first, then the rest
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
                # Direct detection of the reference marker
                label = f"Marker {target_id} (ref) OK"
                cv2.putText(vis, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return True, rvec, tvec, vis

            # Detected a non-reference marker — compensate to get reference pose.
            # offset is already expressed in marker m's frame (accounts for 180° flip).
            # R_ref2cam = R_det2cam @ R_ref2m  (chain: ref→marker_m→camera).
            R_det2cam, _ = cv2.Rodrigues(rvec)
            offset = self.offset_to_ref.get(target_id, np.zeros(3))
            R_ref2m = self.R_ref2m.get(target_id, np.eye(3))
            t_ref_in_cam = (R_det2cam @ offset + tvec.flatten()).reshape(3, 1)
            R_ref2cam = R_det2cam @ R_ref2m
            rvec_ref, _ = cv2.Rodrigues(R_ref2cam)

            label = f"Marker {target_id}→ref({ref_id}) OK"
            cv2.putText(vis, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            return True, rvec_ref, t_ref_in_cam, vis

        cv2.putText(vis, f"None of markers {self.all_marker_ids} found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False, None, None, vis

    def get_robot_pose(self):
        """
        Query TF to get the transform from base_link to the designated effector frame (bracelet_link).
        For eye-in-hand calibration, this transform corresponds to gripper-to-base (T_g2b).
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.robot_base_frame,
                self.args.robot_effector_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            
            # Convert quaternion (x, y, z, w) to rotation matrix
            rot_matrix = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            t_vector = np.array([t.x, t.y, t.z])
            return True, rot_matrix, t_vector
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed from {self.args.robot_base_frame} to {self.args.robot_effector_frame}: {str(e)}")
            return False, None, None

    def execute_move(self, joints_rad):
        self.moveit2.move_to_configuration(joints_rad)
        self.moveit2.wait_until_executed()

    def get_searched_pose(self, pose, rx, ry, rz):
        """Rotate pose relative to its current orientation by rx, ry, rz radians in tool frame."""
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
        # Settle and verify that we can capture a sharp frame where the marker is seen
        best_sharpness = -1.0
        best_frame_data = None
        start_time = time.time()
        
        # Check frames over a 1.5 second window to find the sharpest, most in-focus frame
        while time.time() - start_time < 1.5:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            
            if frame is not None:
                ok_marker, rvec, tvec, vis = self.detect_marker(frame)
                if ok_marker:
                    # Calculate image sharpness (variance of Laplacian)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame_data = (rvec, tvec, vis, frame)
            
            time.sleep(0.05) # Poll at ~20 Hz
            
        if best_frame_data is None:
            self.get_logger().warn(f"  [Capture Fail] ArUco Marker {self.args.marker_id} was not detected in this pose.")
            # Fallback: save whatever frame we had as a diagnostics image
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is not None:
                cv2.imwrite(os.path.join(self.args.save_dir, f"pose_{idx+1:02d}_NO_MARKER.png"), frame)
            return False
            
        rvec, tvec, vis, best_raw_frame = best_frame_data
        
        ok_tf, rot_g2b, t_g2b = self.get_robot_pose()
        if not ok_tf:
            self.get_logger().error("  [Capture Fail] Could not query robot transform from TF.")
            return False
            
        # Success - Save frame and store measurements
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
        self.get_logger().info(f"Computing eye-in-hand calibration using {self.sample_count} samples...")
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
        
        for name, method in solvers:
            try:
                # In eye-in-hand calibration:
                # R_gripper2base (R_g2b), t_gripper2base (t_g2b)
                # R_target2cam (R_t2c), t_target2cam (t_t2c)
                # Returns: R_cam2gripper (rotation of camera relative to gripper)
                #          t_cam2gripper (translation of camera relative to gripper)
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
            except Exception as e:
                self.get_logger().warn(f"  Solver {name}: FAILED - {str(e)}")
                
        if best_R is None:
            self.get_logger().error("All hand-eye calibration solvers failed. Check your marker detections and robot poses.")
            return False
            
        tf_trans = best_t.flatten()
        quat = R.from_matrix(best_R).as_quat()  # [qx, qy, qz, qw]
        rpy = R.from_matrix(best_R).as_euler("xyz", degrees=True)
        
        yaml_path = os.path.join(self.args.save_dir, "eye_in_hand_calib.yaml")
        with open(yaml_path, "w") as f:
            f.write("# Auto-generated Eye-in-Hand Hand-Eye Calibration Result\n")
            f.write(f"# Solver used: {best_name}\n")
            f.write(f"# Robot Effector Link: {self.args.robot_effector_frame}\n")
            f.write(f"# Camera Frame: {self.args.camera_frame}\n\n")
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
            
        self.get_logger().info("================================================================================")
        self.get_logger().info("CALIBRATION COMPLETED SUCCESSFULLY!")
        self.get_logger().info(f"Resulting transform: {self.args.robot_effector_frame} -> {self.args.camera_frame}")
        self.get_logger().info(f"Translation (meters) : X={tf_trans[0]:.5f}, Y={tf_trans[1]:.5f}, Z={tf_trans[2]:.5f}")
        self.get_logger().info(f"Euler Angles (deg)   : R={rpy[0]:.3f}, P={rpy[1]:.3f}, Y={rpy[2]:.3f}")
        self.get_logger().info(f"Quaternion (xyzw)    : [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        self.get_logger().info("--------------------------------------------------------------------------------")
        self.get_logger().info("Copy-paste standard TF publisher command:")
        self.get_logger().info(f"ros2 run tf2_ros static_transform_publisher {tf_trans[0]:.6f} {tf_trans[1]:.6f} {tf_trans[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} {self.args.robot_effector_frame} {self.args.camera_frame}")
        self.get_logger().info("================================================================================")
        self.get_logger().info(f"Calibration configurations exported to: {yaml_path}")
        return True

    def run(self):
        # Wait for camera stream
        self.get_logger().info("Waiting for active camera images...")
        rate = self.create_rate(10)
        while rclpy.ok():
            with self.lock:
                frame_ready = self.latest_frame is not None
            if frame_ready:
                break
            rate.sleep()
            
        self.get_logger().info("Camera stream connected. Initializing GUI...")
        
        cv2.namedWindow("Eye-in-Hand Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Eye-in-Hand Calibration", 848, 480)
        
        total_poses = len(self.poses)
        
        if self.args.mode == "auto":
            if self.args.auto_pose_type == "dynamic":
                self.get_logger().info("DYNAMIC MODE: Localizing marker for dynamic look-at pose generation...")
                # Settle for 1 second to ensure we have a good initial image
                time.sleep(1.0)
                ok_marker, marker_pos = self.find_marker_in_base()
                
                # Get camera's current pose in base frame as start/home pose
                try:
                    tf_start = self.tf_buffer.lookup_transform(
                        self.args.robot_base_frame,
                        self.args.camera_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=2.0)
                    )
                    t_start = tf_start.transform.translation
                    q_start = tf_start.transform.rotation
                    p_start = np.array([t_start.x, t_start.y, t_start.z])
                    R_start = R.from_quat([q_start.x, q_start.y, q_start.z, q_start.w]).as_matrix()
                except Exception as e:
                    self.get_logger().error(f"Failed to get starting camera pose: {str(e)}")
                    return
                
                if not ok_marker:
                    self.get_logger().warn("Marker NOT detected at startup! Using fallback look-at point 0.4m along camera optical axis.")
                    marker_pos = p_start + R_start @ np.array([0.0, 0.0, 0.4])
                
                # Generate dynamic effector poses
                self.get_logger().info("Generating dynamic target poses...")
                target_poses = self.generate_dynamic_poses(p_start, R_start, marker_pos)
                
                total_poses = len(target_poses)
                self.get_logger().info(f"Visiting {total_poses} dynamic look-at calibration poses...")
                
                for i, (pose_e, desc) in enumerate(target_poses):
                    if not rclpy.ok():
                        break
                        
                    self.get_logger().info(f"Moving to Pose {i+1}/{total_poses} ({desc})...")
                    
                    success = self.execute_pose_move(pose_e)
                    if not success:
                        self.get_logger().warn(f"Failed to plan/execute move to Pose {i+1}. Skipping.")
                        continue
                        
                    # Wait for the arm to settle and display settling visual feed
                    settle_start = time.time()
                    while time.time() - settle_start < self.args.settle_time:
                        with self.lock:
                            curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                        if curr_frame is not None:
                            _, _, _, vis = self.detect_marker(curr_frame)
                            cv2.putText(vis, f"Pose {i+1}/{total_poses} settling...", (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.imshow("Eye-in-Hand Calibration", vis)
                            cv2.waitKey(30)
                            
                    # Settle done, capture point
                    self.get_logger().info(f"Capturing data at Pose {i+1}...")
                    captured = self.capture_sample(i)
                    
                    # Active Search fallback for dynamic Cartesian poses
                    if not captured:
                        self.get_logger().warn("  [Active Search] ArUco Marker not seen at nominal pose. Starting Active Search...")
                        # Small angular search offsets in radians (rx, ry, rz in tool/camera frame)
                        search_offsets = [
                            (0.0,  0.10, 0.0),  # Yaw +6 deg
                            (0.0, -0.10, 0.0),  # Yaw -6 deg
                            (0.10, 0.0,  0.0),  # Pitch +6 deg
                            (-0.10, 0.0, 0.0),  # Pitch -6 deg
                            (0.0,  0.0,  0.26), # Roll +15 deg
                            (0.0,  0.0, -0.26), # Roll -15 deg
                        ]
                        
                        for idx_s, (rx, ry, rz) in enumerate(search_offsets):
                            self.get_logger().info(f"    [Active Search {idx_s+1}/{len(search_offsets)}] Moving with offset: rx={math.degrees(rx):.1f}°, ry={math.degrees(ry):.1f}°, rz={math.degrees(rz):.1f}°...")
                            pose_searched = self.get_searched_pose(pose_e, rx, ry, rz)
                            if self.execute_pose_move(pose_searched):
                                time.sleep(1.0) # Settle briefly
                                captured = self.capture_sample(i)
                                if captured:
                                    self.get_logger().info("    [Active Search] SUCCESS! Marker found and captured.")
                                    break
                            else:
                                self.get_logger().warn(f"    [Active Search] Move failed for offset {idx_s+1}.")
                        
                        # Return to nominal pose if active search also failed (to keep it clean)
                        if not captured:
                            self.get_logger().error("    [Active Search] FAILED — Marker could not be found.")
                            self.execute_pose_move(pose_e)

                    # Show capture visual feedback
                    with self.lock:
                        curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                    if curr_frame is not None:
                        _, _, _, vis = self.detect_marker(curr_frame)
                        color = (0, 255, 0) if captured else (0, 0, 255)
                        status_str = f"Captured Sample {self.sample_count}" if captured else "CAPTURE FAILED"
                        cv2.putText(vis, status_str, (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.imshow("Eye-in-Hand Calibration", vis)
                        cv2.waitKey(1000)
            else:
                self.get_logger().info(f"STATIC AUTO MODE: Visiting {total_poses} predefined static poses automatically...")
                
                for i, joints in enumerate(self.poses):
                    if not rclpy.ok():
                        break
                        
                    self.get_logger().info(f"Moving to Pose {i+1}/{total_poses}...")
                    self.execute_move(joints)
                    
                    # Wait for the arm to settle and display settling visual feed
                    settle_start = time.time()
                    while time.time() - settle_start < self.args.settle_time:
                        with self.lock:
                            curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                        if curr_frame is not None:
                            _, _, _, vis = self.detect_marker(curr_frame)
                            cv2.putText(vis, f"Pose {i+1}/{total_poses} settling...", (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.imshow("Eye-in-Hand Calibration", vis)
                            cv2.waitKey(30)
                            
                    # Settle done, capture point
                    self.get_logger().info(f"Capturing data at Pose {i+1}...")
                    captured = self.capture_sample(i)
                    
                    # Active Search fallback for static Joint configuration
                    if not captured:
                        self.get_logger().warn("  [Active Search] ArUco Marker not seen at nominal joints. Starting Active Search...")
                        # Small angular search joint offsets in radians (joint_6, joint_7)
                        search_joint_offsets = [
                            ( 0.10, 0.0),  # joint_6 +6 deg
                            (-0.10, 0.0),  # joint_6 -6 deg
                            ( 0.0,  0.26), # joint_7 +15 deg
                            ( 0.0, -0.26), # joint_7 -15 deg
                            ( 0.10, 0.26), # joint_6 +6, joint_7 +15
                            (-0.10, -0.26),# joint_6 -6, joint_7 -15
                        ]
                        
                        for idx_s, (dj6, dj7) in enumerate(search_joint_offsets):
                            self.get_logger().info(f"    [Active Search {idx_s+1}/{len(search_joint_offsets)}] Moving with joint offsets: j6={math.degrees(dj6):.1f}°, j7={math.degrees(dj7):.1f}°...")
                            searched_joints = list(joints)
                            searched_joints[5] += dj6  # joint_6 is index 5
                            searched_joints[6] += dj7  # joint_7 is index 6
                            
                            self.execute_move(searched_joints)
                            time.sleep(1.0) # Settle briefly
                            captured = self.capture_sample(i)
                            if captured:
                                self.get_logger().info("    [Active Search] SUCCESS! Marker found and captured.")
                                break
                                
                        # Return to nominal joints if active search failed
                        if not captured:
                            self.get_logger().error("    [Active Search] FAILED — Marker could not be found.")
                            self.execute_move(joints)

                    # Show capture visual feedback
                    with self.lock:
                        curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                    if curr_frame is not None:
                        _, _, _, vis = self.detect_marker(curr_frame)
                        color = (0, 255, 0) if captured else (0, 0, 255)
                        status_str = f"Captured Sample {self.sample_count}" if captured else "CAPTURE FAILED"
                        cv2.putText(vis, status_str, (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.imshow("Eye-in-Hand Calibration", vis)
                        cv2.waitKey(1000)
                    
        else:  # Interactive / Manual Mode
            self.get_logger().info("INTERACTIVE MODE ACTIVE.")
            self.get_logger().info("Use RViz or manual kinesthetic guide to move the robot arm.")
            self.get_logger().info("Ensure the wrist camera can clearly see the ArUco marker.")
            self.get_logger().info("Commands:")
            self.get_logger().info("  [Press Enter] in terminal to capture a sample at the current pose.")
            self.get_logger().info("  [Type 'q' and Enter] to finish capturing and calculate calibration.")
            
            # Interactive keyboard thread
            user_input = None
            def get_input():
                nonlocal user_input
                while rclpy.ok():
                    inp = input().strip().lower()
                    with self.lock:
                        user_input = inp if inp else "c"
                    if inp == "q":
                        break
                        
            input_thread = threading.Thread(target=get_input, daemon=True)
            input_thread.start()
            
            idx = 0
            while rclpy.ok():
                with self.lock:
                    curr_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                    cmd = user_input
                    user_input = None  # Reset
                    
                if cmd == "q":
                    self.get_logger().info("Interactive pose capture finished by user request.")
                    break
                elif cmd == "c":
                    self.get_logger().info(f"Triggering interactive capture for Pose #{idx+1}...")
                    captured = self.capture_sample(idx)
                    if captured:
                        idx += 1
                    else:
                        self.get_logger().warn("Interactive capture failed. Try adjusting the arm pose or marker orientation.")
                        
                # Update visual display continuously
                if curr_frame is not None:
                    _, _, _, vis = self.detect_marker(curr_frame)
                    cv2.putText(vis, "INTERACTIVE MODE", (10, vis.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(vis, f"Samples: {self.sample_count} | [Enter] Capture | [q] Solve", (10, vis.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow("Eye-in-Hand Calibration", vis)
                    cv2.waitKey(30)
                else:
                    rate.sleep()
                    
        cv2.destroyAllWindows()
        self.solve_calibration()
        
        # Return to insert_to_container home position if return_home is set
        if self.args.return_home:
            self.get_logger().info("Returning to insert_to_container home position...")
            # Calibrated home position joints in radians (alphabetical sorting: joint_1 to joint_7):
            # joint_1:  0.0000, joint_2: -0.3049, joint_3: -3.1416,
            # joint_4: -1.6607, joint_5:  0.0000, joint_6: -1.7928, joint_7: -0.0006
            home_joints = [0.0000, -0.3049, -3.1416, -1.6607, 0.0000, -1.7928, -0.0006]
            self.execute_move(home_joints)
            self.get_logger().info("Arm successfully returned to home position.")

    def find_marker_in_base(self):
        self.get_logger().info("Searching for marker to initialize autonomous trajectory...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is not None:
                ok_marker, rvec, tvec, _ = self.detect_marker(frame)
                if ok_marker:
                    try:
                        tf = self.tf_buffer.lookup_transform(
                            self.args.robot_base_frame,
                            self.args.camera_frame,
                            rclpy.time.Time(),
                            timeout=rclpy.duration.Duration(seconds=1.0)
                        )
                        t = tf.transform.translation
                        q = tf.transform.rotation
                        R_c2b = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                        t_c2b = np.array([t.x, t.y, t.z])
                        
                        p_marker_cam = tvec.flatten()
                        p_marker_base = R_c2b @ p_marker_cam + t_c2b
                        self.get_logger().info(f"Marker localized in base frame at: X={p_marker_base[0]:.3f}, Y={p_marker_base[1]:.3f}, Z={p_marker_base[2]:.3f}")
                        return True, p_marker_base
                    except Exception as e:
                        self.get_logger().warn(f"TF lookup failed while localizing marker: {str(e)}")
            time.sleep(0.2)
        return False, None

    def generate_dynamic_poses(self, p_start, R_start, marker_pos):
        try:
            tf_nominal = self.tf_buffer.lookup_transform(
                self.args.robot_effector_frame,
                self.args.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            t_c2e = np.array([tf_nominal.transform.translation.x,
                              tf_nominal.transform.translation.y,
                              tf_nominal.transform.translation.z])
            q_c2e = tf_nominal.transform.rotation
            R_c2e = R.from_quat([q_c2e.x, q_c2e.y, q_c2e.z, q_c2e.w]).as_matrix()
        except Exception as e:
            self.get_logger().error(f"Failed to lookup nominal camera-effector offset: {str(e)}. Using standard fallback.")
            t_c2e = np.zeros(3)
            R_c2e = np.eye(3)

        gs = self.args.grid_size
        hv = self.args.height_variation
        rv = np.radians(self.args.roll_variation)
        
        # Grid of (dx, dy, dz, roll) offsets in camera's start frame
        offsets = [
            (0.00,      0.00,      0.00,       0.0),       # Pose 1: Home
            (-1.0 * gs, 0.00,      -0.3 * hv,  -1.0 * rv), # Pose 2: Left, closer, roll -
            (1.0 * gs,  0.00,      -0.3 * hv,  1.0 * rv),  # Pose 3: Right, closer, roll +
            (0.00,      -1.0 * gs, 0.3 * hv,   0.0),       # Pose 4: Up/Forward
            (0.00,      1.0 * gs,  0.3 * hv,   0.0),       # Pose 5: Down/Backward
            (-0.7 * gs, -0.7 * gs, -0.6 * hv,  1.3 * rv),  # Pose 6: Diagonal Top-Left
            (0.7 * gs,  0.7 * gs,  -0.6 * hv,  -1.3 * rv), # Pose 7: Diagonal Bottom-Right
            (0.7 * gs,  -0.7 * gs, 0.6 * hv,   -1.3 * rv), # Pose 8: Diagonal Top-Right
            (-0.7 * gs, 0.7 * gs,  0.6 * hv,   1.3 * rv),  # Pose 9: Diagonal Bottom-Left
            (0.00,      0.00,      -1.0 * hv,  0.7 * rv),  # Pose 10: Center-Close
            (0.00,      0.00,      1.3 * hv,   -0.7 * rv), # Pose 11: Center-Far
            (-0.5 * gs, 0.5 * gs,  0.3 * hv,   1.0 * rv),  # Pose 12: Extra diverse pose
        ]
        
        target_poses = []
        for idx, (dx, dy, dz, roll) in enumerate(offsets):
            p_cam = p_start + R_start @ np.array([dx, dy, dz])
            
            v_z = marker_pos - p_cam
            norm_z = np.linalg.norm(v_z)
            if norm_z < 1e-4:
                z_c = R_start[:, 2]
            else:
                z_c = v_z / norm_z
                
            x_home = R_start[:, 0]
            v_x = x_home - np.dot(x_home, z_c) * z_c
            norm_x = np.linalg.norm(v_x)
            if norm_x < 1e-4:
                x_c = R_start[:, 0]
            else:
                x_c = v_x / norm_x
                
            y_c = np.cross(z_c, x_c)
            R_look = np.column_stack((x_c, y_c, z_c))
            
            R_z = np.array([
                [np.cos(roll), -np.sin(roll), 0.0],
                [np.sin(roll), np.cos(roll),  0.0],
                [0.0,          0.0,           1.0]
            ])
            R_cam = R_look @ R_z
            
            # Effector target pose
            R_eff = R_cam @ R_c2e.T
            p_eff = p_cam - R_eff @ t_c2e
            
            pose_e = Pose()
            pose_e.position.x = p_eff[0]
            pose_e.position.y = p_eff[1]
            pose_e.position.z = p_eff[2]
            q_e = R.from_matrix(R_eff).as_quat()
            pose_e.orientation.x = q_e[0]
            pose_e.orientation.y = q_e[1]
            pose_e.orientation.z = q_e[2]
            pose_e.orientation.w = q_e[3]
            
            desc = f"dx={dx:+.2f}m, dy={dy:+.2f}m, dz={dz:+.2f}m, roll={np.degrees(roll):+.1f}°"
            target_poses.append((pose_e, desc))
            
        return target_poses

    def execute_pose_move(self, pose):
        try:
            self.moveit2.move_to_pose(
                pose=pose,
                target_link=self.args.robot_effector_frame,
                frame_id=self.args.robot_base_frame
            )
            success = self.moveit2.wait_until_executed()
            return success
        except Exception as e:
            self.get_logger().error(f"Motion planning/execution error: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Kinova Gen3 7DOF Eye-in-Hand Hand-Eye Calibration")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "interactive"],
                        help="Calibration mode: 'auto' (visits preset joint configurations) or 'interactive' (user guides arm manually)")
    parser.add_argument("--auto-pose-type", type=str, default="dynamic", choices=["dynamic", "static"],
                        help="Type of poses for auto mode: 'dynamic' (auto-generates a look-at hemisphere around the marker) or 'static' (legacy hardcoded joints)")
    parser.add_argument("--grid-size", type=float, default=0.10,
                        help="Lateral translation offset step size in meters for dynamic pose generation (default: 10 cm)")
    parser.add_argument("--height-variation", type=float, default=0.06,
                        help="Depth/height translation variation range in meters for dynamic pose generation (default: 6 cm)")
    parser.add_argument("--roll-variation", type=float, default=15.0,
                        help="Roll angle variation range in degrees for dynamic pose generation (default: 15°)")
    parser.add_argument("--return-home", action="store_true", default=True,
                        help="Return the robot to the insert_to_container home position after calibration completes")
    parser.add_argument("--poses-file", type=str, default=None,
                        help="Optional file path to load custom joint angles (each line having 7 space-separated angles in radians)")
    parser.add_argument("--camera-topic", type=str, default="/camera/color/image_raw",
                        help="ROS2 Image topic for the wrist camera feed")
    parser.add_argument("--camera-info-topic", type=str, default="/camera/color/camera_info",
                        help="ROS2 CameraInfo topic to read intrinsics dynamically")
    parser.add_argument("--robot-base-frame", type=str, default="world",
                        help="Base planning link frame (matches combine_cameras reference_frame, typically 'world')")
    parser.add_argument("--robot-effector-frame", type=str, default="bracelet_link",
                        help="Robot effector wrist link carrying the camera (typically 'bracelet_link')")
    parser.add_argument("--camera-frame", type=str, default="camera_color_optical_frame",
                        help="Camera optical frame to calibrate (typically 'camera_color_optical_frame')")
    parser.add_argument("--aruco-dict", type=str, default="DICT_4X4_50",
                        help="ArUco dictionary configuration name (matches combine_cameras aruco_dictionary_name)")
    parser.add_argument("--marker-id", type=int, default=0,
                        help="Reference marker ID — calibration is expressed relative to this marker (BL=0, BR=1, TR=2, TL=3)")
    parser.add_argument("--all-marker-ids", type=str, default="0,1,2,3",
                        help="Comma-separated list of all corner marker IDs to try at each pose (any visible one is used)")
    parser.add_argument("--rotated-180-ids", type=str, default="1,2",
                        help="Comma-separated marker IDs that are physically rotated 180° around z relative to the reference (markers 1 and 2 on this board)")
    parser.add_argument("--rect-width", type=float, default=0.23,
                        help="Centre-to-centre distance between BL(0) and BR(1) markers in metres (x-axis, measured: 23 cm)")
    parser.add_argument("--rect-height", type=float, default=0.21,
                        help="Centre-to-centre distance between BL(0) and TL(3) markers in metres (y-axis, measured: 21 cm)")
    parser.add_argument("--marker-size", type=float, default=0.05,
                        help="Physical size of one individual marker square in meters (5 cm measured)")
    parser.add_argument("--settle-time", type=float, default=3.0,
                        help="Settle wait time in seconds at each auto pose before capturing frame")
    parser.add_argument("--max-vel", type=float, default=0.15,
                        help="Max joint velocity scaling for MoveIt2 trajectory execution")
    parser.add_argument("--max-accel", type=float, default=0.10,
                        help="Max joint acceleration scaling for MoveIt2 trajectory execution")
    parser.add_argument("--save-dir", type=str, default=os.path.expanduser("~/Calibration_data/eye_in_hand"),
                        help="Directory to save calibration parameters yaml file and verification photos")
    
    # Parse only ROS2 args internally via rclpy, parse raw script args via argparse
    rcl_args = []
    script_args = sys.argv[1:]
    if "--ros-args" in sys.argv:
        idx = sys.argv.index("--ros-args")
        rcl_args = sys.argv[idx:]
        script_args = sys.argv[1:idx]
        
    args = parser.parse_args(script_args)
    
    rclpy.init(args=rcl_args)
    node = AutoCalibEyeInHandNode(args)
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    # Run ROS spin executor in a background daemon thread
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
