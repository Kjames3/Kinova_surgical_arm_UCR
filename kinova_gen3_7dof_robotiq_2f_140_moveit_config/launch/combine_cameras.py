#!/usr/bin/env python3
"""
Combine Cameras & Multi-Camera ArUco Marker Sensor Fusion Node

================================================================================
Goal & System Overview
================================================================================
The objective of this script is to perform real-time, multi-camera sensor fusion 
to accurately determine the 3D center point and spatial orientation of a square 
target area (e.g., a container or docking tray) located on a table in front of the 
Kinova Gen3 7DOF robot.

The physical square target is marked by 4 distinct ArUco markers, one positioned 
at each corner:
  - Corner 1: Top-Left (TL)
  - Corner 2: Top-Right (TR)
  - Corner 3: Bottom-Right (BR)
  - Corner 4: Bottom-Left (BL)

The system integrates feed inputs from up to three different cameras:
  1. Intel RealSense D435i Camera : Static table observer providing an overview.
  2. OAK-D Depth Camera           : Static observer from an alternative vantage point.
  3. Wrist-Attached Camera        : Dynamic eye-in-hand camera on the Kinova 
                                    Gen3 arm providing high-resolution, close-up 
                                    views as the arm approaches the target.

================================================================================
How the Fusion Pipeline Works
================================================================================
1. 2D Marker Detection:
   Each camera independently subscribes to its raw RGB image feed. The node 
   converts these images into grayscale and runs OpenCV's ArUco marker detection.

2. 3D Pose Estimation relative to Camera:
   Using the camera calibration parameters (intrinsics matrix K and distortion 
   coefficients D) obtained dynamically from each camera's /camera_info topic, 
   the node solves the Perspective-n-Point (PnP) problem via cv2.solvePnP. This 
   reconstructs the 3D position [x, y, z] of each detected marker in the local 
   camera optical frame.

3. Spatial Transformation to Reference Frame:
   Using ROS 2 tf2_ros, the node looks up the dynamic transform from the target 
   common reference frame (typically "world" or "base_link") to the camera's 
   optical frame at the exact image frame timestamp. It transforms the 3D marker 
   position from camera-local coordinates to reference-frame coordinates.

4. Multi-Camera Sensor Fusion:
   Measurements of each marker ID are collected from all reporting cameras. For 
   each corner marker, the active measurements (updated within a timeout 
   threshold) are averaged to cancel out high-frequency noise and resolve spatial 
   biases inherent to different viewpoints and lighting.

5. Geometric Fallbacks for Partial Occlusions:
   - Full Detection (4 corners): The center is the mean of TL, TR, BR, BL.
   - 3-Corner Occlusion Fallback: If one corner is occluded, its 3D position is 
     geometrically reconstructed using the remaining three. For example:
       TL_estimated = TR + (BL - BR)
     Then all 4 corners are averaged to find the center.
   - 2-Opposite-Corner Fallback: If only opposite corners are visible (e.g. TL and BR),
     the center is computed as their midpoint.

6. Spatial Orientation Computation (Pose Generation):
   The node computes a complete orthonormal coordinate frame (Pose) at the center:
     - X-axis points from left to right along the top/bottom edges.
     - Z-axis points normal to the square surface (straight up from the table).
     - Y-axis is orthonormalized to complete the frame (pointing forward-left).
   This 3D rotation matrix is converted into a quaternion.

7. TF & Topic Broadcasting:
   The calculated center is published as a geometry_msgs/PoseStamped message and 
   broadcast as a dynamic TF frame ("marker_square_center"). This allows motion 
   planners (like MoveIt!) and clients (like insert_to_container.py) to target the 
   exact 3D container pose dynamically.

================================================================================
Usage Instructions
================================================================================
Run the node with custom parameters:
  ros2 run kortex_bringup combine_cameras.py \
    --ros-args -p marker_id_tl:=10 -p marker_id_tr:=11 \
               -p marker_id_br:=12 -p marker_id_bl:=13
"""

import sys
import math
import time
import numpy as np
import cv2

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray, Pose, TransformStamped
from cv_bridge import CvBridge

# TF2 Imports
import tf2_ros


# ==============================================================================
# Mathematical Helpers for Coordinate Transforms and Geometry
# ==============================================================================
def rotate_vector(v: np.ndarray, q) -> np.ndarray:
    """
    Rotate a 3D vector v by a geometry_msgs/Quaternion q using Rodrigues' formula.
    Avoids external library dependencies and is highly performant.
    """
    v_arr = np.array(v, dtype=np.float32)
    q_xyz = np.array([q.x, q.y, q.z], dtype=np.float32)
    w = q.w
    
    # Rodrigues' formula: v_rot = v + 2 * cross(q_xyz, cross(q_xyz, v) + w * v)
    cross1 = np.cross(q_xyz, v_arr) + w * v_arr
    v_rot = v_arr + 2.0 * np.cross(q_xyz, cross1)
    return v_rot


def transform_point(point: np.ndarray, transform_stamped: TransformStamped) -> np.ndarray:
    """
    Transform a 3D point from a local source frame to the target frame 
    using a geometry_msgs/TransformStamped message.
    """
    q = transform_stamped.transform.rotation
    t = transform_stamped.transform.translation
    
    # 1. Rotate the point
    rotated = rotate_vector(point, q)
    # 2. Translate the point
    transformed = rotated + np.array([t.x, t.y, t.z], dtype=np.float32)
    return transformed


def rotation_matrix_to_quaternion(R: np.ndarray) -> list:
    """
    Convert a 3x3 orthonormal rotation matrix to a quaternion [x, y, z, w].
    Employs Shepperd's robust method to ensure numerical stability.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    
    norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    return [qx / norm, qy / norm, qz / norm, qw / norm]


# ==============================================================================
# ROS 2 Combine Cameras and Sensor Fusion Node
# ==============================================================================
class CombineCamerasNode(Node):
    def __init__(self):
        super().__init__("combine_cameras_node")
        self.get_logger().info("Initializing Combine Cameras Multi-Sensor Fusion Node...")
        
        self.bridge = CvBridge()
        self._cb_group = ReentrantCallbackGroup()
        
        # ----------------------------------------------------------------------
        # Declare Parameters
        # ----------------------------------------------------------------------
        # Coordinate frame parameters
        self.declare_parameter("reference_frame", "world")
        self.declare_parameter("target_frame_id", "marker_square_center")
        
        # Physical parameters of the target
        self.declare_parameter("marker_size", 0.50)             # meters (side length of marker: 50cm)
        self.declare_parameter("container_z_offset", 0.0)       # meters (vertical height offset of container top relative to markers)
        self.declare_parameter("aruco_dictionary_name", "DICT_4X4_50")
        self.declare_parameter("measurement_timeout", 1.0)       # seconds to cache a visual measurement
        
        # Corner Marker IDs
        self.declare_parameter("marker_id_bl", 0)   # Bottom-Left Corner
        self.declare_parameter("marker_id_br", 1)   # Bottom-Right Corner
        self.declare_parameter("marker_id_tr", 2)   # Top-Right Corner
        self.declare_parameter("marker_id_tl", 3)   # Top-Left Corner
        
        # Camera 1 Configurations: RealSense D435i
        self.declare_parameter("camera_realsense_enabled", True)
        self.declare_parameter("realsense_image_topic", "/realsense/camera/color/image_raw")
        self.declare_parameter("realsense_info_topic", "/realsense/camera/color/camera_info")
        
        # Camera 2 Configurations: OAK-D
        self.declare_parameter("camera_oakd_enabled", True)
        self.declare_parameter("oakd_image_topic", "/oakd/oak/rgb/image_raw")
        self.declare_parameter("oakd_info_topic", "/oakd/oak/rgb/camera_info")
        
        # Camera 3 Configurations: Kinova Wrist-Attached Camera
        self.declare_parameter("camera_kinova_enabled", True)
        self.declare_parameter("kinova_image_topic", "/camera/color/image_raw")
        self.declare_parameter("kinova_info_topic", "/camera/color/camera_info")
        
        # Read parameters
        self.reference_frame = self.get_parameter("reference_frame").value
        self.target_frame_id = self.get_parameter("target_frame_id").value
        self.marker_size = self.get_parameter("marker_size").value
        self.container_z_offset = self.get_parameter("container_z_offset").value
        self.measurement_timeout = self.get_parameter("measurement_timeout").value
        
        self.marker_id_tl = self.get_parameter("marker_id_tl").value
        self.marker_id_tr = self.get_parameter("marker_id_tr").value
        self.marker_id_br = self.get_parameter("marker_id_br").value
        self.marker_id_bl = self.get_parameter("marker_id_bl").value
        self.corner_ids = {self.marker_id_tl, self.marker_id_tr, self.marker_id_br, self.marker_id_bl}
        
        # ----------------------------------------------------------------------
        # Setup OpenCV ArUco Dictionary
        # ----------------------------------------------------------------------
        dict_name = self.get_parameter("aruco_dictionary_name").value
        self.aruco_dict_id = self._get_aruco_dict_id(dict_name)
        
        # ----------------------------------------------------------------------
        # TF2 Setup
        # ----------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # ----------------------------------------------------------------------
        # Fused Measurements State Database
        # ----------------------------------------------------------------------
        # Schema: { marker_id: { camera_name: { 'timestamp': float, 'position': np.array([x, y, z]) } } }
        self.measurements = {}
        
        # Camera Intrinsics Cache: { camera_name: (K_matrix, dist_coeffs, optical_frame_id) }
        self.camera_intrinsics = {}
        
        # ----------------------------------------------------------------------
        # GUI Visualization and Debug Image Database
        # ----------------------------------------------------------------------
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("visualize_grid", True)
        self.enable_visualization = self.get_parameter("enable_visualization").value
        self.visualize_grid = self.get_parameter("visualize_grid").value
        
        # Cache for latest annotated frames and reception timestamps for the GUI dashboard
        # Schema: { camera_name: annotated_opencv_image }
        self.debug_images = {}
        # Schema: { camera_name: timestamp_received }
        self.last_image_received = {}
        # Cache of the latest calculated center for live visual diagnostics
        self.latest_center = None
        
        # ----------------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------------
        # Fused Center Pose
        self.center_pub = self.create_publisher(
            PoseStamped, "/fused_marker_square_center", 10
        )
        # Visual feedback: PoseArray of the four fused/reconstructed corners
        self.corners_pub = self.create_publisher(
            PoseArray, "/fused_corners", 10
        )
        
        # ----------------------------------------------------------------------
        # Initialize Subscribers Dynamically Based on Parameters
        # ----------------------------------------------------------------------
        self.subs = []
        
        # RealSense
        if self.get_parameter("camera_realsense_enabled").value:
            self._init_camera_topics("realsense")
            
        # OAK-D
        if self.get_parameter("camera_oakd_enabled").value:
            self._init_camera_topics("oakd")
            
        # Kinova Arm Camera
        if self.get_parameter("camera_kinova_enabled").value:
            self._init_camera_topics("kinova")
            
        # ----------------------------------------------------------------------
        # Main Sensor Fusion Coordination Timer (10 Hz)
        # ----------------------------------------------------------------------
        self.timer = self.create_timer(0.1, self._fuse_and_publish_center, callback_group=self._cb_group)
        self.get_logger().info("Initialization complete. Awaiting camera feeds...")

    def _get_aruco_dict_id(self, dict_name: str) -> int:
        """Map standard dictionary string name to OpenCV constant."""
        mapping = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        }
        if dict_name not in mapping:
            self.get_logger().warn(f"Unknown ArUco dictionary '{dict_name}'. Defaulting to DICT_4X4_50.")
            return cv2.aruco.DICT_4X4_50
        return mapping[dict_name]

    def _init_camera_topics(self, name: str):
        """Set up subscriptions for a specific camera name."""
        image_topic = self.get_parameter(f"{name}_image_topic").value
        info_topic = self.get_parameter(f"{name}_info_topic").value
        
        self.get_logger().info(f"Subscribing to camera '{name}': Image={image_topic}, Info={info_topic}")
        
        # Lambda default arguments cache the specific camera 'name' string
        sub_info = self.create_subscription(
            CameraInfo,
            info_topic,
            lambda msg: self._camera_info_cb(msg, name),
            10,
            callback_group=self._cb_group
        )
        sub_image = self.create_subscription(
            Image,
            image_topic,
            lambda msg: self._image_cb(msg, name),
            10,
            callback_group=self._cb_group
        )
        self.subs.extend([sub_info, sub_image])

    # ----------------------------------------------------------------------
    # Callback Handlers
    # ----------------------------------------------------------------------
    def _camera_info_cb(self, msg: CameraInfo, camera_name: str):
        """Cache camera intrinsics once they are published."""
        if camera_name in self.camera_intrinsics:
            return # Already cached
            
        K = np.array(msg.k, dtype=np.float32).reshape((3, 3))
        D = np.array(msg.d, dtype=np.float32)
        optical_frame = msg.header.frame_id
        
        self.camera_intrinsics[camera_name] = (K, D, optical_frame)
        self.get_logger().info(f"Received and cached CameraInfo for '{camera_name}'. Optical Frame: {optical_frame}")

    def _image_cb(self, msg: Image, camera_name: str):
        """Process incoming raw image feed and perform marker detection."""
        try:
            # Convert ROS Image to OpenCV Image
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error for camera '{camera_name}': {e}")
            return

        now = self.get_clock().now().nanoseconds / 1e9
        self.last_image_received[camera_name] = now

        # CameraInfo not yet received — show raw feed but skip ArUco detection
        if camera_name not in self.camera_intrinsics:
            if self.enable_visualization:
                self._cache_annotated_feed(camera_name, cv_img.copy())
            return

        K, D, optical_frame = self.camera_intrinsics[camera_name]
        
        # Prepare annotated image copy for visual diagnostics if GUI is enabled
        annotated_img = cv_img.copy() if self.enable_visualization else None
            
        # Convert to grayscale for ArUco processing
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers with dynamic OpenCV API fallback compatibility
        corners, ids, _ = self._detect_aruco_markers(gray)
        
        # Draw detected markers on the annotated copy
        if self.enable_visualization and ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated_img, corners, ids)
            
        # If no markers were found, we can still cache the clean camera feed with a status banner
        if ids is None or len(ids) == 0:
            if self.enable_visualization:
                self._cache_annotated_feed(camera_name, annotated_img)
            return
            
        ids = ids.flatten()
        
        # Look up spatial transform from target world frame to camera frame
        try:
            # Look up transform at the exact image stamp to support active eye-in-hand motion
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                optical_frame,
                msg.header.stamp,
                timeout=RclpyDuration(seconds=0.05)
            )
        except Exception:
            # Fallback to the latest available transform if time sync buffer lag occurs
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.reference_frame,
                    optical_frame,
                    rclpy.time.Time(),
                    timeout=RclpyDuration(seconds=0.05)
                )
            except Exception as e:
                self.get_logger().warn(
                    f"Could not look up transform from {self.reference_frame} to {optical_frame}: {e}",
                    throttle_duration_sec=10.0
                )
                if self.enable_visualization:
                    self._cache_annotated_feed(camera_name, annotated_img)
                return
                
        # Parse detected markers
        for idx, marker_id in enumerate(ids):
            if marker_id not in self.corner_ids:
                continue # Skip markers that aren't designated corners of the container
                
            marker_corners = corners[idx][0] # 4x2 corners
            
            # Solve Perspective-n-Point to get 3D coordinates relative to camera lens
            success, rvec, tvec = self._estimate_single_marker_pose(marker_corners, self.marker_size, K, D)
            if not success:
                continue
                
            # Draw 3D axis on the marker if GUI is active
            if self.enable_visualization:
                if hasattr(cv2, "drawFrameAxes"):
                    cv2.drawFrameAxes(annotated_img, K, D, rvec, tvec, self.marker_size * 0.8)
                else:
                    cv2.aruco.drawAxis(annotated_img, K, D, rvec, tvec, self.marker_size * 0.8)
                
            local_pos = tvec.flatten()
            
            # Transform local position (camera lens coordinates) to global reference coordinates
            global_pos = transform_point(local_pos, transform)
            
            # Store measurement in dynamic database
            self._store_measurement(marker_id, camera_name, global_pos)
            
        if self.enable_visualization:
            self._cache_annotated_feed(camera_name, annotated_img)

    def _detect_aruco_markers(self, gray_img: np.ndarray):
        """Detect ArUco markers with API safety across all OpenCV 4.x versions."""
        if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector"):
            # Modern OpenCV 4.7+ API
            dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            return detector.detectMarkers(gray_img)
        else:
            # Legacy OpenCV API
            dictionary = cv2.aruco.Dictionary_get(self.aruco_dict_id)
            params = cv2.aruco.DetectorParameters_create()
            return cv2.aruco.detectMarkers(gray_img, dictionary, parameters=params)

    def _estimate_single_marker_pose(self, corners: np.ndarray, marker_size: float, K: np.ndarray, D: np.ndarray):
        """
        Estimate 3D position [x,y,z] of a single marker using PnP.
        Utilizes direct cv2.solvePnP for universal compatibility.
        """
        # Define 3D object points of the marker in its local coordinate system (center is origin)
        obj_points = np.array([
            [-marker_size / 2.0,  marker_size / 2.0, 0.0],
            [ marker_size / 2.0,  marker_size / 2.0, 0.0],
            [ marker_size / 2.0, -marker_size / 2.0, 0.0],
            [-marker_size / 2.0, -marker_size / 2.0, 0.0]
        ], dtype=np.float32)
        
        # solvePnP returns rotation (rvec) and translation (tvec) vectors
        success, rvec, tvec = cv2.solvePnP(obj_points, corners.astype(np.float32), K, D)
        return success, rvec, tvec

    def _store_measurement(self, marker_id: int, camera_name: str, pos: np.ndarray):
        """Update the database with the latest transformed marker coordinates."""
        now = self.get_clock().now().nanoseconds / 1e9
        if marker_id not in self.measurements:
            self.measurements[marker_id] = {}
        self.measurements[marker_id][camera_name] = {
            "timestamp": now,
            "position": pos
        }

    def _cache_annotated_feed(self, name: str, img: np.ndarray):
        """Add modern banner, resize to 640x360, and cache annotated image."""
        if img is None:
            return
            
        h, w = img.shape[:2]
        
        # Draw elegant semi-transparent banner at the top
        banner_height = 40
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Set camera display name
        display_names = {
            "realsense": "INTEL REALSENSE D435i",
            "oakd": "OAK-D DEPTH CAMERA",
            "kinova": "KINOVA EYE-IN-HAND CAMERA"
        }
        display_name = display_names.get(name, name.upper())
        
        # Draw status text and circle
        cv2.circle(img, (20, 20), 6, (80, 200, 120), -1, cv2.LINE_AA) # Green dot for online feed
        cv2.putText(
            img,
            f"{display_name} - ONLINE",
            (40, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 245),
            1,
            cv2.LINE_AA
        )
        
        # Resize to standard uniform shape for the dashboard layout
        resized = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
        self.debug_images[name] = resized

    def _create_offline_placeholder(self, name: str) -> np.ndarray:
        """Create a professional dark gray placeholder panel for offline cameras."""
        panel = np.zeros((360, 640, 3), dtype=np.uint8)
        panel[:] = [30, 25, 25] # Slate background
        
        display_names = {
            "realsense": "INTEL REALSENSE D435i",
            "oakd": "OAK-D DEPTH CAMERA",
            "kinova": "KINOVA EYE-IN-HAND CAMERA"
        }
        display_name = display_names.get(name, name.upper())
        
        # Draw grey status circle and "OFFLINE" ring
        cv2.circle(panel, (320, 140), 22, (50, 50, 55), -1, cv2.LINE_AA)
        cv2.circle(panel, (320, 140), 22, (80, 80, 200), 2, cv2.LINE_AA) # Soft reddish ring
        
        # Center display name text
        name_size = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        name_x = (640 - name_size[0]) // 2
        cv2.putText(
            panel,
            display_name,
            (name_x, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 190),
            2,
            cv2.LINE_AA
        )
        
        # Center subtext
        text = "FEED OFFLINE / AWAITING ACTIVE TOPIC"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        text_x = (640 - text_size[0]) // 2
        cv2.putText(
            panel,
            text,
            (text_x, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (100, 100, 110),
            1,
            cv2.LINE_AA
        )
        return panel

    def draw_diagnostics_panel(self, width=640, height=360) -> np.ndarray:
        """Create a premium, dark-mode real-time visual diagnostics dashboard panel."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = [35, 30, 30] # BGR for slate gray [30, 30, 35]
        
        # Title
        cv2.putText(panel, "MULTI-CAMERA FUSION DIAGNOSTICS", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 245), 2, cv2.LINE_AA)
        
        # Draw horizontal divider
        cv2.line(panel, (20, 50), (width - 20, 50), (65, 60, 60), 1)
        
        # Status
        now = self.get_clock().now().nanoseconds / 1e9
        
        # Calculate active corners
        fused_corners = {}
        for m_id in self.corner_ids:
            if m_id in self.measurements:
                valid_positions = [rec["position"] for rec in self.measurements[m_id].values() if now - rec["timestamp"] <= self.measurement_timeout]
                if len(valid_positions) > 0:
                    fused_corners[m_id] = np.mean(valid_positions, axis=0)
                    
        num_corners = len(fused_corners)
        
        # Target status
        status_text = "FUSING & ACTIVE" if num_corners >= 2 else "AWAITING MARKERS"
        status_color = (80, 200, 120) if num_corners >= 2 else (80, 150, 240) # BGR: green or amber/orange
        
        cv2.putText(panel, "System Status:", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, status_text, (160, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)
        
        # Fused coordinates (X, Y, Z)
        cv2.putText(panel, "Fused Center:", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 180), 1, cv2.LINE_AA)
        if self.latest_center is not None and num_corners >= 2:
            x, y, z = self.latest_center
            cv2.putText(panel, f"X: {x:+.4f} m", (160, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 230), 1, cv2.LINE_AA)
            cv2.putText(panel, f"Y: {y:+.4f} m", (160, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 230), 1, cv2.LINE_AA)
            cv2.putText(panel, f"Z: {z:+.4f} m", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 230), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "X: N/A", (160, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 110), 1, cv2.LINE_AA)
            cv2.putText(panel, "Y: N/A", (160, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 110), 1, cv2.LINE_AA)
            cv2.putText(panel, "Z: N/A", (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 110), 1, cv2.LINE_AA)

        # Detected corners detail
        cv2.putText(panel, "Corners Seen:", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 180), 1, cv2.LINE_AA)
        corners_label = f"{num_corners} / 4"
        if num_corners == 4:
            corners_label += " (Perfect)"
            label_color = (80, 200, 120)
        elif num_corners == 3:
            corners_label += " (Reconstructed)"
            label_color = (80, 180, 240)
        elif num_corners == 2:
            corners_label += " (Midpoint)"
            label_color = (80, 180, 240)
        else:
            corners_label += " (Insufficient)"
            label_color = (80, 80, 220)
        cv2.putText(panel, corners_label, (160, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)
        
        # Active camera feeds status
        cv2.putText(panel, "Active Camera Feeds:", (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 180), 1, cv2.LINE_AA)
        
        cameras = ["realsense", "oakd", "kinova"]
        names = ["RealSense D435i", "OAK-D Depth", "Kinova Wrist"]
        
        for idx, (cam, display_name) in enumerate(zip(cameras, names)):
            y_pos = 265 + idx * 25
            cv2.putText(panel, display_name, (45, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 210), 1, cv2.LINE_AA)
            
            # Check if camera has sent an image recently
            last_img_time = self.last_image_received.get(cam, 0.0)
            is_active = (now - last_img_time < 2.0) and self.get_parameter(f"camera_{cam}_enabled").value
            
            status_dot_color = (80, 200, 120) if is_active else (80, 80, 220)
            status_text = "ONLINE" if is_active else "OFFLINE"
            
            # Draw status circle
            cv2.circle(panel, (30, y_pos - 5), 4, status_dot_color, -1, cv2.LINE_AA)
            cv2.putText(panel, status_text, (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_dot_color, 1, cv2.LINE_AA)
            
        return panel

    def show_debug_windows(self):
        """Assemble the 2x2 multi-camera dashboard and display it using OpenCV."""
        if not self.enable_visualization:
            return
            
        # Get active images or construct placeholders if offline
        rs_panel = self.debug_images.get("realsense")
        if rs_panel is None:
            rs_panel = self._create_offline_placeholder("realsense")
            
        oak_panel = self.debug_images.get("oakd")
        if oak_panel is None:
            oak_panel = self._create_offline_placeholder("oakd")
            
        kv_panel = self.debug_images.get("kinova")
        if kv_panel is None:
            kv_panel = self._create_offline_placeholder("kinova")
            
        # Generate the live diagnostics panel
        diag_panel = self.draw_diagnostics_panel()
        
        # Assemble 2x2 dashboard layout
        top_row = np.hstack((rs_panel, oak_panel))
        bottom_row = np.hstack((kv_panel, diag_panel))
        dashboard = np.vstack((top_row, bottom_row))
        
        # Display the window
        try:
            cv2.imshow("Multi-Camera ArUco Detection Dashboard", dashboard)
            # waitKey(1) processes display events and keeps window responsive
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(
                f"Failed to display GUI window: {e}. "
                "Disabling GUI visualization (head-less mode active)."
            )
            self.enable_visualization = False

    # ----------------------------------------------------------------------
    # Core Sensor Fusion & Publisher Loop
    # ----------------------------------------------------------------------
    def _fuse_and_publish_center(self):
        """Consolidate sensor coordinates, perform fallbacks, and publish output."""
        now = self.get_clock().now().nanoseconds / 1e9
        
        # 1. Filter out stale measurements and average valid ones for each corner ID
        fused_corners = {}
        for m_id in self.corner_ids:
            if m_id not in self.measurements:
                continue
                
            valid_positions = []
            for camera_name, record in list(self.measurements[m_id].items()):
                # Check if measurement is within the cache timeout window
                if now - record["timestamp"] <= self.measurement_timeout:
                    valid_positions.append(record["position"])
                else:
                    # Clean up expired data
                    self.measurements[m_id].pop(camera_name)
                    
            if len(valid_positions) > 0:
                # Sensor Fusion: compute average across all cameras
                fused_corners[m_id] = np.mean(valid_positions, axis=0)

        # Retrieve specific corners
        tl = fused_corners.get(self.marker_id_tl)
        tr = fused_corners.get(self.marker_id_tr)
        br = fused_corners.get(self.marker_id_br)
        bl = fused_corners.get(self.marker_id_bl)
        
        num_detected = sum([tl is not None, tr is not None, br is not None, bl is not None])
        
        if num_detected < 2:
            self.get_logger().info(
                "Insufficient corner markers detected. Need at least 2 opposite or 3 adjacent.",
                throttle_duration_sec=10.0
            )
            return
            
        center = None
        
        # ----------------------------------------------------------------------
        # Case A: Perfect Detection (All 4 Corners Visible)
        # ----------------------------------------------------------------------
        if num_detected == 4:
            center = (tl + tr + br + bl) / 4.0
            
        # ----------------------------------------------------------------------
        # Case B: Partial Occlusion Fallback (3 Corners Visible)
        # ----------------------------------------------------------------------
        elif num_detected == 3:
            # Reconstruct the missing 4th corner using parallel vector properties
            if tl is None:
                tl = tr + (bl - br)
                self.get_logger().debug("Reconstructed Top-Left (TL) corner marker.")
            elif tr is None:
                tr = tl + (br - bl)
                self.get_logger().debug("Reconstructed Top-Right (TR) corner marker.")
            elif br is None:
                br = bl + (tr - tl)
                self.get_logger().debug("Reconstructed Bottom-Right (BR) corner marker.")
            elif bl is None:
                bl = br + (tl - tr)
                self.get_logger().debug("Reconstructed Bottom-Left (BL) corner marker.")
                
            center = (tl + tr + br + bl) / 4.0
            
        # ----------------------------------------------------------------------
        # Case C: Minimal Fallback (2 Opposite Corners Visible)
        # ----------------------------------------------------------------------
        elif num_detected == 2:
            if tl is not None and br is not None:
                center = (tl + br) / 2.0
                self.get_logger().info("Partial fusion: Midpoint computed from TL & BR.", throttle_duration_sec=10.0)
            elif tr is not None and bl is not None:
                center = (tr + bl) / 2.0
                self.get_logger().info("Partial fusion: Midpoint computed from TR & BL.", throttle_duration_sec=10.0)
            else:
                self.get_logger().warn(
                    "Only adjacent corners visible. Midpoint cannot resolve center without orientation context.",
                    throttle_duration_sec=10.0
                )
                return

        if center is None:
            return

        # Apply the container Z height offset relative to the marker plane.
        # Since markers are typically placed on a flat table surface, the top opening
        # of the container is elevated by a physical height offset.
        center[2] += self.container_z_offset
        self.latest_center = center

        # ----------------------------------------------------------------------
        # Compute Coordinate Frame Pose (Rotation Matrix -> Quaternion)
        # ----------------------------------------------------------------------
        stamp = self.get_clock().now().to_msg()
        
        # We can only compute a rigorous orthonormal coordinate orientation
        # if we have fully resolved/reconstructed all 4 corners
        if tl is not None and tr is not None and br is not None and bl is not None:
            # Vector along local X (left-to-right top and bottom average)
            vx = (tr - tl) + (br - bl)
            # Vector along local Y (bottom-to-top left and right average)
            vy = (tl - bl) + (tr - br)
            
            # Compute normal vector (Z) representing surface perpendicular orientation
            vz = np.cross(vx, vy)
            
            # Normalize vectors to build orthogonal base
            ux = vx / np.linalg.norm(vx)
            uz = vz / np.linalg.norm(vz)
            
            # Orthonormalize Y vector to guarantee pure 3D rotation frame matrix
            uy = np.cross(uz, ux)
            
            # Rotation matrix R
            R = np.column_stack((ux, uy, uz))
            
            # Convert rotation matrix to quaternion
            q = rotation_matrix_to_quaternion(R)
            
            # Publish PoseArray representing individual fused corners for RViz visualization
            self._publish_visual_corners(stamp, tl, tr, br, bl, q)
        else:
            # Fallback: align target coordinate orientation with the reference frame (Identity)
            q = [0.0, 0.0, 0.0, 1.0]

        # ----------------------------------------------------------------------
        # Publish PoseStamped & Broadcast TF
        # ----------------------------------------------------------------------
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.reference_frame
        pose_msg.pose.position.x = float(center[0])
        pose_msg.pose.position.y = float(center[1])
        pose_msg.pose.position.z = float(center[2])
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])
        
        self.center_pub.publish(pose_msg)
        
        # Broadcast TF Frame
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.reference_frame
        t.child_frame_id = self.target_frame_id
        t.transform.translation.x = float(center[0])
        t.transform.translation.y = float(center[1])
        t.transform.translation.z = float(center[2])
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        
        self.tf_broadcaster.sendTransform(t)
        
        self.get_logger().info(
            f"Published Square Center Target in {self.reference_frame}: "
            f"[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]",
            throttle_duration_sec=5.0
        )

    def _publish_visual_corners(self, stamp, tl, tr, br, bl, q):
        """Publish corner positions as PoseArray for RViz visualization."""
        array_msg = PoseArray()
        array_msg.header.stamp = stamp
        array_msg.header.frame_id = self.reference_frame
        
        for point in [tl, tr, br, bl]:
            pose = Pose()
            pose.position.x = float(point[0])
            pose.position.y = float(point[1])
            pose.position.z = float(point[2])
            pose.orientation.x = float(q[0])
            pose.orientation.y = float(q[1])
            pose.orientation.z = float(q[2])
            pose.orientation.w = float(q[3])
            array_msg.poses.append(pose)
            
        self.corners_pub.publish(array_msg)


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main(args=None):
    rclpy.init(args=args)
    
    node = CombineCamerasNode()
    
    # We spin the executor in a background thread so the main thread
    # is dedicated entirely to the OpenCV GUI event loop (required for thread-safe cv2.imshow).
    if node.enable_visualization:
        import threading
        
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        
        node.get_logger().info("OpenCV Visual Dashboard is active on the main thread.")
        
        try:
            while rclpy.ok() and node.enable_visualization:
                node.show_debug_windows()
                # Run at ~30 Hz (33ms sleep) to avoid pegging the CPU
                time.sleep(0.033)
        except KeyboardInterrupt:
            node.get_logger().info("Shutdown signal received via keyboard. Exiting...")
        finally:
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()
            spin_thread.join(timeout=1.0)
    else:
        # Standard spin for headless execution
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("Shutdown signal received via keyboard. Exiting...")
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
