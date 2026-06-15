# calibration.launch.py — Hand-eye calibration for global RealSense D435I
#
# Standalone launch — does NOT include cameras.launch.py, which carries the
# (wrong) calibration TF.  Instead, launches only what calibration needs:
#   1. Robot arm + MoveIt2
#   2. RealSense D435I (RGB only — depth not needed for ArUco pose estimation)
#   3. ArUco marker detector + TF bridge
#   4. easy_handeye2 rqt_calibrator GUI
#
# easy_handeye2 provides a dummy base_link → global_camera_color_optical_frame
# TF during calibration (eye_on_base mode).  This is replaced when you save.
#
# Workflow:
#   Terminal 1:
#     ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config robot.launch.py \
#       robot_ip:=192.168.1.10
#   Terminal 2:
#     ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config calibration.launch.py \
#       robot_ip:=192.168.1.10
#   In rqt_calibrator GUI:
#     - Move robot to 15–20 positions covering full workspace
#     - Keep ArUco marker (ID 8, DICT_4X4_50, 203.2 mm) visible to RealSense
#     - Click "Take sample" at each pose (need ≥12 well-distributed samples)
#     - Click "Compute" → verify reprojection error < 0.5 px
#     - Click "Save" → copy result into cameras.launch.py calibration_tf
#
# NOTE: tracking_marker_frame = 'aruco_marker' (published by aruco_tf_bridge).
#       For ChArUco board support, replace aruco_detector / aruco_tf_bridge with
#       a ChArUco-capable detector and change tracking_marker_frame to
#       'charuco_board'.

import os

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration("robot_ip")

    # ── Robot arm + MoveIt2 ───────────────────────────────────────────────────
    # RViz disabled: rqt_calibrator is the UI for this session.
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(
                    "kinova_gen3_7dof_robotiq_2f_140_moveit_config"
                ),
                "launch",
                "robot.launch.py",
            )
        ),
        launch_arguments={
            "robot_ip":    robot_ip.perform(context),
            "launch_rviz": "false",
        }.items(),
    )

    # ── RealSense D435I — RGB stream only ────────────────────────────────────
    # Depth is not needed: ArUco gives 6DOF pose from 2D image + marker size.
    # publish_tf=True so realsense publishes color_optical_frame into the TF tree.
    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="global_camera",
        namespace="global_camera",
        output="screen",
        parameters=[{
            "camera_name":      "global_camera",
            "enable_color":     True,
            "enable_depth":     False,
            "publish_tf":       True,
            "enable_accel":     False,
            "enable_gyro":      False,
            "depth_module.profile": "424x240x15",
            "rgb_camera.profile":   "424x240x15",
        }],
    )

    # ── ArUco marker detector ─────────────────────────────────────────────────
    # Publishes TF: global_camera_color_optical_frame → aruco_marker
    aruco_detector = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("my_handeye_config"),
                "launch",
                "aruco_detector.launch.py",
            )
        ),
    )

    aruco_tf_bridge = Node(
        package="my_handeye_config",
        executable="aruco_tf_bridge",
        name="aruco_tf_bridge",
        output="screen",
    )

    # ── easy_handeye2 calibration server + rqt GUI ───────────────────────────
    # eye_on_base: camera is fixed, not on the end-effector.
    # Dummy TF (base_link → global_camera_color_optical_frame at 1 m forward)
    # is published by easy_handeye2 automatically.
    easy_handeye2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("easy_handeye2"),
                "launch",
                "calibrate.launch.py",
            )
        ),
        launch_arguments={
            "name":                    "kinova_realsense",
            "calibration_type":        "eye_on_base",
            "robot_base_frame":        "base_link",
            "robot_effector_frame":    "tool_frame",
            "tracking_base_frame":     "global_camera_color_optical_frame",
            "tracking_marker_frame":   "aruco_marker",
            "freehand_robot_movement": "true",
        }.items(),
    )

    return [
        robot,
        realsense,
        aruco_detector,
        aruco_tf_bridge,
        easy_handeye2,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.10",
            description="Robot IP address."),
        OpaqueFunction(function=launch_setup),
    ])
