# cameras.launch.py — all cameras + calibration TFs
#
# Launch AFTER robot.launch.py is fully running
# ("You can start planning now!" must appear in move_group output first).
# The wrist camera and robot control share the same 192.168.1.x NIC;
# launching cameras while the robot is still initialising causes
# BaseCyclicClient::Refresh timeouts from network congestion.
#
# Workflow:
#   Terminal 1:  ros2 launch ... robot.launch.py robot_ip:=192.168.1.10
#   Terminal 2:  ros2 launch ... cameras.launch.py robot_ip:=192.168.1.10
#   Terminal 3:  ros2 launch ... fusion.launch.py
#
# TF tree after this launch:
#   world → base_link                              (robot.launch.py)
#     base_link → global_camera_color_optical_frame  (calibration_tf here)
#       → global_camera_depth_optical_frame          (depth_optical_tf here)
#     base_link → global_camera_link                 (oak_camera_node.py)
#     base_link → ... → end_effector_link            (URDF / robot_state_publisher)
#       → camera_link                                (URDF, calibrated)
#         → camera_color_frame                       (kinova_vision static TF)
#         → camera_depth_frame                       (kinova_vision static TF)
#
# NOTE — Wrist camera calibration is baked into the URDF (gen3_macro.xacro):
#   end_effector_link → camera_link  xyz=(-0.0494305, 0.049587, 0.00395126)
#                                    rpy=(0.66454839, 0.30604363, 1.09121110)
#                     ↔ quat: (qx=0.200804 qy=0.290464 qz=0.442318 qw=0.824417)
#   camera_link → camera_color_frame (identity — kinova_vision publishes this)
#   camera_link → camera_depth_frame (-0.0195, -0.005, 0 — kinova_vision publishes this)
#
# Adding separate static_transform_publisher nodes for camera_color_frame or
# camera_depth_frame here would create a TF multi-parent loop because
# kinova_vision ALSO publishes camera_link → camera_color/depth_frame.

import os

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    robot_ip            = LaunchConfiguration("robot_ip")
    launch_wrist_camera = LaunchConfiguration("launch_wrist_camera")
    launch_oak_camera   = LaunchConfiguration("launch_oak_camera")

    # ── Global RealSense D435I calibration TFs ───────────────────────────────
    # eye-to-base: base_link → global_camera_color_optical_frame
    # Source: easy_handeye2 calibration result
    # Quaternion norm: 0.999979 (OK)
    calibration_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="calibration_tf_publisher",
        output="log",
        arguments=[
            "--x",  "0.99",
            "--y",  "-0.13",
            "--z",  "0.77",
            "--qx", "0.6220",
            "--qy", "0.6099",
            "--qz", "-0.3475",
            "--qw", "-0.3469",
            "--frame-id",       "base_link",
            "--child-frame-id", "global_camera_color_optical_frame",
        ],
    )

    # D435I depth-to-color extrinsic (from EEPROM via rs-enumerate-devices -c).
    # Color→Depth = inverse of EEPROM Depth→Color:
    #   tx=+0.014603 ty≈0 tz≈0 → inverted tx=-0.014601
    depth_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="depth_optical_tf_publisher",
        output="log",
        arguments=[
            "--x",  "-0.014601", "--y", "-0.000191", "--z", "-0.000399",
            "--qx", "0.000372", "--qy", "-0.002654", "--qz", "0.003147", "--qw", "0.999991",
            "--frame-id",       "global_camera_color_optical_frame",
            "--child-frame-id", "global_camera_depth_optical_frame",
        ],
    )

    # ── RealSense USB sanity check ───────────────────────────────────────────
    # MIPI errors and "reduced performance" warnings indicate USB 2.x port.
    # Plug into a blue USB 3.0 port for full resolution and stable operation.
    usb_check = ExecuteProcess(
        cmd=["bash", "-c",
             "rs-enumerate-devices 2>/dev/null | grep -i 'usb type' | "
             "grep -v '3\\.' "
             "&& echo 'WARNING: RealSense on USB 2.x — plug into blue USB 3.0 port!' "
             "|| echo 'RealSense USB OK'"],
        output="screen",
    )

    # ── Global RealSense D435I node ──────────────────────────────────────────
    # respawn=True: auto-restart after MIPI disconnect / USB re-enumeration.
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="global_camera",
        namespace="global_camera",
        output="screen",
        respawn=True,
        respawn_delay=3.0,
        parameters=[{
            "camera_name":                    "global_camera",
            "align_depth.enable":             True,
            "pointcloud.enable":              True,
            "pointcloud.stream_filter":       2,
            "pointcloud.stream_index_filter": 0,
            "enable_sync":                    False,
            # USB 2.1 bandwidth limit — switch to 640x480x15 on USB 3.0
            "depth_module.profile":           "424x240x15",
            "rgb_camera.profile":             "424x240x15",
            "enable_color":                   True,
            "enable_depth":                   True,
            "publish_tf":                     False,
            "enable_accel":                   False,
            "enable_gyro":                    False,
        }],
    )

    # ── Wrist camera — Kinova Gen3 built-in vision module (RTSP) ────────────
    # Delayed 5 s: robot control and RTSP share the 192.168.1.x NIC.
    # Launching immediately congests the link during the 1 kHz cyclic init.
    #
    # depth_registration=false: publishes native 480×270 depth cloud instead of
    # upsampling to 1280×720 color resolution.  The upsampled cloud fills every
    # color pixel with a depth value (~920K pts, ~613K valid), which overwhelms
    # the pointcloud_fusion sanity filter.  Native resolution yields ~5K–30K
    # valid pts — a useful wrist contribution at much lower CPU cost.
    # kinova_vision still publishes the camera_link static TFs even without
    # depth_registration, so the TF tree is unaffected.
    wrist_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("kinova_vision"),
                "launch",
                "kinova_vision.launch.py",
            )
        ),
        launch_arguments={
            "device":             robot_ip.perform(context),
            "camera":             "camera",
            "launch_color":       "true",
            "launch_depth":       "true",
            "depth_registration": "false",
            "max_color_pub_rate": "15.0",
            # 10 Hz depth: depth_registration TF lookup occasionally stalls
            # at higher rates, causing 2–15 Hz jitter.
            "max_depth_pub_rate": "10.0",
            # RTSP recv timeout (ns) prevents stream dropout on congested link.
            "depth_rtsp_element_config": "depth latency=100 timeout=10000000",
            "color_rtsp_element_config": "color latency=100",
        }.items(),
        condition=IfCondition(launch_wrist_camera),
    )

    wrist_camera_delayed = TimerAction(
        period=5.0,
        actions=[wrist_camera],
    )

    # ── OAK-D standalone Python driver ──────────────────────────────────────
    # Uses depthai v3 API directly; avoids depthai_ros_driver ABI conflict
    # with /usr/local/lib/libdepthai-core.so.
    # Publishes base_link → global_camera_link TF from T_BASE_CAM in the script.
    usb_cleanup = ExecuteProcess(
        cmd=["bash", "-c", "sudo fuser -k /dev/bus/usb/*/* 2>/dev/null || true"],
        output="log",
    )

    _oak_script = os.path.expanduser("~/workspace/ros2_kortex_ws/oak_camera_node.py")
    oak_camera = ExecuteProcess(
        cmd=["python3", _oak_script],
        output="screen",
        emulate_tty=True,
        # DEPTHAI_WATCHDOG_INITIAL_DELAY lets USB settle after cleanup.
        additional_env={"DEPTHAI_WATCHDOG_INITIAL_DELAY": "5000"},
        condition=IfCondition(launch_oak_camera),
    )

    # ── Wrist point cloud from native depth ──────────────────────────────────
    # kinova_vision with depth_registration=false does NOT generate a point
    # cloud.  wrist_pcl_node.py builds one directly from the raw depth image.
    wrist_pcl = ExecuteProcess(
        cmd=["python3", "/home/lab/workspace/ros2_kortex_ws/wrist_pcl_node.py"],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(launch_wrist_camera),
    )

    return [
        calibration_tf,
        depth_optical_tf,
        usb_check,
        realsense_node,
        usb_cleanup,
        wrist_camera_delayed,
        wrist_pcl,
        oak_camera,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.10",
            description="Robot IP — used for wrist camera RTSP stream."),
        DeclareLaunchArgument(
            "launch_wrist_camera",
            default_value="true",
            description="Launch Kinova built-in wrist camera via RTSP."),
        DeclareLaunchArgument(
            "launch_oak_camera",
            default_value="true",
            description="Launch OAK-D via oak_camera_node.py (depthai v3 standalone driver)."),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
