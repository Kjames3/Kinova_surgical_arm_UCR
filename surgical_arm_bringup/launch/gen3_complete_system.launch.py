import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    robot_ip      = LaunchConfiguration("robot_ip")
    use_fake      = LaunchConfiguration("use_fake_hardware")
    launch_wrist  = LaunchConfiguration("launch_wrist_camera")
    launch_oak    = LaunchConfiguration("launch_oak_camera")

    # ── 1. Robot arm + MoveIt + RViz ─────────────────────────────────────────
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('kinova_gen3_7dof_robotiq_2f_140_moveit_config'),
            '/launch/robot.launch.py'
        ]),
        launch_arguments={
            'robot_ip':         robot_ip,
            'use_fake_hardware': use_fake,
        }.items()
    )

    # ── 2. Intel RealSense D435i ─────────────────────────────────────────────
    # respawn=True: auto-restart after USB re-enumeration (common on USB 2.1).
    #
    # USB BANDWIDTH NOTE: The D435i is currently on a USB 2.1 port.
    # USB 2.1 max ≈ 480 Mbit/s. Color + Depth simultaneously exceeds this,
    # causing VIDIOC_QBUF → ENODEV disconnect loops (1 frame then freeze).
    # PERMANENT FIX: plug into a blue USB 3.0 port, then re-enable depth.
    # WORKAROUND: depth/IMU disabled; combine_cameras.py only needs RGB.
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace="realsense",
        output="screen",
        respawn=True,
        respawn_delay=3.0,
        parameters=[{
            "camera_namespace":             "realsense",
            "enable_color":                 True,
            "enable_depth":                 False,   # USB 2.1 bandwidth limit
            "enable_infra1":                False,
            "enable_infra2":                False,
            "enable_gyro":                  False,
            "enable_accel":                 False,
            "align_depth.enable":           False,
            "rgb_camera.color_profile":     "640x480x15",
            "publish_tf":                   False,
        }],
    )

    # ── 3. OAK-D Pro Wide — standalone depthai v3 driver ─────────────────────
    # depthai_ros_driver has an ABI conflict with /usr/local/lib/libdepthai-core.so
    # and crashes on startup.  oak_camera_node.py uses the depthai v3 Python API
    # directly and avoids the conflict.
    # Publishes: /oakd/oak/rgb/image_raw, /oakd/oak/rgb/camera_info
    #            base_link → global_camera_link TF
    _oak_script = os.path.expanduser(
        "~/workspace/ros2_kortex_ws/oak_camera_node.py"
    )
    oak_camera = ExecuteProcess(
        cmd=["python3", _oak_script],
        output="screen",
        emulate_tty=True,
        additional_env={"DEPTHAI_WATCHDOG_INITIAL_DELAY": "5000"},
        condition=IfCondition(launch_oak),
    )

    # ── 4. Kinova wrist camera — kinova_vision RTSP driver ───────────────────
    # Delayed 5 s so the robot control loop is fully established before the
    # RTSP stream competes for bandwidth on the 192.168.1.x NIC.
    # Publishes: /camera/color/image_raw, /camera/color/camera_info
    #            camera_link → camera_color_frame TF (via kinova_vision)
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
            "max_depth_pub_rate": "10.0",
            "depth_rtsp_element_config": "depth latency=100 timeout=10000000",
            "color_rtsp_element_config": "color latency=100",
        }.items(),
        condition=IfCondition(launch_wrist),
    )

    wrist_camera_delayed = TimerAction(period=5.0, actions=[wrist_camera])

    oak_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_world_to_oakd",
        arguments=[
            "0.362562", "0.719192", "1.279846",               # Translation (x, y, z)
            "-0.341494", "-0.888985", "0.299234", "0.059552", # Rotation (qx, qy, qz, qw)
            "world", "global_camera_link"
        ]
    )

    return [
        robot_launch,
        realsense_node,
        oak_camera,
        oak_static_tf,
        wrist_camera_delayed,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.10",
            description="IP address of the Kinova Gen3 arm."),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Use simulated hardware instead of physical arm."),
        DeclareLaunchArgument(
            "launch_wrist_camera",
            default_value="true",
            description="Launch Kinova wrist camera via kinova_vision RTSP driver."),
        DeclareLaunchArgument(
            "launch_oak_camera",
            default_value="true",
            description="Launch OAK-D via oak_camera_node.py (depthai v3 standalone)."),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
