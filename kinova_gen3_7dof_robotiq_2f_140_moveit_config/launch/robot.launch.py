# Copyright (c) 2023 PickNik, Inc.
#
# Licensed under the Apache License, Version 2.0

# NETWORK TIP: Connect robot via a dedicated ethernet NIC.
# Do NOT share the robot's 192.168.1.x subnet with Wi-Fi or other devices.
# The wrist camera RTSP stream also runs over the same IP, so launch
# cameras.launch.py only AFTER robot.launch.py is fully stable
# ("You can start planning now!" appears in move_group output).

# ── FRAME REFERENCE ──────────────────────────────────────────────────────────
# world → base_link (static, identity)
# base_link → global_camera_color_optical_frame
#   x=0.99 y=-0.13 z=0.77
#   qx=0.6220 qy=0.6099 qz=-0.3475 qw=-0.3469
#   ⚠ STALE — redo easy_handeye2 calibration
# end_effector_link → pen_tip
#   xyz="0 0 -0.15452"  (154.52 mm down tool axis, EE -Z direction)
# end_effector_link → camera_color_frame
#   x=-0.0494305 y=0.049587 z=0.00395126
#   qx=0.200804 qy=0.290464 qz=0.442318 qw=0.824417
# base_link → global_camera_link (OAK-D)
#   x=0.48 y=0.72 z=1.0
#   qx=-0.341494 qy=-0.888985 qz=0.299234 qw=0.059552
# ─────────────────────────────────────────────────────────────────────────────

import os

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    robot_ip                      = LaunchConfiguration("robot_ip")
    use_fake_hardware             = LaunchConfiguration("use_fake_hardware")
    gripper_max_velocity          = LaunchConfiguration("gripper_max_velocity")
    gripper_max_force             = LaunchConfiguration("gripper_max_force")
    launch_rviz                   = LaunchConfiguration("launch_rviz")
    use_sim_time                  = LaunchConfiguration("use_sim_time")
    use_internal_bus_gripper_comm = LaunchConfiguration("use_internal_bus_gripper_comm")

    launch_arguments = {
        "robot_ip":                        robot_ip,
        "use_fake_hardware":               use_fake_hardware,
        "gripper":                         "thesis_ee",
        "vision":                          "true",
        # gripper_joint_name omitted — thesis_ee has no movable gripper joints
        "dof":                             "7",
        "gripper_max_velocity":            gripper_max_velocity,
        "gripper_max_force":               gripper_max_force,
        "use_internal_bus_gripper_comm":   use_internal_bus_gripper_comm,
        # Keys must match xacro:arg names in gen3.xacro (include _ms suffix).
        "session_inactivity_timeout_ms":    "120000",
        "connection_inactivity_timeout_ms": "10000",
    }

    moveit_config = (
        MoveItConfigsBuilder(
            "gen3",
            package_name="kinova_gen3_7dof_robotiq_2f_140_moveit_config",
        )
        .robot_description(mappings=launch_arguments)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    # to_moveit_configs() auto-loads sensors_3d.yaml into sensors_3d.
    # Clear it so to_dict() does not include unserialisable list-of-dicts.
    moveit_config.sensors_3d = {}

    moveit_config.moveit_cpp.update(
        {"use_sim_time": use_sim_time.perform(context) == "true"}
    )

    # ── ROS2 control ─────────────────────────────────────────────────────────
    ros2_controllers_path = os.path.join(
        get_package_share_directory(
            "kinova_gen3_7dof_robotiq_2f_140_moveit_config"
        ),
        "config",
        "ros2_controllers.yaml",
    )

    # NOTE: Do NOT set RMW_IMPLEMENTATION here.  The Kortex hardware interface
    # communicates with the arm via the Kortex API (not ROS transport), so a
    # different RMW on this node gives no latency benefit for the 1 kHz loop.
    # Forcing CycloneDDS on ros2_control_node while everything else uses
    # FastRTPS (the default) causes cross-process service calls — controller
    # spawners, `ros2 control list_controllers`, MoveIt — to time out because
    # the two middleware implementations cannot reliably interoperate for
    # ROS2 services.  All nodes must share the same RMW.
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[("/controller_manager/robot_description", "/robot_description")],
        output="both",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # --controller-manager-timeout 300 → wait up to 5 min for the Kortex hardware
    # driver to finish initialising before giving up.  The default (30 s) causes
    # the spawner to exit before the controller manager is ready on a physical arm,
    # which means joint_state_broadcaster never activates and RViz shows a frozen pose.
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager",
                   "--controller-manager-timeout", "300"],
    )

    robot_traj_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager",
                   "--controller-manager-timeout", "300"],
    )

    robot_pos_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["twist_controller", "--inactive", "-c", "/controller_manager",
                   "--controller-manager-timeout", "300"],
    )

    fault_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["fault_controller", "-c", "/controller_manager",
                   "--controller-manager-timeout", "300"],
        condition=UnlessCondition(use_fake_hardware),
    )

    # ── MoveIt2 (no OctoMap) ─────────────────────────────────────────────────
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
        # Suppress the "Joint 'finger_joint' not found in model 'gen3'" flood.
        # thesis_ee has no gripper joint; the KortexMultiInterfaceHardware plugin
        # still advertises one internally (via gripper_joint_name param in the
        # URDF hardware section) even with use_internal_bus_gripper_comm=false.
        # The messages are harmless — planning and execution are unaffected —
        # but they flood the terminal at ~1 kHz and hide real errors.
        # FATAL silences only this logger; all other move_group output is unchanged.
        ros_arguments=["--log-level", "moveit_robot_model.robot_model:=FATAL"],
    )

    # ── TF: world → base_link ────────────────────────────────────────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )

    # ── RViz ─────────────────────────────────────────────────────────────────
    rviz_config_file = (
        get_package_share_directory(
            "kinova_gen3_7dof_robotiq_2f_140_moveit_config"
        )
        + "/config/moveit.rviz"
        if os.path.exists(
            get_package_share_directory(
                "kinova_gen3_7dof_robotiq_2f_140_moveit_config"
            )
            + "/config/moveit.rviz"
        )
        else "/tmp/moveit_simple.rviz"
    )

    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        ),
        condition=IfCondition(launch_rviz),
    )

    robot_keepalive = ExecuteProcess(
        cmd=["python3", os.path.join(
            os.path.expanduser("~"), "workspace", "ros2_kortex_ws", "robot_keepalive.py"
        )],
        output="log",
    )

    return [
        ros2_control_node,
        robot_state_publisher,
        joint_state_broadcaster_spawner,
        robot_traj_controller_spawner,
        robot_pos_controller_spawner,
        fault_controller_spawner,
        move_group_node,
        static_tf,
        robot_keepalive,
        delay_rviz_after_joint_state_broadcaster_spawner,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "robot_ip",
            description="IP address by which the robot can be reached."),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Start robot with fake hardware mirroring command to its states."),
        DeclareLaunchArgument("gripper_max_velocity", default_value="100.0"),
        DeclareLaunchArgument("gripper_max_force",    default_value="100.0"),
        DeclareLaunchArgument(
            "use_internal_bus_gripper_comm",
            # thesis_ee has no physical gripper on the Kortex internal RS-485 bus.
            # Setting false prevents kortex.ros2_control.xacro from emitting
            # <joint name="finger_joint"> into the ros2_control URDF section,
            # which would cause MoveIt to error "Joint 'finger_joint' not found
            # in model 'gen3'" on every cycle.
            default_value="false"),
        DeclareLaunchArgument("use_sim_time",  default_value="false"),
        DeclareLaunchArgument("launch_rviz",   default_value="true"),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
