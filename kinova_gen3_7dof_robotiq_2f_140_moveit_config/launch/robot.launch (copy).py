# moveit_gazebo_obb.py — Kinova Gen3 + Gazebo + YOLOv8 OBB pick-and-place
# Modelled directly on the lab's working robot.launch.py
# ROS2: Humble | Gazebo (gz_sim) | MoveIt2

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):

    launch_rviz   = LaunchConfiguration("launch_rviz")
    use_sim_time  = LaunchConfiguration("use_sim_time")
    world         = LaunchConfiguration("world")

    # -----------------------------------------------------------------------
    # xacro arguments — same pattern as your working robot.launch.py
    # -----------------------------------------------------------------------
    launch_arguments = {
        "robot_ip":                       "192.168.1.10",  # unused in sim
        "use_fake_hardware":              "true",
        "arm":                            "gen3",
        "gripper":                        "robotiq_2f_140",
        "gripper_joint_name":             "finger_joint",
        "dof":                            "7",
        "vision":                         "false",
        "sim_gazebo":                     "true",
        "sim_ignition":                   "false",
        "gripper_max_velocity":           "100.0",
        "gripper_max_force":              "100.0",
        "use_internal_bus_gripper_comm":  "false",
    }

    # -----------------------------------------------------------------------
    # MoveIt config — exact same builder pattern as your robot.launch.py
    # -----------------------------------------------------------------------
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

    moveit_config.moveit_cpp.update({"use_sim_time": True})

    # -----------------------------------------------------------------------
    # Gazebo simulation
    # -----------------------------------------------------------------------
    kinova_moveit_pkg = get_package_share_directory("kinova_moveit_config")

    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=os.path.join(kinova_moveit_pkg, "worlds"),
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch"
            ), "/gz_sim.launch.py"
        ]),
        launch_arguments=[
            ("gz_args", [
                world, ".sdf",
                " -v 4",
                " -r",
                " --physics-engine gz-physics-bullet-featherstone-plugin"
            ])
        ]
    )

    # -----------------------------------------------------------------------
    # Spawn robot into Gazebo from /robot_description topic
    # -----------------------------------------------------------------------
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "/robot_description",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "1.02",
            "-R", "0.0",
            "-P", "0.0",
            "-Y", "0.0",
            "-name", "gen3",
            "-allow_renaming", "false",
        ],
    )

    # -----------------------------------------------------------------------
    # Robot state publisher
    # -----------------------------------------------------------------------
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": True},
        ],
    )

    # -----------------------------------------------------------------------
    # TF: world → base_link
    # -----------------------------------------------------------------------
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
        parameters=[{"use_sim_time": True}],
    )

    # -----------------------------------------------------------------------
    # move_group node  (same pattern as your working robot.launch.py)
    # -----------------------------------------------------------------------
    move_group_node = Node(
        additional_env={
            "AMENT_PREFIX_PATH": (
                "/home/lab/workspace/ros2_kortex_ws/install/moveit_ros_perception:"
                + os.environ.get("AMENT_PREFIX_PATH", "")
            )
        },
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
        ],
    )

    # -----------------------------------------------------------------------
    # Our MoveItPy arm controller (pick-and-place logic)
    # -----------------------------------------------------------------------
    moveit_py_node = Node(
        name="moveit_py",
        package="kinova_moveit_config",
        executable="arm_control_from_UI.py",
        output="both",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
        ],
    )

    # -----------------------------------------------------------------------
    # RViz
    # -----------------------------------------------------------------------
    rviz_config_file = os.path.join(
        get_package_share_directory("kinova_gen3_7dof_robotiq_2f_140_moveit_config"),
        "config", "moveit.rviz",
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
            {"use_sim_time": True},
        ],
    )

    # -----------------------------------------------------------------------
    # ROS ↔ Gazebo bridge for the simulated camera
    # -----------------------------------------------------------------------
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/image_raw@sensor_msgs/msg/Image@gz.msgs.Image"],
        output="screen",
    )

    # -----------------------------------------------------------------------
    # Controller spawners — exact names from kortex ros2_controllers.yaml
    # -----------------------------------------------------------------------
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
        ],
    )

    robot_traj_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "-c", "/controller_manager",
        ],
    )

    robot_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "robotiq_gripper_controller",
            "-c", "/controller_manager",
        ],
    )

    # Start RViz after joint_state_broadcaster is ready
    delay_rviz = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        ),
        condition=IfCondition(launch_rviz),
    )

    return [
        gazebo_resource_path,
        gazebo,
        gz_spawn_entity,
        robot_state_publisher,
        static_tf,
        move_group_node,
        bridge,
        moveit_py_node,
        joint_state_broadcaster_spawner,
        robot_traj_controller_spawner,
        robot_hand_controller_spawner,
        delay_rviz,
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "world",
            default_value="arm_on_the_table",
            description="Gazebo world .sdf filename (no extension)",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock",
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
