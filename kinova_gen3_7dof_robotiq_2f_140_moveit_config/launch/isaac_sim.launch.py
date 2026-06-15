"""
Launch file for controlling the Kinova Gen3 7DOF + Robotiq 2F-140 via Isaac Sim 4.x.

Assumes Isaac Sim is already running (started separately via isaac_sim_gen3.py)
and publishing /isaac_joint_states + subscribing to /isaac_joint_commands.

Usage:
  ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config isaac_sim.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("vision", default_value="false"),
        DeclareLaunchArgument(
            "isaac_joint_commands",
            default_value="/isaac_joint_commands",
            description="Topic Isaac Sim subscribes to for joint position commands",
        ),
        DeclareLaunchArgument(
            "isaac_joint_states",
            default_value="/isaac_joint_states",
            description="Topic Isaac Sim publishes joint states on",
        ),
    ]

    launch_rviz = LaunchConfiguration("launch_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    vision = LaunchConfiguration("vision")
    isaac_joint_commands = LaunchConfiguration("isaac_joint_commands")
    isaac_joint_states = LaunchConfiguration("isaac_joint_states")

    description_arguments = {
        "robot_ip": "xxx.yyy.zzz.www",
        "use_fake_hardware": "false",
        "gripper": "thesis_ee",
        "dof": "7",
        "sim_isaac": "true",
        "vision": vision,
        "isaac_joint_commands": isaac_joint_commands,
        "isaac_joint_states": isaac_joint_states,
    }

    moveit_config = (
        MoveItConfigsBuilder("gen3", package_name="kinova_gen3_7dof_robotiq_2f_140_moveit_config")
        .robot_description(mappings=description_arguments)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description, {"use_sim_time": use_sim_time}],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="both",
        parameters=[
            moveit_config.robot_description,
            PathJoinSubstitution(
                [
                    FindPackageShare("kinova_gen3_7dof_robotiq_2f_140_moveit_config"),
                    "config",
                    "ros2_controllers.yaml",
                ]
            ),
            {"use_sim_time": use_sim_time},
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )

    # Delay controller spawning until ros2_control_node is ready
    delay_controllers = TimerAction(
        period=3.0,
        actions=[
            joint_state_broadcaster_spawner,
            joint_trajectory_controller_spawner,
        ],
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="log",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time}],
        arguments=["--ros-args", "--log-level", "fatal"],
        condition=IfCondition(launch_rviz),
    )

    rviz_config_path = os.path.join(
        get_package_share_directory("kinova_gen3_7dof_robotiq_2f_140_moveit_config"),
        "config",
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_path],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(launch_rviz),
    )

    # Delay RViz until joint_state_broadcaster is up so TF is available
    delay_rviz = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        )
    )

    # Populate table + pole collision objects once MoveIt's ApplyPlanningScene
    # service is up. 5 s after spawners fire is conservative but reliable.
    setup_planning_scene = Node(
        package="kortex_bringup",
        executable="setup_planning_scene.py",
        output="screen",
    )
    delay_planning_scene = TimerAction(
        period=8.0,
        actions=[setup_planning_scene],
    )

    return LaunchDescription(
        declared_arguments
        + [
            robot_state_publisher,
            ros2_control_node,
            delay_controllers,
            move_group_node,
            delay_rviz,
            delay_planning_scene,
        ]
    )
