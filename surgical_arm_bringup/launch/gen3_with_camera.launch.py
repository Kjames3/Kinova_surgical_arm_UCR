from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    surgical_arm_bringup_dir = get_package_share_directory('surgical_arm_bringup')
    
    # Launch the robot
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([surgical_arm_bringup_dir, '/launch/gen3.launch.py']),
        launch_arguments={
            'robot_ip': '192.168.1.10',
            'gripper': 'robotiq_2f_140',
            'gripper_joint_name': 'finger_joint'
        }.items()
    )
    
    # Static transform from wrist to camera (Commented out: URDF provides the true CAD offsets)
    # camera_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='camera_tf_publisher',
    #     arguments=['0', '0', '0', '0', '0', '0', 'bracelet_link', 'camera_link']
    # )
    
    return LaunchDescription([
        robot_launch,
        # camera_tf
    ])
