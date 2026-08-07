import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- Arguments ---
    name = LaunchConfiguration('name')
    namespace = LaunchConfiguration('namespace')
    
    # 1. Start ArUco Ros node to detect the calibration marker
    # We assume the camera is publishing to /camera/color/image_raw and /camera/color/camera_info
    aruco_node = Node(
        package='aruco_ros',
        executable='single',
        name='aruco_single',
        parameters=[{
            'image_is_rectified': True,
            'marker_size': 0.1,         # SIZE OF YOUR ARUCO MARKER IN METERS (update this!)
            'marker_id': 200,           # ID OF YOUR ARUCO MARKER (update this!)
            'reference_frame': 'camera_color_optical_frame',
            'camera_frame': 'camera_color_optical_frame',
            'marker_frame': 'camera_marker'
        }],
        remappings=[
            ('/camera_info', '/camera/color/camera_info'),
            ('/image', '/camera/color/image_raw')
        ]
    )

    # 2. Start easy_handeye2
    # We include the standard easy_handeye2 launch file and pass our specific TF frames.
    easy_handeye2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('easy_handeye2'),
                'launch',
                'calibrate.launch.py'
            )
        ),
        launch_arguments={
            'name': name,
            'namespace': namespace,
            'calibration_type': 'eye_in_hand',
            
            # The fixed frame of the robot (base)
            'robot_base_frame': 'base_link',
            
            # The frame that the camera is physically attached to
            'robot_effector_frame': 'bracelet_link',
            
            # The frame of the camera that observes the marker
            'tracking_base_frame': 'camera_color_optical_frame',
            
            # The frame of the marker itself
            'tracking_marker_frame': 'camera_marker',
            
            # Set to false because we already have MoveIt running via the kinova drivers
            'freehand_robot_movement': 'false',
        }.items()
    )

    return LaunchDescription([
        DeclareLaunchArgument('name', default_value='kinova_eye_in_hand_calib'),
        DeclareLaunchArgument('namespace', default_value=''),
        aruco_node,
        easy_handeye2_launch
    ])
