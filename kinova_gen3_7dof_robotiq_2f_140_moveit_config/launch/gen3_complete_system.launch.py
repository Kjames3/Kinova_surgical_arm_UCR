from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value='192.168.1.10')
    use_fake_arg = DeclareLaunchArgument('use_fake_hardware', default_value='false')

    robot_ip = LaunchConfiguration('robot_ip')
    use_fake = LaunchConfiguration('use_fake_hardware')

    # 1. Robot arm + MoveIt + RViz + Kinova wrist camera (thesis_ee end effector)
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('kinova_gen3_7dof_robotiq_2f_140_moveit_config'),
            '/launch/robot.launch.py'
        ]),
        launch_arguments={
            'robot_ip': robot_ip,
            'use_fake_hardware': use_fake,
        }.items()
    )

    # 2. Intel RealSense D435i — topics at /realsense/camera/color/image_raw
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('realsense2_camera'),
            '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'camera_namespace': 'realsense',
            'enable_color': 'true',
            'enable_depth': 'true',
            'align_depth.enable': 'true'
        }.items()
    )

    # 3. OAK-D — topics at /oakd/oak/rgb/image_raw
    oakd_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('depthai_ros_driver'),
            '/launch/camera.launch.py'
        ]),
        launch_arguments={
            'namespace': 'oakd',
            'name': 'oak'
        }.items()
    )

    return LaunchDescription([
        robot_ip_arg,
        use_fake_arg,
        robot_launch,
        realsense_launch,
        oakd_launch,
    ])
