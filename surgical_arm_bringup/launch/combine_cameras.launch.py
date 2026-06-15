from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    enable_vis_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable OpenCV Visual Dashboard popup window'
    )

    # NOTE: Do NOT set RMW_IMPLEMENTATION here.  The camera nodes
    # (realsense2_camera_node, depthai/OAK-D) launched by
    # gen3_complete_system.launch.py use FastRTPS (the system default).
    # Forcing CycloneDDS on this node would put it on a different
    # middleware bus, making all camera topics invisible → cameras OFFLINE.
    # All nodes in the system must share the same RMW.
    #
    # AXIS SIGN TUNING
    # If the fused container coordinate has the wrong sign on any axis, adjust
    # the x_sign / y_sign / z_sign parameters here (only +1.0 or -1.0).
    # Check the "Center pipeline — raw:" log line: the raw value is the
    # camera→world TF result before sign correction.  If the raw value already
    # has the correct sign, the corresponding sign param should be +1.0.
    combine_cameras_node = Node(
        package='surgical_arm_bringup',
        executable='combine_cameras.py',
        name='combine_cameras_node',
        output='screen',
        parameters=[{
            'enable_visualization': LaunchConfiguration('enable_visualization'),
            # Global RealSense D435i (eye-to-base calibrated TF):
            #   raw X is already positive when container is in front of robot → +1.0
            #   raw Y matches world Y after TF → +1.0
            #   raw Z is negative from camera frame → flip to positive → -1.0
            'x_sign':  1.0,
            'y_sign':  1.0,
            'z_sign': -1.0,
        }]
    )

    return LaunchDescription([
        enable_vis_arg,
        combine_cameras_node
    ])
