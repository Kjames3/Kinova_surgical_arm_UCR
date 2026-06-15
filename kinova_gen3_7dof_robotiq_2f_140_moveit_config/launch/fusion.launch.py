# fusion.launch.py — point cloud fusion node
#
# Fuses RealSense D435I + OAK-D + wrist camera clouds into
# /fused_pointcloud (frame: world) at 10 Hz.
#
# Launch AFTER cameras.launch.py is streaming all cameras.
#
# Workflow:
#   Terminal 1:  ros2 launch ... robot.launch.py  robot_ip:=192.168.1.10
#   Terminal 2:  ros2 launch ... cameras.launch.py robot_ip:=192.168.1.10
#   Terminal 3:  ros2 launch ... fusion.launch.py

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    fusion_script = os.path.expanduser(
        "~/workspace/ros2_kortex_ws/pointcloud_fusion.py"
    )
    return LaunchDescription([
        ExecuteProcess(
            cmd=["python3", fusion_script],
            output="screen",
            emulate_tty=True,
        )
    ])
