#!/usr/bin/env python3
"""One-shot bring-up for the Kinova Gen3 insertion experiment.

Replaces six manual terminals: arm + MoveIt, wrist camera, front RealSense,
front-camera extrinsic (static TF), marker fusion (combine_cameras.py) and the
planning scene (setup_planning_scene.py).

Run it BY PATH.  On the robot PC ``ros2 launch surgical_arm_bringup ...``
resolves to a stale second checkout, so the scripts started here are located
relative to this file (realpath), never through the package::

    source ~/ros2_kortex_ws/install/setup.bash
    ros2 launch ~/workspace/ros2_kortex_ws/src/Kinova_surgical_arm_UCR/surgical_arm_bringup/launch/insertion_stack.launch.py \\
        front_cam_calibration:=~/calib/front_cam.yaml

Arguments (all use_* default true): robot_ip, use_arm, use_wrist_cam,
use_front_cam, use_fusion, use_planning_scene, front_cam_serial,
sensors_delay_s (3.0, cameras/extrinsic/fusion start after the arm),
planning_scene_delay_s (20.0, after move_group), front_cam_calibration.

front_cam_calibration: YAML with translation{x,y,z} and a quaternion under
rotation_quat / rotation / rotation_quaternion, base_link -> color optical
frame.  Empty means the STALE 2026-10-08 values (camera has since been moved);
accuracy runs need a new calibration file.  A bad file fails the launch.

The interactive insertion.py needs stdin and is NOT launched.  In a second
terminal run:

    source ~/ros2_kortex_ws/install/setup.bash
    python3 ~/workspace/ros2_kortex_ws/src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/insertion.py --ros-args -p real_robot:=true -p execute_motion:=true -p step_by_step:=true -p target_source:=markers -p marker_look_always:=true -p azimuth_mode:=search -p target_x:=0.47 -p target_y:=0.0 -p max_velocity_scaling:=0.15 -p "azimuth_search_joint7_keepout_deg:=[-180.0, -120.0]"
"""
import math
import os
import re

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

SCRIPTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'scripts'))

# 2026-10-08 extrinsic, base_link -> front_cam_color_optical_frame.  STALE.
STALE_XYZ = (1.052352, -0.038352, 0.774732)
STALE_QXYZW = (0.633048, 0.617599, -0.320095, -0.339650)

REQUIRED_PARENT = 'base_link'
CHILD_FRAME = 'front_cam_color_optical_frame'
_QUAT_KEYS = ('rotation_quat', 'rotation', 'rotation_quaternion')


def _comment(text, label):
    m = re.search(r'^\s*#\s*' + label + r'\s*:\s*(\S+)', text,
                  re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def load_extrinsic(path):
    """Parse a hand-eye YAML -> (xyz, qxyzw, source_child_frame or None).

    Raises RuntimeError on any problem; never falls back silently.
    """
    path = os.path.expanduser(str(path))
    try:
        with open(path) as f:
            text = f.read()
        data = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f'Cannot read extrinsic {path}: {exc}')
    if not isinstance(data, dict):
        raise RuntimeError(f'Extrinsic {path}: top level is not a mapping')

    parent = data.get('parent_frame') or _comment(text, 'Parent Frame')
    if parent is not None and str(parent) != REQUIRED_PARENT:
        raise RuntimeError(
            f'Extrinsic {path}: parent frame is {parent!r}, expected '
            f'{REQUIRED_PARENT!r}')
    child = data.get('child_frame') or _comment(text, 'Camera Frame')
    if child is not None and 'color_optical_frame' not in str(child):
        raise RuntimeError(
            f'Extrinsic {path}: child frame {child!r} is not a '
            f'*color_optical_frame; a different transform (e.g. camera_link) '
            f'must not be used here')

    tr = data.get('translation')
    if not isinstance(tr, dict):
        raise RuntimeError(f'Extrinsic {path}: missing translation mapping')
    q = None
    for key in _QUAT_KEYS:
        if isinstance(data.get(key), dict):
            q = data[key]
            break
    if q is None:
        raise RuntimeError(
            f'Extrinsic {path}: no quaternion under any of {_QUAT_KEYS}')
    try:
        xyz = tuple(float(tr[k]) for k in 'xyz')
        quat = tuple(float(q[k]) for k in 'xyzw')
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'Extrinsic {path}: bad translation/quaternion: {exc!r}')
    if not all(math.isfinite(v) for v in xyz + quat):
        raise RuntimeError(f'Extrinsic {path}: non-finite value')
    norm = math.sqrt(sum(v * v for v in quat))
    if abs(norm - 1.0) > 0.01:
        raise RuntimeError(
            f'Extrinsic {path}: quaternion norm {norm:.4f} is off by >1%')
    quat = tuple(v / norm for v in quat)
    return xyz, quat, (str(child) if child is not None else None)


def _tf_setup(context):
    if context.perform_substitution(LaunchConfiguration('use_front_cam')).lower() \
            not in ('true', '1'):
        return []
    cal = context.perform_substitution(
        LaunchConfiguration('front_cam_calibration')).strip()
    msgs = []
    if cal:
        xyz, q, _ = load_extrinsic(cal)
        src = os.path.expanduser(cal)
    else:
        xyz, q, src = STALE_XYZ, STALE_QXYZW, 'built-in 2026-10-08 defaults'
        msgs.append(LogInfo(msg=(
            'WARNING: front_cam_calibration not given; using STALE 2026-10-08 '
            'extrinsic. The front camera was moved after that date, so '
            'accuracy runs need a new calibration file.')))
    msgs.append(LogInfo(msg=(
        f'Front-camera extrinsic from {src}: xyz={xyz} qxyzw={q} '
        f'({REQUIRED_PARENT} -> {CHILD_FRAME})')))
    msgs.append(Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='front_cam_static_tf', output='screen',
        arguments=['--x', repr(xyz[0]), '--y', repr(xyz[1]), '--z', repr(xyz[2]),
                   '--qx', repr(q[0]), '--qy', repr(q[1]),
                   '--qz', repr(q[2]), '--qw', repr(q[3]),
                   '--frame-id', REQUIRED_PARENT,
                   '--child-frame-id', CHILD_FRAME]))
    return msgs


def _script(name, *ros_args, condition=None):
    return ExecuteProcess(
        cmd=['python3', os.path.join(SCRIPTS_DIR, name), '--ros-args', *ros_args],
        cwd=SCRIPTS_DIR, output='screen', condition=condition)


def _include(pkg, launch_file, args, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(pkg), 'launch', launch_file])),
        launch_arguments=args.items(), condition=condition)


def generate_launch_description():
    defaults = [
        ('robot_ip', '192.168.1.10', 'Kortex arm IP'),
        ('use_arm', 'true', 'Start arm driver + MoveIt'),
        ('use_wrist_cam', 'true', 'Start Kinova wrist camera'),
        ('use_front_cam', 'true', 'Start front RealSense + extrinsic TF'),
        ('use_fusion', 'true', 'Start combine_cameras.py marker fusion'),
        ('use_planning_scene', 'true', 'Start setup_planning_scene.py'),
        ('front_cam_serial', '_938422070760',
         'RealSense serial (leading underscore keeps it a string)'),
        ('sensors_delay_s', '3.0', 'Delay of cameras/fusion after the arm'),
        ('planning_scene_delay_s', '20.0', 'Delay of planning scene script'),
        ('front_cam_calibration', '',
         'Hand-eye YAML (base_link -> color optical frame); empty = STALE'),
    ]
    ld = LaunchDescription([DeclareLaunchArgument(n, default_value=v, description=d)
                            for n, v, d in defaults])
    lc = LaunchConfiguration

    ld.add_action(_include(
        'kinova_gen3_7dof_robotiq_2f_140_moveit_config', 'robot.launch.py',
        {'robot_ip': lc('robot_ip'), 'launch_rviz': 'false'},
        IfCondition(lc('use_arm'))))

    sensors = [
        _include('kinova_vision', 'kinova_vision.launch.py',
                 {'device': lc('robot_ip'), 'camera': 'camera',
                  'launch_color': 'true', 'launch_depth': 'false'},
                 IfCondition(lc('use_wrist_cam'))),
        _include('realsense2_camera', 'rs_launch.py',
                 {'camera_namespace': 'realsense', 'camera_name': 'front_cam',
                  'serial_no': lc('front_cam_serial'),
                  'enable_color': 'true', 'enable_depth': 'false',
                  'pointcloud.enable': 'false', 'publish_tf': 'false',
                  'rgb_camera.color_profile': '640x480x15'},
                 IfCondition(lc('use_front_cam'))),
        OpaqueFunction(function=_tf_setup),
        _script('combine_cameras.py',
                '-p', 'enable_visualization:=false', '-p', 'fusion_mode:=rays',
                '-p', 'z_sign:=1.0', '-p', 'table_z:=-0.03',
                '-p', 'marker_height_above_table:=0.005',
                '-p', 'realsense_image_topic:=/realsense/front_cam/color/image_raw',
                '-p', 'realsense_info_topic:=/realsense/front_cam/color/camera_info',
                condition=IfCondition(lc('use_fusion'))),
    ]
    ld.add_action(TimerAction(period=lc('sensors_delay_s'), actions=sensors))

    scene = _script('setup_planning_scene.py',
                    condition=IfCondition(lc('use_planning_scene')))
    ld.add_action(TimerAction(period=lc('planning_scene_delay_s'), actions=[scene]))
    return ld
