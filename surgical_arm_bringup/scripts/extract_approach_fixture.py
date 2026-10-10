#!/usr/bin/env python3
"""Read a single-file SQLite bag offline; never initialize ROS or publish."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from moveit_msgs.msg import RobotTrajectory
from tf2_msgs.msg import TFMessage



def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', default='Phase0-1_blended[0]')
    args = parser.parse_args()
    BAG = args.bag.resolve()
    databases = list(BAG.glob('*.db3'))
    if len(databases) != 1:
        raise ValueError('Expected exactly one SQLite database; split bags are unsupported')
    DB = databases[0]
    LABEL = args.phase
    connection = sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True)
    topics = {name: identifier for identifier, name in connection.execute('SELECT id,name FROM topics')}
    events = [(stamp, deserialize_message(blob, String).data) for stamp, blob in connection.execute(
        'SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/insertion/phase'],))]
    starts=[stamp for stamp,text in events if text=='start:'+LABEL]
    if len(starts)!=1: raise ValueError('Expected exactly one selected phase start')
    start=starts[0]
    plan_start = None
    planning_calls = []
    for event_time, text in events:
        if event_time >= start:
            break
        if text == 'plan_start':
            plan_start = event_time
        elif text.startswith('plan_end:') and plan_start is not None:
            planning_calls.append({'start_ns': plan_start, 'end_ns': event_time,
                                   'wall_s': (event_time-plan_start)/1e9,
                                   'outcome': text.split(':',1)[1]})
            plan_start = None
    if not planning_calls or planning_calls[-1]['outcome'] != 'ok':
        raise ValueError('No successful planning call before selected motion phase')
    end, outcome = next((stamp, text.rsplit(':', 1)[1]) for stamp, text in events
                        if stamp > start and text.startswith('end:' + LABEL + ':'))
    plans = list(connection.execute('SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp',
                                   (topics['/insertion/planned_trajectory'], start, end)))
    if len(plans) != 1:
        raise RuntimeError(f'Expected one trajectory in approach phase, found {len(plans)}')
    stamp, raw = plans[0]
    trajectory = deserialize_message(raw, RobotTrajectory)
    if not trajectory.joint_trajectory.points:
        raise RuntimeError('Empty planned trajectory')
    def joint_sample(time):
        t, blob = connection.execute('SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp<=? ORDER BY timestamp DESC LIMIT 1',
                                     (topics['/joint_states'], time)).fetchone()
        msg = deserialize_message(blob, JointState)
        joints = dict(zip(msg.name, msg.position))
        if any(j not in joints for j in trajectory.joint_trajectory.joint_names):
            raise RuntimeError('Recorded joint state is missing a trajectory joint')
        if time - t > 100_000_000:
            raise RuntimeError('Joint sample is older than 100 ms')
        return {'bag_timestamp_ns': t, 'age_s': (time-t)/1e9,
                'positions': {j: joints[j] for j in trajectory.joint_trajectory.joint_names}}
    tip_transforms = []
    for blob, in connection.execute('SELECT data FROM messages WHERE topic_id=?', (topics['/tf_static'],)):
        for transform in deserialize_message(blob, TFMessage).transforms:
            if transform.child_frame_id == 'assembly_tip':
                tip_transforms.append(message_to_ordereddict(transform))
    target = None
    params = {}
    keep = ['table_z', 'container_height', 'hover_above_top', 'approach_clearance', 'insertion_angle_deg',
            'target_depth_mm', 'blend_radius', 'max_velocity_scaling', 'transit_velocity_scaling', 'world_frame', 'ee_link', 'tip_link']
    for blob, in connection.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/insertion/run_info'],)):
        value = json.loads(deserialize_message(blob, String).data)
        if value.get('event') == 'target':
            target = value
        elif value.get('event') == 'params':
            params = {key: value[key] for key in keep if key in value}
    fixture = {'bag': str(BAG), 'bag_db_sha256': digest(DB),
               'phase': LABEL, 'outcome': outcome, 'phase_start_ns': start, 'phase_end_ns': end,
               'phase_duration_s': (end-start)/1e9, 'preceding_planning_call': planning_calls[-1], 'trajectory_recorded_ns': stamp,
               'start_joint_state': joint_sample(start), 'end_joint_state': joint_sample(end),
               'trajectory': message_to_ordereddict(trajectory), 'recorded_tip_transforms': tip_transforms,
               'recorded_target': target, 'scene_parameters': params,
               'limitations': ['World geometry is reconstructed separately; the recorded run had no container.',
                               'Recorded trajectory is a blended approach; endpoint-only planning is not an equivalent constrained path benchmark.']}
    descriptions = list(connection.execute('SELECT data FROM messages WHERE topic_id=?',
                                           (topics['/robot_description'],)))
    if not descriptions:
        raise ValueError('Missing recorded URDF')
    urdf = deserialize_message(descriptions[-1][0], String).data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    urdf_path = args.output.with_suffix('.urdf')
    urdf_path.write_text(urdf)
    fixture['recorded_urdf'] = urdf_path.name
    fixture['recorded_urdf_sha256'] = hashlib.sha256(urdf.encode()).hexdigest()
    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, default=list) + '\n')
    print(f"Extracted {LABEL}: {len(trajectory.joint_trajectory.points)} points, phase duration {(end-start)/1e9:.3f} s, outcome={outcome}")
    print(f"Start joint sample age: {fixture['start_joint_state']['age_s']*1000:.3f} ms")


if __name__ == '__main__':
    main()
