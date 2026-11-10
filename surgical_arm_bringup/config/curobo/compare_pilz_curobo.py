#!/usr/bin/env python3
"""Offline Pilz vs cuRobo Phase 0 comparison on the same start state (M2 item 3).

Pilz side: the Phase 0 trajectory insertion.py published on
/insertion/planned_trajectory in an archived insertion bag (Phase0_approach, or
Phase0-1_blended when the transit was blended). cuRobo side: the planning-only
preview response archived from the same start.

Both are scored with ONE kinematic model (gen3_surgical.urdf next to this file)
so differences come from the planners, not from FK.

    source /opt/ros/humble/setup.bash
    python3 compare_pilz_curobo.py \
        --pair ~/robot-logs/insertion_bags/insertion_20261009_163412 \
               ~/robot-logs/2026-10-09/upright_preview/163412 [--pair ...] [--json out.json]

System Python + ROS (rosbag2_py, PyKDL); no robot, no move_group, no cuRobo.
Tilt uses the preview's definition (bracelet rotation vs the measured
tool-down quaternion), so it is directly comparable to approach_max_tilt_deg.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts'))
from arm_kinematics import KinDynModel  # noqa: E402

URDF = HERE / 'gen3_surgical.urdf'
JOINTS = [f'joint_{i}' for i in range(1, 8)]
PHASE0_LABELS = ('Phase0_approach', 'Phase0-1_blended', 'Phase0_direct')


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pilz_phase0(bag):
    """First Phase 0 trajectory in the bag -> (label, t[N], q[N,7], qd[N,7])."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    from moveit_msgs.msg import RobotTrajectory
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(
        topics=['/insertion/phase', '/insertion/planned_trajectory']))
    label = None
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == '/insertion/phase':
            text = deserialize_message(data, String).data
            label = (text[len('start:'):].split('[')[0]
                     if text.startswith('start:') else None)
        elif label in PHASE0_LABELS:
            jt = deserialize_message(data, RobotTrajectory).joint_trajectory
            idx = [jt.joint_names.index(n) for n in JOINTS]
            t = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points])
            q = np.array([[p.positions[i] for i in idx] for p in jt.points])
            qd = np.array([[p.velocities[i] for i in idx] if p.velocities else [0.] * 7
                           for p in jt.points])
            return label, t, q, qd
    raise ValueError(f'{bag}: no Phase 0 trajectory found')


def curobo_preview(run_dir):
    req = json.loads((Path(run_dir) / 'request.json').read_text())
    resp = json.loads((Path(run_dir) / 'response.json').read_text())
    names = resp['joint_names']
    idx = [names.index(n) for n in JOINTS]
    pts = resp['points']
    t = np.array([p['time_from_start'] for p in pts])
    q = np.array([[p['positions'][i] for i in idx] for p in pts])
    qd = np.array([[p['velocities'][i] for i in idx] for p in pts])
    stages = [p.get('stage') for p in pts]
    return req, resp, t, q, qd, stages


def score(t, q, qd, tip, bracelet, vertical_R, approach_mask=None):
    local_axis = vertical_R.T @ np.array([0., 0., -1.])
    tips, tilts = [], []
    for row in q:
        p, _ = tip.fk(row)
        _, qb = bracelet.fk(row)
        axis = quat_to_R(qb) @ local_axis
        tips.append(p)
        tilts.append(math.degrees(math.acos(max(-1., min(1., -axis[2])))))
    tips, tilts = np.array(tips), np.array(tilts)
    step = np.diff(tips, axis=0)
    out = {
        'duration_s': float(t[-1]),
        'n_points': int(len(t)),
        'joint_path_rad': float(np.sum(np.abs(np.diff(q, axis=0)))),
        'tip_path_m': float(np.sum(np.linalg.norm(step, axis=1))),
        'tip_straight_line_m': float(np.linalg.norm(tips[-1] - tips[0])),
        'peak_joint_vel_rad_s': float(np.max(np.abs(qd))),
        'max_tilt_deg': float(np.max(tilts)),
        'final_tilt_deg': float(tilts[-1]),
        'min_tip_z_m': float(np.min(tips[:, 2])),
        'start_tip_m': tips[0].round(4).tolist(),
        'end_tip_m': tips[-1].round(4).tolist(),
    }
    out['path_ratio'] = out['tip_path_m'] / max(out['tip_straight_line_m'], 1e-9)
    if approach_mask is not None and approach_mask.any():
        out['approach_stage_max_tilt_deg'] = float(np.max(tilts[approach_mask]))
    return out


def compare(bag, run_dir, tip, bracelet):
    label, pt, pq, pqd = pilz_phase0(bag)
    req, resp, ct, cq, cqd, stages = curobo_preview(run_dir)
    vR = quat_to_R(req['approach']['quaternion_xyzw'])
    start = np.array([req['start_joints'][n] for n in JOINTS])
    d = pq[0] - start
    d = np.where([n in ('joint_1', 'joint_3', 'joint_5', 'joint_7') for n in JOINTS],
                 np.remainder(d + math.pi, 2 * math.pi) - math.pi, d)
    mask = np.array([s == 'upright_approach' for s in stages])
    return {
        'bag': Path(bag).name, 'preview': Path(run_dir).name, 'pilz_label': label,
        'start_mismatch_rad': float(np.max(np.abs(d))),
        'curobo_vel_scale': req['vel_scale'],
        'pilz': score(pt, pq, pqd, tip, bracelet, vR),
        'curobo': score(ct, cq, cqd, tip, bracelet, vR, mask),
    }


ROWS = [('duration_s', 'duration [s]', '.1f'), ('tip_path_m', 'tip path [m]', '.3f'),
        ('tip_straight_line_m', 'tip start->end [m]', '.3f'), ('path_ratio', 'path / straight', '.2f'),
        ('joint_path_rad', 'joint path [rad]', '.2f'), ('peak_joint_vel_rad_s', 'peak joint vel', '.3f'),
        ('max_tilt_deg', 'max tilt [deg]', '.1f'), ('final_tilt_deg', 'final tilt [deg]', '.2f'),
        ('min_tip_z_m', 'min tip z [m]', '.3f')]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pair', nargs=2, action='append', required=True, metavar=('BAG', 'PREVIEW_DIR'))
    ap.add_argument('--json', help='write the full results here')
    a = ap.parse_args(argv)
    tip = KinDynModel(str(URDF), 'base_link', 'assembly_tip')
    bracelet = KinDynModel(str(URDF), 'base_link', 'bracelet_link')
    results = [compare(Path(b).expanduser(), Path(p).expanduser(), tip, bracelet) for b, p in a.pair]
    for r in results:
        print(f"\n== {r['bag']}  (Pilz {r['pilz_label']} vs cuRobo preview {r['preview']}; "
              f"start mismatch {r['start_mismatch_rad']:.4f} rad, cuRobo vel_scale {r['curobo_vel_scale']})")
        print(f"  {'':22s}{'Pilz':>10s}{'cuRobo':>10s}")
        for key, name, fmt in ROWS:
            print(f"  {name:22s}{r['pilz'][key]:>10{fmt}}{r['curobo'][key]:>10{fmt}}")
        if 'approach_stage_max_tilt_deg' in r['curobo']:
            print(f"  {'cuRobo upright stage max tilt':32s}{r['curobo']['approach_stage_max_tilt_deg']:.2f} deg")
        print(f"  end tip  Pilz {r['pilz']['end_tip_m']}  cuRobo {r['curobo']['end_tip_m']}")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
