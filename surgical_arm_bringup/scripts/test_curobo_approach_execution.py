#!/usr/bin/env python3
"""Offline tests for the M2 cuRobo approach execution gate. No ROS, no robot.

    python3 test_curobo_approach_execution.py
    CUROBO_ARCHIVE=~/robot-logs/2026-10-09/upright_preview python3 test_curobo_approach_execution.py

With CUROBO_ARCHIVE set, every policy-v1 response.json under it (with a sibling
request.json) is replayed through the gate as a real-data check.
"""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curobo_approach_execution import validate_for_execution, ExecutionRejected, ARM_JOINTS
from curobo_planning_preview import check_preview_parameters
from insertion_geometry import GEN3_BOUNDED_LIMITS

# What insertion.py actually uses on the robot: get_robot_info() reads the URDF
# limits (1.3963 joints 1-4, 1.2218 joints 5-7); 0.4 rad/s^2 is hard-coded.
VEL = {j: (1.3963 if i < 4 else 1.2218) for i, j in enumerate(ARM_JOINTS)}
ACC = {j: 0.4 for j in ARM_JOINTS}
KW = dict(vel_scale=0.15, max_joint_vel=VEL, max_joint_acc=ACC,
          bounded_limits=GEN3_BOUNDED_LIMITS, max_tilt_deg=15.0)
START = dict(zip(ARM_JOINTS, [0.0, -0.30, math.pi - 1e-5, -1.66, 0.0, -1.79, 0.0]))


def synthetic(n=101, T=10.0, dq=0.3):
    """Rest-to-rest cosine move of joint_2 by dq — well inside every limit."""
    pts = []
    for k in range(n):
        t = T * k / (n - 1)
        s = 0.5 * (1 - math.cos(math.pi * t / T))
        sd = 0.5 * math.pi / T * math.sin(math.pi * t / T)
        sdd = 0.5 * (math.pi / T) ** 2 * math.cos(math.pi * t / T)
        q = [START[j] for j in ARM_JOINTS]; v = [0.0] * 7; a = [0.0] * 7
        q[1] += dq * s; v[1] = dq * sd; a[1] = dq * sdd
        if k in (0, n - 1):
            v[1] = 0.0
        pts.append(dict(positions=q, velocities=v, accelerations=a, time_from_start=t))
    return dict(success=True, joint_names=list(ARM_JOINTS), points=pts, error='',
                meta=dict(backend='curobo', validated=True, upright_policy_version=1,
                          approach_max_tilt_deg=1.25))


class Gate(unittest.TestCase):
    def test_good_plan_passes(self):
        s = validate_for_execution(synthetic(), START, **KW)
        self.assertLess(s['peak_velocity_ratio'], 1.0)
        self.assertAlmostEqual(s['duration_s'], 10.0)

    def mutate(self, fn, msg):
        r = synthetic(); fn(r)
        with self.assertRaisesRegex(ExecutionRejected, msg):
            validate_for_execution(r, START, **KW)

    def test_rejections(self):
        cases = {
            'not validated': (lambda r: r['meta'].update(validated=False), 'validated'),
            'stub backend': (lambda r: r['meta'].update(backend='stub'), 'validated'),
            'old policy': (lambda r: r['meta'].pop('upright_policy_version'), 'predates'),
            'tilted': (lambda r: r['meta'].update(approach_max_tilt_deg=16.), 'tilt'),
            'nan tilt': (lambda r: r['meta'].update(approach_max_tilt_deg=float('nan')), 'tilt'),
            'joint names': (lambda r: r['joint_names'].__setitem__(0, 'joint_8'), 'joint names'),
            'one point': (lambda r: r.__setitem__('points', r['points'][:1]), 'two waypoints'),
            'nan pos': (lambda r: r['points'][5]['positions'].__setitem__(2, float('nan')), 'non-finite'),
            'time back': (lambda r: r['points'][5].update(time_from_start=0.0), 'strictly'),
            'time gap': (lambda r: [p.update(time_from_start=p['time_from_start'] + 1.)
                                    for p in r['points'][50:]], 'gap'),
            '2pi seam': (lambda r: [p['positions'].__setitem__(2, p['positions'][2] - 2 * math.pi)
                                   for p in r['points'][50:]], 'jumps'),
            'joint limit': (lambda r: [p['positions'].__setitem__(5, -2.2) for p in r['points']], 'beyond'),
            'too fast': (lambda r: r['points'][50]['velocities'].__setitem__(1, 0.25), 'velocity'),
            'too hard': (lambda r: r['points'][10]['accelerations'].__setitem__(1, 0.5), 'acceleration'),
            'not at rest': (lambda r: r['points'][-1]['velocities'].__setitem__(0, 0.01), 'rest'),
            'wrong start': (lambda r: [p['positions'].__setitem__(0, p['positions'][0] + 0.05)
                                      for p in r['points']], 'from live'),
        }
        for name, (fn, msg) in cases.items():
            with self.subTest(name):
                self.mutate(fn, msg)

    def test_seam_start_is_same_pose(self):
        live = dict(START, joint_3=-math.pi + 1e-5)   # encoder flipped across the seam
        validate_for_execution(synthetic(), live, **KW)

    def test_vel_scale_capped(self):
        with self.assertRaisesRegex(ExecutionRejected, 'vel_scale'):
            validate_for_execution(synthetic(), START, **dict(KW, vel_scale=0.25))

    def test_missing_live_joint(self):
        live = dict(START); live.pop('joint_4')
        with self.assertRaisesRegex(ExecutionRejected, 'no live state'):
            validate_for_execution(synthetic(), live, **KW)


class Params(unittest.TestCase):
    BASE = {'transit_velocity_scaling': .15}

    def test_preview_still_rejects_bare_execute(self):
        with self.assertRaises(ValueError):
            check_preview_parameters(dict(self.BASE, execute_motion=True))

    def test_execute_needs_all_guards(self):
        full = dict(self.BASE, curobo_execute_approach=True, execute_motion=True,
                    step_by_step=True, real_robot=True)
        check_preview_parameters(full)
        for drop in ('execute_motion', 'step_by_step', 'real_robot'):
            with self.subTest(drop), self.assertRaises(ValueError):
                check_preview_parameters(dict(full, **{drop: False}))
        for flag in ('teach_mode', 'replay_mode', 'use_action_server'):
            with self.subTest(flag), self.assertRaises(ValueError):
                check_preview_parameters(dict(full, **{flag: True}))


class JointGoal(unittest.TestCase):
    def test_reorders_to_arm_joint_order(self):
        try:
            from insertion import joint_goal_from_response
        except ImportError:
            self.skipTest('insertion.py needs ROS sourced')
        r = synthetic()
        r['joint_names'] = list(reversed(r['joint_names']))
        for p in r['points']:
            p['positions'] = list(reversed(p['positions']))
        goal = joint_goal_from_response(r, list(ARM_JOINTS))
        self.assertIsInstance(goal, list)
        self.assertAlmostEqual(goal[1], START['joint_2'] + 0.3)
        self.assertAlmostEqual(goal[2], START['joint_3'])
        r['points'][-1]['positions'][-3] = 4.0953          # joint_3, reversed order
        wrapped = joint_goal_from_response(r, list(ARM_JOINTS), {'joint_3'})
        self.assertAlmostEqual(wrapped[2], 4.0953 - 2 * math.pi)


class Archive(unittest.TestCase):
    def test_archived_responses(self):
        root = os.environ.get('CUROBO_ARCHIVE')
        if not root:
            self.skipTest('CUROBO_ARCHIVE not set')
        n = 0
        for resp_path in sorted(Path(root).expanduser().rglob('response.json')):
            req_path = resp_path.with_name('request.json')
            resp = json.loads(resp_path.read_text())
            if not req_path.exists() or resp.get('meta', {}).get('upright_policy_version') != 1:
                continue
            req = json.loads(req_path.read_text())
            with self.subTest(str(resp_path.parent.name)):
                s = validate_for_execution(resp, req['start_joints'],
                                           **dict(KW, vel_scale=req['vel_scale']))
                print(f"\n  {resp_path.parent.name}: {json.dumps(s)}")
            n += 1
        self.assertGreater(n, 0, 'no policy-v1 response/request pairs found')


if __name__ == '__main__':
    unittest.main(verbosity=1)
