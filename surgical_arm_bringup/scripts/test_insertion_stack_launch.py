#!/usr/bin/env python3
"""Offline tests for launch/insertion_stack.launch.py (no ROS graph)."""
import importlib.util
import math
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.realpath(__file__))
LAUNCH = os.path.join(HERE, '..', 'launch', 'insertion_stack.launch.py')
spec = importlib.util.spec_from_file_location('insertion_stack_launch', LAUNCH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

T = 'translation: {x: 1.0, y: -0.5, z: 0.75}\n'


class T_(unittest.TestCase):
    def load(self, text):
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            f.write(text)
        try:
            return mod.load_extrinsic(f.name)
        finally:
            os.unlink(f.name)

    def test_shapes(self):
        for key in ('rotation_quat', 'rotation', 'rotation_quaternion'):
            xyz, q, child = self.load(
                T + key + ': {x: 0.0, y: 0.0, z: 0.0, w: 1.0}\n')
            self.assertEqual(xyz, (1.0, -0.5, 0.75))
            self.assertEqual(q, (0.0, 0.0, 0.0, 1.0))
            self.assertIsNone(child)

    def test_handeye_with_comments(self):
        xyz, q, child = self.load(
            '# Parent Frame: base_link\n# Camera Frame: front_cam_color_optical_frame\n'
            + T + 'rotation_quaternion: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}\n'
            'target: marker\nconsistency_residual: 0.002\n')
        self.assertEqual(child, 'front_cam_color_optical_frame')

    def test_wrong_parent(self):
        with self.assertRaises(RuntimeError):
            self.load('# Parent Frame: world\n' + T
                      + 'rotation: {x: 0, y: 0, z: 0, w: 1}\n')
        with self.assertRaises(RuntimeError):
            self.load('parent_frame: world\n' + T
                      + 'rotation: {x: 0, y: 0, z: 0, w: 1}\n')

    def test_camera_link_child(self):
        with self.assertRaises(RuntimeError):
            self.load('# Camera Frame: front_cam_link\n' + T
                      + 'rotation: {x: 0, y: 0, z: 0, w: 1}\n')

    def test_bad_norm_and_nonfinite(self):
        with self.assertRaises(RuntimeError):
            self.load(T + 'rotation: {x: 0, y: 0, z: 0, w: 1.1}\n')
        with self.assertRaises(RuntimeError):
            self.load(T + 'rotation: {x: .nan, y: 0, z: 0, w: 1}\n')
        with self.assertRaises(RuntimeError):
            self.load(T)

    def test_normalisation(self):
        _, q, _ = self.load(T + 'rotation: {x: 0, y: 0, z: 0, w: 1.005}\n')
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in q)), 1.0, places=12)

    def test_launch_args(self):
        ld = mod.generate_launch_description()
        names = {e.name for e in ld.entities if hasattr(e, 'default_value')}
        want = {'robot_ip', 'use_arm', 'use_wrist_cam', 'use_front_cam',
                'use_fusion', 'use_planning_scene', 'front_cam_serial',
                'planning_scene_delay_s', 'front_cam_calibration',
                'sensors_delay_s'}
        self.assertTrue(want <= names, want - names)

    def test_scripts_exist(self):
        for n in ('combine_cameras.py', 'setup_planning_scene.py'):
            self.assertTrue(os.path.isfile(os.path.join(mod.SCRIPTS_DIR, n)))


if __name__ == '__main__':
    unittest.main()
