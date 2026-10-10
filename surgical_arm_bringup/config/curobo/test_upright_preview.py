#!/usr/bin/env python3
"""Orientation regression tests independent of planner costs and GPU solves."""
import unittest
import numpy as np
from preview_backend import tool_tilts_deg, require_upright, upright_policy
from validate_fk_vs_urdf import rpy
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria


class UprightTests(unittest.TestCase):
    def angles(self, rotations, calibration=np.eye(3)):
        from unittest.mock import patch
        matrices=[]
        for rotation in rotations:
            t=np.eye(4);t[:3,:3]=rotation;matrices.append(t)
        with patch('preview_backend.urdf_transform', side_effect=matrices):
            return tool_tilts_deg({}, ['joint_1'],np.zeros((len(rotations),1)),calibration)

    def test_yaw_is_not_tilt(self):
        np.testing.assert_allclose(self.angles([rpy(0,0,y) for y in (0,1.,3.)]),0,atol=1e-6)

    def test_cone_detects_combined_roll_pitch(self):
        angles=self.angles([rpy(np.radians(12),np.radians(12),0)])
        self.assertGreater(angles[0],15)
        with self.assertRaises(ValueError):require_upright(angles,15.)

    def test_vertical_calibration(self):
        ref=rpy(.2,-.1,.4)
        self.assertAlmostEqual(self.angles([ref],ref)[0],0.,places=5)

    def test_reject_bad_midpoint_not_just_endpoints(self):
        angles=self.angles([np.eye(3),rpy(np.radians(20),0,0),np.eye(3)])
        with self.assertRaises(ValueError):require_upright(angles,15.)

    def test_strict_limit_and_nonfinite(self):
        require_upright(np.array([0.,14.99,15.]),15.)
        for angles in ([15.01],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):require_upright(np.array(angles),15.)
        for limit in (0.,-1.,15.1,float('nan')):
            with self.assertRaises(ValueError):upright_policy({'max_upright_tilt_deg':limit})

    def test_skip_policy_default_fails_closed(self):
        self.assertEqual(upright_policy({}),(15.,True))
        self.assertEqual(upright_policy({'skip_home_move':False}),(15.,False))
        with self.assertRaises(ValueError):upright_policy({'skip_home_move':'false'})


if __name__=='__main__': unittest.main()
