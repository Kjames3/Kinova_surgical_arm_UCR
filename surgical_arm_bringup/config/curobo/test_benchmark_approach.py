"""Numerical regression checks for offline seam handling and path sampling."""
import unittest
import numpy as np
from benchmark_approach import dense_path, normalize_continuous, interpolation_capacity
import torch
from curobo._src.util.trajectory import calculate_traj_steps


class SamplingTests(unittest.TestCase):
    def test_continuous_seam_uses_short_path_without_wrapping_revolute(self):
        raw=np.array([[3.13,3.13],[-3.13,-3.13],[-3.12,-3.12]])
        result=normalize_continuous(raw,[True,False])
        self.assertLess(abs(result[1,0]-result[0,0]),.03)
        np.testing.assert_equal(result[:,1],raw[:,1])
        np.testing.assert_allclose(np.cos(result[:,0]),np.cos(raw[:,0]))
        np.testing.assert_equal(raw[1],[-3.13,-3.13])

    def test_winding_shift_is_constant_over_path(self):
        result=normalize_continuous([[7.0],[7.1],[7.2]],[True])
        np.testing.assert_allclose(np.diff(result[:,0]),[.1,.1])
        self.assertLess(abs(result[0,0]),np.pi)

    def test_sampling_catches_interior_not_just_endpoints(self):
        result=dense_path([[0.,0.],[.2,0.],[.2,.5]],[0.,.01,.02])
        self.assertGreaterEqual(len(result),71)
        self.assertTrue(np.any(np.all(np.isclose(result,[.1,0.]),axis=1)))
        self.assertLessEqual(np.max(np.abs(np.diff(result,axis=0))),.01000001)
        np.testing.assert_allclose(result[-1],[.2,.5])

    def test_time_resolution_enforced_for_stationary_path(self):
        self.assertGreaterEqual(len(dense_path([[0.],[0.]],[0.,1.])),101)

    def test_invalid_time_and_nonfinite_rejected(self):
        for q,t in [([[0.],[1.]],[0.,0.]),([[0.],[1.]],[1.,0.]),
                    ([[0.],[float('nan')]],[0.,1.]),([[0.],[1.]],[0.,float('inf')])]:
            with self.assertRaises(ValueError):dense_path(q,t)


class TimingBufferTests(unittest.TestCase):
    def test_capacity_covers_native_float32_rounding_and_endpoint(self):
        for ceiling in [.2,.3,.5]:
            for steps,knots in [(4,20),(4,32),(8,20)]:
                capacity=interpolation_capacity(ceiling,steps,knots,.025)
                # Include the full ceiling and a sweep of shorter trajectories.
                knot_dt=torch.linspace(.002,ceiling,101,dtype=torch.float32)*steps
                _,actual=calculate_traj_steps(knot_dt,torch.tensor(.025),knots+1,nearest_int=True)
                self.assertGreaterEqual(capacity,int(actual))
        self.assertLess(interpolation_capacity(.3,4,20,.025),1100)

    def test_invalid_buffer_parameters_rejected(self):
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):interpolation_capacity(value,4,20,.025)
            with self.assertRaises(ValueError):interpolation_capacity(.3,4,20,value)


if __name__=='__main__':unittest.main()
