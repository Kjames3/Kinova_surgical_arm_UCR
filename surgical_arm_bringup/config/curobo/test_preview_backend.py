#!/usr/bin/env python3
"""Regression checks for read-only integration boundaries and scene conversion."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from preview_backend import world_from_scene, PreviewPlanner
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from curobo_planning_preview import check_preview_parameters, scene_geometry_signature, validate_upright_response


def pose(x=0.,y=0.,z=0.):
    return {'position':dict(x=x,y=y,z=z),'orientation':dict(x=0.,y=0.,z=0.,w=1.)}


def scene():
    base=dict(header={'frame_id':'world'},operation=0,pose=pose(),planes=[],subframe_names=[],
              meshes=[],mesh_poses=[],primitives=[],primitive_poses=[])
    table=dict(base,id='table',primitives=[dict(type=1,dimensions=[2.,2.,.05])],primitive_poses=[pose(z=-.055)])
    container=dict(base,id='glass_container',pose=pose(x=.2),meshes=[dict(vertices=[dict(x=0.,y=0.,z=0.),dict(x=.09,y=0.,z=0.),dict(x=0.,y=.09,z=.086)],triangles=[dict(vertex_indices=[0,1,2])])],mesh_poses=[pose(z=-.025)])
    return dict(is_diff=False,robot_state={'attached_collision_objects':[]},world=dict(collision_objects=[table,container],octomap={'octomap':{'data':[]}}))


class PreviewTests(unittest.TestCase):
    def test_motion_options_rejected(self):
        params={'transit_velocity_scaling':.15}
        check_preview_parameters(params)
        for flag in ('execute_motion','teach_mode','replay_mode','use_action_server','direct_to_angled_hover','use_current_orientation'):
            with self.subTest(flag=flag), self.assertRaises(ValueError): check_preview_parameters(dict(params,**{flag:True}))
        check_preview_parameters(dict(params,skip_home_move=False))

    def test_limits(self):
        for scale in (0.,.25,float('nan'),float('inf')):
            with self.assertRaises(ValueError): check_preview_parameters({'transit_velocity_scaling':scale})

    def test_composed_scene_pose(self):
        with tempfile.TemporaryDirectory() as d:
            world,bounds=world_from_scene(scene(),Path(d))
            np.testing.assert_allclose(bounds,[[.2,0.,-.025],[.29,.09,.061]])
            self.assertEqual(len(world['mesh']),1)
            self.assertEqual(len(world['cuboid']),1)

    def test_unsupported_scene_fails_closed(self):
        variants=[]
        s=scene();s['world']['collision_objects'][0]['primitives'][0]['type']=3;variants.append(s)
        s=scene();s['world']['collision_objects'][0]['header']['frame_id']='camera';variants.append(s)
        s=scene();s['robot_state']['attached_collision_objects']=[{}];variants.append(s)
        s=scene();s['world']['octomap']['octomap']['data']=[1];variants.append(s)
        s=scene();s['world']['collision_objects']=s['world']['collision_objects'][:1];variants.append(s)
        with tempfile.TemporaryDirectory() as d:
            for s in variants:
                with self.assertRaises(ValueError): world_from_scene(s,Path(d))

    def test_backend_rejects_execution_contract(self):
        backend=PreviewPlanner()
        for request in ({'goal_type':'ptp_pose'}, {'preview_only':True,'goal_type':'linear_pose'}):
            with self.assertRaises(ValueError): backend.plan(request)
        backend.directory.cleanup()

    def test_old_sidecar_cannot_bypass_orientation_check(self):
        for response in ({}, {'meta':{'backend':'curobo','validated':True}}):
            with self.assertRaises(ValueError):validate_upright_response(response,15.)
        meta=dict(upright_policy_version=1,approach_max_tilt_deg=1.,max_upright_tilt_deg=15.,
            stages=[dict(name='upright_approach',upright_required=True)])
        validate_upright_response({'meta':meta},15.)
        meta['approach_max_tilt_deg']=16.
        with self.assertRaises(ValueError):validate_upright_response({'meta':meta},15.)

    def test_geometry_signature(self):
        s=scene(); changed=copy.deepcopy(s)
        changed['world']['collision_objects'].reverse()
        changed['world']['collision_objects'][0]['header']['stamp']={'sec':100}
        self.assertEqual(scene_geometry_signature(s),scene_geometry_signature(changed))
        changed['world']['collision_objects'][0]['pose']['position']['x']+=.1
        self.assertNotEqual(scene_geometry_signature(s),scene_geometry_signature(changed))

    def test_provisional_tool_geometry_is_checked(self):
        import xml.etree.ElementTree as ET
        urdf=(Path(__file__).resolve().parent/'gen3_surgical.urdf').read_text()
        request=dict(preview_only=True,goal_type='ptp_pose',joint_names=[f'joint_{i}' for i in range(1,8)],
            start_joints={f'joint_{i}':0. for i in range(1,8)},vel_scale=.15,robot_description=urdf,scene={})
        backend=PreviewPlanner()
        with patch('preview_backend.world_from_scene',side_effect=RuntimeError('model verified')):
            with self.assertRaisesRegex(RuntimeError,'model verified'): backend.plan(request)
            tree=ET.fromstring(urdf)
            tree.find("./link[@name='thesis_ee']/visual/origin").set('xyz','0 0 0')
            request['robot_description']=ET.tostring(tree,encoding='unicode')
            with self.assertRaisesRegex(ValueError,'collision geometry differs'): backend.plan(request)
        backend.directory.cleanup()

    def test_second_request_under_c_locale(self):
        import locale
        urdf=(Path(__file__).resolve().parent/'gen3_surgical.urdf').read_text(encoding='utf-8')
        request=dict(preview_only=True,goal_type='ptp_pose',joint_names=[f'joint_{i}' for i in range(1,8)],
            start_joints={f'joint_{i}':0. for i in range(1,8)},vel_scale=.15,
            robot_description=urdf+'<!-- Unicode robot description: — -->',scene={})
        backend=PreviewPlanner(); previous=locale.setlocale(locale.LC_CTYPE)
        try:
            locale.setlocale(locale.LC_CTYPE,'C')
            with patch('preview_backend.world_from_scene',side_effect=RuntimeError('model verified')):
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError,'model verified'): backend.plan(request)
        finally:
            locale.setlocale(locale.LC_CTYPE,previous); backend.directory.cleanup()

    def test_preview_dispatch_precedes_legacy_flow(self):
        source=(Path(__file__).resolve().parents[2]/'scripts'/'insertion.py').read_text()
        tree=ast.parse(source)
        run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_run_impl')
        self.assertIsInstance(run.body[0],ast.If)
        self.assertTrue(any(isinstance(n,ast.Return) for n in ast.walk(run.body[0])))
        preview=(Path(__file__).resolve().parents[2]/'scripts'/'curobo_planning_preview.py').read_text()
        for forbidden in ('send_goal','_execute_fjt','_execute_moveit','_recovery_return','_run_phases'):
            self.assertNotIn(forbidden,preview)


if __name__=='__main__': unittest.main()
