#!/usr/bin/env python3
"""Offline tests for source CAD orientation and acknowledged scene application."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import setup_planning_scene as setup
from container_geometry import load_container_geometry, validate_scene_spec, container_top_z
from moveit_msgs.msg import PlanningScene, CollisionObject

DESCRIPTION = Path(__file__).resolve().parents[4] / 'src/ros2_kortex/kortex_description'
MESH = DESCRIPTION / 'grippers/thesis_ee/meshes/Glass_container.STL'


class SceneTests(unittest.TestCase):
    def spec(self):
        _, _, meta = load_container_geometry(MESH)
        return dict(version=1, frame_id='world', container_yaw_rad=0., table_size_m=[2.,2.,.05], table_surface_z_m=-.03,
                    container_center_base_m=[.294,.129,-.03], container_mesh_sha256=meta['source_sha256'])

    def harness(self):
        node=SimpleNamespace(_published=False,_pending=False,_scene_spec=self.spec(),
                             _apply_cli=Mock(),_get_cli=Mock(),get_logger=Mock(),_applied=Mock(),_verified=Mock())
        node.get_parameter=lambda key:SimpleNamespace(value=True if key.endswith('_enabled') else 0.0)
        return node

    def test_insertion_rim_height_includes_board(self):
        self.assertAlmostEqual(container_top_z(-.03,.086,.005),.061)
        self.assertAlmostEqual(container_top_z(-.03,.086,0.),.056)
        self.assertAlmostEqual(container_top_z(-.03,.086,.005)-.030,.031)
        with self.assertRaises(ValueError):container_top_z(-.03,.086,-.005)

    def test_cad_has_upright_height_and_centred_origin(self):
        vertices,faces,_=load_container_geometry(MESH)
        for k,limits in enumerate([(-.045,.045),(-.045,.045),(0.,.086)]):
            self.assertAlmostEqual(min(v[k] for v in vertices),limits[0],places=6)
            self.assertAlmostEqual(max(v[k] for v in vertices),limits[1],places=6)
        self.assertEqual(len(faces),396)

    def test_invalid_scene_rejected(self):
        for key,value in [('frame_id','camera'),('table_size_m',[0.,2.,.05]),('container_center_base_m',[0.,0.,-.04])]:
            spec=copy.deepcopy(self.spec());spec[key]=value
            with self.assertRaises(ValueError):validate_scene_spec(spec)

    def test_applies_marker_centred_hollow_mesh_without_clearing_robot_state(self):
        node=self.harness()
        with patch.object(setup,'get_package_share_directory',return_value=str(DESCRIPTION)):
            setup.PlanningSceneSetup._publish_scene(node)
        scene=node._apply_cli.call_async.call_args.args[0].scene
        self.assertTrue(scene.is_diff);self.assertTrue(scene.robot_state.is_diff)
        self.assertEqual([o.id for o in scene.world.collision_objects],['table','glass_container'])
        table,container=scene.world.collision_objects
        self.assertAlmostEqual(table.primitive_poses[0].position.z+table.primitives[0].dimensions[2]/2,-.03)
        self.assertAlmostEqual(container.mesh_poses[0].position.x,.294)
        self.assertAlmostEqual(container.mesh_poses[0].position.y,.129)
        self.assertEqual(len(container.meshes[0].triangles),396)
        self.assertFalse(node._published)

    def test_board_thickness_raises_container_without_moving_table(self):
        node=self.harness();node._scene_spec['container_center_base_m'][2]=-.025
        with patch.object(setup,'get_package_share_directory',return_value=str(DESCRIPTION)):
            setup.PlanningSceneSetup._publish_scene(node)
        table,container=node._apply_cli.call_async.call_args.args[0].scene.world.collision_objects
        self.assertAlmostEqual(table.primitive_poses[0].position.z+table.primitives[0].dimensions[2]/2,-.03)
        self.assertAlmostEqual(container.mesh_poses[0].position.z,-.025)

    def test_wrong_cad_hash_cannot_silently_publish_partial_scene(self):
        node=self.harness();node._scene_spec['container_mesh_sha256']='wrong'
        with patch.object(setup,'get_package_share_directory',return_value=str(DESCRIPTION)):
            setup.PlanningSceneSetup._publish_scene(node)
        node._apply_cli.call_async.assert_not_called()

    def test_missing_readback_objects_are_not_success(self):
        node=self.harness();node._expected_ids={'table','glass_container'}
        future=Mock();future.result.return_value=SimpleNamespace(scene=PlanningScene())
        setup.PlanningSceneSetup._verified(node,future)
        self.assertFalse(node._published)

    def test_confirmed_readback_is_success(self):
        node=self.harness();node._expected_ids={'table','glass_container'}
        scene=PlanningScene(world=setup.PlanningScene().world)
        scene.world.collision_objects=[CollisionObject(id='table'),CollisionObject(id='glass_container')]
        future=Mock();future.result.return_value=SimpleNamespace(scene=scene)
        setup.PlanningSceneSetup._verified(node,future)
        self.assertTrue(node._published)


if __name__=='__main__':unittest.main()
