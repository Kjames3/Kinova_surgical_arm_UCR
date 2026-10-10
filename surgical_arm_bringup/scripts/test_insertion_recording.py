#!/usr/bin/env python3
"""Offline recorder tests; never construct a ROS node or connect to hardware."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import insertion
from moveit_msgs.msg import PlanningScene, CollisionObject


class RecordingTests(unittest.TestCase):
    def harness(self):
        node = SimpleNamespace(
            _scene_snapshot_cli=Mock(), _scene_snapshot_pub=Mock(),
            _wait_for_future=Mock(), _run_info=Mock(), get_logger=Mock())
        node._scene_snapshot_cli.wait_for_service.return_value = True
        return node

    def test_full_scene_preserves_geometry_and_settings(self):
        node = self.harness()
        scene = PlanningScene()
        scene.world.collision_objects = [CollisionObject(id='table'), CollisionObject(id='container')]
        node._wait_for_future.return_value = SimpleNamespace(scene=scene)
        insertion.AngledInserter._record_scene_snapshot(node)
        node._scene_snapshot_pub.publish.assert_called_once_with(scene)
        request = node._scene_snapshot_cli.call_async.call_args.args[0]
        self.assertEqual(request.components.components, 1023)
        node._run_info.assert_called_once_with('planning_scene_snapshot', success=True,
                                              object_ids=['table', 'container'])

    def test_unavailable_service_reports_missing_snapshot(self):
        node = self.harness()
        node._scene_snapshot_cli.wait_for_service.return_value = False
        insertion.AngledInserter._record_scene_snapshot(node)
        node._scene_snapshot_pub.publish.assert_not_called()
        self.assertFalse(node._run_info.call_args.kwargs['success'])

    def test_timeout_cancels_request(self):
        node = self.harness()
        node._wait_for_future.return_value = None
        insertion.AngledInserter._record_scene_snapshot(node)
        node._scene_snapshot_cli.call_async.return_value.cancel.assert_called_once()
        node._scene_snapshot_pub.publish.assert_not_called()
        self.assertFalse(node._run_info.call_args.kwargs['success'])

    def test_diff_is_not_mislabelled_full_scene(self):
        node = self.harness()
        node._wait_for_future.return_value = SimpleNamespace(scene=PlanningScene(is_diff=True))
        insertion.AngledInserter._record_scene_snapshot(node)
        node._scene_snapshot_pub.publish.assert_not_called()
        self.assertFalse(node._run_info.call_args.kwargs['success'])

    def test_recorder_qos_topics_and_snapshot_order(self):
        with tempfile.TemporaryDirectory() as directory:
            node = self.harness()
            node._BAG_TOPICS = insertion.AngledInserter._BAG_TOPICS
            node._bag_qos_overrides = insertion.AngledInserter._bag_qos_overrides
            node._record_scene_snapshot = Mock()
            node.get_parameter = lambda name: SimpleNamespace(value={
                'bag_dir': directory, 'bag_extra_topics': ['/joint_states', '/extra/compressed']}[name])
            with patch.object(insertion.subprocess, 'Popen') as popen, patch.object(insertion.time, 'sleep'):
                popen.return_value.poll.return_value = None
                proc = insertion.AngledInserter._start_bag(node)
            self.assertIs(proc, popen.return_value)
            command = popen.call_args.args[0]
            self.assertEqual(command.count('/joint_states'), 1)
            for topic in ['/planning_scene', '/monitored_planning_scene', '/robot_description',
                          '/robot_description_semantic', '/insertion/planning_scene_snapshot']:
                self.assertIn(topic, command)
            qos = json.loads(Path(command[command.index('--qos-profile-overrides-path')+1]).read_text())
            self.assertEqual(qos['/tf_static']['durability'], 'transient_local')
            self.assertEqual(qos['/robot_description']['durability'], 'transient_local')
            self.assertNotIn('/joint_states', qos)
            self.assertFalse(any(topic.endswith('/image_raw') for topic in command))
            node._record_scene_snapshot.assert_called_once()

    def test_failed_recorder_does_not_request_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            node = self.harness()
            node._BAG_TOPICS = insertion.AngledInserter._BAG_TOPICS
            node._bag_qos_overrides = insertion.AngledInserter._bag_qos_overrides
            node._record_scene_snapshot = Mock()
            node.get_parameter = lambda name: SimpleNamespace(value=directory if name == 'bag_dir' else [])
            with patch.object(insertion.subprocess, 'Popen') as popen, patch.object(insertion.time, 'sleep'):
                popen.return_value.poll.return_value = 1
                self.assertIsNone(insertion.AngledInserter._start_bag(node))
            node._record_scene_snapshot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
