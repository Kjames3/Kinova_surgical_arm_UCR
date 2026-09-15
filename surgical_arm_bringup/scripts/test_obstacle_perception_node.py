#!/usr/bin/env python3
"""Offline test for obstacle_perception_node.py -- no DDS, no camera, no robot.

Builds the real node, fills its TF buffer by hand (camera extrinsic + arm
frames along the validator's arm), feeds it the ray-cast depth image from
validate_obstacle_perception.py as a sensor_msgs/Image, and decodes what it
publishes.

  source /opt/ros/humble/setup.bash && ./test_obstacle_perception_node.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Reuse the validator's scene and scoring without running its checks.
exec(open(os.path.join(HERE, "validate_obstacle_perception.py")).read().split("# --- 1. back-projection")[0])

import rclpy                                                   # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg             # noqa: E402
from geometry_msgs.msg import TransformStamped                 # noqa: E402
from rclpy.time import Time                                    # noqa: E402
from sensor_msgs.msg import CameraInfo, Image                  # noqa: E402

import obstacle_perception_node as N                           # noqa: E402

depth, _ = render()
mm = np.round(depth * 1000).astype(np.uint16)
for big in (False, True):
    msg = Image(height=H, width=W, encoding="16UC1", is_bigendian=big)
    pad = np.zeros((H, W + 3), dtype=np.dtype(np.uint16).newbyteorder(">" if big else "<"))
    pad[:, :W] = mm
    msg.step, msg.data = (W + 3) * 2, pad.tobytes()
    assert np.array_equal(N.image_to_array(msg), mm), big
f32 = Image(height=H, width=W, encoding="32FC1", step=W * 4, data=depth.astype("<f4").tobytes())
assert np.allclose(N.image_to_array(f32), depth.astype(np.float32))
print("PASS image decode (row padding, big/little endian, 32FC1)")

c = np.array([[0.5, 0.3, TABLE_Z], [0.3, -0.35, TABLE_Z]])
r, h = np.array([0.06, 0.1]), np.array([0.12, 0.08])
ma = N.capsules_to_markers(c, r, h, "base_link", TimeMsg(sec=5), TABLE_Z)
c2, r2, h2 = N.markers_to_capsules(ma, TABLE_Z)
assert np.allclose(c, c2) and np.allclose(r, r2) and np.allclose(h, h2) and ma.markers[0].action == 3
print("PASS markers <-> capsules round trip, DELETEALL first")

rclpy.init()
node = N.ObstaclePerceptionNode()
node.cfg.update(exclude=[(CONTAINER["c"][0], CONTAINER["c"][1], 0.08)], table_z=TABLE_Z)
stamp = Time(seconds=100).to_msg()


def put(child, T, static=False):
    tr = TransformStamped()
    tr.header.frame_id, tr.child_frame_id, tr.header.stamp = "base_link", child, stamp
    tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = T[:3, 3]
    R = T[:3, :3]
    w = np.sqrt(max(0.0, 1 + R.trace())) / 2
    q = ((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w)
    (tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w) = q
    (node.tf_buffer.set_transform_static if static else node.tf_buffer.set_transform)(tr, "test")


put("global_camera_color_optical_frame", T_BC, static=True)
poly = np.vstack((ARM_A[0], ARM_B))
arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
arc /= arc[-1]
for f, si in zip(N.ARM_FRAMES[1:], np.linspace(0, 1, len(N.ARM_FRAMES))[1:]):
    T = np.eye(4)
    T[:3, 3] = [np.interp(si, arc, poly[:, k]) for k in range(3)]
    put(f, T)
node.arm_radii = np.full(len(N.ARM_FRAMES) - 1, 0.06)
node._info_cb(CameraInfo(k=K.ravel().tolist()))
got = []
node.pub = type("Sink", (), {"publish": lambda self, m: got.append(m)})()
img = Image(height=H, width=W, encoding="16UC1", step=W * 2, data=mm.astype("<u2").tobytes())
img.header.frame_id, img.header.stamp = "global_camera_color_optical_frame", stamp
node._depth_cb(img)
assert len(got) == 1, "node did not publish"
cc, rr, hh = N.markers_to_capsules(got[0], TABLE_Z)
labels = assign(cc)
assert set(labels) >= {"cylinder", "box", "bar"} and "phantom" not in labels, labels
assert max(coverage(cc, rr, hh).values()) <= 0
print(f"PASS node path: {len(rr)} capsules -> {sorted(set(labels))}, conservative, arm self-filtered via TF")

img.header.stamp = Time(seconds=500).to_msg()
node.tf_timeout = 0.0
node._depth_cb(img)
assert len(got) == 1 and node.n_skip == 1
print("PASS arm TF unavailable at the depth stamp -> frame skipped, nothing published")
node.destroy_node()
rclpy.try_shutdown()
print("ALL PASS")
