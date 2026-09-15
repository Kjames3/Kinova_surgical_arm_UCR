#!/usr/bin/env python3
"""ROS 2 wrapper for obstacle_perception.py: RealSense depth -> capsule obstacles.

READ-ONLY with respect to the arm: subscribes to depth, camera_info and TF,
publishes markers. It commands nothing.

Output: visualization_msgs/MarkerArray on /barrier/obstacles, one CYLINDER
marker per capsule in base_link, stamped with the DEPTH FRAME's time:

    pose.position  = (cx, cy, table_z + height / 2)
    scale.x = scale.y = 2 * radius,  scale.z = height

`markers_to_capsules` below turns that back into (centers, radii, heights)
for control_barrier. RViz draws the same message, so what you see is exactly
what the barrier would get. The array starts with a DELETEALL marker so
obstacles that disappear are cleared.

A frame is SKIPPED (nothing published) when the arm's TF at the depth stamp is
unavailable: publishing obstacles without the self-filter would put the arm
into its own barrier. What the controller does when obstacles go stale is
barrier item 4.

Run on REAL-1 with robot.launch.py and cameras.launch.py up:

  source /opt/ros/humble/setup.bash && source ~/ros2_kortex_ws/install/setup.bash
  python3 obstacle_perception_node.py --ros-args -p exclude:="[0.45, -0.05, 0.08]"
"""
import os
import sys
import time

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obstacle_perception import (DEFAULT_CFG, depth_image_to_meters,   # noqa: E402
                                 perceive, pose_matrix)

# Segments run between consecutive frame origins. Radii are the cuRobo
# collision-sphere maxima per link (gen3_surgical_spheres.yml) rounded up;
# self_margin is added on top.
ARM_FRAMES = ["base_link", "shoulder_link", "half_arm_1_link", "half_arm_2_link",
              "forearm_link", "spherical_wrist_1_link", "spherical_wrist_2_link",
              "bracelet_link", "end_effector_link", "assembly_tip"]
ARM_RADII = [0.065, 0.045, 0.060, 0.060, 0.050, 0.050, 0.050, 0.060, 0.040]


def image_to_array(msg):
    """sensor_msgs/Image (16UC1 or 32FC1) -> (H, W) array. No cv_bridge on REAL-1."""
    dtype = {"16UC1": np.uint16, "mono16": np.uint16, "32FC1": np.float32}.get(msg.encoding)
    if dtype is None:
        raise ValueError(f"unsupported depth encoding {msg.encoding!r}")
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    per_row = msg.step // dt.itemsize                      # rows may be padded
    arr = np.frombuffer(bytes(msg.data), dtype=dt).reshape(msg.height, per_row)[:, :msg.width]
    return arr.astype(dtype)


def capsules_to_markers(centers, radii, heights, frame, stamp, table_z):
    out = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    out.markers.append(clear)
    for i, (c, r, h) in enumerate(zip(centers, radii, heights)):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = "capsule", i, Marker.CYLINDER, Marker.ADD
        m.pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(table_z + h / 2.0))
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = float(2 * r)
        m.scale.z = float(max(h, 1e-3))
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.35, 0.1, 0.5
        out.markers.append(m)
    return out


def markers_to_capsules(msg, table_z):
    """Inverse of capsules_to_markers -> (centers (k, 3), radii (k,), heights (k,))."""
    ms = [m for m in msg.markers if m.action == Marker.ADD and m.ns == "capsule"]
    centers = np.array([(m.pose.position.x, m.pose.position.y, table_z) for m in ms]).reshape(-1, 3)
    radii = np.array([m.scale.x / 2.0 for m in ms])
    heights = np.array([m.pose.position.z * 2.0 - 2.0 * table_z for m in ms])
    return centers, radii, heights


class ObstaclePerceptionNode(Node):
    def __init__(self):
        super().__init__("obstacle_perception")
        p = self.declare_parameter
        self.depth_topic = p("depth_topic", "/global_camera/global_camera/aligned_depth_to_color/image_raw").value
        info_topic = p("info_topic", "/global_camera/global_camera/aligned_depth_to_color/camera_info").value
        self.base_frame = p("base_frame", "base_link").value
        self.arm_frames = list(p("arm_frames", ARM_FRAMES).value)
        self.arm_radii = np.array(p("arm_radii", ARM_RADII).value, dtype=float)
        if len(self.arm_radii) != len(self.arm_frames) - 1:
            raise ValueError("arm_radii needs one entry per segment (len(arm_frames) - 1)")
        self.cfg = dict(DEFAULT_CFG)
        for key in ("stride", "min_range", "max_range", "table_z", "min_height", "max_height",
                    "self_margin", "cell", "min_points", "inflate", "max_radius", "max_extrude"):
            self.cfg[key] = p(key, DEFAULT_CFG[key]).value
        self.cfg["x_range"] = tuple(p("x_range", list(DEFAULT_CFG["x_range"])).value)
        self.cfg["y_range"] = tuple(p("y_range", list(DEFAULT_CFG["y_range"])).value)
        ex = list(p("exclude", [0.0]).value)                 # flat [x, y, r, x, y, r, ...]
        self.cfg["exclude"] = [tuple(ex[i:i + 3]) for i in range(0, len(ex) - len(ex) % 3, 3)]
        self.tf_timeout = p("tf_timeout", 0.05).value

        self.K = None
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(MarkerArray, "/barrier/obstacles", 10)
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, info_topic, self._info_cb, 10)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, qos)
        self.n_ok = self.n_skip = 0
        self.last_log = time.monotonic()
        self.cost_ms = 0.0
        self.get_logger().info(f"depth={self.depth_topic} exclude={self.cfg['exclude']}")

    def _info_cb(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def _lookup(self, target, source, stamp):
        tf = self.tf_buffer.lookup_transform(
            target, source, stamp, timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
        t, q = tf.transform.translation, tf.transform.rotation
        return pose_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def _depth_cb(self, msg):
        if self.K is None:
            return
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        try:
            T_bc = self._lookup(self.base_frame, msg.header.frame_id, stamp)
            origins = np.array([self._lookup(self.base_frame, f, stamp)[:3, 3]
                                for f in self.arm_frames])
        except Exception as exc:                          # no arm pose at this stamp -> skip
            self.n_skip += 1
            self._maybe_log(f"skip ({type(exc).__name__})")
            return
        t0 = time.perf_counter()
        depth = depth_image_to_meters(image_to_array(msg), msg.encoding)
        centers, radii, heights, stats = perceive(
            depth, self.K, T_bc, origins[:-1], origins[1:], self.arm_radii, self.cfg)
        self.cost_ms = (time.perf_counter() - t0) * 1e3
        self.pub.publish(capsules_to_markers(centers, radii, heights, self.base_frame,
                                             msg.header.stamp, self.cfg["table_z"]))
        self.n_ok += 1
        self._maybe_log(f"{stats['capsules']} capsules, {stats['clusters']} clusters, "
                        f"valid {stats['valid_fraction']:.0%}, self-removed {stats['self_removed']}")

    def _maybe_log(self, text):
        now = time.monotonic()
        if now - self.last_log >= 1.0:
            self.get_logger().info(f"{text} | published {self.n_ok}, skipped {self.n_skip}, "
                                   f"{self.cost_ms:.1f} ms/frame")
            self.last_log = now


def main():
    rclpy.init()
    node = ObstaclePerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
