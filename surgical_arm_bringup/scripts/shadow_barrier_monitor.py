#!/usr/bin/env python3
"""Shadow-mode readout of the obstacle barrier against a real AprilTag.

READ-ONLY BY CONSTRUCTION. This node creates no publishers, no action clients
and no service clients, and it never opens a Kortex low-level session. It
subscribes and it looks up TF. It cannot move the arm.

It answers the question that offline validation cannot: does an obstacle
derived from the CAMERA land in the same base_link the ROBOT uses? A sign or
frame error there is invisible to `validate_obstacle_barrier.py` -- that file
proves the algebra, and would keep proving it while the obstacle sat mirrored
across the workspace.

Three readouts per tick:

  FRAME    the tag's position from the camera, the tool's position from
           forward kinematics, and the distance between them. Jog the tool to
           touch the physical tag: this should read ~0. It is a direct test of
           the hand-eye extrinsic in the robot's own frame, and it does not
           involve the barrier at all.

  CLEARANCE  per-guarded-link capsule clearance with the obstacle placed at
           the tag, so you can watch it shrink as you approach.

  RESPONSE  how hard the barrier would push back. Takes a probe acceleration
           that drives the closest link straight at the obstacle, runs the real
           QP, and reports how much of it survives. Nothing is commanded --
           this is what the filter WOULD have done.

Run on REAL-1 with robot.launch.py and the RealSense already up:

  source /opt/ros/humble/setup.bash
  source ~/ros2_kortex_ws/install/setup.bash
  python3 shadow_barrier_monitor.py --duration 120
"""
import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from control_barrier import (compute_obstacle_hocbf_rows,   # noqa: E402
                             filter_control_qp)


def quat_to_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


class ShadowMonitor(Node):
    def __init__(self, args, model, segs, link_names):
        super().__init__("shadow_barrier_monitor")
        self.args = args
        self.model = model
        self.segs = segs
        self.link_names = link_names
        self.lock = threading.Lock()
        self.frame = None
        self.K = None
        self.D = None
        self.joint_state = None

        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.tag_dict))
        self.detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage,
                                 args.image_topic + "/compressed",
                                 self._image_cb, qos)
        self.create_subscription(CameraInfo, args.camera_info_topic,
                                 self._info_cb, 10)
        self.create_subscription(JointState, args.joint_state_topic,
                                 self._joint_cb, qos)

    # ── inputs ──────────────────────────────────────────────────────────────
    def _image_cb(self, msg):
        # cv_bridge avoided: REAL-1's user Python has a NumPy/OpenCV ABI pair
        # that has segfaulted it.
        f = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if f is not None:
            with self.lock:
                self.frame = f

    def _info_cb(self, msg):
        with self.lock:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.D = np.array(msg.d, dtype=float).reshape(1, -1)

    def _joint_cb(self, msg):
        with self.lock:
            self.joint_state = msg

    # ── geometry ────────────────────────────────────────────────────────────
    def base_from_camera(self):
        """base_link <- camera optical, from TF, or from --extrinsic."""
        if self.args.extrinsic:
            v = [float(x) for x in self.args.extrinsic.split(",")]
            return homogeneous(quat_to_matrix(v[3:7]), v[0:3]), "cli"
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.base_frame, self.args.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception:
            return None, None
        t, q = tf.transform.translation, tf.transform.rotation
        return homogeneous(quat_to_matrix([q.x, q.y, q.z, q.w]),
                           [t.x, t.y, t.z]), "tf"

    def tag_in_base(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            K, D = self.K, self.D
        if frame is None or K is None:
            return None, "waiting for camera"
        T_bc, src = self.base_from_camera()
        if T_bc is None:
            return None, (f"no TF {self.args.base_frame} <- "
                          f"{self.args.camera_frame} (pass --extrinsic)")
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None or self.args.tag_id not in ids.flatten().tolist():
            return None, f"tag {self.args.tag_id} not visible"
        i = ids.flatten().tolist().index(self.args.tag_id)
        s = self.args.tag_size
        obj = np.array([[-s/2, s/2, 0], [s/2, s/2, 0],
                        [s/2, -s/2, 0], [-s/2, -s/2, 0]], dtype=np.float32)
        img = corners[i].reshape(4, 2).astype(np.float32)
        cnt, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not cnt:
            return None, "solvePnP failed"
        # Disambiguate the two planar solutions with the lies-flat prior, the
        # same way validate_handeye_extrinsic.py does. Plain solvePnP returns
        # the mirror solution often enough to matter.
        best, best_tilt = None, None
        for k in range(cnt):
            R, _ = cv2.Rodrigues(rvecs[k])
            T = T_bc @ homogeneous(R, tvecs[k].flatten())
            nz = T[:3, 2]
            tilt = np.degrees(np.arccos(np.clip(abs(nz[2]) / np.linalg.norm(nz),
                                                -1, 1)))
            if best_tilt is None or tilt < best_tilt:
                best, best_tilt = T, tilt
        return (best[:3, 3], float(best_tilt), src), None

    def tool_in_base(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.base_frame, self.args.tool_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as exc:
            return None, str(exc)
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z]), None

    def q_dq(self):
        with self.lock:
            js = self.joint_state
        if js is None:
            return None, None
        by_name = dict(zip(js.name, js.position))
        vel = dict(zip(js.name, js.velocity)) if js.velocity else {}
        names = [f"joint_{i}" for i in range(1, self.model.n + 1)]
        if not all(nm in by_name for nm in names):
            return None, None
        q = np.array([by_name[nm] for nm in names])
        dq = np.array([vel.get(nm, 0.0) for nm in names])
        return q, dq

    # ── the readout ─────────────────────────────────────────────────────────
    def tick(self):
        tag, note = self.tag_in_base()
        tool, tool_err = self.tool_in_base()
        q, dq = self.q_dq()

        if tag is None:
            print(f"  waiting: {note}")
            return
        tag_p, tilt, src = tag
        line = (f"FRAME    tag {tag_p[0]:+.4f} {tag_p[1]:+.4f} {tag_p[2]:+.4f} "
                f"(tilt {tilt:4.1f} deg, extrinsic from {src})")
        if tool is None:
            line += f" | tool TF unavailable: {tool_err}"
            print(line)
            return
        d_tool = np.linalg.norm(tool - tag_p)
        print(line)
        print(f"         tool {tool[0]:+.4f} {tool[1]:+.4f} {tool[2]:+.4f}"
              f"  ->  tool-to-tag {d_tool*1000:8.1f} mm")

        if q is None:
            print("         (no joint state yet; clearance readout skipped)")
            return

        # The capsule stands on the tag, rising --obstacle-height.
        center = np.array([[tag_p[0], tag_p[1], tag_p[2]]])
        radius = np.array([self.args.obstacle_radius])
        height = np.array([self.args.obstacle_height])
        p, J_v, dJdq_v = self.model.point_barrier_terms(q, dq, self.segs)
        A, b, info = compute_obstacle_hocbf_rows(
            p, J_v, dJdq_v, dq, center, radius, height, self.args.link_radius,
            alpha1=self.args.alpha1, alpha2=self.args.alpha2)
        h = info["h"][:, 0]
        closest = int(np.argmin(h))
        print("         clearance mm: " + "  ".join(
            f"{nm.replace('gen3_','')[:12]}={hv*1000:+.0f}"
            for nm, hv in zip(self.link_names, h)))
        print(f"CLEARANCE closest link {self.link_names[closest]} "
              f"at {h[closest]*1000:+.1f} mm"
              + (f"   [{info['degenerate']} DEGENERATE PAIR(S)]"
                 if info["degenerate"] else ""))

        # RESPONSE: a probe acceleration driving the closest link straight at
        # the obstacle. Never commanded -- this only reports what the filter
        # would do with it.
        n_hat = -(-A[closest])           # row is -Jh, so Jh = -A
        Jh = -A[closest]
        denom = Jh @ Jh
        if denom < 1e-12:
            print("RESPONSE  gradient degenerate; no meaningful probe")
            return
        ddq_probe = -Jh * (self.args.probe_accel / denom)
        approach_before = -(Jh @ ddq_probe)
        ddq_f, status = filter_control_qp(ddq_probe, A, b)
        approach_after = -(Jh @ ddq_f)
        print(f"RESPONSE  probe drives closest link at "
              f"{approach_before:+.2f} m/s^2; filter allows "
              f"{approach_after:+.2f} m/s^2  [{status}]")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-topic", default="/realsense/camera/color/image_raw")
    p.add_argument("--camera-info-topic",
                   default="/realsense/camera/color/camera_info")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--camera-frame", default="global_camera_color_optical_frame")
    p.add_argument("--tool-frame", default="assembly_tip")
    p.add_argument("--extrinsic", default=None,
                   help="x,y,z,qx,qy,qz,qw to use instead of a TF lookup")
    p.add_argument("--tag-dict", default="DICT_APRILTAG_36h11")
    p.add_argument("--tag-id", type=int, default=1)
    p.add_argument("--tag-size", type=float, default=0.1437)
    p.add_argument("--obstacle-radius", type=float, default=0.08)
    p.add_argument("--obstacle-height", type=float, default=0.15)
    p.add_argument("--link-radius", type=float, default=0.055)
    p.add_argument("--alpha1", type=float, default=10.0)
    p.add_argument("--alpha2", type=float, default=10.0)
    p.add_argument("--probe-accel", type=float, default=1.0,
                   help="m/s^2 of simulated approach used for the RESPONSE line")
    p.add_argument("--rate", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=60.0)
    return p


def main():
    args = build_parser().parse_args()

    import impedance as imp
    model = imp.KinDynModel(imp.DEFAULT_URDF, imp.DEFAULT_BASE_LINK,
                            imp.DEFAULT_TIP_LINK)
    link_names = [nm for nm in imp.DEFAULT_TABLE_LINKS
                  if nm in model.segment_names()]
    segs = [model.segment_index(nm) for nm in link_names]

    rclpy.init()
    node = ShadowMonitor(args, model, segs, link_names)

    # State the safety property rather than assuming the reader trusts it.
    pubs = [t for t in node.get_publisher_names_and_types_by_node(
        node.get_name(), "/") if not t[0].endswith(("/rosout", "/parameter_events"))]
    print("READ-ONLY: no command publishers, no action clients, no Kortex "
          f"session. (publishers: {[t[0] for t in pubs] or 'none'})")
    print(f"guarding {len(link_names)} links: "
          + ", ".join(nm.replace("gen3_", "") for nm in link_names))
    print(f"obstacle: capsule r={args.obstacle_radius*1000:.0f} mm, "
          f"h={args.obstacle_height*1000:.0f} mm, standing on tag "
          f"{args.tag_id}\n")

    deadline = time.monotonic() + args.duration
    period = 1.0 / max(args.rate, 0.1)
    next_tick = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now >= next_tick:
                next_tick = now + period
                print(f"[t+{args.duration - (deadline - now):6.1f}s]")
                node.tick()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
