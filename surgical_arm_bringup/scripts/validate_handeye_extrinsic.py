#!/usr/bin/env python3
"""Independent accuracy check for an eye-to-hand base_link -> camera extrinsic.

The hand-eye solver's own residual is an AX=XB *consistency* number: it can look
excellent on poorly conditioned data.  This script measures something the solver
never sees.

The calibration board is bolted to the arm, so the board pose expressed in
`bracelet_link` is a physical constant.  Reconstruct it from each observation:

    T_bracelet_board = inv(T_base_bracelet) . T_base_camera . T_cam_board

`T_base_bracelet` comes from TF (robot kinematics), `T_cam_board` from solvePnP,
and `T_base_camera` is the extrinsic under test.  If the extrinsic is right the
result is invariant as the arm moves; the spread over poses IS the extrinsic
error, in millimetres.  A wrong extrinsic makes it swing with arm pose.

Read-only: subscribes and looks up TF.  It never commands the robot.  Move the
arm yourself (teleop / Web App), or just run this alongside another automatic
calibration sweep and let that motion feed it.

A second mode uses the flat markers already lying on the table.  Those are a
different ArUco dictionary from the arm board, so there is no id collision even
though both contain ids 0 and 1.  A marker lying on the table has a known
surface normal (straight up) and a known height, so:

  * the NORMAL check is scale-free -- a wrong --table-marker-size does not move
    it, because planar-target rotation is essentially independent of scale.  It
    tests the extrinsic's rotation on its own.
  * the HEIGHT check needs the true marker size, since size scales the
    camera-to-marker range linearly and therefore biases position along the
    viewing ray.

  python3 validate_handeye_extrinsic.py --mode invariance --extrinsic new
  python3 validate_handeye_extrinsic.py --mode table --extrinsic new
  python3 validate_handeye_extrinsic.py --mode table --table-marker-size 0.04
"""
import argparse
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
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

# Named extrinsics, xyz + quaternion xyzw, base_link -> colour optical frame.
KNOWN_EXTRINSICS = {
    # 2026-09-03 automatic sweep, ANDREFF, 15 samples.
    "new": (1.030907, 0.036634, 0.731535,
            0.639427, 0.611209, -0.327313, -0.332299),
    # Previously published in cameras.launch.py, flagged STALE in robot.launch.py.
    "old": (0.99, -0.13, 0.77, 0.6220, 0.6099, -0.3475, -0.3469),
}


def quat_to_matrix(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=float)
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


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def mean_rotation(rotations):
    """Chordal mean of rotation matrices, projected back onto SO(3)."""
    M = sum(rotations) / float(len(rotations))
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def build_board_object_points(args):
    """Board-frame 3-D corners per marker id, matching calibration_for_cameras.py."""
    ids = [int(x.strip()) for x in args.grid_marker_ids.split(",") if x.strip()]
    expected = args.grid_cols * args.grid_rows
    if len(ids) != expected:
        raise ValueError(f"--grid-marker-ids needs exactly {expected} ids")
    msize = args.marker_size
    pitch = msize + args.marker_gap
    width = args.grid_cols * msize + (args.grid_cols - 1) * args.marker_gap
    height = args.grid_rows * msize + (args.grid_rows - 1) * args.marker_gap
    table = {}
    for index, marker_id in enumerate(ids):
        row, col = divmod(index, args.grid_cols)
        cx = -width / 2.0 + msize / 2.0 + col * pitch
        cy = height / 2.0 - msize / 2.0 - row * pitch
        points = np.array([
            [cx - msize/2, cy + msize/2, 0],
            [cx + msize/2, cy + msize/2, 0],
            [cx + msize/2, cy - msize/2, 0],
            [cx - msize/2, cy - msize/2, 0],
        ], dtype=np.float32)
        table[marker_id] = np.roll(points, args.grid_corner_shift, axis=0)
    return table


class ExtrinsicValidator(Node):
    def __init__(self, args, T_base_cam):
        super().__init__("handeye_extrinsic_validator")
        self.args = args
        self.T_base_cam = T_base_cam
        self.board_points = build_board_object_points(args)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_stamp = None
        self.K = None
        self.D = None
        self.samples = []

        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, args.aruco_dict))
        self.detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters())
        table_dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, args.table_marker_dict))
        self.table_detector = cv2.aruco.ArucoDetector(
            table_dictionary, cv2.aruco.DetectorParameters())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        if args.image_transport == "compressed":
            self.create_subscription(CompressedImage, args.image_topic + "/compressed",
                                     self.compressed_cb, qos)
        else:
            self.create_subscription(Image, args.image_topic, self.raw_cb, qos)
        self.create_subscription(CameraInfo, args.camera_info_topic,
                                 self.info_cb, 10)

    # ── subscriptions ────────────────────────────────────────────────────────
    def info_cb(self, msg):
        with self.lock:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.D = np.array(msg.d, dtype=float).reshape(1, -1)

    def compressed_cb(self, msg):
        # cv_bridge is avoided deliberately: REAL-1's user Python carries a
        # NumPy/OpenCV ABI pair that has segfaulted it.
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is not None:
            with self.lock:
                self.frame = frame
                self.frame_stamp = msg.header.stamp

    def raw_cb(self, msg):
        channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(msg.encoding)
        if channels is None:
            self.get_logger().error(f"unsupported encoding '{msg.encoding}'")
            return
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels)
        if msg.encoding == "rgb8":
            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "mono8":
            frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            frame = raw.copy()
        with self.lock:
            self.frame = frame
            self.frame_stamp = msg.header.stamp

    # ── measurement ──────────────────────────────────────────────────────────
    def solve_board_in_camera(self, frame):
        with self.lock:
            K, D = self.K, self.D
        if K is None:
            return None, "waiting for CameraInfo"
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None, "no markers detected"
        object_points, image_points, visible = [], [], []
        for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
            if marker_id not in self.board_points:
                continue
            object_points.extend(self.board_points[marker_id])
            image_points.extend(marker_corners.reshape(4, 2))
            visible.append(marker_id)
        if len(visible) < self.args.min_visible_markers:
            return None, f"only {len(visible)} board markers visible"
        object_points = np.asarray(object_points, dtype=np.float32)
        image_points = np.asarray(image_points, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, "solvePnP failed"
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
        errors = projected.reshape(-1, 2) - image_points
        rms = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
        if rms > self.args.max_reprojection_error_px:
            return None, f"reprojection RMS {rms:.2f}px too high"
        R, _ = cv2.Rodrigues(rvec)
        return (homogeneous(R, tvec.flatten()), len(visible), rms), None

    def lookup_bracelet(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.robot_base_frame, self.args.robot_effector_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:
            return None, str(exc)
        t = tf.transform.translation
        q = tf.transform.rotation
        return homogeneous(quat_to_matrix([q.x, q.y, q.z, q.w]),
                           [t.x, t.y, t.z]), None

    def pose_is_new(self, T_base_bracelet):
        for sample in self.samples:
            prev = sample["T_base_bracelet"]
            dt = np.linalg.norm(prev[:3, 3] - T_base_bracelet[:3, 3])
            dr = rotation_angle_deg(prev[:3, :3].T @ T_base_bracelet[:3, :3])
            if dt < self.args.min_pose_separation_m and dr < self.args.min_pose_separation_deg:
                return False
        return True

    def try_sample(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        if frame is None:
            return "no image yet"
        T_base_bracelet, err = self.lookup_bracelet()
        if T_base_bracelet is None:
            return f"TF unavailable: {err}"
        if not self.pose_is_new(T_base_bracelet):
            return None  # arm has not moved far enough; stay quiet
        result, err = self.solve_board_in_camera(frame)
        if result is None:
            return err
        T_cam_board, visible, rms = result
        T_bracelet_board = (np.linalg.inv(T_base_bracelet)
                            @ self.T_base_cam @ T_cam_board)
        self.samples.append({
            "T_base_bracelet": T_base_bracelet,
            "T_bracelet_board": T_bracelet_board,
            "visible": visible,
            "rms": rms,
        })
        p = T_bracelet_board[:3, 3]
        self.get_logger().info(
            f"sample {len(self.samples)}: {visible} tags, reproj {rms:.2f}px, "
            f"board in bracelet = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) m")
        return None

    # ── static table-marker check ────────────────────────────────────────────
    def check_table_markers(self):
        """Where does the extrinsic put the flat markers lying on the table?"""
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            K, D = self.K, self.D
        if frame is None:
            return None, "no image yet"
        if K is None:
            return None, "waiting for CameraInfo"
        corners, ids, _ = self.table_detector.detectMarkers(frame)
        if ids is None:
            return None, "no table markers detected"
        wanted = {int(x.strip()) for x in self.args.table_marker_ids.split(",")
                  if x.strip()}
        size = self.args.table_marker_size
        obj = np.array([[-size/2,  size/2, 0], [ size/2,  size/2, 0],
                        [ size/2, -size/2, 0], [-size/2, -size/2, 0]],
                       dtype=np.float32)
        results = []
        for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
            if marker_id not in wanted:
                continue
            image_points = marker_corners.reshape(4, 2).astype(np.float32)
            # A single planar square has TWO poses that reproject almost
            # identically. Plain solvePnP silently returns one of them and is
            # routinely wrong on small, near-frontal markers -- that is the
            # known flaw in the older camera_xcheck.py. Ask for both, then use
            # the physical prior (the marker lies flat) to disambiguate, and
            # report how separable the two actually were.
            count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                obj, image_points, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not count:
                continue
            candidates = []
            for k in range(count):
                R, _ = cv2.Rodrigues(rvecs[k])
                T_base_marker = self.T_base_cam @ homogeneous(
                    R, tvecs[k].flatten())
                # Marker +Z is its outward normal. A marker lying flat points
                # straight up in base_link; the sign flips for a marker rotated
                # in plane, so score the axis, not the direction.
                normal = T_base_marker[:3, 2]
                tilt = float(np.degrees(np.arccos(np.clip(
                    abs(normal[2]) / np.linalg.norm(normal), -1.0, 1.0))))
                candidates.append({
                    "T": T_base_marker,
                    "tilt_deg": tilt,
                    "reproj": float(errors[k][0]) if errors is not None else float("nan"),
                })
            chosen = min(candidates, key=lambda c: c["tilt_deg"])
            rejected = [c for c in candidates if c is not chosen]
            results.append({
                "id": marker_id,
                "position": chosen["T"][:3, 3],
                "tilt_deg": chosen["tilt_deg"],
                "reproj": chosen["reproj"],
                "alternatives": [(c["tilt_deg"], c["reproj"]) for c in rejected],
                "ray": chosen["T"][:3, 3] - self.T_base_cam[:3, 3],
                "pixel": image_points.mean(axis=0),
            })
        if not results:
            return None, "none of the requested table marker ids were seen"
        return results, None

    def report_table(self, results):
        log = self.get_logger()
        log.info("=" * 78)
        log.info(f"STATIC TABLE-MARKER CHECK ({len(results)} marker(s)), "
                 f"assuming table surface z = {self.args.table_z:+.3f} m")
        for r in results:
            p = r["position"]
            log.info(f"  id {r['id']}: base_link = ({p[0]:+.4f}, {p[1]:+.4f}, "
                     f"{p[2]:+.4f}) m | z error {1000*(p[2]-self.args.table_z):+.1f} mm "
                     f"| normal tilt from vertical {r['tilt_deg']:.2f} deg "
                     f"| reproj {r['reproj']:.2f} px")
            for tilt, reproj in r["alternatives"]:
                log.info(f"        (rejected mirror solution: tilt {tilt:.2f} deg, "
                         f"reproj {reproj:.2f} px)")
            # tvec scales linearly with the assumed marker side, so the marker
            # slides along the viewing ray. Back-solve the side length that
            # would place it exactly on the table -- if that lands on a sane
            # printed size, the extrinsic and the table plane agree.
            ray_z = r["ray"][2]
            if abs(ray_z) > 1e-6:
                scale = (self.args.table_z - self.T_base_cam[2, 3]) / ray_z
                log.info(f"        implied marker side for z == table: "
                         f"{1000*scale*self.args.table_marker_size:.1f} mm "
                         f"(assumed {1000*self.args.table_marker_size:.1f} mm)")
        # Pairwise separation is preserved by ANY rigid transform, so it is
        # completely independent of the extrinsic under test. It depends only
        # on the assumed marker size (linearly) and the intrinsics -- which
        # makes it a clean, separate check on those two.
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a, b = results[i], results[j]
                d = float(np.linalg.norm(a["position"] - b["position"]))
                log.info(f"  centre-to-centre id {a['id']} <-> id {b['id']}: "
                         f"{1000*d:.1f} mm at the assumed "
                         f"{1000*self.args.table_marker_size:.1f} mm marker size")
                log.info("        (extrinsic-independent: checks marker size "
                         "and camera intrinsics only)")
        tilts = np.array([r["tilt_deg"] for r in results])
        log.info(f"  ROTATION verdict (scale-free): worst tilt {tilts.max():.2f} deg "
                 "-- a flat marker should read 0")
        if self.args.table_marker_size_known:
            zerr = np.array([abs(r["position"][2] - self.args.table_z)
                             for r in results])
            log.info(f"  HEIGHT verdict: worst |z error| {1000*zerr.max():.1f} mm")
        else:
            log.warn("  HEIGHT verdict SKIPPED: --table-marker-size was not "
                     "given, so the z numbers above are scaled by a guess and "
                     "mean nothing. Rerun with the measured marker side length.")
        log.info("=" * 78)

    # ── verdict ──────────────────────────────────────────────────────────────
    def report(self):
        n = len(self.samples)
        log = self.get_logger()
        log.info("=" * 78)
        if n < 2:
            log.error(f"Only {n} sample(s). The invariance test needs at least "
                      "2 distinct arm poses (ideally 6+). Move the arm while "
                      "this runs.")
            return False
        translations = np.array([s["T_bracelet_board"][:3, 3] for s in self.samples])
        rotations = [s["T_bracelet_board"][:3, :3] for s in self.samples]
        mean_t = translations.mean(axis=0)
        deviations = np.linalg.norm(translations - mean_t, axis=1)
        R_mean = mean_rotation(rotations)
        angles = np.array([rotation_angle_deg(R_mean.T @ R) for R in rotations])

        bracelet_t = np.array([s["T_base_bracelet"][:3, 3] for s in self.samples])
        span = float(np.linalg.norm(bracelet_t.max(axis=0) - bracelet_t.min(axis=0)))

        log.info(f"EXTRINSIC VALIDATION over {n} arm poses "
                 f"(bracelet travel span {span:.3f} m)")
        log.info(f"  board position in bracelet_link, mean : "
                 f"({mean_t[0]:+.4f}, {mean_t[1]:+.4f}, {mean_t[2]:+.4f}) m")
        log.info(f"  position spread : RMS {1000*np.sqrt((deviations**2).mean()):.1f} mm, "
                 f"max {1000*deviations.max():.1f} mm")
        log.info(f"  rotation spread : RMS {np.sqrt((angles**2).mean()):.2f} deg, "
                 f"max {angles.max():.2f} deg")
        log.info("  (this is true accuracy, not the solver's AX=XB residual)")
        log.info("=" * 78)
        return True


def parse_extrinsic(text):
    if text in KNOWN_EXTRINSICS:
        values = KNOWN_EXTRINSICS[text]
    else:
        values = [float(v) for v in text.replace(" ", "").split(",")]
        if len(values) != 7:
            raise ValueError("--extrinsic needs a name or x,y,z,qx,qy,qz,qw")
    return homogeneous(quat_to_matrix(values[3:7]), values[0:3])


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extrinsic", default="new",
                   help="'new', 'old', or x,y,z,qx,qy,qz,qw (base_link -> camera)")
    p.add_argument("--image-topic", default="/realsense/camera/color/image_raw")
    p.add_argument("--camera-info-topic",
                   default="/realsense/camera/color/camera_info")
    p.add_argument("--image-transport", choices=["raw", "compressed"],
                   default="compressed")
    p.add_argument("--robot-base-frame", default="base_link")
    p.add_argument("--robot-effector-frame", default="bracelet_link")
    p.add_argument("--aruco-dict", default="DICT_5X5_50")
    p.add_argument("--marker-size", type=float, default=0.0533)
    p.add_argument("--marker-gap", type=float, default=0.00505)
    p.add_argument("--grid-cols", type=int, default=4)
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-marker-ids", default="2,5,8,11,1,4,7,10,0,3,6,9")
    p.add_argument("--grid-corner-shift", type=int, default=1)
    p.add_argument("--min-visible-markers", type=int, default=4)
    p.add_argument("--max-reprojection-error-px", type=float, default=2.0)
    p.add_argument("--min-pose-separation-m", type=float, default=0.02)
    p.add_argument("--min-pose-separation-deg", type=float, default=4.0)
    p.add_argument("--mode", choices=["invariance", "table"], default="invariance",
                   help="invariance: board-on-arm, needs motion. "
                        "table: flat markers on the table, static.")
    p.add_argument("--table-marker-dict", default="DICT_4X4_50")
    p.add_argument("--table-marker-ids", default="0,1")
    p.add_argument("--table-marker-size", type=float, default=None,
                   help="measured side length in m; without it only the "
                        "scale-free normal check is meaningful")
    p.add_argument("--table-z", type=float, default=-0.03,
                   help="table surface height in base_link (default -0.03)")
    p.add_argument("--samples", type=int, default=8,
                   help="stop once this many distinct poses are collected")
    p.add_argument("--timeout", type=float, default=300.0)
    return p


def main():
    args = build_parser().parse_args()
    try:
        T_base_cam = parse_extrinsic(args.extrinsic)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.table_marker_size_known = args.table_marker_size is not None
    if args.table_marker_size is None:
        args.table_marker_size = 0.04   # placeholder; normals are scale-free

    rclpy.init()
    node = ExtrinsicValidator(args, T_base_cam)
    node.get_logger().info(
        f"Testing extrinsic '{args.extrinsic}': "
        f"t={np.round(T_base_cam[:3, 3], 4).tolist()}")
    node.get_logger().info(
        f"Move the arm through {args.samples} distinct poses "
        f"(>= {args.min_pose_separation_m*100:.0f} cm or "
        f"{args.min_pose_separation_deg:.0f} deg apart). Ctrl-C to score early.")

    if args.mode == "table":
        deadline = time.monotonic() + min(args.timeout, 30.0)
        results, note, last_note = None, None, None
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                results, note = node.check_table_markers()
                if results:
                    break
                if note and note != last_note:
                    node.get_logger().info(f"waiting: {note}")
                    last_note = note
            if results:
                node.report_table(results)
            else:
                node.get_logger().error(f"Table check failed: {note}")
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return 0 if results else 1

    deadline = time.monotonic() + args.timeout
    last_note, ok = None, False
    try:
        while rclpy.ok() and len(node.samples) < args.samples:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                node.get_logger().warn("Timed out waiting for poses.")
                break
            note = node.try_sample()
            if note and note != last_note:
                node.get_logger().info(f"waiting: {note}")
                last_note = note
            elif note is None:
                last_note = None
        ok = node.report()
    except KeyboardInterrupt:
        ok = node.report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
