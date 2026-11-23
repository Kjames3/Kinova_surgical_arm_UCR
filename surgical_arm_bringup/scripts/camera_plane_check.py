#!/usr/bin/env python3
"""Offline, marker-size-independent check of the front camera's marker positions.

combine_cameras.py places each ArUco marker by single-marker PnP, whose depth
scales with the assumed marker_size. This script instead intersects the camera
ray through each marker's pixel centre with the known board plane (z_plane in
world) and compares the two, per marker and as a shift of the fused centre.

Inputs come only from the bag: /marker_observations (corners_px, pos_ref),
camera_info for K/D, and /tf_static + /tf for world -> optical frame.

  source /opt/ros/humble/setup.bash
  python3 camera_plane_check.py --z-plane -0.025 ~/robot-logs/insertion_bags/insertion_20261013_*/
"""
import argparse
import glob
import json
import os
import sqlite3

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def _quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _tf_graph(c, T):
    """child -> (parent, R, t) from the first appearance of each static/dynamic edge."""
    edges = {}
    for topic in ("/tf_static", "/tf"):
        if topic not in T:
            continue
        for (d,) in c.execute("select data from messages where topic_id=?", (T[topic],)):
            for tr in deserialize_message(d, TFMessage).transforms:
                if tr.child_frame_id in edges:
                    continue
                r, p = tr.transform.rotation, tr.transform.translation
                edges[tr.child_frame_id] = (tr.header.frame_id, _quat_R((r.x, r.y, r.z, r.w)),
                                            np.array([p.x, p.y, p.z]))
            if topic == "/tf" and len(edges) > 60:
                break
    return edges


def _pose_in(edges, frame, world):
    R, t = np.eye(3), np.zeros(3)
    while frame != world:
        if frame not in edges:
            raise KeyError(f"no tf path from {frame} to {world}")
        parent, Rp, tp = edges[frame]
        R, t = Rp @ R, Rp @ t + tp
        frame = parent
    return R, t


def analyse(bag, z_plane, world):
    db = sorted(glob.glob(os.path.join(bag, "*.db3")))[0]
    c = sqlite3.connect(db)
    T = {n: i for i, n in c.execute("select id,name from topics")}
    obs = [json.loads(deserialize_message(d, String).data)
           for (d,) in c.execute("select data from messages where topic_id=?", (T["/marker_observations"],))]
    obs = [o for o in obs if o.get("camera") == "realsense" and o.get("markers")]
    if not obs:
        return None
    info_topic = next(n for n in T if n.endswith("camera_info"))
    ci = deserialize_message(c.execute("select data from messages where topic_id=? limit 1",
                                       (T[info_topic],)).fetchone()[0], CameraInfo)
    K = np.array(ci.k).reshape(3, 3)
    R, t = _pose_in(_tf_graph(c, T), obs[0]["optical_frame"], world)

    per = {}
    for o in obs:
        for m in o["markers"]:
            px = np.mean(np.array(m["corners_px"]), axis=0)  # undistorted enough at D435 (D~0)
            ray = R @ np.linalg.solve(K, np.array([px[0], px[1], 1.0]))
            s = (z_plane - t[2]) / ray[2]
            hit = t + s * ray
            per.setdefault(m["id"], []).append((np.array(m["pos_ref"]), hit))
    out = {}
    for mid, pairs in sorted(per.items()):
        pnp = np.median([p for p, _ in pairs], axis=0)
        ray = np.median([h for _, h in pairs], axis=0)
        out[mid] = (pnp, ray, len(pairs))
    return out, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--z-plane", type=float, default=-0.025, help="marker surface height in world (m)")
    ap.add_argument("--world", default="world")
    a = ap.parse_args()
    for b in a.bags:
        if not os.path.isdir(b):
            continue
        try:
            r = analyse(b, a.z_plane, a.world)
        except Exception as exc:
            print(f"{os.path.basename(b.rstrip('/'))}: skipped ({exc})")
            continue
        if r is None:
            continue
        markers, cam = r
        name = os.path.basename(b.rstrip("/"))
        shifts = []
        for mid, (pnp, ray, n) in markers.items():
            d = (ray - pnp) * 1000.0
            dist_pnp = np.linalg.norm(pnp - cam)
            dist_ray = np.linalg.norm(ray - cam)
            shifts.append(d[:2])
            print(f"{name} id{mid} n={n:5d} PnP z={pnp[2]:+.4f}  ray-plane - PnP = "
                  f"({d[0]:+6.1f}, {d[1]:+6.1f}) mm  range ratio ray/PnP = {dist_ray / dist_pnp:.4f}")
        m = np.mean(shifts, axis=0)
        print(f"{name} mean XY shift of visible markers: ({m[0]:+.1f}, {m[1]:+.1f}) mm")


if __name__ == "__main__":
    main()
