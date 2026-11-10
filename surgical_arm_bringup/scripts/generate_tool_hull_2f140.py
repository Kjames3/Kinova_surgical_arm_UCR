#!/usr/bin/env python3
"""Generate the Robotiq 2F-140 table-guard hull, in the bracelet_link frame.

The table check (trajectory_table_check.py) and the impedance table barrier
both guard a tool as the vertices of its convex hull: a plane's closest
approach to a convex body is a hull vertex, so that is exact. Unlike the
thesis_ee, the 2F-140 has moving fingers. This hull is the convex hull of the
gripper's collision geometry over the WHOLE finger range (finger_joint 0 ->
0.7 rad, mimic joints included), so it is valid at any opening, including
while the gripper is moving.

Input is the gripper as kortex_description builds it. Generate the URDF with:

    xacro src/ros2_kortex/kortex_description/robots/gen3.xacro dof:=7 \\
        gripper:=robotiq_2f_140 > /tmp/gen3_2f140.urdf

then run:

    python3 generate_tool_hull_2f140.py /tmp/gen3_2f140.urdf \\
        ../config/2f140_hull_bracelet.csv

The gripper is placed by the URDF (end_effector_link 61.5 mm below
bracelet_link, robotiq_base_link rotated 90 deg about z). The package's
robotiq_2f_140_coupling.stl is NOT referenced by the URDF and so is not in
this hull. If a physical coupling plate sits between the flange and the
gripper, the real fingers are lower than this hull; measure and add an offset.
"""
import argparse
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np

BASE = "bracelet_link"
FINGER = "finger_joint"
FINGER_RANGE = (0.0, 0.7)


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    k = np.asarray(axis, float); k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def tf(xyz, rpy):
    T = np.eye(4); T[:3, :3] = rpy_matrix(*rpy); T[:3, 3] = xyz
    return T


def read_stl(path):
    data = Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
        return np.array(v, float)
    n = (len(data) - 84) // 50
    if struct.unpack("<I", data[80:84])[0] != n:
        raise ValueError(f"{path}: facet count does not match file size")
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype(
        [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


def parse(urdf_text):
    from urdf_parser_py.urdf import URDF
    robot = URDF.from_xml_string(urdf_text)
    return robot


def vec(x, default):
    return default if x is None else list(x)


def link_transforms(robot, finger):
    """bracelet_link -> every descendant link, at finger_joint = `finger`."""
    children = {}
    for j in robot.joints:
        children.setdefault(j.parent, []).append(j)
    out = {BASE: np.eye(4)}
    stack = [BASE]
    while stack:
        parent = stack.pop()
        for j in children.get(parent, []):
            o = j.origin
            T = out[parent] @ tf(vec(o.xyz if o else None, [0, 0, 0]),
                                 vec(o.rpy if o else None, [0, 0, 0]))
            if j.type in ("revolute", "continuous"):
                if j.name == FINGER:
                    a = finger
                elif j.mimic is not None and j.mimic.joint == FINGER:
                    a = (j.mimic.multiplier or 1.0) * finger + (j.mimic.offset or 0.0)
                else:
                    raise SystemExit(f"unexpected free joint {j.name} under {BASE}")
                R = np.eye(4); R[:3, :3] = axis_angle(vec(j.axis, [1, 0, 0]), a)
                T = T @ R
            elif j.type != "fixed":
                raise SystemExit(f"unsupported joint type {j.type} ({j.name})")
            out[j.child] = T
            stack.append(j.child)
    return out


def collision_points(robot, links):
    """{link: (k,3) points in the link frame} from collision meshes and boxes."""
    pts = {}
    for link in robot.links:
        if link.name not in links:
            continue
        for c in (link.collisions or []):
            g = c.geometry
            o = c.origin
            T = tf(vec(o.xyz if o else None, [0, 0, 0]), vec(o.rpy if o else None, [0, 0, 0]))
            if hasattr(g, "filename"):
                fn = g.filename.replace("file://", "")
                if fn.startswith("package://"):
                    raise SystemExit(f"unresolved package:// path {fn}; run xacro with the workspace sourced")
                P = read_stl(fn) * np.asarray(vec(g.scale, [1, 1, 1]))
            elif hasattr(g, "size"):
                s = np.asarray(g.size) / 2
                P = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
            else:
                continue
            P = P @ T[:3, :3].T + T[:3, 3]
            pts.setdefault(link.name, []).append(P)
    return {k: np.vstack(v) for k, v in pts.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf")
    ap.add_argument("out_csv")
    ap.add_argument("--samples", type=int, default=29, help="finger positions over 0..0.7 rad")
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings("ignore")
    from scipy.spatial import ConvexHull

    robot = parse(Path(a.urdf).read_text())
    tool_links = set(link_transforms(robot, 0.0)) - {BASE}
    geom = collision_points(robot, tool_links)
    if not geom:
        raise SystemExit(f"no collision geometry below {BASE}")
    cloud = []
    for f in np.linspace(*FINGER_RANGE, a.samples):
        T = link_transforms(robot, f)
        for name, P in geom.items():
            cloud.append(P @ T[name][:3, :3].T + T[name][:3, 3])
    cloud = np.vstack(cloud)
    hull = cloud[ConvexHull(cloud).vertices]
    hull = hull[np.argsort(hull[:, 2])]                      # lowest (in bracelet z) first
    lo, hi = hull.min(0), hull.max(0)
    header = (
        "# Robotiq 2F-140 convex hull, bracelet_link frame (m). Table-guard points.\n"
        f"# Generated by generate_tool_hull_2f140.py from {Path(a.urdf).name}: collision meshes + finger-pad\n"
        f"# boxes of {sorted(geom)}, swept over {FINGER} {FINGER_RANGE[0]}..{FINGER_RANGE[1]} rad "
        f"({a.samples} samples, mimics applied).\n"
        f"# Bounds x[{lo[0]:.4f},{hi[0]:.4f}] y[{lo[1]:.4f},{hi[1]:.4f}] z[{lo[2]:.4f},{hi[2]:.4f}].\n"
        "# NOT included: robotiq_2f_140_coupling.stl (not referenced by the URDF). Regenerate if remounted.\n"
        "# x,y,z\n")
    Path(a.out_csv).write_text(header + "".join(f"{x:.5f},{y:.5f},{z:.5f}\n" for x, y, z in hull))
    print(f"{len(hull)} hull vertices from {len(cloud)} points; z range {lo[2]:.4f}..{hi[2]:.4f} m "
          f"(reach below bracelet {-lo[2] * 1000:.0f} mm) -> {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
