#!/usr/bin/env python3
"""validate_fk_vs_urdf.py — M1: cuRobo FK vs an independent URDF FK.

Runs in the conda 'curobo' env. robot_state_publisher (what publishes the robot's
TF tree, i.e. what `tf2_echo` reports) is nothing but URDF forward kinematics via
KDL. This script computes that reference FK independently — a from-scratch numpy
chain built by parsing the URDF with the stdlib XML parser (NOT cuRobo's parser) —
and compares it against cuRobo's FK at the same joint state. Agreement validates
cuRobo's kinematics against the robot's TF without needing a live ROS/DDS session
(DDS discovery is unavailable in this sandbox).

Usage:
    source ~/activate_curobo.sh
    python validate_fk_vs_urdf.py
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "gen3_surgical.urdf")
JOINTS = [f"joint_{i}" for i in range(1, 8)]
HOME = {"joint_1": 0.0, "joint_2": -0.3049, "joint_3": -3.1416, "joint_4": -1.6607,
        "joint_5": 0.0, "joint_6": -1.7928, "joint_7": -0.0006}
TARGETS = ["assembly_tip", "bracelet_link"]


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                       # URDF fixed-axis rpy


def axis_angle(axis, a):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def T(R, t):
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def parse_joints(urdf):
    joints = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        origin = j.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split())] if origin is not None else [0, 0, 0]
        rpy_ = [float(v) for v in (origin.get("rpy", "0 0 0").split())] if origin is not None else [0, 0, 0]
        ax = j.find("axis")
        axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
        joints[j.get("name")] = dict(
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=xyz, rpy=rpy_, axis=axis, type=j.get("type"))
    return joints


def urdf_fk(joints, base, target, q):
    # walk child->parent from target up to base, then apply in order
    chain, link = [], target
    child_of = {jd["child"]: (jn, jd) for jn, jd in joints.items()}
    while link != base:
        jn, jd = child_of[link]
        chain.append((jn, jd))
        link = jd["parent"]
    chain.reverse()
    M = np.eye(4)
    for jn, jd in chain:
        M = M @ T(rpy(*jd["rpy"]), jd["xyz"])
        if jd["type"] in ("revolute", "continuous", "prismatic"):
            a = q.get(jn, 0.0)
            if jd["type"] == "prismatic":
                M = M @ T(np.eye(3), np.array(jd["axis"]) * a)
            else:
                M = M @ T(axis_angle(jd["axis"], a), [0, 0, 0])
    return M[:3, 3]


def main():
    joints = parse_joints(URDF)
    cfg = KinematicsCfg.from_basic_urdf(URDF, base_link="base_link", tool_frames=TARGETS)
    kin = Kinematics(cfg)
    jn = kin.joint_names
    q = torch.tensor([[HOME[j] for j in jn]], device="cuda", dtype=torch.float32)
    st = kin.compute_kinematics(JointState.from_position(q, joint_names=jn))

    worst = 0.0
    for name in TARGETS:
        ref = urdf_fk(joints, "base_link", name, HOME)             # independent numpy FK
        i = kin.tool_frames.index(name)
        cur = st.tool_poses.position[0, 0, i].cpu().numpy()        # cuRobo FK
        err = np.linalg.norm(ref - cur) * 1000.0
        worst = max(worst, err)
        print(f"  {name:14s} urdf=({ref[0]:.4f},{ref[1]:.4f},{ref[2]:.4f})  "
              f"cuRobo=({cur[0]:.4f},{cur[1]:.4f},{cur[2]:.4f})  |diff|={err:.4f} mm")
    print(f"=== M1 FK vs URDF (== robot_state_publisher TF): "
          f"{'PASS' if worst < 0.5 else 'FAIL'} (worst {worst:.4f} mm) ===")
    return 0 if worst < 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
