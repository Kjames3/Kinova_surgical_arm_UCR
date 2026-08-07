#!/usr/bin/env python3
"""build_robot_config.py — M1: assemble + validate the cuRobo robot config.

Runs in the conda 'curobo' env. Combines the generated collision spheres
(gen3_surgical_spheres.yml) with kinematics metadata into a cuRobo robot config
(gen3_surgical.yml), then loads it WITH collision and validates:
  * the config loads and reports the right DOF / sphere count
  * the tool-tip collision sphere coincides with the assembly_tip frame (proves
    the hand-authored tool chain is placed correctly in the kinematic tree)

Usage:
    source ~/activate_curobo.sh
    python build_robot_config.py
"""
import os

import torch
import yaml

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

HERE = os.path.dirname(os.path.abspath(__file__))
RESOLVED_URDF = os.path.join(HERE, "gen3_surgical_resolved.urdf")
SPHERES_YML = os.path.join(HERE, "gen3_surgical_spheres.yml")
OUT_YML = os.path.join(HERE, "gen3_surgical.yml")

# Kinova Gen3 7-DOF collision chain, base -> tool, in kinematic order.
CHAIN = ["base_link", "shoulder_link", "half_arm_1_link", "half_arm_2_link",
         "forearm_link", "spherical_wrist_1_link", "spherical_wrist_2_link",
         "bracelet_link"]
JOINTS = [f"joint_{i}" for i in range(1, 8)]
HOME = {"joint_1": 0.0, "joint_2": -0.3049, "joint_3": -3.1416, "joint_4": -1.6607,
        "joint_5": 0.0, "joint_6": -1.7928, "joint_7": -0.0006}


def adjacency_ignore(chain):
    """Ignore self-collision between consecutive links (they always touch)."""
    ig = {}
    for i, ln in enumerate(chain):
        nbrs = []
        if i > 0:
            nbrs.append(chain[i - 1])
        if i < len(chain) - 1:
            nbrs.append(chain[i + 1])
        ig[ln] = nbrs
    return ig


def build():
    spheres = yaml.safe_load(open(SPHERES_YML))["collision_spheres"]
    coll_links = [ln for ln in CHAIN if ln in spheres]
    kin = {
        "format_version": 2.0,
        "urdf_path": RESOLVED_URDF,
        "base_link": "base_link",
        # plan-to tool frame; must be in the dict so RobotCfg.create (IK/planner
        # path) picks it up — the tool_frames kwarg only applies to from_data_dict.
        "tool_frames": ["assembly_tip"],
        "collision_link_names": coll_links,
        "collision_spheres": spheres,
        "collision_sphere_buffer": 0.005,
        "self_collision_ignore": adjacency_ignore(coll_links),
        "self_collision_buffer": {ln: 0.0 for ln in coll_links},
        "cspace": {
            "joint_names": JOINTS,
            "max_acceleration": 15.0,
            "max_jerk": 500.0,
            "cspace_distance_weight": [1.0] * 7,
            "null_space_weight": [1.0] * 7,
            "default_joint_position": [HOME[j] for j in JOINTS],
        },
    }
    cfg = {"robot_cfg": {"kinematics": kin}}
    with open(OUT_YML, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    print(f"Wrote {OUT_YML}")
    return kin


def validate(kin_dict):
    cfg = KinematicsCfg.from_data_dict(kin_dict, tool_frames=["assembly_tip"])
    kin = Kinematics(cfg)
    jn = kin.joint_names
    q = torch.tensor([[HOME[j] for j in jn]], device="cuda", dtype=torch.float32)
    st = kin.compute_kinematics(JointState.from_position(q, joint_names=jn))

    n_spheres = int(st.robot_spheres.shape[-2])
    print(f"loaded OK | dof={len(jn)} | robot_spheres={n_spheres}")

    # world position of the assembly_tip frame. cuRobo reorders tool_frames, so
    # look it up by name rather than assuming a row index.
    ai = kin.tool_frames.index("assembly_tip")
    atip = st.tool_poses.position[0, 0, ai]                   # [3]
    sph = st.robot_spheres[0, 0]                              # [N,4] xyz,r
    d = torch.linalg.norm(sph[:, :3] - atip, dim=-1)
    j = int(torch.argmin(d))
    tip_err_mm = float(d[j]) * 1000.0
    print(f"assembly_tip world=({atip[0]:.3f},{atip[1]:.3f},{atip[2]:.3f})")
    print(f"nearest collision sphere to tip: r={float(sph[j,3])*1000:.1f} mm, "
          f"center-to-tip={tip_err_mm:.3f} mm")
    ok = tip_err_mm < 1.0
    print("=== M1 SPHERE VALIDATION:", "PASS" if ok else "FAIL",
          "(tool-tip sphere sits on assembly_tip) ===")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate(build()) else 1)
