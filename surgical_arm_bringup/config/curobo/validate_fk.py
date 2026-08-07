#!/usr/bin/env python3
"""validate_fk.py — M1 check that cuRobo loads the surgical arm and FK is correct.

Runs in the conda 'curobo' env (source ~/activate_curobo.sh). Loads the generated
gen3_surgical.urdf, tracks both assembly_tip and bracelet_link, and verifies the
fixed link between them is honored: ||assembly_tip - bracelet_link|| must equal
|offset| = sqrt(0.027^2 + 0.414^2) = 414.8795 mm at EVERY joint configuration.
This is rotation-invariant, so it validates FK without needing the live robot.

Usage:
    source ~/activate_curobo.sh
    python validate_fk.py
"""
import math
import os

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen3_surgical.urdf")

# assembly_tip fixed-joint origin in bracelet_link, from thesis_ee_macro.xacro
TIP_OFFSET = (-0.027, 0.000, -0.414)
EXPECTED_MM = math.sqrt(sum(v * v for v in TIP_OFFSET)) * 1000.0

HOME = {"joint_1": 0.0, "joint_2": -0.3049, "joint_3": -3.1416, "joint_4": -1.6607,
        "joint_5": 0.0, "joint_6": -1.7928, "joint_7": -0.0006}


def main():
    cfg = KinematicsCfg.from_basic_urdf(
        URDF, base_link="base_link", tool_frames=["assembly_tip", "bracelet_link"])
    kin = Kinematics(cfg)
    jn = kin.joint_names
    # NOTE: cuRobo REORDERS tool_frames by internal link index, so tool_poses
    # rows follow kin.tool_frames (the reordered list), NOT the input order.
    # Always look up the row by name — never assume row 0 is your first input.
    ai = kin.tool_frames.index("assembly_tip")
    bi = kin.tool_frames.index("bracelet_link")
    print(f"cuRobo loaded chain | dof={len(jn)} | joints={jn}")
    print(f"tool_poses row order (cuRobo-sorted): {kin.tool_frames}")
    print(f"expected fixed offset = {EXPECTED_MM:.4f} mm")

    def dist_at(qdict, label):
        q = torch.tensor([[qdict[j] for j in jn]], device="cuda", dtype=torch.float32)
        st = kin.compute_kinematics(JointState.from_position(q, joint_names=jn))
        p = st.tool_poses.position                 # [B, H, L, 3]
        atip, brac = p[0, 0, ai], p[0, 0, bi]
        d = float(torch.linalg.norm(atip - brac))
        print(f"  {label:9s} |atip-bracelet|={d*1000:.4f} mm  "
              f"atip_world=({atip[0]:.3f},{atip[1]:.3f},{atip[2]:.3f})")
        return d

    configs = [(HOME, "home"), ({j: 0.0 for j in jn}, "zero")]
    for k in range(3):
        configs.append(({j: float(torch.empty(1).uniform_(-2.0, 2.0)) for j in jn},
                        f"random{k}"))

    err_mm = max(abs(dist_at(q, lbl) * 1000 - EXPECTED_MM) for q, lbl in configs)
    print(f"=== max FK offset error: {err_mm:.5f} mm ===")
    ok = err_mm < 0.05
    print("=== M1 FK VALIDATION:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
