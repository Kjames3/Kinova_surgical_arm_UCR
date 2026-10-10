#!/usr/bin/env python3
"""validate_ik.py — M1: collision-aware IK reachability on the surgical arm.

Runs in the conda 'curobo' env. Loads the full robot config (gen3_surgical.yml,
with collision spheres), generates reachable assembly_tip goals from FK of known
joint configs, and solves collision-aware IK back to them. Confirms cuRobo can
plan IK for the assembly_tip tool frame on this robot.

Usage:
    source ~/activate_curobo.sh
    python validate_ik.py
"""
import os
import json
from pathlib import Path

import numpy as np
from trimesh.transformations import quaternion_matrix
from validate_fk_vs_urdf import parse_joints, urdf_transform

import torch
import yaml

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState, GoalToolPose
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(HERE, "gen3_surgical.yml")))
JOINTS = [f"joint_{i}" for i in range(1, 8)]
HOME = [0.0, -0.3049, -3.1416, -1.6607, 0.0, -1.7928, -0.0006]


def main():
    kin_dict = CFG["robot_cfg"]["kinematics"]

    # FK to make a batch of reachable assembly_tip goals from real joint configs.
    kin = Kinematics(KinematicsCfg.from_data_dict(dict(kin_dict), tool_frames=["assembly_tip"]))
    jn = kin.joint_names
    ai = kin.tool_frames.index("assembly_tip")
    torch.manual_seed(1)
    home_by_joint = dict(zip(JOINTS, HOME))
    q0 = torch.tensor([[home_by_joint[j] for j in jn]], device="cuda", dtype=torch.float32)
    # small perturbations around home stay reachable and self-collision-free
    perturb = 0.3 * (torch.rand(3, len(jn), device="cuda") - 0.5)
    qs = torch.cat([q0, q0 + perturb], dim=0)
    n_goals = qs.shape[0]
    st = kin.compute_kinematics(JointState.from_position(qs, joint_names=jn))
    pos = st.tool_poses.position[:, 0, ai]        # [B,3]
    quat = st.tool_poses.quaternion[:, 0, ai]     # [B,4]

    ik = InverseKinematics(InverseKinematicsCfg.create(
        robot=CFG, num_seeds=30, self_collision_check=True,
        position_tolerance=0.005, orientation_tolerance=0.05,
        max_batch_size=n_goals))

    goal = GoalToolPose(tool_frames=ik.kinematics.tool_frames,
                        position=pos.unsqueeze(1).unsqueeze(1).unsqueeze(-2),   # [B,1,1,1,3]
                        quaternion=quat.unsqueeze(1).unsqueeze(1).unsqueeze(-2))
    res = ik.solve_pose(goal_tool_poses=goal)

    succ = res.success.view(-1)
    perr = res.position_error.view(-1) * 1000.0
    n = succ.numel()
    n_ok = int(succ.sum().item())
    print(f"IK goals={n}  success={n_ok}/{n}  "
          f"max_pos_err={float(perr[succ].max()) if n_ok else float('nan'):.3f} mm")
    # Independently verify returned joints, rather than trusting solver metrics.
    solved = res.js_solution.position.detach().cpu().numpy().reshape(-1, len(jn))
    joints = parse_joints(os.path.join(HERE, "gen3_surgical.urdf"))
    position_errors, angle_errors = [], []
    for index, q in enumerate(solved):
        ref = urdf_transform(joints, "base_link", "assembly_tip", dict(zip(jn, q)))
        target_position = pos[index].cpu().numpy()
        target_rotation = quaternion_matrix(quat[index].cpu().numpy())[:3, :3]
        position_errors.append(float(np.linalg.norm(ref[:3, 3] - target_position) * 1000))
        angle_errors.append(float(np.degrees(np.arccos(np.clip(
            (np.trace(ref[:3, :3].T @ target_rotation) - 1) / 2, -1, 1)))))
    ok = n_ok == n and max(position_errors) < 1.0 and max(angle_errors) < 0.5
    print(f"Independent IK verification: max {max(position_errors):.6f} mm / "
          f"{max(angle_errors):.6f} deg")
    if os.environ.get("CUROBO_OFFLINE_OUTPUT"):
        fixture = {"joint_names": jn, "frame": "base_link", "tool_frame": "assembly_tip",
                   "quaternion_order": "wxyz", "seed": 1,
                   "reference_joints": qs.cpu().tolist(), "target_positions": pos.cpu().tolist(),
                   "target_quaternions": quat.cpu().tolist(), "solution_joints": solved.tolist(),
                   "success": succ.cpu().tolist(), "position_error_mm": position_errors,
                   "orientation_error_deg": angle_errors}
        (Path(os.environ["CUROBO_OFFLINE_OUTPUT"]) / "ik_fixtures.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print("=== M1 IK REACHABILITY:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
