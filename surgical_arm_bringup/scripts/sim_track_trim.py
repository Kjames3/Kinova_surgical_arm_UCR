#!/usr/bin/env python3
"""
MuJoCo bench for the cartesian gravity trim in 'track' mode (--trim-error).

WHY THIS EXISTS
---------------
On hardware the trim integrated (q_ref - q) with q_ref stuck at q0 for the whole
track run, so the commanded sweep itself read as droop: joint 2's trim sat at
its 15.6 Nm cap for ~50% of every trim-on track run (2026-08-05/06), and the
no-trim runs scored better. This bench reproduces that offline and scores the
fix (--trim-error task) against it, with no robot.

It runs impedance.py's OWN pieces -- cartesian_impedance_torque, track_setpoint,
trim_error, compute_gravity, the gain constants -- in the same order as the real
loop. Re-anchoring, the barriers, self-collision and the singularity escape are
left out: none of them is active in a well-conditioned track run.

PLANT != MODEL, ON PURPOSE
--------------------------
A trim only has work to do when the gravity model is wrong, so the plant gets a
payload on end_effector_link that the controller's KDL chain does not have
(--payload-mass / --payload-com). Gravity is 'hybrid' as on hardware: exact at
q0, wrong elsewhere -- which is precisely the residual a trim is for.

USAGE
-----
  python3 sim_track_trim.py --compare          # off / joint / task, verdict
  python3 sim_track_trim.py --trim-error joint --duration 60
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_impedance_mujoco as bench  # noqa: E402  (stubs kortex_api for us)
import impedance as imp  # noqa: E402

mujoco = bench.mujoco


def load_plant_with_payload(args, timestep):
    """Pruned base->tip plant plus an unmodelled payload on the last link."""
    joint_names = bench.build_plant_urdf(args.urdf, args.base_link,
                                         args.tip_link)
    tree = ET.parse(bench.PLANT_URDF)
    link = next(lk for lk in tree.getroot().iter("link")
                if lk.get("name") == args.tip_link)
    if args.payload_mass > 0.0:
        old = link.find("inertial")
        if old is not None:
            link.remove(old)
        inert = ET.SubElement(link, "inertial")
        ET.SubElement(inert, "origin",
                      xyz=" ".join(str(v) for v in args.payload_com), rpy="0 0 0")
        ET.SubElement(inert, "mass", value=str(args.payload_mass))
        # Small solid-sphere-ish tensor; gravity only sees mass and CoM.
        i = 0.4 * args.payload_mass * 0.04 ** 2
        ET.SubElement(inert, "inertia", ixx=str(i), iyy=str(i), izz=str(i),
                      ixy="0", ixz="0", iyz="0")
    tree.write(bench.PLANT_URDF)

    m = mujoco.MjModel.from_xml_path(bench.PLANT_URDF)
    m.opt.timestep = timestep
    m.dof_armature[:] = args.armature
    d = mujoco.MjData(m)
    dofadr, qposadr = [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        dofadr.append(int(m.jnt_dofadr[jid]))
        qposadr.append(int(m.jnt_qposadr[jid]))
    return m, d, np.array(dofadr), np.array(qposadr)


def plant_gravity(m, d, q, qposadr, dofadr):
    """True holding torque of the plant at q (what tau_g0 measures on hardware)."""
    d.qpos[qposadr] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return np.array(d.qfrc_bias[dofadr], dtype=float)


def simulate(args, trim_error, track_q_ref=True, vel_ff=True):
    kin = imp.KinDynModel(args.urdf, args.base_link, args.tip_link)
    m, d, dofadr, qposadr = load_plant_with_payload(
        args, 1.0 / (args.rate * args.substeps))
    n = len(qposadr)
    q0 = np.array(args.q0 if args.q0 else bench.HOME_Q0, dtype=float)

    # Hybrid gravity, exactly as the real loop: anchored to the measured
    # holding torque at q0, model supplies the variation.
    tau_g0 = plant_gravity(m, d, q0, qposadr, dofadr)
    g_offset = tau_g0 - kin.gravity(q0)[:n]
    d.qpos[qposadr] = q0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    p_a, quat_des = kin.fk(q0)
    p_b = p_a + np.array(args.track_offset)

    kp_cart = np.concatenate([imp.KP_CART_POS, imp.KP_CART_ORI]) * args.cart_scale
    kd_cart = imp.critical_damping(kp_cart, imp.CART_DAMPING_RATIO)
    kp_null = imp.KP_NULL[:n] * args.null_stiffness
    kd_null = imp.critical_damping(imp.KP_NULL[:n]) * args.kd_scale
    kd_global = imp.KD_GLOBAL[:n] * args.kd_scale
    kp_anchor = args.wrist_anchor * imp.CONTINUOUS_JOINTS[:n]
    tau_max = imp.TAU_MAX[:n]
    i_cap = imp.I_CLAMP_FRAC * tau_max
    k_trim = imp.KI_JOINT[:n] * args.gravity_trim
    acc_max = (imp.MAX_IMP_ACC_LIN, imp.MAX_IMP_ACC_ANG)
    wrench_max = (imp.MAX_EE_FORCE, imp.MAX_EE_TORQUE)
    tip = np.array(args.tip_offset)

    dt = 1.0 / args.rate
    q_ref = q0.copy()
    tau_trim = np.zeros(n)
    rows = []
    for k in range(int(args.duration * args.rate)):
        t = k * dt
        q = np.array(d.qpos[qposadr], dtype=float)
        dq = np.array(d.qvel[dofadr], dtype=float)
        if not np.all(np.isfinite(q)):
            return None

        tau_g = kin.gravity(q)[:n] + g_offset
        p_cur, quat_cur = kin.fk(q)
        J = kin.jacobian(q)
        p_des = imp.track_setpoint(p_a, p_b, t, args.track_period)
        if track_q_ref:
            q_ref = imp.follow_q_ref(kin, q_ref, p_des, quat_des)
        M = kin.mass(q)
        tau, _, _, _ = imp.cartesian_impedance_torque(
            kin, q, dq, p_des, quat_des, q_ref, kp_cart, kd_cart, kp_null,
            kd_null, tau_g, np.zeros(n), imp.K_LIMIT,
            state=(p_cur, quat_cur, J), dyn=(M, np.linalg.inv(M)),
            acc_max=acc_max, wrench_max=wrench_max,
            xdot_des=(imp.track_velocity(p_a, p_b, t, args.track_period)
                      if vel_ff else None))

        if trim_error != "off" and np.max(np.abs(dq)) < args.trim_freeze_thresh:
            task = ((J, p_cur, quat_cur, p_des, quat_des)
                    if trim_error == "task" else None)
            tau_trim = np.clip(
                tau_trim + k_trim * imp.trim_error(trim_error, q, q_ref, task) * dt,
                -i_cap, i_cap)
        tau = tau + tau_trim
        tau = tau - kd_global * dq
        tau = tau + kp_anchor * imp._wrap_rad(q_ref - q)
        ramp = min(1.0, t / args.ramp_time) if args.ramp_time > 0 else 1.0
        tau = tau_g + ramp * (tau - tau_g)

        d.qfrc_applied[dofadr] = np.clip(tau, -tau_max, tau_max)
        for _ in range(args.substeps):
            mujoco.mj_step(m, d)

        # Tool-tip error: flange pose error plus the lever of the orientation
        # error, i.e. where the tip IS versus where the tip SHOULD be.
        R_cur = quat_to_R(quat_cur)
        R_des = quat_to_R(quat_des)
        tip_err = np.linalg.norm((p_cur + R_cur @ tip) - (p_des + R_des @ tip))
        ori = np.rad2deg(np.linalg.norm(imp.orientation_error(quat_des, quat_cur)))
        rows.append((t, tip_err, ori, np.abs(tau_trim) >= 0.98 * i_cap,
                     tau_trim.copy()))
    return rows


def quat_to_R(qt):
    x, y, z, w = qt
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def score(rows, settle):
    t = np.array([r[0] for r in rows])
    keep = t >= settle
    err = np.array([r[1] for r in rows])[keep] * 1e3
    ori = np.array([r[2] for r in rows])[keep]
    pinned = np.array([r[3] for r in rows])[keep]
    trim = np.array([r[4] for r in rows])
    return dict(mean=err.mean(), p95=np.percentile(err, 95), max=err.max(),
                ori95=np.percentile(ori, 95),
                pin_frac=pinned.any(axis=1).mean(),
                trim_max=np.abs(trim).max(axis=0))


def report(label, s):
    print(f"  {label:6s} tip mean {s['mean']:6.1f} mm  p95 {s['p95']:6.1f}  "
          f"max {s['max']:6.1f}  ori p95 {s['ori95']:5.2f} deg  "
          f"trim pinned {100 * s['pin_frac']:4.0f}%  "
          f"|trim|max {np.round(s['trim_max'], 1)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--trim-error", choices=["off", "joint", "task"],
                   default="task")
    p.add_argument("--compare", action="store_true",
                   help="Run legacy, off, joint and task and print a verdict.")
    p.add_argument("--no-track-vel-ff", dest="track_vel_ff",
                   action="store_false",
                   help="Damp absolute EE velocity (the pre-fix behaviour).")
    p.add_argument("--no-track-q-ref", dest="track_q_ref",
                   action="store_false",
                   help="Leave q_ref at q0 (the pre-fix behaviour).")
    p.add_argument("--duration", type=float, default=36.0)
    p.add_argument("--settle", type=float, default=12.0, metavar="S",
                   help="Score from this time on (skips engage + trim learning).")
    p.add_argument("--track-offset", type=float, nargs=3,
                   default=[0.0, 0.10, 0.0])
    p.add_argument("--track-period", type=float, default=6.0)
    p.add_argument("--payload-mass", type=float, default=0.6,
                   help="Unmodelled plant payload (kg); 0 = plant == model.")
    p.add_argument("--payload-com", type=float, nargs=3,
                   default=[0.0, 0.0, 0.10],
                   help="Payload CoM in the tip-link frame (m).")
    p.add_argument("--tip-offset", type=float, nargs=3,
                   default=[0.0, 0.0, 0.26],
                   help="Tool tip in the tip-link frame (m); 2F-140 fingertip.")
    p.add_argument("--gravity-trim", type=float, default=imp.GRAVITY_TRIM_SCALE)
    p.add_argument("--trim-freeze-thresh", type=float,
                   default=imp.TRIM_FREEZE_THRESH)
    p.add_argument("--cart-scale", type=float, default=2.0)
    p.add_argument("--kd-scale", type=float, default=1.0)
    p.add_argument("--null-stiffness", type=float, default=0.4)
    p.add_argument("--wrist-anchor", type=float, default=3.0)
    p.add_argument("--ramp-time", type=float, default=1.0)
    p.add_argument("--rate", type=float, default=1000.0)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--armature", type=float, default=0.01)
    p.add_argument("--q0", type=float, nargs=7, default=None)
    p.add_argument("--urdf", default=imp.DEFAULT_URDF)
    p.add_argument("--base-link", default=imp.DEFAULT_BASE_LINK)
    p.add_argument("--tip-link", default=imp.DEFAULT_TIP_LINK)
    args = p.parse_args()

    print(f"track {args.track_offset} m / {args.track_period}s, unmodelled "
          f"payload {args.payload_mass} kg, scored t >= {args.settle}s")
    # (label, trim, q_ref follows the setpoint). 'legacy' is the hardware
    # behaviour through 2026-10-23: joint trim against a q_ref stuck at q0.
    runs = ([("legacy", "joint", False, False), ("off", "off", True, True),
             ("joint", "joint", True, True), ("task", "task", True, True),
             ("no-ff", "task", True, False)]
            if args.compare else
            [(args.trim_error, args.trim_error, args.track_q_ref,
              args.track_vel_ff)])
    res = {}
    for label, mode, follow, ff in runs:
        rows = simulate(args, mode, follow, ff)
        if rows is None:
            print(f"  {label:6s} DIVERGED")
            continue
        res[label] = score(rows, args.settle)
        report(label, res[label])

    if args.compare and len(res) == 5:
        best = min(("off", "joint", "task"), key=lambda k: res[k]["p95"])
        ok = (res["task"]["p95"] < res["legacy"]["p95"]
              and res["task"]["pin_frac"] == 0.0)
        print(f"best p95: {best}.  "
              + ("PASS: task trim + moving q_ref beats legacy, never pins."
                 if ok else "FAIL: no improvement over legacy."))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
