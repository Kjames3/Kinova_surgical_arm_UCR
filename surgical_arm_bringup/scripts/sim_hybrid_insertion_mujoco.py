#!/usr/bin/env python3
"""
MuJoCo bench for the hybrid insertion controller (hybrid_insertion.py).

Runs the 45-degree needle insertion end to end -- HybridInsertionController
under impedance.TableSafetyFilter with the tool floor moved to the depth stop
(container_geometry.insertion_tool_floor) -- against a MuJoCo arm that has
CONTACT with a real-sized container. Nothing here is a re-implementation of
the robot's code; only the plant and the scenario are simulated.

SCENE
-----
  * table at z=-0.030, the 5 mm marker board, the container on it: 86 mm
    tall, 42 mm inner radius, built as a ring of 24 boxes plus a floor disc
    (a tube mesh collides as its convex hull, which fills the bore).
  * the tool as two capsules at the MEASURED assembly_tip: a 32 mm needle
    (2 mm dia) and, behind it, the 8 mm-radius mounting fixture. The MJCF's
    own `assembly_probe` is pre-2026-10-09 geometry and is switched off.
  * contacts are tool <-> container/table only. Arm links cannot collide, so
    a run cannot pass by leaning on something.

The target tip is the container centre at --depth-mm below the rim; the
needle approaches along a 45-deg axis from 20 mm above the rim.

SCENARIOS (all on the 30%-gravity-error plant unless stated)
  nominal       clean insertion: tip error at depth vs the 3 mm / 2 deg gate
  model-error   + rotor inertia (the mass model 14% off)
  blocked       container floor raised so the needle hits it 15 mm early:
                must abort on axial force and back out
  overshoot     controller told to go 12 mm past the target: the barrier's
                depth stop must hold it short of wall-contact depth

  python3 sim_hybrid_insertion_mujoco.py              # all scenarios
  python3 sim_hybrid_insertion_mujoco.py --scenario nominal --depth-mm 36
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco                                              # noqa: E402

import container_geometry as cg                            # noqa: E402
import impedance as imp                                    # noqa: E402
from hybrid_insertion import HybridInsertionController     # noqa: E402

DEFAULT_MJCF = os.path.expanduser("~/mujoco_models/gen3_thesis_ee.xml")
TIP = np.array([0.108, -0.008, -0.411])          # measured assembly_tip
TABLE_Z = -0.030
BOARD = 0.005
CONT_H = 0.086
CONT_R_IN = 0.042
CONT_WALL = 0.004
CONT_FLOOR = 0.003
NEEDLE_LEN = 0.032
NEEDLE_R = 0.001
FIXTURE_R = 0.008
FIXTURE_LEN = 0.12
ENTRY_ABOVE_RIM = 0.020
GATE_POS, GATE_DEG = 0.003, 2.0

LINK_Z_MIN, TOOL_Z_MIN = 0.08, -0.02
WALL = dict(standoff=0.06, k_wall=1200.0, d_wall=60.0, f_max=90.0)
TOOL_WALL = dict(standoff=0.005, k_wall=6000.0, d_wall=60.0, f_max=90.0)


def build_plant(path, centre, floor_raise=0.0, armature=0.0, tool=None):
    """MJCF + container + tool capsules, torque driven, contacts tool-only.

    `tool` = (mass, com, iquat, diaginertia) for the bracelet body, from
    tool_inertia(). It MUST be set here, before compile: MuJoCo builds each
    body's collision bounding volumes in its inertial frame, so editing
    body_ipos/iquat on the compiled model leaves them stale and the broad
    phase silently drops the tool's contacts (found 2026-10-27: the needle
    passed 16 mm into the container floor with ncon == 0).
    """
    sp = mujoco.MjSpec.from_file(path)
    # No arm self-/env-contact; keep visuals.
    for g in sp.geoms:
        g.contype = 0
        g.conaffinity = 0
    br = sp.body("bracelet_link")
    if tool is not None:
        br.mass, br.ipos, br.iquat, br.inertia = tool
        br.explicitinertial = True
    axis = TIP / np.linalg.norm(TIP)
    back_n = TIP - axis * NEEDLE_LEN
    back_f = TIP - axis * (NEEDLE_LEN + FIXTURE_LEN)
    for name, r, a, b in (("needle", NEEDLE_R, TIP, back_n),
                          ("fixture", FIXTURE_R, back_n, back_f)):
        g = br.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [r, 0, 0]
        g.fromto = np.concatenate((a, b))
        g.contype, g.conaffinity = 1, 0
        g.rgba = [0.9, 0.2, 0.2, 1]
    wb = sp.worldbody
    base_z = TABLE_Z + BOARD
    rim_z = base_z + CONT_H
    tbl = wb.add_geom()
    tbl.type = mujoco.mjtGeom.mjGEOM_PLANE
    tbl.size = [2, 2, 0.01]
    tbl.pos = [0, 0, TABLE_Z]
    tbl.contype, tbl.conaffinity = 0, 1
    nseg = 24
    seg_w = 2 * (CONT_R_IN + CONT_WALL) * np.tan(np.pi / nseg) + 0.002
    for k in range(nseg):
        th = 2 * np.pi * k / nseg
        g = wb.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        rr = CONT_R_IN + CONT_WALL / 2
        g.pos = [centre[0] + rr * np.cos(th), centre[1] + rr * np.sin(th),
                 base_z + CONT_H / 2]
        g.quat = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
        g.size = [CONT_WALL / 2, seg_w / 2, CONT_H / 2]
        g.contype, g.conaffinity = 0, 1
        g.rgba = [0.3, 0.5, 0.9, 0.5]
    fl = wb.add_geom()
    fl.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    ft = CONT_FLOOR + floor_raise
    fl.size = [CONT_R_IN + CONT_WALL, ft / 2, 0]
    fl.pos = [centre[0], centre[1], base_z + ft / 2]
    fl.contype, fl.conaffinity = 0, 1
    m = sp.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    m.opt.timestep = 1e-4
    m.dof_damping[:] = 0.0
    m.dof_armature[:] = armature
    return m, rim_z


def tool_inertia(model):
    """Plant tool inertia = what the controller believes (as in the barrier
    bench): vision bracelet + SolidWorks thesis_ee, in the bracelet frame.
    Returns (mass, com, iquat, diaginertia) for build_plant()."""
    ch = model.chain
    names = model.segment_names()
    sb, se = names.index("gen3_bracelet_link"), names.index("gen3_end_effector_link")
    I = ch.getSegment(sb).getInertia() + ch.getSegment(se).pose(0.0) * \
        ch.getSegment(se).getInertia()
    mass = I.getMass()
    com = np.array([I.getCOG()[i] for i in range(3)])
    Io = I.getRotationalInertia()
    Io = np.array([[Io[3 * r + c] for c in range(3)] for r in range(3)])
    Ic = Io - mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    w, V = np.linalg.eigh(Ic)
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, V.flatten())
    return mass, com, quat, w


def entry_pose(model, tseg, entry_z, seed=11):
    """A 45-deg pose with the tip at `entry_z`, inside the joint stops and the
    gravity reserve, refined so the needle axis is exactly 45 deg."""
    r = np.random.default_rng(seed)
    axis = TIP / np.linalg.norm(TIP)
    lim = 0.6 * imp.TAU_MAX[imp.RESERVE_JOINTS]
    for _ in range(400000):
        q = r.uniform(-2.2, 2.2, model.n)
        p, R, _ = model.point_frame_jacobian(q, tseg, TIP)
        if abs(p[2] - entry_z) > 0.01 or np.hypot(p[0], p[1]) < 0.35:
            continue
        tilt = np.degrees(np.arccos(np.clip(-(R @ axis)[2], -1, 1)))
        if abs(tilt - 45) > 6:
            continue
        if np.any(np.abs(q) > imp.Q_HARD_MAX[:model.n] - 0.15):
            continue
        if np.any(np.abs(model.gravity(q)[imp.RESERVE_JOINTS]) > 0.85 * lim):
            continue
        z, _, _ = model.height_barrier_terms(q, np.zeros(model.n), [
            model.segment_index(nm) for nm in imp.DEFAULT_TABLE_LINKS])
        if np.min(z - 0.055) < LINK_Z_MIN + 0.05:
            continue
        # Refine: tip height exact, axis exactly 45 deg (keep azimuth).
        for _ in range(200):
            p, R, J = model.point_frame_jacobian(q, tseg, TIP)
            a = R @ axis
            az = np.arctan2(a[1], a[0])
            a_des = np.array([np.cos(az) / np.sqrt(2), np.sin(az) / np.sqrt(2),
                              -1 / np.sqrt(2)])
            e = np.concatenate(([entry_z - p[2]], np.cross(a, a_des)))
            Je = np.vstack((J[2:3], J[3:]))
            q = q + np.linalg.lstsq(Je, e, rcond=None)[0] * 0.5
            if np.linalg.norm(e) < 1e-7:
                break
        return q
    raise RuntimeError("no entry pose")


def run(args, scenario):
    model = imp.KinDynModel(imp.DEFAULT_URDF, imp.DEFAULT_BASE_LINK,
                            imp.DEFAULT_TIP_LINK, tool="thesis_ee")
    tseg = model.segment_index(imp.TOOL_HULL_LINK)
    rim_z = TABLE_Z + BOARD + CONT_H
    depth_v = args.depth_mm / 1000.0
    q0 = entry_pose(model, tseg, rim_z + ENTRY_ABOVE_RIM)
    p0, R0, _ = model.point_frame_jacobian(q0, tseg, TIP)
    a = R0 @ (TIP / np.linalg.norm(TIP))
    along = (ENTRY_ABOVE_RIM + depth_v) / (-a[2])        # axial travel
    target = p0 + along * a
    centre = target[:2]

    armature = 0.1 if scenario == "model-error" else 0.0
    floor_raise = 0.0
    cmd_along = along
    if scenario == "blocked":
        # Floor top 15 mm (vertical) above where the tip would end.
        floor_raise = (target[2] + 0.015) - (TABLE_Z + BOARD + CONT_FLOOR)
    if scenario == "overshoot":
        cmd_along = along + 0.012 / (-a[2])
    m, _ = build_plant(args.mjcf, centre, floor_raise, armature,
                       tool=tool_inertia(model))
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)

    g_scale = np.ones(model.n)
    g_scale[3] = 1.0 - 0.3
    filt = imp.TableSafetyFilter(
        model, [model.segment_index(nm) for nm in imp.DEFAULT_TABLE_LINKS],
        0.055, LINK_Z_MIN, (15.0, 15.0), imp.TAU_MAX[:model.n], WALL,
        tool_hull=imp.load_tool_hull("thesis_ee"), tool_k=4,
        tool_z_min=TOOL_Z_MIN, observer_gain=50.0, gravity_reserve=0.4)
    # Entering the insertion phase: depth-stop floor + thin tool wall.
    floor = cg.insertion_tool_floor(rim_z, depth_v)
    assert filt.set_tool_z_min(floor, q0, tool_wall=TOOL_WALL), \
        "floor switch refused at entry"
    ctl = HybridInsertionController(model, tseg, TIP, q0, cmd_along,
                                    speed=args.speed)

    nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "needle")
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fixture")
    worst_wall_f = 0.0
    contact_needle_f = 0.0
    first_wall = None
    lowest = np.inf
    log = []
    T = cmd_along / args.speed + 4.0
    for k in range(int(T * 1000)):
        q, dq = d.qpos.copy(), d.qvel.copy()
        tau_g = model.gravity(q) * g_scale
        tau, info = ctl.torque(q, dq, tau_g, 1e-3, filt.tau_ext)
        tau, finfo = filt.apply(q, dq, tau, tau_g, 1e-3)
        tau = np.clip(tau, -imp.TAU_MAX[:model.n], imp.TAU_MAX[:model.n])
        filt.sent(tau)
        for _ in range(10):
            d.qfrc_applied[:] = tau
            mujoco.mj_step(m, d)
        # contact forces on the tool
        for c in range(d.ncon):
            con = d.contact[c]
            gs = {con.geom1, con.geom2}
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, f6)
            fn = abs(f6[0])
            if fid in gs:
                worst_wall_f = max(worst_wall_f, fn)
                if first_wall is None and fn > 0.5:
                    p_t, _, _ = model.point_frame_jacobian(d.qpos, tseg, TIP)
                    first_wall = (k / 1000, info["phase"], info["s"],
                                  info["lat_err"], info["rot_err_deg"],
                                  p_t[2] - rim_z)
            if nid in gs:
                contact_needle_f = max(contact_needle_f, fn)
        lowest = min(lowest, float(np.min(model.body_point_heights(
            q, tseg, filt.tool_hull))))
        if k % 250 == 0:
            log.append((k / 1000, info["phase"], info["s"], info["lat_err"],
                        info["rot_err_deg"], info["f_ax_contact"]))
        if info["phase"] == "done":
            break

    p, R, _ = model.point_frame_jacobian(d.qpos, tseg, TIP)
    tip_err = np.linalg.norm(p - target)
    _, s, lat = ctl.tip_state(d.qpos)
    ang = np.degrees(np.arccos(np.clip(float((R @ (TIP / np.linalg.norm(TIP)))
                                            @ a), -1, 1)))
    return dict(phase=ctl.phase, abort=ctl.abort_reason, tip_err=tip_err,
                lat=np.linalg.norm(lat), axial_left=along - s, ang=ang,
                wall_f=worst_wall_f, needle_f=contact_needle_f,
                lowest_vs_floor=lowest - floor,
                lowest_vs_wall=lowest - (rim_z - cg.MAX_INSERT_DEPTH_45),
                first_wall=first_wall,
                log=log, target=target, floor=floor)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mjcf", default=DEFAULT_MJCF)
    ap.add_argument("--scenario", nargs="*",
                    default=["nominal", "model-error", "blocked", "overshoot"])
    ap.add_argument("--depth-mm", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=0.005,
                    help="axial insertion speed (m/s)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    fails = []
    for sc in args.scenario:
        r = run(args, sc)
        print(f"\n=== {sc}: phase {r['phase']}"
              + (f"  (abort: {r['abort']})" if r["abort"] else ""))
        print(f"    tip error {r['tip_err']*1000:.2f} mm (lateral "
              f"{r['lat']*1000:.2f} mm, axial short by "
              f"{r['axial_left']*1000:+.2f} mm), angle {r['ang']:.2f} deg")
        print(f"    container contact: fixture/wall {r['wall_f']:.1f} N, "
              f"needle {r['needle_f']:.1f} N;  lowest tool point "
              f"{r['lowest_vs_floor']*1000:+.1f} mm vs floor, "
              f"{r['lowest_vs_wall']*1000:+.1f} mm vs wall-contact depth")
        if r["first_wall"] is not None:
            t_, ph, s_, la, ro, zr = r["first_wall"]
            print(f"    fixture first touched the container at t={t_:.2f}s "
                  f"({ph}, s={s_*1000:.1f} mm, lateral {la*1000:.2f} mm, "
                  f"rot {ro:.2f} deg, tip {zr*1000:+.1f} mm vs rim)")
        if args.verbose:
            for row in r["log"]:
                print("      t=%5.2f %-8s s=%6.1f mm lat=%5.2f mm rot=%4.2f deg "
                      "f_contact=%5.1f N" % (row[0], row[1], row[2] * 1000,
                                             row[3] * 1000, row[4], row[5]))
        ok = True
        if sc in ("nominal", "model-error"):
            ok = (r["phase"] == "hold" and r["lat"] < GATE_POS
                  and r["ang"] < GATE_DEG and r["wall_f"] < 1.0)
        elif sc == "blocked":
            ok = r["phase"] == "done" and r["needle_f"] < 40.0
        elif sc == "overshoot":
            ok = r["lowest_vs_wall"] > 0 and r["lowest_vs_floor"] > -0.001
        print(f"    [{'PASS' if ok else 'FAIL'}]")
        if not ok:
            fails.append(sc)
    print("\n" + "=" * 60)
    if fails:
        print("FAILED: " + ", ".join(fails))
        sys.exit(1)
    print("All insertion scenarios passed.")


if __name__ == "__main__":
    main()
