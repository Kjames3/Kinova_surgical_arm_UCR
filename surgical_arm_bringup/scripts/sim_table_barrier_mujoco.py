#!/usr/bin/env python3
"""
MuJoCo cross-check of the table barrier, with the thesis_ee tool mounted.

WHY THIS EXISTS
---------------
validate_table_barrier.py proves the barrier against a plant integrated from
the SAME KDL model the controller uses -- so any mistake the model makes, the
plant makes too, and the test cannot see it. This bench swaps the plant for
MuJoCo (independent dynamics, integrator and geometry) and adds the error the
real arm is known to have, then asks one question: when an operator shoves the
arm toward the table, does any part of the robot -- arm or tool -- go through?

The controller side is impedance.py's own code: TableSafetyFilter, the exact
object the hot loop runs (observer, wall, table/tool rows, joint stops, speed
limits, gravity reserve, torque limits, one exact projection). Only the
operator is simulated: a hand that pushes with up to F newtons against a still
arm and cannot follow faster than --hand-speed (a constant force source would
fling the free arm at metres per second, which no person does).

The output is the push the barrier holds in EVERY case -- 3 start poses x 3
push sites (needle, flange, tool body) x 3 directions -- per plant. That is
the number to quote: the arm has 39/9 Nm joints and a ~40 N rated payload at
reach, so above some force no controller can keep the tool off the table, and
the gravity reserve exists to keep that force high wherever the operator
takes the arm.

PLANT vs MODEL -- WHAT DIFFERS ON PURPOSE
-----------------------------------------
  * Dynamics: MuJoCo's, from the surgical xacro (gen3 + thesis_ee), not KDL's.
  * Gravity error: the controller's feedforward on joint 4 is scaled by
    (1 - --g-error-j4), default 30% low -- the KDL underestimate measured on
    hardware (CLAUDE.md, impedance accuracy status). Unmodelled gravity is an
    unseen push the barrier was never told about.
  * Joint damping / rotor inertia: 0 by default. The MJCF's 1.0 / 0.1 are
    stabilisers, not measurements, and real friction only HELPS the barrier,
    so leaving them out is the worst case. Plant 'g-err+rotor' adds the
    rotor inertia back (a geared joint reflects it by the ratio squared).
  * Tool inertia: by default the plant carries the SolidWorks tool inertia
    (impedance.TOOL_PAYLOADS, what the controller believes). Plant
    'xacro-tool' keeps the MJCF's pen-era <inertial> instead, as a
    deliberate mass error.

Penetration is scored on MuJoCo's own geometry: every vertex of the tool and
arm-link meshes, plus the measured needle tip -- not on the points the barrier
chose to guard. Contacts are disabled so the table cannot hold the arm up and
hide a failure; the table exists only as a height.

The plant MJCF is ~/mujoco_models/gen3_thesis_ee.xml, written by
mujoco_import_urdf.py. It bakes absolute mesh paths, so regenerate it locally
rather than copying it between machines.

USAGE
-----
  python3 sim_table_barrier_mujoco.py                  # all plants, 20-90 N
  python3 sim_table_barrier_mujoco.py --quick          # 1 plant, rated force
  python3 sim_table_barrier_mujoco.py --plants nominal --forces 30 60
  python3 sim_table_barrier_mujoco.py --gravity-reserve 0.3
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco                                              # noqa: E402

import impedance as imp                                    # noqa: E402

DEFAULT_MJCF = os.path.expanduser("~/mujoco_models/gen3_thesis_ee.xml")
TIP = np.array([0.108, -0.008, -0.411])      # measured assembly_tip, bracelet
TABLE_Z = imp.TABLE_SURFACE_Z

# Barrier settings: impedance.py's argparse defaults.
LINK_Z_MIN = 0.08
TOOL_Z_MIN = -0.02
LINK_RADIUS = imp.KinDynModel.CAPSULE_RADIUS
TOOL_K = 4
ALPHA = (15.0, 15.0)
WALL = dict(standoff=0.06, k_wall=1200.0, d_wall=60.0, f_max=90.0)

ARM_BODIES = ["half_arm_2_link", "forearm_link", "spherical_wrist_1_link",
              "spherical_wrist_2_link", "bracelet_link"]


# --- model (controller side) -------------------------------------------------
class Controller:
    """impedance.TableSafetyFilter -- the hot loop's own filter -- plus the
    commanded gravity feedforward, carrying the configured error."""

    def __init__(self, g_error_j4, observer_gain=50.0, gravity_reserve=0.4):
        self.model = imp.KinDynModel(imp.DEFAULT_URDF, imp.DEFAULT_BASE_LINK,
                                     imp.DEFAULT_TIP_LINK, tool="thesis_ee")
        self.n = self.model.n
        names = self.model.segment_names()
        self.links = [nm for nm in imp.DEFAULT_TABLE_LINKS if nm in names]
        self.segs = [self.model.segment_index(nm) for nm in self.links]
        self.hull = imp.load_tool_hull("thesis_ee")
        self.tseg = self.model.segment_index(imp.TOOL_HULL_LINK)
        self.g_scale = np.ones(self.n)
        self.g_scale[3] = 1.0 - g_error_j4
        self.tau_max = imp.TAU_MAX[:self.n]
        self.filter = imp.TableSafetyFilter(
            self.model, self.segs, LINK_RADIUS, LINK_Z_MIN, ALPHA,
            self.tau_max, WALL, tool_hull=self.hull, tool_k=TOOL_K,
            tool_z_min=TOOL_Z_MIN, observer_gain=observer_gain,
            gravity_reserve=gravity_reserve)

    def reset(self):
        self.filter.reset()

    def step(self, q, dq, tau_task):
        """Return the torque the robot would send, plus filter info."""
        tau_g = self.model.gravity(q) * self.g_scale
        tau, info = self.filter.apply(q, dq, tau_g + tau_task, tau_g, 1e-3)
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        self.filter.sent(tau)
        return tau, info

    def bracelet_pose(self, q):
        f = self.model._scratch_frame
        self.model._fk.JntToCart(self.model._to_jnt(q), f, self.tseg)
        R = np.array([[f.M[i, j] for j in range(3)] for i in range(3)])
        return np.array([f.p[i] for i in range(3)]), R

    def push_jacobian(self, q, site):
        """3xN linear Jacobian (controller's model) of a bracelet-fixed point."""
        J = self.model.jacobian(q)
        p_ee, _ = self.model.fk(q)
        p_b, R = self.bracelet_pose(q)
        return J[:3] - np.cross(p_b + R @ site - p_ee, J[3:].T).T


# --- plant (MuJoCo side) -----------------------------------------------------
class Plant:
    """MuJoCo arm driven by joint torque, scored on its own mesh geometry."""

    def __init__(self, path, damping, armature, tool, substeps, ctrl):
        self.m = mujoco.MjModel.from_xml_path(path)
        m = self.m
        # Torque-driven: the MJCF's position servos and contacts are off.
        m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_ACTUATION
                               | mujoco.mjtDisableBit.mjDSBL_CONTACT)
        m.opt.timestep = 1e-3 / substeps
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        self.substeps = substeps
        m.dof_damping[:] = damping
        m.dof_armature[:] = armature
        self.bid = m.body("bracelet_link").id
        if tool == "solidworks":
            self._set_tool_inertia(ctrl)
        self.d = mujoco.MjData(m)

        # Scoring geometry: hull of every mesh geom on each body, expressed in
        # that body's frame. Hulls are exact for "lowest point" against a plane.
        from scipy.spatial import ConvexHull
        self.score = []          # (body id, label, (k,3) local points)
        for name in ARM_BODIES:
            b = m.body(name).id
            pts = []
            for g in range(m.ngeom):
                if (m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                        or m.geom_group[g] != 1):
                    continue
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:
                                m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                local = v @ R.reshape(3, 3).T + m.geom_pos[g]
                if m.mesh(mid).name in ("elbow", "Full_Assembly"):
                    self.score.append((b, "tool", local[ConvexHull(local).vertices]))
                else:
                    pts.append(local)
            if pts:
                allp = np.vstack(pts)
                self.score.append((b, name, allp[ConvexHull(allp).vertices]))
        self.score.append((self.bid, "tool", TIP[None]))

    def _set_tool_inertia(self, ctrl):
        """Replace the bracelet body's merged inertia with what the controller
        believes: KDL's bracelet + flange segments (vision bracelet + SolidWorks
        tool), combined in the bracelet frame."""
        ch = ctrl.model.chain
        names = ctrl.model.segment_names()
        sb = names.index("gen3_bracelet_link")
        se = names.index("gen3_end_effector_link")
        I_b = ch.getSegment(sb).getInertia()
        T_be = ch.getSegment(se).pose(0.0)       # fixed joint
        I = I_b + T_be * ch.getSegment(se).getInertia()
        mass = I.getMass()
        com = np.array([I.getCOG()[i] for i in range(3)])
        Io = I.getRotationalInertia()             # about the frame origin
        Io = np.array([[Io[3 * r + c] for c in range(3)] for r in range(3)])
        Ic = Io - mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
        w, V = np.linalg.eigh(Ic)
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, V.flatten())
        m = self.m
        m.body_mass[self.bid] = mass
        m.body_ipos[self.bid] = com
        m.body_iquat[self.bid] = quat
        m.body_inertia[self.bid] = w

    def reset(self, q):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)

    def step(self, tau, site, force, hand_speed=np.inf):
        """Hold `tau` for one 1 kHz period while the operator pushes.

        The operator is a hand, not a force source: it delivers the full
        `force` against an arm that is not moving, and less as the push point
        runs away from it, reaching zero at `hand_speed` (m/s) along the push.
        A constant force instead accelerates the free arm to metres per second
        -- a shove no person can follow, and one that no torque-limited brake
        can stop in a few centimetres.
        """
        m, d = self.m, self.d
        fmag = np.linalg.norm(force)
        u = force / fmag if fmag > 0 else force
        jacp = np.zeros((3, m.nv))
        for _ in range(self.substeps):
            d.qfrc_applied[:] = tau
            p = d.xpos[self.bid] + d.xmat[self.bid].reshape(3, 3) @ site
            f = force
            if np.isfinite(hand_speed) and fmag > 0:
                mujoco.mj_jac(m, d, jacp, None, p, self.bid)
                v_along = float(u @ (jacp @ d.qvel))
                f = force * np.clip(1.0 - v_along / hand_speed, 0.0, 1.0)
            mujoco.mj_applyFT(m, d, f, np.zeros(3), p, self.bid,
                              d.qfrc_applied)
            mujoco.mj_step(m, d)

    def clearances(self):
        """(tool clearance to TOOL_Z_MIN, lowest arm-link mesh above the table)."""
        d = self.d
        tool = link = np.inf
        for b, label, pts in self.score:
            z = d.xpos[b][2] + pts @ d.xmat[b].reshape(3, 3)[2]
            if label == "tool":
                tool = min(tool, z.min() - TOOL_Z_MIN)
            else:
                link = min(link, z.min() - TABLE_Z)
        return tool, link


# --- scenarios ---------------------------------------------------------------
def find_pose(ctrl, tilt_deg, seed):
    """Seeded search: needle tilted `tilt_deg` from straight down, tool
    50-200 mm above its floor, inside the joint stops and the gravity
    reserve -- i.e. a pose the filter itself considers safe."""
    r = np.random.default_rng(seed)
    axis = TIP / np.linalg.norm(TIP)
    mdl = ctrl.model
    for _ in range(200000):
        q = r.uniform(-2.2, 2.2, ctrl.n)
        clr = np.min(mdl.body_point_heights(q, ctrl.tseg, ctrl.hull)) - TOOL_Z_MIN
        if not 0.05 < clr < 0.20:
            continue
        z, _, _ = mdl.height_barrier_terms(q, np.zeros(ctrl.n), ctrl.segs)
        if np.min(z - LINK_RADIUS) < LINK_Z_MIN + 0.03:
            continue
        if np.any(np.abs(q) > imp.Q_HARD_MAX[:ctrl.n] - 0.1):
            continue
        g = mdl.gravity(q)[ctrl.filter.reserve_joints]
        if np.any(np.abs(g) > 0.9 * ctrl.filter.reserve_limit):
            continue
        _, R = ctrl.bracelet_pose(q)
        tilt = np.degrees(np.arccos(np.clip(-(R @ axis)[2], -1, 1)))
        if abs(tilt - tilt_deg) < 5.0:
            return q
    raise RuntimeError(f"no start pose at {tilt_deg} deg")


SITES = {"needle tip": TIP, "flange": np.array([0.0, 0.0, -0.0615]),
         "tool body": np.array([0.115, 0.0, -0.20])}
DIRS = {"down": np.array([0, 0, -1.0]),
        "down+fwd": np.array([0.7, 0, -0.7]) / np.hypot(0.7, 0.7),
        "down+side": np.array([0, 0.7, -0.7]) / np.hypot(0.7, 0.7)}
PLANTS = {
    # name: (g_error_j4, damping, armature, tool inertia)
    "nominal":    (0.0, 0.0, 0.0, "solidworks"),
    "g-error":    (0.3, 0.0, 0.0, "solidworks"),
    "g-err+rotor": (0.3, 0.0, 0.1, "solidworks"),
    "xacro-tool": (0.3, 0.0, 0.0, "xacro"),
}


def run(ctrl, plant, q0, site, direction, force, T, guarded=True,
        hand_speed=0.5):
    plant.reset(q0)
    ctrl.reset()
    n = ctrl.n
    worst_tool = worst_link = np.inf
    degraded = 0
    peak_dq = 0.0
    for _ in range(int(T * 1000)):
        q = plant.d.qpos[:n].copy()
        dq = plant.d.qvel[:n].copy()
        t_c, l_c = plant.clearances()
        worst_tool, worst_link = min(worst_tool, t_c), min(worst_link, l_c)
        peak_dq = max(peak_dq, np.max(np.abs(dq)))
        # Operator damping: KD_GLOBAL only, the least any mode applies.
        tau_task = -imp.KD_GLOBAL[:n] * dq
        if guarded:
            tau, info = ctrl.step(q, dq, tau_task)
            degraded += info["status"] == "degraded"
        else:
            tau = ctrl.model.gravity(q) * ctrl.g_scale + tau_task
        plant.step(tau, site, force * direction, hand_speed)
        if not np.all(np.isfinite(plant.d.qpos)):
            raise RuntimeError("plant diverged")
    return dict(tool=worst_tool, link=worst_link, degraded=degraded,
                peak=peak_dq)


def check_model(ctrl, plant):
    """How far the MuJoCo plant is from the controller's model (for the log)."""
    rng = np.random.default_rng(0)
    m, d = plant.m, plant.d
    eg = eM = 0.0
    for _ in range(20):
        q = rng.uniform(-2, 2, ctrl.n)
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        g_plant = d.qfrc_bias.copy()
        M_plant = np.zeros((ctrl.n, ctrl.n))
        mujoco.mj_fullM(m, d, M_plant)
        eg = max(eg, np.max(np.abs(g_plant - ctrl.model.gravity(q) * ctrl.g_scale)))
        M = ctrl.model.mass(q)
        eM = max(eM, np.max(np.abs(M_plant - M)) / np.max(np.abs(M)))
    return eg, eM


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", default=DEFAULT_MJCF)
    ap.add_argument("--plants", nargs="*", default=list(PLANTS),
                    choices=list(PLANTS))
    ap.add_argument("--forces", nargs="*", type=float,
                    default=[15, 20, 25, 30, 40, 50, 60])
    ap.add_argument("--rated-force", type=float, default=20.0,
                    help="Push the barrier must hold in EVERY case (N); the "
                         "run fails if it does not.")
    ap.add_argument("--gravity-reserve", type=float, default=0.4)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--hand-speed", type=float, default=0.5,
                    help="Operator hand speed limit (m/s).")
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="g-error plant, rated force only")
    args = ap.parse_args()
    if args.quick:
        args.plants, args.forces = ["g-error"], [args.rated_force]
    if not os.path.exists(args.mjcf):
        sys.exit(f"{args.mjcf} not found -- generate it with "
                 "mujoco_import_urdf.py (see its header).")

    fails = []
    for pname in args.plants:
        g_err, damp, arm, tool = PLANTS[pname]
        ctrl = Controller(g_err, gravity_reserve=args.gravity_reserve)
        plant = Plant(args.mjcf, damp, arm, tool, args.substeps, ctrl)
        eg, eM = check_model(ctrl, plant)
        print(f"\n=== plant '{pname}': g error j4 {g_err:.0%}, damping {damp}, "
              f"armature {arm}, tool inertia {tool}")
        print(f"    plant vs controller model: gravity up to {eg:.2f} Nm, "
              f"mass matrix up to {eM:.0%}")
        poses = {"needle down": find_pose(ctrl, 0.0, 1),
                 "tilted 45": find_pose(ctrl, 45.0, 2),
                 "sideways": find_pose(ctrl, 90.0, 3)}

        base = run(ctrl, plant, poses["needle down"], TIP, DIRS["down"],
                   args.rated_force, args.duration, guarded=False,
                   hand_speed=args.hand_speed)
        print(f"    unguarded, {args.rated_force:g} N on the needle: tool "
              f"{base['tool']*1000:+.0f} mm  "
              f"[{'went through' if base['tool'] < 0 else 'HELD?!'}]")

        guaranteed = 0.0
        all_held_so_far = True
        t0 = time.time()
        for F in sorted(args.forces):
            worst, bad, deg, peak = np.inf, [], 0, 0.0
            for qn, q0 in poses.items():
                for sn, site in SITES.items():
                    for dn, dv in DIRS.items():
                        r = run(ctrl, plant, q0, site, dv, F, args.duration,
                                hand_speed=args.hand_speed)
                        w = min(r["tool"], r["link"])
                        worst = min(worst, w)
                        deg += r["degraded"] > 0
                        peak = max(peak, r["peak"])
                        if w < -1e-3:
                            bad.append(f"{qn}/{sn}/{dn}: tool "
                                       f"{r['tool']*1000:+.0f} mm, links "
                                       f"{r['link']*1000:+.0f} mm above table")
            n_case = len(poses) * len(SITES) * len(DIRS)
            print(f"    {F:5.0f} N: held {n_case - len(bad)}/{n_case}, worst "
                  f"clearance {worst*1000:+7.1f} mm, degraded in {deg} runs, "
                  f"peak joint speed {peak:.2f} rad/s")
            if all_held_so_far and not bad:
                guaranteed = F
            else:
                all_held_so_far = False
            if F <= args.rated_force and bad:
                for s_ in bad:
                    print("        FAILED " + s_)
                fails.append(f"{pname} at {F:g} N")
        print(f"    -> held in every case up to {guaranteed:g} N "
              f"({time.time() - t0:.0f} s)")

    print("\n" + "=" * 60)
    if fails:
        print("FAILED: " + ", ".join(fails))
        sys.exit(1)
    print(f"Barrier held every push up to the rated {args.rated_force:g} N, "
          f"on every plant.")


if __name__ == "__main__":
    main()
