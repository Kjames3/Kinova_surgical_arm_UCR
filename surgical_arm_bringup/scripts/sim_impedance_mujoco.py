#!/usr/bin/env python3
"""
MuJoCo test bench for impedance.py's cartesian (operational-space) control law.

WHY THIS EXISTS
---------------
Cartesian impedance is the one mode that can command an unbounded torque: the
task torque is tau = Jᵀ * Lambda * F with Lambda = (J M⁻¹ Jᵀ)⁻¹, and Lambda
blows up as the Jacobian loses rank. Provoking that on the real arm means
deliberately driving it into a singularity in TORQUE mode, which is exactly the
experiment nobody should run on hardware first. This bench runs the REAL control
law -- it imports cartesian_impedance_torque from impedance.py, it does not
reimplement it -- against a MuJoCo plant, so the singularity behaviour can be
measured, and a fix compared against the old behaviour, with no robot involved.

It needs no ROS, no Kortex SDK and no robot. It is a sibling of
validate_table_barrier.py, which does the same for the table barrier.

PLANT == MODEL, ON PURPOSE
--------------------------
The plant is built from the SAME URDF the controller's KDL model uses
(config/gen3_2f85.urdf), pruned to exactly the base_link -> end_effector_link
chain that KDL builds. That is deliberate: it makes the bench measure the
control law's singularity handling and nothing else.

Do NOT use ~/mujoco_models/gen3_thesis_ee.xml as the plant here. That MJCF is
generated from the *surgical* xacro (gripper:=thesis_ee) and is a different
robot from the controller's gen3_2f85 URDF -- measured disagreement is up to
12 Nm of gravity torque and ~50% of the mass matrix. Such a mismatch is a real
and separately-tracked problem (see CLAUDE.md, "the model has the wrong end
effector"), but mixing it in here would drown the signal this bench is for.
--check-model prints the residual so this assumption is never silently wrong.

USAGE
-----
  # the fix vs. the behaviour it replaces, on the elbow singularity
  python3 sim_impedance_mujoco.py --scenario reach --guard legacy
  python3 sim_impedance_mujoco.py --scenario reach --guard damped

  # proactive redundancy escape on top of the damped inversion
  python3 sim_impedance_mujoco.py --scenario reach --guard damped --sing-avoid

  # regression: a well-conditioned hold must be IDENTICAL under both guards
  python3 sim_impedance_mujoco.py --scenario hold --guard legacy
  python3 sim_impedance_mujoco.py --scenario hold --guard damped

  # everything at once, with a pass/fail verdict
  python3 sim_impedance_mujoco.py --compare
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco is not installed: pip install mujoco")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impedance as imp   # noqa: E402  (path must be set first)


# The plant is regenerated from the URDF on every run and is a derived artifact,
# so it lives in a temp path and is never committed.
PLANT_URDF = "/tmp/gen3_impedance_plant.urdf"


# =============================================================================
# Plant construction
# =============================================================================
def build_plant_urdf(urdf_path, base_link, tip_link, out_path=PLANT_URDF):
    """Strip a ROS URDF down to the base->tip chain and make MuJoCo accept it.

    Three things have to go:
      * <visual>/<collision> -- they reference package:// meshes MuJoCo cannot
        resolve, and this bench needs inertias only, never geometry.
      * <ros2_control>/<gazebo>/<transmission> -- ROS-only extensions.
      * every link NOT on the base->tip path -- the gripper fingers are a side
        BRANCH, and KDL's getChain() only includes on-path links, so leaving
        their mass in the plant would give the plant inertia the controller's
        model cannot see. That is the very mismatch this bench avoids.

    Returns the ordered list of movable joint names on the chain.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Walk child->parent from the tip to recover the chain, then keep only it.
    parent_of, joint_to_parent = {}, {}
    for j in root.findall("joint"):
        child = j.find("child").attrib["link"]
        parent = j.find("parent").attrib["link"]
        parent_of[child] = parent
        joint_to_parent[child] = j
    chain_links, node = [tip_link], tip_link
    while node != base_link:
        if node not in parent_of:
            raise SystemExit(
                "link '%s' is not an ancestor of '%s' in %s -- check "
                "--base-link/--tip-link" % (base_link, tip_link, urdf_path))
        node = parent_of[node]
        chain_links.append(node)
    chain_links = list(reversed(chain_links))
    keep = set(chain_links)

    for link in root.findall("link"):
        if link.attrib.get("name") not in keep:
            root.remove(link)
            continue
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    for j in list(root.findall("joint")):
        # Drop a joint if EITHER end is gone. The parent test matters for the
        # world->base fixed joint: base_link is on the chain but `world` is not,
        # and leaving that joint behind makes MuJoCo reject the file outright
        # ("URDF joint parent or child missing"). Removing it leaves base_link
        # as the tree root, which is the fixed base we want anyway.
        if (j.find("child").attrib["link"] not in keep
                or j.find("parent").attrib["link"] not in keep):
            root.remove(j)
    for tag in ("ros2_control", "gazebo", "transmission"):
        for el in root.findall(tag):
            root.remove(el)

    # With no geoms left, MuJoCo must take inertia from <inertial> verbatim.
    # balanceinertia repairs any principal-moment triangle violation in the
    # vendor URDF, which MuJoCo rejects outright but KDL silently tolerates.
    mj = ET.Element("mujoco")
    ET.SubElement(mj, "compiler", {"balanceinertia": "true",
                                   "discardvisual": "true",
                                   "fusestatic": "false"})
    root.insert(0, mj)
    with open(out_path, "w") as f:
        f.write(ET.tostring(root, encoding="unicode"))

    joints = [joint_to_parent[l] for l in chain_links[1:]]
    return [j.attrib["name"] for j in joints
            if j.attrib.get("type") in ("revolute", "continuous", "prismatic")]


def load_plant(urdf_path, base_link, tip_link, timestep, armature=0.0):
    """Compile the pruned URDF and return (model, data, dofadr, qposadr).

    ``armature`` adds reflected actuator rotor inertia to every DOF. The vendor
    URDF has none, which leaves the distal joints at ~1e-4 kg m^2 -- so light
    that the controller's own KD_GLOBAL damping, applied as a zero-order hold at
    1 kHz, is DISCRETELY unstable there (kd*h/I > 2) and the plant chatters at
    ~23 rad/s with the control torque identically zero. That is an artifact of
    an unrealistic plant, not a property of the impedance law: the real
    harmonic-drive actuators carry rotor inertia that dominates the distal link
    inertia. Substepping the physics does NOT fix it, because the damping torque
    is only refreshed at the control rate.

    The default value is a conservative round number chosen to make the discrete
    damping comfortably stable. It is NOT a measured Gen3 rotor inertia, and it
    is deliberately absent from the controller's KDL model -- the real arm has
    armature that KDL does not model either, so this mismatch is realistic.
    """
    joint_names = build_plant_urdf(urdf_path, base_link, tip_link)
    m = mujoco.MjModel.from_xml_path(PLANT_URDF)
    m.opt.timestep = timestep
    if armature > 0.0:
        m.dof_armature[:] = armature

    # There are no actuators (the URDF had no transmissions we kept), so torque
    # goes in through qfrc_applied. Assert that rather than assume it: a stray
    # position servo would quietly dominate the impedance and make every number
    # below meaningless.
    if m.nu != 0:
        raise SystemExit("plant has %d actuators; expected 0 (torque is applied "
                         "via qfrc_applied)" % m.nu)

    dofadr, qposadr = [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit("joint '%s' missing from compiled plant" % name)
        dofadr.append(int(m.jnt_dofadr[jid]))
        qposadr.append(int(m.jnt_qposadr[jid]))
    return m, mujoco.MjData(m), np.array(dofadr), np.array(qposadr), joint_names


def check_model_match(m, d, kin, qposadr, dofadr, poses):
    """Max |gravity| and |mass matrix| residual between the plant and KDL.

    The bench's central assumption is plant == model. This quantifies it. Both
    residuals should be at round-off; anything larger means the pruning above
    did not reproduce the KDL chain and the results are not interpretable.
    """
    n = len(qposadr)
    Mfull = np.zeros((m.nv, m.nv))
    dg_max = dM_max = 0.0
    for q in poses:
        d.qpos[qposadr] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        dg_max = max(dg_max, np.max(np.abs(kin.gravity(q) - d.qfrc_bias[dofadr])))
        mujoco.mj_fullM(m, d, Mfull)
        dM_max = max(dM_max, np.max(np.abs(kin.mass(q) - Mfull[np.ix_(dofadr, dofadr)])))
    return dg_max, dM_max


# =============================================================================
# Run
# =============================================================================
class RunResult:
    """Everything the verdict logic needs from one simulated run."""

    def __init__(self):
        self.tau_peak = 0.0          # Nm, max |commanded torque| BEFORE clamping
        self.tau_peak_clamped = 0.0  # Nm, max |torque| actually applied
        self.lambda_peak = 0.0       # max ||Lambda||_2 seen
        self.sigma_min = np.inf      # closest approach to a singularity
        self.dq_peak = 0.0           # rad/s, max |joint velocity|
        self.tip_err = np.nan        # m, final |p_des - p|
        self.clamp_frac = 0.0        # fraction of cycles hitting the TAU_MAX clamp
        self.aborted = None          # reason string, or None
        self.abort_sigma = np.nan    # sigma_min when the guard first tripped
        self.abort_lambda = np.nan   # ||Lambda|| already reached by then
        self.abort_t = np.nan        # s, when it tripped
        self.reached_stop = False    # hit the common --sigma-stop depth
        self.speed_abort_t = np.nan  # s, first breach of MAX_JOINT_SPEED
        self.speed_abort_sigma = np.nan
        self.stopped_on_speed = False
        self.diverged = False        # NaN / non-finite state
        self.steps = 0
        self.trace = []              # (t, sigma_min, tau_peak, lam) per sample


def simulate(args, guard, sing_avoid=False, verbose=False):
    """Run one scenario under one guard and return a RunResult.

    ``guard='legacy'`` reproduces the pre-2026-08-18 behaviour: an undamped
    Lambda inversion (SING_SIGMA_ON = 0 makes damping_factor return 0
    everywhere, which is exactly the old plain np.linalg.inv branch -- the old
    det(Lambda_inv) >= 1e-2 test is not modelled because, as measured, it never
    fired) and the old abort gate on sqrt(det(J Jᵀ)).

    ``guard='damped'`` is the current behaviour: variable-damped inversion and
    an abort gate on sigma_min.
    """
    kin = imp.KinDynModel(args.urdf, args.base_link, args.tip_link)
    # Physics runs --substeps times faster than control. This is not cosmetic:
    # the distal joints of the bare URDF chain carry no actuator armature (the
    # vendor URDF has none, and the real harmonic drives contribute rotor
    # inertia the URDF never models), so their link inertia is ~1e-4 kg m^2.
    # Explicit integration of the controller's own KD_GLOBAL damping on such a
    # light DOF is unstable when kd*h/I > 2, and at h = 1 ms that condition is
    # met: the plant chatters at ~23 rad/s with the control torque identically
    # zero. Substepping fixes the integration without touching the controller
    # or the model, which is what keeps plant == model exact.
    m, d, dofadr, qposadr, joint_names = load_plant(
        args.urdf, args.base_link, args.tip_link,
        1.0 / (args.rate * args.substeps), args.armature)
    n = len(qposadr)

    # Temporarily disable the damping ramp for the legacy comparison. Restored
    # in the finally below so a --compare run cannot leak the override into the
    # next simulate() call.
    saved_sigma_on = imp.SING_SIGMA_ON
    if guard == "legacy":
        imp.SING_SIGMA_ON = 0.0

    try:
        q0 = np.array(args.q0, dtype=float)
        d.qpos[qposadr] = q0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        p0, quat0 = kin.fk(q0)
        if args.scenario == "hold":
            # Well-conditioned regression case: the guards must not change it.
            p_des = p0.copy()
        elif args.scenario == "reach":
            # The realistic way an operator meets the elbow singularity: walk
            # the commanded tip pose outward past the arm's reach. The task
            # spring pulls the elbow straight and parks the arm ON the singular
            # set, where it stays -- a transient pass-through would understate
            # the problem.
            #
            # The setpoint is RAMPED, not stepped. A step of 0.35 m produces a
            # 56 Nm transient on the very first cycle, which then dominates
            # every peak-torque number and hides the singularity behaviour the
            # bench is measuring. Ramping keeps the approach quasi-static so the
            # peaks that survive are attributable to the Jacobian conditioning.
            radial = p0.copy()
            radial[2] = 0.0
            norm = np.linalg.norm(radial)
            direction = radial / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            p_target = p0 + args.reach * direction
            p_des = p0.copy()
        else:
            raise SystemExit("unknown scenario %r" % args.scenario)
        quat_des = quat0.copy()

        kp_cart = np.concatenate([imp.KP_CART_POS, imp.KP_CART_ORI]) * args.cart_scale
        kd_cart = 2.0 * imp.CART_DAMPING_RATIO * np.sqrt(kp_cart)
        kp_null = imp.KP_NULL[:n] * args.null_stiffness
        kd_null = 2.0 * np.sqrt(kp_null)
        tau_max = imp.TAU_MAX[:n]

        res = RunResult()
        tau_sing = np.zeros(n)
        sing_grad = np.zeros(n)
        sing_col = 0
        clamp_hits = 0
        dt = 1.0 / args.rate
        nsteps = int(args.duration * args.rate)
        Minv = np.zeros((n, n))

        for k in range(nsteps):
            q = np.array(d.qpos[qposadr], dtype=float)
            dq = np.array(d.qvel[dofadr], dtype=float)
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
                res.diverged = True
                break

            if args.scenario == "reach":
                # Linear ramp over --ramp seconds, then held at the target so
                # the arm sits in the singular configuration rather than passing
                # through it.
                s = min(1.0, (k * dt) / max(args.ramp, 1e-9))
                p_des = p0 + s * (p_target - p0)

            # Exact gravity feedforward. On hardware this term is the dominant
            # error source, but here plant == model, so making it exact is what
            # isolates the singularity behaviour from gravity droop.
            ramp = (min(1.0, (k * dt) / args.ramp_time)
                    if args.ramp_time > 0.0 else 1.0)

            tau_g = kin.gravity(q)

            M = kin.mass(q)
            Minv = np.linalg.inv(M)

            tau, manip, sigma_min, lam = imp.cartesian_impedance_torque(
                kin, q, dq, p_des, quat_des, q0,
                kp_cart, kd_cart, kp_null, kd_null, tau_g,
                np.zeros(n), imp.K_LIMIT, dyn=(M, Minv), tau_sing=tau_sing)

            # Same scheduled one-column-per-cycle sweep the real loop uses.
            if sing_avoid and sigma_min < saved_sigma_on:
                col, val = imp.sigma_min_gradient(kin, q, column=sing_col)
                sing_grad[col] = val
                sing_col += 1
                if sing_col >= n:
                    sing_col = 0
                    tau_sing = np.clip(args.sing_gain * sing_grad,
                                       -imp.MAX_SING_TORQUE, imp.MAX_SING_TORQUE)
            elif sing_avoid:
                tau_sing = np.zeros(n)
                sing_col = 0

            J = kin.jacobian(q)
            Lam, _ = imp.damped_lambda(J @ Minv @ J.T, sigma_min)
            res.lambda_peak = max(res.lambda_peak, float(np.linalg.norm(Lam, 2)))
            res.sigma_min = min(res.sigma_min, float(sigma_min))

            # Wrist anchor: an UN-projected joint spring on the continuous
            # joints 1/3/5/7. Without it the wrist roll has no restoring
            # authority at all -- the Lambda-weighted task torque and the
            # null-space-projected posture term both leave it at ~0 -- and a
            # well-conditioned hold drifts into a null-space runaway that has
            # nothing to do with singularities. Omitting it made this bench
            # report a 22 rad/s "instability" in the hold case that the real
            # loop does not have.
            if args.wrist_anchor > 0.0:
                tau = tau + ramp * (args.wrist_anchor
                                    * imp.CONTINUOUS_JOINTS[:n]
                                    * imp._wrap_rad(q0 - q))

            # Global joint damping, applied to the TOTAL command outside the
            # null-space projector exactly as the real loop does. It matters a
            # lot here: a damped Lambda deliberately gives up authority along
            # the degenerate direction, and that costs task DAMPING as well as
            # task stiffness, so this joint-space term is what still resists
            # runaway velocity when the arm is in the singular configuration.
            tau = tau - imp.KD_GLOBAL[:n] * args.kd_scale * dq

            # Soft engage: fade the CONTROL torque in while gravity stays full
            # from the first cycle, so the arm neither jumps nor sags at t=0.
            if args.ramp_time > 0.0:
                tau = tau_g + ramp * (tau - tau_g)

            if not np.all(np.isfinite(tau)):
                res.aborted = "non-finite torque"
                res.diverged = True
                break

            # The real loop's last line of defence, reproduced so the bench
            # measures what the ARM would actually receive, not an unclamped
            # number the hardware would never see.
            # Measured on the FULL command, after every torque term, so it is
            # the number the clamp actually sees.
            res.tau_peak = max(res.tau_peak, float(np.max(np.abs(tau))))
            if np.any(np.abs(tau) > tau_max):
                clamp_hits += 1
            tau_cmd = np.clip(tau, -tau_max, tau_max)
            res.tau_peak_clamped = max(res.tau_peak_clamped,
                                       float(np.max(np.abs(tau_cmd))))

            # Record where each guard's abort WOULD fire, but do not let it end
            # the run: the two gates trip at different depths, so stopping on
            # them would compare the guards over different trajectories. The run
            # ends at a common --sigma-stop depth instead (below), which keeps
            # the peak-torque and peak-Lambda numbers directly comparable.
            if res.aborted is None:
                if guard == "legacy" and manip < imp.MIN_MANIPULABILITY:
                    res.aborted = ("legacy gate manip %.2e at t=%.2fs, "
                                   "sigma_min=%.5f" % (manip, k * dt, sigma_min))
                elif guard != "legacy" and sigma_min < imp.SING_SIGMA_ABORT:
                    res.aborted = ("sigma_min gate %.2e at t=%.2fs"
                                   % (sigma_min, k * dt))
                if res.aborted:
                    res.abort_sigma = float(sigma_min)
                    res.abort_lambda = float(np.linalg.norm(Lam, 2))
                    res.abort_t = k * dt
            if args.stop_on_abort and res.aborted:
                res.steps = k + 1
                break

            # Zero-order hold on the torque across the substeps, which is what
            # a 1 kHz controller driving a continuous plant actually does.
            d.qfrc_applied[dofadr] = tau_cmd
            for _ in range(args.substeps):
                mujoco.mj_step(m, d)
            res.dq_peak = max(res.dq_peak, float(np.max(np.abs(d.qvel[dofadr]))))
            res.steps = k + 1

            # The real loop's outermost safety net. On hardware this is what
            # actually stops a singularity runaway, so the bench has to model
            # it: without it, both guards look equally bad because the sim
            # happily runs the arm to 20+ rad/s, which the arm never would.
            dq_now = float(np.max(np.abs(d.qvel[dofadr])))
            if dq_now > imp.MAX_JOINT_SPEED:
                if np.isnan(res.speed_abort_t):
                    res.speed_abort_t = k * dt
                    res.speed_abort_sigma = float(sigma_min)
                if args.stop_on_speed:
                    res.stopped_on_speed = True
                    break

            # Common stopping depth for both guards.
            if sigma_min < args.sigma_stop:
                res.reached_stop = True
                break

            if k % max(1, int(args.rate * args.trace_period)) == 0:
                res.trace.append((k * dt, float(sigma_min),
                                  float(np.max(np.abs(tau))), float(lam)))
                if verbose:
                    print("    t=%5.2fs sigma_min=%.5f |tau|max=%9.2f lam=%.4f"
                          % (k * dt, sigma_min, np.max(np.abs(tau)), lam))

        if res.steps:
            res.clamp_frac = clamp_hits / float(res.steps)
        q = np.array(d.qpos[qposadr], dtype=float)
        if np.all(np.isfinite(q)):
            res.tip_err = float(np.linalg.norm(p_des - kin.fk(q)[0]))
        return res
    finally:
        imp.SING_SIGMA_ON = saved_sigma_on


def describe(label, res):
    print("  %-26s tau_peak=%11.1f Nm  clamped=%6.1f Nm  clamp_hit=%5.1f%%"
          % (label, res.tau_peak, res.tau_peak_clamped, 100.0 * res.clamp_frac))
    print("  %-26s ||Lambda||max=%9.3g  sigma_min=%.5f  dq_peak=%.2f rad/s"
          % ("", res.lambda_peak, res.sigma_min, res.dq_peak))
    print("  %-26s tip_err=%.4f m  steps=%d%s%s"
          % ("", res.tip_err, res.steps,
             "  reached sigma-stop" if res.reached_stop
             else ("  SPEED-STOP" if res.stopped_on_speed else ""),
             "  DIVERGED" if res.diverged else ""))
    if not np.isnan(res.speed_abort_t):
        print("  %-26s MAX_JOINT_SPEED breached at t=%.2fs (sigma_min=%.5f)"
              % ("", res.speed_abort_t, res.speed_abort_sigma))
    if res.aborted:
        print("  %-26s guard tripped: %s  (||Lambda|| already %.3g)"
              % ("", res.aborted, res.abort_lambda))
    else:
        print("  %-26s guard never tripped" % "")


def main():
    p = argparse.ArgumentParser(
        description="MuJoCo bench for impedance.py cartesian singularity handling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", choices=["hold", "reach"], default="reach",
                   help="hold: well-conditioned regression. reach: command a "
                        "tip pose beyond reach, which pulls the elbow straight "
                        "into the arm's dominant singularity.")
    p.add_argument("--guard", choices=["legacy", "damped"], default="damped",
                   help="legacy reproduces the undamped inversion + the old "
                        "sqrt(det(J Jt)) abort gate.")
    p.add_argument("--compare", action="store_true",
                   help="Run the full matrix and print a pass/fail verdict.")
    p.add_argument("--sing-avoid", action="store_true",
                   help="Also enable the null-space singularity escape.")
    p.add_argument("--sing-gain", type=float, default=imp.K_SING)
    p.add_argument("--ramp", type=float, default=5.0, metavar="S",
                   help="Seconds to walk the reach setpoint out to its target. "
                        "Kept slow so the approach is quasi-static.")
    p.add_argument("--reach", type=float, default=0.50, metavar="M",
                   help="How far beyond the startup tip pose to command, "
                        "radially outward, in the reach scenario.")
    p.add_argument("--q0", type=float, nargs=7,
                   default=[0.0, 0.26, 3.14, -2.0, 0.0, -0.93, 1.57],
                   help="Startup joint configuration (the sim Home pose).")
    p.add_argument("--duration", type=float, default=8.0, metavar="S")
    p.add_argument("--rate", type=float, default=1000.0, metavar="HZ",
                   help="Control rate, matching the arm's 1 kHz loop.")
    p.add_argument("--substeps", type=int, default=4, metavar="N",
                   help="Physics steps per control step (zero-order hold on "
                        "the torque across them).")
    p.add_argument("--armature", type=float, default=0.01, metavar="I",
                   help="Reflected actuator rotor inertia added to every DOF "
                        "of the plant. The vendor URDF has none, which leaves "
                        "the distal joints numerically unstable under the "
                        "controller's own damping; see load_plant(). Use 0 to "
                        "get an exact plant==model chain for --check-model.")
    p.add_argument("--cart-scale", type=float, default=1.0)
    p.add_argument("--kd-scale", type=float, default=1.0,
                   help="Scales the global joint damping KD_GLOBAL, as on the arm.")
    p.add_argument("--null-stiffness", type=float, default=0.4)
    p.add_argument("--sigma-stop", type=float, default=0.001, metavar="S",
                   help="End the run when sigma_min first falls below this. "
                        "Applied identically to both guards so their peak "
                        "torque and peak Lambda cover the same approach.")
    p.add_argument("--wrist-anchor", type=float, default=3.0, metavar="K",
                   help="Nm/rad un-projected spring on the continuous joints, "
                        "as on the arm. 0 reproduces the legacy behaviour.")
    p.add_argument("--ramp-time", type=float, default=1.0, metavar="S",
                   help="Soft-engage time for the control torque.")
    p.add_argument("--stop-on-speed", action="store_true", default=True,
                   help="Stop when a joint exceeds MAX_JOINT_SPEED, as the "
                        "real loop does. This is the outermost safety net.")
    p.add_argument("--no-stop-on-speed", dest="stop_on_speed",
                   action="store_false",
                   help="Keep running past the speed limit to see where the "
                        "trajectory would have gone.")
    p.add_argument("--stop-on-abort", action="store_true",
                   help="Stop at the first abort. Off by default so a run "
                        "keeps going and shows what the arm WOULD have done.")
    p.add_argument("--trace-period", type=float, default=0.25, metavar="S")
    p.add_argument("--check-model", action="store_true",
                   help="Report the plant-vs-KDL gravity/inertia residual and exit.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--urdf", default=imp.DEFAULT_URDF)
    p.add_argument("--base-link", default=imp.DEFAULT_BASE_LINK)
    p.add_argument("--tip-link", default=imp.DEFAULT_TIP_LINK)
    args = p.parse_args()

    if args.check_model:
        kin = imp.KinDynModel(args.urdf, args.base_link, args.tip_link)
        m, d, dofadr, qposadr, names = load_plant(
            args.urdf, args.base_link, args.tip_link, 1.0 / args.rate)
        rng = np.random.default_rng(0)
        poses = [np.zeros(7), np.array(args.q0)] + [rng.uniform(-1.5, 1.5, 7)
                                                    for _ in range(8)]
        dg, dM = check_model_match(m, d, kin, qposadr, dofadr, poses)
        print("plant joints: %s" % ", ".join(names))
        print("max |gravity residual| = %.3e Nm" % dg)
        print("max |mass residual|    = %.3e" % dM)
        ok = dg < 1e-6 and dM < 1e-6
        print("plant == model: %s" % ("YES" if ok else "NO -- results are NOT "
                                      "interpretable, the pruning is wrong"))
        return 0 if ok else 1

    if not args.compare:
        print("scenario=%s guard=%s sing_avoid=%s"
              % (args.scenario, args.guard, args.sing_avoid))
        res = simulate(args, args.guard, args.sing_avoid, args.verbose)
        describe("%s/%s" % (args.scenario, args.guard), res)
        return 0

    # --- comparison matrix ---------------------------------------------------
    print("=" * 74)
    print("impedance.py cartesian singularity bench -- plant built from %s"
          % os.path.basename(args.urdf))
    print("=" * 74)

    kin = imp.KinDynModel(args.urdf, args.base_link, args.tip_link)
    m, d, dofadr, qposadr, _ = load_plant(args.urdf, args.base_link,
                                          args.tip_link, 1.0 / args.rate)
    rng = np.random.default_rng(0)
    poses = [np.zeros(7), np.array(args.q0)] + [rng.uniform(-1.5, 1.5, 7)
                                                for _ in range(8)]
    dg, dM = check_model_match(m, d, kin, qposadr, dofadr, poses)
    # Checked on the BARE chain (armature 0), which is what verifies the URDF
    # pruning reproduced the KDL chain. The runs below then add --armature to
    # the plant on top of that verified chain.
    print("plant-vs-model residual (bare chain): gravity %.2e Nm, inertia "
          "%.2e -> %s" % (dg, dM, "OK" if (dg < 1e-6 and dM < 1e-6) else "MISMATCH"))
    print("runs add armature=%.4g kg m^2 per DOF to the plant only\n"
          % args.armature)

    results = {}
    for scenario in ("hold", "reach"):
        args.scenario = scenario
        for guard in ("legacy", "damped"):
            print("[%s / %s]" % (scenario, guard))
            r = simulate(args, guard, sing_avoid=False, verbose=args.verbose)
            describe("%s/%s" % (scenario, guard), r)
            results[(scenario, guard)] = r
        if scenario == "reach":
            print("[reach / damped + --sing-avoid]")
            r = simulate(args, "damped", sing_avoid=True, verbose=args.verbose)
            describe("reach/damped+avoid", r)
            results[("reach", "avoid")] = r
        print()

    # --- verdict -------------------------------------------------------------
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    checks = []

    h_leg, h_dam = results[("hold", "legacy")], results[("hold", "damped")]
    checks.append((
        "well-conditioned hold is unchanged by the fix",
        abs(h_leg.tau_peak - h_dam.tau_peak) < 1e-6
        and abs(h_leg.tip_err - h_dam.tip_err) < 1e-9,
        "legacy tau_peak=%.6f tip_err=%.6f vs damped tau_peak=%.6f tip_err=%.6f"
        % (h_leg.tau_peak, h_leg.tip_err, h_dam.tau_peak, h_dam.tip_err)))
    checks.append((
        "hold stays far from any singularity",
        h_dam.sigma_min > imp.SING_SIGMA_ON,
        "sigma_min=%.4f vs threshold %.4f" % (h_dam.sigma_min, imp.SING_SIGMA_ON)))

    r_leg, r_dam = results[("reach", "legacy")], results[("reach", "damped")]
    checks.append((
        "the reach scenario really does reach a singularity",
        r_leg.sigma_min < imp.SING_SIGMA_ON,
        "sigma_min=%.5f" % r_leg.sigma_min))
    checks.append((
        "damped inversion keeps the arm further from the singular set",
        r_dam.sigma_min > r_leg.sigma_min,
        "legacy is dragged to sigma_min=%.5f; damped stops at %.5f"
        % (r_leg.sigma_min, r_dam.sigma_min)))
    checks.append((
        "damped inversion bounds ||Lambda||",
        r_dam.lambda_peak < r_leg.lambda_peak,
        "legacy %.3g -> damped %.3g" % (r_leg.lambda_peak, r_dam.lambda_peak)))
    checks.append((
        "damped run spends less time saturating the torque clamp",
        r_dam.clamp_frac <= r_leg.clamp_frac + 1e-9,
        "legacy %.1f%% -> damped %.1f%%"
        % (100 * r_leg.clamp_frac, 100 * r_dam.clamp_frac)))
    checks.append((
        "the old gate only fires long after Lambda has already blown up",
        np.isfinite(r_leg.abort_lambda)
        and r_leg.abort_lambda > 10.0 * r_dam.lambda_peak,
        "legacy gate fired at t=%.2fs, by which point ||Lambda|| was already "
        "%.3g -- more than the damped run's peak of %.3g over the WHOLE run"
        % (r_leg.abort_t, r_leg.abort_lambda, r_dam.lambda_peak)))
    checks.append((
        "the damped run never has to abort on this approach at all",
        r_dam.aborted is None,
        "damped abort: %s" % (r_dam.aborted or "none")))
    checks.append((
        "damped run stays numerically sane",
        not r_dam.diverged,
        "diverged=%s" % r_dam.diverged))

    r_avo = results[("reach", "avoid")]
    checks.append((
        "null-space escape improves the closest approach",
        r_avo.sigma_min > r_dam.sigma_min,
        "damped sigma_min=%.5f -> +avoid %.5f" % (r_dam.sigma_min, r_avo.sigma_min)))

    failed = 0
    for name, ok, detail in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        print("         %s" % detail)
        if not ok:
            failed += 1
    print("\n%d/%d checks passed" % (len(checks) - failed, len(checks)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
