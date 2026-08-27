#!/usr/bin/env python3
"""Impedance control for the Kinova Gen3 7-DOF (real hardware).

Real-robot analogue of ``Kuka_impedance_control/scripts/
joint_impedance_control.py`` (a MuJoCo sim). Same two control modes, same
"one loop owns the arm" structure -- but instead of ``mujoco.mj_step`` closing
the physics loop, the physical robot does, over Kinova's low-level cyclic API.

Two modes, selected with ``--mode``:

  joint      (default) -- per-joint gravity-compensated PD spring-damper::

      tau = -Kp * (q - q_des) - Kd * dq + g(q)

  cartesian            -- operational-space impedance: a spring-damper on the
      end-effector POSE (position + orientation), mapped to joint torques with
      the Jacobian, plus a null-space posture task for the redundant 7th DOF::

      F   = Kx * (x_des - x) - Dx * xdot
      tau = Jᵀ * Λ * F  +  N * (posture PD)  +  g(q)

      (Λ = operational-space inertia, N = null-space projector -- ported from
      the Kuka ``cartesian_opspace_torque``.)

Joint mode needs only the Kortex API. Cartesian mode additionally needs a
kinematics/dynamics model (FK, Jacobian, mass matrix); it is built with PyKDL
from the arm URDF (see KinDynModel). PyKDL + urdf_parser_py are imported lazily
so joint mode still runs if they are unavailable.

WHY THIS BYPASSES ros2_control / MoveIt
---------------------------------------
The ``kortex_driver`` ros2_control hardware interface in this workspace does
NOT forward the effort command interface to the actuators -- see
``prepare_command_mode_switch`` in ``kortex_driver/src/hardware_interface.cpp``
("does not support effort command interface"). It only ever sends *position*.
So torque impedance is impossible through ros2_control here. This script talks
straight to the arm via the Kinova Kortex API in LOW_LEVEL_SERVOING with each
actuator in TORQUE control mode -- the only working path to joint torque
control on this robot.

Consequences:
  * Do NOT run this while ``robot.launch.py`` / MoveIt owns the arm. Low-level
    servoing takes exclusive control; stop the ROS2 stack first.
  * Torque mode disables the arm's built-in position safety envelope. A bad
    gain, a NaN, or a stalled loop can let the arm sag or move fast. Start with
    the (low) default gains and keep the e-stop within reach.
  * Cartesian mode adds singularity risk: near a kinematic singularity the
    Jacobian loses rank and the mapped torque blows up. The Lambda inversion is
    variable-damped on sigma_min(J) so the degenerate direction is given up
    smoothly instead of fought (see the SING_* block); ``--sing-avoid`` adds a
    null-space escape that uses the redundant 7th DOF to back away. Near a
    singularity the arm DELIBERATELY stops tracking the unreachable direction,
    so expect tip error there -- that is the trade, not a fault. Still start
    soft. Validate changes offline first with sim_impedance_mujoco.py.

GRAVITY COMPENSATION
--------------------
The MuJoCo reference used the model's ``qfrc_bias`` (exact gravity + Coriolis).
On hardware we don't have that for free:
  * ``--gravity startup`` (default): capture the *measured* load torque at
    startup -- while the arm is still high-level-held, sensed joint torque
    equals the gravity/static load at the start pose -- NEGATE it (the sensed
    value is the load, not the holding effort: see SENSED_LOAD_SIGN) and feed
    that forward as a constant. Accurate near the start pose (which is also the default target).
  * ``--gravity model``: configuration-dependent g(q) from the KinDynModel
    (available whenever the model is loaded). More accurate over large motions,
    but VALIDATE THE SIGN on hardware first (start with 'startup').
  * ``--gravity none``: diagnostic only; the arm will sag.

Run (arm up, ROS2 stack stopped)::

    # joint impedance, hold startup pose
    python3 impedance.py --robot-ip 192.168.1.10
    # cartesian impedance, hold startup EE pose (elbow floats in null space)
    python3 impedance.py --robot-ip 192.168.1.10 --mode cartesian
    # render a virtual environment at the EE (stiff wall / heavier / lighter)
    python3 impedance.py --mode cartesian --interaction-mode render --env spring
    python3 impedance.py --mode cartesian --interaction-mode render \
        --env inertia --km-scale 0.5          # feels heavier (stable dir)
    # softer / snappier via gain scales (apply to whichever mode is active)
    python3 impedance.py --kp-scale 3.5 --kd-scale 2.9
    # joint mode with integral droop-trim + a soft 2s engage, logging to CSV
    python3 impedance.py --ki-scale 0.5 --ramp-time 2 --log run.csv

OPTIONAL EXTRAS
---------------
  * ``--ki-scale`` (joint mode): a bounded integral term
    ``tau += -Ki * integral(q - q_des)`` that cancels the steady-state droop
    left by imperfect gravity comp. OFF by default; the integral torque is
    anti-windup-clamped to a fraction of each joint's torque limit, and
    accumulation is frozen per-joint while that joint is being hand-moved
    (``--ki-freeze-thresh``) so it can't wind up against the operator.
  * ``--ramp-time`` (both modes): fade the control torque from 0 to full over
    N seconds at engage while the gravity feedforward is full immediately, so
    the arm neither jumps nor sags at the moment torque mode turns on.
  * ``--reanchor-time`` (cartesian hold-ee/render, ON at 3 s): hand-guided
    setpoint teaching. While the operator pushes the tip, the task spring, the
    null-space posture term and the wrist anchor fade to ``--yield-scale`` (the
    arm goes limp under the hand; gravity comp stays full so it does not sag).
    Hold the tip still at a new pose for ``--reanchor-time`` seconds and that
    pose AND orientation become the commanded setpoint, with the posture
    reference moved to the current joint configuration so the redundant joints
    stay where they were put instead of being dragged back to q0. Capture
    happens at the current pose, so stiffness returns against ~0 error -- no
    jump. ``--reanchor-time 0`` restores the old hold-the-startup-pose-forever
    behaviour.
  * ``--log PATH`` (both modes): buffer (t, q, dq, tau, tau_g, ramp) in RAM and
    write a CSV on exit -- no file I/O in the 1 kHz loop. ``--log-decimate``
    sets the sample stride (default 10 => 100 Hz at a 1 kHz rate).
"""
import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

import numpy as np
from control_barrier import (compute_table_hocbf_rows, filter_control_qp,
                             table_wall_torque, torque_limit_rows,
                             wall_cap_clearance)

# --- protobuf 3.5.1 (shipped with kortex_api 2.6.0) compat shim for Py>=3.10 --
# kortex_api pins protobuf 3.5.1, whose containers.py references the aliases
# collections.MutableMapping etc. that were moved to collections.abc in 3.10.
# Restore them BEFORE importing anything that pulls in protobuf.
import collections
import collections.abc
for _name in ("MutableMapping", "Mapping", "Sequence", "MutableSequence",
              "Callable", "Iterable"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

# kortex_api is only installed in ~/.venvs/kortex_impedance on REAL-1 (it pins
# protobuf 3.5.1, which would shadow the system protobuf and break colcon and
# torch -- so it is deliberately absent from the laptop). Import it SOFTLY: the
# control law below (KinDynModel, cartesian_impedance_torque, the singularity
# helpers) is pure numpy/KDL and is exercised offline by the MuJoCo harness
# `sim_impedance_mujoco.py`, which must be able to import this module on a
# machine with no robot SDK. Anything that actually touches the arm goes
# through require_kortex_api() and fails there with an actionable message.
KORTEX_API_IMPORT_ERROR = None
try:
    from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.UDPTransport import UDPTransport
    from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import (ActuatorConfig_pb2, Base_pb2,
                                             BaseCyclic_pb2, Session_pb2)
except ImportError as _exc:      # no SDK here -- offline/analysis use only
    KORTEX_API_IMPORT_ERROR = _exc


def require_kortex_api():
    """Fail loudly before any robot connection if the Kortex SDK is missing."""
    if KORTEX_API_IMPORT_ERROR is not None:
        raise SystemExit(
            "kortex_api is not importable ({}).\n"
            "This script only talks to the arm from the REAL-1 venv:\n"
            "  source /home/kinova/ros2_kortex_ws/install/setup.bash\n"
            "  ~/.venvs/kortex_impedance/bin/python impedance.py ...\n"
            "Offline (no arm) use -- CSV analysis, and the MuJoCo singularity\n"
            "harness sim_impedance_mujoco.py -- does not need it."
            .format(KORTEX_API_IMPORT_ERROR))


# --- Connection --------------------------------------------------------------
TCP_PORT = 10000  # Base + ActuatorConfig services
UDP_PORT = 10001  # BaseCyclic real-time (low-level) service
DEFAULT_IP = "192.168.1.10"
DEFAULT_CREDENTIALS = ("admin", "admin")

# --- Desired joint configuration (radians) -----------------------------------
# None => hold the pose the arm is in at startup (recommended: keeps the
# constant gravity feedforward valid). Override with --q-des to spring toward a
# specific configuration (7 values, radians, in joint_1..joint_7 order).
Q_DES = None

# --- Joint-space gains -------------------------------------------------------
# Per-joint stiffness (Nm/rad) and damping (Nm*s/rad). Far lower than the Kuka
# reference: the Gen3 wrist actuators (5-7) saturate at 9 Nm, so a stiff Kp
# there instantly clamps/faults. Tuned soft-but-stable; scale live with
# --kp-scale / --kd-scale. Damping ~ 2*sqrt(Kp) unit-inertia.
KP_JOINT = np.array([30.0, 30.0, 20.0, 20.0, 8.0, 8.0, 6.0])
KD_JOINT = np.array([ 6.0,  6.0,  4.0,  4.0, 2.0, 2.0, 1.5])

# --- Joint integral trim (steady-state gravity droop) ------------------------
# Optional integral term on the joint spring (ported from the sim's
# PIDJointSpaceController Ki):  tau += -Ki * integral(q - q_des) dt.
# OFF by default (--ki-scale 0) so hardware behaviour is unchanged unless asked.
# Cancels the residual sag left by imperfect gravity comp WITHOUT stiffening the
# spring. (Most of the sag it was built for was the inverted gravity feedforward
# fixed on 2026-07-28 -- see SENSED_LOAD_SIGN -- so much less trim is needed now.) Anti-windup: the accumulated integral TORQUE (Ki*integral) is hard-
# capped per joint at I_CLAMP_FRAC * TAU_MAX, so it can never dominate the
# command or wind up while saturated. Only active in --mode joint.
KI_JOINT = np.array([8.0, 8.0, 5.0, 5.0, 2.0, 2.0, 1.5])   # Nm / (rad*s)
I_CLAMP_FRAC = 0.4          # integral torque capped at this fraction of TAU_MAX
# Conditional-integration anti-windup: freeze accumulation on any joint moving
# faster than this (rad/s) so the integral does not wind up against a hand that
# is back-driving the arm (which caused overshoot/ringing on release). Same idea
# and default as free-mode's FREE_MOVE_THRESH; the held integral is still applied
# (constant -> no windup) and resumes accumulating once the joint is released.
I_FREEZE_THRESH = 0.12      # rad/s

# --- Cartesian-space gains (operational space) -------------------------------
# Task stiffness: [x, y, z] in N/m, [rx, ry, rz] in Nm/rad. Damping is derived
# from a unit apparent inertia (the Lambda weighting makes the task ~unit-mass),
# D = 2 * zeta * sqrt(K). Conservative defaults; scale with --kp-scale/--kd-scale.
KP_CART_POS = np.array([200.0, 200.0, 200.0])   # N/m
KP_CART_ORI = np.array([20.0, 20.0, 20.0])      # Nm/rad
CART_DAMPING_RATIO = 1.0

# Null-space posture gains (redundant DOF): pull toward the startup posture q0
# without disturbing the task. Modest -- this torque is added then clamped.
KP_NULL = np.array([10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 4.0])

# Null-space avoidance gains (cartesian mode). Repulsion maps the (dimensionless)
# -collision-gradient to a joint torque; the limit barrier is Nm/rad. Bring up
# with K_REPULSE=0 to confirm cartesian mode is unchanged, then raise it.
K_REPULSE = 40.0             # Nm per unit -d(collision_cost)/dq
K_LIMIT = 25.0               # Nm/rad, joint-limit barrier stiffness

# Self-collision gradient scheduling. The gradient is a finite difference over
# all n joints, so taking it in one go costs n+1 collision_cost() calls --
# measured at 1547 us on REAL-1 (2026-08-11 perf audit), LONGER THAN A WHOLE
# CYCLE. Decimating by 10 only hid that in the average: one cycle in ten still
# ran ~2.6 ms, more than double the watchdog window, and because the stride
# matched --log-decimate 10 every logged sample landed on exactly that cycle.
#
# The sweep is now spread ACROSS cycles: one finite-difference column per cycle,
# published when the sweep completes, and the full gradient still refreshes
# every n cycles -- marginally faster than the old 10-cycle stride. Every column
# is differenced against a FROZEN configuration (collide_q), so the result is a
# true gradient at one pose instead of a mix of poses smeared over the sweep.
#
# Cost per cycle, measured on REAL-1 at the surgical home pose:
#   2026-08-11  193 us ordinary cycle / 386 us on the baseline cycle
#   2026-08-26   64 us ordinary cycle / 128 us on the baseline cycle
# after collision_cost() was vectorised over capsule pairs (see
# _segment_distance_batch). Identical costs to 1.4e-17; identical gradients to
# 1.4e-13. This was still ~66% of all control compute before the change.
COLLIDE_EPS = 1e-4           # finite-difference step (rad); was collision_gradient's default

# Cartesian safety limits.
MAX_CART_FORCE = 80.0        # N   -- clamp task force magnitude per axis
MAX_CART_TORQUE = 15.0       # Nm  -- clamp task moment per axis
MIN_MANIPULABILITY = 1e-3    # sqrt(det(J Jᵀ)); logged only -- see below

# --- Singularity handling (cartesian mode) -----------------------------------
# Distance to a kinematic singularity is sigma_min(J), the SMALLEST singular
# value of the 6xn Jacobian -- not sqrt(det(J Jᵀ)) and not det(J M⁻¹ Jᵀ), which
# is what this file gated on until 2026-08-18. Both determinants are PRODUCTS of
# all six singular values, so the five healthy directions (O(1) each) mask the
# one that is collapsing. Measured on the bundled gen3_2f85 chain, sweeping the
# elbow to straight (joint_4 -> 0, the arm's dominant singularity):
#
#   joint_4   sigma_min   manip=sqrt(det(JJᵀ))   det(J M⁻¹ Jᵀ)   ||Lambda||
#   -0.50      0.0711        2.1e-2               2.9e+06          8.9
#   -0.20      0.0289        6.9e-03              5.0e+05         41.5
#   -0.05      0.0073        1.5e-03              2.7e+04        611
#   -0.01      0.0015        2.9e-04              1.1e+03        1.5e+04
#    0.00      1.0e-05       1.3e-06              2.1e-02        7.8e+08
#
# Two failures follow directly from that table:
#
#   * The Lambda conditioning test `abs(det(Lambda_inv)) >= 1e-2` was still TRUE
#     at the exact singularity (2.06e-2 > 1e-2), so the code took the plain
#     np.linalg.inv branch in the one configuration where it must not, and
#     produced ||Lambda|| = 7.8e8. The pinv fallback was effectively dead code.
#   * The abort gate `manip < 1e-3` still read 1.49e-3 (i.e. PASSING) at
#     joint_4 = -0.05, where ||Lambda|| had already reached 611 -- ~150x nominal.
#     The guard fired only after the torque spike it was meant to prevent.
#
# The replacement is continuous, because a discontinuous switch on a 1 kHz
# torque command is itself a hazard:
#
#   1. Variable-damped inversion (Chiaverini). Lambda_inv is symmetric PSD, so
#      eigh gives Lambda = sum_i (1/e_i) u_i u_iᵀ exactly. Each mode is inverted
#      as e/(e² + lam²) instead of 1/e, which caps the gain of a degenerate
#      direction at 1/(2*lam) rather than letting it run to infinity. lam ramps
#      in smoothly only once sigma_min < SING_SIGMA_ON, so in the validated
#      hold-ee regime (sigma_min ~ 0.07-0.22) lam is exactly 0 and the torque is
#      bit-for-bit what it was before this change.
#   2. No global gain rolloff. The damping already suppresses ONLY the collapsing
#      direction; the five healthy task directions keep full authority, which is
#      the whole point of doing this in the eigenbasis instead of scaling F.
#   3. Null-space escape (opt-in, --sing-avoid). The Gen3 is redundant, so the
#      arm can retreat from the singular set without moving the tool tip: ascend
#      the sigma_min gradient inside the existing null-space projector.
#   4. Abort only on true rank collapse, gated on sigma_min, with a rate-limited
#      warning above it. Aborting drops to the finally-block that restores
#      POSITION mode -- it is a controlled stop, but it ends the run, so it must
#      be the last resort and not the first response.
#
# CAVEAT: J mixes units (m/rad in rows 0-2, dimensionless in rows 3-5), so
# sigma_min is not a physically clean length. The thresholds are therefore
# EMPIRICAL for this chain, taken from the sweep above; re-measure them with
# tools/probe_singularity.py if the tip link or URDF changes.
SING_SIGMA_ON = 0.05         # sigma_min below which DLS damping ramps in
SING_SIGMA_ABORT = 0.002     # sigma_min floor: rank collapse, stop the loop
SING_LAMBDA_MAX = 0.01       # max damping -> caps ||Lambda|| at 1/(2*lam) = 50
SING_WARN_PERIOD = 1.0       # s, rate limit on the proximity warning
K_SING = 30.0                # Nm per unit d(sigma_min)/dq, null-space escape
SING_GRAD_EPS = 1e-3         # finite-difference step (rad) for that gradient
MAX_SING_TORQUE = 8.0        # Nm, per-joint clamp on the null-space escape

# --- (B) Virtual-environment rendering (from virtu-phys-sim controller.c) -----
# In interaction-mode 'render' the EE renders a task-space virtual environment.
# IMPORTANT (learned on hardware 2026-07-20): render is built ON TOP of the
# proven operational-space impedance (`cartesian_impedance_torque`) -- i.e. with
# Lambda weighting AND null-space damping. A naive direct tau=Jᵀ(Kk*x_err) port
# of controller.c (which runs in MuJoCo with EXACT gravity) was too soft to hold
# the real arm against imperfect gravity comp and, with no null-space authority,
# the redundant DOF + limitless continuous joints collapsed the arm into the
# table. So the preset stiffness is expressed in the SAME (Lambda-weighted) units
# as hold-ee's KP_CART_*, and the virtual MASS is added as an extra raw wrench.
#
#   tau = Jᵀ*( Lambda*(kp*x_err - kd*xdot) + Km*acc ) + nullspace(damping+guards)
#
# Km SIGN (critical): Km < 0 ADDS apparent inertia -> arm feels HEAVIER; negative
# acceleration feedback, STABLE. Km > 0 is mass-reduction -> feels LIGHTER, but is
# POSITIVE feedback on a noisy hardware accel estimate and can go UNSTABLE --
# bring --km-scale up slowly, e-stop in hand. Virtual mass is translation-only
# (Km rot = 0). Presets carry Lambda-weighted stiffness `kp` [x,y,z,rx,ry,rz]
# (damping derived like the cartesian gains) and `km` (virtual mass).
ENV_PRESETS = {
    # spring: stiff wall toward the startup pose (= hold-ee gains), no mass. SAFE.
    "spring":  dict(kp=np.array([200.0, 200.0, 200.0, 20.0, 20.0, 20.0]),
                    km=np.zeros(6)),
    # inertia: still holds (moderate stiffness) but feels heavier. Stable dir.
    "inertia": dict(kp=np.array([120.0, 120.0, 120.0, 20.0, 20.0, 20.0]),
                    km=np.array([-4.0, -4.0, -4.0, 0.0, 0.0, 0.0])),
    # passive: soft centering (still can't collapse) + lighter. Risky mass dir.
    "passive": dict(kp=np.array([60.0, 60.0, 60.0, 20.0, 20.0, 20.0]),
                    km=np.array([1.5, 1.5, 1.5, 0.0, 0.0, 0.0])),
}
# Virtual-mass safety: the task-acceleration estimate is a low-passed finite
# difference of xdot (=J*dq) using the ACTUAL loop dt. Heavy filtering (EMA) tames
# the double-derivative noise; the estimate and the mass force are both clamped.
ACC_FILTER_ALPHA = 0.9       # EMA weight on previous accel (higher = smoother)
MAX_TASK_ACC = np.array([8.0, 8.0, 8.0, 40.0, 40.0, 40.0])   # m/s^2 , rad/s^2

# --- Kinematics/dynamics model (cartesian mode) ------------------------------
# URDF providing the arm chain. The bundled gen3 URDF namespaces links with a
# "gen3_" prefix; that is fine -- we hold a RELATIVE startup pose, so absolute
# frame naming is irrelevant, and the arm inertias/kinematics match the real
# arm regardless of the (2f85 vs 2f140) gripper beyond the tip link.
# Bundled in this package's config/ so cartesian mode is self-contained on any
# machine (only the XML is parsed -- PyKDL never loads the referenced meshes).
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(_PKG_DIR, "config", "gen3_2f85.urdf")
DEFAULT_BASE_LINK = "gen3_base_link"
DEFAULT_TIP_LINK = "gen3_end_effector_link"

# Links guarded by the table-avoidance barrier. The distal links do the real
# work (they are what reaches down), but the forearm and half_arm_2 are
# included because an elbow-down posture can swing them at the table while the
# tool tip is nowhere near it. Upper-arm links are omitted: they cannot reach
# the table without the whole arm already being in violation.
def _resolve_table_links(args, model):
    """Resolve --table-links into concrete chain link names.

    Accepts the 'all' / 'tip' shorthands, defaults to DEFAULT_TABLE_LINKS
    filtered to what this chain actually has (so a different URDF or tip link
    does not hard-fail), and validates anything explicitly named.
    """
    if not args.table_avoidance:
        return []

    on_chain = model.segment_names()
    sel = args.table_links

    if sel is None:
        names = [nm for nm in DEFAULT_TABLE_LINKS if nm in on_chain]
        if not names:
            # Different URDF/namespace: fall back to guarding the tip alone
            # rather than silently guarding nothing.
            names = [on_chain[-1]]
        return names

    if len(sel) == 1 and sel[0] == "all":
        return list(on_chain)
    if len(sel) == 1 and sel[0] == "tip":
        return [args.tip_link]

    unknown = [nm for nm in sel if nm not in on_chain]
    if unknown:
        raise SystemExit(
            f"--table-links: unknown link(s) {', '.join(unknown)}.\n"
            f"chain links: {', '.join(on_chain)}")
    # Keep base->tip order so the console report reads consistently.
    return [nm for nm in on_chain if nm in sel]


DEFAULT_TABLE_LINKS = [
    "gen3_half_arm_2_link",
    "gen3_forearm_link",
    "gen3_spherical_wrist_1_link",
    "gen3_spherical_wrist_2_link",
    "gen3_bracelet_link",
    "gen3_end_effector_link",
]

# --- Common torque limits / conventions --------------------------------------
# Per-joint torque saturation (Nm). Gen3 7-DOF: big joints 39 Nm, wrist 9 Nm.
# Command is hard-clamped to +/-TAU_MAX for safety regardless of gains/mode.
TAU_MAX = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])

# Sensed actuator torque -> holding effort. CORRECTED 2026-07-28 after the arm
# drooped the instant torque mode engaged (see below); this was +1.0 before.
#
# The Gen3 reports each actuator's torque as the external LOAD on the joint, not
# the effort needed to hold against it. Feeding it forward unchanged therefore
# commands torque in the direction gravity is ALREADY pulling -- the arm engages
# carrying ~2x its own weight and sinks. Three independent checks agree:
#
#   1. Ground truth from potential energy, tau_hold = +dU/dq with
#      U = sum_k m_k*g*z_com,k(q) (convention-free, see hold_torque_energy):
#      at q0 = [0,15,180,-130,0,55,90] deg it is [-0, -8.14, -0.09, +4.04, -0,
#      +0.76, 0] Nm -- the exact NEGATIVE of the sensed [+0.06, +8.36, +0.49,
#      -4.77, ...] on every loaded joint, magnitudes matching to 2 decimals.
#   2. Hardware, /tmp/imp_trim1.csv: at t=0 the loop commanded tau2=+8.36 (the
#      sensed value) and the arm immediately accelerated DOWNWARD at 0.38 rad/s;
#      q2 rose 15->32 deg.
#   3. The task spring arrested that fall by driving tau2 to -24.8 Nm, i.e. the
#      controller had to fight the feedforward to lift -- negative IS up on j2.
#
# This is why joint mode needed --ki-scale to cancel "steady-state droop", why
# 'free' needed KP_HOLD because "pure gravity comp cannot hold the arm", and why
# only a stiff --cart-scale held the arm: the spring was carrying 2x gravity.
SENSED_LOAD_SIGN = -1.0

# Loop rate. Kinova low-level servoing expects ~1 kHz; too slow and the actuator
# watchdog faults.
#
# MEASURED REALITY (2026-08-11 perf audit, 12 archived runs / 944 s of hardware
# time): the loop achieves 833 Hz (1.200 ms), NOT the 1000 Hz asked for, and has
# done so on every run since 2026-07-30. Compute is not the reason -- it is
# 93-310 us depending on mode, against ~890 us for the blocking
# base_cyclic.Refresh() round-trip. Since compute + comms already exceeds the
# 1 ms budget, the sleep pacer at the bottom of the loop never runs at all and
# the loop free-runs at whatever the transport allows. Lowering --rate will not
# make it faster and raising it will not either; the only levers are
# transport-side. The watchdog evidently tolerates 833 Hz.
#
# 2026-08-26 follow-up, measured against the live arm:
#   * Cartesian+barrier compute was ~298 us, of which collision_cost alone was
#     196 us. Vectorising it (see COLLIDE_EPS) took compute to ~166 us.
#   * The comms floor is genuinely arm-side, not host-side. Read-only
#     RefreshFeedback round-trips at 435 us mean / 834 us max over 3000 frames.
#     Two candidate host-side causes were tested and RULED OUT: pure-Python
#     protobuf (serialize 12 us + parse 72 us, and forcing the cpp backend just
#     crashes on 3.5.1 descriptors), and CPU C-states/governor (REAL-1 idles at
#     'powersave' with a 1048 us C3 exit latency, but pinning all cores awake at
#     5.5 GHz moved the round trip 408 -> 413 us, i.e. not at all).
#   * The remaining ~480 us gap between Refresh and RefreshFeedback is the arm
#     consuming the command at its own cycle boundary. The only lever left is to
#     stop blocking on it: RouterClientSendOptions(andForget=True) and the
#     send-only RefreshCommand() both exist on this API and are unused.
#
# CONFIRMED ON HARDWARE (2026-08-26, two 30 s cartesian hold-ee runs). Note the
# 833 Hz figure above is STALE -- by 2026-08-12 the loop was already at 934 Hz:
#
#   Aug 06   1.223 ms   817 Hz    p99 1.580   max 1.680
#   Aug 12   1.070 ms   934 Hz    p99 1.600   max 2.040
#   Aug 26   1.055 ms   947 Hz    p99 1.246   max 1.460   <- collision_cost vectorised
#
# Cutting 132 us of compute bought only 15 us of mean cycle time. That is the
# point: the loop is NOT compute-bound. It blocks in Refresh until the arm
# services the command on its own ~1 kHz tick, so once total work fits under one
# tick, further compute savings are absorbed into a longer blocking wait rather
# than a faster loop. What the saving DOES buy is tail margin -- p99 -22%, max
# -28% -- i.e. fewer cycles overshoot the tick and slip to the next one, which
# is what the watchdog and the dt_meas-driven integrators actually care about.
# Do not expect further compute optimisation to raise the rate.
#
# What this DID break, until fixed on 2026-08-11: every integrator advanced by a
# nominal 1/rate per cycle while only 833 cycles happened per second, so the
# gravity trim, the joint integral trim and the re-anchor blend all integrated
# at 0.833x their configured gain. They now use the MEASURED cycle time.
DEFAULT_RATE_HZ = 1000.0

# Bounds on the measured cycle time fed to the integrators. The loop genuinely
# does hit ~2.6x nominal on a bad cycle, so the ceiling has to allow real spikes
# through (clamping them would under-integrate exactly when time is passing
# fastest); it exists only to stop a scheduling stall or a clock anomaly from
# injecting one enormous integration step into a torque command. The floor
# guards the first cycle and any zero/negative interval.
DT_MEAS_MIN_FACTOR = 0.1     # x nominal dt
DT_MEAS_MAX_FACTOR = 5.0     # x nominal dt

# Slow-cycle warning threshold, as a multiple of the nominal period. This was
# 3.0, which at --rate 1000 meant nothing was reported below 3 ms -- so a loop
# permanently running at 1.2 ms (20% over budget) never printed a single warning
# and the shortfall went unnoticed for weeks. 1.5 still clears an ordinary
# cycle but catches a genuine stall. The warning stays rate-limited to one line
# every 2 s, and the exit summary reports the achieved rate unconditionally.
SLOW_CYCLE_WARN_FACTOR = 1.5

# Cyclic-frame send timeout (ms). The Kortex default is timeout_ms=10000, and
# impedance.py never passed a RouterClientSendOptions, so it inherited it: ONE
# lost UDP frame blocked the torque loop for TEN SECONDS with the arm in
# LOW_LEVEL_SERVOING and its position safety envelope disabled. The `finally`
# that restores POSITION mode cannot run until that call returns, so the failure
# mode was "arm holds whatever torque it last got, and the recovery path is
# unreachable". Kinova's own low-level examples use 3 ms.
#
# BaseCyclicClient.Refresh waits on a concurrent.futures.Future with this as the
# deadline, so an expiry raises FutureTimeoutError -- caught in the loop below.
# The achieved period is ~1.2 ms (see DEFAULT_RATE_HZ), so 3 ms absorbs a merely
# LATE frame while still failing fast on a lost one.
CYCLIC_TIMEOUT_MS = 3.0
# Consecutive timed-out frames tolerated before aborting. A single drop on a
# 50k-frame run should not end the run; a persistent stall must. Each retry
# recomputes from the last good feedback, so this is bounded staleness
# (3 x ~1.2 ms) rather than an open-ended hang.
MAX_CYCLIC_TIMEOUTS = 3

# Safety envelope: abort if any joint spins faster than this (rad/s).
MAX_JOINT_SPEED = 1.5

# Global joint damping (both modes): a small velocity-proportional torque added
# to the TOTAL command on every joint. Zero when static (does not fight holding
# or the impedance spring), but bleeds off slow drift on low-friction /
# uncontrolled DOFs -- notably the continuous wrist joints 5 & 7, which have no
# null-space restoring authority. Scaled by --kd-scale.
KD_GLOBAL = np.array([2.0, 2.0, 1.5, 1.5, 0.8, 0.8, 0.6])

# Continuous (limitless) joints of the Gen3 7-DOF: 1,3,5,7 (see Q_MIN/Q_MAX).
# These have NO joint-limit barrier and, in cartesian mode, can fall into a
# control dead-zone: the Lambda-weighted task torque maps the EE correction onto
# other joints (giving ~0 on the wrist roll j7), AND the null-space posture term
# toward q0 is annihilated by the null-space projector wherever the joint is not
# in the task null-space. Left with only global damping -- which sets a nonzero
# terminal drift VELOCITY under any residual, never zero -- the wrist creeps
# indefinitely (diagnosed on hardware 2026-07-24, cartesian render). The wrist
# anchor (--wrist-anchor) is a small DIRECT joint-space spring toward q0 on these
# joints, added to the TOTAL command OUTSIDE the null-space projector, so it can
# reach a joint the projector forbids. The driving bias is tiny (~0.02 Nm), so a
# few Nm/rad holds the joint within <1 deg while barely perturbing the EE task.
CONTINUOUS_JOINTS = np.array([True, False, True, False, True, False, True])

# 'free' interaction-mode position hold. 'free' captures a joint-space hold pose
# whenever the arm is (nearly) stationary and applies a light critically-damped
# spring toward it: while |dq| exceeds FREE_MOVE_THRESH the operator is moving it
# (spring OFF, fully fluid); below it the arm holds where released.
# NOTE (2026-07-28): this was originally sized to compensate for gravity comp
# that "under-supports away from q0 and the low-friction arm sinks" -- that was
# the inverted feedforward (see SENSED_LOAD_SIGN), not model error. With the sign
# fixed, gravity comp alone should very nearly hold the arm, so KP_HOLD can
# probably be reduced a lot (--free-hold-scale) for a more fluid hand-guide.
KP_HOLD = np.array([25.0, 25.0, 20.0, 20.0, 12.0, 12.0, 10.0])   # Nm/rad
FREE_MOVE_THRESH = 0.12         # rad/s -- above => "being moved", below => hold

# --- Hand-guided re-anchoring (cartesian hold-ee / render) -------------------
# The cartesian task spring holds the STARTUP EE pose forever, so repositioning
# the arm by hand means fighting it the whole way and losing it on release. With
# re-anchoring the loop watches the tip: while it is being pushed the task
# spring (and the posture/wrist anchors) fade down to YIELD_SCALE -- the arm goes
# limp under the hand, gravity comp still full so it does not sag -- and once the
# tip has been held (nearly) still at a NEW pose for REANCHOR_HOLD_TIME seconds
# the desired position AND orientation are re-captured there, the posture
# reference is moved to the current joint config (so the elbow/redundant DOF is
# left where the operator put it rather than being dragged back to q0), and full
# stiffness fades back in. Capture happens AT the current pose, so the task error
# is ~0 when stiffness returns -- no jump on re-engage.
# Speed thresholds are deliberately WELL ABOVE the arm's own gravity creep. With
# them too tight (first hardware run, 2026-07-28: 0.02 m/s) the loop is unstable:
# residual droop reads as "hand motion" -> spring drops to YIELD_SCALE -> the arm
# droops FASTER -> never goes still -> never re-anchors. That run sat at 15%
# stiffness for 98.9% of 60 s and the arm collapsed into the table. YIELD_TIMEOUT
# is the hard backstop for that failure mode regardless of threshold tuning.
REANCHOR_HOLD_TIME = 3.0      # s of stillness at a new pose before capture
REANCHOR_LIN_SPEED = 0.04     # m/s   -- EE linear speed above this = "being moved"
REANCHOR_ANG_SPEED = 0.25     # rad/s -- EE angular speed above this = ditto
REANCHOR_MIN_POS = 0.02       # m   -- min displacement to count as a NEW pose
REANCHOR_MIN_ORI = 0.10       # rad -- min rotation to count as a NEW pose
YIELD_SCALE = 0.15            # task/posture stiffness fraction while hand-moved
YIELD_BLEND_TIME = 0.4        # s to fade stiffness between yielded and full
YIELD_TIMEOUT = 8.0           # s of continuous yielding -> force full stiffness

# --- Gravity-holding authority (independent of stiffness) --------------------
# Carrying the arm's WEIGHT is the gravity feedforward's job, not the task
# spring's -- a spring stiff enough to hold the arm up is also a spring that
# fights the operator (exactly the "too stiff" / "droops when soft" tradeoff).
# GRAVITY_TRIM is a bounded joint-space integral of the residual droop:
#
#     tau_trim += k_trim * (q_ref - q) * dt      (clamped to +/-I_CLAMP_FRAC*TAU_MAX)
#
# It accumulates ONLY while the arm is nearly still and not being hand-moved, and
# it is applied at FULL authority regardless of the yield scaling -- so the arm
# holds its height even while it is limp under your hand. Same gains and
# anti-windup as the joint-mode --ki-scale trim, reused here for cartesian mode.
GRAVITY_TRIM_SCALE = 0.6      # default --gravity-trim (0 = off, legacy)
TRIM_FREEZE_THRESH = 0.12     # rad/s -- above this, freeze accumulation

# Slow-collapse guard: MAX_JOINT_SPEED only catches a RUNAWAY. A soft-mode droop
# creeps below it (the collapse above peaked at ~0.7 rad/s but averaged far less)
# and ends with the arm on the table. Abort if the EE ever sits this far from its
# commanded position -- by then the pose is unrecoverable by the spring anyway.
MAX_POSE_ERROR = 0.30         # m, while the arm is holding
# Ceiling while the operator IS guiding, where distance from the (stale)
# setpoint is intentional. Still bounded: exempting a guided move entirely let a
# simulated free-fall travel 3 m before the yield timeout noticed. 0.80 m clears
# a 90 deg base-yaw reposition (0.65 m of EE arc) but catches a fall in ~5 s.
MAX_POSE_ERROR_GUIDED = 0.80  # m


# =============================================================================
# Kinematics / dynamics model (PyKDL) -- only used in cartesian mode
# =============================================================================
class KinDynModel:
    """FK, Jacobian, mass matrix and gravity for the arm chain, via PyKDL.

    PyKDL + urdf_parser_py are imported here (lazily) so that joint mode does
    not require them. kdl_parser_py is not packaged for this distro, so the
    URDF->KDL tree conversion is vendored below (from kdl_parser_py).
    """

    # --- Self-collision capsule model (cartesian-mode redundancy resolution) --
    # The arm is approximated as a polyline of equal-radius capsules (one per
    # link). A reactive safety cushion, NOT a certified collision checker; err
    # large on the radius -- the torque clamp bounds the worst case anyway.
    CAPSULE_RADIUS = 0.055      # m, uniform link "thickness"
    COLLISION_MARGIN = 0.08     # m, potential switches on below this clearance

    def __init__(self, urdf_path, base_link, tip_link):
        import PyKDL as kdl
        from urdf_parser_py.urdf import URDF
        self.kdl = kdl

        robot = URDF.from_xml_file(urdf_path)
        tree = self._tree_from_urdf(robot, kdl)
        self.chain = tree.getChain(base_link, tip_link)
        self.n = self.chain.getNrOfJoints()

        self._fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self._jac = kdl.ChainJntToJacSolver(self.chain)
        self._dyn = kdl.ChainDynParam(self.chain, kdl.Vector(0, 0, -9.81))

        # Jacobian time-derivative solver, used by the table barrier for the
        # exact Coriolis term. HYBRID = base-frame axes with the reference
        # point at the segment tip, matching self.jacobian()'s convention.
        self._jdot = kdl.ChainJntToJacDotSolver(self.chain)
        self._jdot.setRepresentation(kdl.ChainJntToJacDotSolver.HYBRID)

        # Scratch objects reused by the hot-loop barrier queries, so a 1 kHz
        # call does not allocate a KDL object per link per cycle.
        self._scratch_frame = kdl.Frame()
        self._scratch_jac = kdl.Jacobian(self.n)
        self._scratch_twist = kdl.Twist()
        self._scratch_qv = kdl.JntArrayVel(self.n)
        self._scratch_cor = kdl.JntArray(self.n)
        # Two slots, because `coriolis` needs q and dq live at the same time.
        self._scratch_jnt = (kdl.JntArray(self.n), kdl.JntArray(self.n))
        # Non-adjacent capsule pair indices, keyed by capsule count.
        self._pair_cache = {}

    # -- vendored from kdl_parser_py (treeFromUrdfModel), rotation of the
    #    inertial origin dropped (URDF inertials here are axis-aligned) --------
    @staticmethod
    def _tree_from_urdf(robot, kdl):
        def to_pose(p):
            if p and p.rpy and len(p.rpy) == 3 and p.xyz and len(p.xyz) == 3:
                return kdl.Frame(kdl.Rotation.RPY(*p.rpy), kdl.Vector(*p.xyz))
            return kdl.Frame()

        def to_joint(j):
            f = to_pose(j.origin)
            if j.type in ("revolute", "continuous"):
                return kdl.Joint(j.name, f.p, f.M * kdl.Vector(*j.axis),
                                 kdl.Joint.RotAxis)
            if j.type == "prismatic":
                return kdl.Joint(j.name, f.p, f.M * kdl.Vector(*j.axis),
                                 kdl.Joint.TransAxis)
            return kdl.Joint(j.name, kdl.Joint.Fixed)

        def to_rbi(inertial):
            o = to_pose(inertial.origin)
            i = inertial.inertia
            rot = kdl.RotationalInertia(i.ixx, i.iyy, i.izz, i.ixy, i.ixz, i.iyz)
            return kdl.RigidBodyInertia(inertial.mass, o.p, rot)

        def add_children(root, tree):
            for jnt in robot.joints:
                if jnt.parent != root:
                    continue
                link = robot.link_map[jnt.child]
                inert = (to_rbi(link.inertial) if link.inertial is not None
                         else kdl.RigidBodyInertia())
                seg = kdl.Segment(jnt.child, to_joint(jnt), to_pose(jnt.origin),
                                  inert)
                tree.addSegment(seg, root)
                add_children(jnt.child, tree)

        tree = kdl.Tree(robot.get_root())
        add_children(robot.get_root(), tree)
        return tree

    def _to_jnt(self, q, slot=0):
        """Copy `q` into a preallocated JntArray and return it.

        Reuses a scratch array instead of allocating one: this runs ~7 times
        per 1 kHz cycle and the allocation was ~30% of its cost.

        The returned object is SHARED and is overwritten by the next call with
        the same `slot`. A call site that needs two JntArrays alive at once --
        `coriolis`, which hands q and dq to a single solver call -- must ask
        for different slots, or both arguments alias the same data and the
        solver silently sees dq twice.
        """
        ja = self._scratch_jnt[slot]
        for i in range(self.n):
            ja[i] = float(q[i])
        return ja

    def fk(self, q):
        """Return (position [3], quaternion [x,y,z,w]) of the tip in base frame."""
        frame = self.kdl.Frame()
        self._fk.JntToCart(self._to_jnt(q), frame)
        p = np.array([frame.p[i] for i in range(3)])
        quat = np.array(frame.M.GetQuaternion())  # (x, y, z, w)
        return p, quat

    def jacobian(self, q):
        """Return the 6xN geometric Jacobian (base frame, tip reference)."""
        J = self.kdl.Jacobian(self.n)
        self._jac.JntToJac(self._to_jnt(q), J)
        return np.array([[J[r, c] for c in range(self.n)] for r in range(6)])

    def segment_index(self, name):
        """1-based segment index whose tip is link `name` (as JntToCart wants).

        Segment s spans joint s, so FK/Jacobian with seg_nr = s returns the
        pose/Jacobian of the tip of the s-th segment. Index 0 is the chain root.
        """
        for s in range(self.chain.getNrOfSegments()):
            if self.chain.getSegment(s).getName() == name:
                return s + 1
        raise ValueError(
            f"link '{name}' is not on the chain; available: "
            + ", ".join(self.chain.getSegment(s).getName()
                        for s in range(self.chain.getNrOfSegments())))

    def segment_names(self):
        """Names of every link on the chain, base -> tip."""
        return [self.chain.getSegment(s).getName()
                for s in range(self.chain.getNrOfSegments())]

    def height_barrier_terms(self, q, dq, seg_indices):
        """Per-link (z, J_z, dJdq_z) for the table barrier.

        Returns only the vertical row of each Jacobian and the z-component of
        the Jacobian-derivative product -- the barrier needs nothing else, and
        materialising full 6xN matrices per link at 1 kHz is pure waste.

        Args:
            q, dq (np.ndarray): joint position/velocity (n,)
            seg_indices (sequence[int]): segment indices from segment_index()

        Returns:
            z (np.ndarray): (m,) link-tip heights in base frame
            J_z (np.ndarray): (m, n) vertical Jacobian rows
            dJdq_z (np.ndarray): (m,) z-component of dJ @ dq (exact, from KDL)
        """
        m = len(seg_indices)
        z = np.empty(m)
        J_z = np.zeros((m, self.n))
        dJdq_z = np.empty(m)

        ja = self._to_jnt(q)
        # JntArrayVel bundles q and dq for the JacDot solver.
        qv = self._scratch_qv
        for i in range(self.n):
            qv.q[i] = float(q[i])
            qv.qdot[i] = float(dq[i])

        frame, jac, twist = (self._scratch_frame, self._scratch_jac,
                             self._scratch_twist)
        for k, s in enumerate(seg_indices):
            self._fk.JntToCart(ja, frame, s)
            z[k] = frame.p[2]

            self._jac.JntToJac(ja, jac, s)
            for c in range(self.n):
                J_z[k, c] = jac[2, c]

            # Twist overload returns the dJ @ dq product directly, so the full
            # 6xN derivative matrix is never formed.
            self._jdot.JntToJacDot(qv, twist, s)
            dJdq_z[k] = twist.vel[2]

        return z, J_z, dJdq_z

    def mass(self, q):
        """Return the NxN joint-space inertia matrix M(q)."""
        M = self.kdl.JntSpaceInertiaMatrix(self.n)
        self._dyn.JntToMass(self._to_jnt(q), M)
        return np.array([[M[r, c] for c in range(self.n)] for r in range(self.n)])

    def coriolis(self, q, dq):
        """Return the Coriolis/centrifugal torque C(q, dq) @ dq (n,).

        The velocity-dependent term of ``M ddq + C dq + g = tau``. Needed by
        the table barrier: it inverts the commanded torque to a nominal
        acceleration and back, and dropping C leaves that round trip wrong by
        ``Minv @ C dq`` -- the same order as the dJ@dq term the HOCBF rows go
        out of their way to get exactly right.

        KDL returns the PRODUCT C(q,dq) @ dq, not the matrix, which is all the
        inversion needs and avoids forming an n x n matrix at 1 kHz.
        """
        c = self._scratch_cor
        # Distinct slots: a shared scratch array would make both arguments the
        # same object, so the solver would receive dq as the position too.
        self._dyn.JntToCoriolis(self._to_jnt(q, 0), self._to_jnt(dq, 1), c)
        return np.array([c[i] for i in range(self.n)])

    def gravity(self, q):
        """Return the N-vector of hold-against-gravity torques g(q).

        KDL's JntToGravity already returns the gravity term in the convention
        ``tau_hold = G(q)`` (static form of ``M qdd + C + G = tau``), so it is
        used AS-IS. It was negated here until 2026-07-28 to "match Kinova's
        sensed holding-torque sign" -- but the sensed torque is the external
        LOAD, the opposite of the holding effort (see SENSED_LOAD_SIGN), so
        matching it inverted the model too. Cross-checked against the
        convention-free energy gradient in ``hold_torque_energy``: they now
        agree to <1e-3 Nm on every joint.
        """
        g = self.kdl.JntArray(self.n)
        self._dyn.JntToGravity(self._to_jnt(q), g)
        return np.array([g[i] for i in range(self.n)])

    def hold_torque_energy(self, q, eps=1e-6):
        """Holding torque from the potential-energy gradient, tau = +dU/dq.

        Ground truth for the gravity sign, independent of every KDL/Kinova sign
        convention: U(q) = sum_k m_k * g * z_com,k(q) over the chain's links, so
        the torque that holds the arm static is its gradient. Slow (2n FK
        sweeps) -- startup validation only, never called in the control loop.
        """
        def U(qq):
            ja = self._to_jnt(qq)
            frame = self.kdl.Frame()
            tot = 0.0
            for s in range(self.chain.getNrOfSegments()):
                self._fk.JntToCart(ja, frame, s + 1)
                inertia = self.chain.getSegment(s).getInertia()
                mass = inertia.getMass()
                if mass <= 0.0:
                    continue
                c = inertia.getCOG()
                w = frame * self.kdl.Vector(c[0], c[1], c[2])
                tot += mass * 9.81 * w[2]
            return tot

        g = np.zeros(self.n)
        for i in range(self.n):
            a = np.array(q, dtype=float); a[i] += eps
            b = np.array(q, dtype=float); b[i] -= eps
            g[i] = (U(a) - U(b)) / (2.0 * eps)
        return g

    def centerline(self, q):
        """De-duplicated world positions of segment origins, base->tip.

        The polyline connecting these points is the capsule model's spine;
        consecutive points define one link capsule each.

        Returns an (m, 3) array. The de-duplication threshold compares
        consecutive segment ORIGINS, whose separation is fixed by the URDF joint
        offsets and does not depend on q, so `m` is constant for a given chain
        -- but `_capsule_pairs` keys its cache on it rather than assuming so.
        """
        ja = self._to_jnt(q)
        frame = self.kdl.Frame()
        pts = []
        for s in range(self.chain.getNrOfSegments() + 1):
            self._fk.JntToCart(ja, frame, s)
            p = np.array([frame.p[i] for i in range(3)])
            if not pts or np.linalg.norm(p - pts[-1]) > 0.02:
                pts.append(p)
        return np.asarray(pts)

    def _capsule_pairs(self, ncap):
        """Cached (i, j) index arrays for the non-adjacent capsule pairs.

        Adjacent capsules share a joint and always touch, so only j >= i + 2 is
        checked. The pair set depends solely on the capsule COUNT, so it is
        built once and reused every cycle instead of being re-enumerated.
        """
        idx = self._pair_cache.get(ncap)
        if idx is None:
            pairs = [(i, j) for i in range(ncap) for j in range(i + 2, ncap)]
            if pairs:
                ii, jj = (np.array(x, dtype=np.intp) for x in zip(*pairs))
            else:
                ii = jj = np.empty(0, dtype=np.intp)
            idx = (ii, jj)
            self._pair_cache[ncap] = idx
        return idx

    def collision_cost(self, q):
        """Quadratic-hinge self-collision potential over non-adjacent capsules.

        Rises smoothly from zero as any non-adjacent link pair's surface
        clearance drops below COLLISION_MARGIN. Adjacent capsules share a joint
        (always touch), so they are skipped.

        Vectorised over all pairs at once (see `_segment_distance_batch`); the
        equivalent Python double loop was 66% of the control cycle's compute.
        """
        pts = self.centerline(q)
        if len(pts) < 3:
            return 0.0
        P, Q = pts[:-1], pts[1:]
        ii, jj = self._capsule_pairs(len(P))
        if ii.size == 0:
            return 0.0
        surf = (_segment_distance_batch(P[ii], Q[ii], P[jj], Q[jj])
                - 2.0 * self.CAPSULE_RADIUS)
        # Quadratic hinge: only clearances inside the margin contribute.
        v = np.maximum(0.0, self.COLLISION_MARGIN - surf)
        return 0.5 * float(v @ v)

    def collision_gradient(self, q, eps=1e-4):
        """-d(collision_cost)/dq direction points toward greater clearance.

        Finite differences: costs n+1 centerline evaluations -- 1547 us on
        REAL-1, longer than a whole control cycle.

        NOT USED BY THE CONTROL LOOP. The loop sweeps the same finite difference
        one column per cycle instead (see COLLIDE_EPS), because decimating a
        call this heavy bounds its average cost but not its worst cycle. Kept
        for offline validation and tests, where taking it in one go is fine.
        """
        g = np.zeros(self.n)
        h0 = self.collision_cost(q)
        for i in range(self.n):
            dq = q.copy()
            dq[i] += eps
            g[i] = (self.collision_cost(dq) - h0) / eps
        return g


# =============================================================================
# Connection
# =============================================================================
class DeviceConnection:
    """Context manager for a Kortex session (TCP for services, UDP for cyclic)."""

    def __init__(self, ip, port, credentials):
        require_kortex_api()
        self.ip = ip
        self.port = port
        self.credentials = credentials
        self.transport = TCPTransport() if port == TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.session_manager = None

    def __enter__(self):
        self.transport.connect(self.ip, self.port)
        info = Session_pb2.CreateSessionInfo()
        info.username = self.credentials[0]
        info.password = self.credentials[1]
        info.session_inactivity_timeout = 60000       # ms
        info.connection_inactivity_timeout = 2000      # ms
        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(info)
        return self.router

    def __exit__(self, *exc):
        if self.session_manager is not None:
            opts = RouterClientSendOptions()
            opts.timeout_ms = 1000
            self.session_manager.CloseSession(opts)
        self.transport.disconnect()


# =============================================================================
# State + math helpers
# =============================================================================
def read_state(feedback, n):
    """Return (q, dq) in radians / rad-per-sec from a BaseCyclic feedback."""
    q = np.array([feedback.actuators[i].position for i in range(n)])
    dq = np.array([feedback.actuators[i].velocity for i in range(n)])
    q = np.deg2rad(_wrap_deg(q))
    dq = np.deg2rad(dq)
    return q, dq


def _wrap_deg(deg):
    """Wrap actuator angles (Kinova reports 0..360) into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _wrap_rad(rad):
    """Shortest-path angle error in (-pi, pi], so a joint spring never takes the
    long way around."""
    return (rad + np.pi) % (2.0 * np.pi) - np.pi


def critical_damping(kp, zeta=1.0):
    """Damping for a unit-apparent-inertia spring of stiffness ``kp``.

        D = 2 * zeta * sqrt(K)

    ALWAYS DERIVE DAMPING FROM THE FINAL, ALREADY-SCALED STIFFNESS. Calling this
    on the base constant and multiplying the result by the same scale afterwards
    is the bug this helper exists to prevent: damping would then scale as
    ``s`` where it must scale as ``sqrt(s)``, so the effective ratio drifts to
    ``zeta * sqrt(s)``.

    That was live until 2026-08-26 on the cartesian task gains, the render-mode
    environment gains and the free-mode hold spring. At ``--cart-scale 0.25`` it
    left the task at HALF the intended damping ratio (underdamped -- the EE
    rings); at ``--cart-scale 4`` it ran at twice (sluggish). Defaults were
    unaffected, since every one of those scales defaults to 1.0 and sqrt(1) = 1,
    which is why it survived so long.

    The loop already gets this right where it blends the re-anchor yield:
    ``engage_d = sqrt(engage)`` scales damping by the root of the stiffness
    fraction. This is the same rule, applied to the CLI scales.
    """
    return 2.0 * zeta * np.sqrt(np.asarray(kp, dtype=float))


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a, b):
    """Hamilton product, (x,y,z,w) convention."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def orientation_error(q_des, q_cur):
    """Rotation-vector error (axis*angle, base frame) from desired to current
    orientation quaternion. Mirrors the Kuka quat-error -> twist step."""
    qe = _quat_mul(q_des, _quat_conj(q_cur))
    if qe[3] < 0.0:               # keep the short way around
        qe = -qe
    vec = qe[:3]
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return 2.0 * vec          # small-angle limit
    angle = 2.0 * np.arctan2(norm, qe[3])
    return (vec / norm) * angle


def _segment_distance(p1, q1, p2, q2):
    """Minimum distance between segments [p1,q1] and [p2,q2].

    Ericson, Real-Time Collision Detection (ClosestPtSegmentSegment).
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-9
    if a <= eps and e <= eps:                 # both degenerate to points
        return float(np.linalg.norm(r))
    if a <= eps:                              # first segment is a point
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:                          # second segment is a point
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _segment_distance_batch(P1, Q1, P2, Q2):
    """Vectorised `_segment_distance` over K segment pairs at once.

    Same algorithm (Ericson, ClosestPtSegmentSegment), with every scalar branch
    rewritten as a `np.where` select so all K pairs are solved in one pass.

    This exists because the scalar version, called from a Python double loop,
    was the single most expensive thing in the 1 kHz control cycle: 195.6 us
    per `collision_cost`, i.e. ~66% of all control compute and ~16% of the
    whole 1.2 ms cycle (measured on REAL-1, 2026-08-26). Batched, the same
    query costs 59.3 us and returns bit-identical costs.

    Args:
        P1, Q1 (np.ndarray): (K, 3) endpoints of the first segment of each pair
        P2, Q2 (np.ndarray): (K, 3) endpoints of the second segment

    Returns:
        np.ndarray: (K,) minimum distance for each pair
    """
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d1, r)
    eps = 1e-9

    # Guarded denominators: the np.where SELECTS the right branch, but both
    # branches are evaluated, so every divisor has to be finite everywhere or
    # numpy warns (and propagates NaN into the selected branch's neighbours).
    a_ok = np.where(a > eps, a, 1.0)
    e_ok = np.where(e > eps, e, 1.0)
    denom = a * e - b * b
    denom_ok = np.where(denom > eps, denom, 1.0)

    # General case: both segments non-degenerate.
    s = np.where(denom > eps, np.clip((b * f - c * e) / denom_ok, 0.0, 1.0), 0.0)
    t = (b * s + f) / e_ok
    # Re-clamp s wherever the unclamped t left [0, 1] (the scalar version's
    # `if t < 0` / `elif t > 1` arms).
    s = np.where(t < 0.0, np.clip(-c / a_ok, 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / a_ok, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)

    # Degenerate arms, applied last so they win over the general case.
    first_pt = a <= eps                       # segment 1 is a point
    s = np.where(first_pt, 0.0, s)
    t = np.where(first_pt, np.clip(f / e_ok, 0.0, 1.0), t)
    second_pt = (e <= eps) & ~first_pt        # segment 2 is a point
    t = np.where(second_pt, 0.0, t)
    s = np.where(second_pt, np.clip(-c / a_ok, 0.0, 1.0), s)
    both_pt = (a <= eps) & (e <= eps)         # both degenerate
    s = np.where(both_pt, 0.0, s)
    t = np.where(both_pt, 0.0, t)

    diff = (P1 + s[:, None] * d1) - (P2 + t[:, None] * d2)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


# Joint-limit barrier (null-space): Gen3 joints 1,3,5,7 are CONTINUOUS -- no
# limit, so +/-inf disables the barrier there; only 2,4,6 are revolute. Values
# from gen3_2f85.urdf.
Q_MIN = np.array([-np.inf, -2.41, -np.inf, -2.66, -np.inf, -2.23, -np.inf])
Q_MAX = np.array([ np.inf,  2.41,  np.inf,  2.66,  np.inf,  2.23,  np.inf])
LIMIT_MARGIN = 0.35          # rad -- barrier activates within this of a limit


def joint_limit_torque(q, k):
    """Repulsive joint torque, nonzero only within LIMIT_MARGIN of a limit.

    Continuous joints have +/-inf limits, so both terms are 0 for them.
    """
    hi = np.maximum(0.0, q - (Q_MAX - LIMIT_MARGIN))
    lo = np.maximum(0.0, (Q_MIN + LIMIT_MARGIN) - q)
    return k * (lo - hi)


def compute_gravity(feedback, n, mode, tau_g0, model, q, g_offset):
    """Gravity torque feedforward vector (Nm), in Kinova command sign.

    mode == 'startup': constant, captured before torque mode engaged.
    mode == 'model'  : configuration-dependent g(q) from the KinDynModel.
    mode == 'hybrid' : model g(q) + a constant offset (tau_g0 - model.g(q0)) so
                       comp is EXACT at the start pose (matches the measured
                       holding torque) and the model supplies the variation as
                       the arm moves. Fixes the model's absolute underestimate --
                       needed for 'free' mode to actually hold the arm up.
    mode == 'none'   : no compensation (arm will sag -- diagnostic only).
    """
    if mode == "none":
        return np.zeros(n)
    if mode == "startup":
        return tau_g0
    if mode == "model":
        return model.gravity(q)[:n]
    if mode == "hybrid":
        return model.gravity(q)[:n] + g_offset
    raise ValueError(f"unknown gravity mode {mode!r}")


def joint_impedance_torque(q, dq, q_des, kp, kd, tau_g):
    """tau = -Kp*(q - q_des) - Kd*dq + g(q)  (the Kuka joint reference law)."""
    tau = -kp * _wrap_rad(q - q_des) - kd * dq
    tau += tau_g
    return tau


def singular_values(J):
    """Singular values of the Jacobian, descending. sigma[-1] is sigma_min."""
    return np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)


def damping_factor(sigma_min):
    """Chiaverini variable damping lam(sigma_min), continuous and C0 at both ends.

    Zero above SING_SIGMA_ON so well-conditioned poses are untouched, rising
    quadratically to SING_LAMBDA_MAX at sigma_min = 0. Quadratic rather than
    linear so the derivative is also zero at the handover point -- a linear ramp
    puts a kink in the torque exactly where the arm is already struggling.
    """
    if sigma_min >= SING_SIGMA_ON:
        return 0.0
    r = 1.0 - sigma_min / SING_SIGMA_ON      # 0 at the threshold, 1 at sigma=0
    return SING_LAMBDA_MAX * r * r


def damped_lambda(Lambda_inv, sigma_min):
    """Operational-space inertia (J M⁻¹ Jᵀ)⁻¹ with singularity-robust inversion.

    Returns ``(Lambda, lam)``. With ``lam == 0`` this is the exact inverse (up to
    the symmetrisation below), so nominal behaviour is unchanged.

    Lambda_inv is symmetric positive-definite in theory; floating-point J M⁻¹ Jᵀ
    is only symmetric to round-off, so it is symmetrised before ``eigh``, whose
    guaranteed-real eigenvalues are what make the per-mode damping well defined.
    """
    A = np.asarray(Lambda_inv, dtype=float)
    A = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(A)
    # eigh can return a tiny NEGATIVE eigenvalue for a theoretically PSD matrix
    # purely from round-off; 1/e would then flip the sign of that mode's torque,
    # driving the arm INTO the singularity. Floor at a value far below anything
    # physical so this only ever catches round-off, never shapes real behaviour.
    evals = np.maximum(evals, 1e-12)
    lam = damping_factor(sigma_min)
    if lam > 0.0:
        inv_e = evals / (evals * evals + lam * lam)
    else:
        inv_e = 1.0 / evals
    Lambda = (evecs * inv_e) @ evecs.T
    return 0.5 * (Lambda + Lambda.T), lam


def sigma_min_gradient(model, q, eps=SING_GRAD_EPS, column=None, base=None):
    """d(sigma_min(J))/dq by central difference -- the null-space escape direction.

    Ascending this gradient moves the arm AWAY from the singular set. There is no
    cheap closed form (it needs dJ/dq_i for every i), so it is a finite
    difference costing 2 Jacobian evaluations per column.

    ``column`` computes a single column i and returns ``(i, value)``, so the
    caller can spread the 7-column sweep across cycles the way the self-collision
    gradient already does -- taking all 7 in one cycle costs 14 KDL Jacobian
    calls and will not fit in a 1 kHz budget. ``base`` is unused for the central
    difference and accepted only to keep the call signature stable.
    """
    n = len(q)
    idx = range(n) if column is None else (column,)
    out = np.zeros(n)
    for i in idx:
        qp = np.array(q, dtype=float)
        qm = np.array(q, dtype=float)
        qp[i] += eps
        qm[i] -= eps
        sp = singular_values(model.jacobian(qp))[-1]
        sm = singular_values(model.jacobian(qm))[-1]
        out[i] = (sp - sm) / (2.0 * eps)
    if column is not None:
        return column, out[column]
    return out


def cartesian_impedance_torque(model, q, dq, p_des, quat_des, q0,
                               kp_cart, kd_cart, kp_null, kd_null, tau_g,
                               tau_collide, k_limit, f_task_extra=None,
                               state=None, dyn=None, tau_sing=0.0):
    """Operational-space impedance torque (ported from Kuka opspace).

    Task spring-damper on the EE pose, mapped through Lambda and Jᵀ, with a
    null-space secondary task: posture PD toward q0, a joint-limit barrier, and
    a precomputed self-collision repulsion ``tau_collide`` (all projected into
    the null space so they never disturb the EE task).

    Returns ``(tau[N], manip, sigma_min, lam_damp)`` where ``sigma_min`` is the
    smallest singular value of J (the singularity metric the caller gates on)
    and ``lam_damp`` is the damping actually applied to the Lambda inversion
    this cycle -- 0.0 whenever the pose is well conditioned.

    ``tau_sing`` (optional n-vec) is the precomputed null-space escape torque
    K_SING * d(sigma_min)/dq. It is passed in rather than computed here because
    the gradient is a finite difference whose 7-column sweep is spread across
    cycles by the caller, exactly as ``tau_collide`` already is.

    ``f_task_extra`` (optional 6-vec) is an extra task-space wrench added as a
    RAW force (not Lambda-weighted) before the Jᵀ map -- used by 'render' mode to
    inject the virtual-mass force Km*acc on top of the impedance.

    ``state`` (optional ``(p_cur, quat_cur, J)``) reuses an FK/Jacobian the
    caller already evaluated this cycle (the re-anchor watchdog needs both), so
    the model is not queried twice per 1 kHz cycle.

    ``dyn`` (optional ``(M, Minv)``) does the same for the joint-space inertia,
    which the table barrier also needs to invert its torque. Building M costs a
    KDL call plus an n^2 Python read-back loop and inverting it is another
    O(n^3), so at 1 kHz doing it twice a cycle is worth avoiding.
    """
    n = model.n
    if state is None:
        p_cur, quat_cur = model.fk(q)
        J = model.jacobian(q)                   # 6 x n
    else:
        p_cur, quat_cur, J = state

    # Task-space pose error and velocity.
    x_err = np.concatenate([p_des - p_cur, orientation_error(quat_des, quat_cur)])
    xdot = J @ dq                               # 6

    # Desired task wrench (impedance), clamped for safety.
    F = kp_cart * x_err - kd_cart * xdot
    F[:3] = np.clip(F[:3], -MAX_CART_FORCE, MAX_CART_FORCE)
    F[3:] = np.clip(F[3:], -MAX_CART_TORQUE, MAX_CART_TORQUE)

    # Operational-space inertia Lambda = (J M^-1 Jᵀ)^-1, with pinv fallback.
    if dyn is None:
        M = model.mass(q)
        Minv = np.linalg.inv(M)
    else:
        M, Minv = dyn
    Lambda_inv = J @ Minv @ J.T
    # sigma_min(J) is the singularity metric everything downstream gates on; the
    # determinant-based `manip` is kept only because the CSV logs and
    # analyze_tracking.py already carry that column. See the SING_* block for
    # why the old det(Lambda_inv) >= 1e-2 test could not detect a singularity.
    sigma = singular_values(J)
    sigma_min = float(sigma[-1])
    manip = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
    Lambda, lam_damp = damped_lambda(Lambda_inv, sigma_min)

    # Optional extra raw task wrench (render-mode virtual mass Km*acc), clamped
    # like F, added inside the Jᵀ map but NOT Lambda-weighted (it is a force).
    if f_task_extra is not None:
        fe = np.asarray(f_task_extra, dtype=float).copy()
        fe[:3] = np.clip(fe[:3], -MAX_CART_FORCE, MAX_CART_FORCE)
        fe[3:] = np.clip(fe[3:], -MAX_CART_TORQUE, MAX_CART_TORQUE)
        tau = J.T @ (Lambda @ F + fe)
    else:
        tau = J.T @ (Lambda @ F)

    # Null-space secondary task: posture PD toward q0 + joint-limit barrier +
    # self-collision repulsion, all damped and projected into the null space.
    Jbar = Minv @ J.T @ Lambda
    tau_null = (kp_null * _wrap_rad(q0 - q)
                + joint_limit_torque(q, k_limit)
                + tau_collide
                + tau_sing
                - kd_null * dq)
    tau += (np.eye(n) - J.T @ Jbar.T) @ tau_null

    tau += tau_g
    return tau, manip, sigma_min, lam_damp


def free_drive_torque(q, tau_g, tau_collide, k_limit):
    """Interaction mode 'free': weightless hand-guiding -- gravity comp only.

    No task spring at all, so the arm is backdrivable and stays wherever the
    human leaves it (real joint friction + the loop's global damping hold it).
    The self-collision repulsion and joint-limit barrier stay ON so the operator
    still cannot drive the arm into itself or a hard stop.
    """
    return tau_g + joint_limit_torque(q, k_limit) + tau_collide


def track_setpoint(p_a, p_b, t, period):
    """Interaction mode 'track': cosine-eased position on the line A<->B.

    s ramps 0->1->0 over one `period` with zero velocity at both endpoints, so
    the moving equilibrium sweeps A -> B -> A smoothly (no velocity step at the
    turnarounds). Orientation is held separately (paper tracks translation only).
    """
    s = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / period))
    return p_a + s * (p_b - p_a)


class ReanchorController:
    """Hand-guided setpoint teaching for cartesian hold-ee / render.

    State machine over the EE twist, one update per control cycle:

      * ``moving``    -- |v| or |w| over the speed thresholds => the operator has
        hold of the tip. Latch ``yielding``: the caller fades the task spring,
        the null-space posture term and the wrist anchor down to ``yield_scale``
        so the arm is easy to push. Gravity comp is NOT touched, so it neither
        sags nor fights.
    A capture only happens out of a latched yield -- i.e. the tip must have
    moved deliberately (over the speed thresholds) before a held pose counts as
    taught. That is what stops a slow gravity collapse, which also ends "still
    and displaced", from cementing itself as the new setpoint: a droop that
    never settles hits ``yield_timeout``, which clears the yield AND blocks
    re-latching until the tip is genuinely still again.

      * ``still`` at a pose further than (``min_pos``, ``min_ori``) from the
        current setpoint -- start/continue a stillness timer; after
        ``hold_time`` seconds capture the current pose as the new setpoint and
        drop the yield (stiffness fades back in against ~0 error, so no jump).
      * ``still`` back within the deadband of the existing setpoint -- the
        operator let go without moving it anywhere new; just drop the yield.

    ``engage`` is the first-order-blended stiffness fraction (``yield_scale`` ->
    1.0 over ``blend_time``) the caller multiplies its gains by.
    """

    def __init__(self, hold_time, lin_speed, ang_speed, min_pos, min_ori,
                 yield_scale, blend_time, yield_timeout, restore_blend):
        self.hold_time = hold_time
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.min_pos = min_pos
        self.min_ori = min_ori
        self.yield_scale = yield_scale
        self.blend_time = blend_time
        self.yield_timeout = yield_timeout
        self.restore_blend = restore_blend
        self._slow_restore = False
        self.engage = 1.0
        self.yielding = False
        self.still_since = None
        self.yield_since = None
        self.blocked = False        # refractory after a timeout, until still

    def update(self, now, dt, p_cur, quat_cur, xdot, p_des, quat_des,
               enabled=True):
        """Advance the machine. Returns (p_des, quat_des, captured, timed_out).

        ``enabled=False`` (e.g. during the torque ramp-in) holds the machine
        idle at full stiffness -- the arm must be under full control before its
        own settling motion is allowed to be read as an operator.
        """
        moving = (np.linalg.norm(xdot[:3]) > self.lin_speed
                  or np.linalg.norm(xdot[3:]) > self.ang_speed)
        displaced = (
            np.linalg.norm(p_cur - p_des) > self.min_pos
            or np.linalg.norm(orientation_error(quat_des, quat_cur)) > self.min_ori)

        captured = False
        timed_out = False
        if not enabled:
            self.yielding = False
            self.still_since = self.yield_since = None
        else:
            if not moving:
                self.blocked = False           # tip is still -> re-arm
            if moving and not self.blocked:
                if not self.yielding:
                    self.yield_since = now
                self.yielding = True
                self.still_since = None
                # Backstop: a yield that never settles is the arm sagging, not
                # an operator. Force full stiffness back on and refuse to
                # re-latch until the tip has actually been still once.
                if now - self.yield_since >= self.yield_timeout:
                    self.yielding = False
                    self.blocked = True
                    self.yield_since = None
                    timed_out = True
            elif self.yielding:
                if not displaced:
                    self.yielding = False      # released where it already was
                    self.still_since = None
                elif self.still_since is None:
                    self.still_since = now
                elif now - self.still_since >= self.hold_time:
                    p_des = p_cur.copy()
                    quat_des = quat_cur.copy()
                    self.yielding = False
                    self.still_since = None
                    captured = True

        target = self.yield_scale if self.yielding else 1.0
        # Coming back from a timeout the setpoint may be far away, so ramping
        # stiffness in at the normal rate would yank the arm toward it. Restore
        # over `restore_blend` instead; a capture needs no such care (it lands on
        # ~zero error by construction).
        if timed_out:
            self._slow_restore = True
        if self.engage >= 1.0:
            self._slow_restore = False
        blend = (max(self.blend_time, self.restore_blend)
                 if self._slow_restore else self.blend_time)
        if blend > 0.0:
            step = dt / blend
            self.engage += float(np.clip(target - self.engage, -step, step))
        else:
            self.engage = target
        return p_des, quat_des, captured, timed_out


def set_control_mode(actuator_config, n, mode_value):
    """Set every arm actuator to a control mode (POSITION or TORQUE)."""
    info = ActuatorConfig_pb2.ControlModeInformation()
    info.control_mode = mode_value
    for dev_id in range(1, n + 1):      # actuator device ids are 1-based
        actuator_config.SetControlMode(info, dev_id)


def return_to_home(base, n, q0, countdown=3):
    """Return robot to its initial position prior to the start of the script after countdown.

    Waits for countdown seconds, then sends a ReachJointAngles action to the Kortex
    BaseClient to move back to q0 (initial joint configuration in radians).
    """
    if q0 is None or base is None:
        return

    print("=" * 70)
    print(f"Returning to initial position in {countdown} seconds ...")
    for s in range(countdown, 0, -1):
        print(f"   returning in {s} ...", end="\r", flush=True)
        time.sleep(1.0)
    print(" " * 40, end="\r")
    print("Moving arm to initial position...")

    action = Base_pb2.Action()
    action.name = "Return to initial position"
    action.application_data = ""

    for i in range(n):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = i
        joint_angle.value = float(np.rad2deg(q0[i]))

    e = threading.Event()

    def check_for_end_or_abort(notification):
        if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()

    notification_handle = None
    try:
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort,
            Base_pb2.NotificationOptions()
        )
        base.ExecuteAction(action)
        finished = e.wait(30.0)
        if finished:
            print("Arm successfully returned to initial position.")
        else:
            print("WARNING: Timed out waiting for arm to return to initial position.")
    except Exception as err:  # noqa: BLE001
        print(f"  (return to initial position failed: {err})")
    finally:
        if notification_handle is not None:
            try:
                base.Unsubscribe(notification_handle)
            except Exception:  # noqa: BLE001
                pass


def analyse_csv(filepath: str):
    """Parse and calculate summary metrics from an impedance control CSV log."""
    if not os.path.exists(filepath):
        print(f"Error: CSV file '{filepath}' not found.")
        return None

    try:
        data = {}
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            for col in header:
                data[col] = []
            for row in reader:
                if not row:
                    continue
                for col_name, val in zip(header, row):
                    data[col_name].append(float(val))

        for col in data:
            data[col] = np.array(data[col])

        # Identify joint indices
        n_joints = sum(1 for col in header if col.startswith("q") and not col.startswith("q_"))
        t = data.get("t", np.array([]))

        print("=" * 70)
        print(f"CSV LOG ANALYSIS: {filepath}")
        if len(t) > 0:
            print(f"  Total Duration : {t[-1] - t[0]:.2f} s ({len(t)} samples)")
            print(f"  Sample Rate    : {len(t) / max(t[-1] - t[0], 1e-3):.1f} Hz (decimated)")

        print("-" * 70)
        print(f"{'Joint':<8} | {'q_min (deg)':<12} | {'q_max (deg)':<12} | {'peak |dq| (rad/s)':<18} | {'peak |tau| (Nm)':<15}")
        print("-" * 70)

        for i in range(1, n_joints + 1):
            q_col = f"q{i}"
            dq_col = f"dq{i}"
            tau_col = f"tau{i}"

            q_deg = np.rad2deg(data[q_col]) if q_col in data else np.array([0.0])
            dq = data[dq_col] if dq_col in data else np.array([0.0])
            tau = data[tau_col] if tau_col in data else np.array([0.0])

            q_min = np.min(q_deg) if len(q_deg) > 0 else 0.0
            q_max = np.max(q_deg) if len(q_deg) > 0 else 0.0
            peak_dq = np.max(np.abs(dq)) if len(dq) > 0 else 0.0
            peak_tau = np.max(np.abs(tau)) if len(tau) > 0 else 0.0

            print(f"Joint {i:<2} | {q_min:<12.1f} | {q_max:<12.1f} | {peak_dq:<18.3f} | {peak_tau:<15.2f}")

        print("=" * 70)
        return data

    except Exception as e:  # noqa: BLE001
        print(f"Failed to analyze CSV file '{filepath}': {e}")
        return None


def display_csv_data(filepath: str = None, show_plot: bool = True):
    """Analyze CSV log file and display visual plots of joint states and torques."""
    if filepath is None:
        print("No CSV filepath provided to display_csv_data().")
        return

    data = analyse_csv(filepath)
    if data is None or "t" not in data or len(data["t"]) == 0:
        return

    if not show_plot:
        return

    try:
        import matplotlib.pyplot as plt

        t = data["t"]
        n_joints = sum(1 for col in data if col.startswith("q") and not col.startswith("q_"))

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.canvas.manager.set_window_title(f"Impedance Control Log: {os.path.basename(filepath)}")

        # 1. Joint Positions
        ax_q = axes[0]
        for i in range(1, n_joints + 1):
            ax_q.plot(t, np.rad2deg(data[f"q{i}"]), label=f"Joint {i}")
        ax_q.set_ylabel("Position (deg)")
        ax_q.set_title("Joint Positions")
        ax_q.grid(True, linestyle="--", alpha=0.6)
        ax_q.legend(loc="upper right", ncol=4, fontsize=8)

        # 2. Joint Velocities
        ax_dq = axes[1]
        for i in range(1, n_joints + 1):
            ax_dq.plot(t, data[f"dq{i}"], label=f"Joint {i}")
        ax_dq.set_ylabel("Velocity (rad/s)")
        ax_dq.set_title("Joint Velocities")
        ax_dq.grid(True, linestyle="--", alpha=0.6)
        ax_dq.legend(loc="upper right", ncol=4, fontsize=8)

        # 3. Commanded Torques
        ax_tau = axes[2]
        for i in range(1, n_joints + 1):
            ax_tau.plot(t, data[f"tau{i}"], label=f"Joint {i}")
        ax_tau.set_xlabel("Time (s)")
        ax_tau.set_ylabel("Torque (Nm)")
        ax_tau.set_title("Commanded Joint Torques")
        ax_tau.grid(True, linestyle="--", alpha=0.6)
        ax_tau.legend(loc="upper right", ncol=4, fontsize=8)

        plt.tight_layout()

        # Save plot to PNG in the workspace so it is viewable on remote laptops / IDEs
        png_path = filepath.rsplit(".", 1)[0] + ".png" if "." in filepath else filepath + ".png"
        try:
            fig.savefig(png_path, dpi=150)
            print(f"Saved plot image to: {png_path}")
        except Exception as err:
            print(f"  (saving plot PNG failed: {err})")

        if show_plot and os.environ.get("DISPLAY"):
            try:
                print("Displaying CSV plots window...")
                plt.show()
            except Exception as err:
                print(f"  (GUI display window unavailable over SSH: {err})")
        else:
            print("Note: Running in headless/SSH session (no DISPLAY). You can open the saved PNG directly in your IDE/laptop.")
        plt.close(fig)

    except Exception as e:  # noqa: BLE001
        print(f"  (displaying CSV plots failed: {e})")



# =============================================================================
# Main control loop
# =============================================================================
def run_impedance(args):
    # Load the kin/dyn model up front if cartesian mode, model gravity, or the
    # no-motion gravity validation needs it.
    model = None
    if (args.mode == "cartesian" or args.gravity in ("model", "hybrid")
            or args.validate_gravity):
        model = KinDynModel(args.urdf, args.base_link, args.tip_link)
    elif args.gravity != "none":
        # Joint mode with a constant feedforward does not otherwise need the
        # model, but the gravity SIGN check does, and an inverted feedforward is
        # exactly as dangerous here. Build it just for the check; joint mode
        # still runs if PyKDL/urdf_parser_py are unavailable (the documented
        # contract), only unvalidated.
        try:
            model = KinDynModel(args.urdf, args.base_link, args.tip_link)
        except Exception as e:  # noqa: BLE001
            print(f"NOTE: gravity sign UNVALIDATED (no kin/dyn model: {e}). "
                  "Engage with the e-stop in hand and watch the first second.")

    # The table barrier is pure kinematics/dynamics -- without the model it
    # cannot run at all. Refuse to start rather than silently engaging torque
    # mode with a safety feature the user believes is active.
    if args.table_avoidance and model is None:
        raise SystemExit(
            "--table-avoidance needs the kin/dyn model, which failed to load "
            "(PyKDL + urdf_parser_py required; source your ROS setup).\n"
            "Refusing to start: torque mode without the barrier you asked for "
            "is more dangerous than not starting.")

    # Resolve (and validate) the guarded link set before opening the robot
    # connection, so a typo in --table-links costs nothing.
    table_names = _resolve_table_links(args, model)

    with DeviceConnection(args.robot_ip, TCP_PORT, DEFAULT_CREDENTIALS) as router, \
         DeviceConnection(args.robot_ip, UDP_PORT, DEFAULT_CREDENTIALS) as router_rt:

        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router_rt)
        actuator_config = ActuatorConfigClient(router)

        n = base.GetActuatorCount().count
        if model is not None and model.n != n:
            raise RuntimeError(
                f"model chain has {model.n} joints but arm has {n}; "
                "check --base-link/--tip-link.")

        kp = KP_JOINT[:n] * args.kp_scale
        kd = KD_JOINT[:n] * args.kd_scale
        tau_max = TAU_MAX[:n]
        # Joint integral trim (A): gain + anti-windup cap. i_tau accumulates the
        # integral TORQUE directly (Ki*integral of q-q_des), so the clamp is a
        # simple magnitude bound; both stay 0 while --ki-scale is 0.
        ki = KI_JOINT[:n] * args.ki_scale
        i_cap = I_CLAMP_FRAC * tau_max
        i_tau = np.zeros(n)
        kd_global = KD_GLOBAL[:n] * args.kd_scale   # drift-bleed, applied to total
        # Cartesian gravity trim: reuses the joint-mode integral gains and the
        # same anti-windup cap, but is applied unscaled by the yield (weight
        # support, not stiffness). See GRAVITY_TRIM_SCALE.
        k_trim = KI_JOINT[:n] * args.gravity_trim
        tau_trim = np.zeros(n)
        # Direct wrist/continuous-joint anchor toward q0 (cartesian, non-free):
        # a small UN-projected joint spring on joints 1,3,5,7, applied to the
        # total command so it reaches the wrist roll that the task + null-space
        # cannot. 0 disables it (legacy behaviour).
        kp_anchor = args.wrist_anchor * CONTINUOUS_JOINTS[:n]
        # 'free'-mode position-hold spring (critically damped for unit inertia).
        kp_hold = KP_HOLD[:n] * args.free_hold_scale
        kd_hold = critical_damping(kp_hold)

        # Cartesian TASK gains scale with --cart-scale (softens EE feel only);
        # posture/null-space gains scale with --kp-scale so self-collision and
        # posture authority stay independent of how compliant the task is set.
        kp_cart = np.concatenate([KP_CART_POS, KP_CART_ORI]) * args.cart_scale
        kd_cart = critical_damping(kp_cart, CART_DAMPING_RATIO)
        # Null-space posture stiffness toward q0. Scaled by --null-stiffness
        # (SEPARATE from --kp-scale) so the redundant DOF can be anchored without
        # stiffening the EE task. This is the restoring term that stops the slow
        # null-space creep of the continuous joints (1,3,5,7): damping alone only
        # sets a terminal drift velocity under residual gravity torque, it never
        # nulls the drift. 0.0 => fully free elbow (legacy behaviour).
        kp_null = KP_NULL[:n] * args.null_stiffness
        # DELIBERATELY NOT critical_damping(kp_null): unlike the springs above,
        # kd_null damps the WHOLE null-space secondary task, and two of its three
        # terms -- the joint-limit barrier and the self-collision repulsion --
        # are live regardless of --null-stiffness. Deriving this from kp_null
        # would drive it to zero at --null-stiffness 0 ("fully free elbow"),
        # leaving those two guards undamped and springy exactly when the operator
        # has asked for the most compliant configuration. It is nominal
        # null-space dissipation trimmed by --kd-scale, not the posture spring's
        # damping partner, so it stays tied to the nominal KP_NULL.
        kd_null = critical_damping(KP_NULL[:n]) * args.kd_scale

        # (B) Virtual-environment gains for interaction-mode 'render'. Stiffness
        # scales with --cart-scale (task softness), damping is derived like the
        # cartesian gains, and Km (virtual mass) scales with --km-scale so the
        # risky mass term is dialed independently from the spring/damper.
        env = ENV_PRESETS[args.env]
        env_kp = env["kp"] * args.cart_scale
        env_kd = critical_damping(env_kp, CART_DAMPING_RATIO)
        env_km = env["km"] * args.km_scale

        # --- Capture start state + gravity feedforward (still position-held) ---
        feedback = base_cyclic.RefreshFeedback()
        q0, _ = read_state(feedback, n)
        tau_g0 = SENSED_LOAD_SIGN * np.array(
            [feedback.actuators[i].torque for i in range(n)])

        # --- No-motion gravity validation (still position-held, no torque) -----
        # Both candidate feedforwards are checked against the potential-energy
        # gradient, which owes nothing to any KDL or Kinova sign convention.
        # Comparing them only against EACH OTHER is what let an inverted
        # feedforward pass for months: the model had been negated to agree with
        # the (load-signed) sensed torque, so the two agreed while both were
        # backwards. A sign disagreement with the energy truth on a loaded joint
        # means the arm will be driven the way gravity is already pulling, so it
        # ABORTS before torque mode engages.
        if model is not None:
            g_truth = model.hold_torque_energy(q0)[:n]
            g_model = model.gravity(q0)[:n]
            print("--- gravity check @ q0 (Nm) ---")
            print(f"  TRUE (dU/dq)     : {np.round(g_truth, 2)}")
            print(f"  sensed (startup) : {np.round(tau_g0, 2)}")
            print(f"  model  g(q0)     : {np.round(g_model, 2)}")
            print(f"  diff (model-true): {np.round(g_model - g_truth, 2)}")
            print(f"  diff (sens-true) : {np.round(tau_g0 - g_truth, 2)}")
            loaded = np.abs(g_truth) > 0.5
            bad = {}
            for label, cand in (("model", g_model), ("sensed", tau_g0)):
                m_bad = loaded & (np.abs(cand) > 0.5) & (
                    np.sign(cand) != np.sign(g_truth))
                if np.any(m_bad):
                    bad[label] = list(np.where(m_bad)[0] + 1)
            for label, joints in bad.items():
                print(f"  !! {label} feedforward has the WRONG SIGN on joints "
                      f"{joints} -- it would push the arm the way gravity "
                      "already pulls.")
            if not bad:
                print("  signs OK vs energy truth (joints with >0.5 Nm load).")
            # Only the feedforward actually selected can hurt us.
            in_use = {"startup": ["sensed"], "none": [],
                      "model": ["model"], "hybrid": ["model", "sensed"]}[args.gravity]
            fatal = [l for l in in_use if l in bad]
            if fatal and not args.allow_gravity_mismatch:
                raise RuntimeError(
                    f"--gravity {args.gravity} uses the {fatal} feedforward, "
                    "which disagrees in SIGN with the energy ground truth. "
                    "Engaging torque would make the arm sink under ~2x gravity. "
                    "Fix the sign (see SENSED_LOAD_SIGN) or, if you have "
                    "verified this is a false alarm, pass "
                    "--allow-gravity-mismatch.")
        if args.validate_gravity:
            print("validate-gravity: no torque engaged. Exiting.")
            return

        # Hybrid gravity offset: anchor model g(q) to the measured holding torque
        # at q0 so comp is exact at the start pose (see compute_gravity).
        g_offset = np.zeros(n)
        if args.gravity == "hybrid":
            g_offset = tau_g0 - model.gravity(q0)[:n]
            print(f"  hybrid g_offset  : {np.round(g_offset, 2)}")

        # Targets: joint -> q_des; cartesian -> startup EE pose (hold in place).
        if args.q_des is not None:
            q_des = np.array(args.q_des[:n])
        else:
            q_des = q0.copy()
        p_des = quat_des = None
        p_a = p_b = None                       # track-mode line endpoints
        # Posture / wrist-anchor reference. Starts at q0 and follows the arm on
        # every re-anchor, so the redundant DOF is left where the operator put it
        # instead of being pulled back to the startup configuration.
        q_ref = q0.copy()
        reanchor = None
        if args.mode == "cartesian":
            p_des, quat_des = model.fk(q0)     # startup EE pose (held)
            if args.interaction_mode == "track":
                p_a = p_des.copy()
                p_b = p_a + np.array(args.track_offset)
            # Hand-guided re-anchoring: only where a STATIC setpoint is held
            # ('track' drives p_des itself; 'free' already captures its own hold
            # pose in joint space).
            if (args.reanchor_time > 0.0
                    and args.interaction_mode in ("hold-ee", "render")):
                reanchor = ReanchorController(
                    args.reanchor_time, args.reanchor_lin_thresh,
                    args.reanchor_ang_thresh, args.reanchor_min_pos,
                    args.reanchor_min_ori, args.yield_scale, args.yield_blend,
                    args.yield_timeout, args.yield_restore_blend)

        print("=" * 70)
        print(f"Kinova Gen3 {args.mode.upper()} IMPEDANCE (low-level torque)")
        print(f"  actuators : {n}")
        print(f"  q_start   : {np.round(np.rad2deg(q0), 1)} deg")
        if args.mode == "joint":
            print(f"  q_des     : {np.round(np.rad2deg(q_des), 1)} deg")
            print(f"  Kp / Kd   : {np.round(kp, 1)} / {np.round(kd, 1)}")
            if args.ki_scale > 0.0:
                print(f"  Ki (trim) : {np.round(ki, 1)}  cap {np.round(i_cap, 1)} Nm")
                print(f"  Ki freeze : |dq|>{args.ki_freeze_thresh} rad/s pauses integ")
        else:
            print(f"  interaction: {args.interaction_mode}")
            print(f"  EE hold   : p={np.round(p_des, 3)} q={np.round(quat_des, 3)}")
            if args.interaction_mode == "free":
                print("  task      : spring OFF (gravity-comp hand-guiding)")
                print(f"  hold      : capture Kp={np.round(kp_hold, 1)} "
                      f"move>{args.free_move_thresh} rad/s")
            elif args.interaction_mode == "render":
                print(f"  env       : {args.env}  (virtual-environment render)")
                print(f"  Kp_cart   : {np.round(env_kp, 1)}  (Lambda-weighted)")
                print(f"  Kd_cart   : {np.round(env_kd, 1)}  "
                      f"(zeta={CART_DAMPING_RATIO:g} at --cart-scale "
                      f"{args.cart_scale:g})")
                print(f"  Km (mass) : {np.round(env_km, 2)}")
                if np.any(env_km > 0.0):
                    print("  !! Km>0 (mass-reduction) is POSITIVE accel feedback "
                          "-- raise --km-scale slowly, e-stop ready.")
            else:
                print(f"  Kp_cart   : {np.round(kp_cart, 1)}")
                print(f"  Kd_cart   : {np.round(kd_cart, 1)}  "
                      f"(zeta={CART_DAMPING_RATIO:g} at --cart-scale "
                      f"{args.cart_scale:g})")
            if args.interaction_mode == "track":
                print(f"  track A->B: {np.round(p_a, 3)} -> {np.round(p_b, 3)}"
                      f"  period {args.track_period}s")
            if args.interaction_mode == "free" or args.null_stiffness <= 0.0:
                print("  elbow     : FREE (posture off; collision/limit guards "
                      "active)")
            else:
                print(f"  posture   : Kp_null={np.round(kp_null, 1)} toward q0 "
                      f"(--null-stiffness {args.null_stiffness}; anti-drift)")
            if reanchor is not None:
                print(f"  re-anchor : ON -- hold the tip still {args.reanchor_time}s "
                      f">{args.reanchor_min_pos*100:.0f}cm / "
                      f"{np.rad2deg(args.reanchor_min_ori):.0f}deg away to re-teach "
                      "pose+orientation")
                print(f"  yield     : stiffness x{args.yield_scale} while hand-moved "
                      f"(blend {args.yield_blend}s); gravity comp stays full")
                print(f"  yield t/o : {args.yield_timeout}s unsettled -> full "
                      "stiffness (anti-collapse backstop)")
            elif args.interaction_mode in ("hold-ee", "render"):
                print("  re-anchor : OFF (--reanchor-time 0)")
            if args.interaction_mode != "free" and args.wrist_anchor > 0.0:
                print(f"  wrist anch: {args.wrist_anchor} Nm/rad direct on "
                      f"cont. joints 1,3,5,7 (un-projected anti-creep)")
            print(f"  cart clamp: F<{MAX_CART_FORCE:.0f} N  M<{MAX_CART_TORQUE:.0f} "
                  f"Nm (task force/moment saturation)")
        print(f"  gravity FF: {args.gravity} x{args.gravity_scale} "
              f"{np.round(tau_g0, 2)}")
        if args.mode == "cartesian" and args.gravity_trim > 0.0:
            print(f"  grav trim : Ki={np.round(k_trim, 1)} cap "
                  f"{np.round(i_cap, 1)} Nm, full authority (unscaled by yield), "
                  f"freezes above {args.trim_freeze_thresh} rad/s")
        if args.mode == "cartesian":
            print(f"  collapse  : abort if EE > {args.max_pose_err} m from "
                  f"commanded position ({args.max_pose_err_guided} m while "
                  "hand-guided)")
        if args.table_avoidance:
            print(f"  table CBF : surfaces held above z={args.table_z_min} m "
                  f"in {args.base_link} "
                  f"(alpha={args.table_alpha[0]:g},{args.table_alpha[1]:g}, "
                  f"link r={args.table_link_radius:g} m)")
            print("  table QP  : barrier + per-joint torque limits solved "
                  "together, so the clamp never overrides the filter")
            print(f"  table link: {len(table_names)} guarded -- "
                  + ", ".join(nm.replace('gen3_', '') for nm in table_names))
            if args.table_standoff > 0.0 and args.table_stiffness > 0.0:
                print(f"  table wall: K={args.table_stiffness:g} N/m "
                      f"D={args.table_damping:g} Ns/m over a "
                      f"{args.table_standoff:g} m standoff, "
                      f"cap {args.table_force_max:g} N/link "
                      f"(peak spring "
                      f"{args.table_stiffness * args.table_standoff:g} N, "
                      f"contact stiffness "
                      f"{2 * args.table_stiffness:g} N/m)")
            else:
                print("  table wall: OFF (CBF constraint only -- will stop the "
                      "arm but not push back)")
        print(f"  ramp-in   : {args.ramp_time}s (control torque fade)")
        if args.log is not None:
            print(f"  log       : {args.log}  (every {args.log_decimate} cycles)")
        print(f"  rate      : {args.rate:.0f} Hz   duration: {args.duration}s")
        print("=" * 70)
        print("!! TORQUE MODE disables the position safety envelope.")
        print("!! Keep the e-stop in hand. Ctrl-C restores position mode.")
        for s in range(args.countdown, 0, -1):
            print(f"   engaging in {s} ...", end="\r", flush=True)
            time.sleep(1.0)
        print(" " * 40, end="\r")

        # --- Switch to low-level torque control -------------------------------
        base.SetServoingMode(Base_pb2.ServoingModeInformation(
            servoing_mode=Base_pb2.LOW_LEVEL_SERVOING))
        feedback = base_cyclic.RefreshFeedback()

        # Seed the command from current feedback so nothing jumps.
        command = BaseCyclic_pb2.Command()
        for i in range(n):
            a = command.actuators.add()
            a.position = feedback.actuators[i].position
            a.torque_joint = feedback.actuators[i].torque
        frame_id = 0

        set_control_mode(actuator_config, n, ActuatorConfig_pb2.TORQUE)

        dt = 1.0 / args.rate         # NOMINAL period: paces the sleep, sets warn
        dt_min = DT_MEAS_MIN_FACTOR * dt
        dt_max = DT_MEAS_MAX_FACTOR * dt
        # Monotonic clock throughout: every elapsed-time comparison below (ramp,
        # track phase, re-anchor timers, duration) and now the integration
        # timestep itself are differences of this clock. time.time() is the wall
        # clock and an NTP correction mid-run would rewind the ramp and jump the
        # track setpoint while the arm is in torque mode.
        t_start = time.monotonic()
        # Seed one nominal period back so the FIRST cycle measures dt, not ~0.
        t_prev = t_start - dt
        last_warn = float("-inf")
        q_hold = q0.copy()          # 'free'-mode captured hold pose
        xdot_prev = np.zeros(6)     # (B) previous task velocity (accel estimate)
        acc_filt = np.zeros(6)      # (B) EMA-filtered task acceleration

        # Rolling self-collision gradient (see COLLIDE_EPS). One finite-difference
        # column per cycle against a frozen configuration; tau_collide holds the
        # last COMPLETED sweep so the applied torque is always a coherent
        # gradient rather than a half-updated one.
        tau_collide = np.zeros(n)   # published repulsion (last complete sweep)
        collide_grad = np.zeros(n)  # sweep under construction
        collide_q = q0.copy()       # configuration this sweep is differenced at
        collide_h0 = 0.0            # baseline cost at collide_q
        collide_col = 0             # next finite-difference column to evaluate

        # Singularity state, mirroring the collision sweep above: one column of
        # d(sigma_min)/dq per cycle, published only when the sweep completes.
        # sigma_min/lam_damp are seeded well-conditioned so the CSV has a valid
        # value for any cycle before the first cartesian evaluation.
        tau_sing = np.zeros(n)      # published escape torque (last full sweep)
        sing_grad_acc = np.zeros(n)  # sweep under construction
        sing_col = 0                # next finite-difference column to evaluate
        sing_warn_t = -1e9          # last proximity warning (rate limiting)
        sigma_min = float("nan")
        lam_damp = 0.0

        # Achieved-rate accounting, reported on exit. The audit had to recover
        # this from CSV sample spacing; the loop now just measures it.
        cyc_n = 0
        cyc_sum = 0.0
        cyc_max = 0.0
        cyc_min = float("inf")
        cyc_over = 0                # cycles that ran longer than the warn threshold
        cyc_timeout_run = 0         # CONSECUTIVE timed-out cyclic frames
        cyc_timeout_total = 0       # timed-out frames over the whole run

        # Bounded cyclic send options (see CYCLIC_TIMEOUT_MS). Built once: the
        # generated stub's default argument is a single shared instance created
        # at import time, so passing our own also avoids mutating that.
        send_opts = RouterClientSendOptions()
        send_opts.timeout_ms = args.cyclic_timeout_ms

        # (F) Loop invariants for the table barrier, hoisted out of the hot loop.
        table_alpha1, table_alpha2 = args.table_alpha
        table_segs = [model.segment_index(nm) for nm in table_names]
        table_radius = (0.0 if args.table_link_radius is None
                        else args.table_link_radius)
        last_table_warn = 0.0
        last_cbf_warn = 0.0
        last_clip_warn = 0.0

        # (D) Data logging: buffer rows in memory (no file I/O in the hot loop --
        # a stalled write could trip the actuator watchdog) and dump on exit.
        # Decimated to --log-decimate so 1 kHz control still logs at a sane rate.
        log_enabled = args.log is not None
        log_rows = []
        log_header = (["t"]
                      + [f"q{i+1}" for i in range(n)]
                      + [f"dq{i+1}" for i in range(n)]
                      + [f"tau{i+1}" for i in range(n)]
                      + [f"taug{i+1}" for i in range(n)]
                      + ["ramp", "engage", "xerr"]
                      + [f"trim{i+1}" for i in range(n)]
                      # Singularity trace: sigma_min is the metric the guards
                      # act on and lam_damp says whether the damped inversion
                      # was engaged, so a post-hoc log review can tell "the arm
                      # went soft" from "the arm was near a singularity and the
                      # controller gave up that direction on purpose". NaN in
                      # joint mode, which never evaluates a Jacobian.
                      + ["sigma_min", "lam_damp"])
        try:
            while True:
                step_start = time.monotonic()
                if args.duration and (step_start - t_start) >= args.duration:
                    break

                # MEASURED cycle period. This is the integration timestep for
                # the gravity trim, the joint integral trim and the re-anchor
                # blend. Using the nominal 1/rate here (as this loop did until
                # 2026-08-11) silently ran all three at 0.833x their configured
                # gain, because the loop delivers ~833 cycles/s while each one
                # claimed to advance time by a full millisecond.
                dt_raw = step_start - t_prev
                t_prev = step_start
                dt_meas = min(max(dt_raw, dt_min), dt_max)
                if frame_id > 0:            # first interval is seeded, not real
                    cyc_n += 1
                    cyc_sum += dt_raw
                    if dt_raw > cyc_max:
                        cyc_max = dt_raw
                    if dt_raw < cyc_min:
                        cyc_min = dt_raw
                    if dt_raw > SLOW_CYCLE_WARN_FACTOR * dt:
                        cyc_over += 1

                q, dq = read_state(feedback, n)

                # Safety: runaway joint speed -> zero torque and abort.
                if np.any(np.abs(dq) > MAX_JOINT_SPEED):
                    print(f"\nABORT: joint speed {np.round(dq, 2)} exceeds "
                          f"{MAX_JOINT_SPEED} rad/s.")
                    break

                tau_g = args.gravity_scale * compute_gravity(
                    feedback, n, args.gravity, tau_g0, model, q, g_offset)

                # Stiffness fraction for this cycle: 1.0 normally, faded toward
                # --yield-scale by the re-anchor watchdog while the operator has
                # hold of the tip. Never applied to the gravity feedforward.
                engage = 1.0
                pose_err = 0.0          # EE position error (cartesian, logged)

                # Per-cycle joint-space inertia cache. Both the operational-
                # space impedance and the table barrier need M(q); whichever
                # runs first fills this in and the other reuses it, so M is
                # built (and inverted) at most once per 1 kHz cycle.
                M_cyc = Minv_cyc = None

                if args.mode == "joint":
                    tau = joint_impedance_torque(q, dq, q_des, kp, kd, tau_g)
                    # (A) Integral trim: accumulate Ki*(q-q_des)*dt as a torque,
                    # clamp for anti-windup, subtract (spring sign). Disabled
                    # when --ki-scale is 0 (ki == 0 -> i_tau stays 0).
                    # Conditional integration: joints moving faster than
                    # --ki-freeze-thresh (being hand-driven) contribute 0 to the
                    # accumulator, so the integral can't wind up against the
                    # operator; the already-stored i_tau is still applied and
                    # resumes growing once the joint is released/still.
                    if args.ki_scale > 0.0:
                        integ = np.where(np.abs(dq) > args.ki_freeze_thresh,
                                         0.0, _wrap_rad(q - q_des))
                        i_tau = np.clip(i_tau + ki * integ * dt_meas,
                                        -i_cap, i_cap)
                        tau = tau - i_tau
                else:
                    # Self-collision gradient, swept one finite-difference
                    # column per cycle (see COLLIDE_EPS). Taking all n+1 columns
                    # in one cycle cost 1547 us -- longer than the whole cycle --
                    # and decimating that only hid the spike in the average.
                    # Active in every interaction mode (guards the free elbow).
                    if collide_col == 0:
                        # Freeze the configuration the whole sweep differences
                        # against, so the published gradient belongs to ONE pose.
                        collide_q = q.copy()
                        collide_h0 = model.collision_cost(collide_q)
                    if collide_h0 == 0.0:
                        # Every capsule pair is outside COLLISION_MARGIN, so the
                        # quadratic hinge is flat and the gradient is exactly
                        # zero -- skip the n perturbations entirely.
                        #
                        # NB (2026-08-11 audit): this early-out does NOT fire on
                        # the current model. CAPSULE_RADIUS 0.055 m makes the
                        # capsule diameter (0.110 m) exceed the minimum wrist
                        # capsule-axis distance (0.106 m), so those (i, i+2)
                        # pairs report a permanent -4.1 mm violation at every
                        # posture and the cost is never zero. Fixing that
                        # geometry is a separate change; this path is here so it
                        # pays off the moment it is.
                        collide_grad[:] = 0.0
                        tau_collide = np.zeros(n)
                        collide_col = 0
                    else:
                        q_pert = collide_q.copy()
                        q_pert[collide_col] += COLLIDE_EPS
                        collide_grad[collide_col] = (
                            (model.collision_cost(q_pert) - collide_h0)
                            / COLLIDE_EPS)
                        collide_col += 1
                        if collide_col >= n:
                            # Sweep complete -- publish it as one coherent
                            # gradient and start the next one next cycle.
                            tau_collide = -K_REPULSE * collide_grad.copy()
                            collide_col = 0

                    if args.interaction_mode == "free":
                        # Weightless hand-guiding: no task spring, no singularity
                        # risk (no Jacobian inversion). Elbow + EE both free.
                        tau = free_drive_torque(q, tau_g, tau_collide, K_LIMIT)
                        # Position capture: while being moved, re-center the hold
                        # pose to the hand (spring off, fully fluid); once still,
                        # hold it with a light damped spring so it does not sink.
                        if np.max(np.abs(dq)) > args.free_move_thresh:
                            q_hold = q.copy()
                        else:
                            tau = tau + kp_hold * _wrap_rad(q_hold - q) - kd_hold * dq
                    else:
                        # hold-ee holds the startup EE pose; render adds the
                        # virtual environment; track sweeps p_des along the A->B
                        # line. FK/Jacobian are evaluated ONCE here and handed to
                        # the impedance law -- the re-anchor watchdog needs the
                        # same pose/twist, and the model is too heavy to query
                        # twice per cycle.
                        p_cur, quat_cur = model.fk(q)
                        J = model.jacobian(q)
                        xdot = J @ dq

                        # Hand-guided re-anchor: fades the task spring down while
                        # the operator moves the tip, re-captures pose AND
                        # orientation once it has been held still somewhere new
                        # for --reanchor-time, and moves the posture reference to
                        # the current joints so the rest of the arm is left free.
                        if reanchor is not None:
                            was_yielding = reanchor.yielding
                            # Gated on the ramp: during soft-engage the arm's own
                            # settling would otherwise latch the yield at t~0 and
                            # never let go (2026-07-28 collapse).
                            p_des, quat_des, captured, timed_out = reanchor.update(
                                step_start, dt_meas, p_cur, quat_cur, xdot,
                                p_des, quat_des,
                                enabled=(step_start - t_start) >= args.ramp_time)
                            if reanchor.yielding and not was_yielding:
                                print("\n[re-anchor] hand motion -- task spring "
                                      "softened; hold the new pose to set it.")
                            if timed_out:
                                print(f"\n[re-anchor] yielded >{args.yield_timeout}s "
                                      "without settling -- restoring full "
                                      "stiffness over "
                                      f"{args.yield_restore_blend}s (reads as "
                                      "drift, not guiding). Pause, then nudge "
                                      "again to re-arm teaching.")
                                if float(np.linalg.norm(p_cur - p_des)) > args.max_pose_err:
                                    print("ABORT: ...and the EE is "
                                          f"{np.linalg.norm(p_cur - p_des):.2f} m "
                                          "from its setpoint, so re-stiffening "
                                          "would snap the arm back. Hold the tip "
                                          "still for "
                                          f"{args.reanchor_time}s to teach a pose "
                                          "instead of moving continuously, or "
                                          "raise --yield-timeout.")
                                    break
                            if captured:
                                q_ref = q.copy()
                                print(f"\n[re-anchor] locked EE at "
                                      f"p={np.round(p_des, 3)} "
                                      f"quat={np.round(quat_des, 3)}")
                            engage = reanchor.engage

                        # Slow-collapse guard (see MAX_POSE_ERROR). Enforced only
                        # while the arm is supposed to be HOLDING: during a
                        # hand-guide the distance from the (stale) setpoint is
                        # the operator's doing, not a failure. Without this
                        # exemption the guard fires on any large deliberate
                        # move -- worst on joint 1, whose base-yaw radius puts
                        # the EE 0.30 m away after only ~38 deg of rotation,
                        # while joints 5 and 7 never reach it at all.
                        pose_err = float(np.linalg.norm(p_cur - p_des))
                        guided = reanchor is not None and reanchor.yielding
                        err_limit = (args.max_pose_err_guided if guided
                                     else args.max_pose_err)
                        if pose_err > err_limit:
                            print(f"\nABORT: EE {pose_err:.2f} m from its "
                                  f"commanded position (> {err_limit} m"
                                  f"{' while hand-guided' if guided else ''}) -- "
                                  "the arm is not holding. Check gravity comp "
                                  "(--gravity-scale / --gravity-trim), or raise "
                                  "--max-pose-err"
                                  f"{'-guided' if guided else ''}.")
                            break

                        # Stiffness scales with `engage`; damping with its sqrt so
                        # the task stays ~critically damped at any yield level.
                        engage_d = float(np.sqrt(engage))

                        if args.interaction_mode == "render":
                            # (B) Virtual mass Km*acc on top of the impedance. The
                            # task acceleration is a low-passed finite diff of
                            # xdot using the ACTUAL loop dt (the loop rate is not
                            # exactly --rate).
                            # dt_meas is the shared measured period computed at
                            # the top of the loop (and clamped strictly > 0), so
                            # render no longer keeps its own t_prev -- this was
                            # the one place that already did the right thing.
                            acc_raw = (xdot - xdot_prev) / dt_meas
                            acc_filt = (ACC_FILTER_ALPHA * acc_filt
                                        + (1.0 - ACC_FILTER_ALPHA) * acc_raw)
                            xdot_prev = xdot
                            acc = np.clip(acc_filt, -MAX_TASK_ACC, MAX_TASK_ACC)
                            task_kp, task_kd = env_kp * engage, env_kd * engage_d
                            f_extra = env_km * acc
                        else:
                            if args.interaction_mode == "track":
                                p_des = track_setpoint(
                                    p_a, p_b, step_start - t_start,
                                    args.track_period)
                            task_kp, task_kd = kp_cart * engage, kd_cart * engage_d
                            f_extra = None

                        M_cyc = model.mass(q)
                        Minv_cyc = np.linalg.inv(M_cyc)
                        tau, manip, sigma_min, lam_damp = cartesian_impedance_torque(
                            model, q, dq, p_des, quat_des, q_ref,
                            task_kp, task_kd, kp_null * engage, kd_null, tau_g,
                            tau_collide, K_LIMIT, f_task_extra=f_extra,
                            state=(p_cur, quat_cur, J),
                            dyn=(M_cyc, Minv_cyc), tau_sing=tau_sing)

                        # Null-space singularity escape: advance ONE column of
                        # the d(sigma_min)/dq finite difference per cycle and
                        # publish the torque when the sweep wraps, so the cost
                        # is 2 Jacobian evaluations per cycle instead of 14.
                        # Only armed once the pose is actually degrading --
                        # above SING_SIGMA_ON the gradient is not even sampled,
                        # so nominal cycles pay nothing.
                        if args.sing_avoid and sigma_min < SING_SIGMA_ON:
                            col, val = sigma_min_gradient(
                                model, q, column=sing_col)
                            sing_grad_acc[col] = val
                            sing_col += 1
                            if sing_col >= n:
                                sing_col = 0
                                # Ascend the gradient: +grad moves AWAY from the
                                # singular set. Clamped so a bad finite
                                # difference near rank collapse (where sigma_min
                                # is least smooth) cannot dominate the command.
                                tau_sing = np.clip(
                                    args.sing_gain * sing_grad_acc,
                                    -MAX_SING_TORQUE, MAX_SING_TORQUE)
                        elif args.sing_avoid:
                            # Well conditioned again: release the escape torque
                            # and restart the sweep, so a stale gradient from a
                            # pose the arm has already left cannot keep pushing.
                            tau_sing = np.zeros(n)
                            sing_col = 0

                        # Proximity warning, then abort only on true rank
                        # collapse. Between the two the damped Lambda has
                        # already suppressed the degenerate direction, so the
                        # right response is to keep holding the arm up and tell
                        # the operator -- not to end the run.
                        if sigma_min < SING_SIGMA_ABORT:
                            print(f"\nABORT: rank collapse (sigma_min "
                                  f"{sigma_min:.2e} < {SING_SIGMA_ABORT:.1e}).")
                            break
                        if (lam_damp > 0.0
                                and step_start - sing_warn_t >= SING_WARN_PERIOD):
                            sing_warn_t = step_start
                            print(f"\n[singularity] sigma_min={sigma_min:.4f} "
                                  f"(< {SING_SIGMA_ON:.3f}) -- damping "
                                  f"lam={lam_damp:.4f}, task authority reduced "
                                  f"along the degenerate direction.")

                # Bounded gravity trim (cartesian): learns the residual holding
                # torque the gravity feedforward is missing and applies it at
                # FULL authority -- never scaled by `engage`, because it is
                # weight support, not stiffness. That is what lets the arm hold
                # its height while the task spring is yielded soft under a hand.
                # Accumulates only when the arm is nearly still and not being
                # guided (conditional-integration anti-windup, as in joint mode);
                # the stored value keeps acting while frozen. Not reset by a
                # re-anchor -- it is a gravity estimate, and dropping it there
                # would make the arm sag at the moment it locks the new pose.
                if args.mode == "cartesian" and args.gravity_trim > 0.0:
                    if (engage >= 1.0
                            and np.max(np.abs(dq)) < args.trim_freeze_thresh):
                        tau_trim = np.clip(
                            tau_trim + k_trim * _wrap_rad(q_ref - q) * dt_meas,
                            -i_cap, i_cap)
                    tau = tau + tau_trim

                # Global velocity damping (both modes): bleeds off slow drift on
                # low-friction / uncontrolled DOFs (continuous wrist joints).
                # Zero when static, so it never fights holding or the spring.
                tau = tau - kd_global * dq

                # Direct continuous-joint anchor (cartesian, non-free): a small
                # UN-projected joint spring toward q0 on joints 1,3,5,7. Damping
                # above only limits drift SPEED; this restoring term nulls the
                # position drift of the wrist roll (j7) that the Lambda-weighted
                # task torque and the null-space-projected posture both leave at
                # ~0. Gated out of 'free' (hand-guiding must stay uncommanded).
                if (args.mode == "cartesian"
                        and args.interaction_mode != "free"
                        and args.wrist_anchor > 0.0):
                    tau = tau + engage * kp_anchor * _wrap_rad(q_ref - q)

                # (E) Soft engage: fade the CONTROL torque in over --ramp-time
                # while the gravity feedforward stays full from the first cycle,
                # so the arm neither jumps (full gains at t=0) nor sags (no
                # gravity) at engage. ramp 0->1; gravity FF is left untouched.
                if args.ramp_time > 0.0:
                    ramp = min(1.0, (step_start - t_start) / args.ramp_time)
                    tau = tau_g + ramp * (tau - tau_g)
                else:
                    ramp = 1.0

                # (F) Table avoidance, applied to every monitored link. Two
                # cooperating layers -- see control_barrier.py for the split:
                #   wall : a real repulsive force, so the arm pushes back and
                #          the operator feels the table before reaching it.
                #   CBF  : a constraint on acceleration that guarantees
                #          non-penetration but exerts no restoring force.
                # The wall is what you feel; the CBF is the backstop.
                if args.table_avoidance:
                    z_pts, J_z, dJdq_z = model.height_barrier_terms(
                        q, dq, table_segs)
                    # Guard the link SURFACE, not its centreline: each monitored
                    # point is offset up by the capsule radius so a link's skin
                    # is what stops at z_min.
                    z_surf = z_pts - table_radius

                    # -- Layer 1: virtual wall (felt resistance) --------------
                    tau_wall, f_wall = table_wall_torque(
                        z_surf, J_z, dq, z_min=args.table_z_min,
                        standoff=args.table_standoff,
                        k_wall=args.table_stiffness,
                        d_wall=args.table_damping,
                        f_max=args.table_force_max)
                    tau = tau + tau_wall

                    # -- Layer 2: HOCBF constraint (hard backstop) -----------
                    # The torque-limit rows are stacked UNDER the barrier rows
                    # and solved in the same projection. Certifying a ddq the
                    # wrist cannot deliver and then saturating tau below would
                    # void the guarantee exactly when the barrier is braking
                    # hardest; here the clamp at the end is left as a backstop
                    # that should never bite.
                    # Inversion bias: the torque that buys NO acceleration.
                    # tau_g is the commanded gravity feedforward (deliberately
                    # that, not model.gravity(q), so --gravity-scale and the
                    # learned trim survive the round trip); tau_c is the
                    # Coriolis/centrifugal term, which was missing until now --
                    # without it ddq_nom is wrong by Minv @ C dq and the
                    # torque rebuilt below does not produce ddq_safe.
                    if M_cyc is None:               # joint / free mode
                        # Nothing else this cycle wants M's inverse, and the
                        # barrier needs exactly one solve against it, so solve
                        # rather than invert: measurably faster and better
                        # conditioned. Cartesian mode still reuses the explicit
                        # Minv it already built for the operational-space term.
                        M_cyc = model.mass(q)
                    M = M_cyc
                    tau_bias = tau_g + model.coriolis(q, dq)

                    if Minv_cyc is not None:
                        ddq_nom = Minv_cyc @ (tau - tau_bias)
                    else:
                        ddq_nom = np.linalg.solve(M, tau - tau_bias)
                    A_cbf, b_cbf = compute_table_hocbf_rows(
                        z_surf, J_z, dJdq_z, dq,
                        z_min=args.table_z_min,
                        alpha1=table_alpha1, alpha2=table_alpha2)
                    A_tau, b_tau = torque_limit_rows(M, tau_bias, tau_max)
                    A = np.vstack((A_cbf, A_tau))
                    b = np.concatenate((b_cbf, b_tau))
                    ddq_safe, cbf_status = filter_control_qp(ddq_nom, A, b)
                    if ddq_safe is not ddq_nom:
                        # Only rebuild the torque when the barrier actually
                        # clipped something -- otherwise M @ ddq_nom + tau_bias
                        # just re-introduces round-trip float error into tau.
                        tau = M @ ddq_safe + tau_bias

                    # A degraded solve means the projection is an approximation
                    # and NOT a safety certificate -- either the guarded links
                    # conflict or the barrier wants more torque than the joint
                    # has. Always report it, independently of --table-verbose:
                    # a safety filter that quietly stops being exact is the one
                    # failure that must not be silent.
                    if (cbf_status == "degraded"
                            and step_start - last_cbf_warn > 1.0):
                        k = int(np.argmin(z_surf - args.table_z_min))
                        print(f"\nWARNING: table CBF degraded (no exact "
                              f"projection) -- closest {table_names[k]} at "
                              f"{z_surf[k] - args.table_z_min:+.3f} m. The "
                              f"barrier may not hold; back the arm off and "
                              f"check --table-alpha / --table-z-min.",
                              flush=True)
                        last_cbf_warn = step_start

                    # Console feedback while testing the barrier: report the
                    # closest link whenever the wall is engaged, rate-limited.
                    if (args.table_verbose and np.any(f_wall > 0.0)
                            and step_start - last_table_warn > 0.5):
                        k = int(np.argmin(z_surf - args.table_z_min))
                        print(f"  [table] {table_names[k]} "
                              f"clr={z_surf[k] - args.table_z_min:+.3f} m  "
                              f"F={f_wall[k]:5.1f} N"
                              + ("" if cbf_status == "inactive"
                                 else f"  CBF:{cbf_status}"),
                              flush=True)
                        last_table_warn = step_start

                    # The clamp below is now redundant with the torque rows, so
                    # if it still bites the projection did not hold -- say so.
                    if (np.any(np.abs(tau) > tau_max + 1e-6)
                            and step_start - last_clip_warn > 1.0):
                        j = int(np.argmax(np.abs(tau) - tau_max))
                        print(f"\nWARNING: torque clamp bit under the table "
                              f"barrier (joint {j + 1}: {tau[j]:+.1f} Nm vs "
                              f"{tau_max[j]:.0f} Nm limit) -- the filtered "
                              f"command is being altered after certification.",
                              flush=True)
                        last_clip_warn = step_start

                tau = np.clip(tau, -tau_max, tau_max)

                if np.any(np.isnan(tau)):
                    print("\nABORT: NaN in torque command.")
                    break

                # (D) Capture a decimated log row (post-clip command torque).
                if log_enabled and frame_id % args.log_decimate == 0:
                    log_rows.append([round(step_start - t_start, 4)]
                                    + [round(v, 5) for v in q]
                                    + [round(v, 5) for v in dq]
                                    + [round(v, 4) for v in tau]
                                    + [round(v, 4) for v in tau_g]
                                    + [round(ramp, 3), round(engage, 3),
                                       round(pose_err, 4)]
                                    + [round(v, 4) for v in tau_trim]
                                    + [round(sigma_min, 6), round(lam_damp, 6)])

                # Build the cyclic command frame.
                frame_id = (frame_id + 1) % 65536
                command.frame_id = frame_id
                for i in range(n):
                    command.actuators[i].command_id = frame_id
                    # Hold position field at measurement so a fallback to
                    # position mode won't jump; torque_joint drives the joint.
                    command.actuators[i].position = feedback.actuators[i].position
                    command.actuators[i].torque_joint = float(tau[i])

                # Bounded-latency cyclic exchange. Without an explicit options
                # object this inherits timeout_ms=10000 and a single lost frame
                # wedges the torque loop for ten seconds (see CYCLIC_TIMEOUT_MS).
                try:
                    feedback = base_cyclic.Refresh(command, 0, send_opts)
                    cyc_timeout_run = 0
                except FutureTimeoutError:
                    cyc_timeout_run += 1
                    cyc_timeout_total += 1
                    if cyc_timeout_run >= args.max_cyclic_timeouts:
                        print(f"\nABORT: {cyc_timeout_run} consecutive cyclic "
                              f"frames timed out at "
                              f"{args.cyclic_timeout_ms:g} ms -- the arm is not "
                              "answering. Restoring position mode.")
                        break
                    # Retry immediately against the last good feedback rather
                    # than ending the run on one dropped datagram. `feedback` is
                    # deliberately left stale; the next pass recomputes from it.
                    continue

                now = time.monotonic()
                if (now - last_warn > 2.0
                        and (now - step_start) > SLOW_CYCLE_WARN_FACTOR * dt):
                    print(f"\nWARNING: loop slow ({(now-step_start)*1e3:.1f} ms "
                          f"work > {SLOW_CYCLE_WARN_FACTOR:g}x the "
                          f"{dt*1e3:.1f} ms target) -- watchdog risk.")
                    last_warn = now

                remaining = dt - (now - step_start)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("\nCtrl-C -- stopping.")
        finally:
            # --- Always restore a safe state -------------------------------
            print("Restoring POSITION control mode ...")
            try:
                set_control_mode(actuator_config, n, ActuatorConfig_pb2.POSITION)
            except Exception as e:  # noqa: BLE001
                print(f"  (control-mode restore failed: {e})")
            try:
                base.SetServoingMode(Base_pb2.ServoingModeInformation(
                    servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING))
            except Exception as e:  # noqa: BLE001
                print(f"  (servoing-mode restore failed: {e})")
            print("Done. Arm returned to high-level position control.")

            # Achieved-rate report. The loop does not reach --rate (see
            # DEFAULT_RATE_HZ) and for weeks nothing said so -- the 833 Hz
            # figure had to be reconstructed from CSV sample spacing after the
            # fact. Print it every run so a regression is visible immediately.
            if cyc_n > 0:
                mean_s = cyc_sum / cyc_n
                print("--- loop timing ---")
                print(f"  cycles     : {cyc_n}")
                print(f"  achieved   : {1.0 / mean_s:.1f} Hz "
                      f"(target {args.rate:.0f} Hz, "
                      f"{100.0 * (1.0 / mean_s) / args.rate:.1f}%)")
                print(f"  cycle time : mean {mean_s * 1e3:.3f} ms  "
                      f"min {cyc_min * 1e3:.3f}  max {cyc_max * 1e3:.3f}")
                print(f"  over {SLOW_CYCLE_WARN_FACTOR:g}x dt: {cyc_over} "
                      f"cycles ({100.0 * cyc_over / cyc_n:.2f}%)")
                print(f"  cyclic t/o : {cyc_timeout_total} frame(s) exceeded "
                      f"{args.cyclic_timeout_ms:g} ms")

            # Return to initial position prior to start of script after 3s countdown
            if q0 is not None:
                return_to_home(base, n, q0, countdown=3)

            # (D) Dump the buffered log AFTER the arm is safe -- a slow/failed
            # write must never delay the control-mode restore above.
            if log_enabled and log_rows:
                try:
                    with open(args.log, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(log_header)
                        w.writerows(log_rows)
                    print(f"Wrote {len(log_rows)} log rows to {args.log}")
                    display_csv_data(args.log)
                except Exception as e:  # noqa: BLE001
                    print(f"  (log write failed: {e})")

def main():
    p = argparse.ArgumentParser(
        description="Impedance control for Kinova Gen3 (real hardware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--robot-ip", default=DEFAULT_IP)
    p.add_argument("--mode", choices=["joint", "cartesian"], default="joint",
                   help="Impedance in joint space or operational (task) space.")
    p.add_argument("--interaction-mode",
                   choices=["free", "hold-ee", "track", "render"],
                   default="hold-ee",
                   help="[cartesian] free: weightless hand-guiding (no task "
                        "spring). hold-ee: hold the startup EE pose while the "
                        "elbow stays free. track: EE sweeps a line between two "
                        "points, compliant to human perturbation. render: render "
                        "a virtual environment (see --env), incl. virtual mass.")
    p.add_argument("--env", choices=list(ENV_PRESETS.keys()), default="spring",
                   help="[render] Virtual-environment preset: spring (stiff wall, "
                        "no virtual mass -- SAFE), inertia (feels heavier; adds "
                        "inertia, stable), passive (feels lighter; mass-reduction, "
                        "positive accel feedback -- bring up km-scale slowly).")
    p.add_argument("--km-scale", type=float, default=1.0,
                   help="[render] Scale the virtual-MASS term Km only (Kk/Kb use "
                        "--cart-scale). Start low for inertia/passive; 0 disables "
                        "virtual mass, leaving a pure stiffness/damping render.")
    p.add_argument("--track-offset", type=float, nargs=3,
                   default=[0.0, 0.10, 0.0], metavar=("DX", "DY", "DZ"),
                   help="[track] Point B = startup EE position + this offset "
                        "(m, base frame); oscillates A(startup) <-> B.")
    p.add_argument("--track-period", type=float, default=6.0,
                   help="[track] Seconds for one full A->B->A cycle.")
    p.add_argument("--free-hold-scale", type=float, default=1.0,
                   help="[free] Scale the position-hold stiffness (damping "
                        "follows as sqrt, keeping it critically damped). 0 = "
                        "pure gravity comp (no hold; arm sinks if model "
                        "imperfect).")
    p.add_argument("--free-move-thresh", type=float, default=FREE_MOVE_THRESH,
                   help="[free] Joint speed (rad/s) above which the arm is "
                        "treated as hand-moved (hold pose re-centers); below it, "
                        "the captured pose is held.")
    p.add_argument("--q-des", type=float, nargs="+", default=Q_DES,
                   metavar="RAD",
                   help="[joint] Desired joint config (7 rad). "
                        "Default: hold startup pose.")
    p.add_argument("--kp-scale", type=float, default=1.0,
                   help="Multiply joint + null-space (posture) stiffness gains. "
                        "In cartesian mode this sets posture/self-collision "
                        "authority; use --cart-scale for task softness.")
    p.add_argument("--kd-scale", type=float, default=1.0,
                   help="Multiply damping gains (incl. global drift damping).")
    p.add_argument("--ki-scale", type=float, default=0.0,
                   help="[joint] Integral-trim gain scale (default 0 = OFF). "
                        ">0 adds a bounded integral term that cancels the "
                        "steady-state gravity droop; integral torque is capped "
                        f"at {I_CLAMP_FRAC:g}*TAU_MAX (anti-windup). Start ~0.5.")
    p.add_argument("--ki-freeze-thresh", type=float, default=I_FREEZE_THRESH,
                   help="[joint] Conditional-integration anti-windup: joints "
                        "moving faster than this (rad/s) pause integral "
                        "accumulation, so it doesn't wind up while you hand-move "
                        "the arm (prevents overshoot/ringing on release).")
    p.add_argument("--cart-scale", type=float, default=1.0,
                   help="[cartesian] Scale ONLY the task stiffness -- softens "
                        "the EE feel WITHOUT weakening posture/self-collision "
                        "authority. Damping tracks as sqrt(scale), so the "
                        "damping RATIO stays put (before 2026-08-26 damping "
                        "scaled linearly, leaving 0.25 underdamped and 4 "
                        "sluggish).")
    p.add_argument("--null-stiffness", type=float, default=0.4,
                   help="[cartesian hold-ee/track/render] Null-space posture "
                        "stiffness toward q0, as a fraction of KP_NULL. Anchors "
                        "the redundant DOF so the continuous joints (1,3,5,7) "
                        "don't slowly creep -- damping alone can't null that "
                        "drift. 0 = fully free elbow (legacy). Projected into the "
                        "null space, so it never stiffens the EE task.")
    p.add_argument("--wrist-anchor", type=float, default=3.0,
                   help="[cartesian hold-ee/track/render] Direct joint-space "
                        "stiffness (Nm/rad) toward q0 on the continuous joints "
                        "1,3,5,7, applied OUTSIDE the null-space projector. Fixes "
                        "the residual wrist-roll (j7) creep that the task torque "
                        "and null-space posture both leave uncontrolled. Small is "
                        "enough (~3); 0 disables (legacy). Not applied in 'free'.")
    p.add_argument("--reanchor-time", type=float, default=REANCHOR_HOLD_TIME,
                   help="[cartesian hold-ee/render] Seconds the operator must "
                        "hold the tip still at a NEW pose before it becomes the "
                        "commanded pose+orientation (the posture reference moves "
                        "with it, leaving the other joints free). 0 disables "
                        "re-anchoring (legacy: the startup pose is held forever).")
    p.add_argument("--reanchor-lin-thresh", type=float,
                   default=REANCHOR_LIN_SPEED,
                   help="[re-anchor] EE linear speed (m/s) above which the tip "
                        "counts as hand-moved (softens, resets the hold timer).")
    p.add_argument("--reanchor-ang-thresh", type=float,
                   default=REANCHOR_ANG_SPEED,
                   help="[re-anchor] EE angular speed (rad/s) with the same role.")
    p.add_argument("--reanchor-min-pos", type=float, default=REANCHOR_MIN_POS,
                   help="[re-anchor] Minimum distance (m) from the current "
                        "setpoint for a held pose to count as NEW -- below it the "
                        "arm just re-stiffens on the old setpoint.")
    p.add_argument("--reanchor-min-ori", type=float, default=REANCHOR_MIN_ORI,
                   help="[re-anchor] Minimum rotation (rad) with the same role.")
    p.add_argument("--yield-scale", type=float, default=YIELD_SCALE,
                   help="[re-anchor] Fraction of task/posture/wrist stiffness "
                        "kept while the tip is being hand-moved. Lower = limper "
                        "under the hand. Gravity comp is never scaled, so the arm "
                        "does not sag. 1.0 = never soften (fight the operator).")
    p.add_argument("--yield-timeout", type=float, default=YIELD_TIMEOUT,
                   help="[re-anchor] Safety backstop: if the tip keeps 'moving' "
                        "this long without ever settling it is the arm drifting, "
                        "not an operator -- full stiffness is restored and the "
                        "yield cannot re-latch until the tip is genuinely still. "
                        "Prevents the droop->soften->droop collapse.")
    p.add_argument("--gravity-trim", type=float, default=GRAVITY_TRIM_SCALE,
                   help="[cartesian] Bounded joint-space integral that learns the "
                        "holding torque the gravity feedforward is missing, "
                        "applied at FULL authority (never scaled by the yield) "
                        "and capped at "
                        f"{I_CLAMP_FRAC:g}*TAU_MAX. This is the knob for 'holds "
                        "its position without getting stiffer' -- raise it if the "
                        "arm still droops, lower it if you feel slow bobbing. "
                        "0 = off (legacy).")
    p.add_argument("--trim-freeze-thresh", type=float, default=TRIM_FREEZE_THRESH,
                   help="[cartesian] Joint speed (rad/s) above which the gravity "
                        "trim stops accumulating, so it never winds up against "
                        "the operator while the arm is being guided.")
    p.add_argument("--gravity-scale", type=float, default=1.0,
                   help="Multiply the whole gravity feedforward. Blunt manual "
                        "alternative to --gravity-trim: >1 adds holding torque "
                        "everywhere (1.05-1.15 for mild droop). Raise slowly -- "
                        "too high and the arm drifts UPWARD on its own.")
    p.add_argument("--max-pose-err", type=float, default=MAX_POSE_ERROR,
                   help="[cartesian] Abort if the EE gets this far (m) from its "
                        "commanded position WHILE NOT BEING HAND-GUIDED. Catches "
                        "a slow collapse that stays under the MAX_JOINT_SPEED "
                        "runaway guard. Deliberate moves are exempt, so this "
                        "does not limit how far you can reposition the arm -- "
                        "note the base yaw (joint 1) alone covers this in ~38 "
                        "deg of rotation.")
    p.add_argument("--max-pose-err-guided", type=float,
                   default=MAX_POSE_ERROR_GUIDED,
                   help="[cartesian] The same guard's ceiling WHILE hand-guided, "
                        "where being far from the setpoint is intentional. Must "
                        "exceed --max-pose-err. Sized to clear a 90 deg base-yaw "
                        "reposition while still catching a fall.")
    p.add_argument("--yield-restore-blend", type=float, default=2.0,
                   help="[re-anchor] Seconds to fade stiffness back in after a "
                        "yield TIMEOUT (as opposed to a capture, which lands on "
                        "zero error). Longer than --yield-blend because the "
                        "setpoint may be far away and a fast restore would pull "
                        "the arm back sharply.")
    p.add_argument("--yield-blend", type=float, default=YIELD_BLEND_TIME,
                   help="[re-anchor] Seconds to fade stiffness between the "
                        "yielded and full level (0 = instant step -- jerky).")
    p.add_argument("--max-cart-torque", type=float, default=25.0,
                   help="[cartesian] Per-axis clamp on the task MOMENT (Nm). The "
                        "orientation spring saturates here, so too low a value "
                        "caps wrist authority regardless of --cart-scale (the old "
                        "hard-coded 15 let orientation run away past ~21 deg).")
    p.add_argument("--gravity", choices=["startup", "none", "model", "hybrid"],
                   default=None,
                   help="Gravity feedforward source. Default: 'hybrid' in "
                        "cartesian mode (model g(q) anchored to the measured "
                        "holding torque at q0 -- needed for 'free' to hold up), "
                        "'startup' in joint mode ('model'/'hybrid' need the URDF).")
    p.add_argument("--allow-gravity-mismatch", action="store_true",
                   help="Engage even if the selected gravity feedforward "
                        "disagrees in sign with the energy ground truth at q0. "
                        "This guard exists because an inverted feedforward made "
                        "the arm sink under ~2x gravity; only override it if you "
                        "have independently verified the sign.")
    p.add_argument("--validate-gravity", action="store_true",
                   help="Print model g(q0) vs sensed holding torque at startup, "
                        "then exit WITHOUT engaging torque (no motion).")
    p.add_argument("--urdf", default=DEFAULT_URDF,
                   help="[cartesian/model] URDF for the kin/dyn model.")
    p.add_argument("--base-link", default=DEFAULT_BASE_LINK)
    p.add_argument("--tip-link", default=DEFAULT_TIP_LINK)
    p.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ,
                   help="Control loop rate (Hz).")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Auto-stop after N seconds (0 = until Ctrl-C).")
    p.add_argument("--cyclic-timeout-ms", type=float, default=CYCLIC_TIMEOUT_MS,
                   help="Per-frame timeout (ms) on the low-level cyclic "
                        "exchange. The Kortex default is 10000, which blocks "
                        "the torque loop for ten seconds on one lost frame and "
                        "makes the position-mode restore unreachable. Must "
                        "exceed the loop period (~1.2 ms achieved).")
    p.add_argument("--max-cyclic-timeouts", type=int,
                   default=MAX_CYCLIC_TIMEOUTS,
                   help="Consecutive timed-out cyclic frames before aborting. "
                        "Absorbs an isolated dropped datagram without ending "
                        "the run; a persistent stall still aborts promptly.")
    p.add_argument("--countdown", type=int, default=5,
                   help="Seconds to wait before engaging torque mode.")
    p.add_argument("--ramp-time", type=float, default=1.0,
                   help="Soft-engage: seconds to fade control torque 0->1 at "
                        "start (gravity FF stays full). 0 = engage instantly.")
    p.add_argument("--log", default=None, metavar="PATH",
                   help="Write a CSV of t,q,dq,tau,taug,ramp to PATH (buffered "
                        "in RAM, flushed on exit). Off by default.")
    p.add_argument("--log-decimate", type=int, default=10,
                   help="Log every Nth control cycle (10 @ 1 kHz => 100 Hz).")
    p.add_argument("--table-avoidance", action="store_true",
                   help="Enable the HOCBF table-avoidance safety filter, which "
                        "keeps the --tip-link origin above --table-z-min.")
    p.add_argument("--table-z-min", type=float, default=0.08, metavar="Z",
                   help="Barrier height for --table-avoidance: minimum allowed "
                        "z of the --tip-link origin, in the --base-link frame "
                        "(metres). Table surface sits at z=-0.03, so the "
                        "default 0.05 leaves ~8 cm of clearance. Raise it to "
                        "cover hardware standing on the table.")
    p.add_argument("--table-alpha", type=float, nargs=2, default=(15.0, 15.0),
                   metavar=("A1", "A2"),
                   help="HOCBF gains (alpha1 alpha2) for --table-avoidance. "
                        "Higher = the barrier engages later and brakes harder; "
                        "lower = a softer, earlier slowdown. Both must be > 0.")
    p.add_argument("--table-links", nargs="*", default=None, metavar="LINK",
                   help="Links guarded by the barrier (default: the six distal "
                        "links). Pass 'all' for every link on the chain, or "
                        "'tip' for the --tip-link only.")
    p.add_argument("--table-link-radius", type=float, default=None, metavar="R",
                   help="Treat each guarded link as a cylinder of this radius "
                        "(m), so its SURFACE stops at --table-z-min rather "
                        "than its centreline. Default: the capsule radius "
                        f"({KinDynModel.CAPSULE_RADIUS}). 0 = guard centrelines.")
    p.add_argument("--table-standoff", type=float, default=0.06, metavar="D",
                   help="Thickness of the band above --table-z-min in which "
                        "the virtual wall pushes back (m). 0 disables the "
                        "wall, leaving only the HOCBF constraint.")
    p.add_argument("--table-stiffness", type=float, default=1200.0, metavar="K",
                   help="Virtual-wall stiffness (N/m). This is what you FEEL "
                        "when pushing the arm at the table; raise it if the "
                        "table still feels soft.")
    p.add_argument("--table-damping", type=float, default=60.0, metavar="D",
                   help="Virtual-wall damping (N*s/m), one-sided (resists "
                        "approach only). Raise with stiffness to stop bounce.")
    p.add_argument("--table-force-max", type=float, default=90.0, metavar="F",
                   help="Per-link cap on virtual-wall force (N). Bounds the "
                        "worst case if z_min or the model is wrong. Must "
                        "exceed --table-stiffness * --table-standoff (default "
                        "72 N) or the cap binds before the link reaches the "
                        "table and the last few mm go soft.")
    p.add_argument("--table-verbose", action="store_true",
                   help="Print the closest guarded link and wall force while "
                        "the wall is engaged (rate-limited to 2 Hz).")
    p.add_argument("--sing-avoid", action="store_true",
                   help="Cartesian mode: use the redundant 7th DOF to retreat "
                        "from kinematic singularities, by ascending the "
                        "d(sigma_min)/dq gradient inside the null-space "
                        "projector (so the tool tip does not move). Only "
                        "active below sigma_min = %.3f; costs 2 extra Jacobian "
                        "evaluations on those cycles. The damped Lambda "
                        "inversion is ALWAYS on and does not need this flag -- "
                        "this is the proactive layer on top of it."
                        % SING_SIGMA_ON)
    p.add_argument("--sing-gain", type=float, default=K_SING, metavar="K",
                   help="Nm per unit d(sigma_min)/dq for --sing-avoid "
                        "(default %(default)s, per-joint clamp "
                        + "%.0f Nm)." % MAX_SING_TORQUE)
    args = p.parse_args()

    if args.sing_gain < 0.0:
        p.error("--sing-gain must be >= 0 (a negative gain would DESCEND the "
                "gradient, driving the arm into the singularity).")

    if args.table_avoidance:
        if args.table_alpha[0] <= 0.0 or args.table_alpha[1] <= 0.0:
            p.error("--table-alpha values must both be > 0 "
                    "(HOCBF forward-invariance requires positive gains).")
        if args.table_stiffness < 0.0 or args.table_damping < 0.0:
            p.error("--table-stiffness/--table-damping must be >= 0.")
        if args.table_standoff < 0.0:
            p.error("--table-standoff must be >= 0 (0 disables the wall).")
        if args.table_force_max <= 0.0:
            p.error("--table-force-max must be > 0.")
        # A cap below the spring's peak turns the last stretch before the table
        # into a constant push instead of a stiff wall -- the one place the
        # wall has to be hard. Refuse rather than silently detune it.
        cap_clr = wall_cap_clearance(args.table_standoff, args.table_stiffness,
                                     args.table_force_max)
        if cap_clr > 0.0:
            p.error(
                f"--table-force-max ({args.table_force_max:g} N) is below the "
                f"virtual wall's peak spring force "
                f"({args.table_stiffness * args.table_standoff:g} N = "
                f"--table-stiffness * --table-standoff), so the cap binds "
                f"{cap_clr * 1000:.0f} mm above the table and the wall goes "
                f"soft right where it must be stiff.\nRaise --table-force-max "
                f"to >= {args.table_stiffness * args.table_standoff:g}, or "
                f"lower --table-stiffness/--table-standoff.")
        if args.table_link_radius is None:
            args.table_link_radius = KinDynModel.CAPSULE_RADIUS

    # Gravity default depends on mode: 'model' (config-dependent g(q), valid
    # across the workspace) for cartesian, 'startup' (constant) for joint.
    if args.gravity is None:
        args.gravity = "hybrid" if args.mode == "cartesian" else "startup"

    # Apply the configurable task-moment clamp (read as a module global inside
    # cartesian_impedance_torque). The per-joint TAU_MAX clamp still bounds the
    # final command, so raising this cannot exceed the actuators' torque limits.
    global MAX_CART_TORQUE
    MAX_CART_TORQUE = args.max_cart_torque

    if args.q_des is not None and len(args.q_des) < len(KP_JOINT):
        p.error(f"--q-des needs {len(KP_JOINT)} values, got {len(args.q_des)}")

    if args.ki_scale < 0.0:
        p.error("--ki-scale must be >= 0")
    if args.null_stiffness < 0.0:
        p.error("--null-stiffness must be >= 0")
    if args.wrist_anchor < 0.0:
        p.error("--wrist-anchor must be >= 0")
    if args.max_cart_torque <= 0.0:
        p.error("--max-cart-torque must be > 0")
    if args.reanchor_time < 0.0:
        p.error("--reanchor-time must be >= 0 (0 disables re-anchoring)")
    if not 0.0 <= args.yield_scale <= 1.0:
        p.error("--yield-scale must be in [0, 1]")
    if args.yield_blend < 0.0:
        p.error("--yield-blend must be >= 0")
    if args.yield_restore_blend < 0.0:
        p.error("--yield-restore-blend must be >= 0")
    if args.yield_timeout <= 0.0:
        p.error("--yield-timeout must be > 0 (it is the anti-collapse backstop)")
    if args.yield_timeout <= args.reanchor_time:
        p.error(f"--yield-timeout ({args.yield_timeout}) must exceed "
                f"--reanchor-time ({args.reanchor_time}), or the yield is torn "
                "down before a pose can ever be taught")
    if args.gravity_trim < 0.0:
        p.error("--gravity-trim must be >= 0")
    if args.trim_freeze_thresh < 0.0:
        p.error("--trim-freeze-thresh must be >= 0")
    if args.max_pose_err <= 0.0:
        p.error("--max-pose-err must be > 0")
    if args.max_pose_err_guided < args.max_pose_err:
        p.error(f"--max-pose-err-guided ({args.max_pose_err_guided}) must be >= "
                f"--max-pose-err ({args.max_pose_err}); guided moves are the "
                "permissive case")
    if not 0.0 <= args.gravity_scale <= 1.5:
        p.error("--gravity-scale must be in [0, 1.5]")
    if args.gravity_scale > 1.2:
        print(f"WARNING: --gravity-scale {args.gravity_scale} over-compensates "
              "by >20%; the arm may drift upward. E-stop ready.")
    if args.gravity_trim > 0.0 and args.mode != "cartesian":
        print("NOTE: --gravity-trim applies to cartesian mode; joint mode has "
              "the equivalent --ki-scale trim. Ignored here.")
    if (args.reanchor_time > 0.0 and args.mode == "cartesian"
            and args.interaction_mode in ("free", "track")):
        print(f"NOTE: re-anchoring does not apply to interaction-mode "
              f"{args.interaction_mode} (it has no static EE setpoint); ignored.")
    if args.ki_freeze_thresh < 0.0:
        p.error("--ki-freeze-thresh must be >= 0 (a negative value would freeze "
                "integration every cycle and silently disable the trim)")
    if args.log_decimate < 1:
        p.error("--log-decimate must be >= 1")
    if args.cyclic_timeout_ms <= 0.0:
        p.error("--cyclic-timeout-ms must be > 0")
    if args.max_cyclic_timeouts < 1:
        p.error("--max-cyclic-timeouts must be >= 1")
    if args.cyclic_timeout_ms < 2.0:
        # The blocking Refresh round trip measures ~0.9 ms against the real arm
        # (2026-08-26), so anything under ~2 ms starts timing out healthy
        # frames and the retry path becomes the normal path.
        print(f"WARNING: --cyclic-timeout-ms {args.cyclic_timeout_ms:g} is close "
              "to the measured ~0.9 ms cyclic round trip; healthy frames may "
              "time out.")
    if args.ki_scale > 0.0 and args.mode != "joint":
        print("NOTE: --ki-scale only affects joint mode; ignored in cartesian.")
    if args.interaction_mode == "render" and args.mode != "cartesian":
        p.error("--interaction-mode render requires --mode cartesian")
    if (args.env != "spring" or args.km_scale != 1.0) and (
            args.mode != "cartesian" or args.interaction_mode != "render"):
        print("NOTE: --env/--km-scale only apply to cartesian --interaction-mode "
              "render; ignored here.")

    try:
        run_impedance(args)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
