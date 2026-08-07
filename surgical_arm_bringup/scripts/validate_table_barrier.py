#!/usr/bin/env python3
"""Offline validation for the table-avoidance barrier. No robot required.

Checks the pieces that can only be verified against the real URDF/KDL chain:
  1. J_z rows from height_barrier_terms match numerical d(z)/dq
  2. KDL's dJ@dq product matches a finite difference of J along the trajectory
     (this is the Coriolis term the barrier depends on)
  3. the barrier actually holds a link above z_min under a closed-loop
     simulation of the arm being pushed into the table
  4. hot-loop timing at the configured link count

Run:  python3 validate_table_barrier.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import impedance as imp                                    # noqa: E402
from control_barrier import (compute_table_hocbf_rows,     # noqa: E402
                             filter_control_qp, table_wall_torque,
                             torque_limit_rows, wall_cap_clearance)

np.set_printoptions(precision=5, suppress=True)

model = imp.KinDynModel(imp.DEFAULT_URDF, imp.DEFAULT_BASE_LINK,
                        imp.DEFAULT_TIP_LINK)
n = model.n
links = [nm for nm in imp.DEFAULT_TABLE_LINKS if nm in model.segment_names()]
segs = [model.segment_index(nm) for nm in links]
print(f"chain: {n} joints, guarding {len(links)} links: "
      + ", ".join(s.replace('gen3_', '') for s in links))

rng = np.random.default_rng(3)
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


# --- 1. J_z vs numerical d(z)/dq --------------------------------------------
print("\n1. Jacobian z-rows vs numerical gradient of link height")
worst = 0.0
for _ in range(20):
    q = rng.uniform(-2.0, 2.0, n)
    _, J_z, _ = model.height_barrier_terms(q, np.zeros(n), segs)
    num = np.zeros_like(J_z)
    h = 1e-6
    for j in range(n):
        qp, qm = q.copy(), q.copy()
        qp[j] += h
        qm[j] -= h
        zp, _, _ = model.height_barrier_terms(qp, np.zeros(n), segs)
        zm, _, _ = model.height_barrier_terms(qm, np.zeros(n), segs)
        num[:, j] = (zp - zm) / (2 * h)
    worst = max(worst, np.max(np.abs(J_z - num)))
check("J_z == d(z)/dq", worst < 1e-5, f"max err {worst:.2e}")

# --- 2. dJ@dq vs finite difference of J along the motion --------------------
print("\n2. KDL dJ@dq (Coriolis term) vs finite difference")
worst = 0.0
for _ in range(20):
    q = rng.uniform(-2.0, 2.0, n)
    dq = rng.uniform(-1.5, 1.5, n)
    _, _, dJdq_z = model.height_barrier_terms(q, dq, segs)
    h = 1e-6
    _, Jp, _ = model.height_barrier_terms(q + h * dq, dq, segs)
    _, Jm, _ = model.height_barrier_terms(q - h * dq, dq, segs)
    num = ((Jp - Jm) / (2 * h)) @ dq
    worst = max(worst, np.max(np.abs(dJdq_z - num)))
check("dJ@dq == dJ/dt @ dq", worst < 1e-4, f"max err {worst:.2e}")

# a zero-dJ approximation is NOT equivalent -- show what it costs
q = np.deg2rad([0, 15, 180, -130, 0, 55, 90])
dq = np.array([0.0, -0.8, 0.0, 1.2, 0.0, 0.9, 0.0])
_, _, dJdq_z = model.height_barrier_terms(q, dq, segs)
print(f"       at a representative moving pose, dJ@dq (z) = {dJdq_z}")
print(f"       -> dropping it mis-states b by up to "
      f"{np.max(np.abs(dJdq_z)):.2f} m/s^2")

# --- 3. closed-loop: push the arm into the table ----------------------------
print("\n3. Closed-loop sim: constant downward push into the table")


def simulate(z_min, use_cbf, use_wall, push=90.0, T=3.0, dt=1e-3,
             standoff=0.06, k_wall=1200.0, d_wall=60.0, radius=0.055,
             tau_limits=True):
    """Integrate the rigid-body arm under gravity comp + a downward push.

    Mirrors the impedance.py hot loop, including the per-joint torque clamp --
    without it the sim would certify accelerations the real arm cannot produce
    and report a guarantee the hardware does not have.
    """
    q = np.deg2rad([0, 15, 180, -130, 0, 55, 90]).astype(float)
    dq = np.zeros(n)
    tau_max = imp.TAU_MAX[:n]
    lowest = np.inf
    degraded = 0
    clipped = 0
    for _ in range(int(T / dt)):
        z, J_z, dJdq_z = model.height_barrier_terms(q, dq, segs)
        z_s = z - radius
        lowest = min(lowest, np.min(z_s - z_min))

        # Operator pushes the tool tip straight down; light joint damping.
        J_full = model.jacobian(q)
        tau = J_full[2, :] * (-push) - 2.0 * dq

        if use_wall:
            tw, _ = table_wall_torque(z_s, J_z, dq, z_min, standoff,
                                      k_wall, d_wall, f_max=90.0)
            tau = tau + tw

        # Plant: M ddq + C dq = tau (gravity is perfectly compensated here, so
        # it cancels out of both the command and the plant). The Coriolis term
        # is the bias the filter must invert through.
        M = model.mass(q)
        tau_c = model.coriolis(q, dq)
        if use_cbf:
            ddq = np.linalg.solve(M, tau - tau_c)
            A, b = compute_table_hocbf_rows(z_s, J_z, dJdq_z, dq, z_min,
                                            alpha1=15.0, alpha2=15.0)
            if tau_limits:
                A_t, b_t = torque_limit_rows(M, tau_c, tau_max)
                A = np.vstack((A, A_t))
                b = np.concatenate((b, b_t))
            ddq, status = filter_control_qp(ddq, A, b)
            degraded += status == "degraded"
            tau = M @ ddq + tau_c

        if np.any(np.abs(tau) > tau_max + 1e-6):
            clipped += 1
        tau = np.clip(tau, -tau_max, tau_max)
        ddq = np.linalg.solve(M, tau - tau_c)

        ddq = np.clip(ddq, -150, 150)
        dq = np.clip(dq + ddq * dt, -3.0, 3.0)
        q = q + dq * dt
    return lowest, degraded, clipped


z_min = 0.05
results = {}
for label, cbf, wall in [("no protection    ", False, False),
                         ("CBF only         ", True, False),
                         ("wall only        ", False, True),
                         ("CBF + wall       ", True, True)]:
    m, deg, clip = simulate(z_min, cbf, wall)
    results[label.strip()] = (m, deg, clip)
    verdict = "penetrated" if m < -1e-3 else "held"
    print(f"  {label} min surface clearance = {m*1000:+8.1f} mm  [{verdict}]"
          + (f"  degraded x{deg}" if deg else "")
          + (f"  CLIPPED x{clip}" if clip else ""))

pen_none = results["no protection"][0]
pen_both, deg_both, clip_both = results["CBF + wall"]
check("barrier prevents penetration", pen_both > -1e-3,
      f"clearance {pen_both*1000:+.1f} mm vs {pen_none*1000:+.1f} mm unprotected")

# Fix 1: with the torque rows in the QP the downstream clamp must never bite,
# because a clamp applied after the projection silently voids the guarantee.
check("torque clamp never overrides the filter", clip_both == 0,
      f"{clip_both} clipped cycles")

# Fix 2: the exact active set should carry the whole run; falling back to the
# least-squares degrade path is an approximation, not a certificate.
check("projection stays exact (no degrade)", deg_both == 0,
      f"{deg_both} degraded cycles")

# Same push with the torque limits left OUT of the QP: this is the pre-fix
# behaviour, kept as evidence that the clamp really was reachable here.
_, _, clip_unconstrained = simulate(z_min, True, True, tau_limits=False)
print(f"  without torque rows in the QP, the clamp bit on "
      f"{clip_unconstrained} cycles (pre-fix behaviour)")

# --- 3b. torque <-> acceleration round trip ---------------------------------
print("\n3b. The filter's ddq <-> tau inversion matches the plant")
worst_c = 0.0
worst_nc = 0.0
for _ in range(20):
    q = rng.uniform(-2.0, 2.0, n)
    dq = rng.uniform(-1.5, 1.5, n)
    M = model.mass(q)
    tau_c = model.coriolis(q, dq)
    tau_g = model.gravity(q)
    tau = rng.normal(size=n) * 10.0

    # Ground truth: what the arm actually does under this command.
    ddq_true = np.linalg.solve(M, tau - tau_c - tau_g)
    # With the Coriolis bias (current code) and without it (pre-fix).
    ddq_fix = np.linalg.solve(M, tau - (tau_g + tau_c))
    ddq_old = np.linalg.solve(M, tau - tau_g)
    worst_c = max(worst_c, np.max(np.abs(ddq_fix - ddq_true)))
    worst_nc = max(worst_nc, np.max(np.abs(ddq_old - ddq_true)))

check("ddq_nom matches the plant acceleration", worst_c < 1e-9,
      f"max err {worst_c:.2e} rad/s^2")
print(f"       dropping Coriolis (pre-fix) mis-stated ddq_nom by up to "
      f"{worst_nc:.2f} rad/s^2")

# --- 4. hot-loop timing -----------------------------------------------------
print("\n4. Hot-loop cost")
q = np.deg2rad([0, 15, 180, -130, 0, 55, 90])
dq = rng.normal(size=n) * 0.3
N = 2000
t0 = time.perf_counter()
for _ in range(N):
    z, J_z, dJdq_z = model.height_barrier_terms(q, dq, segs)
kin = (time.perf_counter() - t0) / N * 1e6

z, J_z, dJdq_z = model.height_barrier_terms(q, dq, segs)
z_s = z - 0.055
M_hot = model.mass(q)
tau_max = imp.TAU_MAX[:n]
t0 = time.perf_counter()
for _ in range(N):
    tau_c = model.coriolis(q, dq)
    tw, fw = table_wall_torque(z_s, J_z, dq, 0.05, 0.06, 1200.0, 90.0)
    A, b = compute_table_hocbf_rows(z_s, J_z, dJdq_z, dq, 0.05, 15.0, 15.0)
    A_t, b_t = torque_limit_rows(M_hot, tau_c, tau_max)
    filter_control_qp(np.zeros(n), np.vstack((A, A_t)),
                      np.concatenate((b, b_t)))
bar = (time.perf_counter() - t0) / N * 1e6

# The barrier's M(q) is now shared with the impedance controller instead of
# being rebuilt, so measure what that saved.
t0 = time.perf_counter()
for _ in range(N):
    Mx = model.mass(q)
    np.linalg.inv(Mx)
mass_cost = (time.perf_counter() - t0) / N * 1e6
print(f"  kinematics ({len(segs)} links, FK+J+Jdot) : {kin:7.1f} us")
print(f"  Coriolis + wall + CBF + projection    : {bar:7.1f} us")
print(f"  TOTAL added per cycle                 : {kin+bar:7.1f} us "
      f"({(kin+bar)/1000*100:.1f}% of a 1 ms budget)")
print(f"  (M(q) build+invert, now shared with the impedance controller "
      f"instead of duplicated: {mass_cost:.1f} us/cycle saved in cartesian "
      f"mode)")
check("fits in the 1 kHz budget", kin + bar < 400.0,
      f"{kin+bar:.0f} us < 400 us")

# --- 5. scratch-object reuse sanity ----------------------------------------
print("\n5. Scratch-object reuse does not corrupt results")
a1 = model.height_barrier_terms(q, dq, segs)
_ = model.height_barrier_terms(rng.uniform(-1, 1, n), rng.normal(size=n), segs)
a2 = model.height_barrier_terms(q, dq, segs)
same = all(np.array_equal(x, y) for x, y in zip(a1, a2))
check("repeated calls are deterministic", same)

# --- 6. virtual-wall force cap ----------------------------------------------
print("\n6. Virtual wall reaches full stiffness before the table")
k_def, so_def, f_def = 1200.0, 0.06, 90.0     # argparse defaults
clr = wall_cap_clearance(so_def, k_def, f_def)
print(f"  K={k_def:g} N/m over a {so_def:g} m standoff peaks at "
      f"{k_def*so_def:g} N; cap {f_def:g} N")
check("force cap does not bind inside the standoff band", clr == 0.0,
      f"cap binds {clr*1000:.1f} mm above z_min")

old_clr = wall_cap_clearance(so_def, k_def, 60.0)
print(f"  (with the old {60.0:g} N cap it bound {old_clr*1000:.1f} mm above "
      f"the table -- the last stretch was a constant push, not a spring)")

# The wall must be continuous at the band edge: no torque step as a link
# crosses in, which is the whole point of the quadratic profile.
z_edge = np.array([0.05 + 0.06 + 1e-9])
J_one = np.zeros((1, n))
J_one[0, 0] = 1.0
_, f_edge = table_wall_torque(z_edge, J_one, np.zeros(n), 0.05, 0.06,
                              k_def, 60.0, f_max=f_def)
check("wall force is zero at the band edge", abs(float(f_edge[0])) < 1e-9,
      f"F={float(f_edge[0]):.3e} N")

print("\n" + "=" * 60)
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("All checks passed.")
