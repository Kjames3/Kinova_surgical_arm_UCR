#!/usr/bin/env python3
"""Offline validation for the obstacle (capsule) barrier. No robot, no camera.

Deliberately does NOT import impedance.py or KDL: this checks the barrier
ALGEBRA, and mixing in the real chain would mean a KDL bug and a barrier bug
land in the same number. `validate_table_barrier.py` covers the KDL side.
The chain here is a synthetic 7-joint arm whose Jacobian and dJ@dq are
differentiated numerically to machine precision, so any mismatch below is the
barrier's fault.

Checks:
  1. Jh == d(h)/dq                    -- the constraint row is the true gradient
  2. drift == h_ddot - Jh @ ddq       -- the term with no analogue in the plane
  3. the curvature term is not negligible at realistic speeds
  4. sphere / infinite cylinder / finite capsule agree with hand geometry
  5. reach-over: a finite capsule does not constrain a point above it
  6. closed-loop -- a chain driven at an obstacle is stopped, not slowed
  7. degenerate cases are reported, not silently passed off as protection
  8. timing at a realistic point x obstacle count

Run:  python3 validate_obstacle_barrier.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from control_barrier import (compute_obstacle_hocbf_rows,   # noqa: E402
                             filter_control_qp)

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(11)
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


# ── synthetic chain ─────────────────────────────────────────────────────────
# A 7-joint arm of unit-ish links with alternating joint axes. Only three
# things matter: it is nonlinear in q, it spans 3-D, and we can differentiate
# it exactly by finite differences.
N = 7
AXES = np.array([[0, 0, 1.], [0, 1, 0], [0, 0, 1], [0, 1, 0],
                 [0, 0, 1], [0, 1, 0], [1, 0, 0]])
LINKS = np.array([[0, 0, .28], [0, 0, .21], [0, 0, .21], [0, 0, .21],
                  [0, 0, .21], [0, 0, .10], [0, 0, .11]])


def _rot(axis, angle):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def fk(q, upto=N):
    """Position of the frame after joint `upto`, base frame."""
    R = np.eye(3)
    p = np.zeros(3)
    for i in range(upto):
        R = R @ _rot(AXES[i], q[i])
        p = p + R @ LINKS[i]
    return p


def jac(q, upto=N, h=1e-7):
    J = np.zeros((3, N))
    for j in range(N):
        qp, qm = q.copy(), q.copy()
        qp[j] += h
        qm[j] -= h
        J[:, j] = (fk(qp, upto) - fk(qm, upto)) / (2 * h)
    return J


def dJdq(q, dq, upto=N, h=1e-6):
    """(dJ/dt) @ dq, by differencing J along the motion. Cheap, for the sim."""
    return (jac(q + h * dq, upto) - jac(q - h * dq, upto)) @ dq / (2 * h)


# Differencing a Jacobian that is ITSELF a finite difference costs about four
# digits: the cheap dJ@dq above is only good to ~5e-4 m/s^2, which is larger
# than the residual the accuracy checks are trying to resolve. Those checks use
# the 4th-order versions below so the harness is not the thing being measured.
# On the robot dJ@dq is exact from KDL's ChainJntToJacDotSolver, validated in
# validate_table_barrier.py -- it is an INPUT here, not something this file
# is responsible for producing.
def jac_hi(q, upto=N, h=1e-5):
    J = np.zeros((3, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        J[:, j] = (-fk(q + 2 * h * e, upto) + 8 * fk(q + h * e, upto)
                   - 8 * fk(q - h * e, upto) + fk(q - 2 * h * e, upto)) / (12 * h)
    return J


def dJdq_hi(q, dq, upto=N, h=1e-4):
    return (-jac_hi(q + 2 * h * dq, upto) + 8 * jac_hi(q + h * dq, upto)
            - 8 * jac_hi(q - h * dq, upto)
            + jac_hi(q - 2 * h * dq, upto)) @ dq / (12 * h)


def barrier_value(q, center, radius, height, link_radius, upto=N):
    """Ground-truth capsule clearance, computed independently of the module."""
    p = fk(q, upto)
    t = p[2] - center[2]
    s = min(max(t, 0.0), height)
    closest = np.array([center[0], center[1], center[2] + s])
    return np.linalg.norm(p - closest) - (radius + link_radius)


OBST = dict(center=np.array([0.35, 0.10, -0.03]), radius=0.05,
            height=0.18, link_radius=0.055)


def rows_for(q, dq, obst=OBST, precise=False, **kw):
    p = fk(q)[None, :]
    J = (jac_hi if precise else jac)(q)
    dJ = (dJdq_hi if precise else dJdq)(q, dq)
    A, b, info = compute_obstacle_hocbf_rows(
        p, J[None, :, :], dJ[None, :], dq,
        obst["center"][None, :], np.array([obst["radius"]]),
        np.array([obst["height"]]), np.array([obst["link_radius"]]), **kw)
    return A, b, info


def sample_q(obst=OBST, lo=0.06, hi=0.60):
    """Random configurations whose tip sits a sane distance from the capsule."""
    while True:
        q = rng.uniform(-1.2, 1.2, N)
        h = barrier_value(q, obst["center"], obst["radius"], obst["height"],
                          obst["link_radius"])
        if lo < h < hi:
            return q


# --- 1. Jh vs numerical d(h)/dq ---------------------------------------------
print("\n1. constraint row Jh vs numerical gradient of the barrier")
worst = 0.0
for _ in range(40):
    q = sample_q()
    A, _, _ = rows_for(q, np.zeros(N))
    Jh = -A[0]                                   # A = -Jh
    num = np.zeros(N)
    eps = 1e-7
    for j in range(N):
        qp, qm = q.copy(), q.copy()
        qp[j] += eps
        qm[j] -= eps
        num[j] = (barrier_value(qp, OBST["center"], OBST["radius"],
                                OBST["height"], OBST["link_radius"])
                  - barrier_value(qm, OBST["center"], OBST["radius"],
                                  OBST["height"], OBST["link_radius"])) / (2 * eps)
    worst = max(worst, np.max(np.abs(Jh - num)))
check("Jh == d(h)/dq", worst < 1e-5, f"max err {worst:.2e}")

# --- 2. drift is the exact ddq-independent part of h_ddot -------------------
# h_ddot = Jh @ ddq + drift. Differentiate h twice along a real trajectory and
# compare. This is the term the plane barrier does not have.
print("\n2. drift == the ddq-independent part of h_ddot")
# Differentiating the gradient ONCE is far better conditioned than
# differencing h twice: h_ddot = (dJh/dt) @ dq + Jh @ ddq, so drift is
# exactly (dJh/dt) @ dq. Check 1 already established Jh is the true gradient,
# so this leans on nothing unverified.
def Jh_of(qq):
    A, _, _ = rows_for(qq, np.zeros(N), precise=True)
    return -A[0]

# Sweep the differencing step rather than trusting one: drift_num is itself a
# finite difference of an already-differenced Jacobian, so a fixed eps floors
# out around 1e-5 on harness noise alone. Requiring the BEST eps to be tiny is
# the honest form of this test -- a real error in drift would not shrink.
worst = np.inf
for _ in range(12):
    q = sample_q()
    dq = rng.uniform(-1.5, 1.5, N)
    A, b, _ = rows_for(q, dq, precise=True)
    Jh = -A[0]
    h1 = barrier_value(q, OBST["center"], OBST["radius"], OBST["height"],
                       OBST["link_radius"])
    h1_dot = Jh @ dq
    drift = b[0] - 10.0 * h1_dot - 10.0 * (h1_dot + 10.0 * h1)
    best = min(abs(drift - (Jh_of(q + eps * dq) - Jh_of(q - eps * dq)) @ dq
                   / (2 * eps))
               for eps in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5))
    worst = min(worst, best) if worst is np.inf else max(worst, best)
check("drift == (dJh/dt) @ dq", worst < 1e-6,
      f"worst-over-samples of best-over-eps: {worst:.2e} m/s^2")

# And confirm the whole row reproduces h_ddot end to end. A second difference
# is roundoff-dominated (error ~ eps*|h|/dt^2), so a fixed dt proves nothing --
# sweep dt and require the error to actually bottom out near zero. A genuine
# modelling error would floor at a constant instead of dipping.
q = sample_q()
dq = rng.uniform(-1.0, 1.0, N)
ddq = rng.uniform(-2.0, 2.0, N)
A, b, _ = rows_for(q, dq, precise=True)
Jh = -A[0]
h1 = barrier_value(q, OBST["center"], OBST["radius"], OBST["height"],
                   OBST["link_radius"])
h1_dot = Jh @ dq
drift = b[0] - 10.0 * h1_dot - 10.0 * (h1_dot + 10.0 * h1)
predicted = Jh @ ddq + drift

errs = []
for dt in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
    def h_at(k):
        qk = q + k * dt * dq + 0.5 * (k * dt) ** 2 * ddq
        return barrier_value(qk, OBST["center"], OBST["radius"],
                             OBST["height"], OBST["link_radius"])
    num = (h_at(1) - 2 * h_at(0) + h_at(-1)) / dt ** 2
    errs.append((dt, abs(predicted - num)))
for dt, e in errs:
    print(f"        dt={dt:.0e}  |error| = {e:.2e}")
check("Jh @ ddq + drift == h_ddot (best dt)", min(e for _, e in errs) < 1e-5,
      f"best {min(e for _, e in errs):.2e} m/s^2 of {abs(predicted):.3f} predicted")

# --- 3. does the curvature term actually matter? ----------------------------
print("\n3. size of the curvature term (the piece a plane barrier lacks)")
worst = 0.0
for _ in range(400):
    q = sample_q()
    dq = rng.uniform(-2.0, 2.0, N)
    p = fk(q)
    J = jac(q)
    t = p[2] - OBST["center"][2]
    on_shaft = 0.0 < t < OBST["height"]
    mask = np.array([1.0, 1.0, 0.0 if on_shaft else 1.0])
    v = (J @ dq) * mask
    s = min(max(t, 0.0), OBST["height"])
    e = p - np.array([OBST["center"][0], OBST["center"][1],
                      OBST["center"][2] + s])
    d = np.linalg.norm(e)
    nrm = e / d
    worst = max(worst, (v @ v - (nrm @ v) ** 2) / d)
check("curvature term is non-negligible", worst > 0.5,
      f"max {worst:.2f} m/s^2 at |dq| <= 2 rad/s")

# --- 4. geometry sanity for the three capsule cases -------------------------
print("\n4. sphere / infinite cylinder / finite capsule geometry")
P = np.array([[0.40, 0.0, 0.30]])
Jz = np.zeros((1, 3, N))
Jz[0, :, 0] = np.eye(3)[:, 0]
c = np.array([[0.10, 0.0, 0.0]])
_, _, i_sph = compute_obstacle_hocbf_rows(P, Jz, np.zeros((1, 3)), np.zeros(N),
                                          c, np.array([0.05]), np.array([0.0]))
_, _, i_inf = compute_obstacle_hocbf_rows(P, Jz, np.zeros((1, 3)), np.zeros(N),
                                          c, np.array([0.05]),
                                          np.array([np.inf]))
_, _, i_cap = compute_obstacle_hocbf_rows(P, Jz, np.zeros((1, 3)), np.zeros(N),
                                          c, np.array([0.05]),
                                          np.array([0.20]))
d_sphere = np.linalg.norm(P[0] - c[0]) - 0.05           # full 3-D distance
d_cyl = np.hypot(P[0, 0] - c[0, 0], P[0, 1] - c[0, 1]) - 0.05
d_capsule = np.linalg.norm(P[0] - np.array([0.10, 0.0, 0.20])) - 0.05
check("sphere (height 0)", abs(i_sph["h"][0, 0] - d_sphere) < 1e-12,
      f"{i_sph['h'][0,0]:.6f} vs {d_sphere:.6f}")
check("infinite cylinder", abs(i_inf["h"][0, 0] - d_cyl) < 1e-12,
      f"{i_inf['h'][0,0]:.6f} vs {d_cyl:.6f}")
check("finite capsule uses the top cap", abs(i_cap["h"][0, 0] - d_capsule) < 1e-12,
      f"{i_cap['h'][0,0]:.6f} vs {d_capsule:.6f}")

# --- 5. reach-over ----------------------------------------------------------
print("\n5. reach-over: a finite capsule must not wall off the column above it")
above = np.array([[0.10, 0.0, 0.45]])                    # directly over it
_, _, i_fin = compute_obstacle_hocbf_rows(above, Jz, np.zeros((1, 3)),
                                          np.zeros(N), c, np.array([0.05]),
                                          np.array([0.20]))
_, _, i_infinite = compute_obstacle_hocbf_rows(above, Jz, np.zeros((1, 3)),
                                               np.zeros(N), c,
                                               np.array([0.05]),
                                               np.array([np.inf]))
check("finite capsule leaves clearance overhead",
      i_fin["h"][0, 0] > 0.15,
      f"h = {i_fin['h'][0,0]*1000:.0f} mm above a 200 mm object")
check("infinite cylinder would forbid it",
      i_infinite["h"][0, 0] < 0.0,
      f"h = {i_infinite['h'][0,0]*1000:.0f} mm (negative = blocked)")

# --- 6. closed loop: drive the tip straight at the obstacle -----------------
print("\n6. closed loop -- chain commanded into the obstacle")


def simulate(use_cbf, T=2.0, dt=1e-3):
    q = np.array([0.30, 0.45, 0.0, -0.55, 0.0, 0.35, 0.0])
    dq = np.zeros(N)
    obst = dict(center=np.array([0.0, 0.0, -0.03]), radius=0.05,
                height=0.60, link_radius=0.055)
    tgt = np.array([obst["center"][0], obst["center"][1], 0.25])
    worst_h = np.inf
    for _ in range(int(T / dt)):
        p = fk(q)
        J = jac(q)
        # crude resolved-rate drive straight at the obstacle axis
        ddq_nom = (J.T @ (900.0 * (tgt - p)) - 60.0 * dq)
        ddq_nom = np.clip(ddq_nom, -60, 60)
        if use_cbf:
            A, b, _ = compute_obstacle_hocbf_rows(
                p[None, :], J[None, :, :], dJdq(q, dq)[None, :], dq,
                obst["center"][None, :], np.array([obst["radius"]]),
                np.array([obst["height"]]), np.array([obst["link_radius"]]),
                alpha1=20.0, alpha2=20.0)
            ddq, _ = filter_control_qp(ddq_nom, A, b)
        else:
            ddq = ddq_nom
        dq = dq + ddq * dt
        q = q + dq * dt
        worst_h = min(worst_h, barrier_value(q, obst["center"], obst["radius"],
                                             obst["height"],
                                             obst["link_radius"]))
    return worst_h


def simulate_standoff(standoff, T=2.0, dt=1e-3):
    """Same drive, but the capsule is inflated by `standoff` for the filter.

    Scored against the TRUE surface, so a positive result means real clearance.
    """
    q = np.array([0.30, 0.45, 0.0, -0.55, 0.0, 0.35, 0.0])
    dq = np.zeros(N)
    obst = dict(center=np.array([0.0, 0.0, -0.03]), radius=0.05,
                height=0.60, link_radius=0.055)
    tgt = np.array([obst["center"][0], obst["center"][1], 0.25])
    worst_h = np.inf
    for _ in range(int(T / dt)):
        p = fk(q)
        J = jac(q)
        ddq_nom = np.clip(J.T @ (900.0 * (tgt - p)) - 60.0 * dq, -60, 60)
        A, b, _ = compute_obstacle_hocbf_rows(
            p[None, :], J[None, :, :], dJdq(q, dq)[None, :], dq,
            obst["center"][None, :], np.array([obst["radius"]]),
            np.array([obst["height"]]),
            np.array([obst["link_radius"] + standoff]),
            alpha1=20.0, alpha2=20.0)
        ddq, _ = filter_control_qp(ddq_nom, A, b)
        dq = dq + ddq * dt
        q = q + dq * dt
        worst_h = min(worst_h, barrier_value(q, obst["center"], obst["radius"],
                                             obst["height"],
                                             obst["link_radius"]))
    return worst_h


h_off = simulate(False)
h_on = simulate(True)
h_standoff = simulate_standoff(0.010)
print(f"  no filter        : min clearance {h_off*1000:+12.6f} mm")
print(f"  CBF, no standoff : min clearance {h_on*1000:+12.6f} mm")
print(f"  CBF + 10 mm stand: min clearance {h_standoff*1000:+12.6f} mm")
check("unfiltered drive really does penetrate", h_off < -0.005,
      "(otherwise the test proves nothing)")
# A CBF forbids further approach at the boundary but applies no restoring
# force, so h -> 0 asymptotically is CORRECT, not a near miss. Demanding real
# clearance from the constraint alone is the mistake that made the table
# barrier feel slack. Require only that it never crosses.
check("CBF never crosses the surface", h_on > -1e-9,
      f"{h_on*1000:+.3e} mm -- rides the boundary, by construction")
check("standoff buys real clearance", h_standoff > 0.008,
      f"{h_standoff*1000:+.3f} mm of a 10 mm standoff retained")

# --- 7. degenerate handling -------------------------------------------------
print("\n7. degenerate pairs are reported, not silently 'safe'")
at_axis = np.array([[0.10, 0.0, 0.10]])
A_d, b_d, i_d = compute_obstacle_hocbf_rows(at_axis, Jz, np.zeros((1, 3)),
                                            np.zeros(N), c, np.array([0.05]),
                                            np.array([0.20]))
check("point on the axis is flagged", i_d["degenerate"] == 1,
      f"degenerate={i_d['degenerate']}")
check("its clearance reads negative", i_d["h"][0, 0] < 0,
      f"h = {i_d['h'][0,0]*1000:.1f} mm")
check("no NaN or inf escapes",
      np.all(np.isfinite(A_d)) and np.all(np.isfinite(b_d)))
_, _, i_ok = compute_obstacle_hocbf_rows(P, Jz, np.zeros((1, 3)), np.zeros(N),
                                         c, np.array([0.05]), np.array([0.20]))
check("well-conditioned pair is not flagged", i_ok["degenerate"] == 0)

# --- 8. timing --------------------------------------------------------------
print("\n8. hot-loop cost (6 links x 8 obstacles)")
m, k = 6, 8
Pm = rng.uniform(-0.5, 0.5, (m, 3))
Jm = rng.uniform(-1, 1, (m, 3, N))
dJm = rng.uniform(-1, 1, (m, 3))
cm = rng.uniform(-0.5, 0.5, (k, 3))
rm = np.full(k, 0.05)
hm = np.full(k, 0.2)
dqm = rng.uniform(-1, 1, N)
t0 = time.perf_counter()
REP = 2000
for _ in range(REP):
    compute_obstacle_hocbf_rows(Pm, Jm, dJm, dqm, cm, rm, hm, 0.055)
us = (time.perf_counter() - t0) / REP * 1e6
check(f"row build under 150 us", us < 150.0, f"{us:.1f} us for {m*k} rows")

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
