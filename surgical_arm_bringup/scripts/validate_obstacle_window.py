#!/usr/bin/env python3
"""Offline validation for the obstacle sliding window. No robot, no camera.

Barrier item 2: `cull_obstacles` bounds how many obstacles the torque loop
pays for, `ObstacleWindow` bounds how many rows the QP sees. Both are only
worth having if bounding them does not quietly remove protection, so:

  1. info["h_dot"] is the barrier's true rate (== Jh @ dq)
  2. capsule_clearance agrees with the rows function's h
  3. ordering: no dropped pair is more urgent than a kept one (beyond
     hysteresis), over random states
  4. a window that keeps everything gives the SAME filtered ddq as no window
  5. cap and ignore-distance are respected; overflow is reported
  6. hysteresis stops near-tie rows from flickering, but not real changes
  7. cull_obstacles keeps exactly the nearest-k obstacles
  8. closed loop: a chain driven into a 40-obstacle field with a 6-row cap
     never penetrates any obstacle, same as enforcing every row
  9. cost of select() and cull at realistic sizes

Run:  python3 validate_obstacle_window.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from control_barrier import (ObstacleWindow, capsule_clearance,   # noqa: E402
                             compute_obstacle_hocbf_rows, cull_obstacles,
                             filter_control_qp)

rng = np.random.default_rng(7)
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


# ── synthetic chain (same construction as validate_obstacle_barrier.py) ─────
N = 7
AXES = np.array([[0, 0, 1.], [0, 1, 0], [0, 0, 1], [0, 1, 0],
                 [0, 0, 1], [0, 1, 0], [1, 0, 0]])
LINKS = np.array([[0, 0, .28], [0, 0, .21], [0, 0, .21], [0, 0, .21],
                  [0, 0, .21], [0, 0, .10], [0, 0, .11]])
MONITORED = (3, 5, 7)          # elbow, wrist, tip


def _rot(axis, angle):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def fk(q, upto=N):
    R, p = np.eye(3), np.zeros(3)
    for i in range(upto):
        R = R @ _rot(AXES[i], q[i])
        p = p + R @ LINKS[i]
    return p


def jac(q, upto=N, h=1e-7):
    J = np.zeros((3, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = h
        J[:, j] = (fk(q + e, upto) - fk(q - e, upto)) / (2 * h)
    return J


def dJdq(q, dq, upto=N, h=1e-6):
    return (jac(q + h * dq, upto) - jac(q - h * dq, upto)) @ dq / (2 * h)


def arm_terms(q, dq):
    p = np.array([fk(q, u) for u in MONITORED])
    J = np.array([jac(q, u) for u in MONITORED])
    dJ = np.array([dJdq(q, dq, u) for u in MONITORED])
    return p, J, dJ


def random_obstacles(k, lo=(-0.6, -0.6, -0.05), hi=(0.6, 0.6, 0.05)):
    c = rng.uniform(lo, hi, (k, 3))
    r = rng.uniform(0.02, 0.08, k)
    hgt = rng.choice([0.0, 0.1, 0.3, np.inf], k)
    return c, r, hgt


LINK_R = np.array([0.06, 0.05, 0.02])

# --- 1 + 2. h_dot and clearance ----------------------------------------------
print("\n1-2. info['h_dot'] and capsule_clearance")
worst_hd = worst_h = 0.0
for _ in range(50):
    q, dq = rng.uniform(-1.2, 1.2, N), rng.uniform(-1.5, 1.5, N)
    p, J, dJ = arm_terms(q, dq)
    c, r, hg = random_obstacles(8)
    A, b, info = compute_obstacle_hocbf_rows(p, J, dJ, dq, c, r, hg, LINK_R)
    worst_hd = max(worst_hd, np.max(np.abs(info["h_dot"].ravel() - (-A @ dq))))
    worst_h = max(worst_h, np.max(np.abs(capsule_clearance(p, c, r, hg, LINK_R) - info["h"])))
check("h_dot == Jh @ dq", worst_hd < 1e-12, f"max err {worst_hd:.1e}")
check("capsule_clearance == rows h", worst_h < 1e-12, f"max err {worst_h:.1e}")

# --- 3. ordering guarantee -----------------------------------------------------
print("\n3. no dropped pair is more urgent than a kept one")
viol = 0
for trial in range(2000):
    m, k = rng.integers(1, 7), rng.integers(1, 12)
    win = ObstacleWindow(max_rows=rng.integers(1, 10), horizon=0.2,
                         ignore_clearance=0.3, hysteresis=0.01)
    for _ in range(3):                         # a few cycles so hysteresis engages
        h = rng.uniform(-0.05, 0.5, (m, k))
        hd = rng.uniform(-1.0, 1.0, (m, k))
        keep, info = win.select(h, hd)
        pred = (h + 0.2 * np.minimum(hd, 0)).ravel()
        dropped = np.setdiff1d(np.flatnonzero(pred < 0.3 - 0.01), keep)
        if keep.size and dropped.size and pred[dropped].min() < pred[keep].max() - 0.01 - 1e-12:
            viol += 1
check("dropped pred >= kept pred - hysteresis", viol == 0, f"({viol} violations in 6000 cycles)")

# --- 4. a window that keeps everything changes nothing ------------------------
print("\n4. full window == no window")
worst = 0.0
n_active = 0
for _ in range(300):
    q, dq = rng.uniform(-1.2, 1.2, N), rng.uniform(-2, 2, N)
    p, J, dJ = arm_terms(q, dq)
    c, r, hg = random_obstacles(6)
    A, b, info = compute_obstacle_hocbf_rows(p, J, dJ, dq, c, r, hg, LINK_R)
    ddq_nom = rng.uniform(-40, 40, N)
    x_full, s_full = filter_control_qp(ddq_nom, A, b)
    keep, _ = ObstacleWindow(max_rows=10**6, ignore_clearance=np.inf).select(info["h"], info["h_dot"])
    x_win, s_win = filter_control_qp(ddq_nom, A[keep], b[keep])
    if s_full in ("exact",) and s_win in ("exact",):
        worst = max(worst, np.max(np.abs(x_full - x_win)))
        n_active += 1
check("same filtered ddq (row order is irrelevant)", worst < 1e-9,
      f"max diff {worst:.1e} over {n_active} active cases")

# --- 5. cap / ignore / overflow ------------------------------------------------
print("\n5. cap, ignore distance, overflow report")
win = ObstacleWindow(max_rows=4, horizon=0.0, ignore_clearance=0.2, hysteresis=0.0)
h = np.array([[0.01, 0.50, 0.03], [0.02, -0.01, 0.25], [0.19, 0.30, 0.04]])
keep, info = win.select(h, np.zeros_like(h))
check("penetrating pair kept first", keep[0] == 4)
check("cap respected", keep.size == 4, f"kept {keep.size}")
check("far pairs never candidates", info["n_candidates"] == 6, f"candidates {info['n_candidates']}")
check("overflow reported with the dropped clearance",
      info["overflow"] and abs(info["dropped_min_pred"] - 0.04) < 1e-12,
      f"dropped_min_pred {info['dropped_min_pred']}")
win = ObstacleWindow(max_rows=4, horizon=0.5, ignore_clearance=0.2, hysteresis=0.0)
keep, info = win.select(np.array([[0.25, 0.25]]), np.array([[-0.2, 0.0]]))
check("closing pair beyond ignore distance is pulled in by the horizon",
      list(keep) == [0], f"kept {list(keep)}")

# --- 6. hysteresis -------------------------------------------------------------
print("\n6. hysteresis")
win = ObstacleWindow(max_rows=1, horizon=0.0, ignore_clearance=1.0, hysteresis=0.01)
flips, last = 0, None
for i in range(200):
    wobble = 0.004 * np.sin(i)                 # near-tie, well under hysteresis
    keep, _ = win.select(np.array([[0.100 + wobble, 0.100 - wobble]]), np.zeros((1, 2)))
    flips += last is not None and keep[0] != last
    last = keep[0]
check("near-tie rows do not flicker", flips == 0, f"{flips} swaps in 200 cycles")
keep, _ = win.select(np.array([[0.100, 0.080]]), np.zeros((1, 2)))
check("a real change (> hysteresis) still swaps", keep[0] == 1)

# --- 7. cull ---------------------------------------------------------------------
print("\n7. cull_obstacles keeps the nearest k")
bad = 0
for _ in range(300):
    q = rng.uniform(-1.2, 1.2, N)
    p = np.array([fk(q, u) for u in MONITORED])
    c, r, hg = random_obstacles(30)
    kk = rng.integers(1, 30)
    idx = cull_obstacles(p, c, r, hg, LINK_R, kk)
    hmin = capsule_clearance(p, c, r, hg, LINK_R).min(axis=0)
    brute = np.sort(hmin)[:kk]
    bad += not np.allclose(np.sort(hmin[idx]), brute)
check("culled set == brute-force nearest k", bad == 0, f"({bad}/300 mismatches)")

# --- 8. closed loop in an obstacle field ---------------------------------------
print("\n8. closed loop: 40 obstacles, 6-row cap vs every row")


def simulate(max_rows, T=2.5, dt=1e-3, seed=3):
    r_ = np.random.default_rng(seed)
    c = np.column_stack([r_.uniform(-0.5, 0.5, 40), r_.uniform(-0.5, 0.5, 40),
                         np.full(40, -0.03)])
    rad = r_.uniform(0.02, 0.05, 40)
    hg = r_.choice([0.05, 0.15, 0.3], 40)
    c[0] = [0.0, 0.0, -0.03]; rad[0] = 0.05; hg[0] = 0.6       # the one in the path
    q = np.array([0.30, 0.45, 0.0, -0.55, 0.0, 0.35, 0.0])
    dq = np.zeros(N)
    tgt = np.array([0.0, 0.0, 0.25])
    win = ObstacleWindow(max_rows=max_rows, horizon=0.2, ignore_clearance=0.30) if max_rows else None
    worst, rows_used, overflow = np.inf, 0, 0
    for it in range(int(T / dt)):
        p, J, dJ = arm_terms(q, dq)
        ddq_nom = np.clip(J[-1].T @ (900.0 * (tgt - p[-1])) - 60.0 * dq, -60, 60)
        if it % 33 == 0:                                         # ~30 Hz perception
            sel = cull_obstacles(p, c, rad, hg, LINK_R, 12)
            if win:
                win.reset()
        A, b, info = compute_obstacle_hocbf_rows(p, J, dJ, dq, c[sel], rad[sel], hg[sel],
                                                 LINK_R, alpha1=20.0, alpha2=20.0)
        if max_rows is None:
            keep = np.arange(A.shape[0])
        else:
            keep, winfo = win.select(info["h"], info["h_dot"])
            overflow += winfo["overflow"]
        rows_used = max(rows_used, keep.size)
        ddq = filter_control_qp(ddq_nom, A[keep], b[keep])[0] if keep.size else ddq_nom
        dq = dq + ddq * dt
        q = q + dq * dt
        p_now = np.array([fk(q, u) for u in MONITORED])
        worst = min(worst, capsule_clearance(p_now, c, rad, hg, LINK_R).min())
    return worst, rows_used, overflow


t0 = time.perf_counter()
w_all, r_all, _ = simulate(None)
w_win, r_win, ovf = simulate(6)
print(f"  every row  : min clearance {w_all*1000:+.3f} mm, up to {r_all} rows")
print(f"  6-row cap  : min clearance {w_win*1000:+.3f} mm, up to {r_win} rows, "
      f"overflow on {ovf} cycles  ({time.perf_counter()-t0:.0f} s)")
check("windowed arm never penetrates any of the 40 obstacles", w_win > -1e-3,
      f"{w_win*1000:+.3f} mm")
check("window never exceeds its cap", r_win <= 6)
check("windowed result matches enforcing every row", abs(w_win - w_all) < 1e-3,
      f"diff {abs(w_win-w_all)*1000:.3f} mm")

# --- 9. cost -----------------------------------------------------------------------
print("\n9. cost")
win = ObstacleWindow(max_rows=12)
h, hd = rng.uniform(0, 0.4, (6, 8)), rng.uniform(-1, 1, (6, 8))
n = 20000
t0 = time.perf_counter()
for _ in range(n):
    win.select(h, hd)
sel_us = (time.perf_counter() - t0) / n * 1e6
p = rng.uniform(-0.5, 0.5, (6, 3))
c, r, hg = random_obstacles(200)
t0 = time.perf_counter()
for _ in range(2000):
    cull_obstacles(p, c, r, hg, 0.05, 8)
cull_us = (time.perf_counter() - t0) / 2000 * 1e6
print(f"  select() 6 points x 8 obstacles: {sel_us:.1f} us/cycle (laptop)")
print(f"  cull 200 obstacles -> 8        : {cull_us:.1f} us/frame (perception rate)")
check("select() well inside a 1 ms cycle", sel_us < 100, f"{sel_us:.1f} us")

print("\n" + ("All checks passed." if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
