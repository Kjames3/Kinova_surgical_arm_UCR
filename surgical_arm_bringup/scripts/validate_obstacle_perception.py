#!/usr/bin/env python3
"""Offline validation for obstacle_perception.py. No robot, no camera.

A depth image is RAY-CAST from a known scene through the real RealSense
extrinsic (cameras.launch.py, 2026-09-03) and intrinsics of the 424x240
aligned stream, so every answer is known exactly:

  table plane, a cylinder, a small box, a 30 cm bar (forces tiling),
  the insertion container (must be excluded), and the ARM reaching down to
  5 cm above the table (must be self-filtered).

Checks:
  1. back-projection + extrinsic reproduce the ray hits exactly
  2. the pipeline finds exactly the three real obstacles
  3. every capsule set is CONSERVATIVE: it covers the whole true object,
     including the back half the camera cannot see
  4. no phantom capsule sits on the arm, the container or empty table
  5. with self-filtering off the arm shows up -- check 2 is not vacuous
  6. with the exclusion off the container shows up
  7. wide objects are tiled; no capsule exceeds max_radius + inflate
  8. still correct with 3 mm depth noise and 2 % dropouts
  9. cost per frame

Run:  python3 validate_obstacle_perception.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from control_barrier import capsule_clearance                        # noqa: E402
from obstacle_perception import (DEFAULT_CFG, depth_to_points,       # noqa: E402
                                 perceive, point_segment_distance, pose_matrix,
                                 transform_points)

rng = np.random.default_rng(5)
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


# ── camera ────────────────────────────────────────────────────────────────────
T_BC = pose_matrix((1.030907, 0.036634, 0.731535),
                   (0.639427, 0.611209, -0.327313, -0.332299))
W, H = 424, 240
K = np.array([[308.0, 0, 212.0], [0, 308.0, 120.0], [0, 0, 1]])
TABLE_Z = -0.030

# ── scene ─────────────────────────────────────────────────────────────────────
CYL = dict(c=(0.50, 0.30), r=0.04, z0=TABLE_Z, z1=0.09)
BOX = dict(lo=(0.30, -0.40, TABLE_Z), hi=(0.36, -0.30, 0.05))
BAR = dict(lo=(0.20, 0.40, TABLE_Z), hi=(0.50, 0.45, 0.01))   # 30 cm along x, fully in view
CONTAINER = dict(c=(0.45, -0.05), r=0.045, z0=TABLE_Z, z1=TABLE_Z + 0.086)
ARM_A = np.array([[0.00, 0.00, 0.15], [0.25, 0.10, 0.45], [0.45, 0.15, 0.35]])
ARM_B = np.array([[0.25, 0.10, 0.45], [0.45, 0.15, 0.35], [0.46, 0.16, 0.02]])
ARM_R = np.array([0.06, 0.06, 0.035])


def ray_plane(o, D, z):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - o[2]) / D[:, 2]
    return np.where(t > 0, t, np.inf)


def ray_vcyl(o, D, c, r, z0, z1):
    oc = o[:2] - np.asarray(c)
    a = (D[:, :2] ** 2).sum(1)
    b = 2 * D[:, :2] @ oc
    cc = oc @ oc - r * r
    disc = b * b - 4 * a * cc
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-b - np.sqrt(disc)) / (2 * a)
    z = o[2] + t * D[:, 2]
    t_side = np.where((disc >= 0) & (t > 0) & (z >= z0) & (z <= z1), t, np.inf)
    t_top = ray_plane(o, D, z1)
    xy = o[:2] + t_top[:, None] * D[:, :2]
    t_top = np.where(((xy - c) ** 2).sum(1) <= r * r, t_top, np.inf)
    return np.minimum(t_side, t_top)


def ray_box(o, D, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (np.asarray(lo) - o) / D
        t2 = (np.asarray(hi) - o) / D
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)


def ray_capsule(o, D, a, b, r):
    """iq's analytic ray-capsule intersection, vectorised over rays."""
    ba, oa = b - a, o - a
    baba, baoa = ba @ ba, ba @ oa
    bard = D @ ba
    rdoa = D @ oa
    dd = (D * D).sum(1)
    k2 = baba * dd - bard * bard
    k1 = baba * rdoa - baoa * bard
    k0 = baba * (oa @ oa) - baoa * baoa - r * r * baba
    h = k1 * k1 - k2 * k0
    out = np.full(len(D), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-k1 - np.sqrt(h)) / k2
        y = baoa + t * bard
        body = (h >= 0) & (y > 0) & (y < baba) & (t > 0)
        out[body] = t[body]
        for cap in (a, b):
            oc = o - cap
            bb = D @ oc
            c = oc @ oc - r * r
            hh = bb * bb - dd * c
            tc = (-bb - np.sqrt(hh)) / dd
            ok = (hh >= 0) & (tc > 0) & ~body
            out[ok] = np.minimum(out[ok], tc[ok])
    return out


def render(with_arm=True):
    v, u = np.mgrid[0:H, 0:W]
    dirs_c = np.column_stack(((u.ravel() - K[0, 2]) / K[0, 0],
                              (v.ravel() - K[1, 2]) / K[1, 1], np.ones(u.size)))
    D = dirs_c @ T_BC[:3, :3].T            # cam z component of each ray is 1 -> t == depth
    o = T_BC[:3, 3]
    t = ray_plane(o, D, TABLE_Z)
    t = np.minimum(t, ray_vcyl(o, D, CYL["c"], CYL["r"], CYL["z0"], CYL["z1"]))
    t = np.minimum(t, ray_box(o, D, BOX["lo"], BOX["hi"]))
    t = np.minimum(t, ray_box(o, D, BAR["lo"], BAR["hi"]))
    t = np.minimum(t, ray_vcyl(o, D, CONTAINER["c"], CONTAINER["r"], CONTAINER["z0"], CONTAINER["z1"]))
    if with_arm:
        for a, b, r in zip(ARM_A, ARM_B, ARM_R):
            t = np.minimum(t, ray_capsule(o, D, a, b, r))
    depth = np.where(np.isfinite(t), t, 0.0).reshape(H, W)
    hits = (o + t[:, None] * D).reshape(H, W, 3)
    return depth, hits


def surface_samples():
    """Points on the FULL surface of each real obstacle, back side included."""
    out = {}
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    zs = np.linspace(CYL["z0"] + 0.016, CYL["z1"], 12)
    out["cylinder"] = np.array([(CYL["c"][0] + CYL["r"] * np.cos(a), CYL["c"][1] + CYL["r"] * np.sin(a), z)
                                for a in ang for z in zs])
    for name, box in (("box", BOX), ("bar", BAR)):
        lo, hi = np.array(box["lo"]), np.array(box["hi"])
        g = rng.uniform(0, 1, (3000, 3))
        p = lo + g * (hi - lo)
        face = rng.integers(0, 2, 3000)
        axis = rng.integers(0, 2, 3000)                     # x or y faces + top
        p[np.arange(3000), axis] = np.where(face == 1, hi[axis], lo[axis])
        top = rng.uniform(0, 1, 500)[:, None] * (hi - lo) + lo
        top[:, 2] = hi[2]
        pts = np.vstack((p, top))
        out[name] = pts[pts[:, 2] > TABLE_Z + 0.016]
    return out


CFG = dict(DEFAULT_CFG)
CFG.update(table_z=TABLE_Z, exclude=[(CONTAINER["c"][0], CONTAINER["c"][1], 0.08)])


def run(depth, cfg=CFG, arm=True):
    a, b, r = (ARM_A, ARM_B, ARM_R) if arm else (np.zeros((0, 3)),) * 2 + (np.zeros(0),)
    return perceive(depth, K, T_BC, a, b, r, cfg)


def coverage(centers, radii, heights):
    worst = {}
    for name, pts in surface_samples().items():
        h = capsule_clearance(pts, centers, radii, heights, 0.0)
        worst[name] = float(h.min(axis=1).max())          # worst-covered surface point
    return worst


def assign(centers):
    """Which true object each capsule centre belongs to (or 'phantom')."""
    labels = []
    for c in centers:
        d = {"cylinder": np.hypot(c[0] - CYL["c"][0], c[1] - CYL["c"][1]) - CYL["r"]}
        for name, box in (("box", BOX), ("bar", BAR)):
            lo, hi = np.array(box["lo"][:2]), np.array(box["hi"][:2])
            d[name] = float(np.linalg.norm(np.maximum(np.maximum(lo - c[:2], c[:2] - hi), 0)))
        name = min(d, key=d.get)
        labels.append(name if d[name] < 0.06 else "phantom")
    return labels


# --- 1. back-projection -----------------------------------------------------------
print("\n1. back-projection and extrinsic")
depth, hits = render()
pts_c, valid = depth_to_points(depth, K, stride=1, min_range=0.0, max_range=np.inf)
ok_px = depth.ravel() > 0
err = np.abs(transform_points(pts_c, T_BC) - hits.reshape(-1, 3)[ok_px]).max()
check("points == ray hits", err < 1e-9, f"max err {err:.1e} m over {ok_px.sum()} px")

# --- 2-4. the clean scene -----------------------------------------------------------
print("\n2-4. clean scene")
t0 = time.perf_counter()
centers, radii, heights, stats = run(depth)
dt_ms = (time.perf_counter() - t0) * 1e3
labels = assign(centers)
print(f"  stats {stats}")
for c, r, h, lab in zip(centers, radii, heights, labels):
    print(f"    capsule ({c[0]:+.3f}, {c[1]:+.3f})  r={r*1000:.0f} mm  h={h*1000:.0f} mm  -> {lab}")
check("three real obstacles, all found", set(labels) >= {"cylinder", "box", "bar"} and stats["clusters"] == 3,
      f"clusters={stats['clusters']}")
cov = coverage(centers, radii, heights)
check("conservative: whole surface of every object inside a capsule",
      all(v <= 0.0 for v in cov.values()),
      "worst clearance " + ", ".join(f"{k} {v*1000:+.1f} mm" for k, v in cov.items()))
check("no phantom capsules", "phantom" not in labels)
arm_d = point_segment_distance(centers, ARM_A, ARM_B) - ARM_R[None, :]
check("no capsule on the arm", arm_d.min() > 0.05, f"nearest {arm_d.min()*1000:.0f} mm")
cont_d = np.hypot(centers[:, 0] - CONTAINER["c"][0], centers[:, 1] - CONTAINER["c"][1]).min()
check("no capsule on the excluded container", cont_d > 0.08, f"nearest {cont_d*1000:.0f} mm")
check("arm points were actually in the slab and removed", stats["self_removed"] > 20,
      f"removed {stats['self_removed']}")

# --- 5-6. the filters are load-bearing ----------------------------------------------
print("\n5-6. filters are load-bearing")
_, _, _, s_noarm = run(depth, arm=False)
check("self-filter off -> the arm appears as an obstacle", s_noarm["clusters"] > stats["clusters"],
      f"{s_noarm['clusters']} clusters vs {stats['clusters']}")
cfg_noexcl = dict(CFG, exclude=[])
c2, _, _, s_noexcl = run(depth, cfg_noexcl)
near = np.hypot(c2[:, 0] - CONTAINER["c"][0], c2[:, 1] - CONTAINER["c"][1]).min()
check("exclusion off -> the container appears", near < 0.06, f"nearest capsule {near*1000:.0f} mm")

# --- 7. tiling -------------------------------------------------------------------------
print("\n7. tiling")
bar_caps = [i for i, lab in enumerate(labels) if lab == "bar"]
check("30 cm bar is split into several capsules", len(bar_caps) >= 2, f"{len(bar_caps)} capsules")
check("no capsule wider than max_radius + inflate",
      radii.max() <= CFG["max_radius"] + CFG["inflate"] + 1e-9, f"max r {radii.max()*1000:.0f} mm")

# --- 8. noise ---------------------------------------------------------------------------
print("\n8. 3 mm depth noise + 2 % dropouts (5 frames)")
ok_all, worst_cov = True, -np.inf
for k in range(5):
    noisy = depth + rng.normal(0, 0.003, depth.shape) * (depth > 0)
    noisy[rng.uniform(size=depth.shape) < 0.02] = 0.0
    cn, rn, hn, sn = run(noisy)
    ln = assign(cn)
    ok_all &= set(ln) >= {"cylinder", "box", "bar"} and "phantom" not in ln and sn["clusters"] == 3
    worst_cov = max(worst_cov, max(coverage(cn, rn, hn).values()))
check("same three obstacles, no phantoms, every frame", ok_all)
check("still conservative under noise", worst_cov <= 0.0, f"worst {worst_cov*1000:+.1f} mm")

# --- 9. cost ------------------------------------------------------------------------------
print("\n9. cost")
n = 10
t0 = time.perf_counter()
for _ in range(n):
    run(depth)
ms = (time.perf_counter() - t0) / n * 1e3
print(f"  perceive() 424x240, stride 2: {ms:.1f} ms/frame (laptop); camera runs at 15 Hz = 66 ms")
check("fits the camera frame period", ms < 66.0, f"{ms:.1f} ms")

print("\n" + ("All checks passed." if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
