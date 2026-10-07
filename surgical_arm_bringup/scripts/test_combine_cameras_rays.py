#!/usr/bin/env python3
"""Offline test for the ray-based fusion in combine_cameras.py (fusion_mode:=rays):
a marker on a table seen by an overhead camera and a table-height side camera,
with a known answer.  Checks the centre ray is exact under perspective and lens
distortion, that one ray on the plane and two free rays both recover the marker,
that a grazing lone view is reported as ill-conditioned, and that with realistic
PnP range error the ray fusion beats averaging positions.

  source /opt/ros/humble/setup.bash && ./test_combine_cameras_rays.py
"""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combine_cameras as cc

fails = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond: fails.append(name)

def look_at(origin, target):
    """Camera-to-world rotation with optical z towards target."""
    z = np.asarray(target, float) - origin; z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
    return np.column_stack([x, np.cross(z, x), z])

K = np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1]])
D = np.array([0.12, -0.25, 0.001, -0.0008, 0.09])
MARKER = np.array([0.40, 0.05, 0.05]); SIZE = 0.05
h = SIZE / 2
CORNERS = MARKER + np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
CAMS = {"front": np.array([1.03, 0.04, 0.73]),      # overhead-ish, as calibrated 2026-09-03
        "side":  np.array([0.45, -0.90, 0.12])}     # table height, to the robot's right
R = {n: look_at(o, MARKER) for n, o in CAMS.items()}

def pixels(name, pts):
    rvec, _ = cv2.Rodrigues(R[name].T)
    uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, -R[name].T @ CAMS[name], K, D)
    return uv.reshape(-1, 2)

rays = {}
for name in CAMS:
    ray_cam = cc.marker_centre_ray(pixels(name, CORNERS), K, D)
    rays[name] = R[name] @ ray_cam
    true = (MARKER - CAMS[name]) / np.linalg.norm(MARKER - CAMS[name])
    err = np.degrees(np.arccos(np.clip(rays[name] @ true, -1, 1)))
    check(f"{name}: centre ray is exact", err < 1e-3, f"{err:.5f} deg")

p, res, cond = cc.solve_rays([CAMS["front"]], [rays["front"]], plane_z=MARKER[2])
check("one overhead ray on the plane recovers the marker", np.linalg.norm(p - MARKER) < 1e-5, np.round(p, 5).tolist())
p, res, cond = cc.solve_rays(list(CAMS.values()), [rays["front"], rays["side"]])
check("two free rays triangulate the marker, height included", np.linalg.norm(p - MARKER) < 1e-5, np.round(p, 5).tolist())
check("exact rays leave no miss distance", res.max() < 1e-6, f"{1000*res.max():.5f} mm")
check("two rays, no plane: well conditioned", cond > 0.2, f"{cond:.3f}")
lone = cc.solve_rays([CAMS["side"]], [rays["side"]], plane_z=MARKER[2])
check("lone grazing ray on the plane is flagged ill-conditioned", lone is None or lone[2] < 0.02,
      "rejected" if lone is None else f"{lone[2]:.4f}")
check("one ray without a plane is refused", cc.solve_rays([CAMS["front"]], [rays["front"]]) is None)
check("a point behind the camera is refused",
      cc.solve_rays([CAMS["front"]], [-rays["front"]], plane_z=MARKER[2]) is None)

# Realistic errors: PnP on a 5 cm marker gets range wrong by a few percent but
# direction right to ~0.1 deg.  Compare both fusions over 400 draws.
rng = np.random.default_rng(1)
err_pos, err_ray, err_front = [], [], []
for _ in range(400):
    positions, noisy, origins, weights = [], [], [], []
    for name, o in CAMS.items():
        d = rays[name] + np.radians(0.1) * rng.standard_normal(3)
        d /= np.linalg.norm(d)
        rng_true = np.linalg.norm(MARKER - o)
        positions.append(o + d * rng_true * (1 + 0.03 * rng.standard_normal()))
        noisy.append(d); origins.append(o); weights.append(1 / rng_true ** 2)
    d0, d1 = (np.linalg.norm(q) for q in positions)          # the node's 2-camera rule
    avg = (positions[0] / d0 + positions[1] / d1) / (1 / d0 + 1 / d1)
    sol = cc.solve_rays(origins, noisy, weights, plane_z=MARKER[2])
    one = cc.solve_rays(origins[:1], noisy[:1], plane_z=MARKER[2])
    err_pos.append(np.linalg.norm(avg[:2] - MARKER[:2]))
    err_ray.append(np.linalg.norm(sol[0][:2] - MARKER[:2]))
    err_front.append(np.linalg.norm(one[0][:2] - MARKER[:2]))
rms = lambda e: 1000 * float(np.sqrt(np.mean(np.square(e))))
print(f"      xy RMS: position average {rms(err_pos):.1f} mm | front ray on plane {rms(err_front):.1f} mm | both rays on plane {rms(err_ray):.1f} mm")
check("ray fusion beats position averaging", rms(err_ray) < 0.5 * rms(err_pos))
check("the side camera improves on the front camera alone", rms(err_ray) < rms(err_front))

print("\n%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
