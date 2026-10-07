#!/usr/bin/env python3
"""Offline test for validate_handeye_extrinsic.py --mode wrist: synthesises a
flat table marker seen by a wrist camera with a KNOWN eye-in-hand transform from
eight arm poses, then checks the scorer ranks the true transform first with a
~zero spread, puts the marker where it really is, and survives an npz round trip.

  source /opt/ros/humble/setup.bash && ./test_validate_wrist_handeye.py
"""
import os, sys, tempfile
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_handeye_extrinsic as v

K = np.array([[1297.67, 0, 620.91], [0, 1298.63, 238.28], [0, 0, 1.0]])
D = np.zeros((1, 5))
SIZE, TABLE_Z = 0.05, -0.03
MARKERS = {0: np.array([0.40, 0.05, TABLE_Z]), 1: np.array([0.47, -0.04, TABLE_Z])}
candidates = {name: v.parse_handeye(name)[1] for name in v.WRIST_HANDEYE}
TRUE = "urdf"

def look_down(xyz, roll, pitch, yaw):
    """Effector pose with the tool axis roughly straight down."""
    return v.homogeneous(v.rpy_to_matrix(np.pi + roll, pitch, yaw), xyz)

rng = np.random.default_rng(0)
samples = []
for k in range(8):
    T_base_ee = look_down([0.42 + 0.05*rng.uniform(-1, 1), 0.04*rng.uniform(-1, 1),
                           0.30 + 0.06*rng.uniform(-1, 1)],
                          0.25*rng.uniform(-1, 1), 0.25*rng.uniform(-1, 1), rng.uniform(-1.0, 1.0))
    T_cam_base = np.linalg.inv(T_base_ee @ candidates[TRUE])
    markers = {}
    for marker_id, centre in MARKERS.items():
        h = SIZE / 2
        world = centre + np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        cam = (T_cam_base[:3, :3] @ world.T).T + T_cam_base[:3, 3]
        uv = (K @ (cam / cam[:, 2:3]).T).T[:, :2]
        if (cam[:, 2] > 0.05).all() and (uv > 0).all() and (uv[:, 0] < 1280).all() and (uv[:, 1] < 720).all():
            markers[marker_id] = uv
    if markers:
        samples.append({"T_base_ee": T_base_ee, "markers": markers})
assert len(samples) >= 5, f"synthetic poses see the markers only {len(samples)} times"

fails = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond: fails.append(name)

scores = v.score_wrist_samples(samples, K, D, candidates, SIZE, TABLE_Z)
best = min((k for k in scores if scores[k]), key=lambda k: scores[k]["spread_rms_mm"])
check("true transform ranks first", best == TRUE, best)
check("true transform spread ~ 0", scores[TRUE]["spread_rms_mm"] < 0.1, f"{scores[TRUE]['spread_rms_mm']:.4f} mm")
check("true transform: marker flat", scores[TRUE]["tilt_deg"] < 0.1, f"{scores[TRUE]['tilt_deg']:.3f} deg")
check("true transform: marker on the table", scores[TRUE]["z_err_mm"] < 0.1, f"{scores[TRUE]['z_err_mm']:.4f} mm")
for marker_id, centre in MARKERS.items():
    got = scores[TRUE]["per_marker"][marker_id]["mean"]
    check(f"marker {marker_id} recovered in base_link", np.linalg.norm(got - centre) < 1e-4, np.round(got, 4).tolist())
# 16 mm along the tool axis: a pure offset moves every observation together when the
# tool only translates, so the arm must also ROTATE for it to show up as spread.
check("16 mm tool-axis error is visible", scores["nominal_z13"]["spread_rms_mm"] > 1.0, f"{scores['nominal_z13']['spread_rms_mm']:.2f} mm")
check("41 deg error is gross", scores["easy_handeye2"]["spread_rms_mm"] > 30.0, f"{scores['easy_handeye2']['spread_rms_mm']:.1f} mm")

path = os.path.join(tempfile.mkdtemp(prefix="wristtest_"), "capture.npz")
v.save_wrist_samples(path, samples, K, D)
loaded, K2, D2 = v.load_wrist_samples(path)
again = v.score_wrist_samples(loaded, K2, D2, candidates, SIZE, TABLE_Z)
check("npz round trip reproduces the scores",
      all(abs(again[k]["spread_rms_mm"] - scores[k]["spread_rms_mm"]) < 1e-9 for k in scores))
label, T = v.parse_handeye("0.1,0.2,0.3,0,0,0,1")
check("custom x,y,z,quat candidate parses", label == "custom" and np.allclose(T[:3, 3], [0.1, 0.2, 0.3]) and np.allclose(T[:3, :3], np.eye(3)))
check("urdf and nominal_z13 share one orientation",
      np.allclose(candidates["urdf"][:3, :3], candidates["nominal_z13"][:3, :3], atol=1e-9))

print("\n%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
