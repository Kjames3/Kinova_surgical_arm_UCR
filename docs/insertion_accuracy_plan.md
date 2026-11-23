# Insertion accuracy: findings and per-run correction design (2026-11-23)

Goal (professor): make the tip land on the marker-square centre automatically,
or calibrate the offset out on every run. All findings below are offline,
from the labelled bags in `~/robot-logs/insertion_bags/`.

## What the bags say

Tools: `scripts/insertion_accuracy_audit.py` (FK tip vs target),
`scripts/camera_plane_check.py` (marker positions, size-independent).

1. **The arm is not the problem.** In all 12 labelled bags, FK puts the tip
   within 0.1 mm of the commanded target. Every measured offset is target
   error (cameras) or model error (TCP).
2. **The 2026-11-10 runs had no cameras.** No camera topics are in the bags;
   `target_source=auto` silently used the nominal params. Their 25 mm spread is
   container placement. Fixed in `18c5f54`: auto now warns and logs
   `auto:params_fallback`.
3. **The front camera sees only the far half of the board.** At both
   2026-10-13 positions the board centre projects below the image (v ≈ 521 of
   480). Only markers 0 and 3 are visible, so the centre is extrapolated from
   two markers plus the remembered board template, and the tool plate occludes
   marker 0 during Phase 4b. Two-point extrapolation makes orientation error
   grow with distance, which fits a position-dependent offset.
4. **The marker-size scale is about 6% off, and the extrinsic has absorbed it.** Ray /
   board-plane intersection gives ranges 1.05–1.07× the PnP ranges, and PnP puts
   markers 40–50 mm above the board. That fits 53 mm markers being treated as
   50 mm; the bird's-eye calibration grid is 53.3 mm. Position A still lands
   on centre because the front extrinsic was fitted with the same wrong size.
   **Do not change `marker_size` alone.** It moves targets by about 30 mm until
   the front extrinsic is re-fitted with the same size.

## Per-run correction: design

The "tip and target in the same camera image" check needs a camera that sees
the board centre and the tool at hover. As mounted, the front RealSense does
not (finding 3). Options, best first:

**A. Wrist-camera look pose (recommended; no new hardware).** Since FK is
exact (finding 1), the wrist camera pose at a look pose is known to within the
hand-eye error. From about 0.3 m it sees all four markers at about 0.5 mm/px, so no
extrapolation and range error is small. `insertion.py` already has a
`marker_look` phase and `target_source:=markers`; this option makes it the
default target source and records the wrist-only centre next to the fused one.
Per-run correction = fresh wrist-camera centre. Remaining one-time
calibrations: wrist hand-eye (`validate_wrist_handeye.py`) and TCP.

**B. Re-aim the front camera** so the whole board and container are in view,
then re-fit its extrinsic with the true marker size. This fixes findings 3 and 4
together and makes a front-camera same-image tip check possible later.

**C. Tool fiducial (same-image check)**, only after B or with a camera that
sees the hover point: a small ArUco on the tool, tag→tip offset found once by
pivoting, correction = marker centre − tag-predicted tip, both at known height
(ray / plane, size-independent).

## To do at the arm (2026-11-24)

1. Caliper the table markers; note the value, **do not** change `marker_size` yet.
2. Cameras up, including the wrist camera; confirm `/fused_marker_square_center`.
3. Grab one bird's-eye frame and one front frame with the tool at hover, to
   confirm coverage (offline projection says the bird's-eye camera misses it).
4. Labelled runs with `target_source:=markers`, 3 repeats × 3 or more positions across y.
