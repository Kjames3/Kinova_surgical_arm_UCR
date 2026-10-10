# Session handoff — 2026-10-09

## Current state

cuRobo supports planning-only free-space approach previews through
`insertion.py -p planner_backend:=curobo`. Pilz remains the default.
Physical execution is rejected for cuRobo, including recovery/fallback motion.

`skip_home_move:=false` previews the configured tool-down ready joint pose before
an upright approach. `skip_home_move:=true` rejects starts outside the upright
cone. `curobo_max_tilt_deg` defaults to 15 degrees. Orientation is checked along
the densely sampled path, not just at the endpoint. Initial reorientation from
the measured Kinova Home (~89.69 degrees from calibrated down) is explicitly
separate from the upright approach.

## Validation completed

- Full offline model qualification: seven stages PASS.
- Earlier endpoint-only trajectory suite: 12/12 PASS after the timing-buffer fix.
- Upright previews: all three archived starts/board positions PASS; maximum
  approach tilt below 1.26 degrees.
- Two live-state, planning-only previews on REAL-1 PASS, with no motion sent.
  Ready transition 19.5 s, hold 0.1 s, upright approach 8.5 s; maximum approach
  tilt 1.252 degrees. Warm solver time 3.65 s for both stages combined.
- Negative live test: skipping ready from the tilted start is correctly rejected.
- Six upright, ten preview, and seven benchmark regression tests PASS.

Detailed private artifacts remain on the laptop:
`~/robot-logs/2026-10-09/{timing_fix_suite,live_curobo_preview,upright_preview}/`.
Do not publish raw requests/URDFs from those directories. The derived browser
preview is `http://localhost:8765/curobo_upright_preview.html` while its local HTTP
server runs. It includes separate stages, a shaft-axis guide and a tilt graph.

## Resume

See README.md for sidecar, SSH tunnel, ROS domain and command setup. The current
arrangement runs cuRobo on the laptop and the ROS client on REAL-1 through
`/tmp/curobo_live_preview.sock`. Existing processes may not survive overnight;
check before starting duplicates. No startup persistence is configured here.
Restart the sidecar when scene geometry or speed scaling changes.

REAL-1 uses the built workspace `/home/kinova/ros2_kortex_ws`. The ROS helpers are
already synced there with backups and checksum checks. Do not git-pull over its
untracked working files. The repository on the laptop is the git authority.

The current MoveIt scene has the virtual container at the observed board centre
(0.348545, -0.105976) m, yaw 4.451 degrees, bottom -0.025 m. It sits on a 5 mm
board above the -0.030 m table. A bringup restart loses the scene; board movement
requires a new scene capture, not blindly restoring these coordinates.

## Remaining before physical motion

1. Verify the physical shaft/down reference, measured TCP and provisional CAD
   placement. The live tool collision mesh still differs from the visual CAD
   placement used for the qualified cuRobo spheres.
2. Add a separate, explicitly guarded approach-only execution path with fresh
   state/scene checks and controller-compatible trajectory validation.
3. Perform the first supervised slow free-space approach with the container
   absent. Pivot and insertion are not qualified by these previews.

## External model dependency

The source tool xacro and CAD are in the separate workspace tree
`src/ros2_kortex/kortex_description/grippers/thesis_ee/`. That directory currently
appears untracked in that separate repository; it is not published by this
surgical-arm commit. Preserve it and arrange its versioning before expecting a
fresh clone on another machine to reproduce the model. Generated cuRobo URDF,
YAML, spheres and provenance reports are included here, but resolved mesh paths
are machine-local. Regeneration explicitly depends on that external source.
The surgical TCP is `assembly_tip` at (0.108, -0.008, -0.411) from bracelet_link;
historical pen/gripper parameters do not apply.
