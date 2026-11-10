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

---

# M2 update — 2026-11-09 (branch `feature/curobo-m2`, not on main)

## Guarded approach-only execution (code done, NOT hardware-tested)

`-p curobo_execute_approach:=true` lets the cuRobo backend execute the validated
Phase 0 approach (ready pose, then upright approach) and nothing else. It refuses
to start unless `execute_motion`, `step_by_step` and `real_robot` are all true.
`curobo_approach_execution.validate_for_execution` re-checks the raw waypoints
against insertion.py's own limits before AND after the operator answers the
`[Step]` prompt. The live start and the scene signature are re-checked after the
prompt too. There is no recovery or fallback motion: a failure leaves the arm
where it stopped. Ctrl+C cancels the in-flight FJT goal. Artifacts, including
`execution_status.json`, go to `curobo_result_dir`.

Not verified: whether the joint-torque force guard is armed on this path. Its
setup may live only in the Pilz flow. Do not rely on it for the first run.

## Pilz vs cuRobo (offline, `compare_pilz_curobo.py`)

Same three 2026-10-09 starts, same FK model. **The goals are NOT the same.** The
Pilz runs used `azimuth_mode:=search`; the previews used `tangential` (search is
rejected by the preview). Both end 0.210 m from the container centre, but at
azimuth -150 vs -108 deg (163412) and 165 vs -66 deg (165447). z differs by 5 mm (0.236 vs
0.241). So these numbers compare the planners' character, not the same task:

| start | Pilz T / tip path | cuRobo T / tip path | peak joint vel P / C |
|---|---|---|---|
| 163412 (Kinova Home, 89.6 deg tilt) | 16.8 s / 0.895 m | 27.6 s / 0.977 m | 0.434 / 0.183 |
| 163937 (ready) | 3.9 s / 0.312 m | 8.5 s / 0.326 m | 0.349 / 0.181 |
| 165447 (ready) | 4.8 s / 0.346 m | 2.0 s / 0.111 m | 0.278 / 0.162 |

The cuRobo upright stage stays at <= 1.26 deg everywhere. cuRobo runs pinned at
its 0.15 velocity cap (Pilz peaked 0.28–0.43 rad/s), which is why it is slower.
**Chaining consequence:** a cuRobo approach ends at the tangential hover. Any
later Pilz pivot must use the same azimuth, so never mix `search` with cuRobo.

## First hardware run checklist (container ABSENT)

1. `/sync-real1` the branch files: insertion.py, curobo_planning_preview.py,
   curobo_approach_execution.py, insertion_geometry.py, curobo_planner_*.py,
   CMakeLists.txt. REAL-1's insertion.py and server predate main's refactor.
   Rebuild surgical_arm_bringup on REAL-1, because the new module goes through
   install(PROGRAMS).
2. Accept or resolve the live tool collision-mesh vs CAD-sphere mismatch above.
3. Laptop sidecar + reverse tunnel exactly as README "Planning-only integration".
   Load the scene with setup_planning_scene. The preview requires a
   `glass_container` *scene object* even with the physical container removed.
4. Run a preview first (`execute_motion:=false`) and check it in RViz.
5. Then, with a hand on the e-stop:
   ```bash
   ros2 run surgical_arm_bringup insertion.py --ros-args \
     -p planner_backend:=curobo -p real_robot:=true \
     -p execute_motion:=true -p curobo_execute_approach:=true -p step_by_step:=true \
     -p skip_home_move:=false -p transit_velocity_scaling:=0.10 \
     -p azimuth_mode:=tangential -p curobo_socket:=/tmp/curobo_live_preview.sock
   ```
   Restart the sidecar whenever the speed scale changes. Pull logs the same day
   with /pull-run.
