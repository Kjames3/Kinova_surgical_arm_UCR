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

The joint-torque force guard is NOT armed on this path, and the same is true of
Pilz Phase 0: `_guarded()` only wraps Phase 4a/4b. That is deliberate. The guard
trips on torque *change* from the start of the motion (6 Nm on j1-4), and a large
free-space transit changes gravity torque by more than that, so it would
false-trip. The e-stop and Ctrl+C (which cancels the FJT goal) are the
protection for this run.

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
   While bringup is up, take the fair-comparison dry runs (no motion, about 2 min,
   all from the same pose). The recipe is in the compare_pilz_curobo.py docstring:
   Pilz pose-goal with `plan_dump_dir:=~/curobo_planning_runs/pilz`, the cuRobo
   preview, then Pilz with `pilz_joint_goal_file:=<that cuRobo run>/response.json`.
   Use tangential azimuth throughout. container_base_offset now defaults to the
   5 mm board, which closes the old 5 mm height gap.
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

## Motionless REAL-1 test — 2026-11-09 21:30 (bringup up, arm parked, no motion sent)

The branch scripts were synced to `~/ros2_kortex_ws` with backups (tag
`pre_curobo_m2_20261109_212824`). `insertion_geometry.py` and
`curobo_approach_execution.py` were symlinked into REAL-1's install tree by
hand, because REAL-1's CMakeLists.txt has uncommitted local edits and was not
overwritten. Scene: `~/insertion_scenes/curobo_preview_20261009/scene.json`,
the virtual container at (0.3485, -0.1060). The parked start was far from
upright (j2 = 1.30 rad). Artifacts: `~/robot-logs/2026-11-09/curobo_m2_motionless/`.

| test | result |
|---|---|
| cuRobo preview, live | PASS: 31.1 s = ready transition 22.5 s + upright approach 8.5 s, approach tilt 1.25 deg |
| execution gate on that live plan | PASS: vel ratio 0.99, acc 0.77, start err 3e-7 rad |
| `execute_motion:=true` without `curobo_execute_approach` | rejected at startup |
| Pilz pose-goal dry run + plan dump | PASS; later Phase 3 dry-run planning fails -31 (planned from the parked pose; unrelated) |
| Pilz joint-goal dry run | PASS after two fixes (below) |

Fair comparison: same start, end tips 0.0 mm apart. Three cuRobo previews from this
start were identical (sd 0.000 on every metric): in this setup the sidecar is
deterministic, so repeat trials are not needed.

| | Pilz pose goal | Pilz joint goal | cuRobo |
|---|---|---|---|
| duration | 5.6 s | 17.7 s | 31.1 s |
| tip path | 0.52 m | 3.91 m | 1.81 m |
| joint path | 4.69 rad | 12.95 rad | 16.74 rad |
| peak joint vel | **0.615** | 0.183 | 0.181 rad/s |
| max tool tilt | 33.8 deg | **140.8 deg** | 70.9 deg (ready stage); 1.25 deg in approach |
| tool-container clearance (AABB) | 0.165 | 0.237 | 0.183 m |

Reading:
- **Pilz at transit scaling 0.15 runs 3.4x faster in joint space than cuRobo**
  (0.615 vs 0.18 rad/s). Its PTP respects joint scaling (the joint-goal run peaks
  at 0.183), so the excess comes from the blended LIN segment, which is limited in
  Cartesian space. "Match Pilz" therefore means joint vel scale ~0.45, not 0.15.
- cuRobo is slower and longer here mostly because `skip_home_move:=false` routes
  it through the ready pose first. A Pilz dry run does not plan its home move
  (the home move runs only on execute), so the pose-goal Pilz row omits roughly
  15 s of home move. A fully fair end-to-end comparison needs that home move
  counted, or cuRobo run with `skip_home_move:=true` from an upright start.
- A Pilz PTP to the same joints turns the tool 140.8 deg (past horizontal) on
  the way. That is the case for an upright-constrained planner.

Bugs found and fixed during this test: the joint goal was passed as a dict
(Pilz needs a list), and the joint goal kept cuRobo's winding (joint_3 = 4.095)
→ Pilz -2. It is now wrapped to (-pi, pi], so joint_3 turns 3.19 rad the other
way to the same pose. The `pilz_joint_goal_file` + execute rejection moved to
startup; before, it fired only after the home move.

## Trajectory table check (barrier item b) — 2026-11-09, offline only

Every trajectory insertion.py sends through FJT or MoveIt (Pilz, OMPL, cuRobo,
recovery) is now checked before it is published. The check is FK along the
path, densified to 0.01 rad, over the six impedance-barrier links (as spheres
of 0.055 m) plus the thesis_ee hull, against the table at z = -0.03. Params:
`table_check` (default true), `table_check_tool` (thesis_ee | none),
`table_check_margin` (0.01 m). Fail-closed: a tool with no hull (2f140) is
refused; with motion enabled, the checker is built at startup. Native Kortex
LIN is refused while the check is on: it has no joint trajectory to check.

If the arm starts inside the margin, the rule becomes "never closer than the
start, minus 2 mm", so a parked-low arm can still move away. Two archived
2026-10-10 home moves started 5.3 and 8.3 mm above the table under this model.
All 152 archived planned trajectories pass; the closest approach was 5.3 mm,
the parked start just mentioned. Link heights agree with the impedance
barrier's height_barrier_terms to 1e-6 m on 50 random poses.

Insertion tip floor (same day): for the insertion-depth phases only
(`INSERTION_DEPTH_LABELS`: Phase4*, Phase6a*), the tip vertex alone may go down
to (target height above table - `table_check_tip_allowance`, 5 mm). Every other
hull vertex and every link keeps the 10 mm margin. At startup, before the home
move, a target leaving less than `table_check_tip_min_floor` (3 mm) is refused.
At the default 30 mm depth the target is 61 mm above the table, so the floor is
56 mm: not binding today.

2F-140 hull (same day): `config/2f140_hull_bracelet.csv` has 409 vertices.
It comes from the URDF collision meshes and pad boxes, swept over finger_joint
0..0.7 rad, and reaches 297 mm below the bracelet. It is registered in both the
trajectory check and impedance.py's TOOL_HULLS (`--tool 2f140`), so torque mode
now guards the fingers too. Offline, validate_table_barrier.py section 9 shows a
20 N push driving the fingers 56 mm into the table with links-only (the old
2F-140 behaviour), and all 27 pushes held with the hull (worst +25 mm). Caveat:
the package's robotiq_2f_140_coupling.stl is not referenced by the URDF. If a
physical coupling plate is fitted, the real fingers sit lower than the hull, so
measure it. Raising --table-z-min for the 2F-140 is no longer needed, but it is
harmless. The validator now takes ~15 min.

Not hardware-run. On REAL-1, `trajectory_table_check.py` needs the same
manual install-tree symlink as the other new modules.
