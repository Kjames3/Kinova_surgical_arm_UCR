# Surgical-arm offline model

This directory rebuilds and validates the Gen3 7-DOF + 1.567 kg `thesis_ee`
model without a ROS graph or robot connection. It is the model prerequisite
for trajectory planning; the sidecar's real motion-planning backend is still
unimplemented. The model does not include a table/container world yet.

From the workspace root on the laptop:

```bash
source ~/activate_curobo.sh
python src/Kinova_surgical_arm_UCR/surgical_arm_bringup/config/curobo/reproduce_offline.py
```

Requires the isolated cuRobo environment, NVIDIA CUDA GPU, ROS Humble's system
Python/xacro at `/opt/ros/humble`, and the source `src/ros2_kortex` checkout.
The runner uses ROS Python only to expand xacro in a subprocess. No colcon
build or installed workspace overlay is needed. It registers a temporary
ament prefix pointing at the **source** description to avoid stale install
files. Source xacros and live launch configurations are never changed.

Each invocation creates a new ignored `offline_runs/<UTC timestamp>/` folder:

- Generated URDFs, configuration and collision spheres.
- `fk_fixtures.json`: 34 deterministic joint configurations and independent
  base-to-tip/base-to-bracelet transforms. Arbitrary FK configurations are
  mathematical checks, not executable or collision-qualified poses.
- `ik_fixtures.json`: four fixed-seed Cartesian goals, reference joints,
  returned joint solutions, success flags, independently checked pose errors.
  Quaternions are explicitly **wxyz**, positions are metres in `base_link`.
- Per-step logs, sphere surface-coverage report and HOME collision report.
- `manifest.json`: validation exit codes, file SHA-256 hashes, cuRobo commit,
  version, dirty state, Python, torch, CUDA and GPU; `pip-freeze.txt` records
  installed dependencies. Retain the referenced source revisions/files.

The runner returns nonzero if any check fails, and stops immediately if model
generation fails so stale generated files cannot produce a passing result.
Use `--output /absolute/new/directory` to choose an artifact location; existing
directories are refused. Generated resolved URDF/YAML paths are machine-local;
rerun generation on another machine rather than copying absolute paths.
Random seeds make the inputs reproducible; floating-point GPU solver results
need not be bitwise identical across different GPUs/library versions.

## Geometry and limits of qualification

- TCP is the 2026-10-09 measured `(0.108, -0.008, -0.411)` m in
  `bracelet_link`. The xacro expansion explicitly checks this calibration.
  A subsequent calibration change requires review of this check/model.
- The source tool collision transform disagrees with its visual CAD placement.
  **Only the generated offline URDF** substitutes both visual geometries
  (assembly and elbow bracket) for the tool collisions. This provisional choice
  does not establish that CAD matches the physical tool.
- CAD's nearest vertex is about 20.67 mm from the measured TCP. An explicit
  8 mm-radius sphere chain extends that vertex to the measured endpoint. A gap
  exceeding 30 mm stops generation. This is a conservative offline assumption,
  not a new measurement of the physical tool's shape.
- Deterministic grid groups bound every subdivided mesh triangle with a sphere
  that encloses all three vertices. Outward rounding preserves the bound.
  The exported YAML is independently checked against original mesh vertices
  and fixed-seed barycentric surface samples. This covers **mesh surfaces**;
  it is not a certification of solid interiors or continuous swept motion.
- There are 1,222 spheres across nine links, including 717 on `thesis_ee`
  and 105 on `spherical_wrist_2_link`. The latter uses 20 mm grid spacing
  to resolve the wrist/tool clearance without relaxing 5 mm radius padding.
  This prioritizes coverage over speed. Benchmark and simplify with a retained
  coverage check before claiming planning latency improvements. Conservative
  over-coverage can reject a physically feasible tight insertion.
- Existing adjacent-link self-collision exclusions are retained, with the
  rigid tool adjacent to its wrist carrier; no additional nonadjacent pairs
  are suppressed to make validation pass. Nominal sphere pairs are checked at
  HOME. This is not an all-postures collision qualification.
- Sphere radius padding remains 5 mm (also affects self-collision in this
  cuRobo revision); the additional per-link self-collision buffer is zero.
  Acceleration/jerk settings are inherited provisional planner settings,
  not newly validated hardware limits.

Hardware use still requires physical tool-geometry confirmation, a table and
container world, trajectory validation and the REAL-1 execution checks.

## Reconstruct the virtual container at a recorded marker target

Use the ROS system interpreter to export a scene from a completed SQLite bag:

```bash
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 /usr/bin/python3 \
  src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/export_bag_scene.py \
  /absolute/path/to/bag --output /absolute/path/to/scene/scene.json \
  --container-base-offset <support_height_above_table_in_metres>
```

The support height is zero only when the container sits directly on the table;
when it sits on a marker board, supply the measured board thickness. This input
is required. `scene.json` and the copied `Glass_container.STL` travel together.
The exporter checks the recorded `world`→`base_link` transform, takes the centre
from the recorded insertion target, and derives yaw from the nearest recorded
`/fused_corners` frame. The rounded-square container's sides are assumed aligned
with the board edges; physical alignment must be checked when placed.

The CAD is 90×86×90 mm with **Y up and a corner origin**. The shared
`container_geometry.py` rotates it to Z up, centres XY, and moves the bottom to
Z=0. Original triangles preserve the open cavity. The table dimensions remain
the existing 2×2×0.05 m defaults, and its surface height comes from the recorded
insertion parameter. These are reconstructed scene assumptions, not perception
measurements of obstacles. No unmeasured pole is added by this scene file.

Then, in the isolated cuRobo environment:

```bash
source ~/activate_curobo.sh
python src/Kinova_surgical_arm_UCR/surgical_arm_bringup/config/curobo/validate_offline_scene.py \
  /absolute/path/to/scene/scene.json --output /absolute/path/to/scene/validated
```

Outputs include a normalized mesh, `curobo_world.yml`, a geometry validation
report and a PNG preview. Validation checks the cavity, solid bottom/wall,
dimensions and cuRobo's transformed mesh placement. It does **not** qualify a
robot trajectory against these objects.

For RViz/MoveIt on REAL-1, with the robot stationary and the intended MoveIt
instance running, launch the updated scene script with
`--ros-args -p scene_file:=/absolute/path/to/scene/scene.json`. It uses
`ApplyPlanningScene` followed by `GetPlanningScene`; look for
`Verified MoveIt scene objects: ['glass_container', 'table']` (additional
pre-existing objects may also be listed). It preserves other world objects
and robot attachments. It does not command robot motion. The next insertion
bag's full scene snapshot should contain the two objects.

## Offline approach benchmark

These tools read archived data and run locally; they do not connect to ROS or
command the arm. Extract with ROS system Python, then plan with conda Python:

```bash
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 /usr/bin/python3 \
  src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/extract_approach_fixture.py \
  /absolute/path/to/bag --output /absolute/path/to/benchmark/fixture.json

source ~/activate_curobo.sh
python src/Kinova_surgical_arm_UCR/surgical_arm_bringup/config/curobo/benchmark_approach.py \
  /absolute/path/to/benchmark/fixture.json \
  --world /absolute/path/to/scene/validated/curobo_world.yml \
  --output /absolute/path/to/benchmark/results
```

Extraction requires one SQLite database, exactly one selected phase (default
`Phase0-1_blended[0]`), one published trajectory during that phase, a preceding
successful planning call, fresh joint samples and a recorded URDF. The URDF is
saved next to the fixture. Keep raw robot descriptions private: they can contain
hardware connection parameters. Serve only the generated HTML, not the raw
fixture directory.

The benchmark uses the first *planned* waypoint, verifies it against the measured
start (maximum 0.03 rad difference), and derives the goal pose from independent
FK of the final recorded waypoint. Continuous joint paths are unwrapped as a
whole. Recorded/model FK must agree throughout the reference trajectory. The
recorded goal is **not** shifted by the later 5 mm board correction.

This is a free-space endpoint comparison, not a reproduction of Pilz's blended
waypoint constraints. The reconstructed scene differs from the empty scene in
the historical recording. `preceding_planning_call.wall_s` measures the original
service-call interval, not just optimizer compute. Do not report a like-for-like
speedup. Recorded transit velocity scaling is applied to the loaded joint limits
exactly once (the local cuRobo revision applies YAML velocity scaling twice).
Acceleration uses half that scale on the provisional cuRobo acceleration limits;
these are not claimed equivalent to MoveIt's or validated hardware limits.

`report.json` records environment, fingerprints, initialization, the first
planning call, warm calls, duration, joint travel, independent endpoint errors,
velocity/acceleration ratios, and sampled position-limit/world/self-collision
checks. Validation samples the piecewise-linear interpolated path at at most
10 ms and 0.01 rad per joint. It is not continuous swept-volume certification.
All existing sphere padding and pair exclusions are preserved. A rejected solver
result stays rejected even if a separate check passes. Exit status 1 means the
benchmark did not produce passing results; rejected candidates are saved only
for inspection. `--goal-mode recorded-joints` is a separately labelled diagnostic
that fixes the final configuration instead of asking IK for a pose solution.
`--self-collision-weight` adjusts optimization cost only (default upstream 10000),
never the acceptance checks or geometry.

Render an interactive standalone HTML comparison with **system Python** (numpy
and plotly), outside the isolated conda environment:

```bash
/usr/bin/python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/view_approach_benchmark.py \
  /absolute/path/to/benchmark/results --scene /absolute/path/to/scene/scene.json \
  --output /absolute/path/to/public/approach.html
```

Play/scrub compares both paths at the same elapsed time using reference arm
centrelines. Rejected candidates are red and labelled as rejected. The HTML does
not embed the raw URDF. Numerical regression checks:

```bash
source ~/activate_curobo.sh
python src/Kinova_surgical_arm_UCR/surgical_arm_bringup/config/curobo/test_benchmark_approach.py
```

The relocated-board approach now passes all four pose-goal benchmark calls after
refining `spherical_wrist_2_link` from 35 mm to 20 mm grid spacing (24 → 105
spheres). Other link spheres, the 5 mm radius padding, and pair exclusions are
unchanged. The earlier coarse fit produced a 0.14 mm padded wrist/tool overlap;
the refined fit has at least 2.55 mm padded clearance at the recorded waypoints.
At the former worst waypoint, actual mesh surface separation is bounded at
43.1–45.1 mm by the mesh diagnostic. The full recorded path passes 505 sampled checks; each cuRobo path passes 541.
Full model qualification also passes. These are offline results for this case,
not a hardware-execution qualification or a continuous collision certificate.

For an actual mesh surface-distance comparison at the former worst waypoint,
after qualifying the model, run:

```bash
source ~/activate_curobo.sh
python src/Kinova_surgical_arm_UCR/surgical_arm_bringup/config/curobo/measure_wrist_tool_clearance.py \
  /absolute/path/to/benchmark/results/recorded_trajectory.json --waypoint 35 \
  --output /absolute/path/to/benchmark/results/mesh_clearance.json
```

This computes a bounded surface distance using subdivided wrist vertices and
point-to-triangle queries, with conservative pruning from the qualified tool
sphere cover. It checks one configuration and does not test solid containment.
`diagnose_trajectory_spheres.py TRAJECTORY --output REPORT` separately reports
nominal and padded separations for non-excluded sphere pairs at saved waypoints.

### Timing for larger starting-posture changes

The benchmark now sets `--max-trajectory-dt 0.3` seconds before solver creation,
up from the installed cuRobo default of 0.2. This is the solver's internal timing
ceiling, **not** the output trajectory sample interval or a requested duration.
It allows automatic retiming to satisfy the existing joint limits for large
moves. Output interpolation stays at 25 ms; independent validation stays at
<=10 ms and <=0.01 rad per joint. No velocity, acceleration, jerk, collision,
start-continuity, endpoint or solver-success thresholds are relaxed.

The original no-home start hit the old ceiling at 16.5 s and exceeded velocity
limits. With the new ceiling it produces passing 18.5–19.0 s pose-goal paths.
`--max-trajectory-dt 0.2` retains the old timing ceiling for comparison.

The interpolation buffer is sized from the ceiling and B-spline knot layout
before the first solve (1,001 samples for this model/settings, rather than the
upstream 5,000). It includes rounding/endpoint headroom and covers the maximum
requested duration; it does not truncate the result or coarsen interpolation.
This avoids GPU out-of-memory errors from large sphere-state buffers on the
6 GB laptop. Regression tests compare the bound with cuRobo's native trajectory
sample-count calculation, including float32 rounding near the ceiling.

Reports record the timing ceiling, interpolation sample interval, knot count and
buffer capacity. The fix is currently in the offline benchmark; it is not yet
wired into insertion.py or the live sidecar.

## Planning-only integration with the running arm

`insertion.py` now accepts `planner_backend:=pilz|curobo` (case-insensitive).
Pilz is still the default. **cuRobo currently supports only a free-space approach
preview. It rejects `execute_motion:=true`; it never falls back to Pilz or sends
an action goal.** Action-server mode, teaching/replay, azimuth search,
direct tilted approach, and current-orientation mode are rejected in this mode.

The ROS process reads MoveIt's expanded URDF and complete scene plus a fresh,
complete seven-joint sample. The goal uses the **container mesh in that scene**,
not `target_x/y` or a fresh target-topic replacement. Its top plus
`approach_clearance` defines the hover height; depth/tilt/azimuth use the same
insertion-axis geometry as the normal approach. The scene must already contain
`table` and `glass_container`; moving the marker board requires updating the
scene first. Board thickness is included in the scene's container base height.
Only boxes and triangle meshes in world/base_link are supported. Other shapes,
attached objects, octomaps, or nondefault MoveIt link scaling/padding fail closed.

The qualified tool spheres deliberately use the **provisional visual CAD
placements**, just as `regenerate_urdf.py` does. Live tool visuals and all relevant
kinematic chains are checked against that model; other collision-link shapes
must match. This does not repair the older live tool collision placement or
qualify the CAD-to-hardware fit. Existing cuRobo self-collision exclusions remain;
MoveIt's allowed-collision matrix is not imported to loosen them.

The sidecar uses the benchmark's 0.3 s internal timing ceiling, sized interpolation
buffer, 25 ms output sampling, and <=10 ms / <=0.01 rad dense validation. Velocity
scaling is applied once and currently capped at 0.15; acceleration scaling is
half the velocity scale. Endpoint, start, derivative, collision and bound checks
must pass. Continuous-joint winding is restored to match the measured state.
Before publishing, the ROS client checks that the arm and scene have not changed.
Results go to `~/curobo_planning_runs/<UTC timestamp>/`, and the accepted trajectory
is published on `/display_planned_path` and `/insertion/planned_trajectory`.
The publisher remains alive for five seconds; RViz should subscribe beforehand.
These files contain the full URDF: **do not serve the run directory on HTTP.**

The current test uses the laptop's qualified CUDA environment and an SSH Unix
socket tunnel. No cuRobo installation on REAL-1 is needed for this arrangement.
Run the following only if those processes are not already running:

```bash
# Laptop terminal A, from the workspace root:
source ~/activate_curobo.sh
python src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/curobo_planner_server.py \
  --backend curobo --socket /tmp/curobo_live_preview.sock

# Laptop terminal B:
ssh -N -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=no \
  -R /tmp/curobo_live_preview.sock:/tmp/curobo_live_preview.sock kinova@10.12.140.145

# REAL-1, with bringup and the current table/container scene loaded:
source ~/ros2_kortex_ws/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp PYTHONNOUSERSITE=1
ros2 run surgical_arm_bringup insertion.py --ros-args \
  -p planner_backend:=curobo -p execute_motion:=false -p skip_home_move:=false \
  -p transit_velocity_scaling:=0.15 -p azimuth_mode:=tangential \
  -p curobo_socket:=/tmp/curobo_live_preview.sock
```

Domain 0 matches the bringup checked on 2026-10-09; verify it for a new session.
Restart the sidecar after changing world geometry or speed scale: this version
refuses to reuse its CUDA graphs with a different world. A socket already in use
is never unlinked automatically. If SSH leaves a stale remote socket, confirm
its tunnel is gone before removing that specific socket and reconnecting.
The ROS helper `curobo_planning_preview.py` is installed beside insertion.py by
CMake; REAL-1's current symlink installation has also been updated directly.

```bash
source ~/activate_curobo.sh
python src/Kinova_surgical_arm_UCR/surgical_arm_bringup/config/curobo/test_preview_backend.py
```

Ten preview regression tests cover preview-only options, velocity caps, composed scene
poses, rejected geometry, model differences, scene changes and repeated URDF
reads after a C-locale change. The socket wiring test remains available in
`scripts/test_curobo_wiring.py`. Physical trajectory execution is a separate,
unimplemented gate; do not infer hardware qualification from a preview PASS.


### Upright approach and tool-down ready stage

`curobo_max_tilt_deg` defaults to **15.0**, accepts values in (0,15], and is a
cone angle around world-down, not separate Euler roll/pitch limits. The shaft
axis is calibrated by the existing `vertical_quat_*` bracelet orientation.
The lateral bracelet-to-TCP translation is not the shaft direction. This
reference still needs confirmation against the physical tool.

- `skip_home_move:=false`: preview a separate joint-space transition to the
  script's configured `home_joints` (SRDF home/ready if available, otherwise the
  surgical tool-down fallback). Require that pose to be within 2 degrees of the
  vertical reference and collision-free. If already within 0.001 rad per joint,
  omit this redundant stage. A start outside the upright cone can reorient during
  this explicitly labelled `ready_transition`; it is not reported as an upright
  approach. If the start is already upright, the ready transition must also pass
  the upright check. Add a 0.1 s stationary hold between the two stages.
- `skip_home_move:=true`: begin the approach at the measured joints, and reject
  a start outside the cone. The current Kinova Home pose measures about 89.69
  degrees from the calibrated down direction, so it requires the ready stage.
- `upright_approach`: cuRobo's nonterminal pose cost guides all three orientation
  axes toward the vertical goal (a conservative yaw preference too). The
  independent URDF-based validator checks shaft tilt on the same <=10 ms /
  <=0.01 rad sampled path used for collision validation. A trajectory exceeding
  the requested cone angle is rejected; a soft cost alone never counts as PASS.

Both stages must pass start/endpoint, collision/bounds, velocity and acceleration
checks and have stationary endpoints. The response includes per-stage time,
maximum tilt, and point-by-point tilt. The ROS client requires the upright policy
version acknowledgement, so an old sidecar cannot silently return an endpoint-only
plan. The deliberate insertion tilt/pivot is not part of this preview.

```bash
source ~/activate_curobo.sh
python src/Kinova_surgical_arm_UCR/surgical_arm_bringup/config/curobo/test_upright_preview.py
```

Six additional orientation tests cover yaw invariance, compound roll/pitch,
calibration, excessive intermediate tilt, nonfinite inputs and start-policy
validation. The three archived starts/board positions and the REAL-1 preview
passed on 2026-10-09 with maximum approach tilt <1.26 degrees. Results and a
stage-labelled viewer are in `~/robot-logs/2026-10-09/upright_preview/` and
`http://localhost:8765/curobo_upright_preview.html`. These remain sampled planning
checks, not hardware execution qualification.
