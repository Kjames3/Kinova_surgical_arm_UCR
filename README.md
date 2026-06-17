# UCR Surgical Arm

A ROS 2 workspace package set for a **Kinova Gen3 7-DOF** arm fitted with a custom
`thesis_ee` end effector and a virtual `assembly_tip` TCP. The project drives the
arm to perform **precision container-insertion** tasks: a rigid tool tip is guided
straight down — or tilted at a computed angle — into a small container whose 3D
center and orientation are located in real time by **multi-camera ArUco marker
sensor fusion**. The same robot description and motion stack also run in **NVIDIA
Isaac Sim** for hardware-free development and testing.

This repository is developed at UC Riverside as part of a thesis project.

---

## What's in the repo

The workspace folder contains three ROS 2 packages:

| Package | Purpose |
|---------|---------|
| **`surgical_arm_description`** | URDF/xacro and meshes for the custom `thesis_ee` end effector (the `assembly_tip` virtual TCP link lives here). |
| **`kinova_gen3_7dof_robotiq_2f_140_moveit_config`** | MoveIt 2 configuration and the top-level launch files (`robot.launch.py`, `isaac_sim.launch.py`, `gen3_complete_system.launch.py`, camera launches, planner YAMLs). |
| **`surgical_arm_bringup`** | Application logic — motion scripts (`insert_to_container.py`), the multi-camera fusion node (`combine_cameras.py`), Isaac Sim setup scripts, hand-eye calibration tools, and supporting launch files. Also defines the `InsertContainer` action. |

### Key scripts (`surgical_arm_bringup/scripts/`)

| Script | What it does |
|--------|--------------|
| `insert_to_container.py` | Main task: approach, tilt, and insert the `assembly_tip` into the container using Pilz PTP/LIN/CIRC motions. |
| `combine_cameras.py` | Multi-camera ArUco fusion node; publishes the fused container center pose + TF. |
| `isaac_sim_gen3.py` | Headless/standalone Isaac Sim launcher with the ROS 2 OmniGraph bridge. |
| `isaac_sim_gen3_gui.py` | Loads the robot + scene into a running Isaac Sim GUI session. |
| `isaac_sim_import_urdf.py` | Converts the robot URDF → USD for Isaac Sim (run after any URDF/xacro change). |
| `setup_planning_scene.py` | Publishes table/obstacle collision objects to the MoveIt planning scene. |
| `handeye_calibration*.py`, `calibrate_camera_offsets.py` | Camera ↔ robot extrinsic calibration utilities. |
| `robot_keepalive.py`, `test_pen_tip_offset.py`, `kortex_utils.py` | Connection keepalive, TCP-offset check, shared helpers. |

---

## Build

This is a single package set inside a larger `ros2_kortex_ws`. Build from the
workspace root (RAM-constrained — sequential, 2 compiler threads):

```bash
cd ~/workspace/ros2_kortex_ws

# Full build
colcon build --executor sequential --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_PARALLEL_LEVEL=2

# Rebuild just these packages
colcon build --executor sequential --cmake-args -DCMAKE_BUILD_PARALLEL_LEVEL=2 \
  --packages-select surgical_arm_bringup surgical_arm_description \
                    kinova_gen3_7dof_robotiq_2f_140_moveit_config

# Source after every build (do this in every new terminal)
source install/setup.bash
```

`--symlink-install` means Python scripts and YAML/xacro changes take effect
without a rebuild. C++ changes always require a rebuild.

---

## Running the robot

### 1. Bring up the arm + MoveIt (`gen3` launch scripts)

**Physical arm:**

```bash
ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config robot.launch.py \
  robot_ip:=192.168.1.10 use_fake_hardware:=false
```

**Simulated hardware (no physical robot, fake controllers):**

```bash
ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config robot.launch.py \
  robot_ip:=192.168.1.10 use_fake_hardware:=true
```

Useful `robot.launch.py` arguments: `launch_rviz` (default `true`),
`use_sim_time` (default `false`), `use_internal_bus_gripper_comm`
(default `false` — the `thesis_ee` has no Kortex-bus gripper).

**Complete system in one launch** — arm + MoveIt + RViz + wrist camera + OAK-D:

```bash
ros2 launch surgical_arm_bringup gen3_complete_system.launch.py \
  robot_ip:=192.168.1.10 use_fake_hardware:=false \
  launch_wrist_camera:=true launch_oak_camera:=true
```

Then publish the collision scene (after the launch above is up, before any motion):

```bash
ros2 run surgical_arm_bringup setup_planning_scene.py
```

### 2. Multi-camera fusion (`combine_cameras`)

The cameras themselves are started by `gen3_complete_system.launch.py`. Once the
camera topics are publishing, start the fusion node:

```bash
ros2 launch surgical_arm_bringup combine_cameras.launch.py
# disable the OpenCV dashboard window:
ros2 launch surgical_arm_bringup combine_cameras.launch.py enable_visualization:=false
```

Or run the node directly:

```bash
ros2 run surgical_arm_bringup combine_cameras.py
```

It detects the four corner ArUco markers across the RealSense D435i, OAK-D, and
wrist cameras, fuses them (with geometric fallbacks for partial occlusion), and
publishes the container's center pose as `PoseStamped` plus a TF frame. If a fused
axis has the wrong sign, tune the `x_sign` / `y_sign` / `z_sign` parameters in
`combine_cameras.launch.py` (only `+1.0` / `-1.0`).

### 3. Container insertion (`insert_to_container`)

> ⚠️ Without `execute_motion:=true` the script only **plans** (dry run) and prints
> the trajectory — no motion. Always dry-run first.

```bash
# Dry run (plans only, no motion):
ros2 run surgical_arm_bringup insert_to_container.py \
  --ros-args -p real_robot:=true \
  -p target_x:=0.45 -p target_y:=-0.20 \
  -p insertion_angle_deg:=45.0

# Execute for real:
ros2 run surgical_arm_bringup insert_to_container.py \
  --ros-args -p real_robot:=true -p execute_motion:=true \
  -p target_x:=0.45 -p target_y:=-0.20 \
  -p insertion_angle_deg:=45.0
```

Common parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `target_x`, `target_y` | `0.26`, `0.0` | Container center in the world frame [m]. |
| `insertion_angle_deg` | `45.0` | Tilt from vertical for the angled descent. |
| `insertion_azimuth_deg` | `0.0` | Compass direction to tilt toward (about world Z). |
| `target_depth_mm` | `30.0` | Depth below the container top [mm]. |
| `max_tilt_deg` | `45.0` | Safety cap on tilt angle. |
| `max_velocity_scaling` | `0.15` | Trajectory speed scaling. |
| `execute_motion` | `false` | `false` = plan only; `true` = move the arm. |
| `real_robot` | `false` | Set `true` when driving the physical arm. |
| `return_to_start` | `true` | Retract and return home after insertion. |

The motion runs as a sequence of phases (approach → vertical descend → compute
tilt → CIRC reorient about the tip → angled descend → hold → reverse → return).
Container model is 90 × 90 × 86 mm; valid targets are within ±40 mm of center
and 0 < depth ≤ 80 mm. An `InsertContainer` action interface is also defined for
driving the task from an action server (`use_action_server:=true`).

---

## Running in Isaac Sim

Isaac Sim provides a physics simulation of the arm that talks to the same MoveIt
stack over the ROS 2 bridge (`/isaac_joint_states` ↔ `/isaac_joint_commands`).

**Step 1 — (re)generate the USD** whenever the URDF/xacro changes:

```bash
cd ~/workspace/ros2_kortex_ws
source install/setup.bash

# Render the URDF from xacro with sim_isaac:=true
xacro src/ros2_kortex/kortex_description/robots/gen3.xacro \
    arm:=gen3 dof:=7 gripper:=thesis_ee sim_isaac:=true \
    robot_ip:=xxx use_fake_hardware:=true > /tmp/gen3_isaac.urdf

# Import URDF → USD (headless)
~/isaacsim/python.sh \
  src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_import_urdf.py
```

**Step 2 — launch Isaac Sim** (it loads the robot, scene, and ROS 2 OmniGraph):

```bash
# Standalone window
~/isaacsim/python.sh \
  src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_gen3.py

# OR inside the full Isaac Sim GUI launcher (then press Play to start the bridge)
~/isaacsim/isaac-sim.sh --exec \
  ~/workspace/ros2_kortex_ws/src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_gen3_gui.py
```

**Step 3 — connect MoveIt** to the running simulation:

```bash
ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config isaac_sim.launch.py
```

> If Isaac prints `Pattern '/gen3' did not match any rigid bodies` and
> `/isaac_joint_states` never publishes, the USD is stale/empty — re-run
> `isaac_sim_import_urdf.py` (Step 1).

---

## TF frames

```
world → base_link → joint_1…joint_7 → bracelet_link → assembly_tip
```

`world` is the fixed planning frame. Pilz PTP/LIN/CIRC goals target the IK tip
(`bracelet_link`); the tool-relative motions are computed against the
`assembly_tip` virtual TCP.
