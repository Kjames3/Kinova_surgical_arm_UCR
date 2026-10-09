# Collecting insertion bags on the table

Goal: 40-50 labelled insertion runs at varied marker-square positions, to train
a small model that corrects the tip's error at the centre of the four ArUco
markers. Each run records its own rosbag automatically; your job is to move the
marker square, start the run, and type in one measurement.

Everything runs on **REAL-1** (`ssh kinova@10.12.140.145`, or sit at the desk).
Six terminals. The commands below are the ones used on 2026-10-08.

## Before you start

1. **`thesis_ee` must be on the flange**, not the 2F-140 gripper. The end
   effector is swapped between users; look before you launch.
2. **The arm is shared.** Run the preflight and do not continue until it is clear:

   ```bash
   cd ~/ros2_kortex_ws/src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts
   python3 preflight_coexistence.py
   ```

   If it prints `NOT CLEAR`, ask the owner (Shashwat) to stop their stack.
   Never kill another user's processes and never run `ros2 daemon stop`.
3. Keep a hand near the e-stop. Clear the table of everything but the marker
   square and container.

## Environment (paste into every terminal first)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_kortex_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset ROS_DOMAIN_ID
```

Our stack runs on ROS domain 0 with CycloneDDS. The other stack on this machine
uses domain 42 — if `ROS_DOMAIN_ID` is set you will see their topics, not ours.

## Terminal 1 — arm + MoveIt

```bash
ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config robot.launch.py \
  robot_ip:=192.168.1.10 launch_rviz:=false
```

Wait for the controllers to report active.

## Terminal 2 — wrist camera

```bash
ros2 launch kinova_vision kinova_vision.launch.py \
  device:=192.168.1.10 camera:=camera launch_color:=true launch_depth:=false
```

## Terminal 3 — front RealSense

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=realsense camera_name:=front_cam serial_no:=_938422070760 \
  enable_color:=true enable_depth:=false pointcloud.enable:=false \
  publish_tf:=false rgb_camera.color_profile:=640x480x15
```

If it says the device is unreachable, another process already holds the camera.

## Terminal 4 — front camera extrinsic (2026-10-08 calibration)

```bash
ros2 run tf2_ros static_transform_publisher \
  --x 1.052352 --y -0.038352 --z 0.774732 \
  --qx 0.633048 --qy 0.617599 --qz -0.320095 --qw -0.339650 \
  --frame-id base_link --child-frame-id front_cam_color_optical_frame
```

## Terminal 5 — marker fusion

```bash
cd ~/ros2_kortex_ws/src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts
python3 combine_cameras.py --ros-args \
  -p enable_visualization:=false -p fusion_mode:=rays -p z_sign:=1.0 \
  -p table_z:=-0.03 -p marker_height_above_table:=0.005 \
  -p realsense_image_topic:=/realsense/front_cam/color/image_raw \
  -p realsense_info_topic:=/realsense/front_cam/color/camera_info
```

**If this was already running from before 2026-10-09 15:10, restart it** — the
older process does not publish `/marker_observations` or the 5 Hz camera JPEGs
the bag records.

Check that the markers are seen (should print a pose at ~10 Hz):

```bash
ros2 topic echo --once /fused_marker_square_center
```

**Do not subscribe to the raw camera topics while the arm driver is running**
— no `ros2 topic echo/hz` on `.../image_raw`, no rqt image view, no RViz image
display. A second subscriber makes DDS broadcast the raw video onto the arm's
Ethernet link; the arm driver then times out and the arm silently stops
accepting motion (terminal 1 shows `timeout detected: BaseCyclicClient::Refresh`).
If that happens, restart terminal 1.

## Terminal 6 — the insertion run (repeat per run)

```bash
ros2 run surgical_arm_bringup insertion.py --ros-args \
  -p real_robot:=true -p execute_motion:=true -p step_by_step:=true \
  -p target_source:=markers -p azimuth_mode:=search \
  -p target_x:=0.47 -p target_y:=0.0 -p max_velocity_scaling:=0.15
```

- `target_x` / `target_y` are only the rough spot the wrist camera looks at
  first; the real target is measured from the markers. Change them if the
  square is far from (0.47, 0.0).
- `step_by_step:=true` asks `[Step] Execute <phase>? [Y/n]` before every move.
  Press Enter to continue. Answering `n` stops all further motion in that run.
- Add `-p execute_motion:=false` for a dry run (plans only, no bag).

### At the hold prompt — the measurement

When the tip is at depth the script stops and asks:

```
Tip at target (...).  Type the measured tip offset from the centre as two numbers in mm,
  X then Y in the world frame (example: 4 -6), then ENTER.
  Or just ENTER to reverse without a measurement:
```

Measure with a ruler where the tip is relative to the centre of the marker
square and type the two numbers, e.g. `1.5 -2`, then Enter. Type numbers, not the
letters "dx dy"; anything that is not two numbers is asked for again.

- `dx`: positive = tip is farther from the robot base than the centre
  (toward the front camera).
- `dy`: positive = tip is to the robot's left, seen from behind the robot.
- If you cannot measure, just press Enter. The run is still recorded but has no
  label, so it cannot be used for the accuracy model.

Do not touch the arm or the container while measuring.

## What to vary

| | |
|---|---|
| Positions | 12-15 across the reachable table area, roughly a 4×4 grid plus a few near the edges |
| Repeats | 3 runs per position, **without moving the square between them** |
| Rotation | rotate the square (20-45°) at about a third of the positions |

After moving the square, check terminal 5's echo again before starting. Note
anything unusual (bumped table, marker partly covered, run aborted) in the log.

## Run log

Copy this and fill a line per run — the bag name is printed as
`Recording bag: ~/insertion_bags/insertion_<date>_<time>`.

| # | Bag (time) | Position (rough) | Rotated? | dx mm | dy mm | Notes | Who |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |

## After the first run — check the bag

```bash
ros2 bag info ~/insertion_bags/$(ls -t ~/insertion_bags | head -1)
```

These must all have a non-zero count:

- `/insertion/throttled/camera/color/image_raw/compressed` (wrist camera, 5 Hz)
- `/insertion/throttled/realsense/front_cam/color/image_raw/compressed`
- `/marker_observations`
- `/insertion/run_info` (3 or 4 messages)
- `/insertion/planned_trajectory`
- `/insertion/phase`

If an image topic is 0, that camera is not publishing or terminal 5 is the old code. If
`/marker_observations` is 0, restart terminal 5.

## When you are done

1. Ctrl-C terminals 6 → 1 in that order.
2. Tell Kamren the bags are there, or pull them yourself from the laptop:

   ```bash
   rsync -a kinova@10.12.140.145:insertion_bags/ ~/robot-logs/insertion_bags/
   ```

Bags stay in `~/insertion_bags` on REAL-1 (not `/tmp`), so they survive a
reboot, but copy them off the same day anyway.

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `No marker centre available -- refusing to insert` | Markers not in view, or terminal 5 not running |
| Prompts return instantly / phases skipped | Terminal has no real stdin (tmux quirk) — run terminal 6 in a plain terminal |
| `guard_trip` / motion STOPPED during insert | Force guard saw contact; the arm extracts along the axis. Note it in the log |
| Planning failed (code −31 / −1) | Square is out of reach at this angle; move it closer and rerun |
| `FJT error -5 ... it did not get there`, arm never moved | Arm driver stopped itself after cyclic timeouts — see the raw-camera warning above; restart terminal 1 |
| Arm does anything unexpected | E-stop. Do not rerun until someone has looked at it |
