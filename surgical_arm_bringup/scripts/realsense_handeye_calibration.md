# RealSense eye-to-hand calibration — run procedure

Recalibrates `base_link → global_camera_color_optical_frame` for the fixed
RealSense D435i using the 12-marker board bolted to the arm.

Everything below runs **on REAL-1** (`kinova@10.12.140.145`), where the arm is
cabled. Four terminals, in this order. All four must share the same
`ROS_DOMAIN_ID` — the camera and robot were both on domain 0 for the
2026-09-03 run, so check with `ros2 topic list` if a node cannot see another.

Source the workspace in every terminal first:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_kortex_ws/install/setup.bash
```

## Terminal 1 — RealSense driver

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=realsense \
  camera_name:=camera \
  enable_color:=true \
  enable_depth:=false \
  pointcloud.enable:=false \
  rgb_camera.color_profile:=640x480x15
```

Keep the colour profile at 640x480. The D435i is on a USB 2.1 port that
disconnect-loops (`VIDIOC_QBUF → ENODEV`) at higher bandwidth.

## Terminal 2 — robot + MoveIt

```bash
ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config robot.launch.py \
  robot_ip:=192.168.1.10 \
  launch_rviz:=false
```

Wait for `You can start planning now!` before starting terminal 4.

## Terminal 3 — image view (optional, for aiming the camera)

```bash
ros2 run rqt_image_view rqt_image_view
```

## Terminal 4 — the calibration itself

Dry run first. `--automatic-plan-only` validates every sweep target and the
return leg **without moving the arm**:

```bash
ros2 run surgical_arm_bringup calibration_for_cameras.py \
  --camera realsense \
  --mode eye-to-hand \
  --robot-base-frame base_link \
  --robot-effector-frame bracelet_link \
  --target-type grid \
  --aruco-dict DICT_5X5_50 \
  --marker-size 0.0533 \
  --marker-gap 0.00505 \
  --grid-cols 4 \
  --grid-rows 3 \
  --grid-marker-ids 2,5,8,11,1,4,7,10,0,3,6,9 \
  --grid-corner-shift 1 \
  --image-transport compressed \
  --capture-mode automatic \
  --max-vel 0.10 \
  --max-accel 0.08 \
  --automatic-plan-only
```

Then the real run. This one **moves the arm** — stay at the E-stop. The board
is not in MoveIt's collision model, so keep the whole sweep volume clear:

```bash
ros2 run surgical_arm_bringup calibration_for_cameras.py \
  --camera realsense \
  --mode eye-to-hand \
  --robot-base-frame base_link \
  --robot-effector-frame bracelet_link \
  --target-type grid \
  --aruco-dict DICT_5X5_50 \
  --marker-size 0.0533 \
  --marker-gap 0.00505 \
  --grid-cols 4 \
  --grid-rows 3 \
  --grid-marker-ids 2,5,8,11,1,4,7,10,0,3,6,9 \
  --grid-corner-shift 1 \
  --image-transport compressed \
  --capture-mode automatic \
  --allow-automatic-motion \
  --max-vel 0.10 \
  --max-accel 0.08 \
  --motion-countdown 15 \
  --save-dir /home/kinova/Calibration_data/realsense_eye_to_hand/$(date +%Y-%m-%d)_auto_compressed_01
```

`--automatic-plan-only` passing does **not** mean the run will succeed — it
validates the modelled robot, not the board's clearance, and it does not
exercise the controller.

## Publishing the result — read this before copy-pasting

The node prints a ready-made `static_transform_publisher` line whose child
frame is `camera_color_optical_frame`. **That is the wrist camera's frame.**
Publishing it verbatim re-parents the fixed table camera onto the moving arm.
The correct child frame is `global_camera_color_optical_frame`:

```bash
ros2 run tf2_ros static_transform_publisher <x> <y> <z> <qx> <qy> <qz> <qw> \
  base_link global_camera_color_optical_frame
```

To make it permanent, edit `calibration_tf` in
`kinova_gen3_7dof_robotiq_2f_140_moveit_config/launch/cameras.launch.py`. That
package's `install/` tree on REAL-1 holds **file copies, not symlinks**, so a
src-only edit is silently ignored — rebuild the package or copy the file into
`install/.../share/.../launch/` as well.

## Validating the result

The solver's own residual (`translation RMS`, `rotation RMS`) is an AX=XB
*consistency* number and looks good even on poorly conditioned data. Score it
independently instead:

```bash
# static: needs a large AprilTag lying flat on the table, no arm motion
python3 validate_handeye_extrinsic.py --mode table \
  --extrinsic <x>,<y>,<z>,<qx>,<qy>,<qz>,<qw> \
  --table-marker-dict DICT_APRILTAG_36h11 --table-marker-ids 1,2 \
  --table-marker-size 0.1437

# stronger: board-in-bracelet must stay constant as the arm moves.
# Run it alongside another calibration sweep and it harvests poses passively.
python3 validate_handeye_extrinsic.py --mode invariance --extrinsic <...>
```

Use a **large** tag, not small markers. A single planar square has two poses
that reproject almost identically; a 143.7 mm tag36h11 separated them 3.07 px
vs 0.08 px, while a 51 mm marker managed only 0.31 vs 0.07 and the older
`camera_xcheck.py` silently returned the mirror solution.

The 2026-09-03 result scored +6.5 mm height error and 1.78° normal tilt against
a tag on the table; the superseded pre-move extrinsic scored +80.4 mm / 3.24°.

## Troubleshooting

**`execute_trajectory` aborts mid-sweep with `STATUS_ABORTED`.** Check
`ros2_control_node`'s log for `State tolerances failed for joint N` — `N` is a
**0-based index** into the JTC `joints` list, so `joint 2` means `joint_3`. A
`Position Error` of 6.2836 (2π) is a wrap, not mistracking: joints 1/3/5/7 are
continuous, the driver wraps feedback into [-π, π], and a path crossing the
seam keeps counting past it. Fixed by setting those four joints'
`trajectory`/`goal` tolerances to `0.0` in `ros2_controllers.yaml` (0.0 =
check disabled). Controller params load at configure time, so restart
terminal 2 after editing.

**Board detected but samples rejected.** The saved `pose_NN_ok.png` files are
the *annotated* frames — re-detecting markers in them undercounts, because the
drawn overlays sit on top of the markers. Read the green banner text instead;
a good frame reads `Grid OK: 12 tags, RMS 1.03px`.

**Spurious marker ids.** The checkerboard sheet in the scene decodes as
`DICT_4X4_50` ids 5, 8, 17, 24, 37, 42. The arm board is `DICT_5X5_50` and the
small table markers are `DICT_4X4_50`, so their shared ids 0/1 never collide.

**Thin sample diversity.** 15 samples spanning only ~38° of rotation leaves
hand-eye translation weakly observable. Raise `--auto-rotation-scale` before
adding more samples at the same spread.
