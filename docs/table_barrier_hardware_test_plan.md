# Table barrier: hardware test plan (first run on the arm)

Written 2026-10-27. Goal: confirm on the real arm that when an operator pushes the
end effector toward the table under `impedance.py --mode cartesian`, the virtual
wall resists and the HOCBF keeps every guarded point above its floor.

## 0. Gate: do not start until all of these are true

- [x] The momentum-observer / joint-limit-rows / NNLS work is **committed** (`b38bc9f`), and
      `sim_table_barrier_mujoco.py` + `validate_table_barrier.py` pass on the laptop.
- [ ] REAL-1's subrepo is updated. Its uncommitted edits are dated 2026-06-30 to
      2026-10-06, older than the laptop's later commits to the same files
      (e.g. combine_cameras.py `8361057`, 2026-10-13), so they are most likely stale
      copies. `git stash` them (don't discard them), then pull. As of 2026-10-27 it is at `663cd44`, **60 commits
      behind** laptop `main`, with its own uncommitted edits (moveit config,
      `robot.launch.py`, `cameras.launch.py`, `combine_cameras.py`,
      `insert_to_container.py`, ...). Stash or commit those **before** `git pull`.
      Then check that the md5 of `impedance.py` / `control_barrier.py` on REAL-1
      matches the laptop.
- [ ] Offline checks on REAL-1, nothing else: `--help`, `py_compile`, and run
      `validate_table_barrier.py` there.
- [ ] Measure the loop rate on REAL-1 with the barrier enabled. The tool hull added
      ~70 µs, and the observer adds one extra M(q) per cycle. If `--log` shows cycle
      overruns or cyclic timeouts in step 2, stop.
- [ ] Decide on the tool (see section 1). **As of 2026-10-27 the 2F-140 stays on
      (shared with other users) -> option B.**
- [x] Barrier columns added to `--log` (2026-10-27): `tb_status` (0 inactive,
      1 exact, 2 blocked, 3 degraded), `tb_clear` (smallest clearance over all
      guarded rows, m), `tb_row` (which row), `tb_fwall` (N), `tau_ext1..7`,
      `dt_max` (worst cycle since the previous logged row). Not yet seen in a
      real log: check the header after step 1.
- [ ] E-stop within reach of a second person; the table is clear except the test area.

## 1. Tool on the flange: 2F-140 (read from the arm 2026-10-27: 1.025 kg, CoM z 0.073)

The barrier only guards a tool that has a convex hull. **Only `thesis_ee` has one.**
With `--tool 2f140`, impedance.py prints `WARNING -- no hull ... nothing below the
flange is guarded`. The fingers hang about 0.2 m below the flange and would reach
the table first.

Options:
- **A (recommended for the thesis):** swap to `thesis_ee` and run `--tool thesis_ee`.
  This exercises the code path that matters (tool-hull rows), and the hull is exact.
- **B (if the 2F-140 stays on):** raise the link floor so the flange stays high enough
  for the fingers to clear the table: `--table-z-min 0.25 --table-tool-points 0`
  (table at −0.03 + ~0.2 m gripper + margin). This only tests the link rows, not the
  tool-hull rows. Measure the closed-finger length first and adjust the number.
- **C:** generate a 2F-140 hull CSV the same way as `thesis_ee_hull_bracelet.csv`.

## 2. Run sequence (each step must pass before the next one)

Common prefix, run on REAL-1 with `robot.launch.py` **not** running:

```bash
source /home/kinova/ros2_kortex_ws/install/setup.bash
cd ~/workspace/ros2_kortex_ws/src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts
PY=/home/kinova/.venvs/kortex_impedance/bin/python
# Option B (2F-140 on). The gripper is ~0.2 m long along the flange axis; in
# free mode the wrist can point it straight down, so the link floor covers that.
COMMON="--robot-ip 192.168.1.10 --tool 2f140 --mode cartesian --table-avoidance --table-verbose --table-tool-points 0"
# Option A (thesis_ee): --tool thesis_ee, drop --table-tool-points 0
```

Every run has a `--duration`. If you have to stop early, use `kill -INT <pid>` only.
Run in a real terminal on REAL-1 or in `tmux`, not as a bare ssh command (an
ssh-timeout SIGHUP skips the code that restores position mode). Log files get
**unique names** with the date and step, and are pulled the same day (`/pull-run`).

| Step | Purpose | Extra flags | Pass |
|---|---|---|---|
| 1 | Hold, barrier on, nobody touching | `--interaction-mode hold-ee --duration 30 --log /tmp/tb_20261027_s1_hold.csv` | Holds to the existing 0.5 mm level; filter `inactive` throughout; no `degraded` warnings; no cyclic timeouts |
| 2 | Observer at rest | same, `--duration 60` | Observer `tau_ext` settles to a steady value (mostly joint 4, the gravity error plus about 0.4 Nm from the wrist self-collision capsules). Record that value; it is the offset seen when nobody is pushing |
| 3 | Gentle push, high floor, soft wall | `--interaction-mode free --duration 60 --table-z-min 0.35` (2F-140; thesis_ee: `0.20 --table-tool-z-min 0.10`) ` --table-stiffness 400 --table-force-max 40 --table-standoff 0.06` | You feel the wall start about 6 cm above the floor; tool lowest point stays ≥ 0.10 m; filter `exact`/`inactive` only |
| 4 | Same floor, default wall | defaults for the wall (K 1200, F 90) | Wall feels stiff; penetration below the floor ≤ 1 mm |
| 5 | Lower the floor step by step | 2F-140: `--table-z-min` 0.35 → 0.30 → 0.25 (floor; never below). thesis_ee: `--table-tool-z-min` 0.05 → 0.0 → −0.02 | Same pass criteria at each floor height |
| 6 | Harder and faster pushes, push from different spots (flange, mid-tool, elbow) | defaults | Penetration ≤ 1 mm; note any `degraded` and the pose it happened in |
| 7 | Observer ablation, **high floor only** | step 3 flags + `--table-observer-gain 0` | Expected to be noticeably softer or to penetrate. This is the hardware confirmation of the 2026-10-26 sim finding. Never at a low floor |

Optional later: sweep `--table-observer-gain` (25 / 50 / 100) at step-4 settings to
trade estimator noise against how fast it reacts.

## 3. Abort immediately if any of these happen

- the arm moves on its own in `free` with no hand on it (possible observer runaway
  or feedback);
- buzzing/chatter at the wall (lower `--table-stiffness` or raise `--table-damping`);
- any `degraded` warning at a low floor, or the torque-clip backstop warning;
- cyclic timeouts.

After every run, confirm the safe state: `GetServoingMode() == SINGLE_LEVEL` and every
actuator `POSITION`.

## 4. What to record per run

Log path, tool, floor heights, K/D/F, observer gain, push site and rough force,
lowest tool-hull z and lowest link z (from the log), counts of each filter status,
the `tau_ext` value with nobody pushing, max cycle time. Score at the **tool tip / hull**,
never at the flange.

## Known caveats going in

- The wrist self-collision capsules are always slightly "violated", adding a
  constant +0.408 Nm on joint 4. The observer will treat that as an external
  push (harmless but biased).
- KDL's gravity is about 30% too low at joint 4. The observer absorbs this too, so
  `tau_ext` is not zero even with nobody touching the arm.
- `--table-z-min` / `--table-tool-z-min` assume the table surface is at z ≈ −0.03 in
  the base frame. Re-check that by jogging if the table or mount has moved.
