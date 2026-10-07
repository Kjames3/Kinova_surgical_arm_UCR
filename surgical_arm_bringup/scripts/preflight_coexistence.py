#!/usr/bin/env python3
"""Is somebody else's ROS stack alive on this machine?  Run before any session.

REAL-1 is shared.  A second stack (the KinovaARM voice pick-and-place project,
/mnt/ros_workspace, user `shash`) runs on the same ROS_DOMAIN_ID=42 against the
same arm, with its own URDF, its own camera extrinsics and its own node that
sends motion goals.  If it is up while we bring ours up:

  * two robot_state_publishers / static publishers feed one TF tree, so
    `world -> assembly_tip` and the camera frames silently come from whichever
    message arrived last;
  * two ros2_control_nodes fight over the arm's single Kortex session;
  * their arm_controller can move the arm under our script.

Read-only: it runs `ps`, nothing else.  It does not touch the ROS graph, so it
works before anything of ours is sourced and cannot disturb a running stack.

  ./preflight_coexistence.py            # exit 0 = clear, 1 = something is up
  ./preflight_coexistence.py --mine     # also list our own ROS processes
"""
import argparse
import collections
import getpass
import re
import subprocess
import sys

# What each kind of process means for us.  Order matters: first match wins.
KINDS = [
    ("arm session", re.compile(r"ros2_control_node|impedance\.py|gamepad_teleop\.py")),
    ("motion client", re.compile(r"arm_controller|insertion\.py|insert_to_container\.py|move_group(\s|$)")),
    ("tf publisher", re.compile(r"static_transform_publisher|robot_state_publisher|camera_tf_broadcaster")),
    ("camera driver", re.compile(r"realsense2_camera_node|kinova_vision_node|oak_camera_node|depthai")),
    ("ros node", re.compile(r"--ros-args|/opt/ros/\S+/bin/ros2 (run|launch)")),
]
BLOCKING = ("arm session", "motion client", "tf publisher")


def list_processes():
    out = subprocess.run(["ps", "-eo", "pid=,user=,etime=,args="],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            rows.append({"pid": int(parts[0]), "user": parts[1],
                         "etime": parts[2], "cmd": parts[3]})
    return rows


def classify(cmd):
    # `ros2 run` / `ros2 launch` wrappers duplicate the node they started.
    if re.search(r"/bin/ros2 (run|launch)\b", cmd) or "ros2-daemon" in cmd:
        return None
    for kind, pattern in KINDS:
        if pattern.search(cmd):
            return kind
    return None


def tf_child_frame(cmd):
    """Child frame a static publisher / camera_tf_broadcaster owns, or None."""
    m = re.search(r"--child-frame-id\s+(\S+)", cmd) or re.search(r"child_frame:=(\S+)", cmd)
    if m:
        return m.group(1)
    if "static_transform_publisher" in cmd:
        # legacy positional form: x y z yaw pitch roll [or quat] parent child
        positional = cmd.split("--ros-args")[0].split()[1:]
        if len(positional) >= 8 and not positional[-1].startswith("-"):
            return positional[-1]
    return None


def short(cmd, width=96):
    cmd = re.sub(r"--params-file \S+", "", cmd)
    return cmd if len(cmd) <= width else cmd[:width - 3] + "..."


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mine", action="store_true",
                        help="also list this user's own ROS processes")
    args = parser.parse_args()

    me = getpass.getuser()
    found = collections.defaultdict(list)       # (user, kind) -> rows
    tf_owners = collections.defaultdict(list)   # child frame -> rows
    for row in list_processes():
        kind = classify(row["cmd"])
        if kind is None:
            continue
        found[(row["user"], kind)].append(row)
        frame = tf_child_frame(row["cmd"]) if kind == "tf publisher" else None
        if frame:
            tf_owners[frame].append(row)

    problems = []
    others = sorted({user for user, _ in found if user != me})
    for user in others:
        print(f"\n== ROS processes owned by '{user}' ==")
        for kind, _ in KINDS:
            rows = found.get((user, kind), [])
            if not rows:
                continue
            blocking = kind in BLOCKING
            print(f"  [{'BLOCK' if blocking else ' info'}] {kind}: {len(rows)}")
            for row in rows:
                print(f"      {row['pid']:>8} up {row['etime']:>11}  {short(row['cmd'])}")
            if blocking:
                problems.append(f"{user} has {len(rows)} {kind} process(es) running")

    if args.mine:
        print(f"\n== our own ROS processes ('{me}') ==")
        mine = [(kind, row) for (user, kind), rows in found.items() if user == me
                for row in rows]
        for kind, row in sorted(mine, key=lambda kr: kr[1]["pid"]):
            print(f"      {row['pid']:>8} {kind:14s} {short(row['cmd'], 80)}")
        if not mine:
            print("      none")

    duplicates = {f: rows for f, rows in tf_owners.items() if len(rows) > 1}
    if duplicates:
        print("\n== TF frames with more than one static publisher ==")
        for frame, rows in sorted(duplicates.items()):
            owners = ", ".join(f"{r['user']}:{r['pid']}" for r in rows)
            print(f"  [BLOCK] {frame}: {owners}")
            problems.append(f"TF frame '{frame}' has {len(rows)} publishers")

    sessions = [row for (user, kind), rows in found.items() if kind == "arm session"
                for row in rows]
    if len(sessions) > 1:
        problems.append(f"{len(sessions)} processes want the arm's single Kortex session")

    print()
    if problems:
        print("NOT CLEAR -- do not bring our stack up or move the arm:")
        for problem in problems:
            print(f"  - {problem}")
        print("Ask the owner to stop their stack (their dashboard, or Ctrl-C on "
              "their launches). Do not kill another user's processes.")
        return 1
    print("CLEAR: no other user's arm, motion or TF processes are running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
