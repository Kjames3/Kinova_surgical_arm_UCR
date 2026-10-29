#!/usr/bin/env python3
"""Terminal menu for demoing impedance.py + the table barrier.

Pick a scenario, tweak a few parameters, confirm, and it runs impedance.py
with the assembled command line (printed first, so it is reproducible).

    source /home/kinova/ros2_kortex_ws/install/setup.bash
    ~/.venvs/kortex_impedance/bin/python demo_menu.py

Stop robot.launch.py first -- low-level servoing is exclusive.
Every torque run has a --duration (no open-ended runs from this menu);
Ctrl-C still stops early and impedance.py restores position mode.
"""
import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMPEDANCE = os.path.join(HERE, "impedance.py")
LOG_DIR = os.path.expanduser("~/demo-logs")
HOME_FILE = os.path.join(LOG_DIR, "demo_home_pose.json")

BARRIER = [
    ("table-z-min", "Barrier height for the tip origin (m, base frame)", 0.08, float),
    ("table-standoff", "Virtual-wall band thickness (m, keep > 0)", 0.06, float),
    ("table-stiffness", "Virtual-wall stiffness (N/m)", 1200.0, float),
]

# key, title, what to show, base args, editable params, torque?, note
SCENARIOS = [
    ("1", "Gravity check (no torque)",
     "Prints model vs measured gravity at the current pose. Safe warm-up.",
     ["--mode", "cartesian", "--validate-gravity"], [], False, None),
    ("2", "Free hand-guiding",
     "Arm floats weightless; push it anywhere, it stays where you leave it.",
     ["--mode", "cartesian", "--interaction-mode", "free"],
     [("free-hold-scale", "Position-hold stiffness scale", 1.0, float)], True, None),
    ("3", "Hold end-effector (spring)",
     "Tip is held in place; push it and it springs back. "
     "Hold it still at a new pose for a moment and it re-anchors there.",
     ["--mode", "cartesian", "--interaction-mode", "hold-ee"],
     [("cart-scale", "Task stiffness scale (lower = softer)", 1.0, float)], True, None),
    ("4", "Free guiding + table barrier",
     "Hand-guide the arm down toward the table: the virtual wall pushes back, "
     "the CBF forbids penetration.",
     ["--mode", "cartesian", "--interaction-mode", "free",
      "--table-avoidance", "--table-verbose"], BARRIER, True,
     "Barrier is rated ~20 N of push (wrist-limited). Don't lean on it."),
    ("5", "Hold end-effector + table barrier",
     "Spring hold, plus the wall when you push the tip toward the table.",
     ["--mode", "cartesian", "--interaction-mode", "hold-ee",
      "--table-avoidance", "--table-verbose"],
     [("cart-scale", "Task stiffness scale", 1.0, float)] + BARRIER, True,
     "Barrier is rated ~20 N of push (wrist-limited). Don't lean on it."),
    ("6", "Rendered environment",
     "Arm renders a virtual spring / inertia / passive environment at the tip.",
     ["--mode", "cartesian", "--interaction-mode", "render"],
     [("env", "Environment (spring | inertia | passive)", "spring", str)], True, None),
    ("7", "Setpoint tracking (EXPERIMENTAL)",
     "Tip sweeps A->B->A along an offset. Moving-q_ref fix is NOT yet "
     "hardware-tested; last hardware result was 66-92 mm tip error.",
     ["--mode", "cartesian", "--interaction-mode", "track"],
     [("track-period", "Seconds per A->B->A cycle", 8.0, float),
      ("track-offset", "Sweep offset DX DY DZ (m)", "0 0.05 0", str)], True,
     "First hardware run of the track fix -- keep hands near the e-stop."),
]


def ask(prompt, default, cast):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            print("    invalid, try again")


def yes(prompt):
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def build(sc, robot_ip, tool, python):
    key, title, _, base, params, torque, _ = sc
    cmd = [python, IMPEDANCE, "--robot-ip", robot_ip, "--tool", tool] + base
    for name, label, default, cast in params:
        if name == "table-z-min" and tool != "thesis_ee":
            # Only thesis_ee has a tool hull in impedance.py; any other tool is
            # unguarded below the flange, so raise the flange floor instead.
            # 0.30 = table (-0.03) + 2F-140 length (~0.25) + ~8 cm margin.
            print(f"    {tool}: no tool hull -- barrier guards the FLANGE only; "
                  "default raised to 0.30 so the fingers clear the table")
            default = 0.30
        val = ask(label, default, cast)
        if name == "table-standoff" and val <= 0:
            print("    standoff 0 disables the wall (CBF alone penetrates); using 0.06")
            val = 0.06
        if name == "env" and val not in ("spring", "inertia", "passive"):
            print(f"    unknown env {val!r}; using spring")
            val = "spring"
        cmd += [f"--{name}"] + (str(val).split() if name == "track-offset" else [str(val)])
    if torque:
        dur = ask("Duration (s, 5-300)", 60.0, float)
        cmd += ["--duration", str(min(max(dur, 5.0), 300.0))]
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = title.split(" (")[0].lower().replace(" ", "_").replace("+", "and")
        cmd += ["--log", os.path.join(LOG_DIR, f"demo_{key}_{slug}_{stamp}.csv")]
    return cmd


def _arm(robot_ip):
    """Import impedance.py's Kortex helpers lazily (needs the kortex venv)."""
    sys.path.insert(0, HERE)
    import impedance as imp
    return imp


def save_home(robot_ip):
    """Record the current joint angles as the pose every demo starts from."""
    imp = _arm(robot_ip)
    with imp.DeviceConnection(robot_ip, imp.TCP_PORT, imp.DEFAULT_CREDENTIALS) as router:
        fb = imp.BaseCyclicClient(router).RefreshFeedback()
        q_deg = [fb.actuators[i].position for i in range(len(fb.actuators))]
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(HOME_FILE, "w") as f:
        json.dump({"q_deg": q_deg, "saved": datetime.datetime.now().isoformat()}, f)
    print("Saved demo home (deg): " + ", ".join(f"{v:.1f}" for v in q_deg))
    print("Tip: run demo 1 here once to confirm the gravity check passes.")


def move_home(robot_ip):
    """Joint-space move (arm's own planner, position mode) to the demo home,
    or to Kinova's built-in 'Home' action if none has been saved."""
    imp = _arm(robot_ip)
    with imp.DeviceConnection(robot_ip, imp.TCP_PORT, imp.DEFAULT_CREDENTIALS) as router:
        base = imp.BaseClient(router)
        servo = imp.Base_pb2.ServoingModeInformation()
        servo.servoing_mode = imp.Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(servo)
        if os.path.exists(HOME_FILE):
            with open(HOME_FILE) as f:
                q_deg = json.load(f)["q_deg"]
            print("Target: saved demo home " + ", ".join(f"{v:.1f}" for v in q_deg))
            imp.return_to_home(base, len(q_deg), imp.np.deg2rad(q_deg), countdown=3)
            return
        req = imp.Base_pb2.RequestedActionType()
        req.action_type = imp.Base_pb2.REACH_JOINT_ANGLES
        home = next((a for a in base.ReadAllActions(req).action_list
                     if a.name == "Home"), None)
        if home is None:
            print("No saved demo home and no Kinova 'Home' action found.")
            return
        print("No saved demo home -- using Kinova's built-in 'Home' pose.")
        q = [j.value for j in home.reach_joint_angles.joint_angles.joint_angles]
        imp.return_to_home(base, len(q), imp.np.deg2rad(q), countdown=3)


def run(cmd):
    # Ctrl-C reaches the child via the terminal's process group; the menu
    # ignores it while waiting so impedance.py's cleanup runs undisturbed.
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        rc = subprocess.call(cmd, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    finally:
        signal.signal(signal.SIGINT, old)
    print(f"\n--- impedance.py exited with code {rc} ---")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default="192.168.1.10")
    ap.add_argument("--tool", choices=["thesis_ee", "2f140", "none", "legacy"],
                    help="What is on the flange right now (asked if omitted)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print the commands, never run them")
    a = ap.parse_args()

    print("=" * 64)
    print(" Kinova Gen3 -- impedance control + table barrier demo")
    print("=" * 64)
    tool = a.tool
    while tool not in ("thesis_ee", "2f140", "none", "legacy"):
        tool = input("Tool on the flange? LOOK first (thesis_ee/2f140/none) [thesis_ee]: ").strip() or "thesis_ee"
    print("Checklist: robot.launch.py stopped | e-stop in reach | area clear\n")

    while True:
        for k, title, desc, *_ in SCENARIOS:
            print(f"  {k}) {title}\n       {desc}")
        saved = "saved pose" if os.path.exists(HOME_FILE) else "Kinova Home -- none saved yet"
        print(f"  h) Move arm to demo home ({saved})  <- do this before each demo")
        print("  s) Save current pose as demo home")
        print("  q) quit")
        choice = input("\nSelect demo: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return
        if choice in ("h", "s"):
            if a.dry_run:
                print("(dry run -- arm not touched)\n")
            elif choice == "s":
                save_home(a.robot_ip)
            elif yes("Arm will MOVE (position mode). Area clear?"):
                try:
                    move_home(a.robot_ip)
                except KeyboardInterrupt:
                    print("\ncancelled")
            print()
            continue
        sc = next((s for s in SCENARIOS if s[0] == choice), None)
        if sc is None:
            print("unknown choice\n")
            continue
        print(f"\n== {sc[1]} ==  (Enter keeps the default)")
        cmd = build(sc, a.robot_ip, tool, sys.executable)
        print("\nCommand:\n  " + shlex.join(cmd))
        if sc[6]:
            print(f"\n  NOTE: {sc[6]}")
        if sc[5]:
            print("  Torque engages after the countdown. Ctrl-C stops early.")
        if a.dry_run:
            print("(dry run -- not executed)\n")
        elif yes("\nRun it?"):
            run(cmd)
        print()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
