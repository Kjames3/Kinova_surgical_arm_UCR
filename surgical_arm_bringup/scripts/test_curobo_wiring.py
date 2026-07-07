#!/usr/bin/env python3
"""test_curobo_wiring.py — M0 end-to-end wiring test for the cuRobo sidecar.

Proves the ROS<->conda boundary works: launches curobo_planner_server.py inside
the conda 'curobo' env (via `conda run`) and drives it with CuroboPlannerClient
from THIS (system) interpreter, over the Unix socket. Uses the STUB backend, so
no cuRobo/torch is needed yet — this is purely a plumbing check.

Run:
    python3 test_curobo_wiring.py
    CUROBO_CONDA=~/miniforge3/bin/conda python3 test_curobo_wiring.py   # override

Exit code 0 = PASS.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from curobo_planner_client import CuroboPlannerClient, CuroboPlannerError  # noqa: E402

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4",
               "joint_5", "joint_6", "joint_7"]
START = {j: 0.1 * i for i, j in enumerate(JOINT_NAMES)}
GOAL = {j: START[j] + 0.2 for j in JOINT_NAMES}


def _resolve_conda():
    for cand in (os.environ.get("CUROBO_CONDA"),
                 os.environ.get("CONDA_EXE"),
                 os.path.expanduser("~/miniforge3/bin/conda"),
                 shutil.which("conda")):
        if cand and os.path.exists(os.path.expanduser(cand)):
            return os.path.expanduser(cand)
    raise SystemExit("FAIL: could not locate a conda binary (set CUROBO_CONDA).")


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [ok] {msg}")


def main():
    conda = _resolve_conda()
    server = os.path.join(_HERE, "curobo_planner_server.py")
    sock = os.path.join(tempfile.mkdtemp(prefix="curobo_wire_"), "planner.sock")
    print(f"conda={conda}\nserver={server}\nsocket={sock}\n")

    # 0) Before the server is up, the client must fail gracefully (Pilz fallback).
    down = CuroboPlannerClient(sock)
    _check(down.ping(timeout=1.0) is False,
           "ping returns False when sidecar is down (graceful fallback)")

    # Launch the sidecar in the conda env, isolated from ROS/system site-packages.
    env = dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1")
    proc = subprocess.Popen(
        [conda, "run", "--no-capture-output", "-n", "curobo",
         "python", server, "--socket", sock, "--backend", "stub",
         "--n-points", "8", "--duration", "1.6"],
        cwd=_HERE, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        client = CuroboPlannerClient(sock)
        _check(client.wait_until_ready(timeout=40.0),
               "sidecar (conda env) came up and answered ping")

        resp = client.plan_ptp(
            joint_names=JOINT_NAMES, start_joints=START, ee_link="bracelet_link",
            position=[0.26, 0.0, 0.30], quaternion=[0.0, 0.0, 0.0, 1.0],
            vel_scale=0.25, goal_joints=GOAL)

        _check(resp.get("success") is True, "plan_ptp returned success")
        _check(resp.get("meta", {}).get("backend") == "stub", "backend is stub")
        _check(resp["joint_names"] == JOINT_NAMES, "joint_names round-tripped in order")
        pts = resp["points"]
        _check(len(pts) == 8, f"got requested 8 waypoints (got {len(pts)})")

        times = [p["time_from_start"] for p in pts]
        _check(all(b >= a for a, b in zip(times, times[1:])),
               "time_from_start is monotonic non-decreasing")
        _check(abs(times[-1] - 1.6) < 1e-6, "final time matches requested duration")

        first = pts[0]["positions"]
        last = pts[-1]["positions"]
        _check(all(abs(a - START[j]) < 1e-9 for a, j in zip(first, JOINT_NAMES)),
               "first waypoint equals start joints")
        _check(all(abs(a - GOAL[j]) < 1e-9 for a, j in zip(last, JOINT_NAMES)),
               "last waypoint equals goal joints (stub interpolation works)")
        _check(all(len(p["positions"]) == 7 for p in pts), "each waypoint has 7 DOF")

        # Optional: if ROS is sourced, verify the moveit_msgs conversion too.
        try:
            traj = CuroboPlannerClient.to_robot_trajectory(resp)
            _check(list(traj.joint_trajectory.joint_names) == JOINT_NAMES,
                   "to_robot_trajectory built a RobotTrajectory (ROS present)")
            _check(len(traj.joint_trajectory.points) == 8,
                   "RobotTrajectory has 8 points")
        except ImportError:
            print("  [skip] moveit_msgs not on path — RobotTrajectory conversion "
                  "not checked (run with ROS sourced to include it)")

        print("\n=== M0 WIRING TEST: PASS ===")
        return 0
    except (AssertionError, CuroboPlannerError) as e:
        print(f"\n=== M0 WIRING TEST: FAIL ===\n{e}")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
