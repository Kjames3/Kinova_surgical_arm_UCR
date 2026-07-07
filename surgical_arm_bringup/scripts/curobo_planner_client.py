#!/usr/bin/env python3
"""curobo_planner_client.py — ROS-side client for the cuRobo sidecar planner.

Runs under system Python (ROS 2 Humble) inside insertion.py. Talks to
curobo_planner_server.py (conda env) over a Unix-domain socket using
curobo_planner_protocol.py. See that module for the rationale (two interpreters,
one socket) and the wire contract.

Design goals
------------
* Importable WITHOUT ROS (the socket/JSON layer is pure stdlib), so the M0
  wiring test can exercise it directly. The optional ``to_robot_trajectory``
  helper imports moveit_msgs lazily, only when actually building a ROS message.
* Connect-per-request: simple and robust; Unix-socket connects are ~free.
* Returns data shaped exactly like insertion.py's existing ``_plan()`` so the
  downstream ``_execute_fjt`` path is unchanged (M2 wiring).
"""
import socket
import time

import curobo_planner_protocol as proto


class CuroboPlannerError(Exception):
    pass


class CuroboPlannerClient:
    def __init__(self, socket_path=proto.DEFAULT_SOCKET_PATH, timeout=30.0):
        self.socket_path = socket_path
        self.timeout = timeout

    # -- transport ---------------------------------------------------------
    def _request(self, payload, timeout=None):
        """Send one request dict, return the response dict. Raises on transport
        failure (so callers can fall back to Pilz)."""
        to = self.timeout if timeout is None else timeout
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(to)
            sock.connect(self.socket_path)
        except OSError as e:
            raise CuroboPlannerError(
                f"cannot reach planner sidecar at {self.socket_path}: {e}")
        try:
            proto.send_msg(sock, payload)
            return proto.recv_msg(sock)
        except (OSError, proto.ProtocolError) as e:
            raise CuroboPlannerError(f"planner request failed: {e}")
        finally:
            sock.close()

    def ping(self, timeout=2.0):
        try:
            resp = self._request(proto.make_request(proto.GOAL_PING), timeout=timeout)
            return bool(resp.get("success"))
        except CuroboPlannerError:
            return False

    def wait_until_ready(self, timeout=20.0, poll=0.25):
        """Block until the sidecar answers a ping or timeout elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ping():
                return True
            time.sleep(poll)
        return False

    # -- high-level goals --------------------------------------------------
    def plan_ptp(self, joint_names, start_joints, ee_link, position, quaternion,
                 vel_scale=0.25, world_id="insert_scene", **extra):
        req = proto.make_request(
            proto.GOAL_PTP, joint_names=list(joint_names),
            start_joints=dict(start_joints), ee_link=ee_link,
            target={"position": list(position), "quaternion": list(quaternion)},
            vel_scale=vel_scale, world_id=world_id, **extra)
        return self._request(req)

    def plan_linear(self, joint_names, start_joints, ee_link, position, quaternion,
                    lock_axes=("x", "y"), vel_scale=0.25, world_id="insert_scene",
                    **extra):
        req = proto.make_request(
            proto.GOAL_LINEAR, joint_names=list(joint_names),
            start_joints=dict(start_joints), ee_link=ee_link,
            target={"position": list(position), "quaternion": list(quaternion)},
            lock_axes=list(lock_axes), vel_scale=vel_scale, world_id=world_id,
            **extra)
        return self._request(req)

    def plan_reorient(self, joint_names, start_joints, ee_link, hold_link,
                      position, quaternion, vel_scale=0.25,
                      world_id="insert_scene", **extra):
        req = proto.make_request(
            proto.GOAL_REORIENT, joint_names=list(joint_names),
            start_joints=dict(start_joints), ee_link=ee_link, hold_link=hold_link,
            target={"position": list(position), "quaternion": list(quaternion)},
            vel_scale=vel_scale, world_id=world_id, **extra)
        return self._request(req)

    # -- ROS conversion (lazy import; only used inside the ROS node) --------
    @staticmethod
    def to_robot_trajectory(resp):
        """Convert a planner response dict into a moveit_msgs/RobotTrajectory,
        matching what insertion.py's _plan() returns. Import is lazy so this
        module stays usable without ROS on the path."""
        if not resp.get("success"):
            raise CuroboPlannerError(resp.get("error") or "planner reported failure")
        from moveit_msgs.msg import RobotTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from builtin_interfaces.msg import Duration as RosDuration

        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = list(resp["joint_names"])
        for p in resp["points"]:
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in p["positions"]]
            pt.velocities = [float(x) for x in p.get("velocities", [])]
            pt.accelerations = [float(x) for x in p.get("accelerations", [])]
            t = float(p["time_from_start"])
            pt.time_from_start = RosDuration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            traj.joint_trajectory.points.append(pt)
        return traj
