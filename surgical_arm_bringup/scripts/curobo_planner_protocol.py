#!/usr/bin/env python3
"""curobo_planner_protocol.py — wire protocol for the cuRobo sidecar planner.

Shared by two processes that run under DIFFERENT Python interpreters:
  * the sidecar server  (conda env 'curobo', torch 2.4.1+cu124)  — see
    curobo_planner_server.py
  * the ROS-side client (system Python, ROS 2 Humble)            — see
    curobo_planner_client.py

Because the two interpreters cannot share a torch/CUDA stack, they talk over a
local Unix-domain socket instead of sharing memory. This module is the ONLY code
both sides import, so it MUST stay pure standard library.

Framing
-------
Each message is a 4-byte big-endian unsigned length prefix followed by a UTF-8
JSON body. Simple, self-delimiting, and language-agnostic.

Request  (client -> server)
---------------------------
    {
      "protocol": 1,
      "goal_type": "ptp_pose" | "linear_pose" | "reorient_about_link" | "ping",
      "joint_names": ["joint_1", ...],          # canonical order
      "start_joints": {"joint_1": 0.0, ...},    # live arm state (IK seed)
      "ee_link": "bracelet_link",               # IK tip cuRobo plans to
      "target": {"position": [x,y,z], "quaternion": [x,y,z,w]},
      "hold_link": "assembly_tip",              # reorient: keep fixed
      "lock_axes": ["x","y"],                   # linear: task-space axis lock
      "vel_scale": 0.25,
      "world_id": "insert_scene",
      "goal_joints": {"joint_1": 0.1, ...}      # optional; stub planner only
    }

Response (server -> client)
---------------------------
    {
      "protocol": 1,
      "success": true,
      "joint_names": ["joint_1", ...],
      "points": [
        {"positions": [...], "velocities": [...], "accelerations": [...],
         "time_from_start": 0.35},
        ...
      ],
      "error": "",
      "meta": {"backend": "stub"}
    }
"""
import json
import struct

PROTOCOL_VERSION = 1
DEFAULT_SOCKET_PATH = "/tmp/curobo_planner.sock"

# Goal types
GOAL_PTP = "ptp_pose"
GOAL_LINEAR = "linear_pose"
GOAL_REORIENT = "reorient_about_link"
GOAL_PING = "ping"

ALL_GOAL_TYPES = (GOAL_PTP, GOAL_LINEAR, GOAL_REORIENT, GOAL_PING)


class ProtocolError(Exception):
    """Raised on framing / decoding / contract violations."""


def _recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ProtocolError("socket closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock, obj):
    """Send a dict as a length-prefixed JSON frame."""
    body = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">I", len(body)) + body)


def recv_msg(sock):
    """Receive one length-prefixed JSON frame, returning a dict."""
    (length,) = struct.unpack(">I", _recv_exactly(sock, 4))
    return json.loads(_recv_exactly(sock, length).decode("utf-8"))


def make_request(goal_type, **kw):
    if goal_type not in ALL_GOAL_TYPES:
        raise ProtocolError(f"unknown goal_type {goal_type!r}")
    req = {"protocol": PROTOCOL_VERSION, "goal_type": goal_type}
    req.update(kw)
    return req


def make_response(success, joint_names=None, points=None, error="", meta=None):
    return {
        "protocol": PROTOCOL_VERSION,
        "success": bool(success),
        "joint_names": list(joint_names or []),
        "points": list(points or []),
        "error": error,
        "meta": meta or {},
    }


def validate_request(req):
    """Cheap sanity checks; raises ProtocolError on a malformed request."""
    if not isinstance(req, dict):
        raise ProtocolError("request is not a JSON object")
    if req.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol mismatch: got {req.get('protocol')}, "
            f"expected {PROTOCOL_VERSION}")
    if req.get("goal_type") not in ALL_GOAL_TYPES:
        raise ProtocolError(f"bad goal_type {req.get('goal_type')!r}")
    return True
