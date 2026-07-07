#!/usr/bin/env python3
"""curobo_planner_server.py — sidecar motion-planning server (runs in conda env).

Runs under the isolated conda 'curobo' Python (torch 2.4.1+cu124). Listens on a
Unix-domain socket and answers plan requests from the ROS-side client
(curobo_planner_client.py), which runs under system Python. See
curobo_planner_protocol.py for the wire contract and the rationale for the split.

M0 scope
--------
Only the STUB backend is implemented. It proves the ROS<->conda wiring end to end
by returning a well-formed, SAFE hold-in-place trajectory (every waypoint equals
the start joint state) — enough for a dry-run round-trip without touching cuRobo.

The real CUROBO backend (M1+) drops in behind the same Planner interface: build
the robot config + collision world once at startup (warm the JIT kernels), then
translate each request into a cuRobo plan and emit the same waypoint list.

Run (normally launched by the client/test via `conda run -n curobo`):
    python curobo_planner_server.py --socket /tmp/curobo_planner.sock --backend stub
"""
import argparse
import os
import signal
import socket
import sys
import traceback

import curobo_planner_protocol as proto


# ---------------------------------------------------------------------------
# Planner backends
# ---------------------------------------------------------------------------
class StubPlanner:
    """M0 placeholder. Returns a hold-in-place trajectory.

    If the request carries optional 'goal_joints', it linearly interpolates from
    start to goal (to exercise the waypoint machinery); otherwise it holds the
    start pose. It never consults the Cartesian target — pose/IK is M1+.
    """

    name = "stub"

    def __init__(self, n_points=10, duration=2.0):
        self.n_points = n_points
        self.duration = duration

    def plan(self, req):
        joint_names = req.get("joint_names") or sorted(
            (req.get("start_joints") or {}).keys())
        if not joint_names:
            return proto.make_response(
                False, error="no joint_names / start_joints in request",
                meta={"backend": self.name})

        start = req.get("start_joints") or {}
        goal = req.get("goal_joints") or start  # default: hold in place
        q0 = [float(start.get(j, 0.0)) for j in joint_names]
        q1 = [float(goal.get(j, start.get(j, 0.0))) for j in joint_names]

        n = max(2, int(self.n_points))
        pts = []
        for i in range(n):
            a = i / (n - 1)
            pos = [p0 + a * (p1 - p0) for p0, p1 in zip(q0, q1)]
            pts.append({
                "positions": pos,
                "velocities": [0.0] * len(joint_names),
                "accelerations": [0.0] * len(joint_names),
                "time_from_start": round(a * self.duration, 6),
            })
        return proto.make_response(
            True, joint_names=joint_names, points=pts,
            meta={"backend": self.name, "goal_type": req.get("goal_type"),
                  "stub": True})


class CuroboPlanner:
    """Real backend — implemented in M1. Placeholder so the CLI/interface exist."""

    name = "curobo"

    def __init__(self, **kw):
        # M1: build RobotConfig + Scene here and warm the solver with a throwaway
        # plan so the first real request doesn't pay JIT-compile latency.
        raise NotImplementedError(
            "CUROBO backend lands in M1 (robot config + collision world). "
            "Use --backend stub for M0 wiring.")

    def plan(self, req):  # pragma: no cover - not reachable in M0
        raise NotImplementedError


def make_planner(backend, **kw):
    if backend == "stub":
        return StubPlanner(**kw)
    if backend == "curobo":
        return CuroboPlanner(**kw)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Server loop
# ---------------------------------------------------------------------------
def _handle_conn(conn, planner, log):
    with conn:
        while True:
            try:
                req = proto.recv_msg(conn)
            except proto.ProtocolError:
                return  # client closed / EOF
            try:
                proto.validate_request(req)
                if req["goal_type"] == proto.GOAL_PING:
                    resp = proto.make_response(
                        True, meta={"backend": planner.name, "pong": True})
                else:
                    resp = planner.plan(req)
            except Exception as e:  # never let one bad request kill the server
                log(f"request error: {e}\n{traceback.format_exc()}")
                resp = proto.make_response(
                    False, error=f"{type(e).__name__}: {e}",
                    meta={"backend": planner.name})
            proto.send_msg(conn, resp)


def serve(socket_path, backend, n_points, duration):
    def log(msg):
        print(f"[curobo_planner_server] {msg}", flush=True)

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    planner = make_planner(backend, n_points=n_points, duration=duration)
    log(f"backend={planner.name} python={sys.executable}")

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(8)
    log(f"listening on {socket_path}")

    stopping = {"flag": False}

    def _stop(signum, frame):
        stopping["flag"] = True
        try:
            srv.close()
        except OSError:
            pass

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not stopping["flag"]:
            try:
                conn, _ = srv.accept()
            except OSError:
                break  # socket closed by signal handler
            _handle_conn(conn, planner, log)
    finally:
        try:
            srv.close()
        finally:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
        log("shut down")


def main(argv=None):
    ap = argparse.ArgumentParser(description="cuRobo sidecar planner server")
    ap.add_argument("--socket", default=proto.DEFAULT_SOCKET_PATH)
    ap.add_argument("--backend", default="stub", choices=("stub", "curobo"))
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--duration", type=float, default=2.0)
    args = ap.parse_args(argv)
    serve(args.socket, args.backend, args.n_points, args.duration)


if __name__ == "__main__":
    main()
