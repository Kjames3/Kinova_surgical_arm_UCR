#!/usr/bin/env python3
"""
robot_keepalive.py — periodic Kortex session heartbeat.

The Kortex arm disconnects sessions that are idle for longer than
session_inactivity_timeout_ms (default 2 min).  This script calls
RefreshFeedback every 30 s to keep the session alive during planning
or long pauses between motions.

Run by robot.launch.py in the background.
"""

import time
import sys

try:
    import kortex_api
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
    from kortex_api.SessionManager import SessionManager
    from kortex_api.autogen.messages import Session_pb2
    import kortex_api.TCPTransport as TCPTransport
except ImportError:
    # kortex_api Python bindings not available in this environment;
    # keepalive is a no-op (the ROS2 driver's own connection handles session renewal).
    print("[robot_keepalive] kortex_api not importable — keepalive is a no-op.", flush=True)
    while True:
        time.sleep(60)

ROBOT_IP   = "192.168.1.10"
ROBOT_PORT = 10000
INTERVAL_S = 30   # heartbeat every 30 s (session timeout is 120 s)

def main():
    transport = TCPTransport.TCPTransport()
    transport.connect(ROBOT_IP, ROBOT_PORT)
    router = RouterClient(transport, lambda kException: None)

    session_info = Session_pb2.CreateSessionInfo()
    session_info.username        = "admin"
    session_info.password        = "admin"
    session_info.session_inactivity_timeout   = 120000
    session_info.connection_inactivity_timeout = 10000

    session_manager = SessionManager(router)
    session_manager.CreateSession(session_info)

    base_cyclic = BaseCyclicClient(router)
    print(f"[robot_keepalive] Started — pinging {ROBOT_IP} every {INTERVAL_S} s", flush=True)

    try:
        while True:
            try:
                base_cyclic.RefreshFeedback()
            except Exception as exc:
                print(f"[robot_keepalive] RefreshFeedback error: {exc}", flush=True)
            time.sleep(INTERVAL_S)
    finally:
        session_manager.CloseSession()
        transport.disconnect()

if __name__ == "__main__":
    main()
