#!/usr/bin/env python3
"""Offline test for analyze_insertion_bag.py: writes a synthetic bag with known
answers (straight 100 mm LIN, 1 mm pivot wobble, joint_5 at half its limit, a
3 Nm effort step, an aborted phase) and checks every reported metric.

  source /opt/ros/humble/setup.bash && ./test_analyze_insertion_bag.py
"""
import math, os, shutil, subprocess, sys, csv, tempfile
TMP = tempfile.mkdtemp(prefix="bagtest_")
import numpy as np, rosbag2_py
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from control_msgs.msg import JointTrajectoryControllerState
out = os.path.join(TMP, "insertion_test"); shutil.rmtree(out, ignore_errors=True)
w = rosbag2_py.SequentialWriter()
w.open(rosbag2_py.StorageOptions(uri=out, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
for name, typ in [("/joint_states","sensor_msgs/msg/JointState"),("/tf","tf2_msgs/msg/TFMessage"),("/tf_static","tf2_msgs/msg/TFMessage"),
                  ("/insertion/phase","std_msgs/msg/String"),("/joint_trajectory_controller/controller_state","control_msgs/msg/JointTrajectoryControllerState")]:
    w.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))
T0 = 1000.0
def stamp(t): return Time(sec=int(t), nanosec=int(round((t % 1) * 1e9)))
def put(topic, msg, t): w.write(topic, serialize_message(msg), int(round(t * 1e9)))
J = [f"joint_{i}" for i in range(1, 8)]
# timeline (s after T0): idle 0-1, PhaseA 1-3 (tip +100mm x), idle 3-4.5, PhaseB 4.5-6.5 (pivot, 1mm wobble, tau step), PhaseC start 7 no end, bag ends 8
def tip(t):
    if t < 1: return (0.40, 0.0, 0.30)
    if t < 3: return (0.40 + 0.1 * (t - 1) / 2, 0.0, 0.30)
    if 4.5 <= t < 6.5: return (0.50, 0.001 * math.sin(math.pi * (t - 4.5) / 2), 0.30)
    return (0.50, 0.0, 0.30)
st = TransformStamped(); st.header.stamp = stamp(T0); st.header.frame_id = "world"; st.child_frame_id = "base_link"; st.transform.rotation.w = 1.0
put("/tf_static", TFMessage(transforms=[st]), T0)
for k in range(0, 8000):
    t = k / 1000.0; ta = T0 + t
    js = JointState(); js.header.stamp = stamp(ta); js.name = J + ["finger_joint"]
    dq5 = 0.6109 * math.sin(math.pi * (t - 1) / 2) if 1 <= t < 3 else 0.0     # joint_5 peak = 0.5 of 1.2218
    js.position = [0.0]*8; js.velocity = [0.0,0.0,0.0,0.0,float(dq5),0.0,0.0,0.0]
    js.effort = [0.0, 10.0 + (3.0 if 5.0 <= t < 6.5 else 0.0), 0.0,0.0,0.0,0.0,0.0,0.0]        # +3 Nm on joint_2 inside PhaseB
    put("/joint_states", js, ta)
    if k % 10 == 0:
        c = JointTrajectoryControllerState(); c.header.stamp = stamp(ta); c.joint_names = J
        c.error.positions = [0.0,0.0,0.0,0.0,0.0,0.0, 0.01 if 1 <= t < 3 else 0.0]           # 0.573 deg on joint_7 in PhaseA
        put("/joint_trajectory_controller/controller_state", c, ta)
    if k % 50 == 0:
        tr = TransformStamped(); tr.header.stamp = stamp(ta); tr.header.frame_id = "base_link"; tr.child_frame_id = "assembly_tip"
        tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = tip(t); tr.transform.rotation.w = 1.0
        put("/tf", TFMessage(transforms=[tr]), ta)
for t, txt in [(1, "start:PhaseA_lin"), (3, "end:PhaseA_lin:ok"), (4.5, "start:Phase3_tip_rotate"), (6.5, "end:Phase3_tip_rotate:ok"), (7, "start:PhaseC_abort")]:
    put("/insertion/phase", String(data=txt), T0 + t + 0.0005)
del w
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_insertion_bag.py")
r = subprocess.run([sys.executable, script, out, out, "--csv", os.path.join(TMP, "p.csv"), "--plot"], capture_output=True, text=True)
print(r.stdout[-2500:], r.stderr[-1500:])
rows = {(row["phase"]): row for row in csv.DictReader(open(os.path.join(TMP, "p.csv")))}
A, B, C = rows["PhaseA_lin"], rows["Phase3_tip_rotate"], rows["PhaseC_abort"]
def near(v, want, tol, what):
    v = float(v); assert abs(v - want) <= tol, f"{what}: got {v}, want {want}±{tol}"; print(f"  PASS {what}: {v:.3f}")
near(A["motion_s"], 2.0, 0.01, "A motion_s"); near(A["idle_before_s"], 1.0, 0.01, "A idle")
near(A["path_mm"], 100.0, 1.0, "A path"); near(A["max_dev_mm"], 0.0, 0.05, "A straightness")
near(A["mean_speed_mms"], 50.0, 1.0, "A mean speed")
near(A["vel_ratio"], 0.5, 0.01, "A vel ratio"); assert A["vel_ratio_joint"] == "joint_5"
near(A["track_err_deg"], math.degrees(0.01), 0.01, "A tracking"); assert A["track_err_joint"] == "joint_7"
near(B["idle_before_s"], 1.5, 0.01, "B idle"); near(B["max_from_start_mm"], 1.0, 0.05, "B pivot wobble")
near(B["dtau_max_nm"], 3.0, 1e-6, "B effort step"); assert B["dtau_joint"] == "joint_2"
near(A["dtau_max_nm"], 0.0, 1e-9, "A no effort change")
assert C["ok"] == "False"; near(C["motion_s"], 1.0, 0.01, "C aborted runs to bag end")
assert os.path.exists(os.path.join(TMP, "insertion_test.png")); print("  PASS plot written")
print("ALL PASS")
