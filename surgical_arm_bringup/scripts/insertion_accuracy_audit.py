#!/usr/bin/env python3
"""Offline accuracy audit of recorded insertion bags (no robot, no ROS graph).

For every bag with a labelled tip offset, reconstruct from /joint_states and the
bag's own /robot_description:

  fk_err   where forward kinematics put the tip at the hold, minus the commanded
           target (mm) -- the arm's own tracking error
  meas     the operator-measured offset from the marker centre (mm)
  meas-fk  what is left once tracking error is removed: the target error
           (camera / extrinsic / TCP model)
  az, tilt the approach direction actually flown, from the tip's motion during
           Phase4b_insert (same convention as insertion_geometry: az 0 = +X,
           the direction the tip travels while descending)

Usage (source ROS first, for rclpy.serialization and urdf_parser_py):
  source /opt/ros/humble/setup.bash
  python3 insertion_accuracy_audit.py ~/robot-logs/insertion_bags/insertion_2026*/
"""
import argparse
import glob
import json
import math
import os
import sqlite3
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from urdf_parser_py.urdf import URDF


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = map(lambda f: f, (math.cos(r), math.sin(r), math.cos(p),
                                               math.sin(p), math.cos(y), math.sin(y)))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class Chain:
    """Serial chain root -> tip from a URDF string, FK with numpy."""

    def __init__(self, urdf_xml, root, tip):
        robot = URDF.from_xml_string(urdf_xml)
        by_child = {j.child: j for j in robot.joints}
        joints, link = [], tip
        while link != root:
            if link not in by_child:
                raise ValueError(f"{tip} is not below {root}")
            j = by_child[link]
            joints.append(j)
            link = j.parent
        self.joints = joints[::-1]

    def fk(self, q_by_name):
        T = np.eye(4)
        for j in self.joints:
            o = j.origin
            A = np.eye(4)
            if o is not None:
                A[:3, :3] = _rot(o.rpy or (0, 0, 0))
                A[:3, 3] = o.xyz or (0, 0, 0)
            T = T @ A
            if j.type in ("revolute", "continuous"):
                M = np.eye(4)
                M[:3, :3] = _axis_angle(j.axis, q_by_name[j.name])
                T = T @ M
            elif j.type == "prismatic":
                M = np.eye(4)
                M[:3, 3] = np.asarray(j.axis) * q_by_name[j.name]
                T = T @ M
        return T


def _read(db):
    c = sqlite3.connect(db)
    topics = {n: (i, t) for i, n, t in c.execute("select id,name,type from topics")}

    def rows(name):
        if name not in topics:
            return []
        return c.execute("select timestamp,data from messages where topic_id=? order by timestamp",
                         (topics[name][0],)).fetchall()

    info = [(t, json.loads(deserialize_message(d, String).data)) for t, d in rows("/insertion/run_info")]
    phase = [(t, deserialize_message(d, String).data) for t, d in rows("/insertion/phase")]
    urdf = [deserialize_message(d, String).data for _, d in rows("/robot_description")]
    return c, topics, info, phase, urdf


def _joints_between(c, topics, t0, t1):
    tid = topics["/joint_states"][0]
    out = []
    for t, d in c.execute("select timestamp,data from messages where topic_id=? and timestamp between ? and ?",
                          (tid, t0, t1)):
        m = deserialize_message(d, JointState)
        out.append((t, dict(zip(m.name, m.position))))
    return out


def analyse(bag_dir, world, tip):
    dbs = sorted(glob.glob(os.path.join(bag_dir, "*.db3")))
    if not dbs:
        return None
    c, topics, info, phase, urdf = _read(dbs[0])
    ev = {}
    for _, e in info:
        ev.setdefault(e.get("event"), e)
    if "measured_offset" not in ev or "target" not in ev or not urdf:
        return None
    marks = {p: t for t, p in phase}
    t_ins0 = marks.get("start:Phase4b_insert")
    t_ins1 = marks.get("end:Phase4b_insert:ok")
    t_ext = marks.get("start:Phase6a_extract")
    if None in (t_ins0, t_ins1, t_ext):
        return None
    chain = Chain(urdf[0], world, tip)

    hold = _joints_between(c, topics, t_ins1, t_ext)
    tips = np.array([chain.fk(q)[:3, 3] for _, q in hold[::50]])
    p_hold = np.median(tips, axis=0)

    start = _joints_between(c, topics, t_ins0, t_ins0 + int(2e8))[:1]
    p_start = chain.fk(start[0][1])[:3, 3]
    d = p_hold - p_start
    az = math.degrees(math.atan2(d[1], d[0]))
    tilt = math.degrees(math.atan2(math.hypot(d[0], d[1]), -d[2]))

    tg, mo = ev["target"], ev["measured_offset"]
    fk_err = (p_hold - np.array([tg["x"], tg["y"], tg["z"]])) * 1000.0
    meas = np.array([mo["dx_mm"], mo["dy_mm"]])
    q7 = math.degrees(np.median([q["joint_7"] for _, q in hold]))
    return dict(bag=os.path.basename(bag_dir.rstrip("/")), tx=tg["x"], ty=tg["y"],
                az=az, tilt=tilt, q7=q7, fk=fk_err, meas=meas, resid=meas - fk_err[:2],
                logged_az=ev.get("approach", {}).get("azimuth_deg"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+", help="bag directories")
    ap.add_argument("--world", default="world")
    ap.add_argument("--tip", default="assembly_tip")
    a = ap.parse_args()
    print(f"{'bag':26} {'target x,y':>15} {'az':>7} {'tilt':>5} {'j7':>7} "
          f"{'fk_err x,y,z mm':>20} {'meas x,y':>12} {'meas-fk x,y':>13}")
    rows = []
    for b in a.bags:
        if not os.path.isdir(b):
            continue
        try:
            r = analyse(b, a.world, a.tip)
        except Exception as exc:  # one broken bag must not hide the rest
            print(f"{os.path.basename(b.rstrip('/')):26} skipped: {exc}", file=sys.stderr)
            continue
        if r is None:
            continue
        rows.append(r)
        az = f"{r['az']:+7.1f}" + ("" if r["logged_az"] is None else f" (log {r['logged_az']:+.1f})")
        print(f"{r['bag']:26} {r['tx']:7.4f},{r['ty']:7.4f} {az} {r['tilt']:5.1f} {r['q7']:+7.1f} "
              f"{r['fk'][0]:+6.1f},{r['fk'][1]:+6.1f},{r['fk'][2]:+6.1f} "
              f"{r['meas'][0]:+5.1f},{r['meas'][1]:+5.1f} {r['resid'][0]:+6.1f},{r['resid'][1]:+6.1f}")
    if not rows:
        print("no labelled bags with a complete Phase4b found")


if __name__ == "__main__":
    main()
