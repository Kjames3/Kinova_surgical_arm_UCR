#!/usr/bin/env python3
"""analyze_insertion_bag.py -- per-phase motion analysis of insertion.py rosbags.

insertion.py records every executed run to ~/insertion_bags/insertion_<stamp>/
(record_bag:=true) and publishes a marker on /insertion/phase around every
executed motion: "start:<label>" and "end:<label>:ok|fail". This script cuts
the bag at those markers and scores each phase, so speed changes are measured
rather than guessed.

WHAT IT REPORTS, PER PHASE
--------------------------
  motion time      marker start -> end
  idle before      end of the previous phase -> this start: planning time plus
                   any step_by_step prompt (so compare runs with the same mode)
  tip path         assembly_tip path length, straight-line chord, max deviation
                   from that chord (straightness of LIN phases), max distance
                   from the start point (stillness of tip-pivot rotations),
                   mean and peak tip speed -- from /tf, world -> tip link
  joint speed      peak |dq| per joint and the worst ratio to the joint's
                   velocity limit (headroom = room to go faster)
  tracking error   peak |desired - actual| from the JTC controller_state
                   (large error = the controller is already struggling)
  effort change    peak |tau - tau_at_phase_start| per joint from
                   /joint_states effort -- a CONTACT PROXY, not a force.
                   Gravity torque changes with posture, so moving phases show
                   non-zero values with no contact at all. Compare the same
                   phase across runs, e.g. Phase4b_insert with and without
                   a jam.

Across several bags it also prints mean/std of motion time per phase, and
--csv writes one row per (bag, phase) for later tuning or learning.

USAGE
-----
  # Pull bags off REAL-1 first (REAL-1's python cannot import matplotlib):
  rsync -a kinova@10.12.140.145:insertion_bags/ ~/robot-logs/insertion_bags/

  source /opt/ros/humble/setup.bash
  ./analyze_insertion_bag.py ~/robot-logs/insertion_bags/insertion_20260914_*
  ./analyze_insertion_bag.py BAG --plot --csv phases.csv

Needs rosbag2_py and tf2_ros (ROS 2 Humble). Nothing here talks to a robot.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

import rclpy.time
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from tf2_ros import BufferCore


PHASE_TOPIC = "/insertion/phase"
JS_TOPIC = "/joint_states"
JTC_TOPIC = "/joint_trajectory_controller/controller_state"
ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
# Gen3 7-DOF: 80 deg/s on joints 1-4, 70 deg/s on 5-7 (Pilz reported
# "limit is 1.2218" for joint_5 on REAL-1).
DEFAULT_VEL_LIMITS = [1.3963] * 4 + [1.2218] * 3


# --- Bag reading -------------------------------------------------------------
def read_bag(path, tf_cache_s):
    """Return phases, joint samples, JTC error samples and a filled tf buffer."""
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=path, storage_id="")
    conv = rosbag2_py.ConverterOptions("", "")
    reader.open(storage, conv)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_cls = {name: get_message(tt) for name, tt in types.items()}

    buf = BufferCore(Duration(seconds=tf_cache_s))
    markers = []                      # (t, text)
    js_t, js_q, js_dq, js_tau = [], [], [], []
    jtc_t, jtc_err = [], []
    js_index = jtc_index = None
    t_first = None

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        t = t_ns * 1e-9
        if t_first is None:
            t_first = t
        if topic not in msg_cls:
            continue
        if topic in ("/tf", "/tf_static"):
            msg = deserialize_message(raw, msg_cls[topic])
            for tr in msg.transforms:
                if topic == "/tf_static":
                    buf.set_transform_static(tr, "bag")
                else:
                    buf.set_transform(tr, "bag")
        elif topic == PHASE_TOPIC:
            markers.append((t, deserialize_message(raw, msg_cls[topic]).data))
        elif topic == JS_TOPIC:
            msg = deserialize_message(raw, msg_cls[topic])
            if js_index is None:
                if not all(j in msg.name for j in ARM_JOINTS):
                    continue
                js_index = [list(msg.name).index(j) for j in ARM_JOINTS]
            js_t.append(t)
            js_q.append([msg.position[i] for i in js_index])
            js_dq.append([msg.velocity[i] for i in js_index] if len(msg.velocity) == len(msg.name) else [np.nan] * 7)
            js_tau.append([msg.effort[i] for i in js_index] if len(msg.effort) == len(msg.name) else [np.nan] * 7)
        elif topic == JTC_TOPIC:
            msg = deserialize_message(raw, msg_cls[topic])
            if jtc_index is None:
                if not all(j in msg.joint_names for j in ARM_JOINTS):
                    continue
                jtc_index = [list(msg.joint_names).index(j) for j in ARM_JOINTS]
            # Humble fills both the legacy desired/actual/error and the newer
            # reference/feedback/error; error is the same field either way.
            e = msg.error.positions
            if len(e) != len(msg.joint_names):
                continue
            jtc_t.append(t)
            jtc_err.append([e[i] for i in jtc_index])

    js = dict(t=np.array(js_t), q=np.array(js_q), dq=np.array(js_dq), tau=np.array(js_tau))
    jtc = dict(t=np.array(jtc_t), err=np.array(jtc_err))
    return pair_markers(markers), js, jtc, buf, t_first


def pair_markers(markers):
    """Turn start/end markers into [(label, t_start, t_end, ok)] in start order."""
    open_ = {}
    phases = []
    for t, text in markers:
        if text.startswith("start:"):
            open_[text[len("start:"):]] = t
        elif text.startswith("end:"):
            label, _, status = text[len("end:"):].rpartition(":")
            if label in open_:
                phases.append((label, open_.pop(label), t, status == "ok"))
    for label, t0 in open_.items():          # run aborted mid-phase
        phases.append((label, t0, None, False))
    return sorted(phases, key=lambda p: p[1])


# --- Metrics -----------------------------------------------------------------
def tip_path(buf, world, tip, t0, t1, dt):
    pts, ts = [], []
    for t in np.arange(t0, t1 + 1e-9, dt):
        try:
            tr = buf.lookup_transform_core(world, tip, rclpy.time.Time(seconds=float(t)))
        except Exception:
            continue
        p = tr.transform.translation
        pts.append((p.x, p.y, p.z))
        ts.append(t)
    return np.array(ts), np.array(pts)


def path_metrics(ts, pts):
    if len(pts) < 2:
        return {}
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    chord = pts[-1] - pts[0]
    chord_len = float(np.linalg.norm(chord))
    if chord_len > 1e-6:
        u = chord / chord_len
        rel = pts - pts[0]
        dev = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    else:
        dev = np.linalg.norm(pts - pts[0], axis=1)
    speed = seg / np.maximum(np.diff(ts), 1e-9)
    return dict(
        path_mm=float(seg.sum() * 1000), chord_mm=chord_len * 1000,
        max_dev_mm=float(dev.max() * 1000),
        max_from_start_mm=float(np.linalg.norm(pts - pts[0], axis=1).max() * 1000),
        mean_speed_mms=float(seg.sum() / max(ts[-1] - ts[0], 1e-9) * 1000),
        peak_speed_mms=float(speed.max() * 1000))


def window(d, t0, t1):
    if len(d["t"]) == 0:
        return None
    m = (d["t"] >= t0) & (d["t"] <= t1)
    return m if m.any() else None


def analyse_bag(path, args):
    phases, js, jtc, buf, t_first = read_bag(path, args.tf_cache)
    name = os.path.basename(os.path.normpath(path))
    vel_lim = np.array(args.vel_limits)
    rows = []
    print(f"\n=== {name} ===")
    if not phases:
        print("  no /insertion/phase markers -- was this recorded by insertion.py >= 0e22991?")
        return rows, None
    if len(js["t"]) == 0:
        print("  WARNING: no arm joints on /joint_states")

    prev_end = t_first
    for label, t0, t1, ok in phases:
        row = dict(bag=name, phase=label, ok=ok and t1 is not None,
                   start_s=t0 - t_first, idle_before_s=t0 - prev_end)
        t1_eff = t1 if t1 is not None else (js["t"][-1] if len(js["t"]) else t0)
        row["motion_s"] = t1_eff - t0
        prev_end = t1_eff

        ts, pts = tip_path(buf, args.world, args.tip_link, t0, t1_eff, args.tip_dt)
        row.update(path_metrics(ts, pts))

        m = window(js, t0, t1_eff)
        if m is not None:
            dq = js["dq"][m]
            if np.isnan(dq).all():            # no velocity field: difference q
                q, tt = js["q"][m], js["t"][m]
                dq = np.diff(q, axis=0) / np.maximum(np.diff(tt), 1e-9)[:, None]
            peak = np.nanmax(np.abs(dq), axis=0)
            ratio = peak / vel_lim
            row["peak_dq"] = peak
            row["vel_ratio"] = float(np.nanmax(ratio))
            row["vel_ratio_joint"] = ARM_JOINTS[int(np.nanargmax(ratio))]
            tau = js["tau"][m]
            if not np.isnan(tau).all():
                d_tau = np.abs(tau - tau[0])
                row["peak_dtau"] = np.nanmax(d_tau, axis=0)
                row["dtau_max_nm"] = float(np.nanmax(d_tau))
                row["dtau_joint"] = ARM_JOINTS[int(np.nanargmax(np.nanmax(d_tau, axis=0)))]

        m = window(jtc, t0, t1_eff)
        if m is not None:
            e = np.abs(jtc["err"][m])
            row["track_err_deg"] = float(np.degrees(e.max()))
            row["track_err_joint"] = ARM_JOINTS[int(np.argmax(e.max(axis=0)))]
        rows.append(row)

    print_bag(rows, js, t_first)
    return rows, (name, phases, js, jtc, t_first)


# --- Output ------------------------------------------------------------------
def fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else "-"


def print_bag(rows, js, t_first):
    hdr = (f"  {'phase':<26}{'ok':>3}{'idle s':>8}{'move s':>8}{'path mm':>9}{'dev mm':>8}"
           f"{'mm/s':>7}{'peak':>7}{'|dq|/lim':>13}{'trk deg':>13}{'d-tau Nm':>13}")
    print(hdr)
    for r in rows:
        vel = f"{fmt(r.get('vel_ratio'), '.2f')} {r.get('vel_ratio_joint', '')[-1:]}" if "vel_ratio" in r else "-"
        trk = f"{fmt(r.get('track_err_deg'), '.2f')} {r.get('track_err_joint', '')[-1:]}" if "track_err_deg" in r else "-"
        tau = f"{fmt(r.get('dtau_max_nm'), '.1f')} {r.get('dtau_joint', '')[-1:]}" if "dtau_max_nm" in r else "-"
        print(f"  {r['phase']:<26}{'Y' if r['ok'] else 'N':>3}{fmt(r['idle_before_s'], '.1f'):>8}"
              f"{fmt(r['motion_s'], '.1f'):>8}{fmt(r.get('path_mm'), '.0f'):>9}{fmt(r.get('max_dev_mm'), '.1f'):>8}"
              f"{fmt(r.get('mean_speed_mms'), '.0f'):>7}{fmt(r.get('peak_speed_mms'), '.0f'):>7}"
              f"{vel:>13}{trk:>13}{tau:>13}")
    total = (js["t"][-1] - t_first) if len(js["t"]) else rows[-1]["start_s"] + rows[-1]["motion_s"]
    moving = sum(r["motion_s"] for r in rows)
    print(f"  total {total:.1f} s: moving {moving:.1f} s ({100 * moving / max(total, 1e-9):.0f}%), "
          f"idle/planning {total - moving:.1f} s")
    print("  (joint letter after a ratio = worst joint; 'dev mm' = max deviation from the "
          "start->end chord, 'd-tau' = contact proxy, see --help)")
    for r in rows:
        if "Rotate" in r["phase"] or "rotate" in r["phase"]:
            if "max_from_start_mm" in r:
                print(f"  {r['phase']}: tip moved at most {r['max_from_start_mm']:.2f} mm from its start (pivot quality)")


def print_summary(all_rows):
    bags = {r["bag"] for r in all_rows}
    if len(bags) < 2:
        return
    print(f"\n=== {len(bags)} bags: motion time per phase ===")
    by = {}
    for r in all_rows:
        if r["ok"]:
            by.setdefault(r["phase"], []).append(r["motion_s"])
    for phase, v in by.items():
        v = np.array(v)
        print(f"  {phase:<26} n={len(v):<3} mean {v.mean():6.1f} s   std {v.std():5.2f} s")


CSV_FIELDS = ["bag", "phase", "ok", "start_s", "idle_before_s", "motion_s", "path_mm",
              "chord_mm", "max_dev_mm", "max_from_start_mm", "mean_speed_mms",
              "peak_speed_mms", "vel_ratio", "vel_ratio_joint", "track_err_deg",
              "track_err_joint", "dtau_max_nm", "dtau_joint"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS + [f"peak_dq_{j}" for j in ARM_JOINTS]
                           + [f"peak_dtau_{j}" for j in ARM_JOINTS], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            for key in ("peak_dq", "peak_dtau"):
                if key in r:
                    out.update({f"{key}_{j}": float(v) for j, v in zip(ARM_JOINTS, r[key])})
            w.writerow(out)
    print(f"\nwrote {path}")


def make_plot(bag_path, data, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name, phases, js, jtc, t_first = data
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    if len(js["t"]):
        t = js["t"] - t_first
        dq = js["dq"] if not np.isnan(js["dq"]).all() else np.gradient(js["q"], js["t"], axis=0)
        for i, j in enumerate(ARM_JOINTS):
            ax[0].plot(t, np.abs(dq[:, i]) / args.vel_limits[i], lw=0.8, label=j)
            if not np.isnan(js["tau"]).all():
                ax[2].plot(t, js["tau"][:, i], lw=0.8, label=j)
    if len(jtc["t"]):
        for i, j in enumerate(ARM_JOINTS):
            ax[1].plot(jtc["t"] - t_first, np.degrees(jtc["err"][:, i]), lw=0.8, label=j)
    ax[0].set_ylabel("|dq| / limit"); ax[0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("tracking error (deg)")
    ax[2].set_ylabel("joint effort (Nm)"); ax[2].set_xlabel("time since bag start (s)")
    t_last = js["t"][-1] if len(js["t"]) else max(p[1] for p in phases)
    for a in ax:
        for label, t0, t1, ok in phases:
            a.axvspan(t0 - t_first, (t1 if t1 else t_last) - t_first,
                      color="tab:green" if ok else "tab:red", alpha=0.08)
    for label, t0, t1, ok in phases:
        ax[0].text(t0 - t_first, 1.02, label, rotation=60, fontsize=7,
                   transform=ax[0].get_xaxis_transform(), va="bottom")
    ax[0].legend(ncol=7, fontsize=7, loc="upper right")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.normpath(bag_path)), f"{name}.png")
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bags", nargs="+", help="bag directories recorded by insertion.py")
    p.add_argument("--world", default="world")
    p.add_argument("--tip-link", default="assembly_tip")
    p.add_argument("--tip-dt", type=float, default=0.02, metavar="S",
                   help="sample spacing for the tip path (tf interpolates between messages)")
    p.add_argument("--vel-limits", type=float, nargs=7, default=DEFAULT_VEL_LIMITS, metavar="RAD_S")
    p.add_argument("--tf-cache", type=float, default=3600.0, metavar="S",
                   help="tf buffer length; must cover the whole bag")
    p.add_argument("--csv", metavar="FILE", help="write one row per (bag, phase)")
    p.add_argument("--plot", action="store_true", help="write a PNG next to each bag")
    args = p.parse_args()

    all_rows = []
    for bag in args.bags:
        if not os.path.isdir(bag):
            print(f"skip {bag}: not a bag directory", file=sys.stderr)
            continue
        rows, data = analyse_bag(bag, args)
        all_rows += rows
        if args.plot and data is not None:
            make_plot(bag, data, args)
    print_summary(all_rows)
    if args.csv and all_rows:
        write_csv(args.csv, all_rows)


if __name__ == "__main__":
    main()
