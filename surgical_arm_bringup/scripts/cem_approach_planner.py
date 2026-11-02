#!/usr/bin/env python3
"""Multi-start constrained CEM for choosing the insertion approach.

Offline and ROS-free: no robot, no move_group, no cuRobo. Only PyKDL +
urdf_parser_py (source /opt/ros/humble/setup.bash) and NumPy.

The insertion in insertion.py is fixed except for a few free choices, and the
tip pose alone does not decide them. This script searches over them:

  azimuth  compass direction the tool leans toward (rotation about world Z)
  tilt     lean of the tool from vertical
  elbow    the 7-DOF arm's redundancy: a null-space target for joint_3,
           relative to the start posture (joint_3 ~ the swivel/elbow angle)

For each candidate it solves IK along the same waypoint chain insertion.py
executes -- ready (vertical above the centre) -> hover (vertical) -> pivot
about the tip to the tilted orientation -> straight descent to the target --
and scores it.

Hard constraints (any violation makes the candidate infeasible):
  * IK converges at every sample on the path
  * bounded joints (2, 4, 6) keep --limit-margin-deg from their hard stops
  * no joint jumps between consecutive path samples (IK branch flips, and
    the joint_3 2*pi seam abort)
  * every link origin stays above the table, and outside / above the container
  * the tilted tool clears the container's inner wall at full depth

Task cost (lower is better, feasible candidates only):
  * joint-limit room, with joint_6 weighted heaviest (Phase 4a guard trips)
  * manipulability (smallest singular value of J) along the descent
  * joint travel from the start posture
  * deviation from the nominal tilt

Multi-start: K independent CEM runs, started at evenly spaced azimuths. Each
one refines its own Gaussian, so the arm's distinct good solutions are all
found instead of the search collapsing onto the first. Results are merged,
duplicates dropped, and the best reported along with the runner-ups.

  source /opt/ros/humble/setup.bash
  ./cem_approach_planner.py --target-x 0.40 --target-y 0.10
  ./cem_approach_planner.py --target-x 0.40 --target-y 0.10 --out /tmp/plan.json
  ./cem_approach_planner.py --selftest

Units on the command line are mm / deg; internally m / rad.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Generated surgical URDF (thesis_ee, assembly_tip at the 2026-10-09 measured
# offset). It is stored under config/curobo/ for historical reasons; nothing
# here uses cuRobo -- only the XML is parsed.
DEFAULT_URDF = os.path.join(os.path.dirname(_HERE), "config", "curobo",
                            "gen3_surgical.urdf")

from insertion_geometry import (
    CONTAINER_HEIGHT_M, CONTAINER_INNER_R, CONTAINER_WIDTH_M, GEN3_BOUNDED_LIMITS,
    TABLE_Z, insertion_axis_geometry, orientation_error, quat_slerp,
    tilt_quaternion, wrap_angle as wrap)

# Values mirrored from insertion.py parameter defaults so the two agree.
CONTAINER_BASE_OFFSET = 0.005    # m above table (container_base_offset)
CONTAINER_OUTER_R = CONTAINER_WIDTH_M / 2.0
APPROACH_CLEARANCE = 0.18        # m above container top = ready height
TIP_RADIUS = 0.003               # m, check_wall_clearance default
HARD_LIMITS = np.array([GEN3_BOUNDED_LIMITS.get(f"joint_{i}", np.inf)
                        for i in range(1, 8)])
CONTINUOUS = ~np.isfinite(HARD_LIMITS)
VERTICAL_QUAT = (-0.003, -0.007, 0.000, 1.000)  # bracelet at tool-down, xyzw
# cuRobo/MoveIt default posture; tool-down at (0.285, -0.007, 0.236).
DEFAULT_START = [0.0, -0.3049, -3.1416, -1.6607, 0.0, -1.7928, -0.0006]

LINK_RADIUS = 0.055              # m, as impedance.py's capsule model


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------
class ArmKinematics:
    """FK / Jacobian / per-link positions / damped least-squares IK."""

    def __init__(self, urdf, base_link="base_link", tip_link="assembly_tip"):
        from impedance import KinDynModel
        self.model = KinDynModel(urdf, base_link, tip_link)
        self.kdl = self.model.kdl
        self.n = self.model.n
        self.n_seg = self.model.chain.getNrOfSegments()
        self.seg_names = self.model.segment_names()
        self._frame = self.kdl.Frame()

    def fk(self, q):
        return self.model.fk(q)

    def jacobian(self, q):
        return self.model.jacobian(q)

    def link_points(self, q):
        """Base-frame origin of every segment, (n_seg, 3)."""
        jnt = self.model._to_jnt(q)
        pts = np.empty((self.n_seg, 3))
        for s in range(self.n_seg):
            self.model._fk.JntToCart(jnt, self._frame, s + 1)
            pts[s] = [self._frame.p[i] for i in range(3)]
        return pts

    def ik(self, p_des, quat_des, seed, q_null=None, null_mask=None,
           iters=150, pos_tol=5e-4, ori_tol=math.radians(0.3)):
        """DLS IK with a null-space pull toward q_null on null_mask joints.

        Bounded joints are clamped to the hard limits. Returns q or None.
        """
        q = np.array(seed, float)
        I = np.eye(self.n)
        lam2 = 0.03 ** 2
        for _ in range(iters):
            p, quat = self.fk(q)
            e = np.concatenate([p_des - p, orientation_error(quat_des, quat)])
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < ori_tol:
                return q
            J = self.jacobian(q)
            JJt = J @ J.T
            J_pinv = J.T @ np.linalg.inv(JJt + lam2 * np.eye(6))
            dq = J_pinv @ e
            if q_null is not None:
                pull = np.where(null_mask, q_null - q, 0.0)
                dq += (I - J_pinv @ J) @ (0.5 * pull)
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, -HARD_LIMITS, HARD_LIMITS)
        return None


# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
class InsertionProblem:
    """Evaluate one (azimuth, tilt, elbow) candidate: constraints + cost."""

    def __init__(self, kin, args):
        self.kin = kin
        self.a = args
        self.q_start = np.array(args.start_q, float)
        self.cx, self.cy = args.target_x / 1000.0, args.target_y / 1000.0
        self.top = TABLE_Z + CONTAINER_BASE_OFFSET + CONTAINER_HEIGHT_M
        self.depth = args.depth_mm / 1000.0
        self.target_z = self.top - self.depth
        self.ready_z = self.top + APPROACH_CLEARANCE
        self.qv = np.array(VERTICAL_QUAT, float)
        self.qv /= np.linalg.norm(self.qv)
        self.margin = math.radians(args.limit_margin_deg)
        self.null_mask = np.zeros(kin.n, bool)
        self.null_mask[2] = True                       # joint_3
        # Segments that must stay out of the container: everything proximal
        # of the tool. The tool itself (assembly_tip) is meant to go in.
        self.guard = [i for i, nm in enumerate(kin.seg_names)
                      if nm != "assembly_tip"]
        self.joint_w = np.array(args.joint_weights, float)

    def path_poses(self, azimuth, tilt):
        """[(tip_xyz, quat, label)] in execution order."""
        a = self.a
        hover, axis, descent = insertion_axis_geometry(
            self.cx, self.cy, self.ready_z, self.target_z, tilt, azimuth)
        hover, axis = np.array(hover), np.array(axis)
        q_t = tilt_quaternion(self.qv, azimuth, tilt)
        poses = [(np.array([self.cx, self.cy, self.ready_z]), self.qv, "ready"),
                 (hover, self.qv, "hover")]
        for k in range(1, a.pivot_samples + 1):
            poses.append((hover, quat_slerp(self.qv, q_t, k / a.pivot_samples),
                          "tilted" if k == a.pivot_samples else "pivot"))
        for k in range(1, a.descent_samples + 1):
            poses.append((hover + axis * descent * k / a.descent_samples, q_t,
                          "target" if k == a.descent_samples else "descent"))
        return poses

    def evaluate(self, x):
        """x = (azimuth, tilt, elbow) rad. Returns dict with cost, feasible, ..."""
        azimuth, tilt, elbow = float(x[0]), float(x[1]), float(x[2])
        out = {"x": [azimuth, tilt, elbow], "feasible": False}
        viol = 0.0
        reasons = []

        # Analytic wall check (insertion.check_wall_clearance, target at centre)
        lateral = self.depth * math.tan(tilt) + TIP_RADIUS
        if lateral > CONTAINER_INNER_R:
            viol += (lateral - CONTAINER_INNER_R) * 1000.0
            reasons.append("container wall")

        q_null = self.q_start.copy()
        q_null[2] += elbow
        seed = self.q_start
        path = []
        for p, quat, label in self.path_poses(azimuth, tilt):
            q = self.kin.ik(p, quat, seed, q_null, self.null_mask)
            if q is None:
                out.update(cost=1e3 + 100.0 + viol * 10.0,
                           reason=f"no IK at {label}", violation=viol + 100.0)
                return out
            path.append((label, q))
            seed = q

        Q = np.array([q for _, q in path])
        # Branch flips / seam: no big joint jump between samples
        jumps = np.abs(np.diff(np.vstack([self.q_start, Q]), axis=0))[1:]
        max_jump = float(jumps.max()) if len(jumps) else 0.0
        if max_jump > self.a.max_jump:
            viol += (max_jump - self.a.max_jump) * 10.0
            reasons.append(f"joint jump {math.degrees(max_jump):.0f} deg")

        # Joint-limit room
        bounded = np.isfinite(HARD_LIMITS)
        room = np.full(self.kin.n, np.inf)
        room[bounded] = (HARD_LIMITS[bounded] - np.abs(Q[:, bounded])).min(axis=0)
        if room[bounded].min() < self.margin:
            viol += (self.margin - room[bounded].min()) * 10.0
            j = int(np.argmin(np.where(bounded, room, np.inf)))
            reasons.append(f"joint_{j + 1} near limit")

        # Table and container clearance at every sample
        min_table = np.inf
        min_cont = np.inf
        for _, q in path:
            pts = self.kin.link_points(q)[self.guard]
            min_table = min(min_table,
                            float((pts[:, 2] - TABLE_Z - LINK_RADIUS).min()))
            r = np.hypot(pts[:, 0] - self.cx, pts[:, 1] - self.cy)
            inside = r < CONTAINER_OUTER_R + LINK_RADIUS
            if inside.any():
                min_cont = min(min_cont, float(
                    (pts[inside, 2] - self.top - LINK_RADIUS).min()))
        if min_table < 0.0:
            viol += -min_table * 100.0
            reasons.append("link below table")
        if min_cont < 0.0:
            viol += -min_cont * 100.0
            reasons.append("link inside container")

        out["violation"] = viol
        out["path"] = [(label, q.tolist()) for label, q in path]
        if viol > 0.0:
            out.update(cost=1e3 + viol, reason=", ".join(reasons))
            return out

        # ---- task cost ----
        a = self.a
        soft = math.radians(a.soft_margin_deg)
        short = np.clip(soft - room, 0.0, None)
        c_limit = float(np.sum(self.joint_w * (short / soft) ** 2))

        descent = [q for label, q in path if label in ("tilted", "descent", "target")]
        sig = min(float(np.linalg.svd(self.kin.jacobian(q), compute_uv=False)[-1])
                  for q in descent)
        c_manip = a.sigma_ref / max(sig, 1e-6)

        travel = np.abs(np.diff(np.vstack([self.q_start, Q]), axis=0)).sum(axis=0)
        c_travel = float(travel.max())

        c_tilt = ((tilt - math.radians(a.tilt_deg)) / math.radians(10.0)) ** 2

        terms = {"limit": a.w_limit * c_limit, "manip": a.w_manip * c_manip,
                 "travel": a.w_travel * c_travel, "tilt": a.w_tilt * c_tilt}
        out.update(feasible=True, cost=sum(terms.values()), terms=terms,
                   info={"min_room_deg": {f"joint_{j + 1}": math.degrees(room[j])
                                          for j in np.flatnonzero(bounded)},
                         "sigma_min": sig,
                         "max_travel_deg": math.degrees(c_travel),
                         "max_jump_deg": math.degrees(max_jump),
                         "table_clear_mm": min_table * 1000.0})
        return out


# --------------------------------------------------------------------------
# Multi-start constrained CEM
# --------------------------------------------------------------------------
def cem(problem, mean0, std0, lo, hi, rng, n_samples, n_elite, iters,
        alpha=0.7, std_min=None):
    """One CEM run. x[0] is an angle (wrapped); x[1:] are clipped to [lo, hi]."""
    mean, std = np.array(mean0, float), np.array(std0, float)
    std_min = np.zeros_like(std) if std_min is None else std_min
    best = None
    history = []
    for it in range(iters):
        X = mean + std * rng.standard_normal((n_samples, len(mean)))
        X[:, 0] = [wrap(v) for v in X[:, 0]]
        X[:, 1:] = np.clip(X[:, 1:], lo[1:], hi[1:])
        if it == 0:
            X[0] = mean                     # always evaluate the start point
        results = [problem.evaluate(x) for x in X]
        costs = np.array([r["cost"] for r in results])
        order = np.argsort(costs)
        elite = X[order[:n_elite]]
        if best is None or costs[order[0]] < best["cost"]:
            best = results[order[0]]
        # Refit. The azimuth mean is a circular mean around the current mean.
        d0 = np.array([wrap(v - mean[0]) for v in elite[:, 0]])
        new_mean = np.concatenate([[wrap(mean[0] + d0.mean())],
                                   elite[:, 1:].mean(axis=0)])
        new_std = np.concatenate([[d0.std()], elite[:, 1:].std(axis=0)])
        mean = np.concatenate([[wrap(mean[0] + alpha * wrap(new_mean[0] - mean[0]))],
                               alpha * new_mean[1:] + (1 - alpha) * mean[1:]])
        std = np.maximum(alpha * new_std + (1 - alpha) * std, std_min)
        history.append({"iter": it, "best": float(costs[order[0]]),
                        "n_feasible": int(sum(r["feasible"] for r in results))})
    return best, history


def multi_start(problem, args, rng):
    K = args.starts
    lo = np.array([-math.pi, math.radians(args.tilt_min_deg), math.radians(-args.elbow_range_deg)])
    hi = np.array([math.pi, math.radians(args.tilt_max_deg), math.radians(args.elbow_range_deg)])
    std0 = np.array([math.pi / K, math.radians(5.0), math.radians(args.elbow_range_deg) / 2])
    std_min = np.array([math.radians(1.0), math.radians(0.5), math.radians(2.0)])
    runs = []
    for k in range(K):
        az0 = wrap(-math.pi + (k + 0.5) * 2 * math.pi / K)
        mean0 = [az0, math.radians(args.tilt_deg), 0.0]
        t0 = time.monotonic()
        best, hist = cem(problem, mean0, std0, lo, hi, rng, args.samples,
                         args.elite, args.iters, std_min=std_min)
        runs.append({"start": k, "start_azimuth_deg": math.degrees(az0),
                     "best": best, "history": hist,
                     "seconds": time.monotonic() - t0})
        b = best
        tag = (f"cost {b['cost']:7.3f}" if b["feasible"]
               else f"INFEASIBLE ({b.get('reason', '?')})")
        print(f"  start {k}: az0 {math.degrees(az0):+7.1f} -> az {math.degrees(b['x'][0]):+7.1f}"
              f"  tilt {math.degrees(b['x'][1]):5.1f}  elbow {math.degrees(b['x'][2]):+6.1f}"
              f"  {tag}  [{runs[-1]['seconds']:.1f} s]", flush=True)
    return runs


def distinct_modes(runs, az_tol_deg=15.0, q_tol=0.3):
    """Feasible run bests, cheapest first, with duplicates of a mode dropped."""
    feas = sorted((r["best"] for r in runs if r["best"]["feasible"]),
                  key=lambda b: b["cost"])
    modes = []
    for b in feas:
        qb = np.array(b["path"][-1][1])
        dup = False
        for m in modes:
            qm = np.array(m["path"][-1][1])
            d = np.abs(qb - qm)
            d[CONTINUOUS] = np.abs([wrap(v) for v in (qb - qm)[CONTINUOUS]])
            if (abs(math.degrees(wrap(b["x"][0] - m["x"][0]))) < az_tol_deg
                    and d.max() < q_tol):
                dup = True
                break
        if not dup:
            modes.append(b)
    return modes


def describe(b):
    az, tilt, elbow = (math.degrees(v) for v in b["x"])
    t, i = b["terms"], b["info"]
    room = "  ".join(f"{k} {v:5.1f}" for k, v in i["min_room_deg"].items())
    return (f"az {az:+7.1f}  tilt {tilt:5.1f}  elbow {elbow:+6.1f} | cost {b['cost']:.3f} "
            f"= limit {t['limit']:.3f} + manip {t['manip']:.3f} + travel {t['travel']:.3f} "
            f"+ tilt {t['tilt']:.3f}\n"
            f"      limit room deg: {room} | sigma_min {i['sigma_min']:.3f} | "
            f"max travel {i['max_travel_deg']:.0f} deg | max jump {i['max_jump_deg']:.1f} deg | "
            f"table clear {i['table_clear_mm']:.0f} mm")


# --------------------------------------------------------------------------
# Self-test: IK round trip + geometry agreement
# --------------------------------------------------------------------------
def selftest(kin, args):
    fails = []

    def check(name, cond, detail=""):
        print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if len(str(detail)) else ""))
        if not cond:
            fails.append(name)

    rng = np.random.default_rng(0)
    q0 = np.array(DEFAULT_START)
    p0, quat0 = kin.fk(q0)
    check("default start is tool-down near (0.285, -0.007, 0.236)",
          np.linalg.norm(p0 - [0.285, -0.007, 0.236]) < 0.005, np.round(p0, 3))
    ok = 0
    for _ in range(20):
        qt = q0 + rng.uniform(-0.4, 0.4, kin.n)
        qt = np.clip(qt, -HARD_LIMITS + 0.1, HARD_LIMITS - 0.1)
        p, quat = kin.fk(qt)
        q = kin.ik(p, quat, q0)
        if q is not None:
            pe, qe = kin.fk(q)
            ok += (np.linalg.norm(pe - p) < 1e-3
                   and np.linalg.norm(orientation_error(quat, qe)) < 0.01)
    check("IK round-trips random nearby poses", ok >= 18, f"{ok}/20")
    hover, axis, D = insertion_axis_geometry(0.26, 0.0, 0.236, 0.026,
                                             math.radians(45), math.radians(-90))
    check("geometry matches insertion.py (az -90 hover on +Y)",
          abs(hover[0] - 0.26) < 1e-9 and abs(hover[1] - 0.21) < 1e-9, np.round(hover, 3))
    qt = tilt_quaternion(np.array([0, 0, 0, 1.0]), 0.0, math.radians(45))
    x = qt[:3]; w = qt[3]
    down = np.array([0, 0, -1.0])
    rot = down + 2 * w * np.cross(x, down) + 2 * np.cross(x, np.cross(x, down))
    check("azimuth 0 leans the tool axis toward +X", rot[0] > 0.6, np.round(rot, 3))
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-x", type=float, default=400.0, help="container centre X, mm (base frame)")
    p.add_argument("--target-y", type=float, default=0.0, help="container centre Y, mm")
    p.add_argument("--depth-mm", type=float, default=30.0, help="tip depth below rim (keep <= 39 at 45 deg)")
    p.add_argument("--tilt-deg", type=float, default=45.0, help="nominal tilt")
    p.add_argument("--tilt-min-deg", type=float, default=30.0)
    p.add_argument("--tilt-max-deg", type=float, default=50.0)
    p.add_argument("--elbow-range-deg", type=float, default=90.0,
                   help="joint_3 null-space target range around the start posture")
    p.add_argument("--start-q", type=float, nargs=7, default=DEFAULT_START,
                   help="start joint vector, rad (default: MoveIt/cuRobo home)")
    p.add_argument("--starts", type=int, default=6, help="number of CEM starts (K)")
    p.add_argument("--samples", type=int, default=32, help="samples per iteration")
    p.add_argument("--elite", type=int, default=6)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pivot-samples", type=int, default=6)
    p.add_argument("--descent-samples", type=int, default=6)
    p.add_argument("--limit-margin-deg", type=float, default=5.7,
                   help="hard: bounded joints keep this from their stops (insertion.py default)")
    p.add_argument("--soft-margin-deg", type=float, default=35.0,
                   help="limit cost starts inside this much room")
    p.add_argument("--max-jump", type=float, default=0.35, help="rad between path samples")
    p.add_argument("--joint-weights", type=float, nargs=7,
                   default=[0, 1, 0, 1, 0, 3, 0], help="limit-cost weight per joint")
    p.add_argument("--sigma-ref", type=float, default=0.05)
    p.add_argument("--w-limit", type=float, default=1.0)
    p.add_argument("--w-manip", type=float, default=1.0)
    p.add_argument("--w-travel", type=float, default=0.3)
    p.add_argument("--w-tilt", type=float, default=0.2)
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--out", help="write the chosen plan + alternatives as JSON")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    kin = ArmKinematics(args.urdf)
    if args.selftest:
        sys.exit(selftest(kin, args))

    prob = InsertionProblem(kin, args)
    print(f"Container ({args.target_x:.0f}, {args.target_y:.0f}) mm, rim z "
          f"{prob.top * 1000:.0f} mm, tip target z {prob.target_z * 1000:.0f} mm, "
          f"ready z {prob.ready_z * 1000:.0f} mm")
    print(f"CEM: {args.starts} starts x {args.iters} iters x {args.samples} samples")
    t0 = time.monotonic()
    runs = multi_start(prob, args, np.random.default_rng(args.seed))
    modes = distinct_modes(runs)
    print(f"\n{len(modes)} distinct feasible mode(s) in {time.monotonic() - t0:.1f} s")
    for i, m in enumerate(modes):
        print(("BEST " if i == 0 else f"alt {i}") + " " + describe(m))
    if not modes:
        reasons = sorted({r["best"].get("reason", "?") for r in runs})
        print("No feasible approach. Closest failures: " + "; ".join(reasons))

    if args.out:
        def plan(b):
            return {"azimuth_deg": math.degrees(b["x"][0]),
                    "tilt_deg": math.degrees(b["x"][1]),
                    "elbow_deg": math.degrees(b["x"][2]),
                    "cost": b["cost"], "terms": b["terms"], "info": b["info"],
                    "waypoints": [{"label": lbl, "q": q} for lbl, q in b["path"]]}
        doc = {"container_mm": [args.target_x, args.target_y], "depth_mm": args.depth_mm,
               "start_q": list(args.start_q),
               "best": plan(modes[0]) if modes else None,
               "alternatives": [plan(m) for m in modes[1:]],
               "runs": [{"start_azimuth_deg": r["start_azimuth_deg"],
                         "history": r["history"], "seconds": r["seconds"],
                         "feasible": r["best"]["feasible"],
                         "reason": r["best"].get("reason")} for r in runs]}
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {args.out}")
    sys.exit(0 if modes else 2)


if __name__ == "__main__":
    main()
