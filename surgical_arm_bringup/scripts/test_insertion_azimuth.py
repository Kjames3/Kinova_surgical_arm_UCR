#!/usr/bin/env python3
"""Offline test for the approach-direction search in insertion.py
(azimuth_mode:=search): the insertion-axis geometry, 2*pi unwrapping and the
least-joint-motion ranking, with hand-built IK chains whose order is known.

  source /opt/ros/humble/setup.bash && ./test_insertion_azimuth.py
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import insertion as ins

fails = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond: fails.append(name)

J = [f"joint_{i}" for i in range(1, 8)]
CONT = {"joint_1", "joint_3", "joint_5", "joint_7"}

# geometry: 45 deg, azimuth -90 -> hover sits on the +Y side, 180 mm + depth above the target
hover, axis, D = ins.insertion_axis_geometry(0.26, 0.0, 0.236, 0.026, math.radians(45), math.radians(-90))
check("axis is a unit vector tilted 45 deg", abs(math.hypot(*axis[:2]) - math.sin(math.radians(45))) < 1e-9
      and abs(axis[2] + math.cos(math.radians(45))) < 1e-9)
check("hover + axis * descent lands on the target",
      max(abs(h + a * D - t) for h, a, t in zip(hover, axis, (0.26, 0.0, 0.026))) < 1e-9, f"D={D*1000:.1f} mm")
check("azimuth -90 puts the hover on +Y", abs(hover[0] - 0.26) < 1e-9 and abs(hover[1] - 0.21) < 1e-9, [round(v, 3) for v in hover])
h0, _, D0 = ins.insertion_axis_geometry(0.26, 0.0, 0.236, 0.026, 0.0, 1.0)
check("zero tilt is a plain vertical descent", abs(D0 - 0.21) < 1e-9 and abs(h0[0] - 0.26) < 1e-9 and abs(h0[1]) < 1e-9)

# unwrap: continuous joints snap to the seed's turn, bounded joints are left alone
q = ins.unwrap_to_seed([3.10, 0.5, -3.0, 1.0, 6.4, 0.2, 0.0], [-3.10, 0.5, 3.0, 1.0, 0.1, 0.2, 0.0], J, CONT)
check("continuous joints unwrap to the seed", abs(q[0] - (3.10 - 2*math.pi)) < 1e-9 and abs(q[2] - (-3.0 + 2*math.pi)) < 1e-9
      and abs(q[4] - (6.4 - 2*math.pi)) < 1e-9)
check("bounded joints are not unwrapped", q[1] == 0.5 and q[3] == 1.0 and q[5] == 0.2)

start = [0.0, 0.3, 0.0, 1.5, 0.0, 0.8, 0.0]
def chain(d1=0.0, d2=0.0, d6=0.0, wrap=0.0):
    """Four waypoints, each moving joint_1 by d1, joint_2 by d2, joint_6 by d6 from the last."""
    out, q = [], list(start)
    for _ in range(4):
        q = list(q); q[0] += d1; q[1] += d2; q[5] += d6
        w = list(q); w[2] += wrap            # same pose, joint_3 reported a turn away
        out.append(w)
    return out
solved = {
    -90.0: chain(d1=0.30),                    # 1.2 rad on one joint
    -60.0: chain(d1=0.10, d2=0.05),           # 0.4 rad max: the easy one
      0.0: chain(d1=0.10, d2=0.10),           # same max as -60, more in total
     30.0: chain(d1=0.05, wrap=2 * math.pi),  # tiny motion once the 2*pi is unwrapped
     60.0: chain(d6=0.32),                    # joint_6 ends at 2.08 rad: inside its 2.09 limit, but only just
     90.0: None,                              # no IK
}
ranked = ins.rank_azimuth_candidates(start, solved, J, CONT)
order = [c["azimuth_deg"] for c in ranked]
check("unreachable direction is dropped", 90.0 not in order)
check("direction at a joint limit is dropped", 60.0 not in order, f"margin floor 0.10 rad")
check("a 2*pi alias is not counted as motion", order[0] == 30.0, f"max {ranked[0]['max_travel']:.3f} rad")
check("least largest-joint-move wins", order[:3] == [30.0, -60.0, 0.0], order)
check("total travel breaks the tie", ranked[1]["max_travel"] == ranked[2]["max_travel"]
      and ranked[1]["total_travel"] < ranked[2]["total_travel"])
check("the big single-joint move ranks last", order[-1] == -90.0)
check("no candidates -> empty list", ins.rank_azimuth_candidates(start, {0.0: None}, J, CONT) == [])
loose = ins.rank_azimuth_candidates(start, solved, J, CONT, limit_margin=0.0)
check("limit margin is configurable", 60.0 in [c["azimuth_deg"] for c in loose])

# marker-centre filtering (target_source:=markers)
c = ins.robust_centre([(0.300, 0.010), (0.301, 0.011), (0.299, 0.009), (0.300, 0.010), (0.302, 0.010)], 0.005)
check("median of steady centre samples", c is not None and abs(c[0] - 0.300) < 1e-9 and abs(c[1] - 0.010) < 1e-9, c)
check("spread is reported", c is not None and abs(c[2] - 0.002) < 1e-9)
check("one outlier does not move the median but is caught by the spread limit",
      ins.robust_centre([(0.300, 0.010)] * 4 + [(0.360, 0.010)], 0.005) is None)
check("no samples -> None", ins.robust_centre([], 0.005) is None)
check("even sample count averages the middle pair",
      abs(ins.robust_centre([(0.0, 0.0), (0.002, 0.0)], 0.005)[0] - 0.001) < 1e-12)

print("\n%d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
