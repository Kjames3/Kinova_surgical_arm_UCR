#!/usr/bin/env python3
"""Execution gate for a cuRobo Phase 0 (free-space approach) trajectory.

ROS-free on purpose: every check here runs offline against archived sidecar
responses (see test_curobo_approach_execution.py). insertion.py calls
``validate_for_execution`` immediately before handing the trajectory to
``_execute_fjt`` (see ``execute_approach``); any failure raises ``ExecutionRejected`` and NO motion is sent.

Scope is deliberately narrow (M2): only the validated upright approach a
preview already produces. Pivot, descent and recovery stay on Pilz / disabled.

The checks are independent of the sidecar's own validation. The sidecar reports
``validated`` against cuRobo's limits; this gate re-checks the raw waypoints
against the limits insertion.py itself enforces on the arm, so a sidecar bug or
a config drift between the two cannot slip a trajectory through.
"""
import math

ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))
CONTINUOUS = frozenset({"joint_1", "joint_3", "joint_5", "joint_7"})

# Hard caps for the first hardware runs. Tighten, don't loosen, from the CLI.
MAX_VEL_SCALE = 0.15          # mirrors check_preview_parameters
START_TOL_RAD = 0.01          # first waypoint vs live arm
MAX_STEP_RAD = 0.25           # largest per-waypoint jump on any joint
MAX_DT_S = 0.5                # largest gap between waypoints
MAX_DURATION_S = 90.0
LIMIT_SLACK = 1.02            # numeric slack on velocity/acceleration checks
END_REST_VEL = 1e-3


class ExecutionRejected(ValueError):
    pass


def _wrap(d):
    return math.remainder(d, 2.0 * math.pi)


def validate_for_execution(response, live_joints, *, vel_scale,
                           max_joint_vel, max_joint_acc, bounded_limits,
                           max_tilt_deg):
    """Return a summary dict, or raise ExecutionRejected.

    response      sidecar response dict (already passed validate_upright_response)
    live_joints   {name: rad} sampled just before execution
    vel_scale     the transit scaling the plan was requested with
    max_joint_vel / max_joint_acc   insertion.py's per-joint limits (rad/s, rad/s^2)
    bounded_limits  {joint: |q| limit} for the non-continuous joints
    """
    def bad(msg):
        raise ExecutionRejected(f"cuRobo approach rejected for execution: {msg}")

    meta = response.get("meta") or {}
    if not response.get("success") or meta.get("backend") != "curobo" or not meta.get("validated"):
        bad("response is not a successful, validated cuRobo plan")
    if meta.get("upright_policy_version") != 1:
        bad("response predates upright-path validation")
    tilt = meta.get("approach_max_tilt_deg", float("nan"))
    if not (math.isfinite(tilt) and tilt <= max_tilt_deg):
        bad(f"approach tilt {tilt} deg exceeds {max_tilt_deg} deg")
    if not (0.0 < vel_scale <= MAX_VEL_SCALE):
        bad(f"vel_scale {vel_scale} outside (0, {MAX_VEL_SCALE}]")

    names = list(response.get("joint_names") or [])
    if sorted(names) != sorted(ARM_JOINTS) or len(set(names)) != 7:
        bad(f"joint names {names} are not exactly the 7 arm joints")
    missing = [n for n in names if n not in live_joints]
    if missing:
        bad(f"no live state for {missing}")

    pts = response.get("points") or []
    if len(pts) < 2:
        bad("fewer than two waypoints")

    t_prev = -1.0
    q_prev = None
    peak_v = {n: 0.0 for n in names}
    peak_a = {n: 0.0 for n in names}
    for k, p in enumerate(pts):
        q, v, a = p.get("positions"), p.get("velocities"), p.get("accelerations")
        t = p.get("time_from_start")
        if not (q and v and a) or not (len(q) == len(v) == len(a) == 7):
            bad(f"waypoint {k} lacks 7 positions/velocities/accelerations")
        vals = list(q) + list(v) + list(a) + [t]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vals):
            bad(f"waypoint {k} has a non-finite value")
        if k == 0 and t != 0.0:
            bad("first waypoint must be at t=0")
        if k > 0 and not (t > t_prev):
            bad(f"time not strictly increasing at waypoint {k}")
        if k > 0 and t - t_prev > MAX_DT_S:
            bad(f"{t - t_prev:.3f} s gap at waypoint {k}")
        for j, n in enumerate(names):
            if n in bounded_limits and abs(q[j]) > bounded_limits[n]:
                bad(f"{n}={q[j]:.3f} beyond +/-{bounded_limits[n]} at waypoint {k}")
            if q_prev is not None and abs(q[j] - q_prev[j]) > MAX_STEP_RAD:
                bad(f"{n} jumps {q[j] - q_prev[j]:.3f} rad at waypoint {k} (2pi seam?)")
            vmax = vel_scale * max_joint_vel.get(n, 0.8)
            if abs(v[j]) > vmax * LIMIT_SLACK:
                bad(f"{n} velocity {v[j]:.3f} > {vmax:.3f} rad/s at waypoint {k}")
            amax = max_joint_acc.get(n, 0.4)
            if abs(a[j]) > amax * LIMIT_SLACK:
                bad(f"{n} acceleration {a[j]:.3f} > {amax:.3f} rad/s^2 at waypoint {k}")
            peak_v[n] = max(peak_v[n], abs(v[j]))
            peak_a[n] = max(peak_a[n], abs(a[j]))
        t_prev, q_prev = t, q

    if t_prev > MAX_DURATION_S:
        bad(f"duration {t_prev:.1f} s exceeds {MAX_DURATION_S} s")
    for end in (pts[0], pts[-1]):
        if max(abs(x) for x in end["velocities"]) > END_REST_VEL:
            bad("trajectory does not start and end at rest")

    start_err = {}
    for j, n in enumerate(names):
        d = pts[0]["positions"][j] - live_joints[n]
        start_err[n] = abs(_wrap(d) if n in CONTINUOUS else d)
    worst = max(start_err, key=start_err.get)
    if start_err[worst] > START_TOL_RAD:
        bad(f"first waypoint is {start_err[worst]:.4f} rad from live {worst}")

    return {
        "n_points": len(pts),
        "duration_s": t_prev,
        "approach_max_tilt_deg": tilt,
        "max_start_error_rad": start_err[worst],
        "peak_velocity_ratio": max(peak_v[n] / (vel_scale * max_joint_vel.get(n, 0.8)) for n in names),
        "peak_acceleration_ratio": max(peak_a[n] / max_joint_acc.get(n, 0.4) for n in names),
    }


def execute_approach(node, response, traj, start, params, out, check_unchanged):
    """Send the validated approach. Every gate re-runs AFTER the operator says yes."""
    import json
    from insertion_geometry import GEN3_BOUNDED_LIMITS
    vel_scale = (params['transit_velocity_scaling'] if params['transit_velocity_scaling'] > 0
                 else params['max_velocity_scaling'])
    kw = dict(vel_scale=vel_scale, max_joint_vel=node.max_joint_vel, max_joint_acc=node.max_joint_acc,
              bounded_limits=GEN3_BOUNDED_LIMITS, max_tilt_deg=params['curobo_max_tilt_deg'])
    summary = validate_for_execution(response, start, **kw)
    node.get_logger().info('cuRobo approach execution gate PASS (planned start): '+json.dumps(summary))
    status = {'motion_sent': False, 'gate_planned_start': summary}
    def write_status():
        (out/'execution_status.json').write_text(json.dumps(status, indent=2)+'\n')
    write_status()
    label = 'Phase0_curobo_approach'
    if not node._wait_for_user(label):
        status['declined'] = True; write_status()
        return response
    try:
        check_unchanged()
        _, live = node._preview_joint_sample
        status['gate_live'] = validate_for_execution(response, live, **kw)
    except (ValueError, ExecutionRejected) as e:
        status['rejected_after_confirm'] = str(e); write_status()
        raise
    status['motion_sent'] = True; write_status()
    ok = node._execute_fjt(traj, label, timeout=summary['duration_s'] + 30.)
    status['fjt_ok'] = bool(ok); write_status()
    if not ok:
        # No recovery or fallback motion: the operator decides what happens next.
        raise RuntimeError(f'{label} failed; arm left where it stopped (no recovery motion)')
    node.get_logger().info(f'{label} complete.' + ('' if params.get('curobo_hybrid_insertion')
                           else ' Pivot/descent are not cuRobo-qualified; stopping here.'))
    return response
