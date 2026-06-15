import math
from moveit_msgs.msg import RobotTrajectory, RobotState, DisplayTrajectory

# Safety limits
MAX_JOINT_VEL_RAD_S  = 0.8    # firmware limit ~1.22 rad/s; 0.8 gives 35% margin
MAX_JOINT_ACC_RAD_S2 = 0.4    # conservative; raise to 0.6 if moves feel too sluggish
MAX_TARGET_DIST_M = 0.60
MAX_WAYPOINT_JUMP_RAD = 1.5   # ~86° per step; anything larger is a planning error
MIN_TRAJ_DURATION_S = 0.5
SUSPICIOUS_STEP_RAD = 1.0

# Human-readable MoveIt error code mapping
MOVEIT_ERROR_CODES = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -4: "CONTROL_FAILED",
    -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE",
    -31: "NO_IK_SOLUTION",
}

def get_error_string(error_code_val: int) -> str:
    return MOVEIT_ERROR_CODES.get(error_code_val, f"UNKNOWN_ERROR_CODE_{error_code_val}")

def quat_multiply(q1, q2):
    """Hamilton product of two (x,y,z,w) tuples."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )

def rotate_vector_by_quat(v, q):
    """Rotate vector v=(x,y,z) by unit quaternion q=(x,y,z,w)."""
    qv  = (v[0], v[1], v[2], 0.0)
    qc  = (-q[0], -q[1], -q[2], q[3])
    res = quat_multiply(quat_multiply(q, qv), qc)
    return res[0], res[1], res[2]

def validate_trajectory(traj: RobotTrajectory, label: str, logger, display_traj_pub=None, real_robot: bool = False) -> bool:
    """
    Sanity-check a planned trajectory before execution.
    """
    pts = traj.joint_trajectory.points
    names = traj.joint_trajectory.joint_names

    # 1. Empty
    if not pts:
        logger.error(f"  [VALIDATE] {label}: trajectory is EMPTY — aborting.")
        return False

    # 2. Duration
    last = pts[-1].time_from_start
    dur  = last.sec + last.nanosec * 1e-9
    if dur < MIN_TRAJ_DURATION_S:
        if len(pts) > 5:
            logger.error(
                f"  [VALIDATE] {label}: trajectory duration {dur:.2f} s < "
                f"{MIN_TRAJ_DURATION_S} s minimum with {len(pts)} waypoints.\n"
                f"  This usually means velocity scaling was ignored and the arm\n"
                f"  would move at 100% speed.  Aborting for safety.")
            return False
        else:
            logger.info(
                f"  [VALIDATE] {label}: short trajectory {dur:.2f} s with "
                f"{len(pts)} pts — arm already near goal, allowing.")

    # 3. Joint jumps between consecutive waypoints
    max_jump = 0.0
    max_jump_info = ""
    for i in range(1, len(pts)):
        for j, name in enumerate(names):
            if j >= len(pts[i].positions) or j >= len(pts[i-1].positions):
                continue
            jump = abs(pts[i].positions[j] - pts[i-1].positions[j])
            if jump > max_jump:
                max_jump = jump
                max_jump_info = f"{name} pt{i-1}→{i}: {math.degrees(jump):.1f}°"
    if max_jump > MAX_WAYPOINT_JUMP_RAD:
        logger.error(
            f"  [VALIDATE] {label}: LARGE JOINT JUMP detected!\n"
            f"  Worst: {max_jump_info} ({math.degrees(max_jump):.1f}° > "
            f"{math.degrees(MAX_WAYPOINT_JUMP_RAD):.0f}° limit)\n"
            f"  This is a planning error (wrong IK solution / joint flip).\n"
            f"  DO NOT execute — it would destroy the end-effector.\n"
            f"  Fix: reduce joint_delta_rad or increase approach_clearance.")
        return False

    # 4. Joint limits
    _JOINT_LIMITS = {
        "joint_1": 2.41, "joint_2": 2.41, "joint_3": math.pi,
        "joint_4": 2.41, "joint_5": 2.41, "joint_6": 2.41, "joint_7": 2.41,
    }
    limit_violated = False
    for pt in pts:
        for j, name in enumerate(names):
            if j >= len(pt.positions):
                continue
            lim = _JOINT_LIMITS.get(name, 2.41)
            pos = pt.positions[j]
            pos_norm = (pos + math.pi) % (2 * math.pi) - math.pi
            if abs(pos_norm) > lim:
                logger.warn(
                    f"  [VALIDATE] {label}: {name} position {math.degrees(pos):.1f}° "
                    f"may exceed limit ±{math.degrees(lim):.0f}°")
                limit_violated = True
                break
        if limit_violated:
            break

    # 5. Velocity peaks
    max_vel = 0.0
    max_vel_joint = ""
    for pt in pts:
        for j, v in enumerate(pt.velocities):
            if abs(v) > max_vel:
                max_vel = abs(v)
                max_vel_joint = names[j] if j < len(names) else f"j{j}"
    if max_vel > MAX_JOINT_VEL_RAD_S:
        logger.warn(
            f"  [VALIDATE] {label}: peak velocity {max_vel:.3f} rad/s on "
            f"{max_vel_joint} exceeds soft limit {MAX_JOINT_VEL_RAD_S} rad/s.\n"
            f"  clamp_traj_limits will fix this before execution.")

    # Summary log
    logger.info(
        f"  [VALIDATE] {label}: OK — {len(pts)} pts, {dur:.1f} s, "
        f"max_jump={math.degrees(max_jump):.1f}°, peak_vel={max_vel:.3f} rad/s")

    # 6. Publish to RViz
    if display_traj_pub:
        try:
            disp = DisplayTrajectory()
            disp.trajectory.append(traj)
            disp.trajectory_start = RobotState()
            display_traj_pub.publish(disp)
        except Exception as exc:
            logger.warn(f"  [DISPLAY] Could not publish preview: {exc}")

    # 7. On real robot
    if real_robot and pts:
        first = {names[j]: round(math.degrees(pts[0].positions[j]), 1)
                 for j in range(min(len(names), len(pts[0].positions)))}
        last_pt = {names[j]: round(math.degrees(pts[-1].positions[j]), 1)
                   for j in range(min(len(names), len(pts[-1].positions)))}
        logger.info(
            f"\n  ── Trajectory preview: {label} ──\n"
            f"  Duration : {dur:.1f} s\n"
            f"  Waypoints: {len(pts)}\n"
            f"  Start (°): {first}\n"
            f"  End   (°): {last_pt}\n"
            f"  ─────────────────────────────────\n"
            f"  [RViz] Ghost trajectory published to /display_planned_path")
        ans = input(f"  Execute '{label}' on real robot? [y/N]: ").strip().lower()
        if ans != "y":
            logger.warn(f"  [VALIDATE] User rejected '{label}' — aborting phase.")
            return False

    return True

def smooth_traj_velocities(traj: RobotTrajectory) -> RobotTrajectory:
    pts = traj.joint_trajectory.points
    n   = len(pts)
    if n < 2:
        return traj
    if not pts[0].positions:
        return traj

    num_joints = len(pts[0].positions)
    times = []
    for pt in pts:
        times.append(pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9)

    vels = [[0.0] * num_joints for _ in range(n)]
    accs = [[0.0] * num_joints for _ in range(n)]

    for j in range(num_joints):
        for i in range(n):
            if i == 0:
                vels[i][j] = 0.0
            elif i == n - 1:
                vels[i][j] = 0.0
            else:
                dt_prev = times[i]   - times[i-1]
                dt_next = times[i+1] - times[i]
                if dt_prev <= 0.0 or dt_next <= 0.0:
                    vels[i][j] = 0.0
                else:
                    v_prev = (pts[i].positions[j]   - pts[i-1].positions[j]) / dt_prev
                    v_next = (pts[i+1].positions[j] - pts[i].positions[j])   / dt_next
                    if v_prev * v_next <= 0.0:
                        vels[i][j] = 0.0
                    else:
                        vels[i][j] = 0.5 * (v_prev + v_next)

        for i in range(n):
            if abs(vels[i][j]) > MAX_JOINT_VEL_RAD_S:
                vels[i][j] = math.copysign(MAX_JOINT_VEL_RAD_S, vels[i][j])

        for i in range(n):
            if i == 0 or i == n - 1:
                accs[i][j] = 0.0
            else:
                dt_prev = times[i]   - times[i-1]
                dt_next = times[i+1] - times[i]
                if dt_prev <= 0.0 or dt_next <= 0.0:
                    accs[i][j] = 0.0
                else:
                    accs[i][j] = (vels[i+1][j] - vels[i-1][j]) / (dt_prev + dt_next)

        for i in range(n):
            if abs(accs[i][j]) > MAX_JOINT_ACC_RAD_S2:
                accs[i][j] = math.copysign(MAX_JOINT_ACC_RAD_S2, accs[i][j])

    for i, pt in enumerate(pts):
        pt.velocities    = list(vels[i])
        pt.accelerations = list(accs[i])

    return traj

def clamp_traj_limits(traj: RobotTrajectory) -> RobotTrajectory:
    pts = traj.joint_trajectory.points
    if len(pts) < 2:
        return traj

    num_joints = len(pts[0].positions) if pts[0].positions else 0
    if num_joints == 0:
        return traj

    times_ns = []
    for pt in pts:
        times_ns.append(
            pt.time_from_start.sec * 1_000_000_000
            + pt.time_from_start.nanosec)

    vel_limit_ns   = MAX_JOINT_VEL_RAD_S
    acc_limit_ns   = MAX_JOINT_ACC_RAD_S2

    def _min_dt_ns(disp: float) -> int:
        if abs(disp) < 1e-9:
            return 0
        return int(abs(disp) / vel_limit_ns * 1e9) + 1

    for i in range(1, len(pts)):
        min_dt = 0
        for j in range(num_joints):
            if j >= len(pts[i].positions) or j >= len(pts[i-1].positions):
                continue
            disp = abs(pts[i].positions[j] - pts[i-1].positions[j])
            min_dt = max(min_dt, _min_dt_ns(disp))
        cur_dt = times_ns[i] - times_ns[i-1]
        if cur_dt < min_dt:
            stretch = min_dt - cur_dt
            for k in range(i, len(pts)):
                times_ns[k] += stretch

    for i in range(len(pts) - 2, -1, -1):
        min_dt = 0
        for j in range(num_joints):
            if j >= len(pts[i+1].positions) or j >= len(pts[i].positions):
                continue
            disp = abs(pts[i+1].positions[j] - pts[i].positions[j])
            min_dt = max(min_dt, _min_dt_ns(disp))
        cur_dt = times_ns[i+1] - times_ns[i]
        if cur_dt < min_dt:
            stretch = min_dt - cur_dt
            for k in range(i + 1):
                times_ns[k] -= stretch

    if times_ns[0] < 0:
        offset = -times_ns[0]
        times_ns = [t + offset for t in times_ns]

    for i, pt in enumerate(pts):
        pt.time_from_start.sec     = times_ns[i] // 1_000_000_000
        pt.time_from_start.nanosec = times_ns[i] %  1_000_000_000

    smooth_traj_velocities(traj)

    return traj
