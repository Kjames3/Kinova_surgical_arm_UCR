#!/usr/bin/env python3
"""Shared, ROS-free insertion geometry: quaternions, tilt, approach axis, wall check.

One copy of what insertion.py and insert_to_container.py used to each carry
(they were byte-identical), so the planner, the ROS nodes and offline tools
all compute the same poses. Pure Python (math only): importable from system
python, the kortex_impedance venv, and offline tests alike.

Conventions: quaternions are (x, y, z, w) tuples; lengths in metres; angles
in radians unless the name says _deg. Frames are the robot base / world
(identical here). Container-scene helpers (STL, rim height) live in
container_geometry.py.
"""
import math

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

# Container physical dimensions (metres)
CONTAINER_WIDTH_M  = 0.090   # 90 mm
CONTAINER_HEIGHT_M = 0.086   # 86 mm
CONTAINER_WALL_M   = 0.003   # 3 mm safety margin from inner wall
CONTAINER_INNER_R  = (CONTAINER_WIDTH_M / 2.0) - CONTAINER_WALL_M  # 42 mm

MAX_TILT_DEG = 25.0   # hard limit — beyond this hits the wall at shallow depths

TABLE_Z = -0.030      # table surface in the base frame (insertion.py table_z default)

# Gen3 7-DOF position limits of the bounded joints [rad]; 1/3/5/7 are continuous.
GEN3_BOUNDED_LIMITS = {"joint_2": 2.24, "joint_4": 2.57, "joint_6": 2.09}


# ---------------------------------------------------------------------------
# Quaternions (x, y, z, w)
# ---------------------------------------------------------------------------

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

def quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])

def quat_normalize(q):
    n = math.sqrt(sum(v*v for v in q))
    return tuple(v/n for v in q)

def quat_slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    dot = sum(a*b for a, b in zip(q0, q1))
    # Ensure shortest path
    if dot < 0.0:
        q1 = tuple(-v for v in q1)
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        # Linear interpolation for nearly identical quaternions
        result = tuple(a + t*(b-a) for a, b in zip(q0, q1))
        return quat_normalize(result)
    theta_0 = math.acos(dot)
    theta   = theta_0 * t
    sin_theta   = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize(tuple(s0*a + s1*b for a, b in zip(q0, q1)))

def quat_from_axis_angle(axis_xyz, angle_rad):
    """Build unit quaternion from rotation axis and angle."""
    ax, ay, az = axis_xyz
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    ax, ay, az = ax/n, ay/n, az/n
    s = math.sin(angle_rad / 2.0)
    c = math.cos(angle_rad / 2.0)
    return (ax*s, ay*s, az*s, c)

def orientation_error(q_des, q_cur):
    """Rotation vector (base frame, rad) taking q_cur to q_des, shortest way."""
    e = quat_multiply(q_des, quat_conjugate(q_cur))
    if e[3] < 0.0:
        e = tuple(-v for v in e)
    return (2.0 * e[0], 2.0 * e[1], 2.0 * e[2])

def wrap_angle(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Tilted insertion
# ---------------------------------------------------------------------------

def tilt_quaternion(q_vertical, azimuth_rad, tilt_rad):
    """
    Compute the new EE quaternion after tilting the tool axis by tilt_rad
    toward azimuth_rad, starting from q_vertical (tool pointing straight down).

    The rotation is composed as two successive rotations applied to q_vertical:
      1. Rotate about world Z by azimuth (point the tilt direction)
      2. Rotate about the new X axis by tilt (lean the tool)

    This keeps the tip fixed while the wrist reorients.
    """
    # Step 1: rotation about world Z by azimuth
    q_az = quat_from_axis_angle((0, 0, 1), azimuth_rad)
    # Step 2: tilt the (downward) tool axis toward the azimuth direction.
    # The tilt axis is horizontal, 90° clockwise from the azimuth heading, so
    # that azimuth=0 leans the tool toward world +X (matching the geometry in
    # _run_impl, which places hover on the −X side and descends toward +X).
    # The previous (-sin, +cos) axis leaned toward −X — the opposite way —
    # which put the wrist on the wrong side and made the EE pose unreachable.
    tilt_ax = (math.sin(azimuth_rad), -math.cos(azimuth_rad), 0.0)
    q_tilt  = quat_from_axis_angle(tilt_ax, tilt_rad)
    # Compose: q_new = q_tilt * q_az * q_vertical
    q_new = quat_multiply(q_tilt, quat_multiply(q_az, q_vertical))
    return quat_normalize(q_new)


def insertion_axis_geometry(cont_x, cont_y, ready_z, target_z, tilt_rad, azimuth_rad):
    """Tool axis, descent length and hover point for one approach azimuth.

    The tip ends at (cont_x, cont_y, target_z).  It starts at the hover point,
    which lies back along the tilted tool axis at height ready_z.
    Returns (hover_xyz, tool_axis, descent_m).
    """
    axis = (math.sin(tilt_rad) * math.cos(azimuth_rad),
            math.sin(tilt_rad) * math.sin(azimuth_rad),
            -math.cos(tilt_rad))
    cos_t = math.cos(tilt_rad)
    descent = (ready_z - target_z) / cos_t if cos_t > 1e-3 else 0.0
    hover = (cont_x - axis[0] * descent, cont_y - axis[1] * descent, ready_z)
    return hover, axis, descent


def compute_tilt_geometry(hover_xyz, target_xyz):
    """
    Given the hover position (tip directly above container centre) and
    the target point inside the container, compute:

      azimuth_rad  : rotation about world Z to face the target (0 = +X direction)
      tilt_rad     : angle from vertical toward target (0 = straight down)
      descent_m    : straight-line distance from hover to target
      tool_axis    : unit vector pointing from hover to target (world frame)

    Parameters
    ----------
    hover_xyz  : (x, y, z) world position of tip at hover height
    target_xyz : (x, y, z) world position of target inside container

    Returns dict with keys: azimuth_rad, tilt_rad, descent_m, tool_axis,
                             dx, dy, dz, horizontal_m
    """
    dx = target_xyz[0] - hover_xyz[0]
    dy = target_xyz[1] - hover_xyz[1]
    dz = target_xyz[2] - hover_xyz[2]   # always negative (going down)

    horizontal_m = math.sqrt(dx*dx + dy*dy)
    descent_m    = math.sqrt(dx*dx + dy*dy + dz*dz)

    if descent_m < 1e-4:
        return dict(azimuth_rad=0.0, tilt_rad=0.0, descent_m=0.0,
                    tool_axis=(0.0, 0.0, -1.0),
                    dx=dx, dy=dy, dz=dz, horizontal_m=0.0)

    # Azimuth: direction of horizontal offset in XY plane
    azimuth_rad = math.atan2(dy, dx)

    # Tilt: angle from world -Z toward target
    # dz is negative, so |dz| is the vertical drop
    tilt_rad = math.atan2(horizontal_m, abs(dz))

    # Unit vector pointing from hover to target
    tool_axis = (dx/descent_m, dy/descent_m, dz/descent_m)

    return dict(
        azimuth_rad  = azimuth_rad,
        tilt_rad     = tilt_rad,
        descent_m    = descent_m,
        tool_axis    = tool_axis,
        dx=dx, dy=dy, dz=dz,
        horizontal_m = horizontal_m,
    )


def check_wall_clearance(offset_x_m, offset_y_m, depth_m,
                          tilt_rad, tip_radius_m=0.003,
                          max_tilt_deg=MAX_TILT_DEG):
    """
    Check that the tilted assembly does not hit the container wall.

    At depth d below hover_z, the tip is offset (dx, dy) from centre.
    The assembly body at depth d has an additional lateral offset from tilt:
      lateral_at_depth = d * tan(tilt_rad)
    This must not exceed inner_radius - tip_radius.

    Returns (ok, message).
    """
    r_target = math.sqrt(offset_x_m**2 + offset_y_m**2)
    if r_target > CONTAINER_INNER_R:
        return False, (f"Target is {r_target*1000:.1f} mm from centre — "
                       f"outside inner radius {CONTAINER_INNER_R*1000:.1f} mm")

    if tilt_rad > math.radians(max_tilt_deg):
        return False, (f"Tilt {math.degrees(tilt_rad):.1f}° > "
                       f"max {max_tilt_deg}°")

    # At full depth, the assembly body's lateral extent
    lateral = depth_m * math.tan(tilt_rad)
    effective_r = r_target + lateral + tip_radius_m
    if effective_r > CONTAINER_INNER_R:
        return False, (
            f"Assembly hits wall at depth {depth_m*1000:.0f} mm: "
            f"effective radius {effective_r*1000:.1f} mm > "
            f"inner radius {CONTAINER_INNER_R*1000:.1f} mm.\n"
            f"  Reduce offset or depth.")

    return True, "OK"


# ---------------------------------------------------------------------------
# Joint space
# ---------------------------------------------------------------------------

def unwrap_to_seed(solution, seed, joint_names, continuous_joints):
    """Shift each continuous joint by 2*pi multiples to sit nearest the seed."""
    out = []
    for name, value, ref in zip(joint_names, solution, seed):
        if name in continuous_joints:
            value += 2.0 * math.pi * round((ref - value) / (2.0 * math.pi))
        out.append(value)
    return out
