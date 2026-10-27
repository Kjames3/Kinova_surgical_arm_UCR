"""Table-avoidance safety layer for the Kinova Gen3 low-level torque loop.

Two independent mechanisms, deliberately kept separate:

  1. HOCBF filter (`compute_table_hocbf_rows` + `filter_control_qp`)
     A *constraint*. Projects the commanded joint acceleration onto the set that
     keeps every monitored point above the table. It guarantees non-penetration
     but exerts no restoring force -- at the boundary it only forbids further
     downward acceleration, so a hand pushing the arm into the table feels the
     motion stop rather than feeling pushed back.

  2. Virtual wall (`table_wall_torque`)
     A *force*. An explicit spring-damper repulsion that switches on within a
     standoff band above the barrier. This is what the operator actually feels,
     and it is what makes the arm resist rather than merely stop. It also gives
     the CBF margin to work with, since it starts decelerating the arm early
     instead of at the last instant.

Both act on every monitored link, not just the end effector, so the elbow and
forearm are covered too.

  3. External-torque observer (`MomentumObserver`)
     The HOCBF predicts the arm's acceleration from the torque the controller
     commands. An operator's hand is not in that command, so without an
     estimate of it the filter certifies a motion the arm is not making and
     the push goes straight through -- in MuJoCo, 90 N at the flange drove the
     tool 725 mm through the table with the filter reporting "exact". The
     observer estimates every torque the model does not explain (the push, and
     the joint-4 gravity error) from the arm's momentum, and the filter
     predicts with it. Found 2026-10-26; every validation before that fed the
     push through the filter, which overstated the guarantee.

Agnostic to the robot backend: it consumes kinematics/dynamics state and
returns joint-space quantities.
"""

import numpy as np


def compute_hocbf_rows(h1, h1_dot, Jh, drift, alpha1=10.0, alpha2=10.0):
    """Second-order CBF rows ``A @ ddq <= b`` for any barrier ``h1(q) >= 0``.

    This is the algebra every barrier in this module shares. A barrier on
    configuration alone has relative degree 2 with respect to joint
    acceleration, so one differentiation is not enough to expose ``ddq``:

        h2      = h1_dot + alpha1 * h1
        enforce   h2_dot + alpha2 * h2 >= 0

    with ``h1_ddot = Jh @ ddq + drift``. Substituting and rearranging:

        A = -Jh
        b =  drift + alpha1 * h1_dot + alpha2 * h2

    Callers differ only in how they build ``Jh`` and ``drift``. For the table
    those are row 2 of the Jacobian and the z-component of ``dJ @ dq``; for a
    distance barrier the normal rotates as the arm moves, so ``Jh`` is
    ``n^T J_v`` and ``drift`` picks up a curvature term as well. Getting
    ``drift`` wrong does not break the constraint's *form*, which is what makes
    it a dangerous place for an error -- the QP still solves, the arm still
    moves, and the guarantee is quietly only approximate.

    Args:
        h1 (array_like): (m,) barrier value; positive is safe.
        h1_dot (array_like): (m,) its time derivative, ``Jh @ dq``.
        Jh (array_like): (m, n) gradient of h1 with respect to q.
        drift (array_like): (m,) the part of ``h1_ddot`` independent of ``ddq``.
        alpha1, alpha2 (float): HOCBF gains, both > 0. Larger = tolerates a
            faster approach and intervenes later and harder.

    Returns:
        A (np.ndarray): (m, n) constraint matrix
        b (np.ndarray): (m,) constraint vector
    """
    h1 = np.atleast_1d(np.asarray(h1, dtype=float))
    h1_dot = np.atleast_1d(np.asarray(h1_dot, dtype=float))
    Jh = np.atleast_2d(np.asarray(Jh, dtype=float))
    drift = np.atleast_1d(np.asarray(drift, dtype=float))

    h2 = h1_dot + alpha1 * h1

    A = -Jh
    b = drift + alpha1 * h1_dot + alpha2 * h2
    return A, b


def compute_table_hocbf_rows(z, J_z, dJdq_z, dq, z_min,
                             alpha1=10.0, alpha2=10.0):
    """Build the HOCBF rows ``A @ ddq <= b`` keeping points above ``z_min``.

    Vectorised over monitored points: pass arrays and get one constraint row
    per point.

    The barrier is h1 = z - z_min, which has relative degree 2 w.r.t. joint
    acceleration, so a second-order (high-order) CBF is used:

        h2      = h1_dot + alpha1 * h1
        enforce   h2_dot + alpha2 * h2 >= 0

    Expanding h2_dot = J_z @ ddq + dJ_z @ dq and rearranging into A @ ddq <= b:

        A =  -J_z
        b =   dJdq_z + alpha1 * h1_dot + alpha2 * h2

    Args:
        z (array_like): Height of each monitored point, base frame (m,)
        J_z (array_like): Row 2 (linear z) of each point's Jacobian (m, n)
        dJdq_z (array_like): z-component of the Jacobian-derivative product
            ``dJ @ dq`` for each point (m,). This is the Coriolis/centrifugal
            term of the barrier -- passing zeros makes the guarantee only
            approximate and increasingly wrong as the arm speeds up.
        dq (array_like): Joint velocities (n,)
        z_min (float): Minimum allowed height (table surface + margin)
        alpha1, alpha2 (float): HOCBF gains, both > 0. Larger = the barrier
            tolerates faster approach and intervenes later/harder.

    Returns:
        A (np.ndarray): (m, n) constraint matrix
        b (np.ndarray): (m,) constraint vector
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    J_z = np.atleast_2d(np.asarray(J_z, dtype=float))
    dJdq_z = np.atleast_1d(np.asarray(dJdq_z, dtype=float))
    dq = np.asarray(dq, dtype=float)

    h1 = z - z_min                 # (m,) clearance
    h1_dot = J_z @ dq              # (m,) vertical speed of each point

    # A plane is the one obstacle whose normal never rotates, so its gradient
    # is just the Jacobian's z row and its drift is just dJ@dq -- no curvature
    # term. That degeneracy is what keeps this function so short; do not read
    # it as the general case.
    return compute_hocbf_rows(h1, h1_dot, J_z, dJdq_z, alpha1, alpha2)


# NOTE: a single-point wrapper (`compute_table_hocbf_constraints`, taking a
# full 6xN Jacobian for the EE alone) was removed on 2026-08-11. It had no
# callers, and it was a trap: it guarded ONE point, so anything that reached
# for it expecting table avoidance would have silently dropped the elbow and
# forearm. Guard every link through `compute_table_hocbf_rows` instead -- pass
# `J_full[2, :]` and `(dJ_full[2, :] @ dq)` if you already hold full matrices.


def compute_obstacle_hocbf_rows(p, J_v, dJdq_v, dq, centers, radii, heights,
                                link_radii=0.0, alpha1=10.0, alpha2=10.0,
                                d_min=1e-3):
    """HOCBF rows keeping monitored points clear of vertical capsule obstacles.

    One row per (monitored point, obstacle) pair, in point-major order: pair
    ``(i, j)`` is row ``i * n_obstacles + j``. Stack the result under the table
    rows and the torque rows before calling `filter_control_qp`.

    Each obstacle is a vertical capsule: a segment from ``centers[j]`` rising
    ``heights[j]`` along +z, inflated by ``radii[j]``. That one primitive
    covers the three cases we care about, and the code below needs no branches
    for them beyond the closest-point clamp:

        heights[j] == 0    sphere
        heights[j] == inf  cylinder, infinite upward
        otherwise          finite capsule -- the arm can reach over the top

    Reaching over matters. An obstacle modelled as an infinite cylinder walls
    off the whole column above a 10 cm object, which on a tabletop forbids most
    of the useful workspace; the surgical descent comes from directly above.

    Why the barrier is not just ``distance - radius`` differentiated twice:
    unlike the table, the contact normal ROTATES as the arm moves. That adds a
    curvature term to the drift,

        (v_t^T v_t) / d      with v_t the component of the point's velocity
                             tangential to the normal

    which is the centrifugal relief you get from swinging around an obstacle
    rather than driving at it. It is strictly non-negative, so omitting it
    makes the filter *more* conservative rather than unsafe -- but it is the
    same class of silent modelling error as the old ``zeros()`` stand-in for
    ``dJ @ dq``, which mis-stated the barrier by up to 0.77 m/s^2, and it grows
    with the square of speed. `validate_obstacle_barrier.py` measures it.

    Args:
        p (array_like): (m, 3) monitored point positions, base frame.
        J_v (array_like): (m, 3, n) linear-velocity Jacobian of each point.
        dJdq_v (array_like): (m, 3) the ``dJ @ dq`` product for each point.
            Passing zeros makes the guarantee approximate, exactly as it does
            for the table barrier.
        dq (array_like): (n,) joint velocities.
        centers (array_like): (k, 3) capsule base points, base frame.
        radii (array_like): (k,) capsule radii.
        heights (array_like): (k,) capsule heights along +z; 0 or inf allowed.
        link_radii (array_like): (m,) or scalar radius of each monitored point,
            added to the obstacle radius so the LINK SURFACE is guarded, not
            its centreline. Matches `--table-link-radius`.
        alpha1, alpha2 (float): HOCBF gains, both > 0.
        d_min (float): floor on the centre distance used to form the normal.
            Below it the contact normal is undefined; see ``degenerate`` in the
            returned info.

    Returns:
        A (np.ndarray): (m*k, n) constraint matrix
        b (np.ndarray): (m*k,) constraint vector
        info (dict): ``h`` (m, k) signed clearance, negative means already
            penetrating; ``d`` (m, k) centre distance; ``degenerate`` (int)
            number of pairs closer than ``d_min``, where the row is
            meaningless and the caller must not treat it as protection.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    J_v = np.asarray(J_v, dtype=float)
    if J_v.ndim == 2:
        J_v = J_v[None, :, :]
    dJdq_v = np.atleast_2d(np.asarray(dJdq_v, dtype=float))
    dq = np.asarray(dq, dtype=float)
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    heights = np.atleast_1d(np.asarray(heights, dtype=float))
    link_radii = np.broadcast_to(
        np.asarray(link_radii, dtype=float), (p.shape[0],))

    # Closest point on each capsule axis. Clamping the height parameter is what
    # turns the cylinder into a capsule, and it also decides the normal's
    # geometry: on the shaft the normal is horizontal, past an end cap it is
    # fully 3-D. Track that as a mask rather than branching.
    t = p[:, None, 2] - centers[None, :, 2]                    # (m, k)
    s = np.clip(t, 0.0, heights[None, :])
    closest = np.broadcast_to(centers[None, :, :], (p.shape[0],) + centers.shape).copy()
    closest[:, :, 2] = centers[None, :, 2] + s

    e = p[:, None, :] - closest                                # (m, k, 3)
    d_true = np.linalg.norm(e, axis=2)                         # (m, k)
    d = np.maximum(d_true, d_min)
    normal = e / d[:, :, None]

    # On the shaft the closest point slides with the monitored point, so the
    # z-component of motion neither closes nor opens the gap and must be
    # projected out. Past a cap the closest point is pinned and every axis
    # counts. Same distinction, applied to velocity and to dJ@dq.
    on_shaft = (t > 0.0) & (t < heights[None, :])              # (m, k)
    axis_mask = np.ones((p.shape[0], centers.shape[0], 3))
    axis_mask[:, :, 2] = np.where(on_shaft, 0.0, 1.0)

    h1 = d_true - (radii[None, :] + link_radii[:, None])       # (m, k)

    v = np.einsum('icn,n->ic', J_v, dq)                        # (m, 3)
    v_eff = v[:, None, :] * axis_mask                          # (m, k, 3)
    h1_dot = np.einsum('ijc,ijc->ij', normal, v_eff)           # (m, k)

    Jh = np.einsum('ijc,icn->ijn', normal, J_v)                # (m, k, n)

    drift_lin = np.einsum('ijc,ijc->ij', normal,
                          dJdq_v[:, None, :] * axis_mask)
    # Tangential speed squared over distance: |v_eff|^2 - (n . v_eff)^2.
    curvature = (np.einsum('ijc,ijc->ij', v_eff, v_eff) - h1_dot ** 2) / d
    drift = drift_lin + curvature

    n_joints = J_v.shape[2]
    A, b = compute_hocbf_rows(h1.reshape(-1), h1_dot.reshape(-1),
                              Jh.reshape(-1, n_joints), drift.reshape(-1),
                              alpha1, alpha2)
    info = {"h": h1, "h_dot": h1_dot, "d": d_true,
            "degenerate": int(np.sum(d_true < d_min))}
    return A, b, info


def capsule_clearance(p, centers, radii, heights, link_radii=0.0):
    """(m, k) signed clearance between monitored points and vertical capsules.

    The same closest-point clamp as `compute_obstacle_hocbf_rows`, without the
    Jacobian work, for ranking obstacles OFF the 1 kHz loop (see
    `cull_obstacles`). Negative means penetrating.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    heights = np.atleast_1d(np.asarray(heights, dtype=float))
    link_radii = np.broadcast_to(np.asarray(link_radii, dtype=float), (p.shape[0],))
    s = np.clip(p[:, None, 2] - centers[None, :, 2], 0.0, heights[None, :])
    e = p[:, None, :] - centers[None, :, :]
    e[:, :, 2] -= s
    return np.linalg.norm(e, axis=2) - (radii[None, :] + link_radii[:, None])


def cull_obstacles(p, centers, radii, heights, link_radii, max_obstacles):
    """Indices of at most `max_obstacles` obstacles, nearest to the arm first.

    Stage one of the sliding window, meant for the perception rate (~30 Hz),
    not the torque loop: it bounds how many obstacles `compute_obstacle_hocbf_rows`
    pays for (cost is linear in m * k). An obstacle is ranked by its smallest
    clearance to ANY monitored point, so a far obstacle near the elbow still
    beats a near-ish one nobody is approaching. Speed is deliberately ignored
    here -- between perception frames the arm can reverse, so velocity-based
    ranking belongs to `ObstacleWindow`, which runs every cycle.
    """
    h = capsule_clearance(p, centers, radii, heights, link_radii)
    if h.shape[1] <= max_obstacles:
        return np.argsort(h.min(axis=0), kind="stable")
    return np.argsort(h.min(axis=0), kind="stable")[:max_obstacles]


class ObstacleWindow:
    """Stage two of the sliding window: which (point, obstacle) rows to enforce.

    A rolling CONSTRAINT set, not a rolling map. Every cycle each pair gets a
    predicted clearance over a short horizon,

        pred = h + horizon * min(h_dot, 0)

    i.e. where the gap will be if the current closing speed holds. Pairs that
    are opening are scored on today's clearance alone. Rows are kept in order
    of increasing pred, capped at `max_rows` so the QP's active-set solve stays
    bounded, and pairs whose pred exceeds `ignore_clearance` are not enforced
    at all.

    Guarantee, checked by validate_obstacle_window.py: a dropped pair is never
    more urgent than a kept one by more than `hysteresis`. Hysteresis exists
    because a row that flickers in and out of the QP makes the filtered
    acceleration jump; pairs kept last cycle get their score lowered by
    `hysteresis` so near-ties do not trade places every millisecond.

    When the cap binds, `info["overflow"]` is True and `info["dropped_min_pred"]`
    says how close the most urgent unenforced pair is. A caller must treat a
    small value there as lost protection, not as a tuning detail.
    """

    def __init__(self, max_rows=12, horizon=0.2, ignore_clearance=0.30,
                 hysteresis=0.01):
        self.max_rows = int(max_rows)
        self.horizon = float(horizon)
        self.ignore_clearance = float(ignore_clearance)
        self.hysteresis = float(hysteresis)
        self._prev = np.zeros(0, dtype=int)
        self._shape = None

    def reset(self):
        self._prev = np.zeros(0, dtype=int)
        self._shape = None

    def select(self, h, h_dot):
        """Flat row indices (point-major, as in compute_obstacle_hocbf_rows)."""
        h = np.asarray(h, dtype=float)
        pred = (h + self.horizon * np.minimum(np.asarray(h_dot, dtype=float), 0.0)).ravel()
        score = pred.copy()
        if self._shape == h.shape and self._prev.size:
            # Obstacle indices are only comparable across cycles when the set
            # did not change shape; after a re-cull start fresh.
            score[self._prev] -= self.hysteresis
        cand = np.flatnonzero(score < self.ignore_clearance)
        order = cand[np.argsort(score[cand], kind="stable")]
        keep = order[:self.max_rows]
        dropped = order[self.max_rows:]
        self._prev, self._shape = keep, h.shape
        info = {
            "overflow": dropped.size > 0,
            "n_candidates": int(cand.size),
            "dropped_min_pred": float(pred[dropped].min()) if dropped.size else np.inf,
            "kept_min_pred": float(pred[keep].min()) if keep.size else np.inf,
        }
        return keep, info


class MomentumObserver:
    """Generalised-momentum observer for the unmodelled joint torque.

    With p = M(q) dq the rigid-body dynamics M ddq + C dq + g = tau + tau_ext
    give  dp/dt = tau + tau_ext + C^T dq - g,  so

        r = K_O * ( p - p0 - integral(tau + C^T dq - g + r) dt )

    obeys dr/dt = K_O (tau_ext - r): a first-order low-pass of the true
    external torque with bandwidth K_O (rad/s), using only q, dq and the torque
    actually applied -- no acceleration, no force sensor (De Luca et al.).

    C^T dq is formed as (dM/dt) dq - C dq, which is all KDL offers; dM/dt
    along the motion is a directional finite difference of M, one extra mass
    matrix per cycle.

    `tau_ext` absorbs everything the model misses, which is the point: the
    joint-4 gravity underestimate is as unknown to the barrier as a hand is.
    """

    def __init__(self, n, gain):
        self.gain = float(gain)
        self.n = n
        self.reset()

    def reset(self):
        self.r = np.zeros(self.n)
        self._integral = None
        self._p0 = None

    def update(self, M, M_ahead, h, dq, tau_applied, c_dq, g, dt):
        """Advance one cycle and return the estimate of tau_ext.

        Args:
            M: mass matrix at q
            M_ahead: mass matrix at q + h*dq (for dM/dt @ dq)
            h: the step used for M_ahead (s)
            dq: joint velocity
            tau_applied: torque sent LAST cycle (what the arm actually felt)
            c_dq: Coriolis product C(q, dq) dq
            g: model gravity torque
            dt: cycle period (s)
        """
        p = M @ dq
        if self._integral is None:
            self._p0 = p.copy()
            self._integral = np.zeros(self.n)
            return self.r
        ct_dq = ((M_ahead - M) / h) @ dq - c_dq
        self._integral += (tau_applied + ct_dq - g + self.r) * dt
        self.r = self.gain * (p - self._p0 - self._integral)
        return self.r


def joint_limit_rows(q, dq, q_min, q_max, alpha1=15.0, alpha2=15.0):
    """HOCBF rows ``A @ ddq <= b`` keeping each joint inside its hard limits.

    Stack these with the barrier and torque rows. Without them the filter can
    certify an escape that drives a joint into its mechanical stop: the stop
    then blocks the motion, the arm does something the filter never planned,
    and in MuJoCo that took the tool 130 mm through the table with every
    solve reporting "exact". Infinite limits (continuous joints) give no row.

    Args:
        q, dq (np.ndarray): joint position/velocity (n,)
        q_min, q_max (np.ndarray): limits (n,), +/-inf for none
    """
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    n = q.shape[0]
    lo = np.flatnonzero(np.isfinite(q_min))
    hi = np.flatnonzero(np.isfinite(q_max))
    eye = np.eye(n)
    # q - q_min >= 0 has gradient +e_i; q_max - q >= 0 has -e_i; no drift.
    h1 = np.concatenate((q[lo] - q_min[lo], q_max[hi] - q[hi]))
    Jh = np.vstack((eye[lo], -eye[hi]))
    return compute_hocbf_rows(h1, Jh @ dq, Jh, np.zeros(len(h1)),
                              alpha1, alpha2)


def gravity_reserve_rows(g, Jg, curv, dq, limit, alpha1=15.0, alpha2=15.0):
    """HOCBF rows keeping ``|g_j(q)| <= limit_j`` for the selected joints.

    A barrier is only a guarantee while the motors have torque to spare. An
    operator's push can walk an unloaded arm outward until gravity alone eats
    the shoulder's budget -- in MuJoCo a 30 N shove did exactly that, and once
    gravity plus the push exceeded 39 Nm the elbow fell 240 mm in 0.25 s with
    nothing able to stop it. This keeps the arm in poses where it can still
    hold itself AND push back: limit = (1 - reserve) * tau_max.

    Two-sided, h = limit - g and h = limit + g, both relative degree 2:
        h_ddot = -/+ (Jg @ ddq + curv)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    Jg = np.atleast_2d(np.asarray(Jg, dtype=float))
    curv = np.atleast_1d(np.asarray(curv, dtype=float))
    limit = np.atleast_1d(np.asarray(limit, dtype=float))
    h1 = np.concatenate((limit - g, limit + g))
    Jh = np.vstack((-Jg, Jg))
    drift = np.concatenate((-curv, curv))
    return compute_hocbf_rows(h1, Jh @ dq, Jh, drift, alpha1, alpha2)


def velocity_limit_rows(dq, dq_max, gain=20.0):
    """CBF rows ``A @ ddq <= b`` keeping ``|dq_i| <= dq_max_i``.

    Relative degree 1, so a first-order CBF:  d/dt (dq_max - |dq|) >=
    -gain (dq_max - |dq|). Every barrier guarantee assumes the arm can still
    be braked in time; that stops being true once the motion carries more
    energy than the torque limits can remove in the remaining clearance. In
    MuJoCo, with no speed bound, the projection's own corrections spun the
    wrist to 8-20 rad/s and the tool went through the table. The Gen3 faults
    above its rated joint speed anyway, so the filter never asks for more.

    NOT used by default (TableSafetyFilter velocity_limits=False): the
    inertia-weighted projection fixed the whipping on its own, and as hard
    rows these conflict with the table rows exactly when the tool needs
    braking.
    """
    dq = np.asarray(dq, dtype=float)
    n = dq.shape[0]
    eye = np.eye(n)
    #  ddq <= gain (dq_max - dq)   and   -ddq <= gain (dq_max + dq)
    A = np.vstack((eye, -eye))
    b = np.concatenate((gain * (dq_max - dq), gain * (dq_max + dq)))
    return A, b


def torque_limit_rows(M, tau_bias, tau_max):
    """Rows enforcing ``|M @ ddq + tau_bias| <= tau_max`` as ``A @ ddq <= b``.

    Stack these under the barrier rows before calling `filter_control_qp`.

    Without them the filter certifies an acceleration the actuators cannot
    deliver: the caller maps ddq back through ``tau = M @ ddq + tau_bias`` and
    then saturates to the per-joint limit, and that saturation silently voids
    the barrier's guarantee at exactly the moment it matters -- a hard stop
    above the table is when the torque demand peaks. Folding the limits into
    the same projection means the returned ddq is one the arm can actually
    produce, and the downstream clamp becomes a no-op instead of a leak.

    Making the problem infeasible is a real possibility (the barrier may demand
    more torque than the wrist has), and that is the point: `filter_control_qp`
    then reports "degraded" and the caller can say so, rather than the conflict
    being hidden inside a clip.

    Args:
        M (array_like): (n, n) joint-space inertia matrix
        tau_bias (array_like): (n,) torque that buys no acceleration -- the
            gravity feedforward plus the Coriolis/centrifugal term. Must be the
            SAME bias the caller uses to form ddq_nom and to map the filtered
            ddq back to a torque, or the limits are enforced on a quantity the
            arm never commands.
        tau_max (array_like): (n,) per-joint torque limit, symmetric

    Returns:
        A (np.ndarray): (2n, n) constraint matrix
        b (np.ndarray): (2n,) constraint vector
    """
    M = np.asarray(M, dtype=float)
    tau_bias = np.asarray(tau_bias, dtype=float)
    tau_max = np.abs(np.asarray(tau_max, dtype=float))

    A = np.vstack((M, -M))
    b = np.concatenate((tau_max - tau_bias, tau_max + tau_bias))
    return A, b


def _nnls(E, f, max_iter):
    """Lawson-Hanson NNLS: minimise ||E u - f|| subject to u >= 0.

    Active set with an interpolation step whenever a free variable would go
    negative, so the objective decreases monotonically and the method cannot
    cycle; it terminates finitely (Lawson & Hanson 1974, ch. 23). Returns
    (u, converged).
    """
    k = E.shape[1]
    u = np.zeros(k)
    free = np.zeros(k, dtype=bool)
    it = 0
    w = E.T @ f
    while it < max_iter:
        cand = np.where(free, -np.inf, w)
        t = int(np.argmax(cand))
        if cand[t] <= 1e-10:
            return u, True
        free[t] = True
        while True:
            it += 1
            idx = np.flatnonzero(free)
            z = np.linalg.lstsq(E[:, idx], f, rcond=None)[0]
            if np.all(z > 0.0):
                u[:] = 0.0
                u[idx] = z
                break
            # Step from u toward z only as far as the first variable to hit 0.
            neg = z <= 0.0
            ui = u[idx]
            alpha = np.min(ui[neg] / (ui[neg] - z[neg]))
            u[idx] = ui + alpha * (z - ui)
            free[idx[u[idx] <= 1e-12]] = False
            u[~free] = 0.0
            if it >= max_iter:
                return u, False
        w = E.T @ (f - E @ u)
    return u, False


def filter_control_qp_weighted(ddq_nom, A, b, M):
    """`filter_control_qp` in the inertia metric: min (x-d)' M (x-d), A x <= b.

    The unweighted projection takes the smallest change in joint
    ACCELERATION, which is cheapest in the light wrist joints, so it dumps the
    correction there and whips the wrist. Weighting by M takes the smallest
    change in effort instead (Gauss's principle of least constraint), which
    is what the arm physically does. With M = L L' and y = L' x it is a plain
    projection:  min ||y - L'd||  s.t.  (A L^-T) y <= b.
    """
    L = np.linalg.cholesky(M)
    AL = np.linalg.solve(L, np.atleast_2d(A).T).T        # A L^-T
    y, status = filter_control_qp(L.T @ ddq_nom, AL, b)
    if status == "inactive":
        return ddq_nom, status
    return np.linalg.solve(L.T, y), status


def filter_control_qp(ddq_nom, A, b, iters=None):
    """Project ``ddq_nom`` onto ``{x : A x <= b}`` in the least-squares sense.

    Solves  minimize ||x - ddq_nom||^2  s.t.  A x <= b.

    No external QP solver, and bounded latency by construction -- both matter
    because this runs inside the 1 kHz torque loop with the position safety
    envelope disabled, where an iterative solver's variable iteration count
    shows up directly as cycle jitter.

      * m == 1 (one barrier): exact closed-form halfspace projection.
      * m  > 1: exact least-distance programme (Lawson & Hanson ch. 23). With
        y = x - ddq_nom the problem is  min ||y||  s.t.  -A y >= r,
        r = A ddq_nom - b, whose solution is read off ONE non-negative least
        squares problem  min_{u>=0} ||E u - e||,  E = [-A^T; r^T]:
        y = -res[:n] / res[n]  with res = E u - e, and res == 0 exactly when
        the constraints are infeasible. NNLS is a finite active-set method
        that cannot cycle, and it is happy with more rows than joints.

        This replaced (2026-10-26) a dual active set with no step-length rule,
        which zigzagged and came back "degraded" on 11% of solvable random
        problems -- and, once the tool hull and joint stops added rows, on
        most real barrier problems in MuJoCo, with the tool going through the
        table. An iterative scheme had been rejected before that for the same
        reason this one is exact: an under-converged correction is too WEAK,
        which for a safety filter is the unsafe direction.

    Args:
        ddq_nom (np.ndarray): Nominal desired joint accelerations (n,)
        A (np.ndarray): (m, n) inequality matrix
        b (np.ndarray): (m,) inequality vector
        iters (int): NNLS iteration cap (default 3*m + 10). Lawson-Hanson
            needs far fewer in practice; the cap only bounds worst-case latency.

    Returns:
        ddq_safe (np.ndarray): Safe joint accelerations (n,). Returns the input
            array object itself when no constraint is active, so callers can
            skip downstream work with an `is` check.
        status (str): Which path produced it, so a caller can log a filter that
            is no longer giving the exact answer.

            * ``"inactive"`` -- nominal command already feasible (no-op).
            * ``"exact"``    -- exact projection onto the feasible set.
            * ``"blocked"``  -- the only violated rows are unactuated in this
              configuration; nothing is expressible, input returned unchanged.
            * ``"degraded"`` -- the constraints conflict (no feasible point),
              so the residual was spread across the violated rows by least
              squares. Constraints are still violated; this is NOT a
              certificate.
    """
    A = np.atleast_2d(A)
    b = np.atleast_1d(b)

    resid = A @ ddq_nom - b
    if np.all(resid <= 0.0):
        # Nominal command is already feasible -- the common case away from the
        # table. Return the input object so callers can detect the no-op.
        return ddq_nom, "inactive"

    if A.shape[0] == 1:
        a = A[0]
        aa = float(a @ a)
        if aa < 1e-12:
            # Degenerate row: this point's height is unactuated in the current
            # configuration, so no correction is expressible.
            return ddq_nom, "blocked"
        return ddq_nom - (float(resid[0]) / aa) * a, "exact"

    # Rows with a ~zero gradient cannot be corrected at all (that point's
    # height is unactuated here); leave them out of the solve.
    norms = np.sqrt(np.einsum("ij,ij->i", A, A))
    live = norms >= 1e-6
    if not np.any(resid[live] > 0.0):
        return ddq_nom, "blocked"
    # Condition the LDP: unit-norm rows, then scale so the largest violation
    # is 1. Unscaled, a ~1e3 rad/s^2 correction (torque rows far from the
    # nominal) put res[-1] near 1e-7 and the division below lost the answer
    # -- a feasible problem reported "degraded" on the torque rows alone.
    Al = A[live] / norms[live, None]
    rl = resid[live] / norms[live]
    scale = float(np.max(np.abs(rl)))
    rs = rl / scale

    n = A.shape[1]
    E = np.vstack((-Al.T, rs[None, :]))
    e = np.zeros(n + 1)
    e[-1] = 1.0
    u, converged = _nnls(E, e, 3 * len(rl) + 10 if iters is None else iters)
    res = E @ u - e
    tol = 1e-7
    if converged and abs(res[-1]) > 1e-12 and np.linalg.norm(res) > 1e-10:
        x = ddq_nom - scale * res[:n] / res[-1]
        # Feasibility in the normalised row metric (distance, rad/s^2).
        if np.all(Al @ (x - ddq_nom) + rl <= tol * max(1.0, scale)):
            return x, "exact"

    # Infeasible (the rows genuinely conflict -- e.g. two links where saving
    # one drives the other down, or a barrier the torque envelope cannot
    # honour) or the iteration cap was hit. Degrade predictably: spread the
    # correction across all violated rows in the least-squares sense, which
    # shares the residual out instead of sacrificing one link entirely.
    viol = rl > tol
    delta = np.linalg.lstsq(Al[viol], rl[viol], rcond=None)[0]
    return ddq_nom - delta, "degraded"


def wall_cap_clearance(standoff, k_wall, f_max):
    """Clearance above z_min at which `table_wall_torque`'s force cap binds.

    Returns 0.0 when the cap is high enough that the spring reaches the surface
    un-clipped (the intended tuning). A positive value is the height at which
    the wall stops being a spring and becomes a constant push -- everything
    below it is soft in the only place that has to be hard.
    """
    if standoff <= 0.0 or k_wall <= 0.0:
        return 0.0
    f_surface = k_wall * standoff
    if f_max >= f_surface:
        return 0.0
    s_cap = np.sqrt(f_max / f_surface)      # F = k*standoff*s^2 = f_max
    return float(standoff * (1.0 - s_cap))


def table_wall_torque(z, J_z, dq, z_min, standoff, k_wall, d_wall,
                      f_max=90.0):
    """Joint torque from a virtual spring-damper wall above the table.

    Unlike the HOCBF -- which only constrains acceleration and therefore feels
    slack when you lean on it -- this applies a genuine upward force to any
    monitored point that enters the standoff band, so the arm pushes back.

    Force on point i, with clearance d = z - z_min and penetration s = 1 - d/standoff:

        F = k_wall * standoff * s^2  +  d_wall * s * max(0, -z_dot)

    The quadratic spring makes engagement smooth (zero force *and* zero
    gradient at the band edge, so there is no torque step as a link crosses in)
    while still becoming very stiff at the surface -- the effective stiffness
    dF/dd is 2 * k_wall * s, so it reaches 2 * k_wall right at z_min. Damping
    is one-sided: it resists approach but never sucks a retreating link back
    down. Below the barrier (d < 0) the spring saturates at its cap rather than
    growing without bound, so a bad z_min or a modelling error cannot command a
    wild torque.

    IMPORTANT: the spring alone peaks at ``k_wall * standoff`` at the surface,
    so `f_max` must exceed that or the cap binds INSIDE the band and the last
    stretch before the table is a constant push rather than a stiff wall --
    the opposite of what the quadratic profile is for. `wall_cap_clearance`
    reports where the cap starts binding; keep it at zero.

    Args:
        z (array_like): Monitored point heights (m,)
        J_z (array_like): Row 2 of each point's Jacobian (m, n)
        dq (array_like): Joint velocities (n,)
        z_min (float): Barrier height
        standoff (float): Band thickness above z_min where the wall acts (m)
        k_wall (float): Wall stiffness (N/m)
        d_wall (float): Wall damping (N*s/m)
        f_max (float): Per-point force cap (N)

    Returns:
        tau (np.ndarray): (n,) joint torque
        f (np.ndarray): (m,) per-point applied force, for logging
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    J_z = np.atleast_2d(np.asarray(J_z, dtype=float))
    dq = np.asarray(dq, dtype=float)

    if standoff <= 0.0:
        return np.zeros(J_z.shape[1]), np.zeros(z.shape[0])

    d = z - z_min
    s = np.clip((standoff - d) / standoff, 0.0, 1.0)   # 0 outside band, 1 at/below z_min
    active = s > 0.0
    if not np.any(active):
        return np.zeros(J_z.shape[1]), np.zeros(z.shape[0])

    z_dot = J_z @ dq
    approach = np.maximum(0.0, -z_dot)                 # one-sided damping

    # `s` is already 0 for every point outside the band, so both the spring and
    # the damping term vanish there and the clip leaves them at 0 -- no
    # separate masking by `active` is needed (it was a no-op).
    f = k_wall * standoff * s ** 2 + d_wall * s * approach
    f = np.clip(f, 0.0, f_max)

    # Map the +z point forces to joint torque: tau = sum_i J_z,i^T * f_i
    return J_z.T @ f, f
