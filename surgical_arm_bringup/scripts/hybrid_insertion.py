#!/usr/bin/env python3
"""Hybrid impedance / force-limited control law for the needle insertion.

WHY A HYBRID
------------
The insertion needs two incompatible things at once:
  * ACROSS the needle axis (and in orientation) it must be stiff and accurate:
    the gate is 3 mm / 2 deg at the tip, and the 42 mm bore leaves the
    assembly ~3 mm to spare at the 39 mm limit.
  * ALONG the axis it must be gentle: if the needle meets something it should
    push with a bounded force and stop, never wind up a spring until something
    gives.
A single impedance cannot be both. This law splits the tip's task space with
the insertion axis `a`: stiff impedance (plus a capped integral) on the
lateral subspace and the orientation, and a saturated, rate-limited axial
reference along `a`. Everything is expressed at the NEEDLE TIP (0.41 m past
the flange, where 1 deg is 7 mm), never at the flange.

It runs UNDER impedance.TableSafetyFilter, whose tool floor is moved to the
depth stop (container_geometry.insertion_tool_floor) for the descent -- the
law aims for the target, the barrier guarantees the overshoot limit.

PHASES (self-contained state machine, advanced by `torque()`)
  "descend"  axial reference moves at `speed` from the entry point to the
             target depth; push force saturates at `f_push_max`.
  "hold"     target reached: hold the tip at depth.
  "retract"  aborted (axial contact force above `f_abort`, measured against a
             baseline tared at entry) or commanded: reference backs out along
             -a to the entry point at `speed`.
  "done"     back at the entry point after a retract.

ROS-free and robot-free: it consumes impedance.KinDynModel queries and returns
joint torque. The gravity feedforward is the caller's, as in impedance.py.
"""
import numpy as np


def _rotvec(R):
    """Rotation vector of a rotation matrix (axis * angle)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(c)
    if ang < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * ang / (2.0 * np.sin(ang))


class HybridInsertionController:
    """Tip-frame hybrid controller for one insertion.

    Args:
        model: impedance.KinDynModel (with the tool lumped in)
        seg_index: link the tip is fixed in (impedance.TOOL_HULL_LINK's index)
        tip_local: needle tip in that link's frame (assembly_tip)
        q_entry: joint pose at the entry point -- sets the axis, the
            orientation to hold and the null-space posture
        depth: insertion depth along the axis from the entry point (m)
        speed: axial reference speed (m/s)
        k_lat, d_lat: lateral stiffness (N/m) / damping (N s/m)
        ki_lat, i_lat_max: lateral integral gain (N/(m s)) and its cap (N)
        k_rot, d_rot: orientation stiffness (Nm/rad) / damping (Nm s/rad)
        ki_rot, i_rot_max: orientation integral gain and its cap (Nm). The
            joint-4 gravity error alone holds the tip ~4 deg off a 60 Nm/rad
            spring; the 2 deg gate needs the integral.
        k_ax, d_ax: axial spring/damper toward the moving reference
        ki_ax: axial integral gain (N/(m s)). It can only raise the push UP
            TO f_push_max, never past it -- a steady few newtons of model
            error otherwise stops the descent millimetres short.
        f_push_max: most the axial command may push along +a (N)
        f_abort: axial contact force (tared at entry) that aborts (N). Must
            sit BELOW f_push_max, or a needle blocked by something that
            holds against f_push_max can never trip it.
        stall_lag, stall_time: abort if the axial reference leads the tip by
            more than stall_lag (m) for stall_time (s) -- the needle is
            blocked. Independent of the force estimate, which drifts a few
            newtons with model error.

        Damping/integral gains were tuned in MuJoCo against a plant with
        0.1 kg m^2 rotor inertia per joint (KDL models none): the stiffer
        first set (d 80/4, ki 4000/150) limit-cycled at ~2 Hz there and the
        fixture clipped the container wall.
        k_null, d_null: null-space posture gains toward q_entry
    """

    def __init__(self, model, seg_index, tip_local, q_entry, depth,
                 speed=0.005, k_lat=2500.0, d_lat=160.0, ki_lat=1500.0,
                 i_lat_max=15.0, k_rot=60.0, d_rot=8.0, ki_rot=60.0,
                 i_rot_max=6.0, k_ax=1500.0, d_ax=60.0, ki_ax=3000.0,
                 f_push_max=8.0, f_abort=6.0, stall_lag=0.003,
                 stall_time=0.3, k_null=5.0, d_null=1.0):
        self.model = model
        self.seg = seg_index
        self.tip = np.asarray(tip_local, dtype=float)
        p0, R0, _ = model.point_frame_jacobian(q_entry, seg_index, self.tip)
        self.p_entry = p0
        self.R_des = R0
        # The needle points along the tip offset from the bracelet origin.
        a = R0 @ (self.tip / np.linalg.norm(self.tip))
        self.a = a / np.linalg.norm(a)
        self.P_lat = np.eye(3) - np.outer(self.a, self.a)
        self.depth = float(depth)
        self.speed = float(speed)
        self.k_lat, self.d_lat = k_lat, d_lat
        self.ki_lat, self.i_lat_max = ki_lat, i_lat_max
        self.k_rot, self.d_rot = k_rot, d_rot
        self.ki_rot, self.i_rot_max = ki_rot, i_rot_max
        self.k_ax, self.d_ax = k_ax, d_ax
        self.ki_ax = ki_ax
        self.f_push_max = f_push_max
        if not f_abort < f_push_max:
            raise ValueError("f_abort must be below f_push_max")
        self.f_abort = f_abort
        self.stall_lag, self.stall_time = stall_lag, stall_time
        self._stall_t = 0.0
        self.k_null, self.d_null = k_null, d_null
        self.q_post = np.array(q_entry, dtype=float)
        self.phase = "descend"
        self.s_ref = 0.0
        self.i_lat = np.zeros(3)
        self.i_rot = np.zeros(3)
        self.i_ax = 0.0
        self.f_ax_tare = None
        self.abort_reason = None

    # -- state ---------------------------------------------------------------
    def tip_state(self, q):
        """(tip position, axial depth s, lateral error vector) at q."""
        p, _, _ = self.model.point_frame_jacobian(q, self.seg, self.tip)
        d = p - self.p_entry
        return p, float(self.a @ d), self.P_lat @ d

    def retract(self, reason="commanded"):
        if self.phase in ("descend", "hold"):
            self.phase = "retract"
            self.abort_reason = reason

    # -- control -------------------------------------------------------------
    def torque(self, q, dq, tau_g, dt, tau_ext=None):
        """Joint torque for this cycle, gravity feedforward `tau_g` included.

        `tau_ext` is TableSafetyFilter's observer estimate (or None). Its tip
        force along the axis, tared at entry, is what aborts the insertion.
        Returns (tau, info).
        """
        p, R, J = self.model.point_frame_jacobian(q, self.seg, self.tip)
        v = J[:3] @ dq
        w = J[3:] @ dq
        d = p - self.p_entry
        s = float(self.a @ d)
        v_ax = float(self.a @ v)

        # Contact force estimate along the axis (positive = resisting the
        # descent), from the observer's joint torque: f = pinv(J^T) tau_ext.
        f_ax_contact = 0.0
        if tau_ext is not None:
            f = np.linalg.lstsq(J.T, tau_ext, rcond=None)[0]
            f_raw = -float(self.a @ f[:3])
            if self.f_ax_tare is None:
                self.f_ax_tare = f_raw
            f_ax_contact = f_raw - self.f_ax_tare
            if self.phase == "descend" and f_ax_contact > self.f_abort:
                self.retract(f"axial contact {f_ax_contact:.1f} N > "
                             f"{self.f_abort:.1f} N")

        # Stall: the reference is running away from a tip that is not moving.
        if self.phase == "descend":
            if self.s_ref - s > self.stall_lag:
                self._stall_t += dt
                if self._stall_t >= self.stall_time:
                    self.retract(f"stalled {(self.s_ref - s)*1000:.1f} mm "
                                 f"behind the reference for "
                                 f"{self._stall_t:.2f} s")
            else:
                self._stall_t = 0.0

        # Axial reference and phase transitions.
        if self.phase == "descend":
            self.s_ref = min(self.s_ref + self.speed * dt, self.depth)
            if self.s_ref >= self.depth and abs(s - self.depth) < 0.001:
                self.phase = "hold"
            v_ref = self.speed if self.s_ref < self.depth else 0.0
        elif self.phase == "retract":
            # Back out from wherever the tip actually is, never jump.
            self.s_ref = max(min(self.s_ref, s) - self.speed * dt, 0.0)
            v_ref = -self.speed if self.s_ref > 0.0 else 0.0
            if self.s_ref <= 0.0 and abs(s) < 0.002:
                self.phase = "done"
        else:
            v_ref = 0.0

        # Lateral: stiff spring to the insertion line + capped integral.
        e_lat = -(self.P_lat @ d)
        if self.phase in ("descend", "hold"):
            self.i_lat = self.i_lat + self.ki_lat * e_lat * dt
            n_i = np.linalg.norm(self.i_lat)
            if n_i > self.i_lat_max:
                self.i_lat *= self.i_lat_max / n_i
        f_lat = (self.k_lat * e_lat - self.d_lat * (self.P_lat @ v)
                 + self.P_lat @ self.i_lat)

        # Axial: spring to the moving reference + integral, push saturated.
        # The integral is clamped so spring + integral never exceeds the push
        # limit (anti-windup), and it is dropped on retract.
        e_ax = self.s_ref - s
        f_spring = self.k_ax * e_ax - self.d_ax * (v_ax - v_ref)
        if self.phase in ("descend", "hold"):
            self.i_ax = float(np.clip(self.i_ax + self.ki_ax * e_ax * dt,
                                      -self.f_push_max, self.f_push_max))
        else:
            self.i_ax = 0.0
        f_ax = f_spring + self.i_ax
        if f_ax > self.f_push_max:
            self.i_ax = max(min(self.i_ax, self.f_push_max - f_spring), 0.0)
        f_ax = float(np.clip(f_spring + self.i_ax, -3.0 * self.f_push_max,
                             self.f_push_max))

        # Orientation: hold the entry orientation (roll included -- the tool
        # body is asymmetric and must clear the container wall), with a
        # capped integral for the steady gravity-model error.
        e_rot = _rotvec(self.R_des @ R.T)
        self.i_rot = self.i_rot + self.ki_rot * e_rot * dt
        n_r = np.linalg.norm(self.i_rot)
        if n_r > self.i_rot_max:
            self.i_rot *= self.i_rot_max / n_r
        m = self.k_rot * e_rot - self.d_rot * w + self.i_rot

        F = np.concatenate((f_lat + f_ax * self.a, m))
        tau = tau_g + J.T @ F

        # Null-space posture, projected so it never disturbs the tip.
        Jp = np.linalg.pinv(J)
        N = np.eye(self.model.n) - J.T @ Jp.T
        tau = tau + N @ (self.k_null * (self.q_post - q) - self.d_null * dq)

        info = dict(phase=self.phase, s=s, s_ref=self.s_ref,
                    lat_err=float(np.linalg.norm(self.P_lat @ d)),
                    rot_err_deg=float(np.degrees(np.linalg.norm(e_rot))),
                    f_ax=f_ax, f_ax_contact=f_ax_contact,
                    abort=self.abort_reason)
        return tau, info
