#!/usr/bin/env python3
"""Arm kinematics/dynamics model (PyKDL) for the Gen3 7-DOF.

Moved out of impedance.py so planners, simulators and validators can use the
model without importing the 1 kHz controller. impedance.py re-exports every
name here (imp.KinDynModel, imp.DEFAULT_URDF, ...), so existing callers are
unchanged.

No ROS and no kortex_api: NumPy at import, PyKDL + urdf_parser_py lazily when
a KinDynModel is built (source /opt/ros/humble/setup.bash, or on REAL-1 the
built workspace before the kortex_impedance venv python).
"""
import os

import numpy as np


# --- Kinematics/dynamics model (cartesian mode) ------------------------------
# URDF providing the arm chain. The bundled gen3 URDF namespaces links with a
# "gen3_" prefix; that is fine -- we hold a RELATIVE startup pose, so absolute
# frame naming is irrelevant, and the arm inertias/kinematics match the real
# arm regardless of the (2f85 vs 2f140) gripper beyond the tip link.
# Bundled in this package's config/ so cartesian mode is self-contained on any
# machine (only the XML is parsed -- PyKDL never loads the referenced meshes).
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(_PKG_DIR, "config", "gen3_2f85.urdf")
DEFAULT_BASE_LINK = "gen3_base_link"
DEFAULT_TIP_LINK = "gen3_end_effector_link"

# --- Tool payload (lumped onto the chain) ------------------------------------
# KDL's getChain() keeps only links ON the base->tip path. Everything the URDF
# hangs off the side -- the bundled 2F-85, the vision-module bracket -- is
# silently dropped, so the bare chain models NO tool at all, and the real
# 1.567 kg thesis_ee was ~6 Nm of unmodelled wrist moment. --tool lumps the
# mounted tool's rigid-body inertia into the end_effector_link segment.
#
# Each preset is (mass kg, CoM xyz m, inertia-at-CoM 3x3 kg m^2), all in the
# end_effector_link frame (+Z out of the flange).
#   2f140     -- Robotiq 2F-140 as configured ON THE ARM (Base > ControlConfig >
#                GetToolConfiguration: 1.025 kg, CoM z = 0.073; read 2026-10-23).
#                The kortex URDF's 0.364 kg sum of 2F-140 links is wrong. Inertia
#                is the URDF's tensor scaled to the real mass.
#   thesis_ee -- SolidWorks "Mass Properties.pdf" (1.567 kg, CoM (107.70,
#                120.07, 145.15) mm, L tensor at CoM), all in the Full_Assembly
#                STL frame, mapped through the VISUAL mesh origin -- the one
#                validated by the measured assembly_tip (2026-10-09). The
#                xacro's <inertial> used the pen-era collision mapping, which
#                put the CoM 46 mm behind the flange; this puts it 95 mm to the
#                side of it, level with the flange.
TOOL_PAYLOADS = {
    "2f140": (1.025, (0.0, 0.0, 0.073),
              np.diag([0.002029, 0.003742, 0.002079])),
    "thesis_ee": (1.567, (0.09507, 0.00230, -0.05685),
                  np.array([[0.021982, 0.0000105, 0.0001843],
                            [0.0000105, 0.021976, 0.0001164],
                            [0.0001843, 0.0001164, 0.005741]])),
    "none": None,
}
# The bundled URDF carries the NON-vision bracelet (0.364 kg) and models the
# vision module as an off-path 0.5 kg bracket -- which the chain drops. The arm
# has the vision module; Kinova's own description (gen3.xacro vision:=true)
# replaces the bracelet inertia with this. Applied whenever --tool is given.
VISION_BRACELET = (0.5, (0.000281, 0.011402, -0.029798),
                   np.array([[0.000587, 3e-6, 3e-6],
                             [3e-6, 0.000369, -0.000118],
                             [3e-6, -0.000118, 0.000609]]))


# =============================================================================
# Kinematics / dynamics model (PyKDL) -- only used in cartesian mode
# =============================================================================
class KinDynModel:
    """FK, Jacobian, mass matrix and gravity for the arm chain, via PyKDL.

    PyKDL + urdf_parser_py are imported here (lazily) so that joint mode does
    not require them. kdl_parser_py is not packaged for this distro, so the
    URDF->KDL tree conversion is vendored below (from kdl_parser_py).
    """

    # --- Self-collision capsule model (cartesian-mode redundancy resolution) --
    # The arm is approximated as a polyline of equal-radius capsules (one per
    # link). A reactive safety cushion, NOT a certified collision checker; err
    # large on the radius -- the torque clamp bounds the worst case anyway.
    CAPSULE_RADIUS = 0.055      # m, uniform link "thickness"
    COLLISION_MARGIN = 0.08     # m, potential switches on below this clearance

    def __init__(self, urdf_path, base_link, tip_link, tool=None):
        import PyKDL as kdl
        from urdf_parser_py.urdf import URDF
        self.kdl = kdl

        robot = URDF.from_xml_file(urdf_path)
        tree = self._tree_from_urdf(robot, kdl)
        self.chain = tree.getChain(base_link, tip_link)
        self.n = self.chain.getNrOfJoints()
        # tool=None keeps the bare URDF chain (pre-2026-10-23 behaviour, and
        # what the offline validators were tuned against).
        self.tool = tool
        if tool is not None:
            self.chain = self._with_tool(self.chain, tool)

        self._fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self._jac = kdl.ChainJntToJacSolver(self.chain)
        self._dyn = kdl.ChainDynParam(self.chain, kdl.Vector(0, 0, -9.81))

        # Jacobian time-derivative solver, used by the table barrier for the
        # exact Coriolis term. HYBRID = base-frame axes with the reference
        # point at the segment tip, matching self.jacobian()'s convention.
        self._jdot = kdl.ChainJntToJacDotSolver(self.chain)
        self._jdot.setRepresentation(kdl.ChainJntToJacDotSolver.HYBRID)

        # Scratch objects reused by the hot-loop barrier queries, so a 1 kHz
        # call does not allocate a KDL object per link per cycle.
        self._scratch_frame = kdl.Frame()
        self._scratch_jac = kdl.Jacobian(self.n)
        self._scratch_twist = kdl.Twist()
        self._scratch_qv = kdl.JntArrayVel(self.n)
        self._scratch_cor = kdl.JntArray(self.n)
        # Two slots, because `coriolis` needs q and dq live at the same time.
        self._scratch_jnt = (kdl.JntArray(self.n), kdl.JntArray(self.n))
        # Non-adjacent capsule pair indices, keyed by capsule count.
        self._pair_cache = {}

    def _with_tool(self, chain, tool):
        """Return a copy of `chain` with the vision bracelet and `tool` lumped in.

        Only segment inertias change; joints and frames are copied as-is, so
        FK and Jacobians are bit-identical to the bare chain.
        """
        kdl = self.kdl

        def rbi(mass, com, i_com):
            i = np.asarray(i_com, dtype=float)
            return kdl.RigidBodyInertia(
                mass, kdl.Vector(*com),
                kdl.RotationalInertia(i[0, 0], i[1, 1], i[2, 2],
                                      i[0, 1], i[0, 2], i[1, 2]))

        payload = TOOL_PAYLOADS[tool]
        names = [chain.getSegment(s).getName()
                 for s in range(chain.getNrOfSegments())]
        flange = [nm for nm in names if nm.endswith("end_effector_link")]
        bracelet = [nm for nm in names if nm.endswith("bracelet_link")]
        if payload is not None and not flange:
            raise ValueError("--tool needs end_effector_link on the chain "
                             f"(tip is {names[-1]})")

        out = kdl.Chain()
        for s in range(chain.getNrOfSegments()):
            seg = chain.getSegment(s)
            inertia = seg.getInertia()
            if bracelet and seg.getName() == bracelet[0]:
                inertia = rbi(*VISION_BRACELET)
            if payload is not None and seg.getName() == flange[0]:
                inertia = inertia + rbi(*payload)
            out.addSegment(kdl.Segment(seg.getName(), seg.getJoint(),
                                       seg.getFrameToTip(), inertia))
        return out

    def total_mass(self):
        return sum(self.chain.getSegment(s).getInertia().getMass()
                   for s in range(self.chain.getNrOfSegments()))

    # -- vendored from kdl_parser_py (treeFromUrdfModel), rotation of the
    #    inertial origin dropped (URDF inertials here are axis-aligned) --------
    @staticmethod
    def _tree_from_urdf(robot, kdl):
        def to_pose(p):
            if p and p.rpy and len(p.rpy) == 3 and p.xyz and len(p.xyz) == 3:
                return kdl.Frame(kdl.Rotation.RPY(*p.rpy), kdl.Vector(*p.xyz))
            return kdl.Frame()

        def to_joint(j):
            f = to_pose(j.origin)
            if j.type in ("revolute", "continuous"):
                return kdl.Joint(j.name, f.p, f.M * kdl.Vector(*j.axis),
                                 kdl.Joint.RotAxis)
            if j.type == "prismatic":
                return kdl.Joint(j.name, f.p, f.M * kdl.Vector(*j.axis),
                                 kdl.Joint.TransAxis)
            return kdl.Joint(j.name, kdl.Joint.Fixed)

        def to_rbi(inertial):
            o = to_pose(inertial.origin)
            i = inertial.inertia
            rot = kdl.RotationalInertia(i.ixx, i.iyy, i.izz, i.ixy, i.ixz, i.iyz)
            return kdl.RigidBodyInertia(inertial.mass, o.p, rot)

        def add_children(root, tree):
            for jnt in robot.joints:
                if jnt.parent != root:
                    continue
                link = robot.link_map[jnt.child]
                inert = (to_rbi(link.inertial) if link.inertial is not None
                         else kdl.RigidBodyInertia())
                seg = kdl.Segment(jnt.child, to_joint(jnt), to_pose(jnt.origin),
                                  inert)
                tree.addSegment(seg, root)
                add_children(jnt.child, tree)

        tree = kdl.Tree(robot.get_root())
        add_children(robot.get_root(), tree)
        return tree

    def _to_jnt(self, q, slot=0):
        """Copy `q` into a preallocated JntArray and return it.

        Reuses a scratch array instead of allocating one: this runs ~7 times
        per 1 kHz cycle and the allocation was ~30% of its cost.

        The returned object is SHARED and is overwritten by the next call with
        the same `slot`. A call site that needs two JntArrays alive at once --
        `coriolis`, which hands q and dq to a single solver call -- must ask
        for different slots, or both arguments alias the same data and the
        solver silently sees dq twice.
        """
        ja = self._scratch_jnt[slot]
        for i in range(self.n):
            ja[i] = float(q[i])
        return ja

    def fk(self, q):
        """Return (position [3], quaternion [x,y,z,w]) of the tip in base frame."""
        frame = self.kdl.Frame()
        self._fk.JntToCart(self._to_jnt(q), frame)
        p = np.array([frame.p[i] for i in range(3)])
        quat = np.array(frame.M.GetQuaternion())  # (x, y, z, w)
        return p, quat

    def jacobian(self, q):
        """Return the 6xN geometric Jacobian (base frame, tip reference)."""
        J = self.kdl.Jacobian(self.n)
        self._jac.JntToJac(self._to_jnt(q), J)
        return np.array([[J[r, c] for c in range(self.n)] for r in range(6)])

    def segment_index(self, name):
        """1-based segment index whose tip is link `name` (as JntToCart wants).

        Segment s spans joint s, so FK/Jacobian with seg_nr = s returns the
        pose/Jacobian of the tip of the s-th segment. Index 0 is the chain root.
        """
        for s in range(self.chain.getNrOfSegments()):
            if self.chain.getSegment(s).getName() == name:
                return s + 1
        raise ValueError(
            f"link '{name}' is not on the chain; available: "
            + ", ".join(self.chain.getSegment(s).getName()
                        for s in range(self.chain.getNrOfSegments())))

    def segment_names(self):
        """Names of every link on the chain, base -> tip."""
        return [self.chain.getSegment(s).getName()
                for s in range(self.chain.getNrOfSegments())]

    def height_barrier_terms(self, q, dq, seg_indices):
        """Per-link (z, J_z, dJdq_z) for the table barrier.

        Returns only the vertical row of each Jacobian and the z-component of
        the Jacobian-derivative product -- the barrier needs nothing else, and
        materialising full 6xN matrices per link at 1 kHz is pure waste.

        Args:
            q, dq (np.ndarray): joint position/velocity (n,)
            seg_indices (sequence[int]): segment indices from segment_index()

        Returns:
            z (np.ndarray): (m,) link-tip heights in base frame
            J_z (np.ndarray): (m, n) vertical Jacobian rows
            dJdq_z (np.ndarray): (m,) z-component of dJ @ dq (exact, from KDL)
        """
        m = len(seg_indices)
        z = np.empty(m)
        J_z = np.zeros((m, self.n))
        dJdq_z = np.empty(m)

        ja = self._to_jnt(q)
        # JntArrayVel bundles q and dq for the JacDot solver.
        qv = self._scratch_qv
        for i in range(self.n):
            qv.q[i] = float(q[i])
            qv.qdot[i] = float(dq[i])

        frame, jac, twist = (self._scratch_frame, self._scratch_jac,
                             self._scratch_twist)
        for k, s in enumerate(seg_indices):
            self._fk.JntToCart(ja, frame, s)
            z[k] = frame.p[2]

            self._jac.JntToJac(ja, jac, s)
            for c in range(self.n):
                J_z[k, c] = jac[2, c]

            # Twist overload returns the dJ @ dq product directly, so the full
            # 6xN derivative matrix is never formed.
            self._jdot.JntToJacDot(qv, twist, s)
            dJdq_z[k] = twist.vel[2]

        return z, J_z, dJdq_z

    def point_frame_jacobian(self, q, seg_index, p_local):
        """(p, R, J) for a point fixed in link `seg_index` (e.g. the needle tip).

        p: point position (base frame); R: the link's rotation; J: 6xN with
        rows 0-2 the point's linear velocity and rows 3-5 the link's angular
        velocity -- the Jacobian of the point, not of the link origin, so a
        task-space controller can act at the tool tip 0.41 m past the flange.
        """
        ja = self._to_jnt(q)
        frame, jac = self._scratch_frame, self._scratch_jac
        self._fk.JntToCart(ja, frame, seg_index)
        self._jac.JntToJac(ja, jac, seg_index)
        R = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)])
        J = np.array([[jac[r, c] for c in range(self.n)] for r in range(6)])
        r = R @ np.asarray(p_local, dtype=float)
        p = np.array([frame.p[i] for i in range(3)]) + r
        # v_p = v + w x r = v - r x w
        J[:3] -= np.cross(r, J[3:].T).T
        return p, R, J

    def gravity_terms(self, q, dq, joints, h=1e-5):
        """(g_j, dg_j/dq, d2g_j/dt2 at ddq=0) for the gravity-reserve barrier.

        Finite differences of `gravity`: one central difference per joint for
        the gradient, and a second directional difference along dq for the
        curvature term dq' (d2g/dq2) dq. ~16 gravity calls; KDL's are cheap.
        """
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        g0 = self.gravity(q)
        G = np.empty((self.n, self.n))
        for k in range(self.n):
            e = np.zeros(self.n)
            e[k] = h
            G[:, k] = (self.gravity(q + e) - self.gravity(q - e)) / (2 * h)
        s = np.linalg.norm(dq)
        if s > 1e-9:
            u = dq / s
            hh = 1e-4
            curv = (self.gravity(q + hh * u) - 2 * g0
                    + self.gravity(q - hh * u)) / hh**2 * s**2
        else:
            curv = np.zeros(self.n)
        return g0[joints], G[joints], curv[joints]

    def body_point_heights(self, q, seg_index, pts):
        """Heights (base frame) of points `pts` (k, 3) fixed in link `seg_index`.

        Cheap (one FK + a k x 3 product), so the barrier can scan a whole tool
        hull every cycle and only build Jacobians for the lowest few points.
        """
        frame = self._scratch_frame
        self._fk.JntToCart(self._to_jnt(q), frame, seg_index)
        r_z = np.array([frame.M[2, j] for j in range(3)])
        return frame.p[2] + pts @ r_z

    def body_point_barrier_terms(self, q, dq, seg_index, pts):
        """(z, J_z, dJdq_z) for points `pts` (k, 3) rigidly fixed in a link.

        Same contract as `height_barrier_terms`, for points that are not link
        origins -- the tool hull. Built from the link's own HYBRID Jacobian and
        dJ@dq by rigid-body transport, r = R @ p_local:
            v_p = v + w x r                  -> J_z = Jv_z + r_y Jw_x - r_x Jw_y
            a_p = a + dw x r + w x (w x r)   -> the dJdq_z drift below
        so one FK/Jacobian/JacDot call covers every point on the body.
        """
        ja = self._to_jnt(q)
        qv = self._scratch_qv
        for i in range(self.n):
            qv.q[i] = float(q[i])
            qv.qdot[i] = float(dq[i])
        frame, jac, twist = (self._scratch_frame, self._scratch_jac,
                             self._scratch_twist)
        self._fk.JntToCart(ja, frame, seg_index)
        self._jac.JntToJac(ja, jac, seg_index)
        self._jdot.JntToJacDot(qv, twist, seg_index)

        R = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)])
        J = np.array([[jac[r, c] for c in range(self.n)] for r in range(6)])
        r = np.asarray(pts, dtype=float) @ R.T              # (k, 3) world offsets
        w = J[3:] @ np.asarray(dq, dtype=float)              # link angular velocity
        z = frame.p[2] + r[:, 2]
        J_z = J[2] + np.outer(r[:, 1], J[3]) - np.outer(r[:, 0], J[4])
        dJdq_z = (twist.vel[2] + twist.rot[0] * r[:, 1] - twist.rot[1] * r[:, 0]
                  + w[2] * (r @ w) - r[:, 2] * (w @ w))
        return z, J_z, dJdq_z

    def point_barrier_terms(self, q, dq, seg_indices):
        """Per-link (p, J_v, dJdq_v) for distance barriers.

        The 3-D counterpart of `height_barrier_terms`. A plane barrier needs
        only the vertical row because its normal never rotates; an obstacle's
        normal does, so `compute_obstacle_hocbf_rows` needs all three linear
        rows and the full dJ @ dq vector.

        Deliberately NOT refactored to share code with `height_barrier_terms`.
        That one runs every cycle for the table and skips two thirds of this
        copying on purpose; folding them together would pay the obstacle cost
        even when no obstacle exists.

        Args:
            q, dq (np.ndarray): joint position/velocity (n,)
            seg_indices (sequence[int]): segment indices from segment_index()

        Returns:
            p (np.ndarray): (m, 3) link-tip positions in base frame
            J_v (np.ndarray): (m, 3, n) linear-velocity Jacobians
            dJdq_v (np.ndarray): (m, 3) dJ @ dq (exact, from KDL)
        """
        m = len(seg_indices)
        p = np.empty((m, 3))
        J_v = np.zeros((m, 3, self.n))
        dJdq_v = np.empty((m, 3))

        ja = self._to_jnt(q)
        qv = self._scratch_qv
        for i in range(self.n):
            qv.q[i] = float(q[i])
            qv.qdot[i] = float(dq[i])

        frame, jac, twist = (self._scratch_frame, self._scratch_jac,
                             self._scratch_twist)
        for k, s in enumerate(seg_indices):
            self._fk.JntToCart(ja, frame, s)
            for r in range(3):
                p[k, r] = frame.p[r]

            self._jac.JntToJac(ja, jac, s)
            for r in range(3):
                for c in range(self.n):
                    J_v[k, r, c] = jac[r, c]

            self._jdot.JntToJacDot(qv, twist, s)
            for r in range(3):
                dJdq_v[k, r] = twist.vel[r]

        return p, J_v, dJdq_v

    def mass(self, q):
        """Return the NxN joint-space inertia matrix M(q)."""
        M = self.kdl.JntSpaceInertiaMatrix(self.n)
        self._dyn.JntToMass(self._to_jnt(q), M)
        return np.array([[M[r, c] for c in range(self.n)] for r in range(self.n)])

    def coriolis(self, q, dq):
        """Return the Coriolis/centrifugal torque C(q, dq) @ dq (n,).

        The velocity-dependent term of ``M ddq + C dq + g = tau``. Needed by
        the table barrier: it inverts the commanded torque to a nominal
        acceleration and back, and dropping C leaves that round trip wrong by
        ``Minv @ C dq`` -- the same order as the dJ@dq term the HOCBF rows go
        out of their way to get exactly right.

        KDL returns the PRODUCT C(q,dq) @ dq, not the matrix, which is all the
        inversion needs and avoids forming an n x n matrix at 1 kHz.
        """
        c = self._scratch_cor
        # Distinct slots: a shared scratch array would make both arguments the
        # same object, so the solver would receive dq as the position too.
        self._dyn.JntToCoriolis(self._to_jnt(q, 0), self._to_jnt(dq, 1), c)
        return np.array([c[i] for i in range(self.n)])

    def gravity(self, q):
        """Return the N-vector of hold-against-gravity torques g(q).

        KDL's JntToGravity already returns the gravity term in the convention
        ``tau_hold = G(q)`` (static form of ``M qdd + C + G = tau``), so it is
        used AS-IS. It was negated here until 2026-07-28 to "match Kinova's
        sensed holding-torque sign" -- but the sensed torque is the external
        LOAD, the opposite of the holding effort (see SENSED_LOAD_SIGN), so
        matching it inverted the model too. Cross-checked against the
        convention-free energy gradient in ``hold_torque_energy``: they now
        agree to <1e-3 Nm on every joint.
        """
        g = self.kdl.JntArray(self.n)
        self._dyn.JntToGravity(self._to_jnt(q), g)
        return np.array([g[i] for i in range(self.n)])

    def hold_torque_energy(self, q, eps=1e-6):
        """Holding torque from the potential-energy gradient, tau = +dU/dq.

        Ground truth for the gravity sign, independent of every KDL/Kinova sign
        convention: U(q) = sum_k m_k * g * z_com,k(q) over the chain's links, so
        the torque that holds the arm static is its gradient. Slow (2n FK
        sweeps) -- startup validation only, never called in the control loop.
        """
        def U(qq):
            ja = self._to_jnt(qq)
            frame = self.kdl.Frame()
            tot = 0.0
            for s in range(self.chain.getNrOfSegments()):
                self._fk.JntToCart(ja, frame, s + 1)
                inertia = self.chain.getSegment(s).getInertia()
                mass = inertia.getMass()
                if mass <= 0.0:
                    continue
                c = inertia.getCOG()
                w = frame * self.kdl.Vector(c[0], c[1], c[2])
                tot += mass * 9.81 * w[2]
            return tot

        g = np.zeros(self.n)
        for i in range(self.n):
            a = np.array(q, dtype=float); a[i] += eps
            b = np.array(q, dtype=float); b[i] -= eps
            g[i] = (U(a) - U(b)) / (2.0 * eps)
        return g

    def centerline(self, q):
        """De-duplicated world positions of segment origins, base->tip.

        The polyline connecting these points is the capsule model's spine;
        consecutive points define one link capsule each.

        Returns an (m, 3) array. The de-duplication threshold compares
        consecutive segment ORIGINS, whose separation is fixed by the URDF joint
        offsets and does not depend on q, so `m` is constant for a given chain
        -- but `_capsule_pairs` keys its cache on it rather than assuming so.
        """
        ja = self._to_jnt(q)
        frame = self.kdl.Frame()
        pts = []
        for s in range(self.chain.getNrOfSegments() + 1):
            self._fk.JntToCart(ja, frame, s)
            p = np.array([frame.p[i] for i in range(3)])
            if not pts or np.linalg.norm(p - pts[-1]) > 0.02:
                pts.append(p)
        return np.asarray(pts)

    def _capsule_pairs(self, ncap):
        """Cached (i, j) index arrays for the non-adjacent capsule pairs.

        Adjacent capsules share a joint and always touch, so only j >= i + 2 is
        checked. The pair set depends solely on the capsule COUNT, so it is
        built once and reused every cycle instead of being re-enumerated.
        """
        idx = self._pair_cache.get(ncap)
        if idx is None:
            pairs = [(i, j) for i in range(ncap) for j in range(i + 2, ncap)]
            if pairs:
                ii, jj = (np.array(x, dtype=np.intp) for x in zip(*pairs))
            else:
                ii = jj = np.empty(0, dtype=np.intp)
            idx = (ii, jj)
            self._pair_cache[ncap] = idx
        return idx

    def collision_cost(self, q):
        """Quadratic-hinge self-collision potential over non-adjacent capsules.

        Rises smoothly from zero as any non-adjacent link pair's surface
        clearance drops below COLLISION_MARGIN. Adjacent capsules share a joint
        (always touch), so they are skipped.

        Vectorised over all pairs at once (see `_segment_distance_batch`); the
        equivalent Python double loop was 66% of the control cycle's compute.
        """
        pts = self.centerline(q)
        if len(pts) < 3:
            return 0.0
        P, Q = pts[:-1], pts[1:]
        ii, jj = self._capsule_pairs(len(P))
        if ii.size == 0:
            return 0.0
        surf = (_segment_distance_batch(P[ii], Q[ii], P[jj], Q[jj])
                - 2.0 * self.CAPSULE_RADIUS)
        # Quadratic hinge: only clearances inside the margin contribute.
        v = np.maximum(0.0, self.COLLISION_MARGIN - surf)
        return 0.5 * float(v @ v)

    def collision_gradient(self, q, eps=1e-4):
        """-d(collision_cost)/dq direction points toward greater clearance.

        Finite differences: costs n+1 centerline evaluations -- 1547 us on
        REAL-1, longer than a whole control cycle.

        NOT USED BY THE CONTROL LOOP. The loop sweeps the same finite difference
        one column per cycle instead (see COLLIDE_EPS), because decimating a
        call this heavy bounds its average cost but not its worst cycle. Kept
        for offline validation and tests, where taking it in one go is fine.
        """
        g = np.zeros(self.n)
        h0 = self.collision_cost(q)
        for i in range(self.n):
            dq = q.copy()
            dq[i] += eps
            g[i] = (self.collision_cost(dq) - h0) / eps
        return g


# =============================================================================
# Segment distance (capsule model)
# =============================================================================
def _segment_distance(p1, q1, p2, q2):
    """Minimum distance between segments [p1,q1] and [p2,q2].

    Ericson, Real-Time Collision Detection (ClosestPtSegmentSegment).
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-9
    if a <= eps and e <= eps:                 # both degenerate to points
        return float(np.linalg.norm(r))
    if a <= eps:                              # first segment is a point
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:                          # second segment is a point
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _segment_distance_batch(P1, Q1, P2, Q2):
    """Vectorised `_segment_distance` over K segment pairs at once.

    Same algorithm (Ericson, ClosestPtSegmentSegment), with every scalar branch
    rewritten as a `np.where` select so all K pairs are solved in one pass.

    This exists because the scalar version, called from a Python double loop,
    was the single most expensive thing in the 1 kHz control cycle: 195.6 us
    per `collision_cost`, i.e. ~66% of all control compute and ~16% of the
    whole 1.2 ms cycle (measured on REAL-1, 2026-08-26). Batched, the same
    query costs 59.3 us and returns bit-identical costs.

    Args:
        P1, Q1 (np.ndarray): (K, 3) endpoints of the first segment of each pair
        P2, Q2 (np.ndarray): (K, 3) endpoints of the second segment

    Returns:
        np.ndarray: (K,) minimum distance for each pair
    """
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d1, r)
    eps = 1e-9

    # Guarded denominators: the np.where SELECTS the right branch, but both
    # branches are evaluated, so every divisor has to be finite everywhere or
    # numpy warns (and propagates NaN into the selected branch's neighbours).
    a_ok = np.where(a > eps, a, 1.0)
    e_ok = np.where(e > eps, e, 1.0)
    denom = a * e - b * b
    denom_ok = np.where(denom > eps, denom, 1.0)

    # General case: both segments non-degenerate.
    s = np.where(denom > eps, np.clip((b * f - c * e) / denom_ok, 0.0, 1.0), 0.0)
    t = (b * s + f) / e_ok
    # Re-clamp s wherever the unclamped t left [0, 1] (the scalar version's
    # `if t < 0` / `elif t > 1` arms).
    s = np.where(t < 0.0, np.clip(-c / a_ok, 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / a_ok, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)

    # Degenerate arms, applied last so they win over the general case.
    first_pt = a <= eps                       # segment 1 is a point
    s = np.where(first_pt, 0.0, s)
    t = np.where(first_pt, np.clip(f / e_ok, 0.0, 1.0), t)
    second_pt = (e <= eps) & ~first_pt        # segment 2 is a point
    t = np.where(second_pt, 0.0, t)
    s = np.where(second_pt, np.clip(-c / a_ok, 0.0, 1.0), s)
    both_pt = (a <= eps) & (e <= eps)         # both degenerate
    s = np.where(both_pt, 0.0, s)
    t = np.where(both_pt, 0.0, t)

    diff = (P1 + s[:, None] * d1) - (P2 + t[:, None] * d2)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))
