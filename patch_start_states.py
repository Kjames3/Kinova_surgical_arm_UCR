import re

with open("surgical_arm_bringup/scripts/insert_to_container.py", "r") as f:
    content = f.read()

# Fix Phase 3b start state
phase3_target = """            req_via = self._build_pilz_lin(*ee_via_circ, q_via, vel_scale)
            ok_via, traj_via = self._plan(req_via)
            req_end = self._build_pilz_lin(*ee_end_circ, q_tilted, vel_scale)
            ok_end, traj_end = self._plan(req_end)"""

phase3_replace = """            req_via = self._build_pilz_lin(*ee_via_circ, q_via, vel_scale)
            ok_via, traj_via = self._plan(req_via)
            req_end = self._build_pilz_lin(*ee_end_circ, q_tilted, vel_scale)
            if ok_via:
                rs = req_end.start_state
                rs.is_diff = True
                rs.joint_state.name = traj_via.joint_trajectory.joint_names
                rs.joint_state.position = traj_via.joint_trajectory.points[-1].positions
            ok_end, traj_end = self._plan(req_end)"""

content = content.replace(phase3_target, phase3_replace)

# Fix Phase 6b start state
phase6_target = """            req_via = self._build_pilz_lin(*ee_via_circ, q_via, vel_scale)
            req_vert= self._build_pilz_lin(*ee_start_circ, q_vertical, vel_scale)
            ok_v, t_v = self._plan(req_via)
            ok_r, t_r = self._plan(req_vert)"""

phase6_replace = """            req_via = self._build_pilz_lin(*ee_via_circ, q_via, vel_scale)
            req_vert= self._build_pilz_lin(*ee_start_circ, q_vertical, vel_scale)
            ok_v, t_v = self._plan(req_via)
            if ok_v:
                rs = req_vert.start_state
                rs.is_diff = True
                rs.joint_state.name = t_v.joint_trajectory.joint_names
                rs.joint_state.position = t_v.joint_trajectory.points[-1].positions
            ok_r, t_r = self._plan(req_vert)"""

content = content.replace(phase6_target, phase6_replace)

with open("surgical_arm_bringup/scripts/insert_to_container.py", "w") as f:
    f.write(content)

