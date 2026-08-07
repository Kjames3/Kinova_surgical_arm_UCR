import os
import re

def refactor_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Imports
    if "from robot_model_parser import get_robot_info" not in content:
        content = content.replace("import math", "import math\nfrom robot_model_parser import get_robot_info")

    # 2. Remove globals
    content = re.sub(r'_GEN3_JOINTS\s*=\s*\[.*?\]', '', content, flags=re.DOTALL)
    content = re.sub(r'_GEN3_HOME_JOINTS\s*=\s*\{.*?\}', '', content, flags=re.DOTALL)
    content = re.sub(r'_ASSEMBLY_TIP_OFFSET\s*=\s*\{.*?\}', '', content, flags=re.DOTALL)
    content = re.sub(r'MAX_JOINT_VEL_RAD_S\s*=\s*[0-9.]+', '', content)
    content = re.sub(r'MAX_JOINT_ACC_RAD_S2\s*=\s*[0-9.]+', '', content)

    # 3. Add to __init__
    init_hook = 'self.declare_parameter("n_circ_via_points", 1)'
    new_init_params = """self.declare_parameter("n_circ_via_points", 1)
        
        self.arm_joint_names = self.declare_parameter("arm_joint_names", [
            "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6", "joint_7"
        ]).value
        
        c_joints, m_vel, h_joints = get_robot_info(self)
        self.continuous_joints = c_joints if c_joints else {"joint_1", "joint_3", "joint_5", "joint_7"}
        self.home_joints = h_joints if h_joints else {
            "joint_1":  0.0000, "joint_2": -0.3049, "joint_3": -3.1416,
            "joint_4": -1.6607, "joint_5":  0.0000, "joint_6": -1.7928, "joint_7": -0.0006,
        }
        self.max_joint_vel = m_vel if m_vel else {j: 0.8 for j in self.arm_joint_names}
        self.max_joint_acc = {j: 0.4 for j in self.arm_joint_names}
        self.fjt_topic = self.declare_parameter("fjt_topic", "/joint_trajectory_controller/follow_joint_trajectory").value
        self._assembly_tip_offset = None
"""
    if 'self.arm_joint_names = self.declare_parameter' not in content:
        content = content.replace(init_hook, new_init_params)

    # 4. _execute_fjt topic
    content = content.replace('"/joint_trajectory_controller/follow_joint_trajectory"', 'self.fjt_topic')

    # 5. replace _GEN3_JOINTS with self.arm_joint_names
    content = content.replace('_GEN3_JOINTS', 'self.arm_joint_names')

    # 6. replace _GEN3_HOME_JOINTS with self.home_joints
    content = content.replace('_GEN3_HOME_JOINTS', 'self.home_joints')

    # 7. replace continuous joints
    content = re.sub(r'_CONTINUOUS_JOINTS\s*=\s*\{.*?\}', '', content)
    content = content.replace('_CONTINUOUS_JOINTS', 'self.continuous_joints')

    # 8. Clamping logic
    def clamp_repl(match):
        return """
    def _clamp_traj(self, traj):
        pts = traj.joint_trajectory.points
        names = traj.joint_trajectory.joint_names
        for i, pt in enumerate(pts):
            if not pt.velocities and not pt.accelerations:
                continue
            worst = 1.0
            for j, v in enumerate(pt.velocities):
                jname = names[j] if j < len(names) else self.arm_joint_names[0]
                max_v = self.max_joint_vel.get(jname, 0.8)
                if abs(v) > max_v:
                    worst = max(worst, abs(v) / max_v)
            for j, a in enumerate(pt.accelerations):
                jname = names[j] if j < len(names) else self.arm_joint_names[0]
                max_a = self.max_joint_acc.get(jname, 0.4)
                if abs(a) > max_a:
                    worst = max(worst, (abs(a) / max_a) ** 0.5)"""

    content = re.sub(r'\@staticmethod\s*def _clamp_traj\(traj\):.*?if abs\(a\) > MAX_JOINT_ACC_RAD_S2:\s*worst = max\(worst, \(abs\(a\) / MAX_JOINT_ACC_RAD_S2\) \*\* 0\.5\)', clamp_repl, content, flags=re.DOTALL)
    
    content = content.replace('self._clamp_traj(traj)', 'self._clamp_traj(traj)') # just to be sure it takes self now

    # 9. TF Offset logic in _ee_for_tip and _ee_circ_arc
    # Need to replace `_ASSEMBLY_TIP_OFFSET["x"]` with `self._assembly_tip_offset["x"]`
    content = content.replace('_ASSEMBLY_TIP_OFFSET', 'self._assembly_tip_offset')

    # Add offset fetcher
    fetch_tf_func = """
    def _get_tip_offset(self):
        if self._assembly_tip_offset is None:
            tf = self._get_tf(self.get_parameter("ee_link").value, self.get_parameter("tip_link").value)
            if tf:
                self._assembly_tip_offset = {
                    "x": tf.transform.translation.x,
                    "y": tf.transform.translation.y,
                    "z": tf.transform.translation.z
                }
            else:
                self.get_logger().warn("Could not lookup tf ee_link -> tip_link, using default offset.")
                self._assembly_tip_offset = {"x": -0.027, "y": 0.0, "z": -0.414}
        return self._assembly_tip_offset
"""
    # Insert it before `def _get_tf`
    if 'def _get_tip_offset' not in content:
        content = content.replace('def _get_tf', fetch_tf_func + '\n    def _get_tf')

    # Now make sure `self._get_tip_offset()` is called where needed.
    content = content.replace('self._assembly_tip_offset["x"]', 'self._get_tip_offset()["x"]')
    content = content.replace('self._assembly_tip_offset["y"]', 'self._get_tip_offset()["y"]')
    content = content.replace('self._assembly_tip_offset["z"]', 'self._get_tip_offset()["z"]')

    with open(filepath, 'w') as f:
        f.write(content)

refactor_file("insertion.py")
refactor_file("insert_to_container.py")
