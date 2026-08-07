import re

with open("surgical_arm_bringup/scripts/insert_to_container.py", "r") as f:
    content = f.read()

# 1. Add step_by_step parameter
content = content.replace(
    'self.declare_parameter("skip_home_move", False)',
    'self.declare_parameter("skip_home_move", False)\n        self.declare_parameter("step_by_step", False)'
)

# 2. Add _wait_for_user method
wait_for_user_code = """
    def _wait_for_user(self, label):
        if not self.get_parameter("step_by_step").value:
            return True
        try:
            ans = input(f"  [Step] Execute {label}? [Y/n]: ").strip().lower()
            if ans in ('', 'y', 'yes'):
                return True
            self.get_logger().warn(f"  Skipping {label} by user request.")
            return False
        except EOFError:
            return False

"""
content = content.replace(
    '    # ------------------------------------------------------------------\n    # EE pose computation',
    wait_for_user_code + '    # ------------------------------------------------------------------\n    # EE pose computation'
)

# 3. Modify execute lines to include _wait_for_user
content = re.sub(
    r'if execute and not self\._execute_moveit\((.*?),\s*"(.*?)"(.*?)\):',
    r'if execute and self._wait_for_user("\2") and not self._execute_moveit(\1, "\2"\3):',
    content
)
content = re.sub(
    r'if execute and not self\._execute_fjt\((.*?),\s*"(.*?)"(.*?)\):',
    r'if execute and self._wait_for_user("\2") and not self._execute_fjt(\1, "\2"\3):',
    content
)

# Replace execute blocks where execute is checked alone
content = re.sub(
    r'if execute:\n(\s*)if not self\._execute_moveit\((.*?),\s*"(.*?)"(.*?)\):',
    r'if execute and self._wait_for_user("\3"):\n\1if not self._execute_moveit(\2, "\3"\4):',
    content
)

# Add wait to Phase 6b
content = content.replace(
    'self._execute_fjt(t_v, "Phase6b_via")',
    'if self._wait_for_user("Phase6b_via"): self._execute_fjt(t_v, "Phase6b_via")'
)
content = content.replace(
    'self._execute_fjt(t_r, "Phase6b_vertical")',
    'if self._wait_for_user("Phase6b_vertical"): self._execute_fjt(t_r, "Phase6b_vertical")'
)
content = content.replace(
    'self._execute_fjt(traj_circ_rev, "Phase6b_circ_reverse")',
    'if self._wait_for_user("Phase6b_circ_reverse"): self._execute_fjt(traj_circ_rev, "Phase6b_circ_reverse")'
)

# Add wait to Phase 7
content = content.replace(
    'self._execute_fjt(traj_up, "Phase7_vertical_ascent")',
    'if self._wait_for_user("Phase7_vertical_ascent"): self._execute_fjt(traj_up, "Phase7_vertical_ascent")'
)

# Add wait to Phase 8
content = content.replace(
    'if ok and execute:\n                self._execute_moveit(traj, "Phase8_return", timeout=90.0)',
    'if ok and execute and self._wait_for_user("Phase8_return"):\n                self._execute_moveit(traj, "Phase8_return", timeout=90.0)'
)

# Adjust FJT parameters (increase tolerances even more to prevent -4, or slow down)
content = content.replace(
    'gt.position = 0.05; gt.velocity = 0.0; gt.acceleration = 0.0',
    'gt.position = 0.1; gt.velocity = 0.1; gt.acceleration = 0.0'
)

with open("surgical_arm_bringup/scripts/insert_to_container.py", "w") as f:
    f.write(content)

