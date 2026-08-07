"""
MuJoCo 3.x GUI-mode script for Kinova Gen3 7DOF + thesis end-effector.

Run it with the managed interactive viewer (MuJoCo owns the loop):
  python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/mujoco_sim_gen3_gui.py

This is the analog of the old "load into an already-running Isaac GUI session"
script: we do NOT drive the stepping loop here. mujoco.viewer.launch() runs its
own physics/render threads and the GUI controls play/pause/reset/step; the ROS 2
bridge attaches to that loop through MuJoCo's global control callback and a ROS
executor on a daemon thread.

Use mujoco_sim_gen3.py instead if you want the script to own the loop (passive
viewer, deterministic real-time pacing) — that is the better choice for
scripted/automated runs. Use this one for hands-on inspection, since the managed
viewer lets you drag bodies, toggle visualisation flags and pause physics.

Requires the MJCF at ~/mujoco_models/gen3_thesis_ee.xml. If it is missing, run
  python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/mujoco_import_urdf.py
first — without the model nothing publishes on /isaac_joint_states, and
load_controller then hangs on the ROS side waiting for a first joint state.

Topics:
  Published:  /isaac_joint_states   (sensor_msgs/JointState)
  Subscribed: /isaac_joint_commands (sensor_msgs/JointState)
  Published:  /clock                (rosgraph_msgs/Clock)
"""

import os
import sys
import threading

import mujoco
import mujoco.viewer
import rclpy
from rclpy.executors import MultiThreadedExecutor

# Both scripts live in the same directory, but this one is usually invoked by
# absolute path from a shell whose CWD is elsewhere, so a plain
# `import mujoco_sim_gen3` can miss. Putting the script's own directory on
# sys.path makes the import work regardless of CWD. The bridge deliberately
# lives in one place so the two entry points cannot drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_sim_gen3 import (  # noqa: E402
    Gen3MujocoBridge,
    load_model,
    reset_to_home,
)


def main():
    model, data = load_model()
    reset_to_home(model, data)

    rclpy.init()
    node = Gen3MujocoBridge(model, data)
    node.publish_state(force=True)

    # The managed viewer owns the physics thread, so the executor gets its own
    # daemon thread instead of being spun inline. Daemon so a viewer-window
    # close tears it down with the process.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # mj_step invokes the global control callback once per step, which is our
    # only hook into a loop we do not own. Publishing from here keeps sim time,
    # joint states and /clock in lockstep with the physics the GUI is running —
    # including while the user single-steps or slows the sim down.
    # publish_state() self-throttles on SIM time, so it is cheap at every step.
    def _control_callback(_model, _data):
        node.publish_state()

    mujoco.set_mjcb_control(_control_callback)

    print("=" * 60)
    print("Gen3 ROS 2 bridge configured successfully.")
    print("Press the play/pause control in the MuJoCo viewer to run physics.")
    print("Then in a new terminal run:")
    print("  source ~/workspace/ros2_kortex_ws/install/setup.bash")
    print("  ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config "
          "mujoco_sim.launch.py")
    print("=" * 60)

    try:
        # Blocking: returns when the user closes the viewer window.
        mujoco.viewer.launch(model, data)
    except KeyboardInterrupt:
        print("\n[mujoco_sim_gen3_gui] interrupted — shutting down")
    finally:
        # Clear the global callback first: it holds a reference to the node and
        # would fire against a destroyed node if physics ticked during teardown.
        mujoco.set_mjcb_control(None)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
