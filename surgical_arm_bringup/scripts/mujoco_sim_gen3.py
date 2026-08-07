"""
MuJoCo 3.x startup script for Kinova Gen3 7DOF + thesis end-effector.

Before first run (and after any URDF/xacro change), regenerate the MJCF:
  cd ~/workspace/ros2_kortex_ws
  source install/setup.bash
  python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/mujoco_import_urdf.py

Then start the sim:
  python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/mujoco_sim_gen3.py

Symptom of a stale/missing MJCF: this script exits with
  "Robot MJCF not found at ~/mujoco_models/gen3_thesis_ee.xml"
or MuJoCo raises a compile error on load. Either way /isaac_joint_states never
publishes, which in turn hangs load_controller on the ROS side (the
topic_based_ros2_control system waits forever for its first joint state).
Fix: re-run mujoco_import_urdf.py.

This is the *standalone* variant: it owns the stepping loop and drives a
passive viewer (mujoco.viewer.launch_passive). See mujoco_sim_gen3_gui.py for
the variant that hands the loop to MuJoCo's managed interactive GUI.

Topics (match kortex.ros2_control.xacro defaults):
  Published:  /isaac_joint_states   (sensor_msgs/JointState)
  Subscribed: /isaac_joint_commands (sensor_msgs/JointState)
  Published:  /clock                (rosgraph_msgs/Clock)
"""

import os
import sys
import time

import mujoco
import mujoco.viewer
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

MODEL_PATH = os.path.expanduser("~/mujoco_models/gen3_thesis_ee.xml")

# Kinova SRDF "Home" pose. Also baked into the MJCF as the <keyframe name="home">
# by mujoco_import_urdf.py; this dict is only the fallback for an MJCF that
# predates the keyframe. It approximates the pen-down/insert orientation so
# insert_to_container's startup orientation check passes. The physical-robot
# workflow is unchanged (the user still poses that manually).
HOME_POSE = {
    "joint_1": 0.0,
    "joint_2": 0.26,
    "joint_3": 3.14,
    "joint_4": -2.0,
    "joint_5": 0.0,
    "joint_6": -0.93,   # tuned so assembly tip points along world -Z (joint_6 limit is ±2.23 rad)
    "joint_7": 1.57,    # corrects tool yaw for thesis_ee vs old robotiq_2f_140
}

# The ROS graph still speaks "isaac" because topic_based_ros2_control and the
# vendor kortex.ros2_control.xacro hard-code these names as their defaults.
# Renaming them would mean touching the vendor xacro, so the simulator changed
# and the topic names did not. They are exposed as node parameters below so a
# future rename needs no code edit.
DEFAULT_JOINT_STATES_TOPIC = "isaac_joint_states"
DEFAULT_JOINT_COMMANDS_TOPIC = "isaac_joint_commands"

SIM_TIMESTEP = 0.002        # s — also set in the MJCF; we re-assert it here
PUBLISH_RATE_HZ = 100.0     # joint_states + /clock publish rate


def load_model(model_path=MODEL_PATH):
    """Load the MJCF, or die with the same troubleshooting hint as the docstring."""
    if not os.path.exists(model_path):
        print(
            f"ERROR: Robot MJCF not found at {model_path}.\n"
            f"       Generate it first:\n"
            f"         python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/"
            f"scripts/mujoco_import_urdf.py\n"
            f"       Without it the ROS 2 bridge never publishes joint states and\n"
            f"       load_controller hangs on the ROS side.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as exc:  # MuJoCo raises plain ValueError on a bad compile
        print(
            f"ERROR: MuJoCo failed to compile {model_path}: {exc}\n"
            f"       The MJCF is likely stale or truncated — re-run "
            f"mujoco_import_urdf.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    model.opt.timestep = SIM_TIMESTEP
    data = mujoco.MjData(model)
    return model, data


def reset_to_home(model, data):
    """Put the arm in the Home pose, preferring the MJCF <keyframe name="home">."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("[mujoco_sim_gen3] reset to MJCF keyframe 'home'")
    else:
        # Fallback for an MJCF without the keyframe: write qpos by joint name.
        # qposadr is required — qpos index != joint index for anything that is
        # not a 1-DOF hinge (a free joint alone eats 7 qpos slots).
        mujoco.mj_resetData(model, data)
        for name, value in HOME_POSE.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                print(f"[mujoco_sim_gen3] WARNING: no joint '{name}' in model")
                continue
            data.qpos[model.jnt_qposadr[jid]] = value
        print("[mujoco_sim_gen3] no 'home' keyframe in MJCF — set qpos by joint name")

    # Hold the home pose: position actuators would otherwise drive every joint
    # to ctrl=0 on the first step and the arm would collapse.
    for act_id in range(model.nu):
        jid = _actuator_joint_id(model, act_id)
        if jid >= 0:
            data.ctrl[act_id] = data.qpos[model.jnt_qposadr[jid]]

    mujoco.mj_forward(model, data)


def _actuator_joint_id(model, act_id):
    """Joint id driven by an actuator, or -1 if it is not a joint transmission."""
    if model.actuator_trntype[act_id] != mujoco.mjtTrn.mjTRN_JOINT:
        return -1
    return int(model.actuator_trnid[act_id, 0])


class Gen3MujocoBridge(Node):
    """ROS 2 <-> MuJoCo bridge for the Gen3.

    Replaces the Isaac OmniGraph (ROS2PublishJointState / ROS2SubscribeJointState
    / IsaacArticulationController / ROS2PublishClock) with plain rclpy. Shared by
    both the standalone and the GUI script so the wire format cannot diverge.

    The node does not own a thread: call publish_state() from whatever loop is
    stepping physics.
    """

    def __init__(self, model, data):
        super().__init__("gen3_mujoco_bridge")
        self.model = model
        self.data = data

        # Parameterised so the topic names can be flipped without a code edit
        # (see DEFAULT_* above for why they still say "isaac").
        self.declare_parameter("joint_states_topic", DEFAULT_JOINT_STATES_TOPIC)
        self.declare_parameter("joint_commands_topic", DEFAULT_JOINT_COMMANDS_TOPIC)
        states_topic = self.get_parameter("joint_states_topic").value
        commands_topic = self.get_parameter("joint_commands_topic").value

        # Actuated joints, in actuator order. Cache the address indirection once:
        # qpos/qvel indices are NOT joint indices whenever a joint is not a
        # 1-DOF hinge, so everything downstream goes through jnt_qposadr/jnt_dofadr.
        self._joint_names = []
        self._qposadr = []
        self._dofadr = []
        for act_id in range(model.nu):
            jid = _actuator_joint_id(model, act_id)
            if jid < 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name is None:
                continue
            self._joint_names.append(name)
            self._qposadr.append(int(model.jnt_qposadr[jid]))
            self._dofadr.append(int(model.jnt_dofadr[jid]))

        self._js_pub = self.create_publisher(JointState, states_topic, 10)
        self._clock_pub = self.create_publisher(Clock, "/clock", 10)
        self._cmd_sub = self.create_subscription(
            JointState, commands_topic, self._on_command, 10
        )

        self._publish_period = 1.0 / PUBLISH_RATE_HZ
        self._next_publish_time = 0.0
        self._unknown_warned = set()

        self.get_logger().info(
            f"bridge up — publishing {states_topic} + /clock, "
            f"subscribing {commands_topic}"
        )
        self.get_logger().info(
            f"actuated joints: {', '.join(self._joint_names)}"
        )

    def _on_command(self, msg: JointState):
        """Map incoming joint names -> actuator ids BY NAME and write data.ctrl.

        Name-based mapping is mandatory: MoveIt / topic_based_ros2_control order
        joints by the URDF/controller config, which is not the model's actuator
        order. Index-based mapping silently drives the wrong joints.

        Writing data.ctrl from the executor thread is safe enough here: each
        element is an independent float and a stale-by-one-step command is
        indistinguishable from normal command latency.
        """
        for i, name in enumerate(msg.name):
            if i >= len(msg.position):
                break
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id < 0:
                if name not in self._unknown_warned:
                    self._unknown_warned.add(name)
                    self.get_logger().warning(
                        f"no actuator named '{name}' in the MJCF — ignoring "
                        f"(warned once per name)"
                    )
                continue
            self.data.ctrl[act_id] = msg.position[i]

    def publish_state(self, force=False):
        """Publish joint states and /clock, throttled to PUBLISH_RATE_HZ of SIM time.

        /clock is the heartbeat of the whole stack: the launch file runs with
        use_sim_time:=true, so every node's timers, TF buffer and action servers
        are driven from it. If this stops publishing, MoveIt and the controllers
        stall silently — they are not "crashed", they are waiting for time.
        """
        sim_time = self.data.time
        if not force and sim_time < self._next_publish_time:
            return
        self._next_publish_time = sim_time + self._publish_period

        stamp = self._sim_stamp(sim_time)

        clock_msg = Clock()
        clock_msg.clock = stamp
        self._clock_pub.publish(clock_msg)

        js = JointState()
        # Stamp with SIM time, not wall time — everything downstream is on
        # use_sim_time and would reject/extrapolate wall-stamped data.
        js.header.stamp = stamp
        js.name = list(self._joint_names)
        js.position = [float(self.data.qpos[a]) for a in self._qposadr]
        js.velocity = [float(self.data.qvel[a]) for a in self._dofadr]
        js.effort = [float(self.data.qfrc_actuator[a]) for a in self._dofadr]
        self._js_pub.publish(js)

    @staticmethod
    def _sim_stamp(sim_time):
        stamp = TimeMsg()
        stamp.sec = int(sim_time)
        stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
        return stamp


def main():
    model, data = load_model()
    reset_to_home(model, data)

    rclpy.init()
    node = Gen3MujocoBridge(model, data)
    node.publish_state(force=True)

    print("=" * 60)
    print("MuJoCo running. ROS 2 bridge active on "
          "/isaac_joint_states and /isaac_joint_commands.")
    print("Start the ROS 2 stack in a separate terminal:")
    print("  source ~/workspace/ros2_kortex_ws/install/setup.bash")
    print("  ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config "
          "mujoco_sim.launch.py")
    print("=" * 60)

    try:
        # launch_passive hands US the loop — the direct analog of the standalone
        # Isaac launcher that called simulation_context.step() itself.
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall_start = time.perf_counter()
            sim_start = data.time
            while viewer.is_running():
                mujoco.mj_step(model, data)
                node.publish_state()

                # Non-blocking spin: the physics loop must never wait on ROS.
                rclpy.spin_once(node, timeout_sec=0.0)

                viewer.sync()

                # Real-time pace. Sleeping on the *accumulated* error rather than
                # a fixed dt keeps sim time from drifting behind wall time.
                lag = (data.time - sim_start) - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        print("\n[mujoco_sim_gen3] interrupted — shutting down")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# Guarded so mujoco_sim_gen3_gui.py can import Gen3MujocoBridge without
# starting a second simulation.
if __name__ == "__main__":
    main()
