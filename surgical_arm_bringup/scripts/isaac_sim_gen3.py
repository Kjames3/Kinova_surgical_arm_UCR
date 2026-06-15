"""
Isaac Sim 4.x startup script for Kinova Gen3 7DOF + Robotiq 2F-140.

Before first run (and after any URDF/xacro change), regenerate the USD:
  cd ~/workspace/ros2_kortex_ws
  source install/setup.bash
  xacro src/ros2_kortex/kortex_description/robots/gen3.xacro \
      arm:=gen3 dof:=7 gripper:=thesis_ee sim_isaac:=true \
      robot_ip:=xxx use_fake_hardware:=true > /tmp/gen3_isaac.urdf
  ~/isaacsim/python.sh src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_import_urdf.py

Then start the sim:
  ~/isaacsim/python.sh src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_gen3.py

Symptom of a stale/empty USD: Isaac prints
  [omni.physx.tensors.plugin] Pattern '/gen3' did not match any rigid bodies
  [omni.physx.tensors.plugin] Provided pattern list did not match any articulations
and /isaac_joint_states never publishes, which in turn hangs load_controller
on the ROS side. Fix: re-run isaac_sim_import_urdf.py.

Topics (match kortex.ros2_control.xacro defaults):
  Published:  /isaac_joint_states   (sensor_msgs/JointState)
  Subscribed: /isaac_joint_commands (sensor_msgs/JointState)
  Published:  /clock                (rosgraph_msgs/Clock)
"""

import os
import sys

import numpy as np
from isaacsim import SimulationApp

USD_PATH = os.path.expanduser("~/isaacsim/gen3_thesis_ee.usd")
ROBOT_PRIM = "/gen3"

CONFIG = {"renderer": "RaytracedLighting", "headless": False}

simulation_app = SimulationApp(CONFIG)

import carb
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, prims, stage

extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)

if not os.path.exists(USD_PATH):
    carb.log_error(
        f"Robot USD not found at {USD_PATH}. "
        "Import the URDF via Isaac Utils > URDF Importer first (see script docstring)."
    )
    simulation_app.close()
    sys.exit(1)

# Dome light — the stage is otherwise unlit and the robot appears black.
prims.create_prim(
    "/World/DomeLight",
    "DomeLight",
    attributes={"inputs:intensity": 1000.0, "inputs:texture:format": "latlong"},
)

# Load the robot into the stage
prims.create_prim(
    ROBOT_PRIM,
    "Xform",
    position=np.array([0.0, 0.0, 0.0]),
    usd_path=USD_PATH,
)

simulation_app.update()

# Find the articulation root — its path may differ from ROBOT_PRIM depending
# on how the URDF Importer named things, and the ArticulationController needs
# the exact path (pattern matching is not recursive).
import omni.usd  # noqa: E402
from pxr import UsdGeom, Gf, UsdPhysics  # noqa: E402

_stage = omni.usd.get_context().get_stage()

# Table: 2m × 2m × 0.05m box, surface at z = -0.03 m (matches setup_planning_scene.py)
_table = UsdGeom.Cube.Define(_stage, "/World/Table")
_table.CreateSizeAttr(1.0)
UsdGeom.XformCommonAPI(_table).SetTranslate(Gf.Vec3d(0.0, 0.0, -0.055))
UsdGeom.XformCommonAPI(_table).SetScale(Gf.Vec3f(2.0, 2.0, 0.05))
UsdPhysics.CollisionAPI.Apply(_table.GetPrim())

# Glass container: cylinder r=0.04m h=0.10m, base on table surface
_container = UsdGeom.Cylinder.Define(_stage, "/World/GlassContainer")
_container.CreateRadiusAttr(0.04)
_container.CreateHeightAttr(0.10)
UsdGeom.XformCommonAPI(_container).SetTranslate(Gf.Vec3d(0.50, -0.20, 0.02))
UsdPhysics.CollisionAPI.Apply(_container.GetPrim())

_art_paths = [p.GetPath().pathString for p in _stage.Traverse()
              if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
if not _art_paths:
    carb.log_error(
        f"No ArticulationRootAPI found under {ROBOT_PRIM}. "
        f"Re-run isaac_sim_import_urdf.py to regenerate the USD."
    )
    simulation_app.close()
    sys.exit(1)
ARTICULATION_PATH = _art_paths[0]
print(f"[isaac_sim_gen3] articulation root at {ARTICULATION_PATH}")

try:
    og.Controller.edit(
        {"graph_path": "/ROS2ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
                ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("ROS2Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", ARTICULATION_PATH),
                ("PublishJointState.inputs:topicName", "isaac_joint_states"),
                ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ARTICULATION_PATH)]),
            ],
        },
    )
except Exception as e:
    carb.log_error(f"OmniGraph setup failed: {e}")
    simulation_app.close()
    sys.exit(1)

simulation_app.update()
simulation_context.initialize_physics()
simulation_context.play()

# Preposition arm to Kinova's SRDF "Home" pose — approximates pen-down so
# insert_to_container's startup orientation check passes. Physical-robot workflow is
# unchanged (user still poses that manually); this only affects the sim stage.
from isaacsim.core.prims import SingleArticulation  # noqa: E402
import numpy as _np  # noqa: E402

_home_pose = {
    "joint_1": 0.0,
    "joint_2": 0.26,
    "joint_3": 3.14,
    "joint_4": -2.0,
    "joint_5": 0.0,
    "joint_6": -0.93,   # tuned so assembly tip points along world -Z (joint_6 limit is ±2.23 rad)
    "joint_7": 1.57,    # was 1.57; -90° to correct tool yaw for thesis_ee vs old robotiq_2f_140
}
try:
    _robot = SingleArticulation(prim_path=ARTICULATION_PATH, name="gen3")
    _robot.initialize()
    _positions = _np.array([_home_pose.get(n, 0.0) for n in _robot.dof_names])
    _robot.set_joint_positions(_positions)
    print(f"[isaac_sim_gen3] set initial joint positions: "
          f"{dict(zip(_robot.dof_names, _positions.round(3)))}")
except Exception as _exc:
    carb.log_warn(f"could not set initial joint positions: {_exc}")

print("Isaac Sim running. ROS 2 bridge active on /isaac_joint_states and /isaac_joint_commands.")
print("Start the ROS 2 stack in a separate terminal:")
print("  ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config isaac_sim.launch.py")

while simulation_app.is_running():
    simulation_context.step(render=True)

simulation_context.stop()
simulation_app.close()
