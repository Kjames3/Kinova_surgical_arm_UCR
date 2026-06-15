"""
Isaac Sim 4.x GUI-mode setup script for Kinova Gen3 7DOF + Robotiq 2F-140.

Run via the full Isaac Sim launcher (ROS 2 bridge already loaded):
  ~/isaacsim/isaac-sim.sh --exec ~/workspace/ros2_kortex_ws/src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_gen3_gui.py

The script loads the robot USD and sets up the ROS 2 OmniGraph.
Press Play in the Isaac Sim GUI to start simulation and activate the bridge.

Topics:
  Published: /isaac_joint_states    (sensor_msgs/JointState)
  Subscribed: /isaac_joint_commands (sensor_msgs/JointState)
  Published:  /clock                (rosgraph_msgs/Clock)
"""

import os
import numpy as np
import carb
import omni.graph.core as og
import omni.usd
import usdrt.Sdf
from isaacsim.core.utils import extensions, prims
from pxr import UsdGeom, Gf, UsdPhysics

USD_PATH = os.path.expanduser("~/isaacsim/gen3_thesis_ee.usd")
ROBOT_PRIM = "/gen3"

if not os.path.exists(USD_PATH):
    carb.log_error(
        f"Robot USD not found at {USD_PATH}. "
        "Import the URDF first via Isaac Utils > URDF Importer."
    )
else:
    extensions.enable_extension("isaacsim.ros2.bridge")

    prims.create_prim(
        ROBOT_PRIM,
        "Xform",
        position=np.array([0.0, 0.0, 0.0]),
        usd_path=USD_PATH,
    )

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
                    ("ArticulationController.inputs:robotPath", ROBOT_PRIM),
                    ("PublishJointState.inputs:topicName", "isaac_joint_states"),
                    ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
                    ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_PRIM)]),
                ],
            },
        )
        print("=" * 60)
        print("Gen3 ROS 2 bridge configured successfully.")
        print("Press PLAY in Isaac Sim to activate.")
        print("Then in a new terminal run:")
        print("  source ~/workspace/ros2_kortex_ws/install/setup.bash")
        print("  ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config isaac_sim.launch.py")
        print("=" * 60)
    except Exception as e:
        carb.log_error(f"OmniGraph setup failed: {e}")
