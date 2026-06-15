"""
Headless URDF → USD import for Kinova Gen3 7DOF + Robotiq 2F-140.

Run this whenever the URDF/xacro changes. Overwrites the USD referenced by
isaac_sim_gen3.py so the next launch picks up a correctly articulated robot.

Usage:
  cd ~/workspace/ros2_kortex_ws
  source install/setup.bash

  # (1) Render URDF from xacro with sim_isaac:=true
  xacro src/ros2_kortex/kortex_description/robots/gen3.xacro \
      arm:=gen3 dof:=7 gripper:=thesis_ee sim_isaac:=true \
      robot_ip:=xxx use_fake_hardware:=true > /tmp/gen3_isaac.urdf

  # (2) Import URDF → USD (headless). Script verifies articulation root exists.
  ~/isaacsim/python.sh src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_import_urdf.py

  # (3) Launch Isaac Sim
  ~/isaacsim/python.sh src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_gen3.py
"""

import os
import re
import sys

from isaacsim import SimulationApp

URDF_PATH = "/tmp/gen3_isaac.urdf"
URDF_RESOLVED_PATH = "/tmp/gen3_isaac_resolved.urdf"
USD_PATH = os.path.expanduser("~/isaacsim/gen3_thesis_ee.usd")


def _resolve_package_uris(urdf_text: str) -> str:
    """Replace package://PKG/rel URIs with absolute paths via AMENT_PREFIX_PATH.

    Isaac Sim's Python environment does not resolve ROS package:// URIs even
    when install/setup.bash is sourced, so mesh filenames stay unresolved and
    the URDF importer crashes with 'Used null prim'.  AMENT_PREFIX_PATH is set
    by source install/setup.bash and lists the install prefixes where each
    package's share/ directory lives.
    """
    ament_paths = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(":") if p]
    if not ament_paths:
        print("WARN: AMENT_PREFIX_PATH is empty — source install/setup.bash first.")
        return urdf_text

    def _replace(match):
        pkg, rel = match.group(1), match.group(2)
        for prefix in ament_paths:
            candidate = os.path.join(prefix, "share", pkg, rel)
            if os.path.exists(candidate):
                return candidate
        print(f"WARN: could not resolve package://{pkg}/{rel} — file not found under AMENT_PREFIX_PATH")
        return match.group(0)

    return re.sub(r'package://([^/]+)/([^"<>\s]+)', _replace, urdf_text)

simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd
from isaacsim.core.utils import extensions
from pxr import Usd, UsdPhysics

extensions.enable_extension("isaacsim.asset.importer.urdf")
simulation_app.update()

from isaacsim.asset.importer.urdf import _urdf  # noqa: E402

if not os.path.exists(URDF_PATH):
    print(f"ERROR: URDF not found at {URDF_PATH} — run xacro first (see docstring).")
    simulation_app.close()
    sys.exit(1)

# Resolve package:// URIs to absolute paths so the Isaac Sim importer can find meshes.
with open(URDF_PATH) as _f:
    _urdf_text = _f.read()
_urdf_resolved = _resolve_package_uris(_urdf_text)
with open(URDF_RESOLVED_PATH, "w") as _f:
    _f.write(_urdf_resolved)
print(f"Resolved URDF written to {URDF_RESOLVED_PATH}")

# Direct ImportConfig instantiation — more reliable than URDFCreateImportConfig.
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = True        # thesis_ee pen_tip is massless/geometry-free; merging avoids null-prim crash in importer. TF is still published by robot_state_publisher from the full URDF.
import_config.convex_decomp = False
import_config.fix_base = True
import_config.make_default_prim = True
import_config.self_collision = False
import_config.import_inertia_tensor = True
import_config.distance_scale = 1.0
import_config.density = 0.0
import_config.create_physics_scene = False     # isaac_sim_gen3.py creates its own

try:
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength = 1e7
    import_config.default_position_drive_damping = 1e5
except AttributeError as exc:
    print(f"WARN: could not set drive type explicitly ({exc}); relying on defaults.")

# Fresh stage so the flattened export is deterministic.
context = omni.usd.get_context()
context.new_stage()
simulation_app.update()

print(f"Parsing:  {URDF_RESOLVED_PATH}")
print(f"Target:   {USD_PATH}")

# Omit dest_path — some Isaac 4.x builds treat it as a ref-and-save which then
# collides with our own save below. Import into the in-memory stage instead.
status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_RESOLVED_PATH,
    import_config=import_config,
)

if not status:
    print("ERROR: URDFParseAndImportFile returned False.")
    simulation_app.close()
    sys.exit(1)

print(f"Importer returned prim_path = {prim_path}")

stage = context.get_stage()
if stage is None:
    print("ERROR: no active stage after import.")
    simulation_app.close()
    sys.exit(1)

# Make the robot prim the stage's default so references resolve cleanly.
robot_prim = stage.GetPrimAtPath(prim_path) if prim_path else None
if robot_prim and robot_prim.IsValid():
    stage.SetDefaultPrim(robot_prim)

# Flatten composes session/sublayers into a single SdfLayer so the saved file
# contains the robot geometry, not just a reference to a transient layer.
stage.Flatten().Export(USD_PATH)

# Verify: re-open the saved USD and confirm it has an articulation root.
verify_stage = Usd.Stage.Open(USD_PATH)
if verify_stage is None:
    print(f"ERROR: could not re-open {USD_PATH} for verification.")
    simulation_app.close()
    sys.exit(1)

default_prim = verify_stage.GetDefaultPrim()
art_roots = [p.GetPath() for p in verify_stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
n_joints = sum(1 for p in verify_stage.Traverse()
               if p.IsA(UsdPhysics.Joint))

size_kb = os.path.getsize(USD_PATH) / 1024.0
print(f"USD size: {size_kb:.1f} KB")
print(f"Default prim: {default_prim.GetPath() if default_prim else 'NONE'}")
print(f"Articulation roots: {art_roots if art_roots else 'NONE'}")
print(f"Physics joints in stage: {n_joints}")

if not art_roots:
    print("ERROR: imported USD has no ArticulationRootAPI — Isaac won't simulate it.")
    print("       Check the Isaac Sim console for URDF parser errors.")
    simulation_app.close()
    sys.exit(2)

print("OK — next: ~/isaacsim/python.sh src/ros2_kortex/surgical_arm_bringup/scripts/isaac_sim_gen3.py")
simulation_app.close()
