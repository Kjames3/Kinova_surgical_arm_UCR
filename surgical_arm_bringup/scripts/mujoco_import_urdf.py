"""
Headless URDF → MJCF import for Kinova Gen3 7DOF + thesis_ee.

Run this whenever the URDF/xacro changes. Overwrites the MJCF referenced by
mujoco_sim_gen3.py so the next launch picks up a correctly articulated robot.

Usage:
  cd ~/workspace/ros2_kortex_ws
  source install/setup.bash

  # (1) Render URDF from xacro (vendor xacro arg names are unchanged)
  xacro src/ros2_kortex/kortex_description/robots/gen3.xacro \
      arm:=gen3 dof:=7 gripper:=thesis_ee sim_isaac:=true \
      robot_ip:=xxx use_fake_hardware:=true > /tmp/gen3_mujoco.urdf

  # (2) Import URDF → MJCF. Script verifies joints + actuators exist.
  python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/mujoco_import_urdf.py

  # (3) Launch MuJoCo
  python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/mujoco_sim_gen3.py
"""

import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

import mujoco

URDF_PATH = "/tmp/gen3_mujoco.urdf"
URDF_RESOLVED_PATH = "/tmp/gen3_mujoco_resolved.urdf"
MJCF_PATH = os.path.expanduser("~/mujoco_models/gen3_thesis_ee.xml")

# Meshes we have to rewrite before MuJoCo will accept them. Kept out of the
# source tree (they are derived artifacts, not authored assets) but next to the
# MJCF rather than in /tmp, because the saved MJCF references them by absolute
# path and must survive a reboot.
MESH_CACHE_DIR = os.path.expanduser("~/mujoco_models/meshes")

# MuJoCo's STL loader hard-rejects meshes above this triangle count.
MJ_MAX_FACES = 200000
MJ_DECIMATE_TARGET = 150000

ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]

# Home pose used by every downstream sim/control script. Stored as a MuJoCo
# keyframe so the sim can snap to it without duplicating the numbers.
HOME_QPOS = {
    "joint_1": 0.0,
    "joint_2": 0.26,
    "joint_3": 3.14,
    "joint_4": -2.0,
    "joint_5": 0.0,
    "joint_6": -0.93,
    "joint_7": 1.57,
}


def _resolve_package_uris(urdf_text: str) -> str:
    """Replace package://PKG/rel URIs with absolute paths via AMENT_PREFIX_PATH.

    MuJoCo's URDF parser does not resolve ROS package:// URIs even when
    install/setup.bash is sourced, so mesh filenames stay unresolved and the
    compiler aborts with a 'could not open file' error.  AMENT_PREFIX_PATH is
    set by source install/setup.bash and lists the install prefixes where each
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


def _stl_face_count(path: str) -> int:
    """Triangle count of a binary STL, or -1 if the file is ASCII/unreadable."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(84)
        if len(header) < 84 or header[:5].lower() == b"solid":
            return -1
        return struct.unpack("<I", header[80:84])[0]
    except OSError:
        return -1


def _decimate_stl(path: str) -> str:
    """Write a face-reduced copy of `path` into MESH_CACHE_DIR and return it.

    Returns the original path if decimation is not possible, so a missing
    open3d degrades to a clear MuJoCo error rather than a traceback here.
    """
    os.makedirs(MESH_CACHE_DIR, exist_ok=True)
    out = os.path.join(MESH_CACHE_DIR, os.path.basename(path))
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return out  # cached from a previous run
    try:
        import open3d as o3d
    except ImportError:
        print(f"WARN: {os.path.basename(path)} exceeds MuJoCo's {MJ_MAX_FACES}-face limit "
              f"and open3d is not installed — cannot decimate.")
        return path
    mesh = o3d.io.read_triangle_mesh(path)
    reduced = mesh.simplify_quadric_decimation(MJ_DECIMATE_TARGET)
    reduced.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out, reduced)
    print(f"Decimated {os.path.basename(path)}: "
          f"{len(mesh.triangles)} → {len(reduced.triangles)} faces → {out}")
    return out


def _normalize_meshes(urdf_text: str) -> str:
    """Make every mesh filename something MuJoCo's compiler can actually load.

    Three separate incompatibilities, all of them silent-until-compile:
      * `file://` URI scheme — MuJoCo opens filenames, not URIs.
      * COLLADA `.dae` — the Kinova arm links point at .dae for both visual and
        collision, and MuJoCo has no COLLADA reader. Every one of them ships a
        sibling .STL of the same geometry, so we swap the extension.
      * Oversized STLs — thesis_ee's Full_Assembly.STL is a 650k-triangle CAD
        export and MuJoCo refuses anything over 200k. We decimate into
        MESH_CACHE_DIR instead of editing the package's asset.
    """
    def _replace(match):
        path = match.group(1)
        if path.startswith("file://"):
            path = path[len("file://"):]
        if path.lower().endswith(".dae"):
            for ext in (".STL", ".stl"):
                if os.path.exists(path[:-4] + ext):
                    path = path[:-4] + ext
                    break
            else:
                print(f"WARN: no STL sibling for {path} — MuJoCo cannot read COLLADA.")
        if path.lower().endswith(".stl") and _stl_face_count(path) > MJ_MAX_FACES:
            path = _decimate_stl(path)
        return f'filename="{path}"'

    return re.sub(r'filename="([^"]+)"', _replace, urdf_text)


def _inject_mujoco_compiler(urdf_text: str) -> str:
    """Add <mujoco><compiler .../></mujoco> under <robot> if not already there.

    This is the only hook URDF gives us for MuJoCo-specific compile settings.
    Without it MuJoCo applies its URDF defaults: it prepends its own meshdir to
    every filename (breaking the absolute paths we just resolved), rejects the
    slightly non-PSD inertia tensors the Kinova URDF ships, and drops visual
    geometry.  strippath=false / balanceinertia=true / discardvisual=false fix
    those three in order.
    """
    if "<mujoco>" in urdf_text:
        return urdf_text
    block = ('<mujoco><compiler strippath="false" balanceinertia="true" '
             'discardvisual="false"/></mujoco>\n')
    # Insert right after the opening <robot ...> tag.
    return re.sub(r"(<robot\b[^>]*>)", r"\1\n  " + block, urdf_text, count=1)


if not os.path.exists(URDF_PATH):
    print(f"ERROR: URDF not found at {URDF_PATH} — run xacro first (see docstring).")
    sys.exit(1)

with open(URDF_PATH) as _f:
    _urdf_text = _f.read()
_urdf_resolved = _inject_mujoco_compiler(_normalize_meshes(_resolve_package_uris(_urdf_text)))
with open(URDF_RESOLVED_PATH, "w") as _f:
    _f.write(_urdf_resolved)
print(f"Resolved URDF written to {URDF_RESOLVED_PATH}")

print(f"Parsing:  {URDF_RESOLVED_PATH}")
print(f"Target:   {MJCF_PATH}")

os.makedirs(os.path.dirname(MJCF_PATH), exist_ok=True)

try:
    _model = mujoco.MjModel.from_xml_path(URDF_RESOLVED_PATH)
except Exception as exc:  # mujoco raises a bare ValueError on compile failure
    print(f"ERROR: MuJoCo failed to compile the URDF: {exc}")
    sys.exit(1)

# Record the compiled joint order *before* post-processing, because the
# keyframe qpos vector below must be laid out in exactly this order.
_urdf_joint_order = []
for _j in range(_model.njnt):
    _name = mujoco.mj_id2name(_model, mujoco.mjtObj.mjOBJ_JOINT, _j)
    _urdf_joint_order.append((_name, _model.jnt_type[_j], _model.jnt_qposadr[_j]))
_urdf_nq = _model.nq

mujoco.mj_saveLastXML(MJCF_PATH, _model)
print(f"Raw MJCF written ({_model.nbody} bodies, {_model.njnt} joints, nq={_urdf_nq})")

# ---------------------------------------------------------------------------
# Post-process the MJCF: a URDF carries no actuators, no scene and no lighting,
# so the raw conversion is not simulatable on its own.
# ---------------------------------------------------------------------------
tree = ET.parse(MJCF_PATH)
root = tree.getroot()


def _child(parent, tag):
    found = parent.find(tag)
    if found is None:
        found = ET.SubElement(parent, tag)
    return found


if root.find("option") is None:
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})

worldbody = _child(root, "worldbody")

# Base weld check. The URDF root link has no joint to world, and MuJoCo only
# adds a freejoint when the URDF declares a floating base — so normally there
# is nothing to strip here. We still scan, because a stray freejoint would let
# the whole arm fall through the table and the failure looks like a solver bug.
_removed_free = []
for body in worldbody.findall("body"):
    for tag in ("freejoint", "joint"):
        for jnt in list(body.findall(tag)):
            if tag == "freejoint" or jnt.get("type") == "free":
                body.remove(jnt)
                _removed_free.append(jnt.get("name") or "<unnamed>")
if _removed_free:
    print(f"Base weld: removed floating joint(s) {_removed_free} — base is now fixed to world.")
else:
    print("Base weld: no freejoint found; URDF root imported already welded to world.")

# Scene geometry. Dimensions mirror setup_planning_scene.py exactly so the
# MoveIt planning scene and the sim agree. MuJoCo box/cylinder sizes are
# HALF-extents: table 2.0x2.0x0.05 -> "1.0 1.0 0.025", top face at z=-0.03 so
# the centre sits at -0.055; container r=0.04 h=0.10 -> "0.04 0.05".
if worldbody.find("geom[@name='table']") is None:
    ET.SubElement(worldbody, "geom", {
        "name": "table", "type": "box", "size": "1.0 1.0 0.025",
        "pos": "0 0 -0.055", "rgba": "0.55 0.45 0.35 1",
    })
if worldbody.find("geom[@name='glass_container']") is None:
    ET.SubElement(worldbody, "geom", {
        "name": "glass_container", "type": "cylinder", "size": "0.04 0.05",
        "pos": "0.5 -0.2 0.02", "rgba": "0.6 0.8 0.9 0.35",
    })

# Lighting + ground. A URDF carries no light source, so without this the
# viewport renders solid black and the model looks like a failed import — the
# same trap the previous simulator's dome-light setup existed to avoid. The
# checkered ground texture also gives the eye a depth reference when jogging.
asset = _child(root, "asset")
if asset.find("texture[@name='grid']") is None:
    ET.SubElement(asset, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker",
        "width": "512", "height": "512",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.15 0.2",
    })
if asset.find("material[@name='grid']") is None:
    ET.SubElement(asset, "material", {
        "name": "grid", "texture": "grid", "texrepeat": "8 8", "reflectance": "0.1",
    })
if worldbody.find("light") is None:
    ET.SubElement(worldbody, "light", {
        "name": "top", "pos": "0 0 2.5", "dir": "0 0 -1",
        "directional": "true", "diffuse": "0.7 0.7 0.7",
    })
if worldbody.find("geom[@name='ground']") is None:
    ET.SubElement(worldbody, "geom", {
        "name": "ground", "type": "plane", "size": "5 5 0.1",
        "pos": "0 0 -0.08", "material": "grid",
    })

# Actuators. mujoco_sim_gen3.py maps incoming JointState command names straight
# onto actuator names, so each actuator MUST be named after its joint.
# kp=1000 with dampratio=1 is stiff enough to hold the arm against gravity
# without the position loop ringing at the 2 ms timestep.
_joint_elems = {}
for body in root.iter("body"):
    for jnt in body.findall("joint"):
        if jnt.get("name"):
            _joint_elems[jnt.get("name")] = jnt

_actuated = [n for n in ARM_JOINTS if n in _joint_elems]
# Any extra hinge/slide joint that is not an arm joint is a gripper DOF.
_actuated += [n for n in _joint_elems if n not in ARM_JOINTS]

missing = [n for n in ARM_JOINTS if n not in _joint_elems]
if missing:
    print(f"ERROR: compiled model is missing arm joints: {missing}")
    sys.exit(2)

actuator = _child(root, "actuator")
_existing_act = {a.get("name") for a in actuator}
for name in _actuated:
    # Joint-level damping keeps the link from coasting between control ticks.
    if _joint_elems[name].get("damping") is None:
        _joint_elems[name].set("damping", "1.0")
    if name in _existing_act:
        continue
    ET.SubElement(actuator, "position", {
        "name": name, "joint": name, "kp": "1000", "dampratio": "1",
    })

# Keyframe. qpos is a flat vector over the whole model, so build it from the
# compiled joint order rather than assuming joint_1 lands at index 0.
qpos = [0.0] * _urdf_nq
for name, jtype, adr in _urdf_joint_order:
    if name in HOME_QPOS and jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        qpos[adr] = HOME_QPOS[name]
keyframe = _child(root, "keyframe")
if keyframe.find("key[@name='home']") is None:
    ET.SubElement(keyframe, "key", {
        "name": "home", "qpos": " ".join(f"{v:g}" for v in qpos),
    })

tree.write(MJCF_PATH, encoding="utf-8", xml_declaration=True)

# ---------------------------------------------------------------------------
# Verify: recompile the file we actually wrote. A model that only compiles
# in-memory is worthless to the sim scripts.
# ---------------------------------------------------------------------------
try:
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
except Exception as exc:
    print(f"ERROR: could not re-compile {MJCF_PATH} for verification: {exc}")
    sys.exit(1)

act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(model.njnt)]

size_kb = os.path.getsize(MJCF_PATH) / 1024.0
print(f"MJCF size: {size_kb:.1f} KB")
print(f"Bodies (nbody):    {model.nbody}")
print(f"Joints (njnt):     {model.njnt}")
print(f"Actuators (nu):    {model.nu}")
print(f"Generalised coords (nq): {model.nq}")
print(f"Actuator names: {act_names}")
print(f"Keyframes: {model.nkey}")

missing_j = [n for n in ARM_JOINTS if n not in jnt_names]
missing_a = [n for n in ARM_JOINTS if n not in act_names]
if missing_j:
    print(f"ERROR: written MJCF is missing arm joints {missing_j} — sim cannot articulate it.")
    sys.exit(2)
if missing_a:
    print(f"ERROR: written MJCF is missing actuators {missing_a} — JointState commands "
          f"will not reach the arm.")
    sys.exit(2)

print("OK — next: python3 src/Kinova_surgical_arm_UCR/surgical_arm_bringup/scripts/mujoco_sim_gen3.py")
