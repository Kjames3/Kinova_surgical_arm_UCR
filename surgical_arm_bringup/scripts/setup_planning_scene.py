#!/usr/bin/env python3
"""
Add static collision objects to the MoveIt planning scene.

Currently adds:
  - table         : large flat box representing the surface the robot is bolted to.
  - pole          : vertical cylinder obstacle (e.g. camera/sensor stand).
  - glass_container : mesh collision object loaded from Glass_container.STL,
                      placed on the table in front of the robot at the
                      position where the tip will be inserted.

Usage
-----
  # Terminal 3 — run AFTER launching the robot (Terminal 1):
  ros2 run surgical_arm_bringup setup_planning_scene.py

  # Apply a bag-derived scene and verify its object IDs through MoveIt:
  #   --ros-args -p scene_file:=/absolute/path/scene.json

  # Disable the container:
  ros2 run surgical_arm_bringup setup_planning_scene.py \\
    --ros-args -p container_enabled:=false

  # Move the container:
  ros2 run surgical_arm_bringup setup_planning_scene.py \\
    --ros-args -p container_x:=0.45 -p container_y:=0.15

Parameters
----------
  table_x_size   (float, default 2.0)   : table width  in metres (X axis)
  table_y_size   (float, default 2.0)   : table depth  in metres (Y axis)
  table_thickness(float, default 0.05)  : table box height in metres
  table_z_surface(float, default -0.03) : Z of the table surface in world frame

  pole_enabled   (bool,  default true)  : add the pole collision object
  pole_x         (float, default 0.384) : pole centre X in world frame
  pole_y         (float, default 0.381) : pole centre Y in world frame
  pole_height    (float, default 1.0)   : pole height in metres
  pole_radius    (float, default 0.025) : pole radius in metres
  pole_z_base    (float, default 0.0)   : Z of the bottom of the pole

  container_enabled       (bool,  default true)  : add the glass container
  container_x             (float, default 0.50)  : container centre X in world frame
                                                    (50 cm in front of robot base)
  container_y             (float, default 0.20)  : container centre Y in world frame
                                                    (20 cm to the left)
  container_z_from_table  (float, default 0.0)   : height of container bottom above
                                                    table surface (0 = directly on table)
"""

import os
import json
import math

from container_geometry import load_container_geometry, validate_scene_spec

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import PlanningScene, CollisionObject, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point


def _load_stl_mesh(filepath: str) -> Mesh:
    """Load and orient the hollow container; origin is bottom centre in metres."""
    vertices, triangles, _ = load_container_geometry(filepath)
    mesh = Mesh()
    mesh.vertices = [Point(x=x, y=y, z=z) for x, y, z in vertices]
    mesh.triangles = [MeshTriangle(vertex_indices=indices) for indices in triangles]
    return mesh


class PlanningSceneSetup(Node):
    def __init__(self):
        super().__init__("planning_scene_setup")

        self.declare_parameter("scene_file", "")
        self._scene_spec = None
        filename = self.get_parameter("scene_file").value
        if filename:
            with open(os.path.expanduser(filename)) as stream:
                self._scene_spec = validate_scene_spec(json.load(stream))

        self.declare_parameter("table_x_size",    2.0)
        self.declare_parameter("table_y_size",    2.0)
        self.declare_parameter("table_thickness", 0.05)
        self.declare_parameter("table_z_surface", -0.03)

        self.declare_parameter("pole_enabled", True)
        self.declare_parameter("pole_x",       0.384)
        self.declare_parameter("pole_y",       0.381)
        self.declare_parameter("pole_height",  1.0)
        self.declare_parameter("pole_radius",  0.025)
        self.declare_parameter("pole_z_base",  0.0)

        # Glass container — mesh loaded from Glass_container.STL
        self.declare_parameter("container_yaw_deg", 0.0)
        self.declare_parameter("container_enabled",      True)
        self.declare_parameter("container_x",            0.50)   # 50 cm in front
        self.declare_parameter("container_y",            -0.20)  # 20 cm to the left (from operator's perspective facing the robot)
        self.declare_parameter("container_z_from_table", 0.0)    # 0 = origin at table surface

        self._apply_cli = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._get_cli = self.create_client(GetPlanningScene, "/get_planning_scene")
        self._pending = False
        self._published = False
        self.create_timer(1.5, self._publish_scene)

    def _publish_scene(self):
        if self._published or self._pending:
            return
        if not self._apply_cli.service_is_ready() or not self._get_cli.service_is_ready():
            self.get_logger().warn("Waiting for MoveIt apply/get planning-scene services.", throttle_duration_sec=5.0)
            return

        x     = self.get_parameter("table_x_size").value
        y     = self.get_parameter("table_y_size").value
        thick = self.get_parameter("table_thickness").value
        z_top = self.get_parameter("table_z_surface").value

        pole_enabled = self.get_parameter("pole_enabled").value
        pole_x       = self.get_parameter("pole_x").value
        pole_y       = self.get_parameter("pole_y").value
        pole_h       = self.get_parameter("pole_height").value
        pole_r       = self.get_parameter("pole_radius").value
        pole_z_base  = self.get_parameter("pole_z_base").value

        container_yaw = math.radians(self.get_parameter("container_yaw_deg").value)
        container_enabled     = self.get_parameter("container_enabled").value
        container_x           = self.get_parameter("container_x").value
        container_y           = self.get_parameter("container_y").value
        container_z_from_table = self.get_parameter("container_z_from_table").value

        if self._scene_spec:
            spec = self._scene_spec
            x, y, thick = spec["table_size_m"]
            z_top = spec["table_surface_z_m"]
            container_x, container_y, base_z = spec["container_center_base_m"]
            container_z_from_table = base_z - z_top
            container_yaw = spec["container_yaw_rad"]
            container_enabled = True
            pole_enabled = False  # No unmeasured pole in the reconstructed scene.

        # --- Table collision object ---
        table_co = CollisionObject()
        table_co.header.frame_id = "world"
        table_co.id = "table"
        table_co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [x, y, thick]

        table_pose = Pose()
        table_pose.position.x = 0.0
        table_pose.position.y = 0.0
        table_pose.position.z = z_top - thick / 2.0
        table_pose.orientation.w = 1.0

        table_co.primitives.append(box)
        table_co.primitive_poses.append(table_pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(table_co)

        log_lines = [
            "Planning scene updated:",
            f"  [table] BOX  {x:.2f} x {y:.2f} x {thick:.3f} m",
            f"          centre z = {table_pose.position.z:.3f} m  "
            f"(surface at z = {z_top:.3f} m)",
        ]

        # --- Pole collision object (optional) ---
        if pole_enabled:
            pole_co = CollisionObject()
            pole_co.header.frame_id = "world"
            pole_co.id = "pole"
            pole_co.operation = CollisionObject.ADD

            cylinder = SolidPrimitive()
            cylinder.type = SolidPrimitive.CYLINDER
            cylinder.dimensions = [pole_h, pole_r]

            pole_pose = Pose()
            pole_pose.position.x = pole_x
            pole_pose.position.y = pole_y
            pole_pose.position.z = pole_z_base + pole_h / 2.0
            pole_pose.orientation.w = 1.0

            pole_co.primitives.append(cylinder)
            pole_co.primitive_poses.append(pole_pose)

            scene.world.collision_objects.append(pole_co)

            log_lines += [
                f"  [pole]  CYLINDER  r={pole_r:.3f} m  h={pole_h:.2f} m",
                f"          centre x={pole_x:.3f}  y={pole_y:.3f}  "
                f"z={pole_pose.position.z:.3f} m  (base at z={pole_z_base:.3f} m)",
            ]

        # --- Glass container collision object (optional, STL mesh) ---
        if container_enabled:
            try:
                mesh_path = os.path.join(
                    get_package_share_directory("kortex_description"),
                    "grippers", "thesis_ee", "meshes", "Glass_container.STL",
                )
                if self._scene_spec and self._scene_spec.get("container_mesh_file"):
                    mesh_path = os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(
                        self.get_parameter("scene_file").value))), self._scene_spec["container_mesh_file"])
                _, _, metadata = load_container_geometry(mesh_path)
                if self._scene_spec and metadata["source_sha256"] != self._scene_spec["container_mesh_sha256"]:
                    raise ValueError("Container CAD hash differs from offline scene")
                mesh = _load_stl_mesh(mesh_path)

                container_co = CollisionObject()
                container_co.header.frame_id = "world"
                container_co.id = "glass_container"
                container_co.operation = CollisionObject.ADD

                container_co.meshes.append(mesh)

                mesh_pose = Pose()
                mesh_pose.position.x = container_x
                mesh_pose.position.y = container_y
                # Normalized mesh origin is its bottom centre.
                mesh_pose.position.z = z_top + container_z_from_table
                mesh_pose.orientation.w = math.cos(container_yaw / 2)
                mesh_pose.orientation.z = math.sin(container_yaw / 2)
                container_co.mesh_poses.append(mesh_pose)

                scene.world.collision_objects.append(container_co)

                log_lines += [
                    f"  [glass_container] MESH  Glass_container.STL",
                    f"          origin x={container_x:.3f}  y={container_y:.3f}  "
                    f"z={mesh_pose.position.z:.3f} m  "
                    f"({len(mesh.triangles)} triangles)",
                ]
            except Exception as e:
                self.get_logger().warn(
                    f"  [glass_container] Could not load mesh: {e}\n"
                    "  Check that kortex_description is built and Glass_container.STL exists."
                )

                return  # Never silently apply a table-only scene after a mesh error.

        # Complete replacement for these object IDs, while preserving other world
        # objects. Mark robot_state as a diff so existing attachments survive.
        scene.robot_state.is_diff = True
        self._expected_ids = {obj.id for obj in scene.world.collision_objects}
        self._pending = True
        self._apply_cli.call_async(ApplyPlanningScene.Request(scene=scene)).add_done_callback(self._applied)

        self.get_logger().info("\n".join(log_lines).replace("Planning scene updated:", "Requesting planning scene:"))

    def _applied(self, future):
        try:
            if not future.result().success:
                raise RuntimeError("MoveIt rejected scene")
            request = GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
            self._get_cli.call_async(request).add_done_callback(self._verified)
        except Exception as exc:
            self._pending = False
            self.get_logger().error(f"Scene application failed: {exc}")

    def _verified(self, future):
        self._pending = False
        try:
            ids = {obj.id for obj in future.result().scene.world.collision_objects}
            if not self._expected_ids.issubset(ids):
                raise RuntimeError(f"Readback missing {self._expected_ids - ids}")
            self._published = True
            self.get_logger().info(f"Verified MoveIt scene objects: {sorted(ids)}")
        except Exception as exc:
            self.get_logger().error(f"Scene verification failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = PlanningSceneSetup()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
