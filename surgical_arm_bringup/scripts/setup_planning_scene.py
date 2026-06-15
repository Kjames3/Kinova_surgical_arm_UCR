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
  container_z_from_table  (float, default 0.0)   : offset of mesh SW origin from
                                                    table surface (0 = origin at base)
"""

import os
import struct

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point


def _load_stl_mesh(filepath: str, scale: float = 0.001) -> Mesh:
    """
    Load a binary SolidWorks-exported STL file and return a shape_msgs/Mesh.
    scale converts mm -> m (SW exports in mm by convention).
    """
    mesh = Mesh()
    with open(filepath, "rb") as f:
        f.read(80)  # skip 80-byte header
        n_tris = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_tris):
            f.read(12)  # skip normal vector
            tri = MeshTriangle()
            base_idx = len(mesh.vertices)
            for j in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                mesh.vertices.append(Point(x=x * scale, y=y * scale, z=z * scale))
                tri.vertex_indices[j] = base_idx + j
            f.read(2)  # skip attribute byte count
            mesh.triangles.append(tri)
    return mesh


class PlanningSceneSetup(Node):
    def __init__(self):
        super().__init__("planning_scene_setup")

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
        self.declare_parameter("container_enabled",      True)
        self.declare_parameter("container_x",            0.50)   # 50 cm in front
        self.declare_parameter("container_y",            -0.20)  # 20 cm to the left (from operator's perspective facing the robot)
        self.declare_parameter("container_z_from_table", 0.0)    # 0 = origin at table surface

        self._scene_pub = self.create_publisher(
            PlanningScene, "/planning_scene", 10
        )
        self.create_timer(1.5, self._publish_scene)
        self._published = False

    def _publish_scene(self):
        if self._published:
            return
        self._published = True

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

        container_enabled     = self.get_parameter("container_enabled").value
        container_x           = self.get_parameter("container_x").value
        container_y           = self.get_parameter("container_y").value
        container_z_from_table = self.get_parameter("container_z_from_table").value

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
                mesh = _load_stl_mesh(mesh_path, scale=0.001)

                container_co = CollisionObject()
                container_co.header.frame_id = "world"
                container_co.id = "glass_container"
                container_co.operation = CollisionObject.ADD

                container_co.meshes.append(mesh)

                mesh_pose = Pose()
                mesh_pose.position.x = container_x
                mesh_pose.position.y = container_y
                # Place mesh origin at table surface + any SW-origin offset
                mesh_pose.position.z = z_top + container_z_from_table
                mesh_pose.orientation.w = 1.0
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

        self._scene_pub.publish(scene)

        log_lines.append("  Objects are now visible in RViz — MoveIt will avoid them.")
        self.get_logger().info("\n".join(log_lines))
        self.get_logger().info(
            "You can keep this node running to re-publish on reconnect, "
            "or Ctrl-C once the scene is confirmed in RViz."
        )


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
