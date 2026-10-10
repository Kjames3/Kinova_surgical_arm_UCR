#!/usr/bin/env python3
"""Deterministic conservative surface spheres, including thin CAD shells.

Every triangle is assigned to a grid cell. Its cell's sphere encloses all three
vertices, hence the entire triangle. This avoids the old volumetric fit's holes.
These cover mesh surfaces, not certified solid interiors or swept trajectories.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml
from curobo.robot_parser import UrdfRobotParser

HERE = Path(__file__).resolve().parent
WS = HERE.parents[4]
URDF = HERE / "gen3_surgical.urdf"
# The coarse wrist fit intrudes into the tool's clearance during the recorded
# approach. Refine its surface coverage instead of reducing collision padding
# or excluding the wrist/tool pair.
LINK_SPACING = {"thesis_ee": 0.020, "spherical_wrist_2_link": 0.020}


def resolve_urdf_meshes(src, dst):
    root = ET.parse(src).getroot()
    for mesh in root.findall('.//mesh'):
        uri = mesh.get('filename')
        if not uri.startswith('package://kortex_description/'):
            raise ValueError(f'Expected source package URI, got {uri}')
        path = WS / 'src/ros2_kortex' / uri.removeprefix('package://')
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    ET.indent(root)
    ET.ElementTree(root).write(dst, encoding='unicode')
    return str(dst)


def link_mesh(parser, link):
    meshes = [g.get_trimesh_mesh(transform_with_pose=True)
              for g in parser.get_link_geometry(link, use_collision_mesh=True)]
    meshes = [trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
              for m in meshes if m is not None and len(m.vertices)]
    return trimesh.util.concatenate(meshes) if meshes else None


def surface_spheres(mesh, spacing=0.035):
    # Subdivide large triangles to avoid huge bounding spheres. Deterministic.
    vertices, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=spacing, max_iter=12)
    triangles = vertices[faces]
    keys = np.floor(triangles.mean(axis=1) / spacing).astype(np.int64)
    cells, assignments = np.unique(keys, axis=0, return_inverse=True)
    lower = np.full((len(cells), 3), np.inf)
    upper = np.full((len(cells), 3), -np.inf)
    np.minimum.at(lower, assignments, triangles.min(axis=1))
    np.maximum.at(upper, assignments, triangles.max(axis=1))
    centers = (lower + upper) / 2
    distances = np.linalg.norm(triangles - centers[assignments, None, :], axis=-1)
    radii = np.zeros(len(cells))
    np.maximum.at(radii, assignments, distances.max(axis=1))
    # Outward rounding and 0.1 mm margin preserve coverage in exported YAML.
    centers = np.round(centers, 8)
    radii = np.ceil((radii + 0.0001) * 1e8) / 1e8
    margin = float(np.min(radii[assignments, None] - np.linalg.norm(
        triangles - centers[assignments, None, :], axis=-1)))
    if margin < 0:
        raise ValueError(f'Uncovered triangles: {margin}')
    spheres = [{'center': c.tolist(), 'radius': float(r)} for c, r in zip(centers, radii)]
    return spheres, {'triangles': len(triangles), 'spheres': len(spheres),
                     'minimum_surface_margin_m': margin}


def main():
    resolved = resolve_urdf_meshes(URDF, HERE / 'gen3_surgical_resolved.urdf')
    parser = UrdfRobotParser(resolved, load_meshes=True, mesh_root='', build_scene_graph=True)
    parser.build_link_parent()
    spheres, report = {}, {}
    for link in parser.get_link_names_from_urdf():
        mesh = link_mesh(parser, link)
        if mesh is None:
            continue
        spacing = LINK_SPACING.get(link, 0.035)
        spheres[link], report[link] = surface_spheres(mesh, spacing)
        report[link]['spacing_m'] = spacing
        print(f'{link}: {report[link]}')

    # Cover the short discrepancy between visual CAD endpoint and measured TCP.
    # The bulk tool is covered by thesis_ee spheres, not a fictitious wrist-tip rod.
    from validate_fk_vs_urdf import parse_joints, urdf_transform
    joints = parse_joints(str(URDF))
    tool_transform = urdf_transform(joints, 'bracelet_link', 'thesis_ee', {})
    mesh = link_mesh(parser, 'thesis_ee')
    vertices = trimesh.transform_points(mesh.vertices, tool_transform)
    tip = urdf_transform(joints, 'bracelet_link', 'assembly_tip', {})[:3, 3]
    closest = vertices[np.argmin(np.linalg.norm(vertices - tip, axis=1))]
    gap = float(np.linalg.norm(closest - tip))
    if gap > 0.030:
        raise ValueError(f'CAD/TCP discrepancy {gap:.3f} m exceeds provisional 30 mm limit')
    count = max(2, int(np.ceil(gap / 0.006)) + 1)
    extension = [{'center': p.tolist(), 'radius': 0.008} for p in np.linspace(closest, tip, count)]
    spheres.setdefault('bracelet_link', []).extend(extension)
    report['tcp_extension'] = {'cad_to_tcp_m': gap, 'spheres': count,
                               'status': 'provisional; requires physical geometry validation'}
    (HERE / 'gen3_surgical_spheres.yml').write_text(yaml.safe_dump(
        {'collision_spheres': spheres}, sort_keys=False, default_flow_style=None))
    (HERE / 'sphere_coverage.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Total spheres: {sum(map(len, spheres.values()))}; CAD/TCP gap {gap*1000:.2f} mm')


if __name__ == '__main__':
    main()
