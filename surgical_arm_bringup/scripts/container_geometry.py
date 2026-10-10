#!/usr/bin/env python3
"""Shared, ROS-free container geometry: centre XY, bottom Z, upright opening.

The supplied SolidWorks STL is millimetres with +Y up and origin at a corner.
Use the proper rotation (x,y,z)->(x,-z,y), then centre XY and lower the base.
Keep every original triangle: a convex hull would incorrectly fill the opening.
"""
import hashlib
import math
from pathlib import Path
import struct


def load_container_geometry(path):
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError('Truncated STL')
    count = struct.unpack_from('<I', data, 80)[0]
    if not count or len(data) != 84 + 50 * count:
        raise ValueError('Expected a complete binary STL')
    vertices = []
    triangles = []
    for i in range(count):
        raw = struct.unpack_from('<9f', data, 84 + 50 * i + 12)
        triangles.append([len(vertices) + j for j in range(3)])
        vertices.extend([[v * .001 for v in raw[j:j+3]] for j in (0, 3, 6)])
    if not all(math.isfinite(v) for point in vertices for v in point):
        raise ValueError('Nonfinite mesh vertex')
    lo = [min(v[k] for v in vertices) for k in range(3)]
    hi = [max(v[k] for v in vertices) for k in range(3)]
    extents = [hi[k]-lo[k] for k in range(3)]
    if any(abs(a-b) > .0001 for a,b in zip(extents, [.09, .086, .09])):
        raise ValueError(f'Unexpected container CAD dimensions: {extents}; review axis/scale')
    cx, cz = (lo[0]+hi[0])/2, (lo[2]+hi[2])/2
    vertices = [[x-cx, cz-z, y-lo[1]] for x,y,z in vertices]
    return vertices, triangles, {'source_sha256': hashlib.sha256(data).hexdigest(),
                                'dimensions_m': [extents[0],extents[2],extents[1]],
                                'convention': 'origin at bottom centre; +Z up; metres'}


def validate_scene_spec(spec):
    if spec.get('version') != 1 or spec.get('frame_id') != 'world':
        raise ValueError('Expected scene version 1 in world frame')
    for key, length in [('table_size_m', 3), ('container_center_base_m', 3)]:
        values = spec.get(key, [])
        if len(values) != length or not all(isinstance(v, (float,int)) and math.isfinite(v) for v in values):
            raise ValueError(f'Invalid {key}')
    if any(v <= 0 for v in spec['table_size_m']):
        raise ValueError('Table dimensions must be positive')
    if not math.isfinite(spec['table_surface_z_m']):
        raise ValueError('Invalid table surface')
    if spec['container_center_base_m'][2] < spec['table_surface_z_m'] - 1e-6:
        raise ValueError('Container base is below table surface')
    if not math.isfinite(spec.get('container_yaw_rad', float('nan'))):
        raise ValueError('Missing or invalid container yaw')
    if not isinstance(spec.get('container_mesh_sha256'), str):
        raise ValueError('Missing mesh fingerprint')
    return spec


def container_top_z(table_z, container_height, base_offset):
    """Container rim above its support surface; keep table and board separate."""
    if not all(math.isfinite(v) for v in (table_z, container_height, base_offset)):
        raise ValueError('Nonfinite container height parameter')
    if container_height <= 0 or base_offset < 0:
        raise ValueError('Container height must be positive and support offset nonnegative')
    return table_z + base_offset + container_height
