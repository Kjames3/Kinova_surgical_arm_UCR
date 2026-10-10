#!/usr/bin/env python3
"""Independent checks of exported spheres: surface samples and HOME pairs.

No collision exclusions are added automatically. A collision at HOME is a
model qualification failure to investigate, not a reason to suppress a pair.
"""
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml
from curobo.robot_parser import UrdfRobotParser

from generate_spheres import link_mesh
from validate_fk_vs_urdf import HOME, parse_joints, urdf_transform

HERE = Path(__file__).resolve().parent


def main():
    cfg = yaml.safe_load((HERE / 'gen3_surgical.yml').read_text())['robot_cfg']['kinematics']
    parser = UrdfRobotParser(cfg['urdf_path'], load_meshes=True, mesh_root='', build_scene_graph=True)
    parser.build_link_parent()
    joints = parse_joints(HERE / 'gen3_surgical.urdf')
    rng = np.random.default_rng(42)
    world, coverage = {}, {}
    for link, spheres in cfg['collision_spheres'].items():
        centers = np.array([s['center'] for s in spheres])
        radii = np.array([s['radius'] for s in spheres])
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() or (radii <= 0).any():
            raise ValueError(f'Invalid sphere on {link}')
        mesh = link_mesh(parser, link)
        # Independent barycentric surface samples, including vertices. Test the
        # persisted YAML, not the generator's in-memory coverage certificate.
        indices = rng.choice(len(mesh.faces), 10000, p=mesh.area_faces / mesh.area)
        weights = rng.dirichlet([1, 1, 1], len(indices))
        samples = np.concatenate((mesh.vertices, (mesh.triangles[indices] * weights[:, :, None]).sum(axis=1)))
        worst = -np.inf
        for offset in range(0, len(samples), 256):
            distance = np.linalg.norm(samples[offset:offset+256, None] - centers, axis=-1) - radii
            worst = max(worst, float(distance.min(axis=1).max()))
        coverage[link] = {'samples': len(samples), 'worst_outside_m': worst}
        transform = urdf_transform(joints, 'base_link', link, HOME)
        world[link] = (trimesh.transform_points(centers, transform), radii)

    overlaps = []
    for i, a in enumerate(world):
        for b in list(world)[i+1:]:
            if b in cfg['self_collision_ignore'].get(a, []) or a in cfg['self_collision_ignore'].get(b, []):
                continue
            pa, ra = world[a]
            pb, rb = world[b]
            distance = np.linalg.norm(pa[:, None] - pb, axis=-1) - ra[:, None] - rb
            # Report nominal geometry separately from cuRobo's extra padding.
            minimum = float(distance.min())
            if minimum < 0:
                overlaps.append({'links': [a, b], 'nominal_overlap_m': -minimum})
    report = {'surface_coverage': coverage, 'home_nominal_overlaps': overlaps,
              'qualification_pass': all(v['worst_outside_m'] <= 1e-7 for v in coverage.values()) and not overlaps}
    (HERE / 'collision_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['qualification_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
