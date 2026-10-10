#!/usr/bin/env python3
"""Bound wrist/tool mesh surface distance at one offline trajectory waypoint.

Uses deterministic subdivided wrist vertices and exact point/triangle distances.
A surface-distance bound is not a solid-containment or hardware-clearance proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree
from curobo.robot_parser import UrdfRobotParser
from generate_spheres import link_mesh
from validate_fk_vs_urdf import parse_joints, urdf_transform

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory',type=Path)
    p.add_argument('--waypoint',type=int,required=True)
    p.add_argument('--robot',type=Path,default=HERE/'gen3_surgical.yml')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resolution',type=float,default=.002,help='Maximum subdivided wrist edge in metres')
    args=p.parse_args()
    if not np.isfinite(args.resolution) or not 0<args.resolution<=.01:p.error('resolution must be in (0, 0.01] m')
    data=json.loads(args.trajectory.read_text())
    if not 0<=args.waypoint<len(data['positions']):p.error('waypoint out of range')
    cfg=yaml.safe_load(args.robot.read_text())['robot_cfg']['kinematics']
    parser=UrdfRobotParser(cfg['urdf_path'],load_meshes=True,mesh_root='',build_scene_graph=True)
    parser.build_link_parent()
    joints=parse_joints(cfg['urdf_path'])
    q=dict(zip(data['joint_names'],data['positions'][args.waypoint]))
    names=['spherical_wrist_2_link','thesis_ee']
    meshes=[link_mesh(parser,n).apply_transform(urdf_transform(joints,'base_link',n,q)) for n in names]
    wrist,tool=meshes
    # Faces outside the wrist AABB expanded by 50 mm cannot determine a
    # smaller distance than 50 mm. This avoids querying the full 650k-face CAD.
    crop_distance=.05; bounds=wrist.bounds;tri=tool.triangles
    keep=np.all(tri.max(axis=1)>=bounds[0]-crop_distance,axis=1)&np.all(tri.min(axis=1)<=bounds[1]+crop_distance,axis=1)
    if not keep.any():raise ValueError('No nearby tool faces; increase crop distance')
    tool=tool.submesh([np.flatnonzero(keep)],append=True)
    vertices,faces=trimesh.remesh.subdivide_to_size(wrist.vertices,wrist.faces,max_edge=args.resolution,max_iter=12)
    triangles=vertices[faces]
    max_edge=max(float(np.linalg.norm(triangles[:,i]-triangles[:,(i+1)%3],axis=1).max()) for i in range(3))
    if max_edge>args.resolution*(1+1e-6):raise ValueError('Subdivision did not achieve requested resolution')
    # A tool vertex gives an upper bound on the global mesh distance. Nominal
    # tool spheres conservatively cover its triangles (qualified separately),
    # so point-to-sphere distance can discard wrist samples that cannot improve
    # this upper bound. Retained samples still use actual mesh triangles.
    upper=float(cKDTree(tool.vertices).query(vertices)[0].min())
    tf=urdf_transform(joints,'base_link','thesis_ee',q)
    spheres=cfg['collision_spheres']['thesis_ee']
    centers=trimesh.transform_points([s['center'] for s in spheres],tf)
    radii=np.array([s['radius'] for s in spheres])
    candidates=[]
    for offset in range(0,len(vertices),512):
        block=vertices[offset:offset+512]
        lower=(np.linalg.norm(block[:,None]-centers,axis=-1)-radii).min(axis=1)
        candidates.extend(block[lower<=upper+1e-9])
    candidates=np.asarray(candidates)
    if not len(candidates):raise ValueError('Invalid sphere coverage or empty candidate set')
    distances=[upper]
    for offset in range(0,len(candidates),256):
        _,d,_=trimesh.proximity.closest_point(tool,candidates[offset:offset+256]);distances.append(float(d.min()))
    sampled_min=min(distances)
    # Distance-to-surface is 1-Lipschitz; every point on each subdivided wrist
    # triangle is within max_edge of a queried vertex. Include omitted faces.
    result={'links':names,'waypoint':args.waypoint,'time_s':data['time_s'][args.waypoint],
            'mesh_surface_clearance_lower_m':max(0.,min(sampled_min-max_edge,crop_distance)),
            'mesh_surface_clearance_upper_m':sampled_min,'maximum_subdivision_edge_m':max_edge,
            'source_samples':len(vertices),'mesh_query_samples':len(candidates),
            'sample_pruning':'Qualified nominal tool sphere cover; excluded points farther than a known upper bound','target_faces_retained':len(tool.faces),
            'excluded_target_face_clearance_lower_m':crop_distance,
            'trajectory_sha256':hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
            'robot_yaml_sha256':hashlib.sha256(args.robot.read_bytes()).hexdigest(),
            'scope':'Mesh surface distance at one configuration, not physical or solid-containment qualification'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
