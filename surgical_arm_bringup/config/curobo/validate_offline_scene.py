#!/usr/bin/env python3
"""Build cuRobo world and a preview from a bag-derived scene spec, without ROS."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import trimesh
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from container_geometry import load_container_geometry, validate_scene_spec
from curobo.scene import Scene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=validate_scene_spec(json.loads(args.scene.read_text()))
    source=args.scene.resolve().parent / spec['container_mesh_file']
    vertices,faces,metadata=load_container_geometry(source)
    if metadata['source_sha256'] != spec['container_mesh_sha256']:
        raise ValueError('CAD fingerprint mismatch')
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    # Check the actual solid mesh: empty open interior, material in bottom/wall.
    probes=np.array([[0,0,.04],[0,0,.085],[0,0,.002],[.042,0,.04],[.06,0,.04]])
    inside=mesh.contains(probes).tolist()
    if not mesh.is_watertight or inside != [False,False,True,True,False]:
        raise ValueError(f'Container hollow-geometry checks failed: {inside}')
    bounds=mesh.bounds
    if not np.allclose(bounds,[[-.045,-.045,0],[.045,.045,.086]],atol=1e-6):
        raise ValueError(f'Unexpected normalized bounds: {bounds}')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    mesh_path=output/'container_centered.obj';mesh.export(mesh_path)
    x,y,z=spec['container_center_base_m'];dims=spec['table_size_m'];top=spec['table_surface_z_m']
    yaw=spec['container_yaw_rad']
    world={'cuboid':{'table':{'dims':dims,'pose':[0,0,top-dims[2]/2,1,0,0,0]}},
           'mesh':{'glass_container':{'file_path':str(mesh_path),'pose':[x,y,z,math.cos(yaw/2),0,0,math.sin(yaw/2)]}}}
    (output/'curobo_world.yml').write_text(yaml.safe_dump(world,sort_keys=False))
    scene=Scene.create(world)
    loaded=scene.mesh[0].get_trimesh_mesh(transform_with_pose=True)
    reference=mesh.copy()
    reference.apply_transform(trimesh.transformations.rotation_matrix(yaw,[0,0,1]))
    reference.apply_translation([x,y,z])
    if not np.allclose(loaded.bounds,reference.bounds,atol=1e-6):
        raise ValueError('cuRobo scene transform mismatch')
    report={'passed':True,'frame':'world == base_link (verified from recorded URDF)',
            'quaternion_order':'wxyz','objects':['table','glass_container'],
            'container_world_bounds_m':loaded.bounds.tolist(),'hollow_geometry_probe_results':inside,
            'container_triangles':len(mesh.faces),
            'scope':'geometry/scene loading only; no robot-trajectory collision qualification'}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    edges=loaded.vertices[loaded.edges_unique]
    for ax,indices,names in zip(axes,[(0,1),(0,2)],[('world X (m)','world Y (m)'),('world X (m)','world Z (m)')]):
        ax.add_collection(LineCollection(edges[:,:,indices],colors='steelblue',linewidths=.5))
        centre=np.array([x,y,z]);ax.scatter(*centre[list(indices)],color='red',label='Container bottom centre')
        if indices==(0,1):ax.scatter(0,0,color='black',marker='+',label='Robot base')
        else:ax.axhline(top,color='brown',label='Table surface')
        ax.autoscale();ax.margins(.15);ax.set_aspect('equal');ax.grid();ax.set_xlabel(names[0]);ax.set_ylabel(names[1]);ax.legend(fontsize=8)
    fig.suptitle('Virtual container at recorded ArUco centre — reconstructed scene')
    fig.tight_layout();fig.savefig(output/'scene_preview.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
