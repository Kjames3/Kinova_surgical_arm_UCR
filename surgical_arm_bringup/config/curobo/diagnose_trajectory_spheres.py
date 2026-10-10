#!/usr/bin/env python3
"""Diagnose closest non-excluded nominal sphere pairs along saved waypoints."""
import argparse,json,sys
from pathlib import Path
import numpy as np,yaml
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from validate_fk_vs_urdf import parse_joints,urdf_transform
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('trajectory',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
cfg=yaml.safe_load((HERE/'gen3_surgical.yml').read_text())['robot_cfg']['kinematics']
joints=parse_joints(cfg['urdf_path'])
path=json.loads(args.trajectory.read_text())
worst={}
for k,row in enumerate(path['positions']):
    world={}
    for link,spheres in cfg['collision_spheres'].items():
        tf=urdf_transform(joints,'base_link',link,dict(zip(path['joint_names'],row)))
        centers=np.array([s['center'] for s in spheres]);radii=np.array([s['radius'] for s in spheres])
        world[link]=(centers@tf[:3,:3].T+tf[:3,3],radii)
    for i,a in enumerate(world):
        for b in list(world)[i+1:]:
            if b in cfg['self_collision_ignore'].get(a,[]) or a in cfg['self_collision_ignore'].get(b,[]):continue
            pa,ra=world[a];pb,rb=world[b]
            dist=np.linalg.norm(pa[:,None]-pb,axis=-1)-ra[:,None]-rb
            d=float(dist.min());key=a+' / '+b
            if key not in worst or d<worst[key]['nominal_separation_m']:
                worst[key]={'links':[a,b],'nominal_separation_m':d,'waypoint':k,'time_s':path['time_s'][k],
                            'padded_separation_m':d-2*cfg['collision_sphere_buffer'],
                            'sphere_indices':list(map(int,np.unravel_index(dist.argmin(),dist.shape)))}
result=sorted(worst.values(),key=lambda r:r['nominal_separation_m'])
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result[:5],indent=2))
