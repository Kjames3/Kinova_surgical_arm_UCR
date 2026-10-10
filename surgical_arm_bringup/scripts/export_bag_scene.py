#!/usr/bin/env python3
"""Export an explicitly reconstructed scene at the recorded ArUco target.
Run with ROS system Python. Reads SQLite/CDR only; no ROS node or publication.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import shutil
import xml.etree.ElementTree as ET

from rclpy.serialization import deserialize_message
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray
from container_geometry import load_container_geometry, validate_scene_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--container-base-offset', type=float, required=True,
                        help='Container support height above table, e.g. marker board thickness (m)')
    parser.add_argument('--mesh', type=Path, default=Path(__file__).resolve().parents[4] / 'src/ros2_kortex/kortex_description/grippers/thesis_ee/meshes/Glass_container.STL')
    args = parser.parse_args()
    params, target, urdfs = None, None, set()
    corners = []
    target_time = None
    files = sorted(args.bag.resolve().glob('*.db3'))
    if not files:
        raise ValueError('No SQLite bag files found')
    for path in files:
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as connection:
            topics = {n:i for i,n in connection.execute('SELECT id,name FROM topics')}
            ids = [topics[n] for n in ['/insertion/run_info','/robot_description','/fused_corners'] if n in topics]
            for tid, stamp, data in connection.execute('SELECT topic_id,timestamp,data FROM messages WHERE topic_id IN ('+','.join('?'*len(ids))+') ORDER BY timestamp', ids):
                if tid == topics.get('/fused_corners'):
                    array = deserialize_message(data, PoseArray)
                    if array.header.frame_id == 'world' and len(array.poses) == 4:
                        corners.append((stamp, [[p.position.x,p.position.y] for p in array.poses]))
                    continue
                text = deserialize_message(data, String).data
                if tid == topics.get('/robot_description'):
                    urdfs.add(text)
                else:
                    item = json.loads(text)
                    if item.get('event') == 'params': params = item
                    elif item.get('event') == 'target': target = item; target_time = stamp
    if not params or not target or target.get('source') != 'markers' or target.get('frame') != 'world':
        raise ValueError('Requires recorded marker target and parameters in world')
    if len(urdfs) != 1:
        raise ValueError('Missing or conflicting recorded robot descriptions')
    if not corners:
        raise ValueError('Board orientation requires /fused_corners in world')
    corner_time, points = min(corners, key=lambda entry: abs(entry[0]-target_time))
    if abs(corner_time-target_time) > 500_000_000:
        raise ValueError('No board corners within 0.5 seconds of recorded target')
    tl,tr,br,bl = points
    vx,vy = tr[0]-tl[0]+br[0]-bl[0],tr[1]-tl[1]+br[1]-bl[1]
    if math.hypot(vx,vy) < .01:
        raise ValueError('Degenerate marker-board edge')
    yaw = math.atan2(vy,vx)
    urdf = next(iter(urdfs)); root = ET.fromstring(urdf)
    mount = next(j for j in root.findall('joint') if j.find('child').get('link') == 'base_link')
    origin = mount.find('origin')
    if mount.find('parent').get('link') != 'world' or mount.get('type') != 'fixed' or any(
            abs(float(x)) > 1e-9 for key in ['xyz','rpy'] for x in origin.get(key,'0 0 0').split()):
        raise ValueError('world != base_link: explicit transform required for cuRobo')
    _, _, geometry = load_container_geometry(args.mesh)
    if abs(geometry['dimensions_m'][2] - params['container_height']) > .001:
        raise ValueError('Recorded container height disagrees with CAD')
    spec = {'version':1, 'frame_id':'world', 'table_size_m':[2.0,2.0,.05],
            'table_surface_z_m':params['table_z'],
            'container_center_base_m':[target['x'],target['y'],params['table_z']+args.container_base_offset],
            'container_mesh_sha256':geometry['source_sha256'],
            'container_mesh_file':'Glass_container.STL',
            'container_yaw_rad':yaw,
            'container_dimensions_m':geometry['dimensions_m'],
            'source_bag':str(args.bag.resolve()),'source_target':target,
            'robot_description_sha256':hashlib.sha256(urdf.encode()).hexdigest(),
            'provenance':{'scene':'reconstructed; physical container absent during recording',
                          'xy':'recorded insertion marker target',
                          'table_z':'recorded insertion parameter; not independently remeasured',
                          'table_size':'existing setup_planning_scene defaults',
                          'container_base_offset_m':args.container_base_offset,
                          'yaw':'align rounded-square container sides to recorded board TL->TR edge; physical alignment assumed',
                          'board_corners_xy_m':points, 'board_sample_age_s':(target_time-corner_time)/1e9,
                          'collision_mesh':'original triangles, reoriented and centred; no convex hull'}}
    validate_scene_spec(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.mesh, args.output.parent / 'Glass_container.STL')
    args.output.write_text(json.dumps(spec,indent=2)+'\n')
    print(f'Wrote {args.output}; container bottom centre {spec["container_center_base_m"]}')


if __name__ == '__main__':
    main()
