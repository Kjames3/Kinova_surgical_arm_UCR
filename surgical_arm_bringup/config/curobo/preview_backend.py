#!/usr/bin/env python3
"""Qualified approach settings behind the planning-only socket endpoint.

Inputs are a full MoveIt scene and URDF, not stale scene files. Unsupported
geometry is rejected. All planning is offline; this module has no ROS APIs.
"""
import hashlib
import copy
import json
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh
import yaml
from trimesh.transformations import quaternion_matrix, quaternion_from_matrix
from benchmark_approach import (dense_path, normalize_continuous, interpolation_capacity,
    RobotCfg, RobotSceneCollision, RobotSceneCollisionCfg, MotionPlanner, MotionPlannerCfg,
    ControlSpace, get_task_configs_path, GoalToolPose, JointState, parse_joints, urdf_transform)

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

HERE = Path(__file__).resolve().parent


def pose_matrix(p):
    q=p['orientation']; xyz=p['position']
    quat=np.array([q[k] for k in ('w','x','y','z')], dtype=float)
    if not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1.) > 1e-4:
        raise ValueError('Invalid scene quaternion')
    result=quaternion_matrix(quat)
    result[:3,3]=[xyz[k] for k in ('x','y','z')]
    if not np.isfinite(result).all(): raise ValueError('Invalid scene position')
    return result


def world_from_scene(scene, directory):
    if any(p['padding'] != 0 for p in scene.get('link_padding',[])) or any(p['scale'] != 1 for p in scene.get('link_scale',[])):
        raise ValueError('Nondefault MoveIt link padding/scale is unsupported')
    if scene['is_diff'] or scene['robot_state']['attached_collision_objects'] or scene['world']['octomap']['octomap']['data']:
        raise ValueError('Full scene without attached objects/octomap required')
    world={'cuboid':{},'mesh':{}}; container_bounds=None
    for obj in scene['world']['collision_objects']:
        if obj['header']['frame_id'] not in ('world','base_link') or obj['operation'] not in (0, '\x00', '\\u0000'):
            raise ValueError('Scene object must be ADD in world/base_link')
        if obj['planes'] or obj.get('subframe_names'):
            raise ValueError('Planes/subframes are unsupported')
        base=pose_matrix(obj['pose']) if 'pose' in obj else np.eye(4)
        if len(obj['primitives']) != len(obj['primitive_poses']) or len(obj['meshes']) != len(obj['mesh_poses']):
            raise ValueError('Geometry/pose count mismatch')
        if not obj['primitives'] and not obj['meshes']: raise ValueError('Empty collision object')
        for i,(primitive,pose) in enumerate(zip(obj['primitives'],obj['primitive_poses'])):
            dims=primitive['dimensions']
            if primitive['type'] != 1 or len(dims)!=3 or not np.isfinite(dims).all() or min(dims)<=0:
                raise ValueError('Only finite positive box primitives are supported')
            mat=base@pose_matrix(pose)
            world['cuboid'][f'{obj["id"]}_box_{i}']={'dims':dims,'pose':[*mat[:3,3],*quaternion_from_matrix(mat)]}
        for i,(mesh,pose) in enumerate(zip(obj['meshes'],obj['mesh_poses'])):
            vertices=np.array([[v[k] for k in ('x','y','z')] for v in mesh['vertices']])
            faces=np.array([t['vertex_indices'] for t in mesh['triangles']])
            if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all() or faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(vertices):
                raise ValueError('Invalid triangle mesh')
            mat=base@pose_matrix(pose)
            path=directory/f'mesh_{len(world["mesh"])}.obj'
            trimesh.Trimesh(vertices=vertices,faces=faces,process=False).export(path)
            world['mesh'][f'{obj["id"]}_mesh_{i}']={'file_path':str(path),'pose':[*mat[:3,3],*quaternion_from_matrix(mat)]}
            if obj['id']=='glass_container':
                if container_bounds is not None: raise ValueError('Multiple container meshes')
                transformed=vertices@mat[:3,:3].T+mat[:3,3]
                container_bounds=np.array([transformed.min(0),transformed.max(0)])
    ids=[o['id'] for o in scene['world']['collision_objects']]
    if len(set(ids))!=len(ids) or not {'table','glass_container'}<=set(ids) or container_bounds is None:
        raise ValueError('Unique scene objects including table and hollow container required')
    return world,container_bounds


def tool_tilts_deg(joints, names, path, vertical_rotation):
    """Shaft tilt from world down, calibrated by the tool-down bracelet pose.

    Do not use bracelet->TCP translation: the tool has a lateral offset.
    Yaw about the vertical axis does not change this acceptance metric.
    """
    local_axis = vertical_rotation.T @ np.array([0., 0., -1.])
    axes = np.array([urdf_transform(joints, 'base_link', 'bracelet_link', dict(zip(names,row)))[:3,:3] @ local_axis for row in path])
    return np.degrees(np.arccos(np.clip(-axes[:,2], -1., 1.)))


def require_upright(angles, limit):
    if not np.isfinite(angles).all() or np.max(angles) > limit:
        raise ValueError(f'Upright approach exceeds {limit:g} deg: maximum {np.max(angles):.3f} deg')


def upright_policy(approach):
    limit=float(approach.get('max_upright_tilt_deg',15.))
    if not np.isfinite(limit) or not 0 < limit <= 15.:
        raise ValueError('max_upright_tilt_deg must be in (0,15]')
    skip=approach.get('skip_home_move',True)
    if not isinstance(skip,bool): raise ValueError('skip_home_move must be boolean')
    return limit, skip


class PreviewPlanner:
    name='curobo'

    def __init__(self):
        self.directory=tempfile.TemporaryDirectory(prefix='curobo_preview_')
        self.key=None

    def plan(self, req):
        if req.get('preview_only') is not True or req['goal_type']!='ptp_pose':
            raise ValueError('Only planning-only free-space approach requests are supported')
        names=[f'joint_{i}' for i in range(1,8)]
        if req['joint_names']!=names or set(req['start_joints'])!=set(names):
            raise ValueError('Expected canonical seven arm joints')
        scale=float(req['vel_scale'])
        if not np.isfinite(scale) or not 0<scale<=.15: raise ValueError('Preview scale must be in (0,0.15]')
        directory=Path(self.directory.name)
        live_path=directory/'live.urdf'; live_path.write_text(req['robot_description'], encoding='utf-8')
        live=parse_joints(live_path)
        robot=yaml.safe_load((HERE/'gen3_surgical.yml').read_text(encoding='utf-8'))
        model=parse_joints(robot['robot_cfg']['kinematics']['urdf_path'])
        # The spheres qualify these collision shapes. Reject URDF shape changes
        # even if joint transforms still agree; mesh filesystem roots may differ.
        def collision_shapes(root, live_preview=False):
            shapes={}
            for link in root.findall('link'):
                if link.get('name') not in robot['robot_cfg']['kinematics']['collision_link_names']:
                    continue
                entries=[]
                source_tag = 'visual' if live_preview and link.get('name') == 'thesis_ee' else 'collision'
                for source in link.findall(source_tag):
                    collision = ET.Element('collision')
                    for tag in ('origin', 'geometry'):
                        if source.find(tag) is not None:
                            collision.append(copy.deepcopy(source.find(tag)))
                    for mesh in collision.iter('mesh'):
                        mesh.set('filename',Path(mesh.get('filename')).name)
                    for element in collision.iter():
                        element.text=None; element.tail=None
                    entries.append(ET.tostring(collision))
                shapes[link.get('name')]=entries
            return shapes
        if collision_shapes(ET.fromstring(req['robot_description']), live_preview=True) != collision_shapes(ET.parse(HERE/'gen3_surgical.urdf').getroot()):
            raise ValueError('Live URDF collision geometry differs from sphere-qualified model')
        # Compare each complete collision-link chain, not only one FK sample.
        child_of={j['child']:(n,j) for n,j in model.items()}
        for link in [*robot['robot_cfg']['kinematics']['collision_link_names'],'assembly_tip']:
            while link!='base_link':
                name,joint=child_of[link]
                if name not in live or live[name]!=joint:
                    raise ValueError(f'Live/model kinematics differ at {name}')
                link=joint['parent']
        if not np.allclose(urdf_transform(live,'world','base_link',{}),np.eye(4),atol=1e-9):
            raise ValueError('world/base_link must be identity')
        world,bounds=world_from_scene(req['scene'], directory)
        approach=req['approach']
        values=np.array([approach[k] for k in ('clearance','depth','tilt_deg','azimuth_deg')])
        if not np.isfinite(values).all() or not .1<=approach['clearance']<=.3 or not 0<=approach['depth']<=.039 or not 0<=approach['tilt_deg']<=45:
            raise ValueError('Approach geometry outside supported range')
        x,y=(bounds[0,:2]+bounds[1,:2])/2
        z=bounds[1,2]+approach['clearance']
        azimuth=np.arctan2(y,x)+np.pi/2 if approach['tangential'] else np.radians(approach['azimuth_deg'])
        offset=(approach['clearance']+approach['depth'])*np.tan(np.radians(approach['tilt_deg']))
        goal_pos=np.array([x-offset*np.cos(azimuth),y-offset*np.sin(azimuth),z])
        quat=np.asarray(approach['quaternion_xyzw'],dtype=float)[[3,0,1,2]]
        if not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1)> .01: raise ValueError('Bad goal quaternion')
        quat/=np.linalg.norm(quat)
        # Input orientation describes bracelet_link, while goal is assembly_tip.
        fixed=urdf_transform(live,'bracelet_link','assembly_tip',{})
        target=quaternion_matrix(quat)@fixed
        target[:3,3]=goal_pos
        continuous=[live[n]['type']=='continuous' for n in names]
        original=np.array([[req['start_joints'][n] for n in names]])
        q=normalize_continuous(original,continuous)
        winding=original[0]-q[0]
        def tensor(v): return torch.tensor(v,device='cuda',dtype=torch.float32)
        # Scene timestamps and live robot state do not change the collision model.
        key=hashlib.sha256(json.dumps({'world':world,'scale':scale},sort_keys=True).encode()).hexdigest()
        # Include actual mesh bytes, since paths remain stable between requests.
        key+=hashlib.sha256(b''.join(Path(m['file_path']).read_bytes() for m in world['mesh'].values())).hexdigest()
        if self.key is not None and key!=self.key:
            raise ValueError('Scene/scale changed: restart sidecar to rebuild CUDA graphs')
        if self.key is None:
            robot['robot_cfg']['kinematics']['cspace']['velocity_scale']=1.
            robot['robot_cfg']['kinematics']['cspace']['acceleration_scale']=1.
            model_cfg=RobotCfg.create(robot)
            limits=model_cfg.kinematics.get_joint_limits(); limits.velocity*=scale; limits.acceleration*=scale*.5
            self.checker=RobotSceneCollision(RobotSceneCollisionCfg.load_from_config(robot_config=model_cfg,scene_model=world,collision_activation_distance=0.))
            optimizer=yaml.safe_load((Path(get_task_configs_path())/'trajopt/lbfgs_bspline_trajopt.yml').read_text(encoding='utf-8'))
            optimizer['rollout']['constraint_cfg']['self_collision_cfg']['weight']=10000.
            cfg=MotionPlannerCfg.create(robot=model_cfg,scene_model=world,random_seed=123,position_tolerance=.001,
                orientation_tolerance=.008726646,self_collision_check=True,trajopt_optimizer_configs=[optimizer])
            cfg.trajopt_solver_config.maximum_trajectory_dt=.3
            self.planner=MotionPlanner(cfg)
            solver=self.planner.trajopt_solver
            control=solver.auxiliary_rollout.transition_model.control_space
            if control not in ControlSpace.bspline_types(): raise ValueError('B-spline required')
            solver.config.interpolation_buffer_size=interpolation_capacity(.3,solver.interpolation_steps,
                ControlSpace.spline_total_knots(control,solver.action_horizon),solver.config.interpolation_dt)
            if self.planner.joint_names!=names or self.checker.kinematics.joint_names!=names: raise ValueError('Joint order mismatch')
            self.key=key
        checker=self.checker; planner=self.planner
        if not checker.validate(tensor(q).unsqueeze(1)).all(): raise ValueError('Measured start collides or violates bounds')
        limit, skip_ready = upright_policy(approach)
        vertical_rotation = quaternion_matrix(quat)[:3,:3]
        start_tilt = float(tool_tilts_deg(live,names,q,vertical_rotation)[0])
        if skip_ready and start_tilt > limit:
            raise ValueError(f'Start tilt {start_tilt:.2f} deg exceeds {limit:g}; use skip_home_move:=false to preview the separate ready transition')
        stages=[]

        def solve_stage(start, label, goal_matrix=None, joint_goal=None, upright=False):
            # Nonterminal orientation guidance is deliberately tighter than the
            # hard acceptance cone. A soft cost alone is not a safety check.
            criteria = ToolPoseCriteria(
                non_terminal_pose_axes_weight_factor=[0.,0.,0.,1.,1.,1.] if upright else [0.]*6,
                non_terminal_pose_convergence_tolerance=[0.,np.radians(min(2.,limit/2))] if upright else [0.,0.])
            planner.update_tool_pose_criteria({'assembly_tip':criteria})
            state=JointState.from_position(tensor(start),joint_names=names)
            torch.cuda.synchronize(); before=time.perf_counter()
            try:
                if joint_goal is not None:
                    result=planner.plan_cspace(JointState.from_position(tensor(joint_goal),joint_names=names),state,max_attempts=5)
                else:
                    goal=GoalToolPose(tool_frames=['assembly_tip'],position=tensor(goal_matrix[:3,3]).reshape(1,1,1,1,3),
                        quaternion=tensor(quaternion_from_matrix(goal_matrix)).reshape(1,1,1,1,4))
                    result=planner.plan_pose(goal,state,max_attempts=5)
            finally:
                planner.update_tool_pose_criteria({'assembly_tip':ToolPoseCriteria()})
            torch.cuda.synchronize(); elapsed=time.perf_counter()-before
            if result is None or not result.success.any() or result.interpolated_trajectory is None:
                raise ValueError(f'{label}: cuRobo solver did not succeed')
            plan=result.get_interpolated_plan()
            path=plan.position.cpu().numpy().reshape(-1,7)
            velocity=plan.velocity.cpu().numpy().reshape(-1,7)
            acceleration=plan.acceleration.cpu().numpy().reshape(-1,7)
            ts=np.arange(len(path))*planner.trajopt_solver.config.interpolation_dt
            dense=dense_path(path,ts)
            for offset in range(0,len(dense),64):
                if not checker.validate(tensor(dense[offset:offset+64]).unsqueeze(1)).all():
                    raise ValueError(f'{label}: dense sampled path fails collision/bound validation')
            tilts=tool_tilts_deg(live,names,dense,vertical_rotation)
            if upright: require_upright(tilts,limit)
            end=urdf_transform(live,'base_link','assembly_tip',dict(zip(names,path[-1])))
            expected=goal_matrix if goal_matrix is not None else urdf_transform(live,'base_link','assembly_tip',dict(zip(names,joint_goal[0])))
            pos_err=np.linalg.norm(end[:3,3]-expected[:3,3])
            rot_err=np.arccos(np.clip((np.trace(end[:3,:3].T@expected[:3,:3])-1)/2,-1,1))
            limits=planner.kinematics.get_joint_limits()
            vr=float(np.max(np.abs(velocity)/limits.velocity[1].cpu().numpy()))
            ar=float(np.max(np.abs(acceleration)/limits.acceleration[1].cpu().numpy()))
            if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all() or vr>1.001 or ar>1.001 or pos_err>.001 or rot_err>np.radians(.5) or np.max(np.abs(path[0]-start[0]))>=1e-4:
                raise ValueError(f'{label}: independent endpoint/start/derivative validation failed')
            if joint_goal is not None and np.max(np.abs(path[-1]-joint_goal[0])) > .001:
                raise ValueError('Ready joint goal not reached')
            if np.max(np.abs(velocity[[0,-1]])) > .001 or np.max(np.abs(acceleration[[0,-1]])) > .01:
                raise ValueError(f'{label}: stage endpoints must be at rest')
            meta=dict(name=label,planning_s=elapsed,duration_s=float(ts[-1]),validation_samples=len(dense),
                position_error_mm=float(pos_err*1000),orientation_error_deg=float(np.degrees(rot_err)),
                peak_velocity_ratio=vr,peak_acceleration_ratio=ar,max_tilt_deg=float(np.max(tilts)),
                start_tilt_deg=float(tilts[0]),end_tilt_deg=float(tilts[-1]),upright_required=upright)
            stages.append((path,velocity,acceleration,ts,meta))
            return path[-1:]

        stage_start=q
        already_ready=False
        if not skip_ready:
            ready=approach.get('ready_joints',{})
            if set(ready)!=set(names): raise ValueError('Complete configured ready joint pose required')
            ready_q=np.array([[ready[n] for n in names]],dtype=float)
            if not np.isfinite(ready_q).all(): raise ValueError('Invalid ready pose')
            for i,c in enumerate(continuous):
                if c: ready_q[0,i]=q[0,i]+np.remainder(ready_q[0,i]-q[0,i]+np.pi,2*np.pi)-np.pi
            require_upright(tool_tilts_deg(live,names,ready_q,vertical_rotation),min(2.,limit))
            if not checker.validate(tensor(ready_q).unsqueeze(1)).all(): raise ValueError('Configured ready pose collides or violates bounds')
            already_ready=bool(np.max(np.abs(ready_q-q))<1e-3)
            if not already_ready:
                stage_start=solve_stage(q,'ready_transition',joint_goal=ready_q,upright=start_tilt<=limit)
            require_upright(tool_tilts_deg(live,names,stage_start,vertical_rotation),min(2.,limit))
        solve_stage(stage_start,'upright_approach',goal_matrix=target,upright=True)
        points=[]; stage_meta=[]; elapsed_t=0.
        for path,velocity,acceleration,ts,meta in stages:
            if points:
                if np.max(np.abs(np.array(points[-1]['positions'])-(path[0]+winding)))>1e-4:
                    raise ValueError('Stage join discontinuity')
                # Duplicate stationary endpoint produces an explicit 0.1 s ready hold.
                elapsed_t+=.1
            meta['start_time_s']=elapsed_t
            meta['end_time_s']=elapsed_t+float(ts[-1])
            stage_meta.append(meta)
            angles=tool_tilts_deg(live,names,path,vertical_rotation)
            points.extend({'positions':(row+winding).tolist(),'velocities':v.tolist(),'accelerations':a.tolist(),
                'time_from_start':float(elapsed_t+t),'stage':meta['name'],'tilt_deg':float(angle)}
                for row,v,a,t,angle in zip(path,velocity,acceleration,ts,angles))
            elapsed_t+=float(ts[-1])
        return {'protocol':1,'success':True,'joint_names':names,'points':points,'error':'',
            'meta':{'backend':'curobo','preview_only':True,'validated':True,
                'planning_s':sum(m['planning_s'] for m in stage_meta),'duration_s':elapsed_t,
                'validation_samples':sum(m['validation_samples'] for m in stage_meta),
                'upright_policy_version':1,'stages':stage_meta,'max_upright_tilt_deg':limit,'approach_max_tilt_deg':stage_meta[-1]['max_tilt_deg'],
                'start_tilt_deg':start_tilt,'already_at_ready':already_ready,'shaft_axis_in_bracelet':(vertical_rotation.T@np.array([0.,0.,-1.])).tolist(),
                'goal_position_m':goal_pos.tolist(),'container_bounds_m':bounds.tolist(),
                'live_urdf_sha256':hashlib.sha256(req['robot_description'].encode()).hexdigest(),
                'robot_config_sha256':hashlib.sha256((HERE/'gen3_surgical.yml').read_bytes()).hexdigest(),
                'tool_geometry':'Provisional visual-CAD substitution, matching regenerate_urdf.py; live collision mesh differs',
                'scope':'Ready transition may tilt; approach must stay within configured vertical cone. Planning-only, sampled validation.'}}
