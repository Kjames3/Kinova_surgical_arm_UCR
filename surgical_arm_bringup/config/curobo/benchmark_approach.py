#!/usr/bin/env python3
"""Offline recorded-endpoint cuRobo benchmark. No ROS, socket, or robot execution."""
import argparse
import hashlib
import math
import json
from pathlib import Path
import time
import platform
import curobo

import numpy as np
import torch
import yaml
from trimesh.transformations import quaternion_from_matrix
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from curobo._src.types.robot import RobotCfg
from curobo._src.types.control_space import ControlSpace
from curobo._src.util_file import get_task_configs_path
from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from validate_fk_vs_urdf import parse_joints, urdf_transform

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dense_path(q, t, max_dt=.01, max_dq=.01):
    """Piecewise-linear sampling, <=10 ms and <=0.01 rad per joint step."""
    q, t = np.asarray(q), np.asarray(t)
    if q.ndim != 2 or len(q) < 2 or len(q) != len(t):
        raise ValueError('Invalid trajectory dimensions')
    if not np.isfinite(q).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Nonfinite trajectory or nonincreasing times')
    out = [q[0]]
    for a, b, dt in zip(q[:-1], q[1:], np.diff(t)):
        n = max(1, int(np.ceil(dt/max_dt)), int(np.ceil(np.max(np.abs(b-a))/max_dq)))
        out.extend(a+(b-a)*f for f in np.linspace(0, 1, n+1)[1:])
    return np.asarray(out)


def normalize_continuous(q, continuous):
    q=np.array(q,dtype=float,copy=True)
    if q.ndim!=2 or not len(q) or q.shape[1]!=len(continuous) or not np.isfinite(q).all():
        raise ValueError('Invalid joint trajectory')
    for i,is_continuous in enumerate(continuous):
        if is_continuous:
            q[:,i]=np.unwrap(q[:,i])
            q[:,i]-=2*np.pi*np.round(q[0,i]/(2*np.pi))
    return q


def interpolation_capacity(max_dt, interpolation_steps, total_knots, output_dt):
    """Conservative buffer for cuRobo's rounded B-spline knot interpolation.

    cuRobo adds one output tick per knot interval before integer rounding.
    Include another tick for floating-point rounding and the final endpoint.
    """
    if not all(np.isfinite(v) and v>0 for v in (max_dt,interpolation_steps,total_knots,output_dt)):
        raise ValueError('Interpolation timing inputs must be finite and positive')
    ticks=math.ceil(max_dt*interpolation_steps/output_dt)+2
    return int(total_knots*ticks+1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture', type=Path)
    p.add_argument('--world', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--robot', type=Path, default=HERE/'gen3_surgical.yml')
    p.add_argument('--max-trajectory-dt', type=float, default=0.3,
                   help='Solver time-step ceiling in seconds (upstream default 0.2); limits stay unchanged')
    p.add_argument('--self-collision-weight', type=float, default=10000.0)
    p.add_argument('--goal-mode', choices=['pose','recorded-joints'], default='pose')
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if not np.isfinite(args.max_trajectory_dt) or args.max_trajectory_dt < .002:
        p.error('--max-trajectory-dt must be finite and >= 0.002 s')
    if args.repeats < 1: p.error('--repeats must be positive')
    if not np.isfinite(args.self_collision_weight) or args.self_collision_weight <= 0:
        p.error('--self-collision-weight must be finite and positive')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(args.fixture.read_text())
    fixture_fingerprint=sha(args.fixture)
    if fixture['outcome'] != 'ok': raise ValueError('Recorded phase did not succeed')
    urdf = args.fixture.resolve().parent/fixture['recorded_urdf']
    if sha(urdf) != fixture['recorded_urdf_sha256']: raise ValueError('Recorded URDF hash mismatch')
    robot = yaml.safe_load(args.robot.read_text())
    world = yaml.safe_load(args.world.read_text())
    params=fixture['scene_parameters']
    velocity_scale=float(params.get('transit_velocity_scaling',-1))
    if velocity_scale<0: velocity_scale=float(params['max_velocity_scaling'])
    if not 0<velocity_scale<=1: raise ValueError('Invalid recorded velocity scale')
    # Apply limits once after loading; this installed cuRobo revision scales
    # velocity both in KinematicsLoader and KinematicsParams when using YAML.
    robot['robot_cfg']['kinematics']['cspace']['velocity_scale']=1.0
    robot['robot_cfg']['kinematics']['cspace']['acceleration_scale']=1.0
    recorded_joints = parse_joints(urdf)
    if not np.allclose(urdf_transform(recorded_joints,'world','base_link',{}),np.eye(4),atol=1e-9):
        raise ValueError('Recorded world/base_link transform is not identity')
    model_joints = parse_joints(robot['robot_cfg']['kinematics']['urdf_path'])
    jt = fixture['trajectory']['joint_trajectory']; names = jt['joint_names']
    if len(names) != 7 or set(names) != {f'joint_{i}' for i in range(1,8)}:
        raise ValueError('Unexpected arm joint names')
    q = np.array([pt['positions'] for pt in jt['points']])
    times = np.array([pt['time_from_start']['sec']+pt['time_from_start']['nanosec']*1e-9 for pt in jt['points']])
    # Continuous joints are unwrapped along the recorded path, then shifted as a
    # whole near zero. Never wrap individual waypoints across the +/-pi seam.
    q = normalize_continuous(q, [recorded_joints[n]['type']=='continuous' for n in names])
    start = dict(zip(names, q[0])); endpoint = dict(zip(names, q[-1]))
    target = urdf_transform(recorded_joints, 'base_link', 'assembly_tip', endpoint)
    for row in q:
        values = dict(zip(names,row))
        if not np.allclose(urdf_transform(recorded_joints,'base_link','assembly_tip',values),
                           urdf_transform(model_joints,'base_link','assembly_tip',values),atol=1e-6):
            raise ValueError('Recorded and planning URDF kinematics disagree')
    measured = fixture['start_joint_state']['positions']
    delta = np.array([measured[n]-start[n] for n in names])
    for i,n in enumerate(names):
        if recorded_joints[n]['type']=='continuous': delta[i]=(delta[i]+np.pi)%(2*np.pi)-np.pi
    if np.max(np.abs(delta)) > .03: raise ValueError('Recorded planned start differs from measured start by >0.03 rad')
    report = {'environment':{'python':platform.python_version(),'torch':torch.__version__,
                             'curobo':getattr(curobo,'__version__','unknown'),'cuda':torch.version.cuda,
                             'gpu':torch.cuda.get_device_name(0)},
              'scope':'Offline free-space endpoint comparison, not reproduction of Pilz blended path constraints.',
              'fixture_sha256':fixture_fingerprint,'robot_sha256':sha(args.robot),'world_sha256':sha(args.world),
              'recorded_urdf_sha256':sha(urdf),
              'planning_urdf_sha256':sha(robot['robot_cfg']['kinematics']['urdf_path']),
              'world_mesh_sha256':{name:sha(mesh['file_path']) for name,mesh in world.get('mesh',{}).items()},
              'goal_mode':args.goal_mode,'seed':123,'frame':'base_link == world','tool_frame':'assembly_tip',
              'start_source':'first recorded planned waypoint; measured-start difference checked',
              'start_measurement_max_error_rad':float(np.max(np.abs(delta))),
              'goal_source':'independent recorded-URDF FK of final planned waypoint; no 5 mm goal adjustment',
              'goal_position_m':target[:3,3].tolist(),'goal_quaternion_wxyz':quaternion_from_matrix(target).tolist(),
              'recorded_motion_wall_s':fixture['phase_duration_s'],
              'recorded_planner_time_s':fixture['preceding_planning_call']['wall_s'],
              'velocity_scale':velocity_scale,'acceleration_scale':velocity_scale*.5,
              'timing_note':'Original preceding plan_start/end wall time includes the blended planning service. Different path constraints/world and acceleration limits preclude a like-for-like speedup claim.',
              'collision_scope':'Conservative robot spheres, existing self-collision exclusions, reconstructed table and hollow mesh; sampled validation, not continuous certification.',
              'runs':[]}
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def tensor(v): return torch.tensor(v,device='cuda',dtype=torch.float32)
    robot_model=RobotCfg.create(robot)
    effective_limits=robot_model.kinematics.get_joint_limits()
    effective_limits.velocity *= velocity_scale
    effective_limits.acceleration *= velocity_scale*.5
    report['effective_limits']={key:getattr(effective_limits,key).cpu().tolist() for key in ('position','velocity','acceleration','jerk')}
    print('Creating validation collision checker',flush=True)
    checker=RobotSceneCollision(RobotSceneCollisionCfg.load_from_config(
        robot_config=robot_model, scene_model=world, collision_activation_distance=0.0))
    ordered=checker.kinematics.joint_names
    indices=[names.index(n) for n in ordered]
    q=q[:,indices]; names=ordered
    def validate_path(path, ts):
        dense=dense_path(path,ts); valid=[]; world_bad=[]; self_bad=[]; bound_bad=[]
        for offset in range(0,len(dense),64):
            values=tensor(dense[offset:offset+64]).unsqueeze(1)
            valid.extend(checker.validate(values).reshape(-1).cpu().tolist())
            kin=checker.get_kinematics(values)
            world_bad.extend((checker.get_collision_constraint(kin).reshape(len(values),-1)>0).any(-1).cpu().tolist())
            self_bad.extend((checker.get_self_collision(kin.robot_spheres).reshape(len(values),-1)>0).any(-1).cpu().tolist())
            bound_bad.extend((checker.get_bound(values).reshape(len(values),-1)>0).any(-1).cpu().tolist())
        return {'passed':all(valid),'samples':len(dense),'invalid_samples':len(valid)-sum(valid),
                'world_collision_samples':sum(world_bad),'self_collision_samples':sum(self_bad),
                'joint_bound_samples':sum(bound_bad),'max_sample_dt_s':.01,'max_sample_joint_step_rad':.01}
    report['recorded_validation']=validate_path(q,times)
    def path_data(path, ts):
        poses=np.array([urdf_transform(recorded_joints,'base_link','assembly_tip',dict(zip(names,row)))[:3,3] for row in path])
        links=['base_link','shoulder_link','half_arm_1_link','half_arm_2_link','forearm_link',
               'spherical_wrist_1_link','spherical_wrist_2_link','bracelet_link','assembly_tip']
        lines=[[urdf_transform(recorded_joints,'base_link',link,dict(zip(names,row)))[:3,3].tolist()
                for link in links] for row in path]
        return {'arm_centrelines_m':lines,'joint_names':names,'positions':path.tolist(),'time_s':ts.tolist(),'tip_positions_m':poses.tolist(),
                'duration_s':float(ts[-1]-ts[0]),'joint_travel_rad':np.abs(np.diff(path,axis=0)).sum(axis=0).tolist(),
                'tip_path_length_m':float(np.linalg.norm(np.diff(poses,axis=0),axis=1).sum())}
    (out/'recorded_trajectory.json').write_text(json.dumps(path_data(q,times),indent=2)+'\n')
    report['recorded_duration_s']=float(times[-1]-times[0])
    report['start_valid']=bool(checker.validate(tensor(q[:1]).unsqueeze(1)).all())
    report['goal_configuration_valid']=bool(checker.validate(tensor(q[-1:]).unsqueeze(1)).all())
    save(); print(json.dumps({k:report[k] for k in ('recorded_validation','start_valid','goal_configuration_valid')}),flush=True)
    if not report['start_valid']:
        report['passed']=False; report['failure']='Recorded start invalid in reconstructed scene';save();return 1
    print('Initializing motion planner (first call can compile CUDA kernels)',flush=True)
    torch.cuda.synchronize(); before=time.perf_counter()
    optimizer=yaml.safe_load((Path(get_task_configs_path())/'trajopt/lbfgs_bspline_trajopt.yml').read_text())
    optimizer['rollout']['constraint_cfg']['self_collision_cfg']['weight']=args.self_collision_weight
    report['self_collision_optimizer_weight']=args.self_collision_weight
    planner_cfg=MotionPlannerCfg.create(robot=robot_model,scene_model=world,random_seed=123,
        position_tolerance=.001,orientation_tolerance=.008726646,self_collision_check=True,
        trajopt_optimizer_configs=[optimizer])
    # The upstream 0.2 s ceiling clips automatic retiming on large joint moves.
    # Configure before constructing the solver/CUDA graphs. No velocity,
    # acceleration, jerk, collision or success thresholds are relaxed.
    planner_cfg.trajopt_solver_config.maximum_trajectory_dt=args.max_trajectory_dt
    report['maximum_trajectory_dt_s']=args.max_trajectory_dt
    planner=MotionPlanner(planner_cfg)
    solver=planner.trajopt_solver
    control_space=solver.auxiliary_rollout.transition_model.control_space
    if control_space not in ControlSpace.bspline_types():
        raise ValueError('Timing buffer sizing requires a B-spline planner')
    total_knots=ControlSpace.spline_total_knots(control_space,solver.action_horizon)
    # Interpolation allocates on the first solve, not in the constructor.
    # The upstream 5000-sample buffer wastes GPU memory with 1222 robot spheres.
    # Derive capacity from the timing ceiling; keep 25 ms output sampling.
    solver.config.interpolation_buffer_size=interpolation_capacity(
        args.max_trajectory_dt,solver.interpolation_steps,total_knots,solver.config.interpolation_dt)
    report['interpolation_buffer_samples']=solver.config.interpolation_buffer_size
    report['interpolation_dt_s']=solver.config.interpolation_dt
    report['timing_spline_total_knots']=total_knots
    torch.cuda.synchronize();report['initialization_s']=time.perf_counter()-before
    if planner.joint_names != names: raise ValueError('Planner and checker joint orders disagree')
    goal=GoalToolPose(tool_frames=['assembly_tip'], position=tensor(target[:3,3]).reshape(1,1,1,1,3),
                     quaternion=tensor(quaternion_from_matrix(target)).reshape(1,1,1,1,4))
    report['graph_recorded_start_goal_valid']=planner.graph_planner.check_samples_feasibility(tensor(q[[0,-1]])).cpu().tolist()
    print('Graph start/goal validity:',report['graph_recorded_start_goal_valid'],flush=True)
    state=JointState.from_position(tensor(q[:1]),joint_names=names)
    for trial in range(args.repeats+1):
        torch.cuda.synchronize();before=time.perf_counter()
        if args.goal_mode=='pose':
            result=planner.plan_pose(goal,state.clone(),max_attempts=5)
        else:
            result=planner.plan_cspace(JointState.from_position(tensor(q[-1:]),joint_names=names),state.clone(),max_attempts=5)
        torch.cuda.synchronize();elapsed=time.perf_counter()-before
        run={'trial':trial,'kind':'first_call' if trial==0 else 'warm','wall_s':elapsed,
             'solver_success':bool(result is not None and result.success.any())}
        if result is not None and result.interpolated_trajectory is not None:
            plan=result.get_interpolated_plan()
            path=plan.position.detach().cpu().numpy().reshape(-1,len(names))
            dt=float(planner.trajopt_solver.config.interpolation_dt)
            ts=np.arange(len(path))*dt
            data=path_data(path,ts)
            end=urdf_transform(recorded_joints,'base_link','assembly_tip',dict(zip(names,path[-1])))
            run.update(duration_s=data['duration_s'],joint_travel_rad=data['joint_travel_rad'],
                       tip_path_length_m=data['tip_path_length_m'],validation=validate_path(path,ts),
                       endpoint_position_error_mm=float(np.linalg.norm(end[:3,3]-target[:3,3])*1000),
                       endpoint_orientation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(end[:3,:3].T@target[:3,:3])-1)/2,-1,1)))),
                       start_error_rad=float(np.max(np.abs(path[0]-q[0]))))
            limits=planner.kinematics.get_joint_limits()
            vel=plan.velocity.detach().cpu().numpy().reshape(-1,len(names))
            max_vel=limits.velocity[1].cpu().numpy()
            run['peak_velocity_limit_ratio']=float(np.max(np.abs(vel)/max_vel))
            acc=plan.acceleration.detach().cpu().numpy().reshape(-1,len(names))
            run['peak_acceleration_limit_ratio']=float(np.max(np.abs(acc)/limits.acceleration[1].cpu().numpy()))
            run['passed']=run['solver_success'] and run['peak_acceleration_limit_ratio']<=1.001 and run['validation']['passed'] and run['endpoint_position_error_mm']<=1 and run['endpoint_orientation_error_deg']<=.5 and run['start_error_rad']<1e-3 and run['peak_velocity_limit_ratio']<=1.001
            data['solver_success']=run['solver_success']
            data['passed']=run['passed']
            (out/f'curobo_trajectory_{trial}.json').write_text(json.dumps(data,indent=2)+'\n')
        else:
            run['passed']=False
            if result is not None:
                for metric_name in ('metrics','interpolated_metrics'):
                    metrics=getattr(result,metric_name,None)
                    if metrics is not None and metrics.costs_and_constraints is not None:
                        constraints=metrics.costs_and_constraints.constraints
                        run[metric_name+'_constraint_max']={n:float(v.max()) for n,v in zip(constraints.names,constraints.values)}
                for key in ('position_error','rotation_error'):
                    value=getattr(result,key,None)
                    if value is not None and torch.isfinite(value).all(): run[key]=value.detach().cpu().tolist()
        report['runs'].append(run);save();print(json.dumps(run),flush=True)
    report['passed']=all(r['passed'] for r in report['runs'])
    report['successful_runs']=sum(r['passed'] for r in report['runs'])
    report['recorded_joint_travel_rad']=np.abs(np.diff(q,axis=0)).sum(axis=0).tolist()
    report['warm_median_s']=float(np.median([r['wall_s'] for r in report['runs'][1:]]))
    report['warm_timing_is_successful_planning']=all(r['passed'] for r in report['runs'][1:])
    save();return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
