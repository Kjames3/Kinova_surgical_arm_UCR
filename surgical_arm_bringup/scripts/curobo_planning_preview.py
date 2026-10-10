#!/usr/bin/env python3
"""Read-only cuRobo approach preview. No motion/action or scene-write APIs."""
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone


def check_preview_parameters(params):
    for flag in ('execute_motion', 'teach_mode', 'replay_mode', 'use_action_server',
                 'direct_to_angled_hover', 'use_current_orientation'):
        if params.get(flag, False):
            raise ValueError(f'cuRobo approach preview requires {flag}:=false')
    if params.get('azimuth_mode', '') == 'search':
        raise ValueError('cuRobo preview does not yet support azimuth search')
    if params.get('world_frame', 'world') != 'world' or params.get('tip_link', 'assembly_tip') != 'assembly_tip':
        raise ValueError('cuRobo preview requires world and assembly_tip frames')
    tilt = params.get('curobo_max_tilt_deg', 15.)
    if not math.isfinite(tilt) or not 0 < tilt <= 15.:
        raise ValueError('curobo_max_tilt_deg must be in (0,15]')
    scale = params.get('transit_velocity_scaling', -1.)
    if scale <= 0: scale = params.get('max_velocity_scaling', .25)
    if not math.isfinite(scale) or not 0 < scale <= .15:
        raise ValueError('cuRobo preview currently requires 0 < transit_velocity_scaling <= 0.15')


def validate_upright_response(response, limit):
    meta=response.get('meta',{})
    if meta.get('upright_policy_version') != 1:
        raise ValueError('Sidecar lacks upright-path validation; restart it with current code')
    maximum=meta.get('approach_max_tilt_deg',float('nan'))
    if not math.isfinite(maximum) or maximum > limit or meta.get('max_upright_tilt_deg') != limit:
        raise ValueError('Sidecar upright validation does not satisfy the requested tilt limit')
    stages=meta.get('stages',[])
    if not stages or stages[-1].get('name') != 'upright_approach' or stages[-1].get('upright_required') is not True:
        raise ValueError('Missing validated upright approach stage')


def scene_geometry_signature(scene):
    import copy
    data=copy.deepcopy(scene)
    objects=data['world']['collision_objects']
    for obj in objects:
        obj['header'].pop('stamp',None)
    data['world']['collision_objects']=sorted(objects,key=lambda o:o['id'])
    return json.dumps({'world':data['world'], 'attached':data['robot_state']['attached_collision_objects'],
        'padding':data.get('link_padding',[]), 'scale':data.get('link_scale',[])},sort_keys=True)


def run_preview(node):
    from rcl_interfaces.srv import GetParameters
    from moveit_msgs.srv import GetPlanningScene
    from moveit_msgs.msg import DisplayTrajectory, PlanningSceneComponents
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from curobo_planner_client import CuroboPlannerClient
    import curobo_planner_protocol as proto

    params = {n: p.value for n, p in node._parameters.items()}
    check_preview_parameters(params)
    out = Path(params['curobo_result_dir']).expanduser()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, mode=0o700)
    node.get_logger().info(f'cuRobo PLANNING ONLY; artifacts: {out}')

    def call(client, request):
        if not client.wait_for_service(timeout_sec=5.):
            raise RuntimeError(f'{client.srv_name} unavailable')
        result = node._wait_for_future(client.call_async(request), 10.)
        if result is None: raise RuntimeError(f'{client.srv_name} timed out')
        return result

    cli = node.create_client(GetParameters, '/move_group/get_parameters', callback_group=node._cb_group)
    req = GetParameters.Request(); req.names = ['robot_description']
    urdf = call(cli, req).values[0].string_value
    if not urdf: raise ValueError('MoveIt did not provide its URDF')
    req = GetPlanningScene.Request()
    req.components.components = (PlanningSceneComponents.SCENE_SETTINGS | PlanningSceneComponents.ROBOT_STATE
        | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        | PlanningSceneComponents.OCTOMAP | PlanningSceneComponents.TRANSFORMS
        | PlanningSceneComponents.LINK_PADDING_AND_SCALING)
    scene = call(node._scene_snapshot_cli, req).scene
    if scene.is_diff: raise ValueError('Full scene required')
    if scene.robot_state.attached_collision_objects or scene.world.octomap.octomap.data:
        raise ValueError('Attached objects/octomap are not supported; refusing to omit them')
    objects = scene.world.collision_objects
    if not {'table', 'glass_container'} <= {o.id for o in objects}:
        raise ValueError('Live scene must contain table and glass_container; run setup_planning_scene first')
    container = next(o for o in objects if o.id == 'glass_container')
    if container.header.frame_id not in ('world', 'base_link') or len(container.mesh_poses) != 1:
        raise ValueError('Expected one container mesh in world/base_link')
    # Scene poses are composed and validated by the sidecar. Send all geometry,
    # not a filename or guessed replacement; choose the goal there from its bounds.
    deadline = time.monotonic()+5.
    previous = getattr(node, '_preview_joint_sample', (0., {}))[0]
    while getattr(node, '_preview_joint_sample', (0., {}))[0] <= previous:
        if time.monotonic() > deadline: raise RuntimeError('No fresh complete arm joint state')
        time.sleep(.02)
    _, start = node._preview_joint_sample
    request = proto.make_request(proto.GOAL_PTP, preview_only=True, joint_names=list(node.arm_joint_names),
        start_joints=start, robot_description=urdf, scene=message_to_ordereddict(scene),
        vel_scale=params['transit_velocity_scaling'] if params['transit_velocity_scaling']>0 else params['max_velocity_scaling'],
        approach={'skip_home_move':params['skip_home_move'], 'ready_joints':node.home_joints,
                  'max_upright_tilt_deg':params['curobo_max_tilt_deg'],
                  'clearance':params['approach_clearance'], 'depth':params['target_depth_mm']/1000.,
                  'tilt_deg':params['insertion_angle_deg'], 'azimuth_deg':params['insertion_azimuth_deg'],
                  'tangential':params['azimuth_mode']=='tangential' or (not params['azimuth_mode'] and params['auto_azimuth']),
                  'quaternion_xyzw':[params['vertical_quat_'+a] for a in 'xyzw']})
    (out/'request.json').write_text(json.dumps(request, indent=2, allow_nan=False)+'\n')
    client = CuroboPlannerClient(params['curobo_socket'], timeout=180.)
    response = client._request(request)
    (out/'response.json').write_text(json.dumps(response, indent=2, allow_nan=False)+'\n')
    if response.get('meta', {}).get('backend') != 'curobo' or not response.get('meta', {}).get('validated'):
        raise ValueError(response.get('error') or 'Sidecar did not return a validated cuRobo plan')
    validate_upright_response(response, params['curobo_max_tilt_deg'])
    traj = client.to_robot_trajectory(response)
    # Display only if the stationary arm still agrees with the captured start.
    sample_time, current = node._preview_joint_sample
    if time.monotonic()-sample_time > .5:
        raise ValueError('Joint state stale after planning')
    for n, q in start.items():
        delta = current[n]-q
        if n in ('joint_1','joint_3','joint_5','joint_7'): delta = math.remainder(delta, 2*math.pi)
        if abs(delta) > .01: raise ValueError('Arm moved during planning; preview rejected')
    # Reject a scene update during the solve; never display an old-world plan
    # as if it were validated against the newly returned world.
    latest_scene = call(node._scene_snapshot_cli, req).scene
    if scene_geometry_signature(message_to_ordereddict(latest_scene)) != scene_geometry_signature(request['scene']):
        raise ValueError('Planning scene changed during solve; retry with a fresh sidecar')
    (out/'preview_status.json').write_text(json.dumps({'passed':True, 'motion_sent':False,
        'start_max_change_rad':max(abs(math.remainder(current[n]-q, 2*math.pi)) for n,q in start.items()),
        'scene_unchanged':True}, indent=2)+'\n')
    display = DisplayTrajectory(); display.model_id = 'gen3'
    display.trajectory_start = scene.robot_state
    display.trajectory_start.joint_state.name = list(start)
    display.trajectory_start.joint_state.position = list(start.values())
    display.trajectory_start.joint_state.velocity = []; display.trajectory_start.joint_state.effort = []
    display.trajectory = [traj]
    node._curobo_display_pub = node.create_publisher(DisplayTrajectory, '/display_planned_path',
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node._curobo_display_pub.publish(display)
    node._traj_pub.publish(traj)
    node.get_logger().info('cuRobo preview PASS: '+json.dumps(response['meta']))
    node.get_logger().info('Display published; no motion sent. Holding display publisher for 5 seconds.')
    time.sleep(5.)
    return response
