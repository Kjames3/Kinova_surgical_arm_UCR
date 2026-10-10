#!/usr/bin/env python3
"""Render offline benchmark artifacts as a standalone HTML animation (system Python)."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from container_geometry import load_container_geometry, validate_scene_spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--scene',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report=json.loads((args.results/'report.json').read_text())
    recorded=json.loads((args.results/'recorded_trajectory.json').read_text())
    candidates=[r for r in report['runs'] if (args.results/f"curobo_trajectory_{r['trial']}.json").exists()]
    if not candidates: raise ValueError('No candidate trajectory to display')
    selected=next((r for r in candidates if r['passed']),candidates[-1])
    planned=json.loads((args.results/f"curobo_trajectory_{selected['trial']}.json").read_text())
    spec=validate_scene_spec(json.loads(args.scene.read_text()))
    vertices,faces,meta=load_container_geometry(args.scene.parent/spec['container_mesh_file'])
    if meta['source_sha256']!=spec['container_mesh_sha256']: raise ValueError('Mesh fingerprint mismatch')
    yaw=spec['container_yaw_rad'];c,s=np.cos(yaw),np.sin(yaw)
    vertices=np.asarray(vertices)@np.array([[c,-s,0],[s,c,0],[0,0,1]]).T+spec['container_center_base_m']
    faces=np.asarray(faces)
    fig=go.Figure()
    fig.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],
                           color='#38bdf8',opacity=.5,name='Virtual container',showlegend=True))
    z=spec['table_surface_z_m'];dx,dy,_=spec['table_size_m']
    fig.add_trace(go.Mesh3d(x=[-dx/2,dx/2,dx/2,-dx/2],y=[-dy/2,-dy/2,dy/2,dy/2],z=[z]*4,
                           i=[0,0],j=[1,2],k=[2,3],color='#b6a896',opacity=.2,name='Table top',showlegend=True))
    paths=[recorded,planned];colors=['#fbbf24','#34d399' if selected['passed'] else '#fb7185']
    labels=['Recorded Pilz', 'cuRobo' if selected['passed'] else 'cuRobo rejected candidate']
    for data,color,label in zip(paths,colors,labels):
        tip=np.asarray(data['tip_positions_m'])
        fig.add_trace(go.Scatter3d(x=tip[:,0],y=tip[:,1],z=tip[:,2],mode='lines',name=label+' tip path',line=dict(color=color,width=6)))
    def arm_trace(data,t,color,label):
        ts=np.asarray(data['time_s']);lines=np.asarray(data['arm_centrelines_m'])
        points=np.array([np.interp(t,ts,lines[:,j,k]) for j in range(lines.shape[1]) for k in range(3)]).reshape(-1,3)
        return go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='lines+markers',name=label+' centreline',
                            line=dict(color=color,width=6),marker=dict(size=3,color=color))
    for data,color,label in zip(paths,colors,labels):fig.add_trace(arm_trace(data,0,color,label))
    timeline=np.linspace(0,max(d['time_s'][-1] for d in paths),100)
    fig.frames=[go.Frame(name=str(i),traces=[4,5],data=[arm_trace(d,t,c,l) for d,c,l in zip(paths,colors,labels)]) for i,t in enumerate(timeline)]
    fig.update_layout(template='plotly_dark',paper_bgcolor='#0b1220',height=720,margin=dict(l=0,r=0,t=50,b=30),
        scene=dict(aspectmode='data',xaxis=dict(title='world X (m)',range=[-.3,.65]),yaxis=dict(title='world Y (m)',range=[-.35,.65]),
                   zaxis=dict(title='world Z (m)',range=[-.07,1.05]),camera=dict(eye=dict(x=1.5,y=-1.7,z=1))),
        legend=dict(x=.01,y=.99),
        updatemenus=[dict(type='buttons',direction='left',x=.01,y=1.07,buttons=[
            dict(label='Play',method='animate',args=[None,{'frame':{'duration':max(20,int(timeline[-1]*1000/99)),'redraw':True},'fromcurrent':True,'transition':{'duration':0}}]),
            dict(label='Pause',method='animate',args=[[None],{'mode':'immediate','frame':{'duration':0,'redraw':False}}])])],
        sliders=[dict(currentvalue={'prefix':'Time (s): '},steps=[dict(label=f'{t:.2f}',method='animate',args=[[str(i)],{'mode':'immediate','frame':{'duration':0,'redraw':True},'transition':{'duration':0}}]) for i,t in enumerate(timeline)])])
    status='PASS' if selected['passed'] else 'REJECTED — inspection only'
    text=f'''<h1>Offline approach comparison</h1><p><b>cuRobo: {status}</b> · {html.escape(report['goal_mode'])} goal · {len(report['runs'])} attempts</p>
<p>Recorded path: {recorded['duration_s']:.2f} s · Candidate: {planned['duration_s']:.2f} s · Planning call: {selected['wall_s']:.3f} s ({selected['kind']})</p>
<p>Yellow: recorded Pilz. {'Green' if selected['passed'] else 'Red'}: cuRobo. Drag to rotate; Play or scrub to compare at the same elapsed time.</p>
<p>Recorded path has {report['recorded_validation']['self_collision_samples']} padded self-collision samples. Candidate has {selected['validation']['self_collision_samples']}. This is a reconstructed scene and a free-space endpoint comparison; original blended constraints differ.</p>'''
    page='<html><head><meta charset="utf-8"><title>Offline approach comparison</title><style>body{background:#0b1220;color:#e2e8f0;font:16px system-ui;margin:24px}h1{font-size:25px}p{max-width:1100px;line-height:1.5}</style></head><body>'+text
    page+=fig.to_html(full_html=False,include_plotlyjs=True,auto_play=False,config={'scrollZoom':True,'displaylogo':False})
    page+='<p>Offline animation only. Joint centrelines are reference guides; collision validation uses the robot spheres. No robot commands are sent.</p></body></html>'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(page)
    print(args.output)


if __name__=='__main__':main()
