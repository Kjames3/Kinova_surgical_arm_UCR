#!/usr/bin/env python3
"""Regenerate and qualify the offline model; retain inputs, logs and failures.

Run inside the isolated cuRobo environment. No ROS graph or robot connection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
WS = HERE.parents[4]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'offline_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import curobo
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate the cuRobo environment')
    manifest = {'python': sys.version, 'platform': platform.platform(),
                'curobo': curobo.__version__, 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                'geometry_status': 'offline provisional CAD placement; not hardware qualified',
                'steps': {}}
    curobo_root = Path(curobo.__file__).resolve().parents[1]
    revision = subprocess.run(['git', '-C', str(curobo_root), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    manifest['curobo_commit'] = revision.stdout.strip()
    manifest['curobo_dirty'] = subprocess.run(['git', '-C', str(curobo_root), 'status', '--porcelain'], text=True, capture_output=True).stdout
    (output / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    environment = dict(os.environ, CUROBO_OFFLINE_OUTPUT=str(output))
    steps = [
        ('regenerate_urdf', ['bash', '-c', 'source /opt/ros/humble/setup.bash && exec /usr/bin/python3 "$1"', 'offline-xacro', str(HERE / 'regenerate_urdf.py')]),
    ] + [(name, [sys.executable, str(HERE / (name + '.py'))]) for name in (
        'generate_spheres', 'build_robot_config', 'validate_fk',
        'validate_fk_vs_urdf', 'validate_collision_model', 'validate_ik')]
    for name, command in steps:
        print(f'Running {name}', flush=True)
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT)
        manifest['steps'][name] = {'returncode': result.returncode}
        print(f'  {"PASS" if result.returncode == 0 else "FAIL"}: {output / (name + ".log")}', flush=True)
        if result.returncode and name in ('regenerate_urdf', 'generate_spheres', 'build_robot_config'):
            break  # Never validate stale artifacts after a generation failure.

    inputs = list(HERE.glob('*.py')) + list(HERE.glob('gen3_surgical*'))
    description = WS / 'src/ros2_kortex/kortex_description'
    inputs += list(description.rglob('*.xacro')) + list((description / 'config').glob('*.yaml'))
    resolved = HERE / 'gen3_surgical_resolved.urdf'
    if resolved.exists():
        inputs += [Path(m.get('filename')) for m in ET.parse(resolved).findall('.//mesh')]
    manifest['sha256'] = {str(p.relative_to(WS)): sha256(p) for p in sorted(set(inputs)) if p.is_file()}
    for path in list(HERE.glob('gen3_surgical*')) + [HERE / 'sphere_coverage.json', HERE / 'collision_validation.json']:
        if path.is_file():
            shutil.copy2(path, output / path.name)
    manifest['qualification_pass'] = len(manifest['steps']) == len(steps) and all(s['returncode'] == 0 for s in manifest['steps'].values())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Results: {output}\nModel qualification: {"PASS" if manifest["qualification_pass"] else "FAIL"}')
    return 0 if manifest['qualification_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
