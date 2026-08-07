#!/usr/bin/env python3
"""generate_spheres.py — M1: fit cuRobo collision spheres to the surgical arm.

Runs in the conda 'curobo' env (source ~/activate_curobo.sh). Parses
gen3_surgical.urdf, and for every link that has collision geometry, fits a set of
spheres (in the LINK frame) using cuRobo's sphere_fit. Writes the result as a
cuRobo-style collision_spheres YAML.

The thesis_ee assembly/tool (the long ~0.414 m insertion body ending at
assembly_tip) is the collision-critical part, so it gets a denser budget.

Usage:
    source ~/activate_curobo.sh
    python generate_spheres.py                 # writes gen3_surgical_spheres.yml
"""
import argparse
import os
import re

import trimesh
import yaml

from curobo.robot_parser import UrdfRobotParser
from curobo.sphere_fit import SphereFitType, estimate_sphere_count, fit_spheres_to_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "gen3_surgical.urdf")
# Workspace root = 5 dirs above config/curobo/ (…/ws/src/<repo>/<pkg>/config/curobo).
WS = os.path.abspath(os.path.join(HERE, *([os.pardir] * 5)))
INSTALL = os.path.join(WS, "install")


def resolve_urdf_meshes(src, dst):
    """Rewrite every mesh filename to an absolute filesystem path.

    cuRobo's join_path only skips its asset-root when the path is absolute, so
    'file://…' URIs (arm meshes) and package-relative paths (thesis_ee tool) both
    get mangled. Strip file://, and resolve 'pkg/…' -> install/pkg/share/pkg/….
    """
    txt = open(src).read()

    def repl(m):
        fn = m.group(1)
        if fn.startswith("file://"):
            fn = fn[len("file://"):]
        elif fn.startswith("package://"):
            fn = fn[len("package://"):]                 # -> pkg/rest…
        if not fn.startswith("/"):
            pkg = fn.split("/", 1)[0]                    # install/pkg/share/pkg/rest…
            fn = os.path.join(INSTALL, pkg, "share", fn)
        return f'filename="{fn}"'

    open(dst, "w").write(re.sub(r'filename="([^"]+)"', repl, txt))
    return dst

# Per-link sphere budget: clamp the density-estimated count into a sane range so
# self-collision stays cheap on 6 GB VRAM. Links matching a key get that range.
BUDGET = {
    "default": (4, 10),
    # long thin tool body needs more spheres to be represented without
    # over-inflating; tune after visual inspection.
    "assembly": (8, 16),
    "bracelet": (4, 8),
    "base_link": (4, 8),
}


def _budget_for(link):
    for key, rng in BUDGET.items():
        if key != "default" and key in link:
            return rng
    return BUDGET["default"]


# The thesis_ee assembly is a thin-shell STL that defeats volumetric sphere
# fitting (yields a few 5 mm specks). Since the whole tool is rigid w.r.t.
# bracelet_link and assembly_tip sits at (-0.027, 0, -0.414) in that frame, we
# hand-author a sphere chain there instead — far more reliable for a thin tool.
# Radius tapers from the assembly body near the wrist to the thin tip.
TOOL_CHAIN = dict(
    link="bracelet_link",
    start=(0.0, 0.0, -0.065),     # just past end_effector_link (z=-0.0615)
    end=(-0.027, 0.0, -0.414),    # assembly_tip
    n=12,
    r_start=0.018,                # assembly body
    r_end=0.008,                  # thin tip
)
SKIP_MESH_LINKS = {"thesis_ee"}   # covered by TOOL_CHAIN instead


def tool_chain_spheres():
    a, b = TOOL_CHAIN["start"], TOOL_CHAIN["end"]
    n, r0, r1 = TOOL_CHAIN["n"], TOOL_CHAIN["r_start"], TOOL_CHAIN["r_end"]
    out = []
    for i in range(n):
        t = i / (n - 1)
        c = [round(a[k] + t * (b[k] - a[k]), 5) for k in range(3)]
        out.append({"center": c, "radius": round(r0 + t * (r1 - r0), 5)})
    return out


def link_mesh(parser, link):
    geoms = parser.get_link_geometry(link, use_collision_mesh=True)
    meshes = []
    for g in geoms:
        m = g.get_trimesh_mesh(transform_with_pose=True)  # -> link frame
        if m is not None and len(m.vertices):
            m.fill_holes()
            trimesh.repair.fix_normals(m)
            meshes.append(m)
    if not meshes:
        return None
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "gen3_surgical_spheres.yml"))
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--surface-radius", type=float, default=0.005)
    ap.add_argument("--fit", default="morphit", choices=[t.value for t in SphereFitType])
    args = ap.parse_args()

    fit_type = SphereFitType(args.fit)
    resolved = resolve_urdf_meshes(URDF, os.path.join(HERE, "gen3_surgical_resolved.urdf"))
    parser = UrdfRobotParser(resolved, load_meshes=True, mesh_root="",
                             build_scene_graph=True)
    parser.build_link_parent()
    links = [ln for ln in parser.get_link_names_from_urdf()
             if parser.get_link_geometry(ln, use_collision_mesh=True)]
    print(f"links with collision geometry ({len(links)}): {links}")

    collision_spheres = {}
    total = 0
    for link in links:
        if link in SKIP_MESH_LINKS:
            print(f"  {link:28s} skipped mesh-fit (hand-authored tool chain)")
            continue
        mesh = link_mesh(parser, link)
        if mesh is None:
            print(f"  {link:28s} no usable mesh, skipped")
            continue
        lo, hi = _budget_for(link)
        n = max(lo, min(hi, estimate_sphere_count(mesh, sphere_density=args.density)))
        res = fit_spheres_to_mesh(
            mesh, num_spheres=n, surface_radius=args.surface_radius, fit_type=fit_type)
        centers = res.centers if hasattr(res, "centers") else res.center
        radii = res.radii if hasattr(res, "radii") else res.radius
        spheres = []
        for c, r in zip(centers, radii):
            if float(r) <= 0:
                continue
            spheres.append({"center": [round(float(x), 5) for x in c],
                            "radius": round(float(r), 5)})
        collision_spheres[link] = spheres
        total += len(spheres)
        print(f"  {link:28s} {len(spheres):2d} spheres  (r: "
              f"{min(s['radius'] for s in spheres):.3f}-{max(s['radius'] for s in spheres):.3f} m)")

    # Append the hand-authored tool chain to its carrier link.
    chain = tool_chain_spheres()
    collision_spheres.setdefault(TOOL_CHAIN["link"], []).extend(chain)
    total += len(chain)
    print(f"  {TOOL_CHAIN['link']+' (+tool chain)':28s} {len(chain):2d} spheres  (r: "
          f"{TOOL_CHAIN['r_end']:.3f}-{TOOL_CHAIN['r_start']:.3f} m)")

    with open(args.out, "w") as f:
        yaml.safe_dump({"collision_spheres": collision_spheres}, f,
                       default_flow_style=None, sort_keys=False)
    print(f"\nTotal spheres: {total} across {len(collision_spheres)} links")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
