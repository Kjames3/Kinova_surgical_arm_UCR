#!/usr/bin/env python3
"""Tabletop obstacle perception for the capsule barrier (barrier item 3).

Depth image in, vertical capsules out -- the obstacle primitive that
`control_barrier.compute_obstacle_hocbf_rows` consumes. Pure numpy, no ROS,
no scipy (REAL-1's scipy is built for NumPy 1 and its NumPy is 2), so the
pipeline can be tested offline against a rendered scene.
`obstacle_perception_node.py` is the thin ROS wrapper.

Pipeline, and why each step exists:

  1. depth_to_points   back-project the (color-aligned) depth image
  2. transform_points  camera optical frame -> base_link, via the extrinsic
  3. crop_workspace    keep a slab above the table. The advisor's prior is
                       "obstacles stay on the table", which is what lets us
                       skip an ESDF and fit capsules directly. min_height
                       removes the table itself plus depth noise.
  4. self_filter       drop points near the ARM. Mandatory: an arm that sees
                       itself as an obstacle locks up its own barrier. The
                       arm segments must be taken at the depth frame's
                       timestamp, not "now" -- the node does that via TF.
  5. exclude_regions   drop points inside keep-out cylinders. The container
                       being inserted into IS a tabletop object; without
                       this the barrier would forbid the insertion.
  6. grid_clusters     connected components on a 2-D occupancy grid, which
                       is the natural resolution for on-table objects
  7. fit_capsules      one vertical capsule per cluster, split into tiles
                       when a cluster is wider than max_radius allows

Occlusion is the hard limit of a single eye-to-hand camera: only the
camera-facing half of an object is seen. fit_capsules therefore extrudes each
cluster away from the camera by its own apparent width, capped at max_extrude,
before fitting (an object is assumed roughly as deep as it is wide), and then
inflates. Absent
depth must NOT read as free space -- the staleness/occlusion policy that
enforces this in the loop is barrier item 4; `valid_fraction` from
depth_to_points is the input it needs.
"""

import numpy as np


# ── 1-2. geometry ─────────────────────────────────────────────────────────────
def depth_to_points(depth_m, K, stride=2, min_range=0.15, max_range=3.0):
    """Back-project a depth image (metres) to camera-frame points.

    Returns (points (N, 3), valid_fraction). Zero / NaN / out-of-range pixels
    are dropped; valid_fraction is over the strided pixel grid.
    """
    d = np.asarray(depth_m, dtype=float)[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    ok = np.isfinite(d) & (d > min_range) & (d < max_range)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = d[ok]
    pts = np.column_stack(((u[ok] - cx) * z / fx, (v[ok] - cy) * z / fy, z))
    return pts, float(ok.mean()) if ok.size else 0.0


def depth_image_to_meters(raw, encoding):
    """16UC1 is millimetres (RealSense), 32FC1 is metres."""
    if encoding in ("16UC1", "mono16"):
        return raw.astype(float) * 1e-3
    if encoding == "32FC1":
        return raw.astype(float)
    raise ValueError(f"unsupported depth encoding {encoding!r}")


def quat_to_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_matrix(xyz, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat_xyzw)
    T[:3, 3] = xyz
    return T


def transform_points(pts, T):
    return pts @ T[:3, :3].T + T[:3, 3]


# ── 3-5. filtering ────────────────────────────────────────────────────────────
def crop_workspace(pts, x_range, y_range, table_z, min_height=0.015, max_height=0.40):
    keep = ((pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1])
            & (pts[:, 1] > y_range[0]) & (pts[:, 1] < y_range[1])
            & (pts[:, 2] > table_z + min_height) & (pts[:, 2] < table_z + max_height))
    return pts[keep]


def point_segment_distance(pts, a, b):
    """(N, S) distance from each point to each segment a[s] -> b[s]."""
    ab = b - a                                          # (S, 3)
    ap = pts[:, None, :] - a[None, :, :]                # (N, S, 3)
    denom = np.maximum(np.einsum("sc,sc->s", ab, ab), 1e-12)
    t = np.clip(np.einsum("nsc,sc->ns", ap, ab) / denom, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(pts[:, None, :] - closest, axis=2)


def self_filter(pts, seg_a, seg_b, radii, margin=0.03, chunk=20000):
    """Drop points within radius + margin of any arm segment."""
    if len(pts) == 0 or len(seg_a) == 0:
        return pts
    lim = np.asarray(radii, dtype=float) + margin
    keep = np.ones(len(pts), dtype=bool)
    for i in range(0, len(pts), chunk):
        d = point_segment_distance(pts[i:i + chunk], seg_a, seg_b)
        keep[i:i + chunk] = np.all(d > lim[None, :], axis=1)
    return pts[keep]


def exclude_regions(pts, regions):
    """Drop points inside vertical keep-out cylinders [(x, y, radius), ...]."""
    if len(pts) == 0 or not regions:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    for x, y, r in regions:
        keep &= (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2 > r * r
    return pts[keep]


# ── 6. clustering ─────────────────────────────────────────────────────────────
def grid_clusters(pts, cell=0.01, min_points=15, min_cells=2):
    """8-connected components of occupied xy cells -> list of point-index arrays."""
    if len(pts) == 0:
        return []
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    cells, inverse, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    lookup = {tuple(c): k for k, c in enumerate(cells)}
    label = -np.ones(len(cells), dtype=np.int64)
    n_labels = 0
    for start in range(len(cells)):
        if label[start] >= 0:
            continue
        label[start] = n_labels
        stack = [start]
        while stack:
            k = stack.pop()
            ci, cj = cells[k]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = lookup.get((ci + di, cj + dj))
                    if nb is not None and label[nb] < 0:
                        label[nb] = n_labels
                        stack.append(nb)
        n_labels += 1
    point_label = label[inverse]
    clusters = []
    for lab in range(n_labels):
        idx = np.flatnonzero(point_label == lab)
        if len(idx) >= min_points and np.sum(label == lab) >= min_cells:
            clusters.append(idx)
    return clusters


# ── 7. capsule fitting ────────────────────────────────────────────────────────
def _circle_cover(xy):
    """Center and radius of a circle covering xy: bbox centre, farthest point."""
    c = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    return c, float(np.sqrt(((xy - c) ** 2).sum(axis=1).max()))


def fit_capsules(pts, clusters, table_z, camera_xy, inflate=0.02, max_radius=0.10,
                 max_extrude=0.06):
    """Vertical capsules (centers (k, 3), radii (k,), heights (k,)) covering clusters.

    Each capsule's axis starts at table height and rises to the cluster top, so
    with its radius it covers the cluster as a cylinder and its caps add
    `radius` of margin above and below. Clusters wider than 2 * max_radius are
    tiled so a long object does not become one huge disc.
    """
    camera_xy = np.asarray(camera_xy, dtype=float)
    centers, radii, heights = [], [], []
    for idx in clusters:
        c_pts = pts[idx]
        xy = c_pts[:, :2]
        if max_extrude > 0:
            # Occlusion: we only see the camera-facing surface. Assume the
            # object is as deep (along the horizontal view ray) as it is wide,
            # capped at max_extrude -- uncapped, a 30 cm bar seen broadside
            # was pushed 30 cm back and walled off empty table.
            ray = xy.mean(axis=0) - camera_xy
            ray /= max(np.linalg.norm(ray), 1e-9)
            perp = np.array([-ray[1], ray[0]])
            width = float(np.ptp(xy @ perp)) if len(xy) > 1 else 0.0
            xy = np.vstack((xy, xy + min(width, max_extrude) * ray[None, :]))
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        tile = max_radius * np.sqrt(2.0) - 1e-9   # a tile's half-diagonal <= max_radius
        nx = max(1, int(np.ceil((hi[0] - lo[0]) / tile)))
        ny = max(1, int(np.ceil((hi[1] - lo[1]) / tile)))
        top = float(c_pts[:, 2].max())
        for ix in range(nx):
            for iy in range(ny):
                x0, y0 = lo[0] + ix * tile, lo[1] + iy * tile
                m = ((xy[:, 0] >= x0) & (xy[:, 0] <= x0 + tile)
                     & (xy[:, 1] >= y0) & (xy[:, 1] <= y0 + tile))
                if not m.any():
                    continue
                c, r = _circle_cover(xy[m])
                centers.append((c[0], c[1], table_z))
                radii.append(r + inflate)
                heights.append(max(top - table_z, 0.0))
    if not centers:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    return np.array(centers), np.array(radii), np.array(heights)


def perceive(depth_m, K, T_base_cam, arm_seg_a, arm_seg_b, arm_radii, cfg):
    """Full pipeline. Returns (centers, radii, heights, stats)."""
    pts_c, valid = depth_to_points(depth_m, K, cfg["stride"], cfg["min_range"], cfg["max_range"])
    pts = transform_points(pts_c, T_base_cam)
    n0 = len(pts)
    pts = crop_workspace(pts, cfg["x_range"], cfg["y_range"], cfg["table_z"],
                         cfg["min_height"], cfg["max_height"])
    n_crop = len(pts)
    pts = self_filter(pts, arm_seg_a, arm_seg_b, arm_radii, cfg["self_margin"])
    n_self = len(pts)
    pts = exclude_regions(pts, cfg["exclude"])
    clusters = grid_clusters(pts, cfg["cell"], cfg["min_points"])
    centers, radii, heights = fit_capsules(pts, clusters, cfg["table_z"],
                                           T_base_cam[:2, 3], cfg["inflate"],
                                           cfg["max_radius"], cfg["max_extrude"])
    stats = dict(valid_fraction=valid, points=n0, in_slab=n_crop,
                 self_removed=n_crop - n_self, clusters=len(clusters),
                 capsules=len(radii))
    return centers, radii, heights, stats


DEFAULT_CFG = dict(
    stride=2, min_range=0.15, max_range=3.0,
    x_range=(-0.2, 1.0), y_range=(-0.7, 0.7), table_z=-0.030,
    min_height=0.015, max_height=0.40,
    self_margin=0.03, exclude=[],
    cell=0.01, min_points=15,
    inflate=0.02, max_radius=0.10, max_extrude=0.06,
)
