"""Generate triangular meshes on a hemispherical surface.

Hyperparameters
---------------
n_nodes
    Target number of nodes (final mesh may have slightly fewer if the
    ``min_triangle_height`` constraint forces some nodes to be dropped).
radius
    Hemisphere radius.
min_triangle_height
    Minimum allowed altitude (shortest perpendicular height) of any triangle.
    Triangles thinner than this are eliminated by removing the offending
    interior node and re-triangulating.
nonhomogeneity
    Floating-point number in ``[0, 1]``.
    ``0`` produces an as-uniform-as-possible mesh (Poisson-disk seeding
    refined with Lloyd's relaxation).
    ``1`` produces a heavily randomized mesh (jittered seeding and no
    relaxation).

Output format
-------------
The mesh is written as a text file matching the existing
``tests/resources/shell_contact/input_hemisphere_*.txt`` format:

    *Nodes
    x,y,z
    ...
    *Triangles
    i,j,k        (1-based indices)
    ...

The boundary nodes lie exactly on the equator (``z = 0``) and are listed
first, followed by interior dome nodes.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.spatial import Delaunay, Voronoi


# ---------------------------------------------------------------------------
# Lambert equal-area projection between hemisphere and 2D disk
# ---------------------------------------------------------------------------

def lambert_project(points: np.ndarray, radius: float) -> np.ndarray:
    """Map points on the upper hemisphere (z >= 0) of given radius to the
    Lambert equal-area disk of radius ``R * sqrt(2)``.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    factor = np.sqrt(2.0 * radius / np.maximum(radius + z, 1e-15))
    return np.column_stack([x * factor, y * factor])


def lambert_unproject(uv: np.ndarray, radius: float) -> np.ndarray:
    """Inverse of :func:`lambert_project`."""
    u, v = uv[:, 0], uv[:, 1]
    rho2 = u * u + v * v
    z = radius - rho2 / (2.0 * radius)
    z = np.clip(z, 0.0, radius)
    factor = np.sqrt(2.0 * radius / np.maximum(radius + z, 1e-15))
    x = u / factor
    y = v / factor
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Point distribution in the 2D disk
# ---------------------------------------------------------------------------

def poisson_disk_2d(disk_radius: float,
                    min_dist: float,
                    boundary_points: np.ndarray,
                    k: int = 30,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Bridson's Poisson-disk sampling in a disk, with pre-placed boundary
    points that are kept as the first entries of the returned array.
    """
    if rng is None:
        rng = np.random.default_rng()

    cell_size = min_dist / np.sqrt(2)
    offset = disk_radius + cell_size

    def grid_coords(p):
        return int((p[0] + offset) / cell_size), int((p[1] + offset) / cell_size)

    grid: dict[tuple[int, int], list[int]] = {}
    points: list[np.ndarray] = [np.asarray(bp, dtype=float) for bp in boundary_points]
    active: list[int] = []

    for i, bp in enumerate(points):
        grid.setdefault(grid_coords(bp), []).append(i)
        active.append(i)

    # Seed with the disk centre if it is not blocked by the boundary.
    centre = np.zeros(2)
    if not any(np.linalg.norm(bp - centre) < min_dist for bp in points):
        grid.setdefault(grid_coords(centre), []).append(len(points))
        points.append(centre)
        active.append(len(points) - 1)

    interior_limit = disk_radius - 0.5 * min_dist

    def in_disk(p):
        return p[0] * p[0] + p[1] * p[1] < interior_limit * interior_limit

    def is_far_enough(p):
        gx, gy = grid_coords(p)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                bucket = grid.get((gx + dx, gy + dy))
                if not bucket:
                    continue
                for idx in bucket:
                    q = points[idx]
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_dist * min_dist:
                        return False
        return True

    while active:
        pick = int(rng.integers(len(active)))
        p = points[active[pick]]
        found = False
        for _ in range(k):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            r = rng.uniform(min_dist, 2.0 * min_dist)
            candidate = np.array([p[0] + r * np.cos(theta),
                                  p[1] + r * np.sin(theta)])
            if in_disk(candidate) and is_far_enough(candidate):
                grid.setdefault(grid_coords(candidate), []).append(len(points))
                points.append(candidate)
                active.append(len(points) - 1)
                found = True
                break
        if not found:
            active.pop(pick)

    return np.asarray(points)


# ---------------------------------------------------------------------------
# Lloyd's relaxation (vertex-centroid variant) for non-boundary points
# ---------------------------------------------------------------------------

def lloyd_relaxation(points_2d: np.ndarray,
                     n_boundary: int,
                     disk_radius: float,
                     n_iter: int) -> np.ndarray:
    """Move each non-boundary point toward the centroid of its Voronoi cell.

    Uses the mean of the cell's vertices as the centroid proxy. Boundary
    points (the first ``n_boundary`` entries) stay fixed on the equator.
    Centroids that drift outside the disk are projected back inside.
    """
    if n_iter <= 0 or len(points_2d) <= n_boundary:
        return points_2d

    clip_radius = disk_radius * 0.985
    pts = points_2d.copy()
    for _ in range(n_iter):
        try:
            vor = Voronoi(pts)
        except Exception:
            return pts
        new_pts = pts.copy()
        for i in range(n_boundary, len(pts)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            verts = vor.vertices[region]
            centroid = verts.mean(axis=0)
            r = float(np.linalg.norm(centroid))
            if r > clip_radius:
                centroid = centroid * (clip_radius / r)
            new_pts[i] = centroid
        pts = new_pts
    return pts


# ---------------------------------------------------------------------------
# Minimum-triangle-height enforcement
# ---------------------------------------------------------------------------

def _triangle_min_altitude(p: np.ndarray) -> tuple[float, int]:
    """Return ``(min_altitude, vertex_opposite_longest_edge)`` for a triangle.

    ``p`` is a ``(3, d)`` array of vertex coordinates. The minimum altitude
    of a triangle is ``2 * area / longest_edge``, and the vertex opposite
    the longest edge is the one with the smallest perpendicular distance to
    its opposite side, so it is the natural candidate for removal when a
    triangle is too thin.
    """
    a = float(np.linalg.norm(p[1] - p[2]))
    b = float(np.linalg.norm(p[0] - p[2]))
    c = float(np.linalg.norm(p[0] - p[1]))
    edges = np.array([a, b, c])
    s = 0.5 * edges.sum()
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0.0:
        return 0.0, int(np.argmax(edges))
    area = np.sqrt(area_sq)
    longest = edges.max()
    altitude = 2.0 * area / longest if longest > 0.0 else 0.0
    return altitude, int(np.argmax(edges))


def enforce_min_height(points_2d: np.ndarray,
                       n_boundary: int,
                       radius: float,
                       min_height: float,
                       max_iter: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively drop non-boundary nodes that produce too-thin triangles
    (measured in 3D on the hemisphere) and re-triangulate.
    """
    pts = points_2d
    for _ in range(max_iter):
        nodes = lambert_unproject(pts, radius)
        tri = Delaunay(pts)
        bad: set[int] = set()
        for simplex in tri.simplices:
            altitude, worst = _triangle_min_altitude(nodes[simplex])
            if altitude < min_height:
                idx = int(simplex[worst])
                if idx >= n_boundary:
                    bad.add(idx)
        if not bad:
            return pts, tri.simplices
        keep = np.array([i for i in range(len(pts)) if i not in bad], dtype=int)
        pts = pts[keep]
    tri = Delaunay(pts)
    return pts, tri.simplices


# ---------------------------------------------------------------------------
# Top-level mesh generation
# ---------------------------------------------------------------------------

def _seed_points(spacing: float,
                 disk_radius: float,
                 radius: float,
                 seed: int | None) -> tuple[np.ndarray, int]:
    """Place equispaced boundary points and Poisson-disk-pack interior.

    Uses a fresh RNG seeded from ``seed`` so a given ``spacing`` always
    produces the same point set, which keeps the calibration loop below
    deterministic.
    """
    sub_rng = np.random.default_rng(seed)
    n_boundary = max(6, int(round(2.0 * np.pi * radius / spacing)))
    angles = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    boundary_2d = np.column_stack([
        disk_radius * np.cos(angles),
        disk_radius * np.sin(angles),
    ])
    points_2d = poisson_disk_2d(disk_radius, spacing, boundary_2d, rng=sub_rng)
    return points_2d, n_boundary


def generate_hemisphere_mesh(n_nodes: int,
                             radius: float,
                             min_triangle_height: float,
                             nonhomogeneity: float,
                             seed: int | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Generate a triangular mesh on a hemisphere.

    Returns
    -------
    nodes : (N, 3) float array
        Node coordinates. Equator nodes (``z = 0``) come first.
    triangles : (M, 3) int array
        Triangle vertex indices (0-based).
    """
    if not 0.0 <= nonhomogeneity <= 1.0:
        raise ValueError("nonhomogeneity must be in [0, 1]")
    if n_nodes < 6:
        raise ValueError("n_nodes must be at least 6")
    if radius <= 0:
        raise ValueError("radius must be positive")

    rng = np.random.default_rng(seed)

    # Hemisphere area = 2 * pi * R^2; the Lambert disk has the same area.
    disk_radius = radius * np.sqrt(2.0)
    hemi_area = 2.0 * np.pi * radius * radius

    # Spacing floor implied by min_triangle_height for an equilateral
    # triangle (altitude = side * sqrt(3) / 2). Triangulations may include
    # non-equilateral triangles below this floor, which are filtered later.
    spacing_floor = (2.0 / np.sqrt(3.0)) * min_triangle_height * 1.05

    # Initial spacing guess: equilateral-tile assumption.
    spacing = max(np.sqrt(2.0 * hemi_area / (n_nodes * np.sqrt(3.0))),
                  spacing_floor)

    # Calibration loop: Poisson-disk packing fills ~70% of the equilateral
    # density, so a single closed-form choice undershoots. Iterate (n ~
    # 1/spacing^2) to track the requested count without depending on
    # ``nonhomogeneity``.
    points_2d, n_boundary = _seed_points(spacing, disk_radius, radius, seed)
    for _ in range(8):
        actual = len(points_2d)
        if abs(actual - n_nodes) <= max(2, int(0.05 * n_nodes)):
            break
        new_spacing = max(spacing * np.sqrt(actual / n_nodes), spacing_floor)
        if abs(new_spacing - spacing) / spacing < 0.01:
            break
        spacing = new_spacing
        points_2d, n_boundary = _seed_points(spacing, disk_radius, radius, seed)

    # Lloyd's relaxation: many iterations when uniform, none when fully random.
    n_lloyd = int(round(8 * (1.0 - nonhomogeneity)))
    points_2d = lloyd_relaxation(points_2d, n_boundary, disk_radius, n_lloyd)

    # Jitter interior points; magnitude grows with nonhomogeneity.
    if nonhomogeneity > 0.0 and len(points_2d) > n_boundary:
        n_interior = len(points_2d) - n_boundary
        jitter = rng.normal(scale=0.3 * nonhomogeneity * spacing,
                            size=(n_interior, 2))
        points_2d[n_boundary:] += jitter
        interior = points_2d[n_boundary:]
        rho = np.linalg.norm(interior, axis=1)
        clip_r = disk_radius * 0.98
        mask = rho > clip_r
        if mask.any():
            interior[mask] *= (clip_r / rho[mask])[:, None]
        points_2d[n_boundary:] = interior

    # Jitter boundary angles (keep them on the equator) for nonzero
    # nonhomogeneity; magnitude is a fraction of the equator arc spacing.
    if nonhomogeneity > 0.0:
        max_perturb = (np.pi / n_boundary) * 0.5 * nonhomogeneity
        b_angles = np.arctan2(points_2d[:n_boundary, 1],
                              points_2d[:n_boundary, 0])
        b_angles = b_angles + rng.uniform(-max_perturb, max_perturb, n_boundary)
        points_2d[:n_boundary, 0] = disk_radius * np.cos(b_angles)
        points_2d[:n_boundary, 1] = disk_radius * np.sin(b_angles)

    if min_triangle_height > 0.0:
        points_2d, triangles = enforce_min_height(
            points_2d, n_boundary, radius, min_triangle_height,
        )
    else:
        triangles = Delaunay(points_2d).simplices

    nodes = lambert_unproject(points_2d, radius)
    return nodes, np.asarray(triangles, dtype=int)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_mesh(nodes: np.ndarray, triangles: np.ndarray, output_path: str) -> None:
    """Write the mesh in the ``*Nodes`` / ``*Triangles`` text format with
    1-based triangle indices.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("*Nodes\n")
        for n in nodes:
            f.write(f"{n[0]:.15g},{n[1]:.15g},{n[2]:.15g}\n")
        f.write("*Triangles\n")
        for t in triangles:
            f.write(f"{int(t[0]) + 1},{int(t[1]) + 1},{int(t[2]) + 1}\n")


def plot_mesh(nodes: np.ndarray, triangles: np.ndarray, output_path: str | None = None) -> None:
    """Render a 3D view of the mesh for sanity-checking."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    polys = nodes[triangles]
    collection = Poly3DCollection(polys, alpha=0.6, edgecolor="k", linewidth=0.4)
    collection.set_facecolor((0.6, 0.75, 0.95))
    ax.add_collection3d(collection)
    rng = float(np.max(np.linalg.norm(nodes, axis=1)))
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.set_zlim(0, rng)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{len(nodes)} nodes, {len(triangles)} triangles")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved preview to: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a triangular mesh on a hemispherical surface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-nodes", type=int, required=True,
                        help="Target number of nodes.")
    parser.add_argument("--radius", type=float, default=0.1,
                        help="Hemisphere radius.")
    parser.add_argument("--min-triangle-height", type=float, default=0.001,
                        help="Minimum altitude of any triangle (3D, on the hemisphere).")
    parser.add_argument("--nonhomogeneity", type=float, default=0.0,
                        help="0 = most uniform, 1 = highly randomized.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output text file (auto-named if omitted).")
    parser.add_argument("--plot", action="store_true",
                        help="Show a 3D matplotlib preview of the mesh.")
    parser.add_argument("--save-plot", type=str, default="preview.png",
                        help="Save the preview PNG to this path (implies --plot logic).")
    args = parser.parse_args()

    nodes, triangles = generate_hemisphere_mesh(
        n_nodes=args.n_nodes,
        radius=args.radius,
        min_triangle_height=args.min_triangle_height,
        nonhomogeneity=args.nonhomogeneity,
        seed=args.seed,
    )

    if args.output is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        name = (
            f"hemisphere_n{len(nodes)}_r{args.radius:g}"
            f"_mh{args.min_triangle_height:g}_nh{args.nonhomogeneity:g}.txt"
        )
        args.output = os.path.join(out_dir, name)

    write_mesh(nodes, triangles, args.output)
    print(f"Generated mesh with {len(nodes)} nodes and {len(triangles)} triangles.")
    print(f"Saved to: {args.output}")

    if args.plot or args.save_plot:
        plot_mesh(nodes, triangles, args.save_plot)


if __name__ == "__main__":
    main()
