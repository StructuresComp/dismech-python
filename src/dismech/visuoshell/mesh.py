import dataclasses
import itertools
import typing

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist


@dataclasses.dataclass(frozen=True)
class VisuoShellMesh:
    """Triangle mesh connectivity and hinge stencils for VisuoShell markers."""

    triangles: np.ndarray
    hinge_nodes: np.ndarray
    rest_angles: np.ndarray
    edge_map: dict[tuple[int, int], dict[str, typing.Any]]
    boundary_edges: dict[tuple[int, int], dict[str, typing.Any]]


def build_mesh(points: np.ndarray, method: str = "convex_hull") -> VisuoShellMesh:
    """Build a reference shell mesh and hinge stencils from 3D marker points."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if method == "convex_hull":
        triangles = _convex_hull_triangles(points)
    elif method == "delaunay":
        triangles = _delaunay_sphere_triangles(points)
    else:
        raise ValueError("method must be 'convex_hull' or 'delaunay'")

    edge_map = _edge_map(triangles)
    hinge_nodes = []
    rest_angles = []
    boundary_edges = {}

    for (n0, n1), data in edge_map.items():
        if len(data["triangles"]) == 2:
            tri0, tri1 = data["triangles"]
            tri0_nodes = set(triangles[tri0])
            tri1_nodes = set(triangles[tri1])

            o0 = (tri0_nodes - {n0, n1}).pop()
            o1 = (tri1_nodes - {n0, n1}).pop()
            theta = get_hinge_angle(points[n0], points[n1], points[o0], points[o1])

            if theta < 0:
                o0, o1 = o1, o0
                theta = get_hinge_angle(points[n0], points[n1], points[o0], points[o1])

            hinge_nodes.append([n0, n1, o0, o1])
            rest_angles.append(theta)
        elif len(data["triangles"]) == 1:
            boundary_edges[(n0, n1)] = data

    return VisuoShellMesh(
        triangles=np.asarray(triangles, dtype=np.int64),
        hinge_nodes=np.asarray(hinge_nodes, dtype=np.int32),
        rest_angles=np.asarray(rest_angles, dtype=np.float64),
        edge_map=edge_map,
        boundary_edges=boundary_edges,
    )


def get_hinge_angle(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> float:
    """Signed dihedral angle for the hinge stencil (x0, x1, x2, x3)."""
    e0 = x1 - x0
    e1 = x2 - x0
    e2 = x3 - x0

    n0 = np.cross(e0, e1)
    n1 = np.cross(e2, e0)
    w = np.cross(n0, n1)
    angle = np.arctan2(np.linalg.norm(w), np.dot(n0, n1))
    return -angle if np.dot(e0, w) < 0 else angle


def _convex_hull_triangles(points: np.ndarray) -> np.ndarray:
    hull = ConvexHull(points)
    return _filter_large_triangles(hull.simplices, points)


def _filter_large_triangles(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    threshold = 5 * pdist(points).max() / np.sqrt(len(points))

    def max_edge_length(tri):
        p0, p1, p2 = points[tri]
        return max(
            np.linalg.norm(p0 - p1),
            np.linalg.norm(p1 - p2),
            np.linalg.norm(p0 - p2),
        )

    return np.asarray([tri for tri in triangles if max_edge_length(tri) < threshold])


def _delaunay_sphere_triangles(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit = points / norms
    theta = np.arccos(np.clip(unit[:, 2], -1, 1))
    phi = np.arctan2(unit[:, 1], unit[:, 0])
    return Delaunay(np.column_stack([theta, phi])).simplices


def _edge_map(triangles: np.ndarray) -> dict[tuple[int, int], dict[str, typing.Any]]:
    edge_map: dict[tuple[int, int], dict[str, typing.Any]] = {}

    for tri_idx, tri in enumerate(triangles):
        for n0, n1 in itertools.combinations(tri, 2):
            key = tuple(sorted((int(n0), int(n1))))
            if key not in edge_map:
                edge_map[key] = {"triangles": [tri_idx], "edge_idx": len(edge_map)}
            else:
                edge_map[key]["triangles"].append(tri_idx)

    return edge_map
