import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from dismech.soft_robot import SoftRobot
from dismech.springs import HingeSprings
from dismech.visuoshell import load_nodes
from dismech.visuoshell.mesh import build_mesh


VISUOSHELL_ROOT = Path("/Users/radha/GitRepos/VisuoShell")


def _reference_build_mesh():
    if not VISUOSHELL_ROOT.exists():
        pytest.skip(f"VisuoShell checkout not found at {VISUOSHELL_ROOT}")

    sys.path.insert(0, str(VISUOSHELL_ROOT))
    try:
        return importlib.import_module("src.mesh_convexHull").build_mesh
    finally:
        sys.path.remove(str(VISUOSHELL_ROOT))


def _reference_hinge_nodes_and_angles(reference_hinges):
    hinge_nodes = np.array(
        [[n0, n1, *hinge["opposite"]] for (n0, n1), hinge in reference_hinges.items()],
        dtype=np.int32,
    )
    rest_angles = np.array(
        [hinge["theta_bar"] for hinge in reference_hinges.values()],
        dtype=np.float64,
    )
    return hinge_nodes, rest_angles


@pytest.mark.parametrize(
    "data_dir",
    [
        VISUOSHELL_ROOT / "test_data",
        VISUOSHELL_ROOT / "data",
        VISUOSHELL_ROOT / "dot_detection_pipline" / "example_dot_10mm",
    ],
)
def test_visuoshell_mesh_matches_reference_extraction(data_dir):
    reference_build_mesh = _reference_build_mesh()
    nodes_by_frame = load_nodes(data_dir)
    if not nodes_by_frame:
        pytest.skip(f"no loadable VisuoShell frames found in {data_dir}")

    frame_name = sorted(nodes_by_frame)[0]
    points = nodes_by_frame[frame_name]

    (
        reference_triangles,
        reference_edge_map,
        reference_hinges,
        reference_boundary,
        _reference_tri_to_edges,
        _reference_node_to_tris,
    ) = reference_build_mesh(points)
    reference_hinge_nodes, reference_rest_angles = _reference_hinge_nodes_and_angles(
        reference_hinges
    )

    mesh = build_mesh(points, method="convex_hull")

    assert np.array_equal(mesh.triangles, reference_triangles)
    assert mesh.edge_map == {
        edge: {"triangles": data["tris"], "edge_idx": data["edge_idx"]}
        for edge, data in reference_edge_map.items()
    }
    assert np.array_equal(mesh.hinge_nodes, reference_hinge_nodes)
    assert np.array_equal(mesh.rest_angles, reference_rest_angles)
    assert set(mesh.boundary_edges) == set(reference_boundary)


def test_visuoshell_example_dot_10mm_unused_nodes_match_reference():
    reference_build_mesh = _reference_build_mesh()
    data_dir = VISUOSHELL_ROOT / "dot_detection_pipline" / "example_dot_10mm"
    nodes_by_frame = load_nodes(data_dir)
    if not nodes_by_frame:
        pytest.skip(f"no loadable VisuoShell frames found in {data_dir}")

    points = nodes_by_frame[sorted(nodes_by_frame)[0]]
    reference_triangles, *_ = reference_build_mesh(points)
    mesh = build_mesh(points, method="convex_hull")

    all_nodes = set(range(points.shape[0]))
    reference_unused_nodes = all_nodes - set(np.unique(reference_triangles))
    unused_nodes = all_nodes - set(np.unique(mesh.triangles))

    assert unused_nodes == reference_unused_nodes
    assert unused_nodes == {
        19,
        39,
        52,
        74,
        105,
        111,
        119,
        161,
        169,
        191,
        195,
        196,
        197,
        203,
        209,
        212,
        214,
        216,
        219,
        222,
        223,
        224,
    }


def test_visuoshell_hinge_spring_dof_indices_match_reference_hinges():
    reference_build_mesh = _reference_build_mesh()
    data_dir = VISUOSHELL_ROOT / "dot_detection_pipline" / "example_dot_10mm"
    nodes_by_frame = load_nodes(data_dir)
    if not nodes_by_frame:
        pytest.skip(f"no loadable VisuoShell frames found in {data_dir}")

    points = nodes_by_frame[sorted(nodes_by_frame)[0]]
    _, _, reference_hinges, *_ = reference_build_mesh(points)
    reference_hinge_nodes, _ = _reference_hinge_nodes_and_angles(reference_hinges)

    springs = HingeSprings.from_arrays(
        build_mesh(points, method="convex_hull").hinge_nodes,
        np.ones(reference_hinge_nodes.shape[0]),
        SoftRobot.map_node_to_dof,
    )

    expected_dof_indices = np.hstack(
        [
            np.vstack([SoftRobot.map_node_to_dof(node) for node in reference_hinge_nodes[:, i]])
            for i in range(4)
        ]
    )

    assert np.array_equal(springs.nodes_ind, reference_hinge_nodes)
    assert np.array_equal(springs.ind, expected_dof_indices)
