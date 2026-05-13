import numpy as np

from dismech.visuoshell import (
    VisuoShellForceEstimator,
    get_force_animation_plotly,
    load_nodes,
)
from dismech.visuoshell.visualization import _pyvista_faces


def test_visuoshell_reference_shape_has_zero_elastic_force():
    nodes = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.2],
        ]
    )

    estimator = VisuoShellForceEstimator.from_reference_points(nodes, kb=3.0)

    assert estimator.triangles.shape == (6, 3)
    assert estimator.mesh.hinge_nodes.shape == (9, 4)
    assert np.allclose(estimator.elastic_force(nodes), 0.0)
    assert estimator.energy_value(nodes) == 0.0


def test_visuoshell_midedge_reference_shape_has_zero_elastic_force():
    nodes = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.2],
        ]
    )

    estimator = VisuoShellForceEstimator.from_reference_points(
        nodes,
        kb=3.0,
        use_midedge=True,
    )

    assert estimator.use_midedge
    assert len(estimator.springs) == estimator.triangles.shape[0]
    assert np.allclose(estimator.elastic_force(nodes), 0.0)
    assert estimator.energy_value(nodes) == 0.0


def test_visuoshell_midedge_deformed_shape_has_finite_elastic_force():
    nodes = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.2],
        ]
    )
    deformed = nodes.copy()
    deformed[0, 2] += 0.2

    estimator = VisuoShellForceEstimator.from_reference_points(
        nodes,
        kb=3.0,
        use_midedge=True,
    )
    elastic_force = estimator.elastic_force(deformed)

    assert np.all(np.isfinite(elastic_force))
    assert np.linalg.norm(elastic_force) > 0
    assert np.allclose(estimator.external_balance_force(deformed), -elastic_force)


def test_visuoshell_midedge_matches_legacy_manual_subtraction():
    """End-to-end equivalence: the estimator (which now relies on TriangleEnergy's
    internal zero-referencing) must match the legacy hand-rolled subtraction that
    was previously done in force_estimator.py."""
    from dismech.elastics import TriangleEnergy
    from dismech.visuoshell.force_estimator import (
        _build_midedge_data,
        _build_triangle_springs,
        _face_edge_connectivity,
        _midedge_state_from_nodes,
        _stiffness_values,
    )
    from dismech.visuoshell.mesh import build_mesh

    nodes = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.2],
        ]
    )
    deformed = nodes.copy()
    deformed[0, 2] += 0.25
    deformed[3, 1] += 0.05

    kb = 5.0
    estimator = VisuoShellForceEstimator.from_reference_points(
        nodes, kb=kb, use_midedge=True
    )

    # Recreate the un-calibrated TriangleEnergy and manually subtract reference
    # force/energy — i.e. the pre-refactor behavior.
    mesh = build_mesh(nodes)
    shell_edges, face_edges, signs = _face_edge_connectivity(mesh.triangles)
    edge_ref_len = np.linalg.norm(
        nodes[shell_edges[:, 1]] - nodes[shell_edges[:, 0]], axis=1
    )
    midedge = _build_midedge_data(
        nodes, mesh.triangles, shell_edges, face_edges, signs, edge_ref_len
    )
    kb_values = _stiffness_values(kb, mesh.triangles.shape[0], "n_triangles")
    springs = _build_triangle_springs(midedge, kb_values, 0.5)
    ref_state = _midedge_state_from_nodes(nodes, midedge)
    raw_energy = TriangleEnergy(springs, ref_state)  # default: no zero-ref

    ref_force, _ = raw_energy.grad_hess_energy_linear_elastic(ref_state)
    ref_energy = float(np.sum(raw_energy.get_energy_linear_elastic(ref_state)))

    state = _midedge_state_from_nodes(deformed, midedge)
    raw_force, _ = raw_energy.grad_hess_energy_linear_elastic(state)
    raw_e = float(np.sum(raw_energy.get_energy_linear_elastic(state)))

    legacy_bend_force = raw_force - ref_force
    legacy_bend_energy = raw_e - ref_energy

    # The estimator's new TriangleEnergy returns the same bending force at the
    # node DOFs as the legacy manual subtraction.
    estimator_bend_force, _ = estimator.energy.grad_hess_energy_linear_elastic(state)
    assert np.allclose(estimator_bend_force, legacy_bend_force)

    estimator_bend_energy = float(
        np.sum(estimator.energy.get_energy_linear_elastic(state))
    )
    assert np.isclose(estimator_bend_energy, legacy_bend_energy)


def test_visuoshell_reference_properties_match_internal_energy():
    """The estimator's reference_force/reference_energy properties continue to
    expose the underlying TriangleEnergy calibration so existing diagnostics
    (e.g. examples/midedge_reference_force_check.py) keep working."""
    nodes = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.2],
        ]
    )

    midedge_est = VisuoShellForceEstimator.from_reference_points(
        nodes, kb=3.0, use_midedge=True
    )
    hinge_est = VisuoShellForceEstimator.from_reference_points(nodes, kb=3.0)

    # Midedge path: TriangleEnergy holds the cached reference values.
    assert isinstance(midedge_est.reference_force, np.ndarray)
    assert midedge_est.reference_force.size > 0
    assert np.allclose(
        midedge_est.reference_force, midedge_est.energy.reference_force
    )
    assert np.isclose(
        midedge_est.reference_energy,
        float(np.sum(midedge_est.energy.reference_energy)),
    )

    # Hinge path: no calibration needed, properties return empty/zero.
    assert hinge_est.reference_force.size == 0
    assert hinge_est.reference_energy == 0.0


def test_visuoshell_external_balance_force_is_negative_elastic_force():
    nodes = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.2],
        ]
    )
    deformed = nodes.copy()
    deformed[0, 2] += 0.2

    estimator = VisuoShellForceEstimator.from_reference_points(nodes, kb=3.0)
    elastic_force = estimator.elastic_force(deformed)

    assert np.linalg.norm(elastic_force) > 0
    assert np.allclose(estimator.external_balance_force(deformed), -elastic_force)


def test_visuoshell_force_animation_plotly_builds_frames():
    nodes = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.2],
        ]
    )
    deformed = nodes.copy()
    deformed[0, 2] += 0.2

    estimator = VisuoShellForceEstimator.from_reference_points(nodes, kb=3.0)
    nodes_by_frame = {"frame0.csv": nodes, "frame1.csv": deformed}
    forces_by_frame = {
        name: estimator.external_balance_force(frame_nodes)
        for name, frame_nodes in nodes_by_frame.items()
    }

    fig = get_force_animation_plotly(
        nodes_by_frame,
        forces_by_frame,
        estimator.triangles,
    )

    assert len(fig.data) == 4
    assert len(fig.frames) == 2


def test_load_nodes_splits_visuoshell_tracked_3d_csv(tmp_path):
    path = tmp_path / "tracked_3d.csv"
    path.write_text(
        "frame,point_id,x,y,z\n"
        "0,1,1.0,1.1,1.2\n"
        "0,0,0.0,0.1,0.2\n"
        "1,1,3.0,3.1,3.2\n"
        "1,0,2.0,2.1,2.2\n"
    )
    (tmp_path / "tracked_2d_left.csv").write_text(
        "frame,point_id,x,y\n"
        "0,0,0.0,0.1\n"
    )

    nodes_by_frame = load_nodes(tmp_path)

    assert list(nodes_by_frame) == [
        "tracked_3d_frame_000000.csv",
        "tracked_3d_frame_000001.csv",
    ]
    assert np.allclose(
        nodes_by_frame["tracked_3d_frame_000000.csv"],
        np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]),
    )


def test_pyvista_faces_match_visuoshell_polydata_format():
    triangles = np.array([[2, 0, 1], [1, 3, 2]], dtype=np.int64)

    faces = _pyvista_faces(triangles)

    assert np.array_equal(faces, np.array([3, 2, 0, 1, 3, 1, 3, 2]))
