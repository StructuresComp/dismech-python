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

    assert estimator.triangles.shape == (4, 3)
    assert estimator.mesh.hinge_nodes.shape == (4, 4)
    assert np.allclose(estimator.elastic_force(nodes), 0.0)
    assert estimator.energy_value(nodes) == 0.0


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
