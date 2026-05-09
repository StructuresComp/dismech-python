import numpy as np

from dismech.soft_robot import SoftRobot
from dismech.springs import HingeSprings


def test_hinge_springs_map_each_node_to_its_own_dofs():
    springs = HingeSprings.from_arrays(
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        np.array([10.0]),
        SoftRobot.map_node_to_dof,
    )

    assert np.array_equal(
        springs.ind,
        np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.int32),
    )
