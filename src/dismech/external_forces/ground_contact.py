import typing
import scipy.sparse as sp
import numpy as np

from ..soft_robot import SoftRobot


def compute_ground_contact(robot: SoftRobot, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    # Get z-component indices
    z_indices = np.arange(2, robot.end_node_dof_index, 3)
    dist = q[z_indices] - robot.env.ground_h - robot.env.ground_z

    # Create mask for nodes close enough to the ground
    active_mask = (dist <= robot.env.ground_delta)
    active_indices = z_indices[active_mask]
    active_dist = dist[active_mask]

    if active_indices.size == 0:
        return np.zeros_like(q), np.zeros((q.shape[0], q.shape[0]))

    # Compute force and stiffness only for active points
    K1 = 15 / robot.env.ground_delta
    v = np.exp(-K1 * active_dist)
    f_raw = (-2 * v * np.log(v + 1)) / (K1 * (v + 1)) * robot.env.ground_stiffness
    j_raw = (2 * v * np.log(v + 1) + 2 * v ** 2) / ((v + 1) ** 2) * robot.env.ground_stiffness

    # Allocate global force and Jacobian
    F = np.zeros_like(q)
    J = np.zeros((q.shape[0], q.shape[0]))

    # Accumulate only valid entries
    F[active_indices] = -f_raw
    J[active_indices, active_indices] = -j_raw
    return F, J
