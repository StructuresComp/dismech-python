import typing
import scipy.sparse as sp
import numpy as np

from ..soft_robot import SoftRobot

def compute_viscous_force(robot: SoftRobot, q: np.ndarray, u: np.ndarray) -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, sp.csr_matrix]]:
    """
    Compute linear viscous damping forces and Jacobian.
    F = -η * volume * velocity
    J = -η * volume / dt * Identity

    Parameters
    ----------
    robot : object
        Must contain:
            - sim_params.dt
            - sim_params.sparse (bool)
            - n_dof
            - n_nodes
            - voronoi_ref_len: (n_nodes,)
            - map_node_to_dof(i): returns [ix, iy, iz]
    q : (n_dof,) array
        Current DOF positions.
    u : (n_dof,) array
        Current DOF velocity.

    Returns
    -------
    Fv : (n_dof,) array
        Damping force.
    Jv : (n_dof, n_dof) array or sparse csr_matrix
        Damping Jacobian.
    """
    dt = robot.sim_params.dt
    eta = robot.env.eta   # Assumed: damping coefficient stored here
    n_nodes = np.shape(robot.nodes)[0]
    n_dof = robot.n_dof

    # Per-node Voronoi weights (each node has 3 DOFs)
    # vlen = robot.voronoi_ref_len  # shape (n_nodes,)
    # eta_v = eta * vlen            # shape (n_nodes,)
    eta_v = eta * np.ones((n_nodes, 1))
    eta_dof = np.repeat(eta_v, 3)  # shape (3 * n_nodes,) per DOF

    # Node DOF indices, shape (n_nodes, 3)
    node_dof_indices = np.array([robot.map_node_to_dof(i) for i in range(n_nodes)])  # (n_nodes, 3)
    flat_dof_indices = node_dof_indices.reshape(-1)

    # Force
    Fv = np.zeros(n_dof)
    Fv[flat_dof_indices] = -eta_dof * u[flat_dof_indices]

    # Jacobian
    if robot.sim_params.sparse:
        J_diag = -eta_dof / dt
        Jv = sp.diags((J_diag,), [0], shape=(n_dof, n_dof), format="csr")
    else:
        Jv = np.zeros((n_dof, n_dof))
        J_diag = -eta_dof / dt
        Jv[flat_dof_indices, flat_dof_indices] = J_diag

    print("viscous force", Fv)
    print("viscous jacob", Jv)

    return Fv, Jv