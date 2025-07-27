import typing
import scipy.sparse as sp
import numpy as np

from ..soft_robot import SoftRobot

def compute_damping_force(robot: SoftRobot, q: np.ndarray, u: np.ndarray) -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, sp.csr_matrix]]:
    """
    Compute linear viscous damping forces and Jacobian.
    F = -η * velocity
    J = -η * 1 / dt * Identity

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
    Fd : (n_dof,) array
        Damping force.
    Jd : (n_dof, n_dof) array or sparse csr_matrix
        Damping Jacobian.
    """
    dt = robot.sim_params.dt
    eta = robot.env.eta   # Assumed: damping coefficient stored here
    n_nodes = np.shape(robot.nodes)[0]
    n_dof = robot.n_dof

    # Per-node Voronoi weights (each node has 3 DOFs)
    # vlen = robot.voronoi_ref_len_all  # shape (n_nodes,)
    # eta_v = eta * vlen            # shape (n_nodes,)
    n_nodes_shell = len(np.unique(robot.face_nodes_shell))
    n_nodes_rod = n_nodes - n_nodes_shell
    eta_v_shell = eta * np.ones((n_nodes_shell, 1))
    eta_shell_dof = np.repeat(eta_v_shell, 3)  # shape (3 * n_nodes,) per DOF
    eta_v_rod = eta*0.005 * np.ones((n_nodes_rod, 1))
    eta_rod_dof = np.repeat(eta_v_rod, 3)  # shape (3 * n_nodes,) per DOF

    # Node DOF indices, shape (n_nodes, 3)
    # node_dof_indices = np.array([robot.map_node_to_dof(i) for i in range(n_nodes)])  # (n_nodes, 3)
    shell_node_dof_indices = np.array([
        robot.map_node_to_dof(i) for i in range(n_nodes) if i in robot.face_nodes_shell
    ])
    rod_node_dof_indices = np.array([
        robot.map_node_to_dof(i) for i in range(n_nodes) if i not in robot.face_nodes_shell
    ]) # TO DO: define separate shell and rod nodes inside SoftRobot itself

    flat_shell_dof_indices = shell_node_dof_indices.reshape(-1).astype(int)
    flat_rod_dof_indices = rod_node_dof_indices.reshape(-1).astype(int)

    # Force
    Fd = np.zeros(n_dof)
    Fd[flat_shell_dof_indices] = -eta_shell_dof * u[flat_shell_dof_indices]
    Fd[flat_rod_dof_indices] = -eta_rod_dof * u[flat_rod_dof_indices]

    # Jacobian
    if robot.sim_params.sparse:
        J_shell_diag = -eta_shell_dof / dt
        J_rod_diag = -eta_rod_dof / dt
        J_diag = np.concatenate((J_shell_diag, J_rod_diag))
        Jd = sp.diags((J_diag,), [0], shape=(n_dof, n_dof), format="csr")
    else:
        Jd = np.zeros((n_dof, n_dof))
        J_shell_diag = -eta_shell_dof / dt
        J_rod_diag = -eta_rod_dof / dt
        Jd[flat_shell_dof_indices, flat_shell_dof_indices] = J_shell_diag
        Jd[flat_rod_dof_indices, flat_rod_dof_indices] = J_rod_diag

    return Fd, Jd