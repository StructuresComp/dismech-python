import numpy as np
import scipy.sparse as sp
import typing

from ..soft_robot import SoftRobot

def compute_surface_viscous_drag(robot: SoftRobot, q: np.ndarray, u: np.ndarray) -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, sp.csr_matrix]]:
    """
    Computes surface viscous drag force (proportional to velocity) for shell elements.

    Parameters
    ----------
    robot : SoftRobot
        Must contain:
            - env.rho: fluid density
            - env.cd: linear drag coefficient
            - face_area: (n_faces,) triangle areas
            - face_nodes_shell: (n_faces, 3) triangle node indices
            - map_node_to_dof: function mapping node index -> [ix, iy, iz]
            - sim_params.dt
            - sim_params.sparse: whether to return sparse Jacobian
            - n_dof: number of DOFs

    q : (n_dof,) array
        Current positions.
    q0 : (n_dof,) array
        Previous positions.

    Returns
    -------
    Fd : (n_dof,) array
        Drag force.
    Jd : (n_dof, n_dof) array or sparse matrix
        Jacobian of drag force.
    """
    rho = robot.env.rho
    cd = robot.env.cd
    dt = robot.sim_params.dt
    sparse = robot.sim_params.sparse
    n_dof = robot.n_dof

    Fd = np.zeros(n_dof)
    if not sparse:
        Jd = np.zeros((n_dof, n_dof))

    face_nodes = robot.face_nodes_shell
    face_area = robot.face_area
    n_faces = face_nodes.shape[0]

    # Get DOFs for each vertex of the triangle (3 per face)
    dofs = [np.array([robot.map_node_to_dof(n) for n in face_nodes[:, i]])
            for i in range(3)]  # each is (n_faces, 3)

    # Calculate velocities at each vertex
    us = [u[dofs[i]] for i in range(3)]

    # Per-face linear drag coefficient: shared equally among 3 nodes
    coeff = (rho * cd * face_area / 3.0)[:, None]  # shape (n_faces, 1)

    # Apply forces: F = -coeff * u_i
    for i in range(3):
        force = -coeff * us[i]
        np.add.at(Fd, dofs[i], force)

    # Build Jacobian
    if sparse:
        rows_list = []
        cols_list = []
        data_list = []

        coeff_flat = (rho * cd * face_area / (3.0 * dt))
        for i in range(3):
            # Each block is -coeff * I_3x3
            row_ids = dofs[i].reshape(-1, 1).repeat(3, axis=1)  # (n_faces, 3)
            col_ids = row_ids  # self contribution only

            block_vals = -coeff_flat[:, None] * np.ones((n_faces, 3))  # (n_faces, 3)

            rows_list.append(row_ids.flatten())
            cols_list.append(col_ids.flatten())
            data_list.append(block_vals.flatten())

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)

        Jd = sp.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    else:
        coeff_flat = (rho * cd * face_area / (3.0 * dt))  # (n_faces,)
        for i in range(3):
            idx = dofs[i]
            diag_vals = -coeff_flat[:, None] * np.ones((n_faces, 3))
            np.add.at(Jd, (idx, idx), diag_vals)

    return Fd, Jd
