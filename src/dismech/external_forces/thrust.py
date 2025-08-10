import typing
import scipy.sparse as sp
import numpy as np

from ..soft_robot import SoftRobot

def compute_approx_volume(robot, q):
    """
    Compute an approximate enclosed volume of a closed hemispherical shell.
    Assumes faces are oriented correctly (outward normals).
    """
    face_nodes = robot.face_nodes_shell
    positions = q[0:robot.end_node_dof_index].reshape(-1, 3)
    total_volume = 0.0

    for f in face_nodes:
        v0, v1, v2 = positions[f]
        volume = np.dot(v0, np.cross(v1, v2)) / 6.0  # Signed volume of tetrahedron with origin
        total_volume += volume

    return abs(total_volume)

def compute_approx_volume_and_gradient(robot, q: np.ndarray) -> typing.Tuple[float, np.ndarray]:
    """
    Vectorized computation of:
    - Signed volume enclosed by the shell
    - Gradient of volume with respect to q

    Parameters
    ----------
    robot : object
        Must contain:
            - face_nodes_shell: (n_faces, 3) int, indices of triangle vertices
            - n_nodes: total number of nodes

    q : (n_dof,) array (3 * n_nodes,)
        Flattened position vector of all nodes.

    Returns
    -------
    volume : float
        Total signed volume enclosed by mesh.
    dV_dq : (n_dof,) array
        Gradient of volume with respect to q.
    """
    face_nodes = robot.face_nodes_shell  # (n_faces, 3)
    positions = q[0:robot.end_node_dof_index].reshape(-1, 3)         # (n_nodes, 3)

    v0 = positions[face_nodes[:, 0]]  # (n_faces, 3)
    v1 = positions[face_nodes[:, 1]]
    v2 = positions[face_nodes[:, 2]]

    # Compute per-face signed volume: V = 1/6 * dot(v0, cross(v1, v2))
    cross_v1_v2 = np.cross(v1, v2)      # (n_faces, 3)
    volume_contrib = np.einsum('ij,ij->i', v0, cross_v1_v2)  # (n_faces,)
    total_volume = np.sum(volume_contrib) / 6.0

    # Compute per-face gradients
    grad_i = np.cross(v1, v2) / 6.0  # (n_faces, 3)
    grad_j = np.cross(v2, v0) / 6.0
    grad_k = np.cross(v0, v1) / 6.0

    # Accumulate gradients at each node
    n_nodes = np.shape(robot.nodes)[0]
    grad = np.zeros((n_nodes, 3))
    np.add.at(grad, face_nodes[:, 0], grad_i)
    np.add.at(grad, face_nodes[:, 1], grad_j)
    np.add.at(grad, face_nodes[:, 2], grad_k)

    dV_dq = grad.reshape(-1)  # (n_dof,)

    return abs(total_volume), dV_dq


def compute_thrust_force_and_jacobian(robot, q: np.ndarray, u: np.ndarray) -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, sp.csr_matrix]]:

    dt = robot.sim_params.dt
    q0 = robot.state.q
    k = robot.env.thrust_coeff
    n_dof = robot.n_dof
    n_nodes = robot.n_nodes
    sparse = robot.sim_params.sparse

    # --- Compute volume and rate of change ---
    V_curr, dV_dq = compute_approx_volume_and_gradient(robot, q)
    V_prev = compute_approx_volume(robot, q0)
    dV_dt = (V_curr - V_prev) / dt

    F_thrust = np.zeros(n_dof)

    if dV_dt >= 0:
        J_thrust = sp.csr_matrix((n_dof, n_dof)) if sparse else np.zeros((n_dof, n_dof))
        return F_thrust, J_thrust

    # --- Apply thrust force ---
    thrust_dir = np.array([0, 0, 1])  # along Z
    total_force = -k * dV_dt * thrust_dir

    # print(f"dV/dt: {dV_dt:.5f}, Total thrust: {total_force}")

    #######################################
    # --- choose which nodes receive thrust ---
    # For shell-only thrust:
    sel_nodes = np.unique(robot.face_nodes_shell.ravel())

    # Map selected nodes to DOF indices in [x,y,z,x,y,z,...] order
    sel_dofs_list = [robot.map_node_to_dof(i) for i in sel_nodes]
    sel_dofs = np.asarray(np.concatenate(sel_dofs_list), dtype=int)  # shape (3*n_sel,)

    n_sel = len(sel_nodes)

    # Force distribution (vectorized)
    force_per_node = total_force / n_sel            # shape (3,)
    F_thrust[sel_dofs] = np.tile(force_per_node, n_sel)

    # --- Jacobian on the selected DOFs ---
    # dz pattern should align with DOF ordering: [0,0,1/n_sel, 0,0,1/n_sel, ...]
    dz_sub = np.tile(thrust_dir / n_sel, n_sel)     # shape (3*n_sel,)

    # Restrict the volume gradient to selected DOFs
    dV_dq_sub = dV_dq[sel_dofs]                     # shape (3*n_sel,)

    scale = -k * (1.0 / dt)

    if sparse:
        # Build the outer product on the subset and place it into the global (n_dof x n_dof)
        # Outer: dV_dq_sub[:,None] * dz_sub[None,:]  -> (3*n_sel, 3*n_sel)
        rows_block = np.repeat(sel_dofs, sel_dofs.size)
        cols_block = np.tile(sel_dofs,  sel_dofs.size)
        data_block = scale * (dV_dq_sub[:, None] * dz_sub[None, :]).ravel(order='C')
        J_thrust = sp.csr_matrix((data_block, (rows_block, cols_block)), shape=(n_dof, n_dof))
    else:
        J_thrust = np.zeros((n_dof, n_dof))
        J_thrust[np.ix_(sel_dofs, sel_dofs)] = scale * np.outer(dV_dq_sub, dz_sub)

    return F_thrust, J_thrust



# import typing
# import scipy.sparse as sp
# import numpy as np

# from ..soft_robot import SoftRobot

# def compute_approx_volume(robot: SoftRobot, q: np.ndarray):
#     """
#     Compute an approximate enclosed volume of a closed hemispherical shell.
#     Assumes faces are oriented correctly (outward normals).
#     """
#     face_nodes = robot.face_nodes_shell
#     total_volume = 0.0

#     for f in face_nodes:
#         n1, n2, n3 = robot.map_node_to_dof(f)
#         v0, v1, v2 = q[n1], q[n2], q[n3]  # Get positions of triangle vertices
#         # print(f"v0: {v0}, v1: {v1}, v2: {v2}")
#         volume = np.dot(v0, np.cross(v1, v2)) / 6.0  # Signed volume of tetrahedron with origin
#         total_volume += volume

#     return abs(total_volume)

# def compute_approx_volume_and_gradient(robot: SoftRobot, q: np.ndarray) -> typing.Tuple[float, np.ndarray]:
#     """
#     Vectorized computation of:
#     - Signed volume enclosed by the shell
#     - Gradient of volume with respect to q

#     Parameters
#     ----------
#     robot : object
#         Must contain:
#             - face_nodes_shell: (n_faces, 3) int, indices of triangle vertices
#             - n_nodes: total number of nodes

#     q : (n_dof,) array (3 * n_nodes,)
#         Flattened position vector of all nodes.

#     Returns
#     -------
#     volume : float
#         Total signed volume enclosed by mesh.
#     dV_dq : (n_dof,) array
#         Gradient of volume with respect to q.
#     """
#     face_nodes = robot.face_nodes_shell  # (n_faces, 3)
#     n_nodes = robot.n_nodes
#     # positions = q.reshape(-1, 3)         # (n_nodes, 3)
#     positions = q[:3 * n_nodes].reshape(-1, 3)


#     v0 = positions[face_nodes[:, 0]]  # (n_faces, 3)
#     v1 = positions[face_nodes[:, 1]]
#     v2 = positions[face_nodes[:, 2]]

#     # Compute per-face signed volume: V = 1/6 * dot(v0, cross(v1, v2))
#     cross_v1_v2 = np.cross(v1, v2)      # (n_faces, 3)
#     volume_contrib = np.einsum('ij,ij->i', v0, cross_v1_v2)  # (n_faces,)
#     total_volume = np.sum(volume_contrib) / 6.0

#     # Compute per-face gradients
#     grad_i = np.cross(v1, v2) / 6.0  # (n_faces, 3)
#     grad_j = np.cross(v2, v0) / 6.0
#     grad_k = np.cross(v0, v1) / 6.0

#     # Accumulate gradients at each node
#     grad = np.zeros((n_nodes, 3))
#     np.add.at(grad, face_nodes[:, 0], grad_i)
#     np.add.at(grad, face_nodes[:, 1], grad_j)
#     np.add.at(grad, face_nodes[:, 2], grad_k)

#     dV_dq = grad.reshape(-1)  # (n_dof,)

#     return abs(total_volume), dV_dq


# def compute_thrust_force_and_jacobian(robot: SoftRobot, q: np.ndarray, u: np.ndarray) -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, sp.csr_matrix]]:

#     dt = robot.sim_params.dt
#     q0 = robot.state.q
#     k = robot.env.thrust_coeff
#     n_dof = robot.n_dof
#     n_nodes = np.shape(robot.nodes)[0]
#     n_nodes_shell = len(np.unique(robot.face_nodes_shell))
#     sparse = robot.sim_params.sparse

#     # --- Compute volume and rate of change ---
#     V_curr, dV_dq = compute_approx_volume_and_gradient(robot, q)
#     V_prev = compute_approx_volume(robot, q0)
#     dV_dt = (V_curr - V_prev) / dt

#     F_thrust = np.zeros(n_dof)

#     if dV_dt >= 0:
#         J_thrust = sp.csr_matrix((n_dof, n_dof)) if sparse else np.zeros((n_dof, n_dof))
#         return F_thrust, J_thrust

#     # --- Apply thrust force ---
#     thrust_dir = np.array([0, 0, 1])  # along Z
#     total_force = -k * dV_dt * thrust_dir

#     # print("Total thrust: ", total_force)

#     shell_node_dof_indices = np.array([
#         robot.map_node_to_dof(i) for i in range(n_nodes) if i in robot.face_nodes_shell
#     ]).flatten().astype(int)  # Flatten to 1D array of DOF indices

#     # print("Shell node DOF indices: ", shell_node_dof_indices)
#     # print("shape of shell_node_dof_indices: ", shell_node_dof_indices.shape)

#     force_per_node = total_force / n_nodes_shell

#     F_thrust[shell_node_dof_indices] = np.repeat(force_per_node, n_nodes_shell) # Apply force to each shell node DOF
#     # F_thrust[shell_node_dof_indices] = force_per_node

#     # --- Jacobian ---
#     dz = np.repeat(thrust_dir / n_nodes_shell, n_nodes_shell)  # (3 * n_nodes_shell,)
#     dV_dq_shell = dV_dq[shell_node_dof_indices]                # restrict to shell DOFs

#     if sparse:
#         data = -k * (1 / dt) * dV_dq_shell * dz                # (3 * n_nodes_shell,)
#         J_thrust = sp.csr_matrix(
#             (data, (shell_node_dof_indices, shell_node_dof_indices)),
#             shape=(n_dof, n_dof)
#         )
#     else:
#         J_thrust = np.zeros((n_dof, n_dof))
#         J_thrust[np.ix_(shell_node_dof_indices, shell_node_dof_indices)] = (
#             -k * (1 / dt) * np.outer(dV_dq_shell, dz)
#         )
    
#     # print("F_thrust: ", F_thrust)
#     # print("J_thrust: ", J_thrust)
#     return F_thrust, J_thrust

