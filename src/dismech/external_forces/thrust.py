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
    positions = q.reshape(-1, 3)
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
    positions = q.reshape(-1, 3)         # (n_nodes, 3)

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
    n_nodes = np.shape(robot.nodes)[0]
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


    force_per_node = total_force / n_nodes

    for i in range(n_nodes):
        idx = robot.map_node_to_dof(i)
        F_thrust[idx] = force_per_node

    # --- Jacobian ---
    if sparse:
        dz = np.tile(thrust_dir / n_nodes, n_nodes)  # (n_dof,)
        data = -k * (1 / dt) * dV_dq * dz
        J_thrust = sp.csr_matrix(data[np.newaxis, :] * np.ones((n_dof, 1)))
    else:
        dz = np.tile(thrust_dir / n_nodes, n_nodes)
        J_thrust = -k * (1 / dt) * np.outer(dV_dq, dz)
    
    # print("F_thrust: ", F_thrust)
    # print("J_thrust: ", J_thrust)
    return F_thrust, J_thrust

