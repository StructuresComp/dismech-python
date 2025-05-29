import typing
import numpy as np

from ..soft_robot import SoftRobot
from .ground_friction_helper import get_floor_lambda_fns

_static_dof, _static_fn, _slide_dof, _slide_fn = get_floor_lambda_fns()


def compute_ground_contact(robot: SoftRobot, q: np.ndarray, as_raw=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    # Get z-component indices
    z_indices = np.arange(2, robot.end_node_dof_index, 3)
    dist = q[z_indices] - robot.env.ground_h - robot.env.ground_z

    # print("dist is: ", dist)

    # Create mask for nodes close enough to the ground
    active_mask = (dist <= robot.env.ground_delta)
    active_indices = z_indices[active_mask]
    active_dist = dist[active_mask]

    if active_indices.size == 0:
        return np.zeros_like(q), np.zeros((q.shape[0], q.shape[0]))

    # Compute force and stiffness only for active points
    K1 = 15 / robot.env.ground_delta
    v = np.exp(-K1 * active_dist)
    f_raw = (-2 * v * np.log(v + 1)) / \
        (K1 * (v + 1)) * robot.env.ground_stiffness
    j_raw = (2 * v * np.log(v + 1) + 2 * v ** 2) / \
        ((v + 1) ** 2) * robot.env.ground_stiffness

    if as_raw:
        f_full = np.zeros((f_raw.shape[0], 3))
        f_full[:, 2] = f_raw
        j_full = np.zeros((j_raw.shape[0], 3, 3))
        j_full[:, 2, 2] = j_raw
        return f_full, j_full

    # Allocate global force and Jacobian
    F = np.zeros_like(q)
    J = np.zeros((q.shape[0], q.shape[0]))

    # Accumulate only valid entries
    F[active_indices] = -f_raw
    J[active_indices, active_indices] = -j_raw
    return F, J


def compute_ground_contact_friction(robot: SoftRobot, q: np.ndarray, u: np.ndarray, eps=1e-4) -> typing.Tuple[np.ndarray, np.ndarray]:
    F, J = compute_ground_contact(robot, q, as_raw=True)

    if not F.any():
        return F, J

    u_vec = u[:robot.end_node_dof_index].reshape(-1, 3)

    # Compute normal vectors from contact forces
    f_norms = np.linalg.norm(F, axis=1, keepdims=True)
    valid_force = f_norms > eps

    n_hat = np.zeros_like(F)
    n_hat[valid_force[:, 0]] = F[valid_force[:, 0]] / \
        f_norms[valid_force, None]

    # Compute tangential velocity
    v_dot_n = np.sum(u_vec * n_hat, axis=1, keepdims=True)
    v_proj = v_dot_n * n_hat
    v_tangent = u_vec - v_proj

    v_tangent_norm = np.linalg.norm(v_tangent, axis=1)
    valid_tangent = v_tangent_norm > eps

    if valid_tangent.any():
        # Normalize valid tangent vectors
        v_tangent_hat = np.zeros_like(v_tangent)
        v_tangent_hat[valid_tangent] = v_tangent[valid_tangent] / \
            v_tangent_norm[valid_tangent, None]

        slide_inds = v_tangent_norm > robot.env.ground_vel_tol
        stick_inds = ~slide_inds

        k2 = 15 / robot.env.ground_vel_tol

        def apply_chain_rule(J_local, u_local, f_local, df_dx_fn, df_dfn_fn):
            inputs = np.vstack([
                u_local.T, f_local.T,
                robot.env.ground_mu * np.ones(u_local.shape[0]),
                k2 * np.ones(u_local.shape[0])
            ])
            df_dx = df_dx_fn(*inputs).transpose(2, 0, 1) / robot.sim_params.dt
            df_dfn = df_dfn_fn(*inputs).transpose(2, 0, 1)
            return df_dfn * J_local + df_dx

        if slide_inds.any():
            J[slide_inds] += apply_chain_rule(J[slide_inds], u_vec[slide_inds],
                                              F[slide_inds], _slide_dof, _slide_fn)
            F[slide_inds] += robot.env.ground_mu * \
                f_norms[slide_inds] * v_tangent_hat[slide_inds]
        if stick_inds.any():
            J[stick_inds] += apply_chain_rule(J[stick_inds], u_vec[stick_inds],
                                              F[stick_inds], _static_fn, _static_fn)
            gamma = (2 / (1 + np.exp(-k2 * v_tangent_norm[stick_inds])) - 1)
            F[stick_inds] += gamma[:, None] * robot.env.ground_mu * \
                f_norms[stick_inds] * v_tangent_hat[stick_inds]

    # Scatter into global vector/matrix
    F_full = np.zeros_like(q)
    J_full = np.zeros((q.shape[0], q.shape[0]))

    F_full[:robot.end_node_dof_index] -= F.flatten()
    for i in range(J.shape[0]):
        idx = slice(3*i, 3*(i+1))
        J_full[idx, idx] -= J[i]

    return F_full, J_full
