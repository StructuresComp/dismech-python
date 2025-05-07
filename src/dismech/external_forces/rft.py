import typing
import numpy as np
from ..soft_robot import SoftRobot


import numpy as np
import typing

def compute_rft(
    robot: SoftRobot,
    q: np.ndarray,
    u: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    q_n = q[: robot.end_node_dof_index]
    u_n = u[: robot.end_node_dof_index]

    N = q_n.size // 3
    X = q_n.reshape(N, 3)
    V = u_n.reshape(N, 3)

    T = np.empty_like(X)
    T[0]     = X[1]  - X[0]
    T[-1]    = X[-1] - X[-2]
    T[1:-1]  = X[2:] - X[:-2]
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    T /= np.where(norms < 1e-12, 1.0, norms)

    ct, cn = robot.env.ct, robot.env.cn
    dt     = robot.sim_params.dt

    I3 = np.eye(3)
    M  = (ct - cn) * T[:, :, None] * T[:, None, :] + cn * I3

    F_nodes   = -np.einsum("nij,nj->ni", M, V).reshape(-1)
    J_blocks  = (-M / dt)

    # assemble block-diagonal
    base = 3 * np.arange(N)[:, None]
    rows = base + np.arange(3)
    r_idx = np.repeat(rows, 3, axis=1).ravel()
    c_idx = np.tile(rows,  (1,3)).ravel()

    J_nodes = np.zeros((3 * N, 3 * N))
    J_nodes[r_idx, c_idx] = J_blocks.reshape(-1)

    ndof   = q.size
    F_full = np.zeros(ndof, dtype=q.dtype)
    J_full = np.zeros((ndof, ndof), dtype=q.dtype)

    sl = slice(None, robot.end_node_dof_index)
    F_full[sl]         = F_nodes
    J_full[sl, sl]     = J_nodes

    return F_full, J_full

