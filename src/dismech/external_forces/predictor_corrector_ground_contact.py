import numpy as np
import typing

from ..soft_robot import SoftRobot

def predictor_step_for_ground_contact(robot: SoftRobot, q: np.ndarray) -> typing.Tuple[SoftRobot, bool, np.ndarray]:
    """
    Compute ground contact forces and Jacobian using a predictor-corrector method.

    Parameters
    ----------
    robot : SoftRobot
        The soft robot instance containing environment parameters.
    q : (n_dof,) array
        Flattened position vector of all nodes.

    Returns
    -------
    robot : SoftRobot
        The updated soft robot instance with modified fixed and free DOFs.
    revert_to_start : bool
        Whether to revert to the start of the time step.
    vertically_constrained_nodes : (m,) array
        Indices of nodes that are vertically constrained due to ground contact.
    """
    # Get z-component indices of free nodes
    z_indices = np.arange(2, robot.end_node_dof_index, 3)

    # Compute distances from ground
    ground_level = robot.sim_params.ground_level_for_predictor_corrector
    if ground_level is None:
        Warning("ground_level_for_predictor_corrector is not set in sim_params. Using 0.0 as default.")
        ground_level = 0.0
    dist = q[z_indices] - ground_level

    # Create mask for nodes below the ground with some tolerance
    active_mask = dist <= 0 - robot.env.ground_delta
    active_indices = z_indices[active_mask]

    # only keep those indices which are not in robot.fixed_dof
    dof_indices_to_fix = np.setdiff1d(active_indices, robot.fixed_dof)

    # find the nodes corresponding to these dofs
    vertically_constrained_nodes = dof_indices_to_fix // 3

    if dof_indices_to_fix.size == 0:
        revert_to_start = False
        return robot, revert_to_start, vertically_constrained_nodes

    # move these dofs to lie on the ground
    robot.state.q[dof_indices_to_fix] = ground_level

    # Make the active indices fixed_dof
    robot = robot.fix_dof(dof_indices_to_fix)


    # we need to revert to the start of the time step
    revert_to_start = True

    return robot, revert_to_start, vertically_constrained_nodes

def corrector_step_for_ground_contact(robot: SoftRobot, q_final: np.ndarray, vertically_constrained_nodes: np.ndarray, threshold: float = 1e-6) -> SoftRobot:
    """
    Corrector step for ground contact forces using a predictor-corrector method.

    Parameters
    ----------
    robot : SoftRobot
        The soft robot instance containing environment parameters.
    q : (n_dof,) array
        Flattened position vector of all nodes.

    Returns
    -------
    robot : SoftRobot
        The updated soft robot instance with modified fixed and free DOFs.
    """
    # velocity vector 
    u = (q_final - robot.state.q0) / robot.sim_params.dt
    u_vecs = u[:robot.end_node_dof_index].reshape(-1, 3)
    # check if the velocity for the vertically constrained nodes is greater than a small threshold
    # use normal (z) component to decide
    uz = u_vecs[vertically_constrained_nodes, 2]
    nodes_to_fix = vertically_constrained_nodes[np.abs(uz) < threshold]
    if nodes_to_fix.size > 0:
        robot = robot.fix_nodes(nodes_to_fix)
        print("Fixing nodes: ", nodes_to_fix)

    # free the other nodes
    nodes_to_free = np.setdiff1d(vertically_constrained_nodes, nodes_to_fix, assume_unique=False)
    if nodes_to_free.size > 0:
        robot = robot.free_nodes(nodes_to_free)  

    return robot