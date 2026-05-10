import numpy as np
import typing

from ..soft_robot import SoftRobot

def predictor_step_for_ground_contact(robot: SoftRobot, q: np.ndarray) -> typing.Tuple[SoftRobot, bool, np.ndarray]:
    """
    Predictor step for ground contact (Belt Contact BC).

    Detects nodes whose z-coordinate has reached the belt (``z_i <= z_belt + r_0``,
    using ``env.ground_z + env.ground_h`` and the tolerance ``env.ground_delta``),
    rewinds those nodes' z-DOF in ``robot.state.q`` to the belt surface, and adds
    those z-DOFs to the robot's fixed-DOF set so the subsequent re-solve treats
    them as vertically constrained while ``(x, y)`` continue to follow the EOM.

    Parameters
    ----------
    robot : SoftRobot
        Robot whose env carries ``ground_z``, ``ground_h``, ``ground_delta``.
    q : (n_dof,) array
        Trial position vector from the unconstrained step, used only to detect
        contact. The snap is written to ``robot.state.q`` (the start-of-step
        state the integrator will re-solve from), not to ``q``.

    Returns
    -------
    robot : SoftRobot
        Robot with the newly contacted z-DOFs added to ``fixed_dof``.
    revert_to_start : bool
        True iff any new contacts were added, signalling the caller to rewind
        and re-solve the time step with the updated constraints.
    vertically_constrained_nodes : (m,) array
        Node indices whose z-DOF was *just added* to ``fixed_dof`` by this call
        (the new contact set Ξ). The caller is responsible for accumulating
        this across calls if it needs the full set of currently
        vertically-constrained nodes to pass to the corrector.
    """
    z_indices = np.arange(2, robot.end_node_dof_index, 3)

    ground_level = robot.env.ground_z + robot.env.ground_h
    dist = q[z_indices] - ground_level

    active_mask = dist <= robot.env.ground_delta
    active_indices = z_indices[active_mask]

    dof_indices_to_fix = np.setdiff1d(active_indices, robot.fixed_dof)
    vertically_constrained_nodes = dof_indices_to_fix // 3

    if dof_indices_to_fix.size == 0:
        return robot, False, vertically_constrained_nodes

    robot.state.q[dof_indices_to_fix] = ground_level
    robot = robot.fix_nodes(vertically_constrained_nodes, axis=2, fix_edges=False)

    return robot, True, vertically_constrained_nodes

def corrector_step_for_ground_contact(robot: SoftRobot, q_final: np.ndarray, vertically_constrained_nodes: np.ndarray, threshold: float = 1e-6) -> SoftRobot:
    """
    Corrector step for ground contact (Belt Contact BC).

    For each currently vertically-constrained node, transitions it either to
    the fully constrained state (sticks to the belt) or back to the free state,
    based on the relative velocity between the node and the belt:

        ``||u_node - u_belt|| < threshold``  →  fully constrained
        otherwise                            →  fully freed

    The belt is assumed static (``u_belt = 0``); ``u_node`` is the full 3D
    velocity from ``(q_final - state.q) / dt``. Since z is held during the
    constrained solve, the meaningful component of the criterion is the
    tangential ``(x, y)`` slip.

    Parameters
    ----------
    robot : SoftRobot
        Robot whose z-DOFs at ``vertically_constrained_nodes`` are currently
        fixed.
    q_final : (n_dof,) array
        Position vector returned by the constrained solve.
    vertically_constrained_nodes : (m,) array
        All node indices currently held by the predictor's vertical constraint.
        The caller is responsible for maintaining this cumulative set across
        time steps.
    threshold : float, optional
        Slip-velocity tolerance for the stick decision (paper uses 1e-6 cm/s).

    Returns
    -------
    robot : SoftRobot
        Robot with stuck nodes fully constrained and lifting/sliding nodes
        fully freed.
    """
    if vertically_constrained_nodes.size == 0:
        return robot

    u = (q_final - robot.state.q) / robot.sim_params.dt
    u_node = u[:robot.end_node_dof_index].reshape(-1, 3)[vertically_constrained_nodes]
    # TODO: extend to moving belts by subtracting u_belt before taking the norm.
    slip_speed = np.linalg.norm(u_node, axis=1)

    stuck_mask = slip_speed < threshold
    nodes_to_fix = vertically_constrained_nodes[stuck_mask]
    nodes_to_free = vertically_constrained_nodes[~stuck_mask]

    if nodes_to_fix.size > 0:
        robot = robot.fix_nodes(nodes_to_fix)
    if nodes_to_free.size > 0:
        robot = robot.free_nodes(nodes_to_free)

    return robot
