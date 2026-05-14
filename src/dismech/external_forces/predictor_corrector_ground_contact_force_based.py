"""Force-based predictor-corrector for ground contact.

The predictor is unchanged from the position-based variant in
``predictor_corrector_ground_contact.py`` — penetration detection is purely
kinematic and re-exported here under a parallel name so the time stepper can
import the predictor/corrector as a matched pair.

The corrector replaces the velocity-based stick/release criterion with the
sign-of-the-normal-reaction criterion from unilateral contact mechanics: a
predictor-pinned z-DOF stays constrained while the ground reaction is
compressive (pushing the node up) and is released as soon as it would have
to be tensile (pulling the node down). The reaction force is read directly
from the converged Newton residual on the constrained DOFs — see
``corrector_step_for_ground_contact_force_based`` for the sign convention.
"""

import numpy as np
import typing

from ..soft_robot import SoftRobot
from .predictor_corrector_ground_contact import (
    predictor_step_for_ground_contact as predictor_step_for_ground_contact_force_based,
)

__all__ = [
    "predictor_step_for_ground_contact_force_based",
    "corrector_step_for_ground_contact_force_based",
]


def corrector_step_for_ground_contact_force_based(
    robot: SoftRobot,
    F: np.ndarray,
    vertically_constrained_nodes: np.ndarray,
    release_threshold: float = 0.0,
    no_future_freeing: bool = False,
    frictionless: bool = False,
) -> SoftRobot:
    """
    Corrector step for ground contact using the normal-reaction sign as the
    release criterion.

    For each predictor-pinned node, compare the constraint reaction force on
    its z-DOF against ``release_threshold``::

        F_react_z >= release_threshold  →  stays in contact (compressive)
        F_react_z <  release_threshold  →  released (would be tensile)

    The reaction force at a fixed DOF is exactly the Newton residual at that
    DOF evaluated at the converged constrained configuration. The time
    stepper's ``_solve_step`` already re-evaluates ``F`` at the converged
    ``q`` after the constrained re-solve, so passing that ``F`` here gives
    the reaction force at no extra cost.

        F_react_z = F[3 * node + 2]

    Sign convention: positive = ground pushing up (compressive, physical);
    negative = ground would pull down (tensile, unphysical → release).
    Default ``release_threshold = 0`` releases on strictly tensile reactions.
    Setting it to a small positive value introduces a deadband that releases
    slightly before the reaction crosses zero (useful when numerical noise
    leaves a small residual reaction at the moment of separation).

    Parameters
    ----------
    robot : SoftRobot
        Robot whose z-DOFs at ``vertically_constrained_nodes`` are currently
        fixed by the predictor.
    F : (n_dof,) array
        Converged Newton residual at the constrained ``q_final``. On fixed
        DOFs ``F`` equals the reaction force the constraint absorbs.
    vertically_constrained_nodes : (m,) array
        Node indices currently held by the predictor's vertical constraint
        for this step (the set Ξ returned by the predictor).
    release_threshold : float, optional
        Compression level below which a node is released. Default 0.0
        releases only when the reaction is strictly tensile (negative). A
        small positive value gives a deadband that ignores numerical noise.
    no_future_freeing : bool, optional
        If True, the release branch is suppressed: nodes that would otherwise
        be fully released are left in the predictor's z-only constrained
        state (z stays pinned at the ground, x/y remain free). Stuck nodes
        are still promoted to fully constrained.
    frictionless : bool, optional
        If True, the corrector only touches the z-DOF: stuck nodes keep only
        their z pinned (no tangential constraint) and released nodes have
        only their z released. Models a frictionless ground.

    Returns
    -------
    robot : SoftRobot
        Robot with stuck nodes fully constrained and released nodes either
        fully freed (default) or left z-only constrained when
        ``no_future_freeing`` is set.

    Notes
    -----
    The stick/slide tangential decision is folded into the same binary as
    the normal release decision: when ``frictionless=False`` a node that
    stays in contact is fixed on all three axes, so tangential motion is
    inhibited as long as the normal reaction is compressive. A proper
    Coulomb-style tangential test could be layered on top later by gating
    the x/y constraint on a separate slip check.
    """
    if vertically_constrained_nodes.size == 0:
        return robot

    z_dofs = 3 * vertically_constrained_nodes + 2
    F_react_z = F[z_dofs]

    stuck_mask = F_react_z >= release_threshold
    nodes_to_fix = vertically_constrained_nodes[stuck_mask]
    nodes_to_free = vertically_constrained_nodes[~stuck_mask]

    fix_axis = 2 if frictionless else None
    free_axis = 2 if frictionless else None

    if nodes_to_fix.size > 0:
        robot = robot.fix_nodes(nodes_to_fix, axis=fix_axis)
    if nodes_to_free.size > 0 and not no_future_freeing:
        robot = robot.free_nodes(nodes_to_free, axis=free_axis)

    return robot
