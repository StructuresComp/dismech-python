import typing
import numpy as np


from .time_stepper import TimeStepper
from ..soft_robot import SoftRobot


class ImplicitMidpointTimeStepper(TimeStepper):

    def __init__(self, robot: SoftRobot, min_force=1e-8, dtype=np.float64):
        super().__init__(robot, min_force, dtype)

    def _compute_inertial_force_and_jacobian(self, robot: SoftRobot, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        a_mid = self._compute_acceleration(robot, self._compute_evaluation_position(robot, q))

        # Inertial force: M * a_mid
        inertial_force = robot.mass_matrix @ a_mid

        # Jacobian of inertial force w.r.t q
        # ∂a_mid/∂q_next = 1/(dt) * (∂u_mid/∂q) = 2/(dt^2) * I
        jacobian = (1.0 / robot.sim_params.dt**2) * robot.mass_matrix

        return inertial_force, jacobian

    def _compute_acceleration(self, robot: SoftRobot, q: np.ndarray):
        return 2.0 * (self._compute_evaluation_velocity(robot, q) - robot.state.u) / robot.sim_params.dt

    def _compute_evaluation_position(self, robot, q):
        return 0.5 * (robot.state.q + q)

    def _compute_evaluation_velocity(self, robot, q):
        return 2.0 * (q - robot.state.q) / robot.sim_params.dt
