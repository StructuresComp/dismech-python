import typing
import numpy as np


from .time_stepper import TimeStepper
from ..soft_robot import SoftRobot


class NewmarkBetaTimeStepper(TimeStepper):

    def __init__(self, robot: SoftRobot, min_force=1e-8, dtype=np.float64, beta=0.25):
        super().__init__(robot, min_force, dtype)
        self._beta = beta

    def _compute_inertial_force_and_jacobian(self, robot: SoftRobot, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        # Inertial force: M * a_new
        inertial_force = robot.mass_matrix @ self._compute_acceleration(
            robot, q)

        # Jacobian of inertial force w.r.t. q: (1/(beta*dt²)) * M
        jacobian = (1.0 / (self._beta * robot.sim_params.dt**2)) * \
            robot.mass_matrix

        return inertial_force, jacobian

    def _compute_acceleration(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return (q - robot.state.q - robot.sim_params.dt * robot.state.u) / \
            (self._beta * robot.sim_params.dt**2) - \
            ((1 - 2 * self._beta) / (2 * self._beta)) * robot.state.a
