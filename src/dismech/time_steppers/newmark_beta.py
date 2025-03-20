import typing
import numpy as np
from numba import njit

from .time_stepper import TimeStepper
from ..soft_robot import SoftRobot


class NewmarkBetaTimeStepper(TimeStepper):

    def __init__(self, robot: SoftRobot, min_force=1e-8, dtype=np.float64, beta=0.25, gamma=0.5):
        super().__init__(robot, min_force, dtype)
        self._beta = beta
        self._gamma = gamma
        self._dt = robot.sim_params.dt
        self._dt_sq = self._dt**2
        self._beta_dt_sq = self._beta * self._dt_sq
        self._inv_beta_dt_sq = 1.0 / self._beta_dt_sq
        self._gamma_dt = self._gamma * self._dt
        self._one_minus_gamma = 1.0 - self._gamma
        self._one_minus_2beta = 1.0 - 2 * self._beta
        self._one_minus_2beta_over_2beta = self._one_minus_2beta / \
            (2 * self._beta)
        self._mass_diag = robot.mass_matrix

    def _compute_inertial_force_and_jacobian(self, robot: SoftRobot, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        # Compute acceleration
        acceleration = self._compute_acceleration(robot, q)

        # Inertial force: M * a_new
        inertial_force = self._mass_diag * acceleration

        # Jacobian of inertial force w.r.t. q: (1/(beta*dt²)) * M
        jacobian = self._inv_beta_dt_sq * self._mass_diag

        return inertial_force, jacobian

    def _compute_acceleration(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return (q - robot.state.q - self._dt * robot.state.u) / self._beta_dt_sq - \
            self._one_minus_2beta_over_2beta * robot.state.a

    def _compute_velocity(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        acceleration = self._compute_acceleration(robot, q)
        return robot.state.u + self._dt * (self._one_minus_gamma * robot.state.a + self._gamma_dt * acceleration)

    def _compute_evaluation_velocity(self, robot, q):
        return self._compute_velocity(robot, q)
