import typing
import numpy as np


from .time_stepper import TimeStepper
from ..soft_robot import SoftRobot


class ImplicitEulerTimeStepper(TimeStepper):

    def _compute_inertial_force_and_jacobian(self, robot: SoftRobot, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return (robot.mass_matrix / robot.sim_params.dt) * ((q - robot.state.q) / robot.sim_params.dt - robot.state.u), \
            robot.mass_matrix / robot.sim_params.dt ** 2
